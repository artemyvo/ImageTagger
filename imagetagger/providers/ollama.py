from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from imagetagger.utils.image_prep import prepare_image_for_query
from imagetagger.providers.llm_provider import LlmProviderCancelled, LlmProviderError, LlmRequestCancellation
from imagetagger.providers.http_request import request_json, discard_pooled_connection_for_server


DEFAULT_OLLAMA_SERVER = "http://127.0.0.1:11434"
# Vision-capable LLMs can take a while on busy GPUs; keep this generous.
DEFAULT_TIMEOUT = 300.0


class OllamaError(LlmProviderError):
    pass


class OllamaCancelled(OllamaError, LlmProviderCancelled):
    pass


OllamaCancellation = LlmRequestCancellation


@dataclass(frozen=True)
class OllamaConnection:
    server_url: str
    model_name: str


def fetch_models(server: str, timeout: float = 5.0) -> list[str]:
    payload = request_json(
        server=server,
        default_server=DEFAULT_OLLAMA_SERVER,
        path="/api/tags",
        timeout=timeout,
        error_class=OllamaError,
        cancel_class=OllamaCancelled,
    )

    models = payload.get("models", [])
    if not isinstance(models, list):
        raise OllamaError("Ollama server response did not include a models list.")

    model_names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("name")
            if isinstance(name, str) and name.strip():
                model_names.append(name.strip())

    return model_names

def _encode_image(image_path: Path) -> str:
    prepared_image = prepare_image_for_query(image_path, force_webp_to_png=True)
    return base64.b64encode(prepared_image.content).decode("ascii")


def generate_with_image(
    connection: OllamaConnection,
    image_path: Path,
    prompt: str,
    timeout: float = DEFAULT_TIMEOUT,
    cancellation: LlmRequestCancellation | None = None,
    thread_count: int | None = None,
) -> str:
    payload: dict[str, object] = {
        "model": connection.model_name,
        "prompt": prompt,
        "images": [_encode_image(image_path)],
        "stream": False,
    }

    if thread_count is not None:
        payload["options"] = {"num_thread": max(1, int(thread_count))}

    response_payload = request_json(
        server=connection.server_url,
        default_server=DEFAULT_OLLAMA_SERVER,
        path="/api/generate",
        payload=payload,
        timeout=timeout,
        cancellation=cancellation,
        error_class=OllamaError,
        cancel_class=OllamaCancelled,
    )
    response = response_payload.get("response")
    if not isinstance(response, str) or not response.strip():
        thinking = response_payload.get("thinking")
        _context_exhausted = isinstance(thinking, str) and bool(thinking.strip())
        if _context_exhausted:
            eval_count = response_payload.get("eval_count")
            eval_info = f" eval_count={eval_count}" if isinstance(eval_count, int) else ""
            print(
                f"[debug_prompts] response is empty but thinking is non-empty "
                f"({len(thinking)} chars{eval_info}) — context window likely exhausted by CoT tokens. "
                f"Consider increasing num_ctx on your Ollama server.",
                flush=True,
            )
        done_reason = response_payload.get("done_reason")
        # Discard the pooled connection: the server returned a valid HTTP 200
        # but an empty body, which can leave the TCP stream in an ambiguous
        # state.  A fresh socket on the next retry avoids re-using a stale one.
        discard_pooled_connection_for_server(connection.server_url, DEFAULT_OLLAMA_SERVER)
        reason_detail = f" (done_reason={done_reason!r})" if isinstance(done_reason, str) and done_reason else ""
        try:
            from imagetagger import config as _config
            if _config.load().get("debug_prompts"):
                debug_payload = {
                    k: [f"[base64 image, {len(v)} chars]" for v in val]
                    if k == "images" and isinstance(val, list)
                    else f"[{len(val)} chars]"
                    if k == "prompt" and isinstance(val, str)
                    else val
                    for k, (val) in ((k, payload[k]) for k in payload)
                }
                print(f"[debug_prompts] empty response for file: {image_path}", flush=True)
                print("[debug_prompts] request sent to Ollama:", flush=True)
                print(json.dumps(debug_payload, indent=2, ensure_ascii=False), flush=True)
                debug_response = {
                    k: f"[{len(v)} tokens]" if k == "context" and isinstance(v, list) else v
                    for k, v in response_payload.items()
                } if isinstance(response_payload, dict) else response_payload
                print("[debug_prompts] raw response from LLM:", flush=True)
                print(json.dumps(debug_response, indent=2, ensure_ascii=False), flush=True)
        except Exception:
            pass
        err = OllamaError(f"Ollama returned an empty response{reason_detail}.")
        err.no_backoff = True
        # Context exhaustion: thinking tokens consumed the entire context window,
        # leaving no room for the actual response.  Retrying will produce the same
        # result, so callers should skip further retries and warn the user.
        err.context_exhausted = _context_exhausted
        raise err
    return response.strip()
