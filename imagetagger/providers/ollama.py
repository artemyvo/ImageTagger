from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from imagetagger.utils.image_prep import prepare_image_for_query
from imagetagger.providers.llm_provider import LlmProviderCancelled, LlmProviderError, LlmRequestCancellation
from imagetagger.providers.http_request import request_json


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
        raise OllamaError("Ollama returned an empty response.")
    return response.strip()
