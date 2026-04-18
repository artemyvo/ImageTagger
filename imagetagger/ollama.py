from __future__ import annotations

import base64
import http.client
import json
from dataclasses import dataclass
from urllib.parse import urlparse

from pathlib import Path

from imagetagger.image_prep import prepare_image_for_query
from imagetagger.llm_provider import LlmProviderCancelled, LlmProviderError, LlmRequestCancellation


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


def normalize_server_url(server: str) -> str:
    trimmed = server.strip()
    if not trimmed:
        return DEFAULT_OLLAMA_SERVER

    if "://" not in trimmed:
        trimmed = f"http://{trimmed}"

    parsed = urlparse(trimmed)
    if not parsed.scheme or not parsed.netloc:
        raise OllamaError("Enter a valid Ollama server address.")

    normalized = f"{parsed.scheme}://{parsed.netloc}"
    return normalized.rstrip("/")


def _request_json(
    server: str,
    path: str,
    payload: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    cancellation: LlmRequestCancellation | None = None,
) -> dict:
    server_url = normalize_server_url(server)
    parsed = urlparse(server_url)
    if parsed.scheme == "http":
        connection_class = http.client.HTTPConnection
    elif parsed.scheme == "https":
        connection_class = http.client.HTTPSConnection
    else:
        raise OllamaError("Only http and https Ollama server URLs are supported.")

    request_path = path if path.startswith("/") else f"/{path}"
    data = None
    headers: dict[str, str] = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    connection = connection_class(parsed.netloc, timeout=timeout)
    response: http.client.HTTPResponse | None = None

    try:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
            cancellation.set_active_resource(connection)

        method = "POST" if payload is not None else "GET"
        connection.request(method, request_path, body=data, headers=headers)
        response = connection.getresponse()

        chunks = bytearray()
        while True:
            if cancellation is not None:
                cancellation.raise_if_cancelled()

            chunk = response.read(64 * 1024)
            if not chunk:
                break
            chunks.extend(chunk)

        response_text = chunks.decode("utf-8")
        if response.status >= 400:
            message = f"Ollama server returned HTTP {response.status}."
            details = response_text.strip()
            if details:
                message = f"{message} {details}"
            raise OllamaError(message)

        return json.loads(response_text)
    except OllamaCancelled:
        raise
    except TimeoutError as exc:
        raise OllamaError(
            f"Timed out after {int(timeout)} seconds while contacting Ollama server. "
            "Try again or use a smaller/faster model."
        ) from exc
    except OSError as exc:
        if cancellation is not None and cancellation.is_cancelled():
            raise OllamaCancelled("Request stopped.") from exc
        raise OllamaError(f"Could not reach Ollama server: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama server returned invalid JSON.") from exc
    finally:
        if cancellation is not None:
            cancellation.clear_active_resource(connection)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        connection.close()


def fetch_models(server: str, timeout: float = 5.0) -> list[str]:
    payload = _request_json(server, "/api/tags", timeout=timeout)

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

    payload = _request_json(
        connection.server_url,
        "/api/generate",
        payload=payload,
        timeout=timeout,
        cancellation=cancellation,
    )
    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
        raise OllamaError("Ollama returned an empty response.")
    return response.strip()
