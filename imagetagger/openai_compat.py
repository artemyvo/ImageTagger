from __future__ import annotations

import base64
import http.client
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from imagetagger.image_prep import prepare_image_for_query
from imagetagger.llm_provider import LlmProviderCancelled, LlmProviderError, LlmRequestCancellation


DEFAULT_OPENAI_COMPAT_SERVER = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 300.0


class OpenAiCompatError(LlmProviderError):
    pass


class OpenAiCompatCancelled(OpenAiCompatError, LlmProviderCancelled):
    pass


@dataclass(frozen=True)
class OpenAiCompatConnection:
    server_url: str
    model_name: str


def normalize_server_url(server: str) -> str:
    trimmed = server.strip()
    if not trimmed:
        return DEFAULT_OPENAI_COMPAT_SERVER

    if "://" not in trimmed:
        trimmed = f"http://{trimmed}"

    parsed = urlparse(trimmed)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OpenAiCompatError("Enter a valid server address.")

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
        raise OpenAiCompatError("Only http and https server URLs are supported.")

    request_path = path if path.startswith("/") else f"/{path}"
    data = None
    headers = {"Accept": "application/json"}
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
            message = f"Server returned HTTP {response.status}."
            details = response_text.strip()
            if details:
                message = f"{message} {details}"
            raise OpenAiCompatError(message)

        return json.loads(response_text)
    except OpenAiCompatCancelled:
        raise
    except TimeoutError as exc:
        raise OpenAiCompatError(
            f"Timed out after {int(timeout)} seconds while contacting server. "
            "Try again or use a smaller/faster model."
        ) from exc
    except OSError as exc:
        if cancellation is not None and cancellation.is_cancelled():
            raise OpenAiCompatCancelled("Request stopped.") from exc
        raise OpenAiCompatError(f"Could not reach server: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OpenAiCompatError("Server returned invalid JSON.") from exc
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
    payload = _request_json(server, "/v1/models", timeout=timeout)
    data = payload.get("data")
    if not isinstance(data, list):
        raise OpenAiCompatError("Server response did not include a models list.")

    model_names: list[str] = []
    for model in data:
        if isinstance(model, dict):
            model_id = model.get("id")
            if isinstance(model_id, str) and model_id.strip():
                model_names.append(model_id.strip())

    return model_names


def _encode_image_data_url(image_path: Path) -> str:
    prepared_image = prepare_image_for_query(image_path)
    encoded = base64.b64encode(prepared_image.content).decode("ascii")
    media_type = prepared_image.media_type or "image/jpeg"
    return f"data:{media_type};base64,{encoded}"


def _extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()

    return ""


def generate_with_image(
    connection: OpenAiCompatConnection,
    image_path: Path,
    prompt: str,
    timeout: float = DEFAULT_TIMEOUT,
    cancellation: LlmRequestCancellation | None = None,
) -> str:
    payload: dict[str, object] = {
        "model": connection.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image_data_url(image_path)}},
                ],
            }
        ],
        "stream": False,
    }

    response_payload = _request_json(
        connection.server_url,
        "/v1/chat/completions",
        payload=payload,
        timeout=timeout,
        cancellation=cancellation,
    )

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAiCompatError("Server returned no completion choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise OpenAiCompatError("Server returned an invalid completion payload.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise OpenAiCompatError("Server returned no assistant message.")

    content = _extract_text_content(message.get("content"))
    if not content:
        raise OpenAiCompatError("Server returned an empty response.")
    return content