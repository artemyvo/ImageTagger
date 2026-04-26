from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from imagetagger.utils.image_prep import prepare_image_for_query
from imagetagger.providers.llm_provider import LlmProviderCancelled, LlmProviderError, LlmRequestCancellation
from imagetagger.providers.http_request import request_json


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


def fetch_models(server: str, timeout: float = 5.0) -> list[str]:
    payload = request_json(
        server=server,
        default_server=DEFAULT_OPENAI_COMPAT_SERVER,
        path="/v1/models",
        timeout=timeout,
        error_class=OpenAiCompatError,
        cancel_class=OpenAiCompatCancelled,
    )
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

    response_payload = request_json(
        server=connection.server_url,
        default_server=DEFAULT_OPENAI_COMPAT_SERVER,
        path="/v1/chat/completions",
        payload=payload,
        timeout=timeout,
        cancellation=cancellation,
        error_class=OpenAiCompatError,
        cancel_class=OpenAiCompatCancelled,
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
