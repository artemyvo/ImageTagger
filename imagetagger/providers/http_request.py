from __future__ import annotations

import http.client
import json
from urllib.parse import urlparse

from imagetagger.providers.llm_provider import (
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
    normalize_server_url,
)


def request_json(
    server: str,
    default_server: str,
    path: str,
    payload: dict | None = None,
    timeout: float = 300.0,
    cancellation: LlmRequestCancellation | None = None,
    error_class: type[LlmProviderError] = LlmProviderError,
    cancel_class: type[LlmProviderCancelled] = LlmProviderCancelled,
) -> dict:
    server_url = normalize_server_url(server, default_server)
    parsed = urlparse(server_url)
    if parsed.scheme == "http":
        connection_class = http.client.HTTPConnection
    elif parsed.scheme == "https":
        connection_class = http.client.HTTPSConnection
    else:
        raise error_class("Only http and https server URLs are supported.")

    request_path = path if path.startswith("/") else f"/{path}"
    data = None
    headers: dict[str, str] = {"Accept": "application/json"}

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
            raise error_class(message)

        return json.loads(response_text)
    except cancel_class:
        raise
    except TimeoutError as exc:
        raise error_class(
            f"Timed out after {int(timeout)} seconds while contacting server. "
            "Try again or use a smaller/faster model."
        ) from exc
    except OSError as exc:
        if cancellation is not None and cancellation.is_cancelled():
            raise cancel_class("Request stopped.") from exc
        raise error_class(f"Could not reach server: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise error_class("Server returned invalid JSON.") from exc
    finally:
        if cancellation is not None:
            cancellation.clear_active_resource(connection)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        connection.close()
