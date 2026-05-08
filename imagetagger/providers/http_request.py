from __future__ import annotations

import http.client
import json
import threading
from urllib.parse import urlparse

from imagetagger.providers.llm_provider import (
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
    normalize_server_url,
)


# ---------------------------------------------------------------------------
# Per-thread connection pool.  Each worker thread keeps one open connection
# per (netloc, scheme) so sequential LLM requests to the same server reuse
# the TCP socket instead of re-handshaking on every call.
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def _get_pooled_connection(
    netloc: str,
    scheme: str,
    timeout: float,
    connection_class: type,
) -> tuple[http.client.HTTPConnection, bool]:
    """Return *(connection, was_reused)* from the thread-local pool."""
    pool: dict = getattr(_thread_local, "connections", None)  # type: ignore[assignment]
    if pool is None:
        _thread_local.connections = {}
        pool = _thread_local.connections
    key = (netloc, scheme)
    conn = pool.get(key)
    if conn is not None:
        conn.timeout = timeout
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.settimeout(timeout)
            except Exception:
                pass
        return conn, True
    conn = connection_class(netloc, timeout=timeout)
    pool[key] = conn
    return conn, False


def _discard_pooled_connection(netloc: str, scheme: str) -> None:
    """Remove the connection from the pool and close its socket."""
    pool: dict = getattr(_thread_local, "connections", None)  # type: ignore[assignment]
    if pool is None:
        return
    conn = pool.pop((netloc, scheme), None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _store_pooled_connection(
    netloc: str,
    scheme: str,
    conn: http.client.HTTPConnection,
) -> None:
    pool: dict = getattr(_thread_local, "connections", None)  # type: ignore[assignment]
    if pool is None:
        _thread_local.connections = {}
        pool = _thread_local.connections
    pool[(netloc, scheme)] = conn


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

    method = "POST" if payload is not None else "GET"
    netloc = parsed.netloc
    scheme = parsed.scheme

    connection, was_reused = _get_pooled_connection(netloc, scheme, timeout, connection_class)

    # Attempt the request.  If the pooled connection is stale (OSError on
    # first attempt), discard it and retry once with a fresh connection.
    for attempt in range(2):
        response: http.client.HTTPResponse | None = None
        _discard = False

        try:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
                cancellation.set_active_resource(connection)

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
            _discard = True
            raise
        except TimeoutError as exc:
            _discard = True
            raise error_class(
                f"Timed out after {int(timeout)} seconds while contacting server. "
                "Try again or use a smaller/faster model."
            ) from exc
        except OSError as exc:
            _discard = True
            if cancellation is not None and cancellation.is_cancelled():
                raise cancel_class("Request stopped.") from exc
            if attempt == 0 and was_reused:
                # Stale pooled connection — will retry with a fresh one below.
                pass
            else:
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
            if _discard:
                _discard_pooled_connection(netloc, scheme)

        # Retry with a fresh connection (only reached when attempt==0, was_reused, OSError).
        connection = connection_class(netloc, timeout=timeout)
        _store_pooled_connection(netloc, scheme, connection)

    # Unreachable: the loop always returns or raises before this point.
    raise error_class("Could not reach server.")  # pragma: no cover
