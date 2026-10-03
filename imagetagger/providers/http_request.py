from __future__ import annotations

import http.client
import json
import socket
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


def discard_pooled_connection_for_server(server: str, default_server: str) -> None:
    """Discard the thread-local pooled connection for the given server URL.

    Call this when a technically-successful HTTP response contains invalid
    content (e.g. an empty body from Ollama) so the next request starts with
    a guaranteed-fresh socket instead of potentially reusing a stale one.
    """
    server_url = normalize_server_url(server, default_server)
    parsed = urlparse(server_url)
    _discard_pooled_connection(parsed.netloc, parsed.scheme)


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
        sock: socket.socket | None = None
        _discard = False

        try:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
                cancellation.set_active_resource(connection)

            connection.request(method, request_path, body=data, headers=headers)
            # On a "Connection: close" answer getresponse() hands the socket to
            # the response and clears connection.sock, so cancel needs the
            # socket itself to interrupt reading the body.
            sock = connection.sock
            if cancellation is not None and sock is not None:
                cancellation.set_active_resource(sock)
            response = connection.getresponse()

            chunks = bytearray()
            while True:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()

                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                chunks.extend(chunk)
            if cancellation is not None:
                # A cancel shuts the socket down, which can end the body early.
                cancellation.raise_if_cancelled()

            if response.status >= 400:
                message = f"Server returned HTTP {response.status}."
                # Error pages (e.g. from a proxy) are not always UTF-8.
                details = chunks.decode("utf-8", errors="replace").strip()
                if details:
                    message = f"{message} {details}"
                raise error_class(message)

            try:
                response_text = chunks.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise error_class("Server returned a response that is not UTF-8 text.") from exc
            result = json.loads(response_text)
            if not isinstance(result, dict):
                raise error_class("Server returned unexpected JSON (not an object).")
            return result

        except LlmProviderCancelled as exc:
            # Also the plain LlmProviderCancelled from raise_if_cancelled: the
            # connection may be shut down, so never reuse it.
            _discard = True
            if isinstance(exc, cancel_class):
                raise
            raise cancel_class(str(exc)) from exc
        # socket.timeout is only an alias of TimeoutError from Python 3.10 on.
        # Caught before OSError: a timeout is not a stale connection to retry.
        except (TimeoutError, socket.timeout) as exc:
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
        except http.client.HTTPException as exc:
            # Not an OSError: e.g. BadStatusLine from a server that does not
            # speak HTTP, or ResponseNotReady after a cancel closed the socket.
            _discard = True
            if cancellation is not None and cancellation.is_cancelled():
                raise cancel_class("Request stopped.") from exc
            raise error_class(f"Could not reach server: invalid HTTP answer ({exc!r})") from exc
        except json.JSONDecodeError as exc:
            raise error_class("Server returned invalid JSON.") from exc
        finally:
            if cancellation is not None:
                cancellation.clear_active_resource(connection)
                if sock is not None:
                    cancellation.clear_active_resource(sock)
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
