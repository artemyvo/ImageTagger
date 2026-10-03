"""request_json against a local stub server: success, error mapping, timeouts, cancellation and pooling."""
from __future__ import annotations

import http.client
import socket
import threading
import time
from typing import Any, Dict

import pytest

from imagetagger.providers import http_request
from imagetagger.providers.http_request import discard_pooled_connection_for_server, request_json
from imagetagger.providers.llm_provider import (
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
)

DEFAULT = "http://127.0.0.1:1"  # never contacted: every test passes a server


class CustomError(LlmProviderError):
    pass


class CustomCancelled(CustomError, LlmProviderCancelled):
    pass


def _get(server, path="/api/tags", **kwargs):
    return request_json(server.url, DEFAULT, path, timeout=kwargs.pop("timeout", 5.0), **kwargs)


def _pool() -> Dict[Any, Any]:
    return getattr(http_request._thread_local, "connections", {})


def _in_thread(fn) -> Dict[str, Any]:
    """Run *fn* on a worker thread; the dict gets ``result``/``error`` and ``thread``."""
    outcome: Dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed back to the test
            outcome["error"] = exc

    outcome["thread"] = threading.Thread(target=run, daemon=True)
    outcome["thread"].start()
    return outcome


# ---------------------------------------------------------------------------
# Successful requests
# ---------------------------------------------------------------------------


def test_get_without_payload_returns_the_decoded_json(stub_server):
    stub_server.reply(json_body={"models": [{"name": "m"}]})

    assert _get(stub_server) == {"models": [{"name": "m"}]}

    request = stub_server.last
    assert request.method == "GET"
    assert request.path == "/api/tags"
    assert request.headers["accept"] == "application/json"
    assert "content-type" not in request.headers
    assert request.body == b""


def test_post_sends_the_payload_as_json(stub_server):
    stub_server.reply(json_body={"response": "ok"})

    result = request_json(stub_server.url, DEFAULT, "/api/generate", {"model": "m", "n": [1, 2]}, timeout=5.0)

    assert result == {"response": "ok"}
    request = stub_server.last
    assert request.method == "POST"
    assert request.headers["content-type"] == "application/json"
    assert request.json == {"model": "m", "n": [1, 2]}


def test_path_without_leading_slash_gets_one(stub_server):
    _get(stub_server, path="v1/models")
    assert stub_server.last.path == "/v1/models"


def test_server_without_scheme_or_with_a_path_is_normalized(stub_server):
    """Only scheme://host:port is kept; the request path comes from *path* alone."""
    request_json(f"  127.0.0.1:{stub_server.port}/ignored/prefix/ ", DEFAULT, "/api/tags", timeout=5.0)
    assert stub_server.last.path == "/api/tags"


def test_empty_server_falls_back_to_the_default(stub_server):
    request_json("   ", stub_server.url, "/api/tags", timeout=5.0)
    assert stub_server.last.path == "/api/tags"


def test_unsupported_default_scheme_raises_the_error_class():
    with pytest.raises(CustomError, match="Only http and https"):
        request_json("", "ftp://127.0.0.1:21", "/x", timeout=1.0, error_class=CustomError)


def test_invalid_server_address_is_a_provider_error():
    with pytest.raises(LlmProviderError, match="valid server address"):
        request_json("ftp://127.0.0.1:21", DEFAULT, "/x", timeout=1.0)


# ---------------------------------------------------------------------------
# HTTP errors and bad bodies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (400, b'{"error": "model not found"}', 'Server returned HTTP 400. {"error": "model not found"}'),
        (404, b"404 page not found\n", "Server returned HTTP 404. 404 page not found"),
        (500, b"", "Server returned HTTP 500."),
        (503, b"  \n ", "Server returned HTTP 503."),
    ],
    ids=["4xx-json", "4xx-text", "5xx-empty", "5xx-blank"],
)
def test_http_error_status_raises_the_error_class_with_the_body(stub_server, status, body, expected):
    stub_server.reply(status=status, body=body)

    with pytest.raises(CustomError) as raised:
        _get(stub_server, error_class=CustomError)

    assert str(raised.value) == expected
    assert not raised.value.no_backoff


def test_http_error_keeps_the_keep_alive_connection(stub_server):
    """A complete error answer leaves the socket usable, so the next call reuses it."""
    stub_server.reply(stub_server.Reply(status=500, body=b"busy"), stub_server.Reply(json_body={"ok": 1}))

    with pytest.raises(LlmProviderError):
        _get(stub_server)
    assert _get(stub_server) == {"ok": 1}

    assert stub_server.requests[0].client_port == stub_server.requests[1].client_port


@pytest.mark.parametrize("body", [b"not json", b"", b'{"truncated": '], ids=["text", "empty", "truncated"])
def test_invalid_json_raises_the_error_class(stub_server, body):
    stub_server.reply(body=body)

    with pytest.raises(CustomError, match="Server returned invalid JSON."):
        _get(stub_server, error_class=CustomError)


@pytest.mark.parametrize("status", [200, 502])
def test_non_utf8_body_is_a_provider_error(stub_server, status):
    """E.g. a proxy's Latin-1 error page: batch loops only catch LlmProviderError."""
    stub_server.reply(status=status, body="Passerelle défaillante".encode("latin-1"))

    with pytest.raises(LlmProviderError):
        _get(stub_server)


def test_connection_dropped_mid_body_is_a_provider_error_and_the_next_call_recovers(stub_server):
    stub_server.reply(
        stub_server.Reply(json_body={"response": "x" * 100}, truncate_after=10),
        stub_server.Reply(json_body={"ok": True}),
    )

    with pytest.raises(LlmProviderError):
        _get(stub_server)
    assert _get(stub_server) == {"ok": True}


def test_non_http_answer_is_a_provider_error(stub_server):
    """E.g. the endpoint points at an SSH port: fetching models must fail with a message, not crash."""
    stub_server.reply(raw=b"SSH-2.0-OpenSSH_9.6\r\n")

    with pytest.raises(CustomError, match="Could not reach server"):
        _get(stub_server, error_class=CustomError)


def test_connection_dropped_before_answering_a_fresh_connection_is_not_retried(stub_server):
    stub_server.reply(drop=True)

    with pytest.raises(CustomError, match="Could not reach server"):
        _get(stub_server, error_class=CustomError)

    assert len(stub_server.requests) == 1
    assert _pool() == {}


def test_connection_refused_raises_could_not_reach_server(closed_port):
    started = time.monotonic()
    with pytest.raises(CustomError, match="Could not reach server"):
        request_json(f"127.0.0.1:{closed_port}", DEFAULT, "/api/tags", timeout=2.0, error_class=CustomError)
    assert time.monotonic() - started < 2.0
    assert _pool() == {}


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


def test_server_slower_than_the_timeout_raises_the_error_class(stub_server):
    stub_server.reply(json_body={}, hold=threading.Event())

    started = time.monotonic()
    with pytest.raises(CustomError):
        _get(stub_server, timeout=0.2, error_class=CustomError)

    assert time.monotonic() - started < 2.0
    assert _pool() == {}, "a timed-out connection must not be reused"


def test_timeout_message_says_timed_out(stub_server):
    """Batch loops look for 'Timed out' in the message to flag the image as timed out."""
    stub_server.reply(json_body={}, hold=threading.Event())

    with pytest.raises(LlmProviderError, match="^Timed out after"):
        _get(stub_server, timeout=0.2)


def test_timeout_on_a_reused_connection_is_not_retried(stub_server):
    stub_server.reply(stub_server.Reply(json_body={"warm": 1}), stub_server.Reply(json_body={}, hold=threading.Event()))
    _get(stub_server)  # leaves a pooled connection behind

    with pytest.raises(LlmProviderError):
        _get(stub_server, timeout=0.2)

    assert len(stub_server.requests) == 2, "the timed-out generation was sent to the server again"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancelled_before_sending_raises_without_contacting_the_server(stub_server):
    cancellation = LlmRequestCancellation()
    cancellation.cancel()

    with pytest.raises(LlmProviderCancelled):
        _get(stub_server, cancellation=cancellation)

    assert stub_server.requests == []


def test_cancel_after_a_reused_connection_raises_the_provider_class_and_drops_it(stub_server):
    stub_server.reply(json_body={"models": []})
    _get(stub_server)  # leaves a pooled connection
    assert _pool()
    cancellation = LlmRequestCancellation()
    cancellation.cancel()

    with pytest.raises(CustomCancelled):
        _get(stub_server, cancellation=cancellation, cancel_class=CustomCancelled)

    assert _pool() == {}


def test_cancel_while_waiting_then_server_answers_raises_cancelled_not_the_answer(stub_server, response_begun):
    release = threading.Event()
    stub_server.reply(json_body={"response": "late"}, hold=release)
    cancellation = LlmRequestCancellation()

    outcome = _in_thread(lambda: _get(stub_server, cancellation=cancellation))
    assert response_begun.wait(5)
    cancellation.cancel()
    release.set()
    outcome["thread"].join(5)

    assert not outcome["thread"].is_alive()
    assert isinstance(outcome.get("error"), LlmProviderCancelled), outcome
    assert "Request stopped." in str(outcome["error"])


def test_cancel_in_flight_wins_over_the_timeout_error(stub_server, response_begun):
    """Even when the blocked read only ends at the timeout, the caller sees a cancellation."""
    stub_server.reply(json_body={}, hold=threading.Event())
    cancellation = LlmRequestCancellation()

    outcome = _in_thread(
        lambda: _get(stub_server, timeout=0.3, cancellation=cancellation, cancel_class=CustomCancelled)
    )
    assert response_begun.wait(5)
    cancellation.cancel()
    outcome["thread"].join(5)

    assert isinstance(outcome.get("error"), CustomCancelled), outcome


def test_cancel_just_after_sending_raises_cancelled(stub_server, monkeypatch):
    """Stop pressed right as the request finished sending: the closed connection is 'Idle'."""
    import http.client

    cancellation = LlmRequestCancellation()
    original = http.client.HTTPConnection.getresponse

    def getresponse(self):
        cancellation.cancel()
        return original(self)

    monkeypatch.setattr(http.client.HTTPConnection, "getresponse", getresponse)

    with pytest.raises(LlmProviderCancelled):
        _get(stub_server, cancellation=cancellation)


def test_cancel_after_the_request_finished_does_not_close_the_pooled_connection(stub_server):
    cancellation = LlmRequestCancellation()
    _get(stub_server, cancellation=cancellation)
    cancellation.cancel()

    _get(stub_server)

    assert stub_server.requests[0].client_port == stub_server.requests[1].client_port


def _wait_until_reading_body(cancellation: LlmRequestCancellation) -> None:
    """Bounded wait until the in-flight connection has parsed the status line and headers."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        resources = list(cancellation._active_resources)
        connections = [r for r in resources if isinstance(r, http.client.HTTPConnection)]
        if any(getattr(conn, "_HTTPConnection__response", None) is not None for conn in connections):
            return
        # "Connection: close": getresponse() passed the socket on to the response.
        if any(isinstance(r, socket.socket) for r in resources) and all(c.sock is None for c in connections):
            return
        time.sleep(0.005)
    raise AssertionError("the client never got the response headers")


@pytest.mark.parametrize("phase", ["waiting-for-headers", "reading-body", "reading-body-connection-close"])
def test_cancel_in_flight_returns_promptly(stub_server, response_begun, phase):
    """Stop is pressed on the GUI thread: cancel() must not block and the worker must stop."""
    server_gate = threading.Event()
    if phase == "waiting-for-headers":
        stub_server.reply(json_body={}, hold=server_gate)
    else:
        headers = {"Connection": "close"} if phase.endswith("connection-close") else {}
        stub_server.reply(
            json_body={"response": "x" * 100}, headers=headers, truncate_after=10, hold_body=server_gate
        )
    cancellation = LlmRequestCancellation()
    worker = _in_thread(lambda: _get(stub_server, timeout=5.0, cancellation=cancellation))
    assert response_begun.wait(5)
    if phase != "waiting-for-headers":
        _wait_until_reading_body(cancellation)

    canceller = _in_thread(cancellation.cancel)
    canceller["thread"].join(0.3)
    worker["thread"].join(0.3)
    cancel_returned = not canceller["thread"].is_alive()
    worker_stopped = not worker["thread"].is_alive()
    server_gate.set()  # unblock everything either way
    canceller["thread"].join(5)
    worker["thread"].join(5)

    assert isinstance(worker.get("error"), LlmProviderCancelled), worker
    assert (cancel_returned, worker_stopped) == (True, True)


# ---------------------------------------------------------------------------
# Connection pooling
# ---------------------------------------------------------------------------


def test_sequential_requests_reuse_one_pooled_connection(stub_server):
    _get(stub_server)
    _get(stub_server)
    _get(stub_server)

    ports = {request.client_port for request in stub_server.requests}
    assert len(ports) == 1
    assert list(_pool()) == [(f"127.0.0.1:{stub_server.port}", "http")]


def test_each_thread_has_its_own_connection(stub_server):
    _get(stub_server)
    outcome = _in_thread(lambda: _get(stub_server))
    outcome["thread"].join(5)

    assert "error" not in outcome
    assert stub_server.requests[0].client_port != stub_server.requests[1].client_port


@pytest.mark.parametrize("server_form", ["url", "bare-host-port", "with-path"])
def test_discard_pooled_connection_for_server_forces_a_new_connection(stub_server, server_form):
    server = {
        "url": stub_server.url,
        "bare-host-port": f"127.0.0.1:{stub_server.port}",
        "with-path": f"{stub_server.url}/api/",
    }[server_form]
    _get(stub_server)

    discard_pooled_connection_for_server(server, DEFAULT)
    assert _pool() == {}
    _get(stub_server)

    assert stub_server.requests[0].client_port != stub_server.requests[1].client_port


def test_discard_for_an_unknown_server_or_empty_pool_is_harmless(stub_server):
    discard_pooled_connection_for_server("127.0.0.1:9", DEFAULT)  # empty pool
    _get(stub_server)
    discard_pooled_connection_for_server("127.0.0.1:9", DEFAULT)

    assert len(_pool()) == 1


def test_stale_pooled_connection_is_retried_once_on_a_fresh_one(stub_server):
    """A server that dropped the idle keep-alive socket must not fail the next request."""
    stub_server.reply(
        stub_server.Reply(json_body={"first": 1}, close_after=True),
        stub_server.Reply(json_body={"second": 2}),
    )
    assert _get(stub_server) == {"first": 1}
    assert stub_server.connection_closed.wait(5)

    assert _get(stub_server) == {"second": 2}

    assert len(stub_server.requests) == 2
    assert stub_server.requests[0].client_port != stub_server.requests[1].client_port
    # The fresh connection replaced the stale one in the pool and is reused next time.
    _get(stub_server)
    assert stub_server.requests[2].client_port == stub_server.requests[1].client_port


def test_stale_pooled_connection_to_a_server_that_went_away_raises_after_one_retry(stub_server):
    stub_server.reply(json_body={}, close_after=True)
    _get(stub_server)
    assert stub_server.connection_closed.wait(5)
    stub_server.close()

    with pytest.raises(CustomError, match="Could not reach server"):
        _get(stub_server, error_class=CustomError)

    assert _pool() == {}
