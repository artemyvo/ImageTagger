"""Fixtures for the LLM provider tests: a scripted local HTTP server and clean module state.

``stub_server`` answers on 127.0.0.1 with whatever the test queued via
``stub_server.reply(...)`` (``stub_server.Reply`` builds sequences) and
records every request.  Replies can hold until an event is set, send a
partial body and drop the connection, answer with non-HTTP bytes, or close
the keep-alive connection right after answering.
"""
from __future__ import annotations

import json
import socket
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

from imagetagger.providers import http_request
from imagetagger.utils import image_prep

# Upper bound for any wait inside the stub server, so a broken test cannot hang it.
_MAX_WAIT = 10.0


@dataclass
class Reply:
    """One scripted answer.  ``json_body`` wins over ``body`` when both are given."""

    status: int = 200
    json_body: Any = None
    body: bytes = b""
    headers: Dict[str, str] = field(default_factory=dict)
    # Wait for this event before sending anything (the status line included).
    hold: Optional[threading.Event] = None
    # Send the headers and this many body bytes, then hold on ``hold_body``
    # (if given) and finally drop the connection without sending the rest.
    truncate_after: Optional[int] = None
    hold_body: Optional[threading.Event] = None
    # Close the keep-alive connection after a complete answer, without saying so
    # in the headers - like a server whose idle timeout fired.
    close_after: bool = False
    # Drop the connection after reading the request, without any answer.
    drop: bool = False
    # Write these bytes instead of an HTTP answer, then close the connection.
    raw: Optional[bytes] = None

    def payload(self) -> bytes:
        if self.json_body is not None:
            return json.dumps(self.json_body).encode("utf-8")
        return self.body


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes
    client_port: int

    @property
    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class StubServer:
    Reply = Reply

    def __init__(self) -> None:
        self.requests: List[RecordedRequest] = []
        self.received = threading.Event()  # set on every request; tests clear it as needed
        self.connection_closed = threading.Event()  # set when the server drops any connection
        self._replies: List[Reply] = []
        self._lock = threading.Lock()
        self._events: List[threading.Event] = []

        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive, so pooling is observable

            def log_message(self, *args) -> None:
                pass

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                with stub._lock:
                    stub.requests.append(
                        RecordedRequest(
                            method=self.command,
                            path=self.path,
                            headers={k.lower(): v for k, v in self.headers.items()},
                            body=body,
                            client_port=self.client_address[1],
                        )
                    )
                    reply = stub._replies.pop(0) if len(stub._replies) > 1 else (
                        stub._replies[0] if stub._replies else Reply(json_body={})
                    )
                stub.received.set()

                if reply.hold is not None:
                    reply.hold.wait(_MAX_WAIT)
                if reply.drop or reply.raw is not None:
                    if reply.raw is not None:
                        self.wfile.write(reply.raw)
                        self.wfile.flush()
                    self.close_connection = True
                    return
                payload = reply.payload()
                try:
                    self.send_response(reply.status)
                    headers = {"Content-Type": "application/json", **reply.headers}
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    if reply.truncate_after is not None:
                        self.wfile.write(payload[: reply.truncate_after])
                        self.wfile.flush()
                        if reply.hold_body is not None:
                            reply.hold_body.wait(_MAX_WAIT)
                        self.close_connection = True
                        return
                    self.wfile.write(payload)
                    self.wfile.flush()
                except OSError:
                    self.close_connection = True
                    return
                if reply.close_after:
                    self.close_connection = True

            do_GET = _handle
            do_POST = _handle

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address) -> None:
                pass  # clients hanging up mid-request is what several tests do on purpose

            def shutdown_request(self, request) -> None:
                try:
                    request.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                super().shutdown_request(request)
                stub.connection_closed.set()

        self._server = Server(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()

    def reply(self, *replies: Reply, **kwargs) -> "StubServer":
        """Queue replies in order; the last one keeps answering.  ``reply(status=..)`` makes one."""
        if kwargs:
            replies = replies + (Reply(**kwargs),)
        with self._lock:
            self._replies = list(replies)
        for reply in replies:
            for event in (reply.hold, reply.hold_body):
                if event is not None:
                    self._events.append(event)
        return self

    @property
    def last(self) -> RecordedRequest:
        assert self.requests, "the stub server saw no request"
        return self.requests[-1]

    def close(self) -> None:
        for event in self._events:
            event.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(_MAX_WAIT)


@pytest.fixture
def stub_server() -> Iterator[StubServer]:
    server = StubServer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def response_begun(monkeypatch) -> threading.Event:
    """Set once a client starts reading a response's status line.

    From then on the request is truly in flight: cancelling cannot land in the
    gap between sending the request and starting to read the answer.
    """
    import http.client

    begun = threading.Event()
    original = http.client.HTTPResponse.begin

    def begin(self):
        begun.set()
        return original(self)

    monkeypatch.setattr(http.client.HTTPResponse, "begin", begin)
    return begun


@pytest.fixture
def closed_port() -> int:
    """A 127.0.0.1 port with nothing listening on it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture(autouse=True)
def fresh_connection_pool(monkeypatch) -> Iterator[None]:
    """Each test starts with an empty per-thread connection pool; its connections are closed after."""
    local = threading.local()
    monkeypatch.setattr(http_request, "_thread_local", local)
    yield
    for conn in list(getattr(local, "connections", {}).values()):
        try:
            conn.close()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def fresh_image_prep(monkeypatch) -> None:
    """Module-level image preparation settings and cache are restored after each test."""
    monkeypatch.setattr(image_prep, "_max_image_pixels", image_prep.DEFAULT_MAX_IMAGE_PIXELS)
    monkeypatch.setattr(image_prep, "_resize_warning_pending", False)
    monkeypatch.setattr(image_prep, "_prepared_image_cache", OrderedDict())


@pytest.fixture
def tiny_png(tmp_path) -> Path:
    """A 4x4 PNG, well under the resize limit, so it is sent byte for byte."""
    from PIL import Image

    path = tmp_path / "tiny.png"
    Image.new("RGB", (4, 4), "red").save(path)
    return path
