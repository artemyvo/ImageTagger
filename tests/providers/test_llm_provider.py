"""The provider-neutral layer: GeneratedText, server URL normalization, cancellation and provider dispatch."""
from __future__ import annotations

import base64
from typing import List

import pytest

from imagetagger.providers import ollama, openai_compat
from imagetagger.providers.llm_provider import (
    DEFAULT_VISION_PROVIDER,
    GeneratedText,
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
    _DefaultVisionProvider,
    _OllamaSession,
    _OpenAiCompatSession,
    normalize_server_url,
)
from imagetagger.utils.llm_queries import LlmQueryError

OLLAMA_DEFAULT = ollama.DEFAULT_OLLAMA_SERVER
OPENAI_DEFAULT = openai_compat.DEFAULT_OPENAI_COMPAT_SERVER


# ---------------------------------------------------------------------------
# GeneratedText and the error types
# ---------------------------------------------------------------------------


def test_generated_text_is_a_str_with_a_thinking_trace():
    text = GeneratedText("answer", thinking="because")

    assert isinstance(text, str)
    assert text == "answer"
    assert text.thinking == "because"
    assert text.upper() == "ANSWER"
    assert f"{text}!" == "answer!"
    assert {text: 1}["answer"] == 1


@pytest.mark.parametrize("thinking", [None, 3, ["x"]], ids=["none", "number", "list"])
def test_generated_text_non_string_thinking_becomes_empty(thinking):
    assert GeneratedText("a", thinking=thinking).thinking == ""


def test_generated_text_defaults():
    text = GeneratedText()
    assert text == ""
    assert text.thinking == ""


def test_generated_text_str_methods_drop_the_trace():
    """Derived strings are plain str: callers must read .thinking off the raw answer."""
    derived = GeneratedText(" a ", thinking="t").strip()
    assert type(derived) is str
    assert getattr(derived, "thinking", None) is None


def test_provider_errors_are_query_errors_and_back_off_by_default():
    error = LlmProviderError("x")
    assert isinstance(error, LlmQueryError)
    assert error.no_backoff is False
    assert issubclass(LlmProviderCancelled, LlmProviderError)


def test_no_backoff_set_on_one_error_does_not_leak_to_others():
    first = LlmProviderError("a")
    first.no_backoff = True
    assert LlmProviderError("b").no_backoff is False
    assert LlmProviderError.no_backoff is False


# ---------------------------------------------------------------------------
# normalize_server_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server, expected",
    [
        ("", "http://default:1"),
        ("   ", "http://default:1"),
        ("localhost:11434", "http://localhost:11434"),
        ("  192.168.1.5:8000  ", "http://192.168.1.5:8000"),
        ("http://host:8000/", "http://host:8000"),
        ("https://api.example.com/v1/chat?x=1#f", "https://api.example.com"),
        ("HTTP://Host:8000", "http://Host:8000"),
        ("http://user:pw@host:1", "http://user:pw@host:1"),
        ("http://[::1]:8000/v1", "http://[::1]:8000"),
    ],
)
def test_normalize_server_url(server, expected):
    assert normalize_server_url(server, "http://default:1") == expected


@pytest.mark.parametrize("server", ["ftp://host", "http://", "https:///path", "file:///etc/hosts"])
def test_normalize_server_url_rejects_invalid_addresses(server):
    with pytest.raises(LlmProviderError, match="Enter a valid server address."):
        normalize_server_url(server, "http://default:1")


def test_normalize_server_url_rejects_a_malformed_ipv6_address():
    """normalize_endpoint wraps it, but request_json (so ollama/openai_compat.fetch_models) does not."""
    with pytest.raises(LlmProviderError):
        normalize_server_url("http://[::1", "http://default:1")


def test_normalize_server_url_honours_allowed_schemes():
    assert normalize_server_url("ws://host:1", "", allowed_schemes={"ws"}) == "ws://host:1"
    with pytest.raises(LlmProviderError):
        normalize_server_url("http://host:1", "", allowed_schemes={"https"})


def test_normalize_server_url_returns_the_default_verbatim():
    """The default is trusted as given, even if it would not normalize."""
    assert normalize_server_url("", "http://default:1/") == "http://default:1/"


# ---------------------------------------------------------------------------
# LlmRequestCancellation
# ---------------------------------------------------------------------------


class Resource:
    def __init__(self, fail: bool = False) -> None:
        self.closed = 0
        self._fail = fail

    def close(self) -> None:
        self.closed += 1
        if self._fail:
            raise OSError("already closed")


def test_new_cancellation_is_not_cancelled():
    cancellation = LlmRequestCancellation()
    assert not cancellation.is_cancelled()
    cancellation.raise_if_cancelled()  # no error


def test_cancel_marks_it_cancelled_and_raise_if_cancelled_raises():
    cancellation = LlmRequestCancellation()
    cancellation.cancel()

    assert cancellation.is_cancelled()
    with pytest.raises(LlmProviderCancelled, match="Request stopped."):
        cancellation.raise_if_cancelled()


def test_cancel_closes_active_resources_once():
    cancellation = LlmRequestCancellation()
    first, second = Resource(), Resource()
    cancellation.set_active_resource(first)
    cancellation.set_active_resource(second)

    cancellation.cancel()
    cancellation.cancel()

    assert (first.closed, second.closed) == (1, 1)


def test_cleared_resource_is_not_closed():
    cancellation = LlmRequestCancellation()
    resource = Resource()
    cancellation.set_active_resource(resource)
    cancellation.clear_active_resource(resource)
    cancellation.clear_active_resource(resource)  # clearing twice is harmless

    cancellation.cancel()

    assert resource.closed == 0


def test_cancel_ignores_resources_that_fail_to_close_or_cannot_close():
    cancellation = LlmRequestCancellation()
    failing, fine = Resource(fail=True), Resource()
    for resource in (failing, object(), fine):
        cancellation.set_active_resource(resource)

    cancellation.cancel()

    assert failing.closed == 1
    assert fine.closed == 1


def test_setting_a_resource_after_cancel_closes_it_and_raises():
    cancellation = LlmRequestCancellation()
    cancellation.cancel()
    resource = Resource(fail=True)

    with pytest.raises(LlmProviderCancelled):
        cancellation.set_active_resource(resource)

    assert resource.closed == 1


# ---------------------------------------------------------------------------
# _DefaultVisionProvider: endpoints and dispatch
# ---------------------------------------------------------------------------


def test_default_provider_basics():
    assert isinstance(DEFAULT_VISION_PROVIDER, _DefaultVisionProvider)
    assert DEFAULT_VISION_PROVIDER.display_name == "LLM"
    assert DEFAULT_VISION_PROVIDER.default_endpoint == OLLAMA_DEFAULT


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("", OLLAMA_DEFAULT),
        ("  ", OLLAMA_DEFAULT),
        ("localhost:11434/api", "http://localhost:11434"),
        ("http://gpu-box:8000/v1/", "http://gpu-box:8000"),
        ("https://api.example.com", "https://api.example.com"),
    ],
)
def test_normalize_endpoint(endpoint, expected):
    assert DEFAULT_VISION_PROVIDER.normalize_endpoint(endpoint) == expected


@pytest.mark.parametrize("endpoint", ["ftp://host:1", "http://"])
def test_normalize_endpoint_rejects_invalid_addresses(endpoint):
    with pytest.raises(LlmProviderError, match="valid server address"):
        DEFAULT_VISION_PROVIDER.normalize_endpoint(endpoint)


def test_normalize_endpoint_wraps_parse_errors():
    with pytest.raises(LlmProviderError, match="Invalid IPv6 URL"):
        DEFAULT_VISION_PROVIDER.normalize_endpoint("http://[::1")


@pytest.mark.parametrize("endpoint", ["localhost:abc", "http://host:port/", "http://host:99999"])
def test_endpoint_with_a_bad_port_is_rejected(endpoint):
    """The Fetch button only catches LlmProviderError, so a typo must not escape as another exception."""
    with pytest.raises(LlmProviderError):
        DEFAULT_VISION_PROVIDER.normalize_endpoint(endpoint)
    with pytest.raises(LlmProviderError):
        DEFAULT_VISION_PROVIDER.fetch_models(endpoint, timeout=1.0)


@pytest.mark.parametrize(
    "endpoint, session_type",
    [
        ("", _OllamaSession),
        ("127.0.0.1:11434", _OllamaSession),
        ("https://ollama.lan:11434/", _OllamaSession),
        ("127.0.0.1:8000", _OpenAiCompatSession),
        ("http://localhost:1234/v1", _OpenAiCompatSession),
        ("https://api.example.com", _OpenAiCompatSession),
    ],
)
def test_create_session_picks_the_interface_by_port(endpoint, session_type):
    session = DEFAULT_VISION_PROVIDER.create_session(endpoint, "model-x")

    assert type(session) is session_type
    assert session.endpoint == DEFAULT_VISION_PROVIDER.normalize_endpoint(endpoint)
    assert session.model_name == "model-x"


def test_create_session_rejects_an_invalid_endpoint():
    with pytest.raises(LlmProviderError):
        DEFAULT_VISION_PROVIDER.create_session("ftp://x", "m")


@pytest.fixture
def fetch_calls(monkeypatch) -> List[tuple]:
    calls: List[tuple] = []

    def fake(kind):
        def _fetch(server, timeout=5.0):
            calls.append((kind, server, timeout))
            return [f"{kind}-model"]

        return _fetch

    monkeypatch.setattr(ollama, "fetch_models", fake("ollama"))
    monkeypatch.setattr(openai_compat, "fetch_models", fake("openai"))
    return calls


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("", ("ollama", OLLAMA_DEFAULT, 7.0)),
        ("gpu:11434/x", ("ollama", "http://gpu:11434", 7.0)),
        ("gpu:8000", ("openai", "http://gpu:8000", 7.0)),
    ],
)
def test_fetch_models_dispatches_with_the_normalized_endpoint(fetch_calls, endpoint, expected):
    models = DEFAULT_VISION_PROVIDER.fetch_models(endpoint, timeout=7.0)

    assert fetch_calls == [expected]
    assert models == [f"{expected[0]}-model"]


def test_fetch_models_against_an_openai_compatible_server(stub_server):
    stub_server.reply(json_body={"data": [{"id": "served-model"}]})

    assert DEFAULT_VISION_PROVIDER.fetch_models(f"127.0.0.1:{stub_server.port}", timeout=5.0) == ["served-model"]
    assert stub_server.last.path == "/v1/models"


# ---------------------------------------------------------------------------
# Sessions end to end
# ---------------------------------------------------------------------------


def test_openai_session_generates_via_chat_completions(stub_server, tiny_png):
    stub_server.reply(json_body={"choices": [{"message": {"content": "tags", "reasoning": "why"}}]})
    session = DEFAULT_VISION_PROVIDER.create_session(stub_server.url, "m")

    result = session.generate(tiny_png, "prompt", timeout=5.0, thread_count=4, temperature=0.1, think=True)

    assert (result, result.thinking) == ("tags", "why")
    sent = stub_server.last.json
    assert stub_server.last.path == "/v1/chat/completions"
    assert sent["temperature"] == 0.1
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}
    assert "options" not in sent and "num_thread" not in str(sent)  # thread_count is Ollama-only


def test_ollama_session_generates_via_api_generate(stub_server, tiny_png):
    """Built directly: the stub cannot listen on 11434, which is what selects Ollama."""
    stub_server.reply(json_body={"response": "tags", "thinking": "why"})
    session = _OllamaSession(endpoint=stub_server.url, model_name="m")

    result = session.generate(tiny_png, "prompt", timeout=5.0, thread_count=4, temperature=0.1, think=False)

    assert (result, result.thinking) == ("tags", "why")
    sent = stub_server.last.json
    assert stub_server.last.path == "/api/generate"
    assert sent["options"] == {"num_thread": 4, "temperature": 0.1}
    assert sent["think"] is False
    assert sent["images"] == [base64.b64encode(tiny_png.read_bytes()).decode("ascii")]


@pytest.mark.parametrize("session_type", [_OllamaSession, _OpenAiCompatSession])
def test_session_passes_cancellation_through(stub_server, tiny_png, session_type):
    cancellation = LlmRequestCancellation()
    cancellation.cancel()
    session = session_type(endpoint=stub_server.url, model_name="m")

    with pytest.raises(LlmProviderCancelled):
        session.generate(tiny_png, "prompt", timeout=5.0, cancellation=cancellation)

    assert stub_server.requests == []


def test_sessions_are_frozen_value_objects():
    session = _OllamaSession(endpoint="http://a:1", model_name="m")
    assert session == _OllamaSession(endpoint="http://a:1", model_name="m")
    with pytest.raises(Exception):
        session.model_name = "other"  # type: ignore[misc]
