"""Ollama provider: model listing, the /api/generate payload, and how answers and failures come back."""
from __future__ import annotations

import base64
import json
import threading

import pytest
from PIL import Image

from imagetagger import config as _config
from imagetagger.providers import http_request, ollama
from imagetagger.providers.llm_provider import (
    GeneratedText,
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
)
from imagetagger.providers.ollama import (
    OllamaCancelled,
    OllamaConnection,
    OllamaError,
    fetch_models,
    generate_with_image,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _generate(stub_server, image, timeout=5.0, **kwargs):
    return generate_with_image(
        OllamaConnection(stub_server.url, "llava:7b"), image, "Describe it.", timeout=timeout, **kwargs
    )


# ---------------------------------------------------------------------------
# fetch_models
# ---------------------------------------------------------------------------


def test_fetch_models_lists_names_from_api_tags(stub_server):
    stub_server.reply(
        json_body={
            "models": [
                {"name": "llava:7b", "size": 1},
                {"name": "  qwen2.5vl:3b  "},
                {"name": "   "},
                {"name": 42},
                {"model": "no-name"},
                "not-a-dict",
            ]
        }
    )

    assert fetch_models(stub_server.url, timeout=5.0) == ["llava:7b", "qwen2.5vl:3b"]
    assert stub_server.last.method == "GET"
    assert stub_server.last.path == "/api/tags"


def test_fetch_models_without_models_key_is_empty(stub_server):
    stub_server.reply(json_body={})
    assert fetch_models(stub_server.url, timeout=5.0) == []


def test_fetch_models_with_a_non_list_models_value_is_an_error(stub_server):
    stub_server.reply(json_body={"models": {"name": "x"}})
    with pytest.raises(OllamaError, match="did not include a models list"):
        fetch_models(stub_server.url, timeout=5.0)


@pytest.mark.parametrize("body", [[], "text", None], ids=["list", "string", "null"])
def test_fetch_models_with_a_non_object_answer_is_an_error(stub_server, body):
    stub_server.reply(body=json.dumps(body).encode())
    with pytest.raises(OllamaError):
        fetch_models(stub_server.url, timeout=5.0)


@pytest.mark.parametrize("form", ["{host}:{port}", "  http://{host}:{port}/api/  ", "{host}:{port}/x?y=1"])
def test_fetch_models_normalizes_the_endpoint(stub_server, form):
    server = form.format(host="127.0.0.1", port=stub_server.port)
    fetch_models(server, timeout=5.0)
    assert stub_server.last.path == "/api/tags"


def test_fetch_models_http_error_is_an_ollama_error(stub_server):
    stub_server.reply(status=500, body=b"boom")
    with pytest.raises(OllamaError, match="HTTP 500. boom"):
        fetch_models(stub_server.url, timeout=5.0)


def test_fetch_models_unreachable_server_is_an_ollama_error(closed_port):
    with pytest.raises(OllamaError, match="Could not reach server"):
        fetch_models(f"127.0.0.1:{closed_port}", timeout=2.0)


# ---------------------------------------------------------------------------
# generate_with_image: the request
# ---------------------------------------------------------------------------


def test_generate_posts_model_prompt_and_base64_image(stub_server, tiny_png):
    stub_server.reply(json_body={"response": "a red square"})

    _generate(stub_server, tiny_png)

    request = stub_server.last
    assert request.method == "POST"
    assert request.path == "/api/generate"
    assert request.headers["content-type"] == "application/json"
    assert request.json == {
        "model": "llava:7b",
        "prompt": "Describe it.",
        "images": [base64.b64encode(tiny_png.read_bytes()).decode("ascii")],
        "stream": False,
    }


@pytest.mark.parametrize("think, expected", [(True, True), (False, False), (1, True), (0, False)])
def test_think_is_sent_as_a_bool(stub_server, tiny_png, think, expected):
    """``False`` is sent explicitly: Ollama thinks by default on models that can."""
    stub_server.reply(json_body={"response": "ok"})
    _generate(stub_server, tiny_png, think=think)
    assert stub_server.last.json["think"] is expected


def test_think_none_leaves_the_flag_out(stub_server, tiny_png):
    stub_server.reply(json_body={"response": "ok"})
    _generate(stub_server, tiny_png, think=None)
    assert "think" not in stub_server.last.json


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, None),
        ({"thread_count": 8}, {"num_thread": 8}),
        ({"thread_count": 0}, {"num_thread": 1}),
        ({"thread_count": -3}, {"num_thread": 1}),
        ({"thread_count": 3.9}, {"num_thread": 3}),
        ({"temperature": 0}, {"temperature": 0.0}),
        ({"temperature": "0.7"}, {"temperature": 0.7}),
        ({"thread_count": 2, "temperature": 1.5}, {"num_thread": 2, "temperature": 1.5}),
    ],
    ids=["none", "threads", "zero-threads", "negative-threads", "float-threads", "zero-temp", "str-temp", "both"],
)
def test_thread_count_and_temperature_map_to_options(stub_server, tiny_png, kwargs, expected):
    stub_server.reply(json_body={"response": "ok"})
    _generate(stub_server, tiny_png, **kwargs)
    assert stub_server.last.json.get("options") == expected


def test_webp_is_transcoded_to_png_for_ollama(stub_server, tmp_path):
    path = tmp_path / "tiny.webp"
    Image.new("RGB", (4, 4), "blue").save(path, format="WEBP")
    stub_server.reply(json_body={"response": "ok"})

    _generate(stub_server, path)

    sent = base64.b64decode(stub_server.last.json["images"][0])
    assert sent.startswith(PNG_SIGNATURE)


def test_unreadable_image_fails_before_contacting_the_server(stub_server, tmp_path):
    with pytest.raises(LlmProviderError):
        _generate(stub_server, tmp_path / "missing.png")
    assert stub_server.requests == []


# ---------------------------------------------------------------------------
# generate_with_image: the answer
# ---------------------------------------------------------------------------


def test_non_object_answer_is_an_ollama_error(stub_server, tiny_png):
    stub_server.reply(json_body=["not", "an", "object"])
    with pytest.raises(OllamaError):
        _generate(stub_server, tiny_png)


def test_answer_is_stripped_generated_text_with_thinking(stub_server, tiny_png):
    stub_server.reply(json_body={"response": "\n  red, square \n", "thinking": "  It is red.  ", "done": True})

    result = _generate(stub_server, tiny_png, think=True)

    assert isinstance(result, GeneratedText)
    assert result == "red, square"
    assert result.thinking == "It is red."


@pytest.mark.parametrize("thinking", [None, 42, ["x"]], ids=["missing", "number", "list"])
def test_answer_without_a_string_thinking_field_has_empty_thinking(stub_server, tiny_png, thinking):
    body = {"response": "ok"}
    if thinking is not None:
        body["thinking"] = thinking
    stub_server.reply(json_body=body)

    result = _generate(stub_server, tiny_png)

    assert result == "ok"
    assert result.thinking == ""


@pytest.mark.parametrize(
    "body",
    [{}, {"response": ""}, {"response": "  \n "}, {"response": None}, {"response": ["a"]}],
    ids=["missing", "empty", "blank", "null", "list"],
)
def test_empty_answer_is_an_error_without_backoff(stub_server, tiny_png, body):
    """An empty answer says nothing about load, so auto-threading must not back off."""
    stub_server.reply(json_body=body)

    with pytest.raises(OllamaError, match=r"^Ollama returned an empty response\.$") as raised:
        _generate(stub_server, tiny_png)

    assert raised.value.no_backoff is True
    assert raised.value.context_exhausted is False


def test_empty_answer_reports_the_done_reason(stub_server, tiny_png):
    stub_server.reply(json_body={"response": "", "done_reason": "length"})

    with pytest.raises(OllamaError) as raised:
        _generate(stub_server, tiny_png)

    assert str(raised.value) == "Ollama returned an empty response (done_reason='length')."


@pytest.mark.parametrize("done_reason", ["", 7], ids=["blank", "non-string"])
def test_empty_answer_ignores_a_useless_done_reason(stub_server, tiny_png, done_reason):
    stub_server.reply(json_body={"response": "", "done_reason": done_reason})
    with pytest.raises(OllamaError, match=r"^Ollama returned an empty response\.$"):
        _generate(stub_server, tiny_png)


def test_empty_answer_after_thinking_flags_context_exhaustion(stub_server, tiny_png, capsys):
    stub_server.reply(json_body={"response": "", "thinking": "hmm " * 50, "eval_count": 4096})

    with pytest.raises(OllamaError) as raised:
        _generate(stub_server, tiny_png, think=True)

    assert raised.value.context_exhausted is True
    assert raised.value.no_backoff is True
    out = capsys.readouterr().out
    assert "context window likely exhausted" in out
    assert "eval_count=4096" in out


def test_blank_thinking_is_not_context_exhaustion(stub_server, tiny_png):
    stub_server.reply(json_body={"response": "", "thinking": "   "})
    with pytest.raises(OllamaError) as raised:
        _generate(stub_server, tiny_png)
    assert raised.value.context_exhausted is False


def test_empty_answer_discards_the_pooled_connection(stub_server, tiny_png):
    """The next retry starts on a fresh socket."""
    stub_server.reply(stub_server.Reply(json_body={"response": ""}), stub_server.Reply(json_body={"response": "ok"}))

    with pytest.raises(OllamaError):
        _generate(stub_server, tiny_png)
    assert getattr(http_request._thread_local, "connections", {}) == {}
    assert _generate(stub_server, tiny_png) == "ok"

    assert stub_server.requests[0].client_port != stub_server.requests[1].client_port


def test_good_answer_keeps_the_pooled_connection(stub_server, tiny_png):
    stub_server.reply(json_body={"response": "ok"})
    _generate(stub_server, tiny_png)
    _generate(stub_server, tiny_png)
    assert stub_server.requests[0].client_port == stub_server.requests[1].client_port


def test_debug_prompts_dump_hides_the_image_and_prompt(stub_server, tiny_png, monkeypatch, capsys):
    monkeypatch.setattr(_config, "load", lambda: {"debug_prompts": True})
    stub_server.reply(json_body={"response": "", "context": [1, 2, 3], "done_reason": "stop"})

    with pytest.raises(OllamaError):
        _generate(stub_server, tiny_png, temperature=0.2)

    out = capsys.readouterr().out
    encoded = base64.b64encode(tiny_png.read_bytes()).decode("ascii")
    assert encoded not in out
    assert "Describe it." not in out
    assert "[base64 image, " in out
    assert '"prompt": "[12 chars]"' in out
    assert '"context": "[3 tokens]"' in out
    assert str(tiny_png) in out


def test_debug_prompts_failure_still_raises_the_empty_response_error(stub_server, tiny_png, monkeypatch):
    def broken_load():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(_config, "load", broken_load)
    stub_server.reply(json_body={"response": ""})

    with pytest.raises(OllamaError, match="empty response"):
        _generate(stub_server, tiny_png)


# ---------------------------------------------------------------------------
# Errors and cancellation
# ---------------------------------------------------------------------------


def test_http_error_is_an_ollama_error_with_backoff(stub_server, tiny_png):
    """Server-side failures may be load-related, so they keep the default backoff."""
    stub_server.reply(status=500, json_body={"error": "model runner has unexpectedly stopped"})

    with pytest.raises(OllamaError, match="HTTP 500") as raised:
        _generate(stub_server, tiny_png)

    assert "unexpectedly stopped" in str(raised.value)
    assert raised.value.no_backoff is False


def test_cancelled_request_raises_a_cancellation_and_sends_nothing(stub_server, tiny_png):
    cancellation = LlmRequestCancellation()
    cancellation.cancel()

    with pytest.raises(LlmProviderCancelled):
        _generate(stub_server, tiny_png, cancellation=cancellation)

    assert stub_server.requests == []


def test_cancel_in_flight_raises_ollama_cancelled(stub_server, tiny_png, response_begun):
    stub_server.reply(json_body={"response": "late"}, hold=threading.Event())
    cancellation = LlmRequestCancellation()
    outcome = {}

    def run():
        try:
            _generate(stub_server, tiny_png, cancellation=cancellation, timeout=0.3)
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert response_begun.wait(5), outcome
    cancellation.cancel()
    thread.join(5)

    assert isinstance(outcome.get("error"), OllamaCancelled), outcome


def test_error_classes_fit_the_provider_hierarchy():
    assert issubclass(OllamaError, LlmProviderError)
    assert issubclass(OllamaCancelled, OllamaError)
    assert issubclass(OllamaCancelled, LlmProviderCancelled)
    assert ollama.OllamaCancellation is LlmRequestCancellation
