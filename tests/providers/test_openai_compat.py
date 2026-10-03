"""OpenAI-compatible provider: /v1/models, the chat-completions payload, and parsing the answer."""
from __future__ import annotations

import base64
import json
import threading

import pytest
from PIL import Image

from imagetagger import config as _config
from imagetagger.providers.llm_provider import (
    GeneratedText,
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
)
from imagetagger.providers.openai_compat import (
    OpenAiCompatCancelled,
    OpenAiCompatConnection,
    OpenAiCompatError,
    _extract_text_content,
    fetch_models,
    generate_with_image,
)


def _generate(stub_server, image, timeout=5.0, **kwargs):
    return generate_with_image(
        OpenAiCompatConnection(stub_server.url, "Qwen/Qwen2.5-VL-7B"), image, "Describe it.", timeout=timeout, **kwargs
    )


def _completion(message) -> dict:
    return {"id": "x", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


# ---------------------------------------------------------------------------
# fetch_models
# ---------------------------------------------------------------------------


def test_fetch_models_lists_ids_from_v1_models(stub_server):
    stub_server.reply(
        json_body={
            "object": "list",
            "data": [{"id": "model-a"}, {"id": "  model-b "}, {"id": ""}, {"id": 3}, {"name": "x"}, "bad"],
        }
    )

    assert fetch_models(stub_server.url, timeout=5.0) == ["model-a", "model-b"]
    assert stub_server.last.method == "GET"
    assert stub_server.last.path == "/v1/models"


@pytest.mark.parametrize("body", [{}, {"data": None}, {"data": {"id": "x"}}], ids=["missing", "null", "object"])
def test_fetch_models_without_a_data_list_is_an_error(stub_server, body):
    stub_server.reply(json_body=body)
    with pytest.raises(OpenAiCompatError, match="did not include a models list"):
        fetch_models(stub_server.url, timeout=5.0)


def test_fetch_models_with_an_empty_list_is_empty(stub_server):
    stub_server.reply(json_body={"data": []})
    assert fetch_models(stub_server.url, timeout=5.0) == []


def test_fetch_models_with_a_non_object_answer_is_an_error(stub_server):
    stub_server.reply(json_body=[{"id": "model-a"}])
    with pytest.raises(OpenAiCompatError):
        fetch_models(stub_server.url, timeout=5.0)


def test_fetch_models_normalizes_the_endpoint(stub_server):
    stub_server.reply(json_body={"data": []})
    fetch_models(f" 127.0.0.1:{stub_server.port}/v1/ ", timeout=5.0)
    assert stub_server.last.path == "/v1/models"


def test_fetch_models_http_error_is_an_openai_compat_error(stub_server):
    stub_server.reply(status=401, json_body={"error": {"message": "bad key"}})
    with pytest.raises(OpenAiCompatError, match="HTTP 401.*bad key"):
        fetch_models(stub_server.url, timeout=5.0)


# ---------------------------------------------------------------------------
# generate_with_image: the request
# ---------------------------------------------------------------------------


def test_generate_posts_a_chat_completion_with_a_data_url(stub_server, tiny_png):
    stub_server.reply(json_body=_completion({"role": "assistant", "content": "red"}))

    _generate(stub_server, tiny_png)

    request = stub_server.last
    assert request.method == "POST"
    assert request.path == "/v1/chat/completions"
    encoded = base64.b64encode(tiny_png.read_bytes()).decode("ascii")
    assert request.json == {
        "model": "Qwen/Qwen2.5-VL-7B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe it."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            }
        ],
        "stream": False,
    }


@pytest.mark.parametrize(
    "suffix, pil_format, media_type",
    [(".jpg", "JPEG", "image/jpeg"), (".webp", "WEBP", "image/webp"), (".gif", "GIF", "image/gif")],
)
def test_data_url_carries_the_image_media_type(stub_server, tmp_path, suffix, pil_format, media_type):
    """Unlike Ollama, WEBP is sent as is."""
    path = tmp_path / f"tiny{suffix}"
    Image.new("RGB", (4, 4), "green").save(path, format=pil_format)
    stub_server.reply(json_body=_completion({"content": "ok"}))

    _generate(stub_server, path)

    url = stub_server.last.json["messages"][0]["content"][1]["image_url"]["url"]
    assert url == f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, {}),
        ({"temperature": 0}, {"temperature": 0.0}),
        ({"temperature": "0.3"}, {"temperature": 0.3}),
        ({"think": True}, {"chat_template_kwargs": {"enable_thinking": True}}),
        ({"think": False}, {"chat_template_kwargs": {"enable_thinking": False}}),
        ({"think": None}, {}),
        ({"think": 1, "temperature": 1}, {"temperature": 1.0, "chat_template_kwargs": {"enable_thinking": True}}),
    ],
    ids=["none", "zero-temp", "str-temp", "think", "no-think", "think-none", "both"],
)
def test_temperature_and_think_options(stub_server, tiny_png, kwargs, expected):
    stub_server.reply(json_body=_completion({"content": "ok"}))
    _generate(stub_server, tiny_png, **kwargs)

    sent = stub_server.last.json
    extras = {key: sent[key] for key in ("temperature", "chat_template_kwargs") if key in sent}
    assert extras == expected


def test_unreadable_image_fails_before_contacting_the_server(stub_server, tmp_path):
    with pytest.raises(LlmProviderError):
        _generate(stub_server, tmp_path / "missing.png")
    assert stub_server.requests == []


# ---------------------------------------------------------------------------
# _extract_text_content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, expected",
    [
        ("  plain answer \n", "plain answer"),
        ("", ""),
        (None, ""),
        (42, ""),
        ({"type": "text", "text": "dict"}, ""),
        ([{"type": "text", "text": " one "}, {"type": "output_text", "text": "two"}], "one\ntwo"),
        ([{"type": "image_url", "image_url": {}}, {"type": "text", "text": "only"}], "only"),
        ([{"type": "text", "text": "  "}, {"type": "text", "text": None}, "raw", 7], ""),
        ([{"text": "untyped"}], ""),
        ([], ""),
    ],
    ids=[
        "string",
        "empty-string",
        "none",
        "number",
        "dict",
        "text-parts",
        "skips-non-text-parts",
        "blank-and-junk-parts",
        "untyped-part",
        "empty-list",
    ],
)
def test_extract_text_content(content, expected):
    assert _extract_text_content(content) == expected


# ---------------------------------------------------------------------------
# generate_with_image: the answer
# ---------------------------------------------------------------------------


def test_answer_is_stripped_generated_text(stub_server, tiny_png):
    stub_server.reply(json_body=_completion({"role": "assistant", "content": "  red square \n"}))

    result = _generate(stub_server, tiny_png)

    assert isinstance(result, GeneratedText)
    assert result == "red square"
    assert result.thinking == ""


def test_answer_from_content_parts(stub_server, tiny_png):
    stub_server.reply(json_body=_completion({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}))
    assert _generate(stub_server, tiny_png) == "a\nb"


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"reasoning_content": " vLLM trace "}, "vLLM trace"),
        ({"reasoning": " llama.cpp trace "}, "llama.cpp trace"),
        ({"reasoning_content": "first", "reasoning": "second"}, "first"),
        ({"reasoning_content": "  ", "reasoning": "fallback"}, "fallback"),
        ({"reasoning_content": None, "reasoning": 5}, ""),
    ],
    ids=["reasoning_content", "reasoning", "prefers-reasoning_content", "blank-falls-back", "non-strings"],
)
def test_reasoning_fields_become_thinking(stub_server, tiny_png, fields, expected):
    stub_server.reply(json_body=_completion({"content": "answer", **fields}))

    result = _generate(stub_server, tiny_png, think=True)

    assert result == "answer"
    assert result.thinking == expected


@pytest.mark.parametrize(
    "body, message",
    [
        ({}, "no completion choices"),
        ({"choices": []}, "no completion choices"),
        ({"choices": {"0": {}}}, "no completion choices"),
        ({"choices": ["text"]}, "invalid completion payload"),
        ({"choices": [{"text": "legacy"}]}, "no assistant message"),
        ({"choices": [{"message": "hi"}]}, "no assistant message"),
        (_completion({"content": ""}), "empty response"),
        (_completion({"content": None, "reasoning_content": "thought only"}), "empty response"),
        (_completion({"content": [{"type": "image_url"}]}), "empty response"),
    ],
    ids=[
        "no-choices",
        "empty-choices",
        "choices-not-list",
        "choice-not-dict",
        "no-message",
        "message-not-dict",
        "empty-content",
        "only-reasoning",
        "no-text-parts",
    ],
)
def test_unusable_answers_are_openai_compat_errors(stub_server, tiny_png, body, message):
    stub_server.reply(json_body=body)
    with pytest.raises(OpenAiCompatError, match=message):
        _generate(stub_server, tiny_png)


def test_debug_prompts_dump_hides_the_image_and_prompt(stub_server, tiny_png, monkeypatch, capsys):
    monkeypatch.setattr(_config, "load", lambda: {"debug_prompts": True})
    stub_server.reply(json_body=_completion({"content": ""}))

    with pytest.raises(OpenAiCompatError, match="empty response"):
        _generate(stub_server, tiny_png)

    out = capsys.readouterr().out
    encoded = base64.b64encode(tiny_png.read_bytes()).decode("ascii")
    assert encoded not in out
    assert "Describe it." not in out
    assert f"data:image/png;base64,[{len(encoded)} chars base64]" in out
    assert '"text": "[12 chars]"' in out
    assert str(tiny_png) in out


def test_debug_prompts_failure_still_raises_the_empty_response_error(stub_server, tiny_png, monkeypatch):
    def broken_load():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(_config, "load", broken_load)
    stub_server.reply(json_body=_completion({"content": ""}))

    with pytest.raises(OpenAiCompatError, match="empty response"):
        _generate(stub_server, tiny_png)


# ---------------------------------------------------------------------------
# Errors and cancellation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 429, 500])
def test_http_error_is_an_openai_compat_error(stub_server, tiny_png, status):
    stub_server.reply(status=status, body=json.dumps({"error": {"message": "nope"}}).encode())
    with pytest.raises(OpenAiCompatError, match=f"HTTP {status}"):
        _generate(stub_server, tiny_png)


def test_invalid_json_is_an_openai_compat_error(stub_server, tiny_png):
    stub_server.reply(body=b"<html>gateway</html>")
    with pytest.raises(OpenAiCompatError, match="invalid JSON"):
        _generate(stub_server, tiny_png)


def test_cancelled_request_raises_a_cancellation_and_sends_nothing(stub_server, tiny_png):
    cancellation = LlmRequestCancellation()
    cancellation.cancel()

    with pytest.raises(LlmProviderCancelled):
        _generate(stub_server, tiny_png, cancellation=cancellation)

    assert stub_server.requests == []


def test_cancel_in_flight_raises_openai_compat_cancelled(stub_server, tiny_png, response_begun):
    stub_server.reply(json_body=_completion({"content": "late"}), hold=threading.Event())
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

    assert isinstance(outcome.get("error"), OpenAiCompatCancelled), outcome


def test_error_classes_fit_the_provider_hierarchy():
    assert issubclass(OpenAiCompatError, LlmProviderError)
    assert issubclass(OpenAiCompatCancelled, OpenAiCompatError)
    assert issubclass(OpenAiCompatCancelled, LlmProviderCancelled)
