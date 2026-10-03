"""LlmController batches: thread scheduling, retries, stopping, and summaries."""
from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest

from imagetagger.providers.llm_provider import (
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
)
from imagetagger.ui import llm_controller as llm_controller_module
from imagetagger.utils.sidecar import read_sidecar_data


def _prepare_llm(window, session, monkeypatch, threads: str = "1", retries: str = "0") -> None:
    window.llm_model_name = "fake-model"
    window.llm_threads_input.setText(threads)
    window.llm_retry_input.setText(retries)
    monkeypatch.setattr(window.llm_controller, "_active_provider_session", lambda: session)


def _only_tags(window) -> None:
    window.generate_tags_checkbox.setChecked(True)
    window.generate_description_checkbox.setChecked(False)
    window.generate_vision_checkbox.setChecked(False)
    window.generate_refine_checkbox.setChecked(False)


def _wait_batch(qtbot, window) -> None:
    qtbot.waitUntil(lambda: window.llm_controller.llm_thread is None, timeout=10_000)


def _run(qtbot, window, start) -> None:
    window.list_widget.selectAll()
    start()
    assert window.llm_controller.llm_thread is not None
    _wait_batch(qtbot, window)


def _status(window) -> str:
    return window.statusBar().currentMessage()


def _provider(window) -> str:
    return window._llm_provider.display_name


def _record_items(window, monkeypatch) -> List[dict]:
    """Record every streamed batch payload as the GUI thread applies it."""
    controller = window.llm_controller
    original = controller._on_llm_task_item_ready
    seen: List[dict] = []

    def _spy(payload):
        original(payload)
        snapshot = dict(payload)
        snapshot["_generate_retry_images"] = controller._generate_batch_retry_images
        snapshot["_validate_errors"] = controller._validate_batch_errors
        snapshot["_validate_retry_images"] = controller._validate_batch_retry_images
        seen.append(snapshot)

    monkeypatch.setattr(controller, "_on_llm_task_item_ready", _spy)
    return seen


def _assert_controls_idle(window) -> None:
    assert window.generate_button.text() == "&Generate"
    assert window.generate_button.isEnabled()
    assert window.validate_button.text() == "&Validate"
    assert window.validate_button.isEnabled()
    assert window.ai_find_button.text() == "AI Find"
    assert window.ai_find_input.isEnabled()
    for widget in (
        window.llm_threads_input,
        window.llm_retry_input,
        window.llm_timeout_input,
        window.llm_model_combo,
        window.llm_use_button,
    ):
        assert widget.isEnabled(), widget
    assert window.llm_controller.llm_action_name is None
    assert window.llm_controller.llm_cancel is None


# ---------------------------------------------------------------------------
# _run_parallel_llm_jobs, called directly
# ---------------------------------------------------------------------------


def test_fixed_thread_count_caps_concurrency_and_runs_each_job_once(main_window):
    controller = main_window.llm_controller
    lock = threading.Condition()
    state = {"active": 0, "max": 0, "arrived": 0}
    ran: List[int] = []

    def process_one(position: int, job: int) -> dict:
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            ran.append(job)
            if job < 3:
                # The first three wait for each other: proves 3 really run at once.
                state["arrived"] += 1
                lock.notify_all()
                assert lock.wait_for(lambda: state["arrived"] >= 3, timeout=10)
            state["active"] -= 1
        return {"image_name": f"job{job}", "job": job, "position": position}

    progress: List[str] = []
    items: List[dict] = []
    controller._run_parallel_llm_jobs(
        jobs=list(range(9)),
        requested_threads=3,
        action_prefix="Test",
        cancel_token=LlmRequestCancellation(),
        report_progress=progress.append,
        report_item=items.append,
        process_one=process_one,
    )

    assert state["max"] == 3
    assert sorted(ran) == list(range(9))
    assert sorted(item["job"] for item in items) == list(range(9))
    assert sorted(item["position"] for item in items) == list(range(1, 10))
    assert [message.split(" - ")[0] for message in progress] == [f"Test: processing {n}/9" for n in range(1, 10)]
    assert controller._llm_threads_current == 3
    assert controller._llm_threads_auto_mode is False


def test_fixed_thread_count_is_capped_by_job_count(main_window):
    controller = main_window.llm_controller
    items: List[object] = []
    controller._run_parallel_llm_jobs(
        jobs=["x", "y"],
        requested_threads=8,
        action_prefix="Test",
        cancel_token=LlmRequestCancellation(),
        report_progress=lambda _m: None,
        report_item=items.append,
        process_one=lambda position, job: {"job": job},
    )
    assert controller._llm_threads_current == 2
    assert sorted(item["job"] for item in items) == ["x", "y"]


def test_job_exception_cancels_the_batch_and_propagates(main_window):
    controller = main_window.llm_controller
    token = LlmRequestCancellation()

    def process_one(position: int, job: int) -> dict:
        if job == 1:
            raise LlmProviderError("boom")
        return {"job": job}

    items: List[dict] = []
    with pytest.raises(LlmProviderError, match="boom"):
        controller._run_parallel_llm_jobs(
            jobs=[0, 1, 2],
            requested_threads=1,
            action_prefix="Test",
            cancel_token=token,
            report_progress=lambda _m: None,
            report_item=items.append,
            process_one=process_one,
        )
    assert token.is_cancelled()
    assert items == [{"job": 0}]


def test_threads_status_suffix(main_window):
    controller = main_window.llm_controller
    assert controller._llm_threads_status_suffix() == ""
    try:
        controller._llm_action_name = "Generate"
        controller._llm_threads_auto_mode = False
        controller._llm_threads_current = 3
        assert controller._llm_threads_status_suffix() == " | threads: 3"
        controller._llm_threads_auto_mode = True
        controller._llm_threads_current = 0  # never shown below 1
        assert controller._llm_threads_status_suffix() == " | threads: 1 auto"
    finally:
        controller._llm_action_name = None
        controller._llm_threads_auto_mode = False
        controller._llm_threads_current = 0


# ---------------------------------------------------------------------------
# Auto thread mode through a real Generate batch
# ---------------------------------------------------------------------------


def _ordered_auto_generate(qtbot, make_main_window, image_folder, fake_session, monkeypatch, count, fail):
    """Run Generate (tags) in auto mode over *count* images, finishing them in name order.

    ``fail(stem, attempt)`` returns an exception to raise, or None to answer.
    Returns [(stem, _llm_threads_current right after that image finished)].
    """
    names = [chr(ord("a") + i) for i in range(count)]
    window = make_main_window(image_folder({name: f"tag_{name}" for name in names}))
    controller = window.llm_controller
    reported: Dict[str, threading.Event] = defaultdict(threading.Event)
    attempts: Dict[str, int] = defaultdict(int)
    trace: List[Tuple[str, int]] = []
    original = controller._run_parallel_llm_jobs

    def ordered_runner(**kwargs):
        inner = kwargs["report_item"]

        def report(payload):
            stem = payload["image_path"].stem
            trace.append((stem, controller._llm_threads_current))
            inner(payload)
            reported[stem].set()

        kwargs["report_item"] = report
        return original(**kwargs)

    monkeypatch.setattr(controller, "_run_parallel_llm_jobs", ordered_runner)

    def respond(path: Path, prompt: str) -> str:
        index = names.index(path.stem)
        if index > 0:
            # Finish strictly in order so the scheduling decisions are deterministic.
            assert reported[names[index - 1]].wait(10)
        attempts[path.stem] += 1
        error = fail(path.stem, attempts[path.stem])
        if error is not None:
            raise error
        return f"gen_{path.stem}"

    session = fake_session(respond)
    _prepare_llm(window, session, monkeypatch, threads="0", retries="1")
    _only_tags(window)
    window._cfg["llm_auto_warmup_items"] = 1
    window._cfg["llm_auto_scale_up_every"] = 1
    window._cfg["llm_auto_max_threads"] = 32
    _run(qtbot, window, controller.generate_with_llm)
    assert [stem for stem, _ in trace] == names
    return window, trace, attempts


def test_auto_threads_start_at_one_and_scale_up_after_clean_items(
    qtbot, make_main_window, image_folder, fake_session, monkeypatch
):
    names = [chr(ord("a") + i) for i in range(8)]
    window = make_main_window(image_folder({name: f"tag_{name}" for name in names}))
    controller = window.llm_controller
    trace: List[int] = []
    original = controller._run_parallel_llm_jobs

    def runner(**kwargs):  # records the thread target after each finished image
        inner = kwargs["report_item"]

        def report(payload):
            trace.append(controller._llm_threads_current)
            inner(payload)

        kwargs["report_item"] = report
        return original(**kwargs)

    first_job_parallelism: List[int] = []

    def respond(path: Path, prompt: str) -> str:
        if path.stem == "a":
            first_job_parallelism.append(controller._llm_threads_current)
        return f"gen_{path.stem}"

    monkeypatch.setattr(controller, "_run_parallel_llm_jobs", runner)
    session = fake_session(respond)
    _prepare_llm(window, session, monkeypatch, threads="0")
    _only_tags(window)
    window._cfg["llm_auto_warmup_items"] = 2
    window._cfg["llm_auto_scale_up_every"] = 1
    window._cfg["llm_auto_max_threads"] = 32
    _run(qtbot, window, controller.generate_with_llm)

    assert first_job_parallelism == [1]
    # No scaling before warm-up (2 items); then +1 after scale_up_every * current clean items.
    assert trace == [1, 2, 2, 3, 3, 3, 4, 4]
    assert sorted(path.stem for path in session.calls) == names
    folder = window.records[0].image_path.parent
    for name in names:
        assert (folder / f"{name}.txt").read_text(encoding="utf-8") == f"tag_{name}, gen_{name}"
    # Auto mode state is reset once the batch is over.
    assert controller._llm_threads_auto_mode is False
    assert controller._llm_threads_current == 0


def test_auto_threads_step_down_by_one_on_retry(
    qtbot, make_main_window, image_folder, fake_session, monkeypatch
):
    def fail(stem, attempt):
        if stem in ("d", "e") and attempt == 1:
            return LlmProviderError("server busy")
        return None

    window, trace, attempts = _ordered_auto_generate(
        qtbot, make_main_window, image_folder, fake_session, monkeypatch, 6, fail
    )
    # a:1->2, b:clean, c:->3, d retried: 3->2, e retried but was queued before
    # the step-down (stale epoch): no second step, f: 1 clean of 2 needed.
    assert [current for _, current in trace] == [2, 2, 3, 2, 2, 2]
    assert attempts["d"] == 2 and attempts["e"] == 2
    folder = window.records[0].image_path.parent
    assert (folder / "d.txt").read_text(encoding="utf-8") == "tag_d, gen_d"


def test_auto_threads_halve_on_timeout(qtbot, make_main_window, image_folder, fake_session, monkeypatch):
    def fail(stem, attempt):
        if stem in ("d", "e"):
            return LlmProviderError("Timed out after 5 seconds while generating annotations.")
        return None

    window, trace, attempts = _ordered_auto_generate(
        qtbot, make_main_window, image_folder, fake_session, monkeypatch, 6, fail
    )
    # d timed out: 3 -> 1; e timed out but was queued before (stale epoch): stays 1;
    # f is clean: back up to 2.
    assert [current for _, current in trace] == [2, 2, 3, 1, 1, 2]
    folder = window.records[0].image_path.parent
    assert (folder / "d.txt").read_text(encoding="utf-8") == "tag_d"
    assert (folder / "e.txt").read_text(encoding="utf-8") == "tag_e"


def test_no_backoff_errors_do_not_step_threads_down(
    qtbot, make_main_window, image_folder, fake_session, monkeypatch
):
    def fail(stem, attempt):
        if stem == "d" and attempt == 1:
            error = LlmProviderError("bad answer")
            error.no_backoff = True
            return error
        return None

    _, trace, attempts = _ordered_auto_generate(
        qtbot, make_main_window, image_folder, fake_session, monkeypatch, 6, fail
    )
    assert attempts["d"] == 2
    # d's retry is not a performance signal: it counts as a clean completion.
    assert [current for _, current in trace] == [2, 2, 3, 3, 3, 4]


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


def test_generate_retry_then_success(qtbot, main_window, fake_session, monkeypatch, drain_writes):
    attempts: Dict[str, int] = defaultdict(int)

    def respond(path: Path, prompt: str) -> str:
        attempts[path.stem] += 1
        if path.stem == "b" and attempts["b"] == 1:
            raise LlmProviderError("connection reset")
        return f"gen_{path.stem}"

    session = fake_session(respond)
    _prepare_llm(main_window, session, monkeypatch, retries="2")
    _only_tags(main_window)
    items = _record_items(main_window, monkeypatch)
    _run(qtbot, main_window, main_window.llm_controller.generate_with_llm)
    drain_writes()

    folder = main_window.records[0].image_path.parent
    assert [path.stem for path in session.calls] == ["a", "b", "b", "c"]
    assert (folder / "b.txt").read_text(encoding="utf-8") == "tag_b, gen_b"
    by_stem = {item["image_path"].stem: item for item in items}
    assert by_stem["b"]["retried"] is True and by_stem["b"]["perf_retry"] is True
    assert by_stem["a"]["retried"] is False
    assert items[-1]["_generate_retry_images"] == 1
    assert _status(main_window) == f"Generate complete via {_provider(main_window)} (3 images, 3 new annotations)"


def test_generate_exhausted_retries_leave_image_unchanged(
    qtbot, main_window, fake_session, monkeypatch, drain_writes, message_boxes
):
    def respond(path: Path, prompt: str) -> str:
        if path.stem == "b":
            raise LlmProviderError("model crashed")
        return f"gen_{path.stem}"

    session = fake_session(respond)
    _prepare_llm(main_window, session, monkeypatch, retries="1")
    _only_tags(main_window)
    _run(qtbot, main_window, main_window.llm_controller.generate_with_llm)
    drain_writes()

    folder = main_window.records[0].image_path.parent
    assert [path.stem for path in session.calls] == ["a", "b", "b", "c"]
    assert (folder / "b.txt").read_text(encoding="utf-8") == "tag_b"
    assert (folder / "a.txt").read_text(encoding="utf-8") == "tag_a, gen_a"
    assert (folder / "c.txt").read_text(encoding="utf-8") == "tag_c, gen_c"
    assert _status(main_window) == f"Generate complete via {_provider(main_window)} (2 images, 2 new annotations)"
    assert message_boxes.calls == []


def test_generate_with_nothing_new_reports_empty(qtbot, main_window, fake_session, monkeypatch, message_boxes):
    session = fake_session(lambda path, prompt: f"tag_{path.stem}")  # already there
    _prepare_llm(main_window, session, monkeypatch)
    _only_tags(main_window)
    _run(qtbot, main_window, main_window.llm_controller.generate_with_llm)
    assert message_boxes.titles("information") == ["Generate finished"]
    assert _status(main_window) == "Generate finished"


def test_validate_retries_bad_format_then_accepts(qtbot, main_window, fake_session, monkeypatch):
    attempts: Dict[str, int] = defaultdict(int)

    def respond(path: Path, prompt: str) -> str:
        attempts[path.stem] += 1
        if path.stem == "b" and attempts["b"] == 1:
            return "looks fine to me"  # no fixup headers: treated as a failed attempt
        return "OK"

    session = fake_session(respond)
    _prepare_llm(main_window, session, monkeypatch, retries="1")
    items = _record_items(main_window, monkeypatch)
    _run(qtbot, main_window, main_window.llm_controller.validate_tags_with_llm)

    assert attempts == {"a": 1, "b": 2, "c": 1}
    assert [item["kind"] for item in items] == ["validate_item"] * 3
    assert items[-1]["_validate_retry_images"] == 1
    for record in main_window.records:
        assert read_sidecar_data(record.image_path).validated_by == "fake-model"
    assert _status(main_window) == f"Validate complete via {_provider(main_window)} (3 images checked, 3 clean, 0 fixup files)"


def test_validate_exhausted_retries_count_as_errors(qtbot, main_window, fake_session, monkeypatch, message_boxes):
    def respond(path: Path, prompt: str) -> str:
        if path.stem == "b":
            raise LlmProviderError("Timed out after 5 seconds while validating annotations for b.png.")
        return "OK"

    session = fake_session(respond)
    _prepare_llm(main_window, session, monkeypatch, retries="1")
    items = _record_items(main_window, monkeypatch)
    _run(qtbot, main_window, main_window.llm_controller.validate_tags_with_llm)

    assert [path.stem for path in session.calls] == ["a", "b", "b", "c"]
    by_stem = {item["image_path"].stem: item for item in items}
    assert by_stem["b"]["kind"] == "validate_item_error"
    assert by_stem["b"]["timed_out"] is True
    assert items[-1]["_validate_errors"] == 1
    assert read_sidecar_data(main_window.records[1].image_path).validated is None
    assert main_window.llm_controller.validate_pending_paths == set()
    assert _status(main_window) == (
        f"Validate complete via {_provider(main_window)} (2 images checked, 2 clean, 0 fixup files, 1 error)"
    )
    assert message_boxes.calls == []


def test_ai_find_provider_failure_ends_batch_cleanly(qtbot, main_window, fake_session, monkeypatch):
    session = fake_session(lambda path, prompt: "maybe" if path.stem == "b" else "YES")
    _prepare_llm(main_window, session, monkeypatch)
    main_window.ai_find_input.setText("red square")
    _run(qtbot, main_window, main_window.llm_controller.ai_find_with_llm)

    # a was answered before b failed; whatever happens to the rest, a keeps its match.
    assert read_sidecar_data(main_window.records[0].image_path).ai_find_matches == ["red square"]
    _assert_controls_idle(main_window)


def test_ai_find_failure_on_one_image_does_not_abort_the_batch(
    qtbot, main_window, fake_session, monkeypatch, message_boxes
):
    session = fake_session(lambda path, prompt: "maybe" if path.stem == "b" else "YES")
    _prepare_llm(main_window, session, monkeypatch, retries="1")
    main_window.ai_find_input.setText("red square")
    _run(qtbot, main_window, main_window.llm_controller.ai_find_with_llm)

    assert [path.stem for path in session.calls] == ["a", "b", "b", "c"]
    records = main_window.records
    assert read_sidecar_data(records[0].image_path).ai_find_matches == ["red square"]
    assert read_sidecar_data(records[2].image_path).ai_find_matches == ["red square"]
    assert "found images: 2 of 3" in _status(main_window)
    assert "1 error" in _status(main_window)
    assert message_boxes.titles("warning") == []


# ---------------------------------------------------------------------------
# Stopping a batch
# ---------------------------------------------------------------------------


def _cancellable(session):
    """Make held requests end the way a real provider does once the batch is stopped."""
    original = session.generate

    def generate(image_path, prompt, *, timeout, cancellation=None, **kwargs):
        result = original(image_path, prompt, timeout=timeout, cancellation=cancellation, **kwargs)
        if cancellation is not None and cancellation.is_cancelled():
            raise LlmProviderCancelled("Request stopped.")
        return result

    session.generate = generate
    return session


@pytest.mark.parametrize(
    "action, button, stopping_text, stopped_status",
    [
        ("generate_with_llm", "generate_button", "Stopping generation...", "Request stopped."),
        ("validate_tags_with_llm", "validate_button", "Stopping validation...", "Validation stopped after 1/3 images."),
        ("ai_find_with_llm", "ai_find_button", "Stopping AI Find...", "AI Find stopped after 1/3 images (found images: 1)."),
    ],
)
def test_stop_mid_batch(
    qtbot, main_window, fake_session, monkeypatch, drain_writes, message_boxes,
    action, button, stopping_text, stopped_status,
):
    def respond(path: Path, prompt: str) -> str:
        if action == "generate_with_llm":
            return f"gen_{path.stem}"
        if action == "validate_tags_with_llm":
            return "OK"
        return "YES"

    session = _cancellable(fake_session(respond))
    session.hold.add("b")
    _prepare_llm(main_window, session, monkeypatch)
    _only_tags(main_window)
    main_window.ai_find_input.setText("red square")
    start = getattr(main_window.llm_controller, action)

    main_window.list_widget.selectAll()
    start()
    assert session.started.wait(10), "the request for b never started"
    stop_button = getattr(main_window, button)
    assert stop_button.isEnabled()
    start()  # the same button path: pressing it again while running stops the batch
    assert stop_button.text() == stopping_text
    assert not stop_button.isEnabled()
    session.release()
    _wait_batch(qtbot, main_window)
    drain_writes()

    assert [path.stem for path in session.calls] == ["a", "b"]
    records = main_window.records
    folder = records[0].image_path.parent
    if action == "generate_with_llm":
        assert (folder / "a.txt").read_text(encoding="utf-8") == "tag_a, gen_a"
    elif action == "validate_tags_with_llm":
        assert read_sidecar_data(records[0].image_path).validated_by == "fake-model"
    else:
        assert read_sidecar_data(records[0].image_path).ai_find_matches == ["red square"]
    for record in records[1:]:
        assert record.image_path.with_suffix(".txt").read_text(encoding="utf-8") == f"tag_{record.image_path.stem}"
        sidecar = read_sidecar_data(record.image_path)
        assert sidecar.validated is None and sidecar.ai_find_matches is None
    assert _status(main_window) == stopped_status
    assert message_boxes.calls == []
    assert main_window.llm_controller.validate_pending_paths == set()
    _assert_controls_idle(main_window)


def test_controls_while_generate_runs(qtbot, main_window, fake_session, monkeypatch):
    session = fake_session(lambda path, prompt: f"gen_{path.stem}")
    session.hold.add("a")
    _prepare_llm(main_window, session, monkeypatch)
    _only_tags(main_window)
    main_window.ai_find_input.setText("red square")
    main_window.list_widget.selectAll()
    main_window.llm_controller.generate_with_llm()
    assert session.started.wait(10)
    try:
        assert main_window.generate_button.text() == "&Stop generation"
        assert main_window.generate_button.isEnabled()
        assert not main_window.validate_button.isEnabled()
        assert not main_window.ai_find_button.isEnabled()
        assert not main_window.ai_find_input.isEnabled()
        assert not main_window.llm_threads_input.isEnabled()
        assert not main_window.llm_retry_input.isEnabled()
        # Other actions do nothing while a batch runs.
        main_window.llm_controller.validate_tags_with_llm()
        assert main_window.llm_controller.llm_action_name == "Generate"
    finally:
        session.release()
    _wait_batch(qtbot, main_window)
    _assert_controls_idle(main_window)


# ---------------------------------------------------------------------------
# Generate with description, vision and refine
# ---------------------------------------------------------------------------


def test_generate_description_vision_refine(
    qtbot, make_main_window, image_folder, fake_session, monkeypatch, drain_writes
):
    window = make_main_window(image_folder({"a": "Old picture of things., tag_a", "b": "tag_b"}))

    def query(prompt):
        return lambda *args, **kwargs: SimpleNamespace(prompt=prompt, kwargs=kwargs)

    monkeypatch.setattr(llm_controller_module, "prepare_description_query", query("Q:description"))
    monkeypatch.setattr(llm_controller_module, "prepare_tagging_query", query("Q:tags"))
    vision_inputs: Dict[str, str] = {}

    def vision_query(*, tags_text, user_hint=None):
        return SimpleNamespace(prompt="Q:vision\n" + tags_text)

    refine_inputs: List[Tuple[str, str]] = []

    def refine_query(*, description, reasoning):
        refine_inputs.append((description, reasoning))
        return SimpleNamespace(prompt="Q:refine")

    monkeypatch.setattr(llm_controller_module, "prepare_vision_query", vision_query)
    monkeypatch.setattr(llm_controller_module, "prepare_refine_query", refine_query)

    def respond(path: Path, prompt: str) -> str:
        if prompt == "Q:description":
            return f"A red square {path.stem}."
        if prompt == "Q:tags":
            return "red, square"
        if prompt.startswith("Q:vision"):
            vision_inputs[path.stem] = prompt.split("\n", 1)[1]
            return f"THOUGHT:\nlooks red {path.stem}\nDESCRIPTION:\nA red square on white {path.stem}."
        if prompt == "Q:refine":
            return f"TAGS: crimson, box\nCAPTION: A crimson box {path.stem}."
        raise AssertionError(prompt)

    session = fake_session(respond)
    _prepare_llm(window, session, monkeypatch)
    for checkbox in (
        window.generate_tags_checkbox,
        window.generate_description_checkbox,
        window.generate_vision_checkbox,
        window.generate_refine_checkbox,
    ):
        checkbox.setChecked(True)
    _run(qtbot, window, window.llm_controller.generate_with_llm)
    drain_writes()

    folder = window.records[0].image_path.parent
    # The old description is replaced in place; b gets one inserted first.
    assert (folder / "a.txt").read_text(encoding="utf-8") == "A red square a., tag_a, red, square"
    assert (folder / "b.txt").read_text(encoding="utf-8") == "A red square b., tag_b, red, square"
    # Vision sees this attempt's description and tags.
    assert vision_inputs["a"] == "A red square a.\nred\nsquare"
    for record in window.records:
        stem = record.image_path.stem
        sidecar = read_sidecar_data(record.image_path)
        assert sidecar.description == f"A red square on white {stem}."
        assert sidecar.reasoning == f"looks red {stem}"
        assert sidecar.vision_tags == ["crimson", "box"]
        assert sidecar.vision_caption == f"A crimson box {stem}."
        assert record.has_pending_fixup
    # Refine reads the vision output just written.
    assert sorted(refine_inputs) == [
        ("A red square on white a.", "looks red a"),
        ("A red square on white b.", "looks red b"),
    ]
    assert _status(window) == (
        f"Generate complete via {_provider(window)} "
        "(2 images, 6 new annotations, 2 vision updates, 2 refine fixups)"
    )


def test_refine_without_vision_data_is_skipped(qtbot, main_window, fake_session, monkeypatch, message_boxes):
    session = fake_session(lambda path, prompt: pytest.fail("no request expected"))
    _prepare_llm(main_window, session, monkeypatch, retries="0")
    main_window.generate_tags_checkbox.setChecked(False)
    main_window.generate_description_checkbox.setChecked(False)
    main_window.generate_vision_checkbox.setChecked(False)
    main_window.generate_refine_checkbox.setChecked(True)
    _run(qtbot, main_window, main_window.llm_controller.generate_with_llm)
    assert session.calls == []
    assert message_boxes.titles("information") == ["Generate finished"]


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def test_validate_batch_outcomes_and_summary(qtbot, make_main_window, image_folder, fake_session, monkeypatch):
    window = make_main_window(image_folder({"a": "tag_a", "b": "tag_b", "c": ""}))
    answers = {"a": "OK", "b": "ISSUES: Wrong animal.\nTAGS: dog"}
    session = fake_session(lambda path, prompt: answers[path.stem])
    _prepare_llm(window, session, monkeypatch)
    _run(qtbot, window, window.llm_controller.validate_tags_with_llm)

    assert [path.stem for path in session.calls] == ["a", "b"]  # c has no annotations
    a, b, c = (read_sidecar_data(record.image_path) for record in window.records)
    assert a.validated is not None and a.validated_by == "fake-model" and not a.has_pending_fixup
    assert (b.fixup_issues, b.fixup_tags, b.fixup_model) == ("Wrong animal.", ["dog"], "fake-model")
    assert b.validated is None
    assert window.records[1].has_pending_fixup
    assert c.validated is None
    assert _status(window) == (
        f"Validate complete via {_provider(window)} (2 images checked, 1 clean, 1 fixup file, 1 skipped)"
    )
    assert window.llm_controller.validate_pending_paths == set()


# ---------------------------------------------------------------------------
# AI Find
# ---------------------------------------------------------------------------


def test_ai_find_records_matches(qtbot, main_window, fake_session, monkeypatch, message_boxes):
    session = fake_session(lambda path, prompt: "NO" if path.stem == "b" else "Yes, it does.")
    _prepare_llm(main_window, session, monkeypatch, threads="2")
    main_window.ai_find_input.setText("  Red   Square ")
    _run(qtbot, main_window, main_window.llm_controller.ai_find_with_llm)

    matches = [read_sidecar_data(record.image_path).ai_find_matches for record in main_window.records]
    assert matches == [["red square"], None, ["red square"]]
    assert [record.has_pending_fixup for record in main_window.records] == [True, False, True]
    assert _status(main_window) == "AI Find complete (found images: 2 of 3 for 'Red Square')"
    assert message_boxes.calls == []


def test_ai_find_without_matches_says_so(qtbot, main_window, fake_session, monkeypatch, message_boxes):
    session = fake_session(lambda path, prompt: "NO")
    _prepare_llm(main_window, session, monkeypatch)
    main_window.ai_find_input.setText("red square")
    _run(qtbot, main_window, main_window.llm_controller.ai_find_with_llm)

    assert message_boxes.titles() == ["AI Find finished"]
    assert _status(main_window) == "AI Find complete (found images: 0 of 3 for 'red square')"
    for record in main_window.records:
        assert read_sidecar_data(record.image_path).ai_find_matches is None


# ---------------------------------------------------------------------------
# Guards: nothing starts
# ---------------------------------------------------------------------------


ACTIONS = ["generate_with_llm", "validate_tags_with_llm", "ai_find_with_llm"]


def _guard_setup(window, session, monkeypatch):
    _prepare_llm(window, session, monkeypatch)
    _only_tags(window)
    window.ai_find_input.setText("red square")


def _assert_nothing_started(window, session):
    assert window.llm_controller.llm_thread is None
    assert window.llm_controller.llm_action_name is None
    assert session.calls == []


@pytest.mark.parametrize("action", ACTIONS)
def test_guard_no_selection(make_main_window, fake_session, monkeypatch, message_boxes, action):
    window = make_main_window()  # no images at all
    session = fake_session(lambda path, prompt: "OK")
    _guard_setup(window, session, monkeypatch)
    getattr(window.llm_controller, action)()
    assert message_boxes.titles() == ["No image selected"]
    _assert_nothing_started(window, session)


@pytest.mark.parametrize("action", ACTIONS)
def test_guard_no_model(main_window, fake_session, monkeypatch, message_boxes, action):
    session = fake_session(lambda path, prompt: "OK")
    _only_tags(main_window)
    main_window.ai_find_input.setText("red square")
    main_window.llm_threads_input.setText("1")
    main_window.llm_model_name = ""  # the real session factory needs a model
    main_window.list_widget.selectAll()
    getattr(main_window.llm_controller, action)()
    assert message_boxes.titles() == ["No model selected"]
    _assert_nothing_started(main_window, session)


@pytest.mark.parametrize("value", ["", "two", "-1"])
@pytest.mark.parametrize("action", ACTIONS)
def test_guard_invalid_thread_count(main_window, fake_session, monkeypatch, message_boxes, action, value):
    session = fake_session(lambda path, prompt: "OK")
    _guard_setup(main_window, session, monkeypatch)
    main_window.llm_threads_input.setText(value)
    main_window.list_widget.selectAll()
    getattr(main_window.llm_controller, action)()
    assert message_boxes.titles() == ["Invalid threads"]
    _assert_nothing_started(main_window, session)


def test_guard_generate_with_nothing_enabled(main_window, fake_session, monkeypatch, message_boxes):
    session = fake_session(lambda path, prompt: "x")
    _guard_setup(main_window, session, monkeypatch)
    main_window.generate_tags_checkbox.setChecked(False)
    main_window.list_widget.selectAll()
    main_window.llm_controller.generate_with_llm()
    assert message_boxes.titles() == ["Nothing selected"]
    assert not main_window.generate_button.isEnabled()
    _assert_nothing_started(main_window, session)


def test_guard_validate_without_annotations(make_main_window, image_folder, fake_session, monkeypatch, message_boxes):
    window = make_main_window(image_folder({"a": "", "b": "  "}))
    session = fake_session(lambda path, prompt: "OK")
    _guard_setup(window, session, monkeypatch)
    window.list_widget.selectAll()
    window.llm_controller.validate_tags_with_llm()
    assert message_boxes.titles() == ["No annotations to validate"]
    _assert_nothing_started(window, session)


def test_guard_ai_find_without_query(main_window, fake_session, monkeypatch, message_boxes):
    session = fake_session(lambda path, prompt: "YES")
    _guard_setup(main_window, session, monkeypatch)
    main_window.ai_find_input.setText("   ")
    main_window.list_widget.selectAll()
    main_window.llm_controller.ai_find_with_llm()
    assert message_boxes.titles() == ["Missing search text"]
    _assert_nothing_started(main_window, session)
