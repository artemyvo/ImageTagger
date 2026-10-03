"""Annotation saves: failures are reported, queued saves cannot outlive a delete."""
from __future__ import annotations

import threading

from imagetagger.utils import io_utils


def _fail_writes_to(monkeypatch, *targets) -> None:
    real_write = io_utils.atomic_write_text

    def write(path, content, encoding="utf-8"):
        if path in targets:
            raise OSError(28, "No space left on device")
        real_write(path, content, encoding=encoding)

    monkeypatch.setattr(io_utils, "atomic_write_text", write)


def test_failed_auto_save_shows_one_dialog_for_a_burst(
    qtbot, main_window, monkeypatch, message_boxes, write_errors, drain_writes
):
    write_errors.expect()
    records = main_window.records
    _fail_writes_to(monkeypatch, *(record.text_path for record in records))

    for record in records:
        main_window._set_tags_for_image_path(record.image_path, ["changed"])
    drain_writes()

    qtbot.waitUntil(lambda: "Save failed" in message_boxes.titles("warning"), timeout=5_000)
    qtbot.wait(50)  # let any stray queued failure signals arrive
    save_failed = [text for kind, title, text in message_boxes.calls if title == "Save failed"]
    assert len(save_failed) == 1
    assert "Could not save 3 files" in save_failed[0]
    assert len(write_errors.errors) == 3
    for record in records:
        assert record.text_path.read_text(encoding="utf-8") != "changed"


def test_delete_waits_for_queued_auto_save(main_window, drain_writes):
    """A queued auto-save must not re-create the .txt of an image deleted after it."""
    record = main_window.records[0]
    gate = threading.Event()
    io_utils._enqueue_write_job(lambda: gate.wait(30))
    main_window._set_tags_for_image_path(record.image_path, ["late write"])

    # The delete waits in write order, so release the queue from elsewhere.
    releaser = threading.Timer(0.2, gate.set)
    releaser.start()
    try:
        deleted, _ = main_window._delete_image_and_related_files(record.image_path, confirm=False)
    finally:
        gate.set()
        releaser.cancel()
    drain_writes()

    assert deleted
    assert not record.image_path.exists()
    assert not record.text_path.exists()


def _close_answering(monkeypatch, window, answer) -> tuple:
    """Close *window*; the "close anyway?" question gets *answer*.  Returns (closed, questions)."""
    from PyQt6.QtWidgets import QMessageBox

    questions = []

    def question(parent, title, text, *args, **kwargs):
        questions.append((title, text))
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    window.show()
    return window.close(), questions


def test_close_reports_a_queued_save_that_fails_and_can_be_called_off(
    qtbot, main_window, monkeypatch, message_boxes, write_errors
):
    """The failure dialog waits for the event loop, which closing ends: ask before closing instead."""
    from PyQt6.QtWidgets import QMessageBox

    write_errors.expect()
    record = main_window.records[0]
    _fail_writes_to(monkeypatch, record.text_path)
    main_window._set_tags_for_image_path(record.image_path, ["changed"])

    closed, questions = _close_answering(monkeypatch, main_window, QMessageBox.StandardButton.No)

    assert not closed
    assert main_window.isVisible()
    assert len(questions) == 1
    title, text = questions[0]
    assert title == "Save failed"
    assert "Could not save 1 file" in text and record.text_path.name in text
    qtbot.wait(50)  # the usual deferred dialog must not repeat it
    assert "Save failed" not in message_boxes.titles()


def test_close_anyway_after_a_failed_save(qtbot, main_window, monkeypatch, message_boxes, write_errors):
    from PyQt6.QtWidgets import QMessageBox

    write_errors.expect()
    record = main_window.records[0]
    _fail_writes_to(monkeypatch, record.text_path)
    main_window._set_tags_for_image_path(record.image_path, ["changed"])

    closed, questions = _close_answering(monkeypatch, main_window, QMessageBox.StandardButton.Yes)

    assert closed
    assert len(questions) == 1
    assert "Save failed" not in message_boxes.titles()


def test_close_without_failures_asks_nothing(qtbot, main_window, monkeypatch, message_boxes, write_errors):
    from PyQt6.QtWidgets import QMessageBox

    main_window._set_tags_for_image_path(main_window.records[0].image_path, ["changed"])

    closed, questions = _close_answering(monkeypatch, main_window, QMessageBox.StandardButton.No)

    assert closed
    assert questions == []
    assert main_window.records[0].text_path.read_text(encoding="utf-8") == "changed"
