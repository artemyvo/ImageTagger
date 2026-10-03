"""Background write queue: ordering and failure reporting."""
from __future__ import annotations

import json
import threading

from imagetagger.ui.merge_actions import clear_fixup_sidecar
from imagetagger.utils import io_utils
from imagetagger.utils.sidecar import (
    SidecarData,
    get_sidecar_json_path,
    read_sidecar_data,
    write_sidecar_data,
    write_sidecar_data_async,
)


def _disk_sidecar(image_path) -> dict:
    return json.loads(get_sidecar_json_path(image_path).read_text(encoding="utf-8"))


def test_sync_write_waits_for_queued_async_write(tmp_path, drain_writes):
    """Undo after a merge: the restore must not be overwritten by the still-queued merge."""
    image = tmp_path / "a.png"
    original = SidecarData(description="original", fixup_tags=["x"])
    write_sidecar_data(image, original)

    # Merge (async), queued behind a stalled write, as on a slow disk.
    gate = threading.Event()
    io_utils._enqueue_write_job(lambda: gate.wait(30))
    clear_fixup_sidecar(image)

    # Undo (sync) from another thread while the merge is still queued.
    undo = threading.Thread(target=write_sidecar_data, args=(image, original))
    undo.start()
    undo.join(0.2)
    assert undo.is_alive(), "sync write must wait behind the queued merge"
    gate.set()
    undo.join(10)
    drain_writes()

    on_disk = _disk_sidecar(image)
    assert on_disk["description"] == "original"
    assert on_disk["fixup_tags"] == ["x"]
    assert "validated" not in on_disk
    assert read_sidecar_data(image).fixup_tags == ["x"]


def test_writes_land_in_queue_order(tmp_path, drain_writes):
    path = tmp_path / "a.txt"
    for index in range(50):
        io_utils.bg_write_text(path, f"v{index}")
    io_utils.write_text_in_order(path, "final")
    drain_writes()
    assert path.read_text() == "final"


def test_failed_async_sidecar_write_is_reported_and_not_cached(tmp_path, monkeypatch, write_errors, drain_writes):
    """A failed write must not leave the cache claiming the new content was saved."""
    write_errors.expect()
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData(description="on disk"))

    real_write = io_utils.atomic_write_text

    def failing_write(path, content, encoding="utf-8"):
        if path == get_sidecar_json_path(image):
            raise OSError(28, "No space left on device")
        real_write(path, content, encoding=encoding)

    monkeypatch.setattr(io_utils, "atomic_write_text", failing_write)
    write_sidecar_data_async(image, SidecarData(description="lost"))
    drain_writes()

    assert [path for path, _ in write_errors.errors] == [get_sidecar_json_path(image)]
    assert _disk_sidecar(image)["description"] == "on disk"
    assert read_sidecar_data(image).description == "on disk"


def test_completion_callback_gets_the_error(tmp_path, monkeypatch, write_errors, drain_writes):
    write_errors.expect()
    outcomes = []

    def failing_write(path, content, encoding="utf-8"):
        raise OSError("boom")

    monkeypatch.setattr(io_utils, "atomic_write_text", failing_write)
    io_utils.bg_write_text(tmp_path / "a.txt", "x", on_complete=outcomes.append)
    drain_writes()
    assert len(outcomes) == 1 and isinstance(outcomes[0], OSError)


def test_run_in_write_order_reraises_in_caller(tmp_path):
    def fail():
        raise ValueError("from write thread")

    try:
        io_utils.run_in_write_order(fail)
    except ValueError as exc:
        assert str(exc) == "from write thread"
    else:
        raise AssertionError("expected ValueError")


def test_run_in_write_order_from_write_thread_does_not_deadlock(tmp_path):
    """A completion callback that writes synchronously runs inline instead of waiting on itself."""
    target = tmp_path / "b.txt"
    done = threading.Event()

    def on_complete(error):
        io_utils.write_text_in_order(target, "nested")
        done.set()

    io_utils.bg_write_text(tmp_path / "a.txt", "x", on_complete=on_complete)
    assert done.wait(5)
    assert target.read_text() == "nested"
