"""ImageReloadHelper (watcher + mtime polling + debounce) and MainWindow._on_image_reload.

The helper's class-level intervals are shortened so the real watcher and poll
timer fire within a test; checks wait with qtbot.waitUntil, never sleeps.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QColor, QImage

from imagetagger.ui.image_reload_helper import ImageReloadHelper
from imagetagger.ui.main_window import _ROLE_BADGES, _ROLE_PIXMAP


@pytest.fixture(autouse=True)
def fast_intervals(monkeypatch):
    monkeypatch.setattr(ImageReloadHelper, "DEBOUNCE_INTERVAL_MS", 20)
    monkeypatch.setattr(ImageReloadHelper, "POLLING_INTERVAL_MS", 50)


def _save_png(path: Path, width: int, height: int, color: str = "red") -> Path:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    assert image.save(str(path))
    return path


def _bump_mtime(path: Path, seconds: int = 5) -> None:
    """Make the change visible to mtime polling even on coarse-timestamp filesystems."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + seconds * 1_000_000_000))


class Helper:
    def __init__(self, parent: QObject) -> None:
        self.calls: list = []
        self.helper = ImageReloadHelper(parent, self.calls.append)


@pytest.fixture
def helper():
    parent = QObject()
    made = Helper(parent)
    made.parent = parent  # keep the timers' parent alive
    yield made
    made.helper.set_watched_image(None)


@pytest.fixture
def images(tmp_path):
    return _save_png(tmp_path / "a.png", 16, 16), _save_png(tmp_path / "b.png", 16, 16)


def _watched_files(h: ImageReloadHelper):
    watcher = h._image_file_watcher
    return [] if watcher is None else [Path(p) for p in watcher.files()]


# ---------------------------------------------------------------------------
# ImageReloadHelper
# ---------------------------------------------------------------------------


def test_external_rewrite_invokes_callback_with_the_path(helper, images, qtbot):
    a, _ = images
    helper.helper.set_watched_image(a)
    assert _watched_files(helper.helper) == [a]
    assert helper.helper._image_reload_poll_timer.isActive()
    _save_png(a, 30, 10, "blue")
    _bump_mtime(a)
    qtbot.waitUntil(lambda: helper.calls == [a])
    # The new mtime is the baseline: polling does not report it again.
    helper.helper._poll_watched_image_changes()
    assert not helper.helper._image_reload_pending


def test_touch_only_is_detected_by_polling(helper, images, qtbot):
    a, _ = images
    helper.helper.set_watched_image(a)
    _bump_mtime(a)
    qtbot.waitUntil(lambda: helper.calls == [a])


def test_unchanged_file_is_not_reported(helper, images):
    a, _ = images
    helper.helper.set_watched_image(a)
    helper.helper._poll_watched_image_changes()
    assert not helper.helper._image_reload_pending
    assert not helper.helper._image_reload_debounce_timer.isActive()
    assert helper.calls == []


def test_rapid_changes_are_debounced_into_one_reload(helper, images, qtbot):
    a, _ = images
    h = helper.helper
    h.set_watched_image(a)
    for _ in range(3):
        h._on_watched_image_file_changed(str(a))
    _bump_mtime(a)
    h._poll_watched_image_changes()
    assert h._image_reload_debounce_timer.isActive()
    qtbot.waitUntil(lambda: helper.calls == [a])
    assert not h._image_reload_pending
    assert not h._image_reload_debounce_timer.isActive()
    h._apply_pending_image_reload()  # nothing pending any more
    assert helper.calls == [a]


def test_each_change_restarts_the_debounce(helper, images, monkeypatch):
    a, _ = images
    h = helper.helper
    h.set_watched_image(a)
    starts = []
    monkeypatch.setattr(h, "_image_reload_debounce_timer", _CountingTimer(h._image_reload_debounce_timer, starts))
    h._on_watched_image_file_changed(str(a))
    h._on_watched_image_file_changed(str(a))
    assert len(starts) == 2
    assert helper.calls == []  # nothing until the debounce runs out


class _CountingTimer:
    def __init__(self, timer, starts) -> None:
        self._timer = timer
        self._starts = starts

    def start(self, *args) -> None:
        self._starts.append(args)
        self._timer.start(*args)

    def __getattr__(self, name):
        return getattr(self._timer, name)


def test_change_to_another_path_is_ignored(helper, images):
    a, b = images
    helper.helper.set_watched_image(a)
    helper.helper._on_watched_image_file_changed(str(b))
    assert not helper.helper._image_reload_pending


def test_switching_image_moves_the_watch(helper, images, qtbot):
    a, b = images
    h = helper.helper
    h.set_watched_image(a)
    h.set_watched_image(b)
    assert _watched_files(h) == [b]
    _bump_mtime(a)
    h._poll_watched_image_changes()
    assert not h._image_reload_pending  # a is no longer watched
    _save_png(b, 20, 10, "green")
    _bump_mtime(b)
    qtbot.waitUntil(lambda: helper.calls == [b])


def test_switching_image_cancels_a_pending_reload(helper, images, qtbot):
    a, b = images
    h = helper.helper
    h.set_watched_image(a)
    h._on_watched_image_file_changed(str(a))
    assert h._image_reload_pending
    h.set_watched_image(b)
    assert not h._image_reload_pending
    assert not h._image_reload_debounce_timer.isActive()
    h._apply_pending_image_reload()
    assert helper.calls == []


def test_unwatching_stops_timers_and_releases_the_watcher(helper, images, qtbot):
    a, _ = images
    h = helper.helper
    h.set_watched_image(a)
    h._on_watched_image_file_changed(str(a))
    h.set_watched_image(None)
    assert h._image_file_watcher is None
    assert not h._image_reload_poll_timer.isActive()
    assert not h._image_reload_debounce_timer.isActive()
    h._on_watched_image_file_changed(str(a))  # a late watcher signal is harmless
    h._poll_watched_image_changes()
    assert helper.calls == []
    # Watching again recreates the watcher.
    h.set_watched_image(a)
    assert _watched_files(h) == [a]
    _bump_mtime(a)
    qtbot.waitUntil(lambda: helper.calls == [a])


def test_deleted_file_is_not_reported_by_polling(helper, images):
    a, _ = images
    h = helper.helper
    h.set_watched_image(a)
    a.unlink()
    h._poll_watched_image_changes()
    assert not h._image_reload_pending
    assert helper.calls == []


def test_watcher_event_for_deleted_file_still_calls_back(helper, images, qtbot):
    # The helper leaves "file is gone" to the callback (the main window skips
    # unreadable files as a transient save state).
    a, _ = images
    h = helper.helper
    h.set_watched_image(a)
    a.unlink()
    h._on_watched_image_file_changed(str(a))
    qtbot.waitUntil(lambda: helper.calls == [a])


def test_atomic_replace_is_detected_and_resubscribed(helper, images, qtbot, tmp_path):
    a, _ = images
    h = helper.helper
    h.set_watched_image(a)
    temp = _save_png(tmp_path / "a.tmp.png", 30, 10, "blue")
    _bump_mtime(temp)
    os.replace(temp, a)
    qtbot.waitUntil(lambda: helper.calls[:1] == [a])
    assert _watched_files(h) == [a]  # the watch is back on the new file


def test_file_missing_when_watched_is_polled_once_it_appears(helper, tmp_path, qtbot):
    missing = tmp_path / "later.png"
    h = helper.helper
    h.set_watched_image(missing)
    assert _watched_files(h) == []
    assert h._image_reload_poll_timer.isActive()
    _save_png(missing, 16, 16)
    h._poll_watched_image_changes()  # records the first mtime as the baseline
    _bump_mtime(missing)
    qtbot.waitUntil(lambda: helper.calls == [missing])


# ---------------------------------------------------------------------------
# MainWindow._on_image_reload
# ---------------------------------------------------------------------------


def _badges(window, index):
    return window.list_widget.item(index).data(_ROLE_BADGES) or frozenset()


def test_main_window_watches_the_selected_image(main_window):
    h = main_window._image_reload_helper
    assert main_window.current_index == 0
    assert h._watched_image_path == main_window.records[0].image_path
    main_window.list_widget.setCurrentRow(2)
    assert h._watched_image_path == main_window.records[2].image_path
    assert _watched_files(h) == [main_window.records[2].image_path]


def test_external_edit_refreshes_size_badge_and_preview(main_window, qtbot):
    record = main_window.records[0]
    assert record._image_size == (16, 16)
    assert "✂️" not in _badges(main_window, 0)
    _save_png(record.image_path, 30, 10, "blue")  # 3:1 is not an allowed ratio
    _bump_mtime(record.image_path)
    qtbot.waitUntil(lambda: main_window.statusBar().currentMessage() == "Reloaded image: a.png")
    assert record._image_size == (30, 10)
    assert "✂️" in _badges(main_window, 0)
    assert main_window._record_needs_fixup(record)
    assert main_window.status_ratio_label.text().endswith("1 ratio fixup")
    cached = main_window.image_view_controller._cached_pixmap
    assert cached is not None and (cached.width(), cached.height()) == (30, 10)


def test_external_fix_removes_the_ratio_badge(make_main_window, image_folder, qtbot):
    folder = image_folder({"a": "tag_a", "b": "tag_b"})
    _save_png(folder / "a.png", 30, 10)
    window = make_main_window(folder)
    assert "✂️" in _badges(window, 0)
    _save_png(folder / "a.png", 20, 20, "blue")
    window._on_image_reload(folder / "a.png")
    assert window.records[0]._image_size == (20, 20)
    assert "✂️" not in _badges(window, 0)
    assert not window._record_needs_fixup(window.records[0])


def test_reload_for_a_non_current_image_is_ignored(main_window):
    other = main_window.records[1]
    _save_png(other.image_path, 30, 10)
    main_window.statusBar().showMessage("before")
    main_window._on_image_reload(other.image_path)
    assert other._image_size == (16, 16)
    assert "✂️" not in _badges(main_window, 1)
    assert main_window.statusBar().currentMessage() == "before"


def test_unreadable_file_is_skipped_as_a_transient_save(main_window):
    record = main_window.records[0]
    main_window._show_image(record.image_path)
    cached = main_window.image_view_controller._cached_pixmap
    record.image_path.write_bytes(b"\x89PNG half written")
    main_window.statusBar().showMessage("before")
    main_window._on_image_reload(record.image_path)
    assert record._image_size == (16, 16)
    assert main_window.image_view_controller._cached_pixmap is cached
    assert main_window.statusBar().currentMessage() == "before"


def test_deleted_current_image_does_not_break_reload(main_window, qtbot):
    record = main_window.records[0]
    record.image_path.unlink()
    main_window.statusBar().showMessage("before")
    main_window._on_image_reload(record.image_path)
    assert main_window.statusBar().currentMessage() == "before"
    _save_png(record.image_path, 30, 10)
    main_window._on_image_reload(record.image_path)
    assert record._image_size == (30, 10)


def test_external_edit_refreshes_the_list_thumbnail(main_window):
    record = main_window.records[0]
    before = main_window.list_widget.item(0).data(_ROLE_PIXMAP)
    assert before is not None and before.width() == before.height()  # square 16x16 source
    _save_png(record.image_path, 30, 10, "blue")
    main_window._on_image_reload(record.image_path)
    after = main_window.list_widget.item(0).data(_ROLE_PIXMAP)
    assert after is not None and after.width() > after.height()


def test_crop_refreshes_the_list_thumbnail(main_window):
    """Fixup dialog and Batch Autofix report crops through _on_image_cropped."""
    record = main_window.records[1]
    _save_png(record.image_path, 30, 10, "blue")  # the cropped file on disk
    main_window._on_image_cropped(record.image_path, 30, 10)
    after = main_window.list_widget.item(1).data(_ROLE_PIXMAP)
    assert after is not None and after.width() > after.height()
