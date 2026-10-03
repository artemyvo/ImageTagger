"""Shared fixtures.

Every test runs with Qt offscreen, against a throw-away config.json, with
message boxes answered automatically, and with the background write queue
drained (and checked for unexpected failures) afterwards.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pytest

from imagetagger import config as _config
from imagetagger.utils import io_utils
from imagetagger.utils.sidecar import SidecarData, write_sidecar_data

TESTS_DIR = Path(__file__).resolve().parent

# MainWindows built by tests.  They are hidden, never deleted: a debounced
# tag-list timer can otherwise fire on a deleted widget and abort PyQt6.
_live_windows: list = []


def pytest_collection_modifyitems(config, items) -> None:
    ui_dir = TESTS_DIR / "ui"
    for item in items:
        if ui_dir in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.ui)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_config(tmp_path_factory, monkeypatch) -> Path:
    """Point config load/save at a temp file so the real config.json is never read or written."""
    path = tmp_path_factory.mktemp("config") / "config.json"
    monkeypatch.setattr(_config, "_CONFIG_PATH", path)
    return path


class WriteErrors:
    """Background write failures seen during a test."""

    def __init__(self) -> None:
        self.errors: List[Tuple[Path, BaseException]] = []
        self.expected = False

    def expect(self) -> "WriteErrors":
        self.expected = True
        return self


@pytest.fixture(autouse=True)
def write_errors(monkeypatch) -> Iterator[WriteErrors]:
    """Drain the write queue after the test and fail on unexpected write errors.

    A test that provokes failures on purpose calls ``write_errors.expect()``.
    """
    seen = WriteErrors()
    original = io_utils._report_bg_write_error

    def _record(path: Path, error: BaseException) -> None:
        seen.errors.append((path, error))
        original(path, error)

    monkeypatch.setattr(io_utils, "_report_bg_write_error", _record)
    yield seen
    _drain_writes()
    if not seen.expected:
        assert seen.errors == [], f"unexpected background write failures: {seen.errors}"


def _drain_writes() -> None:
    io_utils.run_in_write_order(lambda: None)


@pytest.fixture
def drain_writes() -> Callable[[], None]:
    """Call it to wait until every write queued so far has landed."""
    return _drain_writes


@pytest.fixture
def stalled_write_queue() -> Iterator[threading.Event]:
    """Hold the background write queue until the returned event is set.

    Writes queued meanwhile stay pending, like on a slow disk.  The queue is
    released when the test ends even if the test does not set the event.
    """
    gate = threading.Event()
    io_utils._enqueue_write_job(lambda: gate.wait(30))
    try:
        yield gate
    finally:
        gate.set()


class MessageBoxes:
    """QMessageBox calls made during a test: (kind, title, text)."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str, str]] = []

    def titles(self, kind: Optional[str] = None) -> List[str]:
        return [title for k, title, _ in self.calls if kind is None or k == kind]


@pytest.fixture(autouse=True)
def message_boxes(monkeypatch) -> MessageBoxes:
    """Record QMessageBox dialogs instead of blocking on them; questions answer Yes."""
    from PyQt6.QtWidgets import QMessageBox

    boxes = MessageBoxes()

    def _fake(kind: str, answer):
        def _show(parent, title, text, *args, **kwargs):
            boxes.calls.append((kind, str(title), str(text)))
            return answer

        return staticmethod(_show)

    Button = QMessageBox.StandardButton
    monkeypatch.setattr(QMessageBox, "information", _fake("information", Button.Ok))
    monkeypatch.setattr(QMessageBox, "warning", _fake("warning", Button.Ok))
    monkeypatch.setattr(QMessageBox, "critical", _fake("critical", Button.Ok))
    monkeypatch.setattr(QMessageBox, "question", _fake("question", Button.Yes))
    return boxes


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


@pytest.fixture
def image_folder(tmp_path) -> Callable[..., Path]:
    """Factory: ``image_folder({"a": "tag_a", ...}, sidecars={"a": SidecarData(...)})``.

    Writes a small PNG plus its .txt for each name, and the given sidecars.
    """
    from PyQt6.QtGui import QColor, QImage

    def _make(
        texts: Dict[str, str],
        sidecars: Optional[Dict[str, SidecarData]] = None,
        folder_name: str = "images",
    ) -> Path:
        folder = tmp_path / folder_name
        folder.mkdir()
        for index, (name, text) in enumerate(texts.items()):
            image = QImage(16, 16, QImage.Format.Format_RGB32)
            image.fill(QColor.fromHsv((index * 40) % 360, 200, 200))
            assert image.save(str(folder / f"{name}.png"))
            (folder / f"{name}.txt").write_text(text, encoding="utf-8")
        for name, data in (sidecars or {}).items():
            write_sidecar_data(folder / f"{name}.png", data)
        return folder

    return _make


class FakeSession:
    """Scripted stand-in for a VisionLlmSession.

    ``respond(image_path, prompt)`` returns the model's answer.  Calls for
    images in ``hold`` block until ``release()`` so a test can change the app
    while a request is in flight.
    """

    def __init__(self, respond: Callable[[Path, str], str]) -> None:
        self._respond = respond
        self.calls: List[Path] = []
        self.hold: set = set()
        self.started = threading.Event()
        self._released = threading.Event()
        self._lock = threading.Lock()

    def release(self) -> None:
        self._released.set()

    def generate(self, image_path: Path, prompt: str, *, timeout: float, cancellation=None, **_kwargs) -> str:
        with self._lock:
            self.calls.append(image_path)
        if image_path.stem in self.hold:
            self.started.set()
            if not self._released.wait(30):
                raise AssertionError(f"FakeSession held {image_path.name} for too long")
        return self._respond(image_path, prompt)


@pytest.fixture
def fake_session() -> Iterator[Callable[[Callable[[Path, str], str]], FakeSession]]:
    sessions: List[FakeSession] = []

    def _make(respond: Callable[[Path, str], str]) -> FakeSession:
        session = FakeSession(respond)
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.release()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


@pytest.fixture
def make_main_window(qtbot) -> Iterator[Callable[..., object]]:
    """Factory: an offscreen MainWindow, optionally with *folder* loaded."""
    from imagetagger.ui.main_window import MainWindow

    def _make(folder: Optional[Path] = None):
        window = MainWindow()
        _live_windows.append(window)
        if folder is not None:
            expected = len(list(folder.glob("*.png")))
            window.load_directory(folder)
            qtbot.waitUntil(
                lambda: window._loader_thread is None and len(window.records) == expected,
                timeout=10_000,
            )
            for record in window.records:
                assert folder in record.image_path.parents, record.image_path
        return window

    yield _make
    for window in _live_windows:
        window.hide()


@pytest.fixture
def main_window(make_main_window, image_folder):
    """A MainWindow loaded with images a, b and c (texts tag_a, tag_b, tag_c)."""
    return make_main_window(image_folder({"a": "tag_a", "b": "tag_b", "c": "tag_c"}))
