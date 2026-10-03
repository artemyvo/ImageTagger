"""Batch Autofix ratio: the worker, run_batch_autofix_ratio and MainWindow.batch_autofix_ratio.

Runs on temp folders of mis-sized PNGs.  The worker thread is real; tests
wait for its finish callback with qtbot.waitUntil and then for the thread
itself, so no QThread outlives its test.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QPushButton, QWidget

from imagetagger.ui import ratio_autofix
from imagetagger.ui.main_window import _ROLE_BADGES
from imagetagger.ui.ratio_autofix import (
    RatioAutofixResult,
    RatioAutofixTarget,
    _RatioAutofixWorker,
    run_batch_autofix_ratio,
    show_batch_autofix_summary,
)
from imagetagger.utils.aspect_ratio import autofix_crop, matching_ratio, parse_allowed_ratios

RATIOS = parse_allowed_ratios("1:1, 2:3, 3:4, 4:5, 16:9")

# name -> size.  Near misses keep >= 99% of the pixels when cropped.
SIZES = {
    "near_square": (101, 100),   # -> 100x100
    "near_wide": (320, 181),     # 16:9 -> 320x180
    "near_tall": (201, 300),     # 2:3 -> 200x300
    "fits": (160, 90),           # already 16:9
    "far": (300, 100),           # 3:1: closest fix keeps only ~59%
}
NEAR = ("near_square", "near_tall", "near_wide")


def _save_png(path: Path, width: int, height: int, color: str = "red") -> Path:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    assert image.save(str(path))
    return path


def _size(path: Path):
    with Image.open(path) as image:
        return image.size


@pytest.fixture
def folder(tmp_path) -> Path:
    root = tmp_path / "batch"
    root.mkdir()
    for name, (width, height) in SIZES.items():
        _save_png(root / f"{name}.png", width, height)
        (root / f"{name}.txt").write_text(f"tag_{name}", encoding="utf-8")
    return root


def _targets(folder: Path, names=None):
    targets = []
    for name in sorted(names or SIZES):
        width, height = SIZES[name]
        candidate = autofix_crop(width, height, RATIOS)
        if candidate is not None:
            targets.append(RatioAutofixTarget(folder / f"{name}.png", (width, height), candidate))
    return targets


def _wait_thread_done(thread) -> None:
    try:
        assert thread.wait(5000)
    except RuntimeError:
        pass  # already deleted: it had finished


class Run:
    def __init__(self) -> None:
        self.items: list = []
        self.finished = None

    def on_item_done(self, result) -> None:
        self.items.append(result)

    def on_finished(self, cropped, failed, stopped, failures) -> None:
        self.finished = (cropped, failed, stopped, list(failures))


@pytest.fixture
def parent(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


def _run(parent, targets, qtbot) -> Run:
    run = Run()
    thread, worker = run_batch_autofix_ratio(parent, targets, run.on_item_done, run.on_finished)
    qtbot.waitUntil(lambda: run.finished is not None, timeout=5000)
    _wait_thread_done(thread)
    return run


# ---------------------------------------------------------------------------
# Targets and the worker
# ---------------------------------------------------------------------------


def test_only_near_misses_are_targets(folder):
    assert [t.image_path.stem for t in _targets(folder)] == sorted(NEAR)


def test_worker_crops_and_reports_in_order(folder):
    targets = _targets(folder)
    worker = _RatioAutofixWorker(targets)
    progress, items, finished = [], [], []
    worker.progress.connect(lambda *args: progress.append(args))
    worker.item_done.connect(items.append)
    worker.finished.connect(lambda *args: finished.append(args))
    worker.run()  # synchronously, in this thread
    assert progress == [(i, 3, t.image_path.name) for i, t in enumerate(targets)]
    assert [(r.image_path, r.new_size, r.error) for r in items] == [
        (t.image_path, (t.candidate.width, t.candidate.height), "") for t in targets
    ]
    assert finished == [(3, 0, False)]


def test_worker_stop_before_start_crops_nothing(folder):
    before = {p: p.read_bytes() for p in folder.glob("*.png")}
    worker = _RatioAutofixWorker(_targets(folder))
    finished = []
    worker.finished.connect(lambda *args: finished.append(args))
    worker.request_stop()
    worker.run()
    assert finished == [(0, 0, True)]
    assert {p: p.read_bytes() for p in folder.glob("*.png")} == before


# ---------------------------------------------------------------------------
# run_batch_autofix_ratio
# ---------------------------------------------------------------------------


def test_batch_crops_near_misses_and_leaves_the_rest_byte_identical(folder, parent, qtbot):
    untouched = {p: p.read_bytes() for p in folder.iterdir() if p.stem not in NEAR or p.suffix == ".txt"}
    run = _run(parent, _targets(folder), qtbot)
    assert run.finished == (3, 0, False, [])
    assert {r.image_path.stem: r.new_size for r in run.items} == {
        "near_square": (100, 100),
        "near_tall": (200, 300),
        "near_wide": (320, 180),
    }
    for name in NEAR:
        path = folder / f"{name}.png"
        width, height = _size(path)
        assert matching_ratio(width, height, RATIOS) is not None
        assert width * height >= 0.99 * SIZES[name][0] * SIZES[name][1]
    for path, data in untouched.items():
        assert path.read_bytes() == data, path.name


def test_batch_crop_is_centred(tmp_path, parent, qtbot):
    # 202x200 with 1-px blue strips at both sides: a centred 200x200 crop removes both.
    path = tmp_path / "strips.png"
    image = QImage(202, 200, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    for x in (0, 201):
        for y in range(200):
            image.setPixelColor(x, y, QColor("blue"))
    assert image.save(str(path))
    candidate = autofix_crop(202, 200, RATIOS)
    assert (candidate.width, candidate.height) == (200, 200)
    run = _run(parent, [RatioAutofixTarget(path, (202, 200), candidate)], qtbot)
    assert run.finished[:3] == (1, 0, False)
    with Image.open(path) as result:
        rgb = result.convert("RGB")
        assert rgb.size == (200, 200)
        assert rgb.getpixel((0, 50)) == (255, 0, 0)
        assert rgb.getpixel((199, 50)) == (255, 0, 0)


def test_progress_dialog_is_shown_and_closed(folder, parent, qtbot):
    run = Run()
    thread, _ = run_batch_autofix_ratio(parent, _targets(folder), run.on_item_done, run.on_finished)
    dialogs = parent.findChildren(QProgressDialog)
    assert len(dialogs) == 1
    dialog = dialogs[0]
    assert dialog.windowTitle() == "Batch Autofix ratio"
    assert dialog.maximum() == 3
    qtbot.waitUntil(lambda: run.finished is not None, timeout=5000)
    _wait_thread_done(thread)
    qtbot.waitUntil(lambda: not parent.findChildren(QProgressDialog))  # deleteLater ran


def test_file_changed_since_listing_is_a_failure(folder, parent, qtbot):
    targets = _targets(folder)
    changed = folder / "near_square.png"
    _save_png(changed, 102, 100, "blue")  # size no longer what the target was computed for
    changed_bytes = changed.read_bytes()
    run = _run(parent, targets, qtbot)
    cropped, failed, stopped, failures = run.finished
    assert (cropped, failed, stopped) == (2, 1, False)
    assert [f.image_path for f in failures] == [changed]
    assert failures[0].new_size is None and failures[0].error
    assert changed.read_bytes() == changed_bytes
    assert _size(folder / "near_wide.png") == (320, 180)  # the others still ran


def test_missing_file_is_a_failure(folder, parent, qtbot):
    (folder / "near_tall.png").unlink()
    run = _run(parent, _targets(folder), qtbot)
    assert run.finished[:3] == (2, 1, False)
    assert run.finished[3][0].image_path.name == "near_tall.png"


def test_stop_leaves_done_images_and_skips_the_rest(folder, parent, qtbot, monkeypatch):
    targets = _targets(folder)
    first_started = threading.Event()
    release = threading.Event()
    real_crop = ratio_autofix.crop_image_file

    def _held_crop(path, box, expected_size=None):
        if path == targets[0].image_path:
            first_started.set()
            assert release.wait(10)
        return real_crop(path, box, expected_size=expected_size)

    monkeypatch.setattr(ratio_autofix, "crop_image_file", _held_crop)
    untouched = {t.image_path: t.image_path.read_bytes() for t in targets[1:]}
    run = Run()
    thread, _ = run_batch_autofix_ratio(parent, targets, run.on_item_done, run.on_finished)
    dialog = parent.findChildren(QProgressDialog)[0]
    qtbot.waitUntil(first_started.is_set)
    dialog.findChild(QPushButton).click()  # the Stop button
    release.set()
    qtbot.waitUntil(lambda: run.finished is not None, timeout=5000)
    _wait_thread_done(thread)
    assert run.finished == (1, 0, True, [])
    assert [r.image_path for r in run.items] == [targets[0].image_path]
    assert _size(targets[0].image_path) == (targets[0].candidate.width, targets[0].candidate.height)
    for path, data in untouched.items():
        assert path.read_bytes() == data


def test_empty_batch_finishes(parent, qtbot):
    run = _run(parent, [], qtbot)
    assert run.finished == (0, 0, False, [])


# ---------------------------------------------------------------------------
# show_batch_autofix_summary
# ---------------------------------------------------------------------------


def test_summary_without_failures_shows_no_box(parent, message_boxes):
    assert show_batch_autofix_summary(parent, 3, 0, False, []) == "Batch Autofix ratio: cropped 3 image(s)"
    assert message_boxes.calls == []


def test_summary_with_failures_and_stop(parent, message_boxes):
    failures = [RatioAutofixResult(Path(f"/x/img{i}.png"), None, "boom") for i in range(17)]
    summary = show_batch_autofix_summary(parent, 2, 17, True, failures)
    assert summary == "Batch Autofix ratio: cropped 2 image(s), 17 failed (stopped)"
    [(kind, title, text)] = message_boxes.calls
    assert (kind, title) == ("warning", "Batch Autofix ratio")
    assert "img0.png: boom" in text and "img14.png: boom" in text
    assert "img15.png" not in text and "… and 2 more" in text


# ---------------------------------------------------------------------------
# MainWindow.batch_autofix_ratio
# ---------------------------------------------------------------------------


def _badges(window, index):
    return window.list_widget.item(index).data(_ROLE_BADGES) or frozenset()


def _row(window, stem):
    return next(i for i, record in enumerate(window.records) if record.image_path.stem == stem)


@pytest.fixture
def window(make_main_window, folder):
    return make_main_window(folder)


def _run_window(window, qtbot):
    thread_holder = []
    window.batch_autofix_ratio_button.click()
    if window._ratio_autofix_thread is not None:
        thread_holder.append(window._ratio_autofix_thread)
    qtbot.waitUntil(lambda: window._ratio_autofix_thread is None, timeout=5000)
    for thread in thread_holder:
        _wait_thread_done(thread)


def test_window_badges_and_button_before_the_batch(window):
    for name in ("near_square", "near_tall", "near_wide", "far"):
        assert "✂️" in _badges(window, _row(window, name)), name
    assert "✂️" not in _badges(window, _row(window, "fits"))
    assert window.batch_autofix_ratio_button.isEnabled()
    assert "Crop 3 listed images" in window.batch_autofix_ratio_button.toolTip()
    assert window.status_ratio_label.text().endswith("4 ratio fixups")


def test_window_batch_crops_and_updates_badges(window, folder, qtbot, message_boxes):
    far_bytes = (folder / "far.png").read_bytes()
    fits_bytes = (folder / "fits.png").read_bytes()
    _run_window(window, qtbot)
    assert message_boxes.titles("question") == ["Batch Autofix ratio"]
    assert "Crop 3 images" in message_boxes.calls[0][2]
    for name in NEAR:
        index = _row(window, name)
        record = window.records[index]
        assert record._image_size == _size(record.image_path)
        assert "✂️" not in _badges(window, index), name
        assert not window._record_needs_fixup(record)
    assert "✂️" in _badges(window, _row(window, "far"))
    assert (folder / "far.png").read_bytes() == far_bytes
    assert (folder / "fits.png").read_bytes() == fits_bytes
    assert window.statusBar().currentMessage() == "Batch Autofix ratio: cropped 3 image(s)"
    assert not window.batch_autofix_ratio_button.isEnabled()  # nothing autofixable is left
    assert window.status_ratio_label.text().endswith("1 ratio fixup")


def test_window_batch_answered_no_changes_nothing(window, folder, qtbot, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.No)
    )
    before = {p: p.read_bytes() for p in folder.glob("*.png")}
    window.batch_autofix_ratio()
    assert window._ratio_autofix_thread is None
    assert {p: p.read_bytes() for p in folder.glob("*.png")} == before
    assert window.batch_autofix_ratio_button.isEnabled()


def test_window_batch_only_crops_listed_images(window, folder, qtbot):
    window.filter_input.setText('"tag_near_square"')
    hidden = [record.image_path.stem for i, record in enumerate(window.records) if window.list_widget.item(i).isHidden()]
    assert "near_wide" in hidden and "near_square" not in hidden
    wide_bytes = (folder / "near_wide.png").read_bytes()
    _run_window(window, qtbot)
    assert _size(folder / "near_square.png") == (100, 100)
    assert (folder / "near_wide.png").read_bytes() == wide_bytes
    assert window.statusBar().currentMessage() == "Batch Autofix ratio: cropped 1 image(s)"


def test_window_batch_reports_failures(window, folder, qtbot, message_boxes):
    _save_png(folder / "near_tall.png", 202, 300)  # changed behind the window's back
    _run_window(window, qtbot)
    assert window.statusBar().currentMessage() == "Batch Autofix ratio: cropped 2 image(s), 1 failed"
    assert message_boxes.titles("warning") == ["Batch Autofix ratio"]
    assert "near_tall.png" in message_boxes.calls[-1][2]
    assert _size(folder / "near_tall.png") == (202, 300)


def test_window_batch_with_nothing_to_fix(make_main_window, image_folder, message_boxes):
    window = make_main_window(image_folder({"a": "tag_a"}))
    window.batch_autofix_ratio()
    assert message_boxes.calls == []
    assert window.statusBar().currentMessage() == "Batch Autofix ratio: no listed image is a near miss"
    assert not window.batch_autofix_ratio_button.isEnabled()


def test_window_button_hidden_without_allowed_ratios(make_main_window, folder, isolated_config):
    isolated_config.write_text('{"allowed_ratios": ""}', encoding="utf-8")
    window = make_main_window(folder)
    assert window.batch_autofix_ratio_button.isHidden()
    assert not window.batch_autofix_ratio_button.isEnabled()
    assert not any("✂️" in _badges(window, i) for i in range(len(window.records)))


def test_window_refuses_a_second_batch_while_running(window, folder, qtbot, monkeypatch):
    release = threading.Event()
    real_crop = ratio_autofix.crop_image_file

    def _held_crop(path, box, expected_size=None):
        assert release.wait(10)
        return real_crop(path, box, expected_size=expected_size)

    monkeypatch.setattr(ratio_autofix, "crop_image_file", _held_crop)
    window.batch_autofix_ratio_button.click()
    thread = window._ratio_autofix_thread
    assert thread is not None
    assert not window.batch_autofix_ratio_button.isEnabled()
    assert window.batch_autofix_ratio_button.toolTip() == "Available when the running batch has finished"
    window.batch_autofix_ratio()  # ignored while busy
    assert window._ratio_autofix_thread is thread
    release.set()
    qtbot.waitUntil(lambda: window._ratio_autofix_thread is None, timeout=5000)
    _wait_thread_done(thread)
    assert window.statusBar().currentMessage() == "Batch Autofix ratio: cropped 3 image(s)"
