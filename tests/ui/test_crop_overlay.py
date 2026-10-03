"""The Fix ratio crop frame: CropOverlay driven through the Fixup dialog's ImagePane.

The pane is built on its own (as FixupDialog builds it) and shown offscreen,
so the preview label has a real size and the frame is laid out over the
scaled, letterboxed image.  Mouse input is sent as synthetic QMouseEvents.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from imagetagger.ui.main_window import _ROLE_BADGES
from imagetagger.ui.merge_dialog import FixupDialog
from imagetagger.ui.panels.image_pane import ImagePane
from imagetagger.utils.aspect_ratio import autofix_crop, parse_allowed_ratios

DEFAULT_RATIOS = parse_allowed_ratios("1:1, 2:3, 3:4, 4:5, 16:9")


def _save_png(path: Path, width: int, height: int) -> Path:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(path))
    return path


def _size_on_disk(path: Path):
    with Image.open(path) as image:
        return image.size


@pytest.fixture
def make_pane(qtbot):
    def _make(image_path: Path, size=(400, 300)) -> ImagePane:
        pane = ImagePane(image_path, confirm_delete=False, delete_image=None, regen_panel=QWidget())
        qtbot.addWidget(pane)
        pane.resize(*size)
        pane.show()
        qtbot.waitUntil(lambda: pane.image_size() is not None)
        return pane

    return _make


@pytest.fixture
def wide(tmp_path) -> Path:
    """300x100 (3:1): closest allowed ratio is 16:9 -> 178x100."""
    return _save_png(tmp_path / "wide.png", 300, 100)


@pytest.fixture
def tall(tmp_path) -> Path:
    """100x300 (1:3): closest allowed ratio is 9:16 -> 100x178."""
    return _save_png(tmp_path / "tall.png", 100, 300)


def _enter(pane: ImagePane, ratios=DEFAULT_RATIOS):
    assert pane.begin_ratio_fix(ratios)
    QApplication.processEvents()  # the crop bar shows: let the label re-layout and rescale
    return pane._crop_overlay


def _display(overlay):
    """Where the scaled image sits in the overlay (computed independently)."""
    label = overlay._label
    pixmap_size = label.pixmap().size()
    contents = label.contentsRect()
    x = contents.x() + (contents.width() - pixmap_size.width()) // 2
    y = contents.y() + (contents.height() - pixmap_size.height()) // 2
    scale = pixmap_size.width() / overlay.image_size()[0]
    return x, y, scale


def _to_widget(overlay, image_x: float, image_y: float) -> QPointF:
    x, y, scale = _display(overlay)
    return QPointF(x + image_x * scale, y + image_y * scale)


def _mouse(widget, kind: str, pos: QPointF, button=Qt.MouseButton.LeftButton) -> None:
    event_type = {
        "press": QEvent.Type.MouseButtonPress,
        "move": QEvent.Type.MouseMove,
        "release": QEvent.Type.MouseButtonRelease,
    }[kind]
    if kind == "move":
        event_button, buttons = Qt.MouseButton.NoButton, button
    elif kind == "press":
        event_button, buttons = button, button
    else:
        event_button, buttons = button, Qt.MouseButton.NoButton
    global_pos = QPointF(widget.mapToGlobal(pos.toPoint()))
    event = QMouseEvent(event_type, pos, global_pos, event_button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def _key(pane: ImagePane, key, modifiers=Qt.KeyboardModifier.NoModifier) -> bool:
    return pane.handle_crop_key(QKeyEvent(QEvent.Type.KeyPress, key, modifiers))


def _wheel(widget, dx: int, dy: int) -> None:
    pos = QPointF(10, 10)
    event = QWheelEvent(
        pos,
        QPointF(widget.mapToGlobal(pos.toPoint())),
        QPoint(dx, dy),
        QPoint(dx, dy),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


# ---------------------------------------------------------------------------
# Entering and leaving crop mode
# ---------------------------------------------------------------------------


def test_begin_shows_frame_with_closest_ratio_centred(make_pane, wide, qtbot):
    pane = make_pane(wide)
    assert not pane.is_crop_mode_active()
    with qtbot.waitSignal(pane.crop_mode_changed) as blocker:
        overlay = _enter(pane)
    assert blocker.args == [True]
    assert pane.is_crop_mode_active()
    assert overlay.isVisible() and pane._crop_bar.isVisible()
    assert overlay.geometry() == overlay._label.rect()
    assert overlay.image_size() == (300, 100)
    # 16:9 keeps the most pixels and is listed first; the frame spans the full height.
    assert pane._crop_ratio_combo.currentIndex() == 0
    assert pane._crop_ratio_combo.currentText().startswith("16:9")
    assert overlay.crop_box() == (61, 0, 239, 100)


def test_candidates_never_include_the_full_image(make_pane, tmp_path):
    pane = make_pane(_save_png(tmp_path / "sq.png", 100, 100))
    _enter(pane)
    labels = [pane._crop_ratio_combo.itemText(i) for i in range(pane._crop_ratio_combo.count())]
    assert labels and not any(label.startswith("1:1 ") for label in labels)


def test_begin_refused_when_no_ratio_crops_anything(make_pane, tmp_path, qtbot):
    pane = make_pane(_save_png(tmp_path / "sq.png", 100, 100))
    with qtbot.waitSignal(pane.status_message) as blocker:
        assert not pane.begin_ratio_fix(parse_allowed_ratios("1:1"))
    assert "no allowed ratio" in blocker.args[0]
    assert not pane.is_crop_mode_active()
    assert not pane._crop_overlay.isVisible()


def test_begin_refused_while_image_is_loading(tmp_path, qtbot):
    pane = ImagePane(_save_png(tmp_path / "w.png", 300, 100), False, None, QWidget())
    qtbot.addWidget(pane)
    assert pane.image_size() is None  # load is still in flight
    assert not pane.begin_ratio_fix(DEFAULT_RATIOS)
    qtbot.waitUntil(lambda: pane.image_size() is not None)


def test_begin_twice_keeps_the_frame(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    overlay.nudge(5, 0)
    box = overlay.crop_box()
    assert pane.begin_ratio_fix(DEFAULT_RATIOS)
    assert overlay.crop_box() == box


def test_cancel_button_leaves_file_untouched(make_pane, wide, qtbot):
    before = wide.read_bytes()
    pane = make_pane(wide)
    overlay = _enter(pane)
    overlay.nudge(-20, 0)
    with qtbot.waitSignal(pane.crop_mode_changed) as blocker:
        pane.crop_cancel_button.click()
    assert blocker.args == [False]
    assert not pane.is_crop_mode_active()
    assert not overlay.isVisible() and not pane._crop_bar.isVisible()
    assert wide.read_bytes() == before
    assert pane.image_size() == (300, 100)


def test_escape_cancels_and_is_consumed(make_pane, wide, qtbot):
    before = wide.read_bytes()
    pane = make_pane(wide)
    _enter(pane)
    with qtbot.waitSignal(pane.status_message) as blocker:
        assert _key(pane, Qt.Key.Key_Escape)
    assert blocker.args == ["Fix ratio cancelled."]
    assert not pane.is_crop_mode_active()
    assert wide.read_bytes() == before
    # Outside crop mode the keys are not the pane's business.
    assert not _key(pane, Qt.Key.Key_Escape)
    assert not _key(pane, Qt.Key.Key_Left)


def test_reloading_the_image_ends_crop_mode(make_pane, wide, qtbot):
    pane = make_pane(wide)
    _enter(pane)
    pane.load_image(wide)
    assert not pane.is_crop_mode_active()
    assert not pane._crop_overlay.isVisible()
    qtbot.waitUntil(lambda: pane.image_size() is not None)


# ---------------------------------------------------------------------------
# Ratio choice and bounds
# ---------------------------------------------------------------------------


def test_every_ratio_choice_fits_the_ratio_and_the_image(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    combo = pane._crop_ratio_combo
    for index in range(combo.count()):
        combo.setCurrentIndex(index)
        candidate = pane._crop_candidates[index]
        left, top, right, bottom = overlay.crop_box()
        assert (right - left, bottom - top) == (candidate.width, candidate.height)
        assert 0 <= left and 0 <= top and right <= 300 and bottom <= 100
        ratio = candidate.ratio
        assert abs((right - left) * ratio.h - (bottom - top) * ratio.w) < max(ratio.w, ratio.h)


def test_changing_ratio_keeps_the_frame_centre(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    overlay.nudge(-10, 0)  # move off-centre first
    left, _, right, _ = overlay.crop_box()
    centre = (left + right) / 2
    one_to_one = next(i for i, c in enumerate(pane._crop_candidates) if c.ratio.label == "1:1")
    pane._crop_ratio_combo.setCurrentIndex(one_to_one)
    left, top, right, bottom = overlay.crop_box()
    assert (right - left, bottom - top) == (100, 100)
    assert abs((left + right) / 2 - centre) <= 1


def test_changing_ratio_near_the_edge_is_clamped(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    overlay.nudge(-1000, 0)
    assert overlay.crop_box()[0] == 0
    smallest = pane._crop_ratio_combo.count() - 1
    pane._crop_ratio_combo.setCurrentIndex(smallest)
    pane._crop_ratio_combo.setCurrentIndex(0)
    left, top, right, bottom = overlay.crop_box()
    assert left >= 0 and right <= 300 and (right - left, bottom - top) == (178, 100)


# ---------------------------------------------------------------------------
# Mouse, keyboard and wheel
# ---------------------------------------------------------------------------


def test_drag_moves_the_frame_by_scaled_distance(make_pane, wide, qtbot):
    pane = make_pane(wide)
    overlay = _enter(pane)
    _, _, scale = _display(overlay)
    assert scale > 1  # the 300x100 image is scaled up to fill the preview
    start = _to_widget(overlay, 150, 50)  # inside the frame: grabs it
    moves = []
    overlay.frame_moved.connect(lambda: moves.append(overlay.crop_box()))
    _mouse(overlay, "press", start)
    assert overlay.crop_box() == (61, 0, 239, 100)  # a press inside does not jump
    _mouse(overlay, "move", start + QPointF(-30 * scale, 40 * scale))
    _mouse(overlay, "release", start + QPointF(-30 * scale, 40 * scale))
    # Only x has room (the frame spans the full height).
    assert overlay.crop_box() == (31, 0, 209, 100)
    assert moves and moves[-1] == overlay.crop_box()
    # After release, moving does nothing.
    _mouse(overlay, "move", start)
    assert overlay.crop_box() == (31, 0, 209, 100)


def test_drag_is_clamped_to_the_image(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    start = _to_widget(overlay, 150, 50)
    _mouse(overlay, "press", start)
    _mouse(overlay, "move", start + QPointF(5000, 0))
    assert overlay.crop_box() == (122, 0, 300, 100)
    _mouse(overlay, "move", start + QPointF(-5000, -5000))
    assert overlay.crop_box() == (0, 0, 178, 100)
    _mouse(overlay, "release", start)


def test_click_outside_the_frame_recentres_it_under_the_pointer(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    _mouse(overlay, "press", _to_widget(overlay, 20, 50))
    _mouse(overlay, "release", _to_widget(overlay, 20, 50))
    assert overlay.crop_box() == (0, 0, 178, 100)  # centre 20 clamps to the left edge
    _mouse(overlay, "press", _to_widget(overlay, 260, 50))
    _mouse(overlay, "release", _to_widget(overlay, 260, 50))
    assert overlay.crop_box() == (122, 0, 300, 100)


def test_right_click_is_ignored(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    _mouse(overlay, "press", _to_widget(overlay, 20, 50), Qt.MouseButton.RightButton)
    assert overlay.crop_box() == (61, 0, 239, 100)
    assert overlay._drag_anchor is None


def test_letterboxed_tall_image_maps_widget_to_image_pixels(make_pane, tall):
    # A tall image in a wide preview: bars left and right, frame moves along y.
    pane = make_pane(tall, size=(500, 300))
    overlay = _enter(pane)
    x, y, scale = _display(overlay)
    assert x > 20  # horizontal letterbox
    assert overlay.crop_box() == (0, 61, 100, 239)
    # A click in the left bar, at image y=260, still maps to image rows.
    _mouse(overlay, "press", QPointF(2, y + 260 * scale))
    _mouse(overlay, "release", QPointF(2, y + 260 * scale))
    assert overlay.crop_box() == (0, 122, 100, 300)
    target = _to_widget(overlay, 50, 100)  # above the frame: recentres at y=100
    _mouse(overlay, "press", target)
    _mouse(overlay, "release", target)
    assert overlay.crop_box() == (0, 11, 100, 189)


def test_downscaled_image_maps_widget_to_image_pixels(make_pane, tmp_path):
    pane = make_pane(_save_png(tmp_path / "big.png", 1500, 500))
    overlay = _enter(pane)
    _, _, scale = _display(overlay)
    assert scale < 0.5
    assert overlay.crop_box() == (306, 0, 1195, 500)  # 16:9 -> 889x500
    start = _to_widget(overlay, 750, 250)
    _mouse(overlay, "press", start)
    _mouse(overlay, "move", start + QPointF(20, 0))
    _mouse(overlay, "release", start + QPointF(20, 0))
    assert overlay.crop_box()[0] == 306 + round(20 / scale)
    # One arrow step moves one preview pixel's worth of image pixels.
    before = overlay.crop_box()[0]
    assert _key(pane, Qt.Key.Key_Left)
    assert overlay.crop_box()[0] == before - round(1 / scale)


def test_overlay_follows_label_resize(make_pane, wide, qtbot):
    pane = make_pane(wide)
    overlay = _enter(pane)
    pane.resize(600, 450)
    qtbot.waitUntil(lambda: overlay._label.width() > 500)
    assert overlay.geometry() == overlay._label.rect()
    # Image-pixel state is independent of the preview scale.
    assert overlay.crop_box() == (61, 0, 239, 100)


def test_arrow_keys_nudge_and_shift_moves_ten_times(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    _, _, scale = _display(overlay)
    step = max(1, round(1 / scale))
    assert _key(pane, Qt.Key.Key_Right)
    assert overlay.crop_box()[0] == 61 + step
    assert _key(pane, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert overlay.crop_box()[0] == 61 + step - 10 * step
    assert _key(pane, Qt.Key.Key_Up)  # no room along y: consumed, no move
    assert overlay.crop_box()[1] == 0
    # Other modifiers are consumed without moving (they belong to shortcuts).
    left = overlay.crop_box()[0]
    assert _key(pane, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert overlay.crop_box()[0] == left
    # Keys crop mode does not use are left to the dialog.
    assert not _key(pane, Qt.Key.Key_A)


def test_wheel_moves_along_the_axis_with_room(make_pane, wide):
    pane = make_pane(wide)
    overlay = _enter(pane)
    _, _, scale = _display(overlay)
    _wheel(overlay, 0, -20)  # vertical scroll on a frame that can only move sideways
    assert overlay.crop_box()[1] == 0
    assert overlay.crop_box()[0] == 61 + round(20 / scale)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def test_apply_crops_the_file_to_the_frame(make_pane, wide, qtbot):
    # Left half red, right half blue: the kept region is verifiable.
    image = QImage(300, 100, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    for x in range(150, 300):
        for y in range(100):
            image.setPixelColor(x, y, QColor("blue"))
    assert image.save(str(wide))

    pane = make_pane(wide)
    overlay = _enter(pane)
    overlay.nudge(1000, 0)  # far right: keeps 122..300
    assert overlay.crop_box() == (122, 0, 300, 100)
    with qtbot.waitSignal(pane.image_cropped) as blocker:
        pane.crop_apply_button.click()
    assert blocker.args == [wide, 178, 100]
    assert _size_on_disk(wide) == (178, 100)
    with Image.open(wide) as result:
        rgb = result.convert("RGB")
        assert rgb.getpixel((0, 50)) == (255, 0, 0)  # column 122 was red
        assert rgb.getpixel((40, 50)) == (0, 0, 255)  # column 162 was blue
    assert not pane.is_crop_mode_active()
    qtbot.waitUntil(lambda: pane.image_size() == (178, 100))


def test_enter_key_applies(make_pane, wide, qtbot):
    pane = make_pane(wide)
    _enter(pane)
    with qtbot.waitSignal(pane.image_cropped):
        assert _key(pane, Qt.Key.Key_Return)
    assert _size_on_disk(wide) == (178, 100)


def test_enter_with_modifier_or_autorepeat_does_not_apply(make_pane, wide):
    before = wide.read_bytes()
    pane = make_pane(wide)
    _enter(pane)
    assert _key(pane, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier, "", True)
    assert pane.handle_crop_key(repeat)
    assert pane.is_crop_mode_active()
    assert wide.read_bytes() == before


def test_apply_refused_when_file_changed_since_framing(make_pane, wide, qtbot, message_boxes):
    pane = make_pane(wide)
    _enter(pane)
    _save_png(wide, 320, 100)  # external edit while the frame is up
    changed = wide.read_bytes()
    cropped = []
    pane.image_cropped.connect(lambda *args: cropped.append(args))
    pane.crop_apply_button.click()
    assert message_boxes.titles("warning") == ["Fix ratio failed"]
    assert cropped == []
    assert wide.read_bytes() == changed
    assert pane.is_crop_mode_active()  # the frame stays up after a failure


def test_own_crop_is_not_reported_as_an_external_edit(make_pane, wide, qtbot):
    pane = make_pane(wide)
    _enter(pane)
    with qtbot.waitSignal(pane.image_cropped):
        pane.crop_apply_button.click()
    qtbot.waitUntil(lambda: pane.image_size() == (178, 100))
    generation = pane._load_generation
    pane._image_reload_helper._schedule_pending_image_reload()
    pane._image_reload_helper._apply_pending_image_reload()
    assert pane._load_generation == generation  # no second load of the same file


def test_autofix_ratio_centre_crops_without_crop_mode(make_pane, tmp_path, qtbot):
    path = _save_png(tmp_path / "near.png", 101, 100)
    pane = make_pane(path)
    candidate = autofix_crop(101, 100, DEFAULT_RATIOS)
    assert candidate is not None
    with qtbot.waitSignal(pane.image_cropped) as blocker:
        assert pane.autofix_ratio(candidate)
    assert blocker.args == [path, 100, 100]
    assert _size_on_disk(path) == (100, 100)
    qtbot.waitUntil(lambda: pane.image_size() == (100, 100))


def test_autofix_ratio_refused_in_crop_mode(make_pane, wide):
    before = wide.read_bytes()
    pane = make_pane(wide)
    _enter(pane)
    assert not pane.autofix_ratio(pane._crop_candidates[0])
    assert wide.read_bytes() == before


# ---------------------------------------------------------------------------
# Main window: record and ✂️ badge follow the crop
# ---------------------------------------------------------------------------


def _badges(window, index):
    return window.list_widget.item(index).data(_ROLE_BADGES) or frozenset()


@pytest.fixture
def wide_window(make_main_window, image_folder):
    folder = image_folder({"a": "tag_a", "b": "tag_b"})
    _save_png(folder / "a.png", 300, 100)
    window = make_main_window(folder)
    assert window.current_index == 0
    assert "✂️" in _badges(window, 0)
    return window


def test_crop_updates_record_badge_and_status(make_pane, wide_window, qtbot):
    record = wide_window.records[0]
    pane = make_pane(record.image_path)
    pane.image_cropped.connect(wide_window._on_image_cropped)
    _enter(pane)
    with qtbot.waitSignal(pane.image_cropped):
        pane.crop_apply_button.click()
    assert record._image_size == (178, 100)
    assert not wide_window._record_needs_fixup(record)
    assert "✂️" not in _badges(wide_window, 0)
    assert not wide_window.status_ratio_label.isVisible()
    # The main preview shows the cropped file (its cache was dropped).
    cached = wide_window.image_view_controller._cached_pixmap
    assert cached is not None and (cached.width(), cached.height()) == (178, 100)


def test_fix_ratio_through_the_fixup_dialog(wide_window, monkeypatch, qtbot):
    record = wide_window.records[0]
    seen = []

    def _exec(dialog):
        pane = dialog._image_pane
        qtbot.waitUntil(lambda: pane.image_size() is not None)
        assert dialog._ratio_fix_available
        dialog._begin_ratio_fix()
        assert pane.is_crop_mode_active()
        assert not dialog._button_bar.isEnabled()  # crop mode parks the rest of the dialog
        with qtbot.waitSignal(pane.image_cropped):
            pane.crop_apply_button.click()
        seen.append(_size_on_disk(record.image_path))
        assert dialog._button_bar.isEnabled()
        dialog.done(QDialog.DialogCode.Rejected)
        return dialog.result()

    monkeypatch.setattr(FixupDialog, "exec", _exec)
    wide_window.open_fixup_dialog()
    assert seen == [(178, 100)]
    assert record._image_size == (178, 100)
    assert "✂️" not in _badges(wide_window, 0)
