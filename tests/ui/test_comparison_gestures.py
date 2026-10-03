"""Mouse and trackpad row actions on the merge dialog's comparison table.

Synthetic QWheelEvent / QMouseEvent objects are sent straight to the table's
viewport, where ComparisonGestureHandler is installed as an event filter.
Most tests use a standalone ComparisonPanel (what FixupDialog embeds); the
end-to-end tests at the bottom open the real dialog through MainWindow with
``merge_table_mouse_actions`` set in the window's config and check what the
merge writes.

The horizontal-scroll "stop idle" pause is measured with time.monotonic();
tests that exercise it swap the handler module's clock for a manual one.
"""
from __future__ import annotations

import time
import types
from typing import Callable, List, Optional

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication, QDialog

from imagetagger import config as _config
from imagetagger.ui.merge_dialog import FixupDialog
from imagetagger.ui.panels import comparison_gesture_handler as gesture_module
from imagetagger.ui.panels.comparison_panel import ComparisonPanel
from imagetagger.utils.sidecar import SidecarData

POINTER_ROW = _config.MERGE_TABLE_HSCROLL_TARGET_POINTER_ROW
SELECTED_ROW = _config.MERGE_TABLE_HSCROLL_TARGET_SELECTED_ROW
POINTER_ON_SELECTED = _config.MERGE_TABLE_HSCROLL_TARGET_POINTER_ON_SELECTED

# Rows of the standard panel (current: old, keep, mine; proposed: keep, new):
#   0  old   | ✕ |        current only, not validated
#   1  keep  |   | keep   exact match, no action
#   2  mine  | ✕ |        current only, not validated
#   3        | ← | new    proposed only
OLD, KEEP, MINE, NEW = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides) -> dict:
    """Explicit ComparisonPanel mouse settings: everything off unless overridden."""
    values = {
        "merge_table_double_click_action_enabled": False,
        "merge_table_swipe_actions_enabled": False,
        "merge_table_horizontal_scroll_actions_enabled": False,
        "merge_table_horizontal_scroll_reverse_enabled": False,
        "merge_table_horizontal_scroll_stop_idle_seconds": 0.45,
        "merge_table_horizontal_scroll_row_target_mode": POINTER_ROW,
    }
    for key, value in overrides.items():
        full = f"merge_table_{key}"
        assert full in values, key
        values[full] = value
    return values


@pytest.fixture
def make_panel(qtbot) -> Callable[..., ComparisonPanel]:
    def _make(
        current=("old", "keep", "mine"),
        proposed=("keep", "new"),
        allow_left_delete: bool = True,
        **settings,
    ) -> ComparisonPanel:
        panel = ComparisonPanel(
            list(current),
            "",
            list(proposed),
            [],
            allow_left_delete=allow_left_delete,
            fixup_tag_keys={tag.casefold() for tag in proposed},
            **_settings(**settings),
        )
        qtbot.addWidget(panel)
        panel.update_difference_highlights()
        panel.resize(700, 400)
        panel.show()
        QApplication.processEvents()
        return panel

    return _make


class Clock:
    """Stands in for the gesture module's ``time`` so idle pauses are exact."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(gesture_module, "time", types.SimpleNamespace(monotonic=fake.monotonic))
    return fake


def _row_point(panel: ComparisonPanel, row: int, column: int = 2) -> QPointF:
    table = panel.comparison_table
    rect = table.visualRect(table.model().index(row, column))
    assert rect.isValid() and rect.height() > 0, (row, column)
    return QPointF(rect.center())


def _below_last_row(panel: ComparisonPanel) -> QPointF:
    table = panel.comparison_table
    rect = table.visualRect(table.model().index(table.rowCount() - 1, 2))
    point = QPointF(rect.center().x(), rect.bottom() + 20)
    assert point.y() < table.viewport().height()
    assert table.rowAt(int(point.y())) == -1
    return point


def _global(panel: ComparisonPanel, pos: QPointF) -> QPointF:
    return QPointF(panel.comparison_table.viewport().mapToGlobal(pos.toPoint()))


def wheel(
    panel: ComparisonPanel,
    pos: QPointF,
    dx: int = 0,
    dy: int = 0,
    *,
    trackpad: bool = True,
    phase: Qt.ScrollPhase = Qt.ScrollPhase.ScrollUpdate,
) -> QWheelEvent:
    """A trackpad scroll (pixel + angle delta) or a mouse wheel notch (angle delta only)."""
    pixel = QPoint(dx, dy) if trackpad else QPoint()
    angle = QPoint(dx, dy)
    event = QWheelEvent(
        pos,
        _global(panel, pos),
        pixel,
        angle,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        phase if trackpad else Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    event.setAccepted(False)
    QApplication.sendEvent(panel.comparison_table.viewport(), event)
    return event


def scroll_over(panel: ComparisonPanel, row: int, dx: int, **kwargs) -> QWheelEvent:
    return wheel(panel, _row_point(panel, row), dx, **kwargs)


def _mouse(panel, kind: QEvent.Type, pos: QPointF, button, buttons) -> None:
    event = QMouseEvent(kind, pos, _global(panel, pos), button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(panel.comparison_table.viewport(), event)


def swipe(panel: ComparisonPanel, row: int, dx: float, dy: float = 0.0, steps: int = 4) -> None:
    """Press on *row* (Proposed column), drag by (dx, dy), release."""
    left = Qt.MouseButton.LeftButton
    start = _row_point(panel, row)
    if dx < 0:  # start near the right edge of the Proposed column so the drag fits
        start = QPointF(panel.comparison_table.viewport().width() - 20, start.y())
    else:
        start = QPointF(20, start.y())
    _mouse(panel, QEvent.Type.MouseButtonPress, start, left, left)
    for step in range(1, steps + 1):
        point = QPointF(start.x() + dx * step / steps, start.y() + dy * step / steps)
        _mouse(panel, QEvent.Type.MouseMove, point, Qt.MouseButton.NoButton, left)
    end = QPointF(start.x() + dx, start.y() + dy)
    _mouse(panel, QEvent.Type.MouseButtonRelease, end, left, Qt.MouseButton.NoButton)


def double_click(panel: ComparisonPanel, row: int, column: int = 2) -> None:
    left = Qt.MouseButton.LeftButton
    pos = _row_point(panel, row, column)
    _mouse(panel, QEvent.Type.MouseButtonPress, pos, left, left)
    _mouse(panel, QEvent.Type.MouseButtonRelease, pos, left, Qt.MouseButton.NoButton)
    _mouse(panel, QEvent.Type.MouseButtonDblClick, pos, left, left)
    _mouse(panel, QEvent.Type.MouseButtonRelease, pos, left, Qt.MouseButton.NoButton)


def current(panel: ComparisonPanel) -> List[str]:
    """The Current column: what the dialog merges."""
    return panel.selected_annotations()


def select(panel: ComparisonPanel, row: int) -> None:
    assert panel.select_comparison_row(row, Qt.FocusReason.OtherFocusReason)
    assert panel.comparison_table.currentRow() == row


def _row_texts(panel: ComparisonPanel, row: int) -> tuple:
    left = panel._current_texts(panel.left_list)
    right = panel._current_texts(panel.right_list)
    left_index, right_index = panel.table_row_map[row]
    return (
        left[left_index] if left_index is not None else None,
        right[right_index] if right_index is not None else None,
    )


INITIAL = ["old", "keep", "mine"]


def test_standard_panel_layout(make_panel):
    panel = make_panel()
    assert [_row_texts(panel, row) for row in range(4)] == [
        ("old", None),
        ("keep", "keep"),
        ("mine", None),
        (None, "new"),
    ]
    assert panel._table_row_action_symbols == {OLD: "✕", MINE: "✕", NEW: "←"}


# ---------------------------------------------------------------------------
# Horizontal scroll: enable flag, direction, reverse
# ---------------------------------------------------------------------------


def test_hscroll_disabled_does_nothing(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=False)
    scroll_over(panel, NEW, -200)
    scroll_over(panel, OLD, 200)
    scroll_over(panel, NEW, -120, trackpad=False)
    assert current(panel) == INITIAL


def test_hscroll_left_applies_and_right_removes(make_panel, clock):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    scroll_over(panel, NEW, -100)
    assert current(panel) == INITIAL + ["new"]

    clock.advance(1)
    scroll_over(panel, OLD, 100)
    assert current(panel) == ["keep", "mine", "new"]


def test_hscroll_reverse_swaps_directions(make_panel, clock):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_reverse_enabled=True)
    scroll_over(panel, NEW, 100)
    assert current(panel) == INITIAL + ["new"]

    clock.advance(1)
    scroll_over(panel, OLD, -100)
    assert current(panel) == ["keep", "mine", "new"]

    # the unreversed directions now hit rows without that action
    clock.advance(1)
    scroll_over(panel, 0, 100)  # "keep": apply is a no-op on an exact match
    clock.advance(1)
    scroll_over(panel, _find_row(panel, "mine", None), 100)  # apply: nothing proposed
    assert current(panel) == ["keep", "mine", "new"]


def test_hscroll_accumulates_small_deltas_to_the_threshold(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    for _ in range(2):
        scroll_over(panel, NEW, -30)
    assert current(panel) == INITIAL  # 60 px < 84 px
    scroll_over(panel, NEW, -30)
    assert current(panel) == INITIAL + ["new"]


def test_hscroll_small_deltas_in_opposite_directions_cancel_out(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    for dx in (-60, 50, -60, 50):
        scroll_over(panel, NEW, dx)
    assert current(panel) == INITIAL


def test_hscroll_accumulator_restarts_on_another_row(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    scroll_over(panel, OLD, 60)
    scroll_over(panel, MINE, 60)  # different row: starts from zero
    assert current(panel) == INITIAL
    scroll_over(panel, MINE, 30)
    assert current(panel) == ["old", "keep"]


def test_mouse_wheel_notch_counts_as_a_full_scroll(make_panel, clock):
    """A tilt/thumb wheel has only angleDelta; one 120-unit notch is enough."""
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    scroll_over(panel, NEW, -120, trackpad=False)
    assert current(panel) == INITIAL + ["new"]
    clock.advance(1)
    scroll_over(panel, MINE, 120, trackpad=False)
    assert current(panel) == ["old", "keep", "new"]


def test_hscroll_action_selects_the_row_it_acts_on(make_panel, clock):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=POINTER_ROW)
    select(panel, OLD)
    scroll_over(panel, NEW, -100)
    assert current(panel) == INITIAL + ["new"]
    # the "new" row moved to the Current side; the table stays consistent
    assert ("new", "new") in [_row_texts(panel, row) for row in range(panel.comparison_table.rowCount())]


# ---------------------------------------------------------------------------
# Horizontal scroll: one action per gesture, idle stop
# ---------------------------------------------------------------------------


def test_hscroll_one_action_until_scrolling_stops(make_panel, clock):
    panel = make_panel(
        horizontal_scroll_actions_enabled=True,
        horizontal_scroll_stop_idle_seconds=0.45,
    )
    scroll_over(panel, OLD, 100)
    assert current(panel) == ["keep", "mine"]

    # The same gesture keeps going (momentum) for two seconds over the rows
    # that slid under the pointer: nothing else happens.
    for _ in range(20):
        clock.advance(0.1)
        scroll_over(panel, OLD, 100, phase=Qt.ScrollPhase.ScrollMomentum)
    assert current(panel) == ["keep", "mine"]

    # Each blocked event pushed the pause out again: 0.4 s is not yet enough.
    clock.advance(0.4)
    scroll_over(panel, OLD, 100)
    assert current(panel) == ["keep", "mine"]

    # After a real pause the next scroll acts again.
    clock.advance(0.5)
    scroll_over(panel, 0, 100)
    assert current(panel) == ["mine"]


def test_hscroll_pause_is_measured_in_real_time(make_panel, qtbot):
    idle = 0.05
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_stop_idle_seconds=idle)
    scroll_over(panel, OLD, 100)
    acted_at = time.monotonic()
    assert current(panel) == ["keep", "mine"]

    scroll_over(panel, 0, 100)  # immediately: still the same gesture
    blocked_at = time.monotonic()
    assert current(panel) == ["keep", "mine"]
    assert blocked_at - acted_at < idle, "machine too slow for this check"

    qtbot.waitUntil(lambda: time.monotonic() > blocked_at + idle * 2, timeout=2000)
    scroll_over(panel, 0, 100)
    assert current(panel) == ["mine"]


def test_hscroll_zero_pause_lets_a_long_scroll_run_several_actions(make_panel, clock):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_stop_idle_seconds=0.0)
    for _ in range(3):
        scroll_over(panel, 0, 100)
    assert current(panel) == []


def test_hscroll_leaving_the_table_ends_the_gesture(make_panel, clock):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    scroll_over(panel, OLD, 100)
    assert current(panel) == ["keep", "mine"]
    QApplication.sendEvent(panel.comparison_table.viewport(), QEvent(QEvent.Type.Leave))
    scroll_over(panel, 0, 100)
    assert current(panel) == ["mine"]


def test_hscroll_scroll_end_event_without_delta_is_ignored(make_panel, clock):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    scroll_over(panel, NEW, -60)
    wheel(panel, _row_point(panel, NEW), 0, 0, phase=Qt.ScrollPhase.ScrollEnd)
    scroll_over(panel, NEW, -30)
    assert current(panel) == INITIAL + ["new"]


# ---------------------------------------------------------------------------
# Horizontal scroll: vertical scrolling is left alone
# ---------------------------------------------------------------------------


def test_vertical_scroll_scrolls_the_table_and_runs_no_action(make_panel):
    tags = [f"tag{i:02d}" for i in range(40)]
    panel = make_panel(
        current=tags,
        proposed=["new"],
        horizontal_scroll_actions_enabled=True,
        swipe_actions_enabled=True,
    )
    bar = panel.comparison_table.verticalScrollBar()
    assert bar.maximum() > 0
    assert bar.value() == 0

    wheel(panel, _row_point(panel, 1), 0, -120, trackpad=False)
    assert bar.value() > 0
    after_notch = bar.value()
    wheel(panel, _row_point(panel, 1), 0, -80)  # trackpad
    assert bar.value() > after_notch
    # Mostly-vertical diagonal scrolls are vertical too.
    wheel(panel, _row_point(panel, 1), 70, -100)
    wheel(panel, _row_point(panel, 1), 70, -100, trackpad=False)
    assert current(panel) == tags


def test_horizontal_scroll_does_not_move_the_vertical_scrollbar(make_panel):
    tags = [f"tag{i:02d}" for i in range(40)]
    panel = make_panel(current=tags, proposed=["new"], horizontal_scroll_actions_enabled=True)
    bar = panel.comparison_table.verticalScrollBar()
    scroll_over(panel, 1, 40)
    assert bar.value() == 0


# ---------------------------------------------------------------------------
# Horizontal scroll: row targeting modes
# ---------------------------------------------------------------------------


def test_target_pointer_row_acts_under_the_pointer_regardless_of_selection(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=POINTER_ROW)
    select(panel, NEW)
    scroll_over(panel, OLD, 100)
    assert current(panel) == ["keep", "mine"]


def test_target_pointer_row_ignores_scrolls_below_the_rows(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=POINTER_ROW)
    select(panel, OLD)
    for _ in range(3):
        wheel(panel, _below_last_row(panel), 100)
    assert current(panel) == INITIAL


def test_target_selected_row_acts_wherever_the_pointer_is(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=SELECTED_ROW)
    select(panel, NEW)
    scroll_over(panel, OLD, -100)  # pointer on "old", selected row is "new"
    assert current(panel) == INITIAL + ["new"]


def test_target_selected_row_works_below_the_rows_too(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=SELECTED_ROW)
    select(panel, MINE)
    wheel(panel, _below_last_row(panel), 100)
    assert current(panel) == ["old", "keep"]


def test_target_selected_row_without_selection_does_nothing(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=SELECTED_ROW)
    panel.comparison_table.setCurrentCell(-1, -1)
    assert panel.comparison_table.currentRow() == -1
    scroll_over(panel, OLD, 100)
    scroll_over(panel, NEW, -100)
    assert current(panel) == INITIAL


def test_target_pointer_on_selected_needs_both(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=POINTER_ON_SELECTED)
    select(panel, NEW)
    scroll_over(panel, OLD, 100)
    scroll_over(panel, OLD, 100)
    assert current(panel) == INITIAL
    scroll_over(panel, NEW, -100)
    assert current(panel) == INITIAL + ["new"]


def test_target_pointer_on_selected_without_selection_does_nothing(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=POINTER_ON_SELECTED)
    panel.comparison_table.setCurrentCell(-1, -1)
    scroll_over(panel, OLD, 100)
    assert current(panel) == INITIAL


def test_unknown_target_mode_falls_back_to_pointer_on_selected(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=99)
    select(panel, MINE)
    scroll_over(panel, OLD, 100)
    assert current(panel) == INITIAL
    scroll_over(panel, MINE, 100)
    assert current(panel) == ["old", "keep"]


def test_moving_off_target_between_events_restarts_the_count(make_panel):
    """Pointer-on-selected: deltas over another row don't add up toward the selected one."""
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=POINTER_ON_SELECTED)
    select(panel, MINE)
    scroll_over(panel, MINE, 60)
    scroll_over(panel, OLD, 60)  # off target: resets
    scroll_over(panel, MINE, 60)
    assert current(panel) == INITIAL
    scroll_over(panel, MINE, 30)
    assert current(panel) == ["old", "keep"]


# ---------------------------------------------------------------------------
# Horizontal scroll: rows without that action
# ---------------------------------------------------------------------------


def test_hscroll_remove_on_a_proposed_only_row_does_nothing(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    scroll_over(panel, NEW, 100)
    assert current(panel) == INITIAL
    # Nothing ran, so there is no pause to wait out: the other direction works at once.
    scroll_over(panel, NEW, -100)
    assert current(panel) == INITIAL + ["new"]


def test_hscroll_apply_on_a_current_only_row_does_nothing(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    scroll_over(panel, OLD, -100)
    scroll_over(panel, OLD, -100)
    assert current(panel) == INITIAL
    scroll_over(panel, OLD, 100)
    assert current(panel) == ["keep", "mine"]


def test_hscroll_apply_on_a_matching_row_keeps_the_value(make_panel):
    panel = make_panel(horizontal_scroll_actions_enabled=True)
    scroll_over(panel, KEEP, -100)
    assert current(panel) == INITIAL


def test_hscroll_apply_on_a_fuzzy_match_replaces_the_current_value(make_panel):
    panel = make_panel(
        current=["red car"],
        proposed=["red cars"],
        horizontal_scroll_actions_enabled=True,
    )
    assert panel.table_row_map == [(0, 0)]
    assert panel._table_row_action_symbols == {0: "←"}
    scroll_over(panel, 0, -100)
    assert current(panel) == ["red cars"]


def test_hscroll_ignored_while_the_table_is_disabled(make_panel):
    """The merge dialog disables the table in crop mode."""
    panel = make_panel(horizontal_scroll_actions_enabled=True, swipe_actions_enabled=True,
                       double_click_action_enabled=True)
    panel.comparison_table.setEnabled(False)
    scroll_over(panel, OLD, 100)
    swipe(panel, OLD, 150)
    double_click(panel, NEW)
    assert current(panel) == INITIAL


# ---------------------------------------------------------------------------
# Swipe
# ---------------------------------------------------------------------------


def test_swipe_disabled_does_nothing(make_panel):
    panel = make_panel(swipe_actions_enabled=False)
    swipe(panel, OLD, 200)
    swipe(panel, NEW, -200)
    assert current(panel) == INITIAL


def test_swipe_right_removes_the_current_value(make_panel):
    panel = make_panel(swipe_actions_enabled=True)
    swipe(panel, MINE, 150)
    assert current(panel) == ["old", "keep"]


def test_swipe_left_applies_the_proposed_value(make_panel):
    panel = make_panel(swipe_actions_enabled=True)
    swipe(panel, NEW, -150)
    assert current(panel) == INITIAL + ["new"]


def test_swipe_ignores_the_hscroll_reverse_setting(make_panel):
    panel = make_panel(swipe_actions_enabled=True, horizontal_scroll_reverse_enabled=True)
    swipe(panel, NEW, -150)
    assert current(panel) == INITIAL + ["new"]


def test_swipe_needs_the_minimum_distance(make_panel):
    panel = make_panel(swipe_actions_enabled=True)
    swipe(panel, OLD, 60)
    swipe(panel, NEW, -60)
    assert current(panel) == INITIAL


def test_swipe_ending_on_another_row_does_nothing(make_panel):
    panel = make_panel(swipe_actions_enabled=True)
    height = panel.comparison_table.visualRect(panel.comparison_table.model().index(OLD, 0)).height()
    swipe(panel, OLD, 150, dy=height + 2, steps=1)
    assert current(panel) == INITIAL


def test_swipe_on_rows_without_that_action_does_nothing(make_panel):
    panel = make_panel(swipe_actions_enabled=True)
    swipe(panel, NEW, 150)  # nothing current to remove
    swipe(panel, OLD, -150)  # nothing proposed to apply
    assert current(panel) == INITIAL


def test_swipe_is_one_action_per_drag(make_panel):
    """Each drag acts once, and a new drag acts again (there is no idle pause)."""
    panel = make_panel(swipe_actions_enabled=True, horizontal_scroll_stop_idle_seconds=5.0)
    swipe(panel, OLD, 300, steps=10)
    assert current(panel) == ["keep", "mine"]
    swipe(panel, 0, 300)
    assert current(panel) == ["mine"]


def test_click_without_drag_changes_nothing(make_panel):
    panel = make_panel(swipe_actions_enabled=True)
    swipe(panel, NEW, 0, steps=1)
    assert current(panel) == INITIAL
    assert panel.comparison_table.currentRow() == NEW


def test_swipe_leaving_the_viewport_cancels(make_panel):
    panel = make_panel(swipe_actions_enabled=True)
    left = Qt.MouseButton.LeftButton
    start = QPointF(20, _row_point(panel, OLD).y())
    end = QPointF(220, start.y())
    _mouse(panel, QEvent.Type.MouseButtonPress, start, left, left)
    QApplication.sendEvent(panel.comparison_table.viewport(), QEvent(QEvent.Type.Leave))
    _mouse(panel, QEvent.Type.MouseButtonRelease, end, left, Qt.MouseButton.NoButton)
    assert current(panel) == INITIAL


# ---------------------------------------------------------------------------
# Double-click
# ---------------------------------------------------------------------------


def test_double_click_disabled_runs_no_action(make_panel):
    panel = make_panel(double_click_action_enabled=False)
    double_click(panel, NEW)
    double_click(panel, OLD)
    assert current(panel) == INITIAL


def test_double_click_runs_the_row_action_and_moves_on(make_panel):
    panel = make_panel(double_click_action_enabled=True)
    double_click(panel, OLD)  # ✕
    assert current(panel) == ["keep", "mine"]
    # the next row that still needs a decision is selected
    table = panel.comparison_table
    assert _row_texts(panel, table.currentRow()) == ("mine", None)

    double_click(panel, table.rowCount() - 1)  # ← on "new"
    assert current(panel) == ["keep", "mine", "new"]


def test_double_click_on_a_row_without_action_changes_nothing(make_panel):
    panel = make_panel(double_click_action_enabled=True)
    double_click(panel, KEEP)
    assert current(panel) == INITIAL
    assert panel.comparison_table.currentRow() == KEEP


def test_double_click_on_current_cell_edits_instead(make_panel, qtbot):
    panel = make_panel(double_click_action_enabled=True)
    double_click(panel, OLD, column=0)
    qtbot.waitUntil(lambda: panel.active_comparison_editor() is not None, timeout=2000)
    assert current(panel) == INITIAL
    panel.comparison_table.setFocus()  # closes the editor unchanged
    qtbot.waitUntil(lambda: panel.active_comparison_editor() is None, timeout=2000)


def test_double_click_on_empty_current_cell_runs_the_action(make_panel):
    panel = make_panel(double_click_action_enabled=True)
    double_click(panel, NEW, column=0)  # proposed-only row: the Current cell is not editable
    assert current(panel) == INITIAL + ["new"]


# ---------------------------------------------------------------------------
# Selection after a gesture
# ---------------------------------------------------------------------------


def test_hscroll_remove_keeps_a_row_selected(make_panel, clock):
    """Like the ✕ button, Enter, double-click and the Right key, a gesture remove moves on."""
    panel = make_panel(horizontal_scroll_actions_enabled=True, horizontal_scroll_row_target_mode=SELECTED_ROW)
    select(panel, OLD)
    scroll_over(panel, OLD, 100)
    assert current(panel) == ["keep", "mine"]
    table = panel.comparison_table
    assert table.currentRow() >= 0
    assert _row_texts(panel, table.currentRow()) == ("mine", None)

    clock.advance(1)
    scroll_over(panel, OLD, 100)  # acts on the selected row again
    assert current(panel) == ["keep"]


def test_swipe_remove_keeps_a_row_selected(make_panel):
    panel = make_panel(swipe_actions_enabled=True)
    swipe(panel, OLD, 150)
    assert current(panel) == ["keep", "mine"]
    assert panel.comparison_table.currentRow() >= 0


def test_hscroll_apply_moves_to_the_next_row_needing_a_decision(make_panel, clock):
    panel = make_panel(
        current=["old"],
        proposed=["new", "other"],
        horizontal_scroll_actions_enabled=True,
        horizontal_scroll_row_target_mode=SELECTED_ROW,
    )
    # rows: old ✕ | new ← | other ←
    select(panel, 1)
    scroll_over(panel, 1, -100)
    assert current(panel) == ["old", "new"]
    table = panel.comparison_table
    assert _row_texts(panel, table.currentRow()) == (None, "other")

    clock.advance(1)
    wheel(panel, _below_last_row(panel), -100)
    assert current(panel) == ["old", "new", "other"]


# ---------------------------------------------------------------------------
# End to end: the gesture's decision is what the merge writes
# ---------------------------------------------------------------------------


class Driver:
    def __init__(self) -> None:
        self.steps: List[Callable[[FixupDialog], Optional[int]]] = []

    def exec(self, dialog: FixupDialog) -> int:
        if self.steps:
            code = self.steps.pop(0)(dialog)
            if code is not None:
                dialog.done(code)
        return dialog.result()


@pytest.fixture
def driver(monkeypatch) -> Driver:
    drv = Driver()
    monkeypatch.setattr(FixupDialog, "exec", lambda self: drv.exec(self))
    return drv


@pytest.fixture
def fixup_window(make_main_window, image_folder):
    folder = image_folder(
        {"a": "old, keep"},
        sidecars={"a": SidecarData(fixup_issues="issue", fixup_tags=["keep", "new"])},
    )
    return make_main_window(folder)


def _find_row(panel: ComparisonPanel, left: Optional[str], right: Optional[str]) -> int:
    for row in range(panel.comparison_table.rowCount()):
        if _row_texts(panel, row) == (left, right):
            return row
    raise AssertionError((left, right))


GESTURES = {
    "hscroll": (
        {"horizontal_scroll_actions_enabled": True, "horizontal_scroll_row_target_mode": POINTER_ROW,
         "horizontal_scroll_stop_idle_seconds": 0.0},
        lambda panel, row, apply: scroll_over(panel, row, -100 if apply else 100),
    ),
    "hscroll-reversed": (
        {"horizontal_scroll_actions_enabled": True, "horizontal_scroll_reverse_enabled": True,
         "horizontal_scroll_row_target_mode": POINTER_ROW, "horizontal_scroll_stop_idle_seconds": 0.0},
        lambda panel, row, apply: scroll_over(panel, row, 100 if apply else -100),
    ),
    "swipe": (
        {"swipe_actions_enabled": True},
        lambda panel, row, apply: swipe(panel, row, -150 if apply else 150),
    ),
    "double-click": (
        {"double_click_action_enabled": True},
        lambda panel, row, apply: double_click(panel, row),
    ),
}


@pytest.mark.parametrize("gesture", sorted(GESTURES))
def test_gesture_decisions_are_what_the_merge_writes(gesture, fixup_window, driver, drain_writes):
    settings, act = GESTURES[gesture]
    mouse_actions = {
        "double_click_action_enabled": False,
        "swipe_actions_enabled": False,
        "horizontal_scroll_actions_enabled": False,
        "horizontal_scroll_reverse_enabled": False,
        "horizontal_scroll_stop_idle_seconds": 0.45,
        "horizontal_scroll_row_target_mode": POINTER_ON_SELECTED,
    }
    mouse_actions.update(settings)
    fixup_window._cfg["merge_table_mouse_actions"] = mouse_actions
    seen = {}

    def decide(dialog):
        dialog.show()
        panel = dialog._comparison_panel
        QApplication.processEvents()
        act(panel, _find_row(panel, "old", None), False)
        act(panel, _find_row(panel, None, "new"), True)
        seen["current"] = current(panel)
        seen["merge_enabled"] = dialog.merge_button.isEnabled()
        dialog._merge_and_next()

    driver.steps.append(decide)
    fixup_window.open_fixup_dialog()
    drain_writes()

    assert seen == {"current": ["keep", "new"], "merge_enabled": True}
    record = fixup_window.records[0]
    assert record.text_path.read_text(encoding="utf-8") == "keep, new"
    assert not record.has_pending_fixup


def test_gestures_off_by_config_leave_the_merge_unchanged(fixup_window, driver, drain_writes):
    fixup_window._cfg["merge_table_mouse_actions"] = {
        "double_click_action_enabled": False,
        "swipe_actions_enabled": False,
        "horizontal_scroll_actions_enabled": False,
        "horizontal_scroll_reverse_enabled": False,
        "horizontal_scroll_stop_idle_seconds": 0.0,
        "horizontal_scroll_row_target_mode": POINTER_ROW,
    }
    seen = {}

    def decide(dialog):
        dialog.show()
        panel = dialog._comparison_panel
        QApplication.processEvents()
        old_row = _find_row(panel, "old", None)
        new_row = _find_row(panel, None, "new")
        scroll_over(panel, old_row, 100)
        swipe(panel, old_row, 150)
        double_click(panel, new_row)
        scroll_over(panel, new_row, -100)
        seen["current"] = current(panel)
        return QDialog.DialogCode.Rejected

    driver.steps.append(decide)
    fixup_window.open_fixup_dialog()
    drain_writes()

    assert seen["current"] == ["old", "keep"]
    assert fixup_window.records[0].text_path.read_text(encoding="utf-8") == "old, keep"
