"""Mouse-gesture event handler for the comparison table.

Handles swipe (touch/trackpad drag), horizontal-scroll (trackpad two-finger
or mouse wheel), and double-click actions on the comparison table viewport.
All keyboard navigation stays in ``FixupDialog.eventFilter``.
"""

from __future__ import annotations

import time
from typing import Callable

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication, QTableWidget


# ── Row-target modes (mirrored from imagetagger.config) ────────────────────
HSCROLL_TARGET_POINTER_ROW = 1
HSCROLL_TARGET_SELECTED_ROW = 2
HSCROLL_TARGET_POINTER_ON_SELECTED = 3


class ComparisonGestureHandler(QObject):
    """Installs as an event filter on a QTableWidget's viewport to handle
    mouse swipe, horizontal-scroll, and double-click gestures.

    All actions are delegated back to the owning dialog via injected callbacks
    so the handler owns no comparison-table state except its own gesture
    tracking variables.
    """

    # Swipe thresholds
    _SWIPE_MIN_DISTANCE_PX = 90
    _SWIPE_MAX_VERTICAL_DRIFT_PX = 48
    _SWIPE_HORIZONTAL_BIAS = 1.2

    # Horizontal-scroll thresholds
    _HSCROLL_TRACKPAD_THRESHOLD_PX = 84.0
    _HSCROLL_MOUSE_NOTCH_EQUIVALENT_PX = 96.0

    def __init__(
        self,
        table: QTableWidget,
        *,
        swipe_enabled: bool,
        hscroll_enabled: bool,
        hscroll_reverse: bool,
        hscroll_stop_idle_seconds: float,
        hscroll_row_target_mode: int,
        double_click_enabled: bool,
        on_select_row: Callable[[int, Qt.FocusReason], None],
        on_remove_row: Callable[[int], bool],
        on_apply_row: Callable[[int], bool],
        on_begin_editing: Callable[[int], bool],
        on_trigger_row_action: Callable[[int], bool],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self._table = table
        self._swipe_enabled = bool(swipe_enabled)
        self._hscroll_enabled = bool(hscroll_enabled)
        self._hscroll_reverse = bool(hscroll_reverse)
        self._hscroll_stop_idle_seconds = max(0.0, float(hscroll_stop_idle_seconds))
        self._double_click_enabled = bool(double_click_enabled)

        if hscroll_row_target_mode in (
            HSCROLL_TARGET_POINTER_ROW,
            HSCROLL_TARGET_SELECTED_ROW,
            HSCROLL_TARGET_POINTER_ON_SELECTED,
        ):
            self._hscroll_row_target_mode = int(hscroll_row_target_mode)
        else:
            self._hscroll_row_target_mode = HSCROLL_TARGET_POINTER_ON_SELECTED

        # Injected callbacks
        self._on_select_row = on_select_row
        self._on_remove_row = on_remove_row
        self._on_apply_row = on_apply_row
        self._on_begin_editing = on_begin_editing
        self._on_trigger_row_action = on_trigger_row_action

        # Swipe tracking state
        self._swipe_drag_active = False
        self._swipe_start_pos: tuple[float, float] | None = None
        self._swipe_row = -1

        # Horizontal-scroll tracking state
        self._hscroll_accumulator_x = 0.0
        self._hscroll_row = -1
        self._hscroll_wait_for_stop = False
        self._hscroll_rearm_after = 0.0

        table.viewport().installEventFilter(self)

    # ------------------------------------------------------------------
    # QObject.eventFilter override
    # ------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        table = self._table
        try:
            if watched is not table.viewport():
                return super().eventFilter(watched, event)
        except RuntimeError:
            return super().eventFilter(watched, event)

        # ── Swipe ──────────────────────────────────────────────────────
        if self._swipe_enabled:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
                row = table.rowAt(int(event.position().y()))  # type: ignore[attr-defined]
                if row >= 0:
                    pos = event.position()  # type: ignore[attr-defined]
                    self._swipe_drag_active = True
                    self._swipe_start_pos = (float(pos.x()), float(pos.y()))
                    self._swipe_row = row
                else:
                    self._reset_swipe()

            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
                handled_swipe = False
                if (
                    self._swipe_drag_active
                    and self._swipe_start_pos is not None
                    and self._swipe_row >= 0
                ):
                    start_x, start_y = self._swipe_start_pos
                    pos = event.position()  # type: ignore[attr-defined]
                    row = table.rowAt(int(pos.y()))
                    if row == self._swipe_row:
                        handled_swipe = self._handle_swipe(
                            row,
                            float(pos.x()) - start_x,
                            float(pos.y()) - start_y,
                        )
                self._reset_swipe()
                if handled_swipe:
                    return True

            elif event.type() == QEvent.Type.Leave:
                self._reset_swipe()

        # ── Horizontal scroll ──────────────────────────────────────────
        if self._hscroll_enabled:
            if event.type() == QEvent.Type.Wheel:
                if not self._wheel_is_mostly_horizontal(event):
                    return super().eventFilter(watched, event)
                pointer_row = table.rowAt(int(event.position().y()))  # type: ignore[attr-defined]
                row = self._hscroll_target_row(pointer_row)
                if row is None:
                    self._reset_hscroll()
                    self._reset_hscroll_blocking()
                    return super().eventFilter(watched, event)
                delta_x = self._hscroll_delta_from_wheel(event)
                if delta_x == 0.0:
                    return super().eventFilter(watched, event)
                if self._handle_hscroll(row, delta_x):
                    return True

            elif event.type() == QEvent.Type.Leave:
                self._reset_hscroll()
                self._reset_hscroll_blocking()

        # ── Double-click ───────────────────────────────────────────────
        if (
            self._double_click_enabled
            and event.type() == QEvent.Type.MouseButtonDblClick
            and event.button() == Qt.MouseButton.LeftButton  # type: ignore[attr-defined]
        ):
            row = table.rowAt(int(event.position().y()))  # type: ignore[attr-defined]
            if row >= 0:
                column = table.columnAt(int(event.position().x()))  # type: ignore[attr-defined]
                if column == 0 and self._on_begin_editing(row):
                    return True
                table.setCurrentCell(row, max(table.currentColumn(), 0))
                if self._on_trigger_row_action(row):
                    return True

        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # Swipe internals
    # ------------------------------------------------------------------

    def _reset_swipe(self) -> None:
        self._swipe_drag_active = False
        self._swipe_start_pos = None
        self._swipe_row = -1

    def _swipe_min_distance_px(self) -> int:
        return max(self._SWIPE_MIN_DISTANCE_PX, QApplication.startDragDistance() * 6)

    def _handle_swipe(self, row: int, delta_x: float, delta_y: float) -> bool:
        min_distance = float(self._swipe_min_distance_px())
        max_vertical_drift = max(float(self._SWIPE_MAX_VERTICAL_DRIFT_PX), min_distance * 0.6)

        if abs(delta_x) < min_distance:
            return False
        if abs(delta_y) > max_vertical_drift:
            return False
        if abs(delta_x) < abs(delta_y) * self._SWIPE_HORIZONTAL_BIAS:
            return False

        self._on_select_row(row, Qt.FocusReason.MouseFocusReason)
        if delta_x > 0:
            return self._on_remove_row(row)
        return self._on_apply_row(row)

    # ------------------------------------------------------------------
    # Horizontal-scroll internals
    # ------------------------------------------------------------------

    def _reset_hscroll(self) -> None:
        self._hscroll_accumulator_x = 0.0
        self._hscroll_row = -1

    def _reset_hscroll_blocking(self) -> None:
        self._hscroll_wait_for_stop = False
        self._hscroll_rearm_after = 0.0

    @staticmethod
    def _wheel_is_mostly_horizontal(event) -> bool:
        pixel_delta = event.pixelDelta()  # type: ignore[attr-defined]
        if not pixel_delta.isNull():
            return abs(pixel_delta.x()) > abs(pixel_delta.y()) * 1.1
        angle_delta = event.angleDelta()  # type: ignore[attr-defined]
        if angle_delta.x() == 0:
            return False
        return abs(angle_delta.x()) > abs(angle_delta.y()) * 1.1

    def _hscroll_delta_from_wheel(self, event) -> float:
        pixel_delta = event.pixelDelta()  # type: ignore[attr-defined]
        if not pixel_delta.isNull():
            return float(pixel_delta.x())
        angle_delta = event.angleDelta()  # type: ignore[attr-defined]
        if angle_delta.x() == 0:
            return 0.0
        return (float(angle_delta.x()) / 120.0) * self._HSCROLL_MOUSE_NOTCH_EQUIVALENT_PX

    def _hscroll_target_row(self, pointer_row: int) -> int | None:
        mode = self._hscroll_row_target_mode
        selected_row = self._table.currentRow()

        if mode == HSCROLL_TARGET_POINTER_ROW:
            return pointer_row if pointer_row >= 0 else None
        if mode == HSCROLL_TARGET_SELECTED_ROW:
            return selected_row if selected_row >= 0 else None
        # Default: only act when pointer is over the selected row.
        if selected_row >= 0 and pointer_row == selected_row:
            return selected_row
        return None

    def _handle_hscroll(self, row: int, delta_x: float) -> bool:
        now = time.monotonic()
        if self._hscroll_wait_for_stop:
            stop_idle = self._hscroll_stop_idle_seconds
            if now < self._hscroll_rearm_after:
                self._hscroll_rearm_after = now + stop_idle
                return True
            self._reset_hscroll_blocking()
            self._reset_hscroll()

        if row != self._hscroll_row:
            self._hscroll_row = row
            self._hscroll_accumulator_x = 0.0

        self._hscroll_accumulator_x += delta_x
        if abs(self._hscroll_accumulator_x) < self._HSCROLL_TRACKPAD_THRESHOLD_PX:
            return True

        self._on_select_row(row, Qt.FocusReason.MouseFocusReason)
        direction_is_right = self._hscroll_accumulator_x > 0
        if self._hscroll_reverse:
            direction_is_right = not direction_is_right

        handled_action = (
            self._on_remove_row(row) if direction_is_right else self._on_apply_row(row)
        )

        self._hscroll_accumulator_x = 0.0
        self._hscroll_row = -1
        if handled_action and self._hscroll_stop_idle_seconds > 0.0:
            self._hscroll_wait_for_stop = True
            self._hscroll_rearm_after = now + self._hscroll_stop_idle_seconds

        return handled_action
