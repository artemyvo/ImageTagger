from __future__ import annotations

import re
import sys
from typing import Callable

from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imagetagger.providers.llm_provider import (
    VisionLlmProvider,
    VisionLlmSession,
)
from imagetagger.ui.shortcuts import native_shortcut_text, platform_key_sequence, platform_key_sequences
from imagetagger.utils.fixup_parser import FixupData
from imagetagger.ui.panels.regenerate_panel import RegeneratePanel
from imagetagger.ui.panels.image_pane import ImagePane
from imagetagger.ui.panels.comparison_panel import ComparisonPanel


class FixupDialog(QDialog):
    NAVIGATE_PREV_CODE = 2
    NAVIGATE_NEXT_CODE = 3

    def __init__(
        self,
        current_tags: list[str],
        fixup_data: FixupData,
        image_path: Path | None = None,
        title_text: str | None = None,
        apply_annotations: Callable[[list[str], str], None] | None = None,
        clear_fixup: Callable[[], bool] | None = None,
        restore_fixup: Callable[[], bool] | None = None,
        can_navigate_prev: bool = False,
        can_navigate_next: bool = False,
        tag_suggestions: list[str] | None = None,
        normalize_annotation: Callable[[str], str] | None = None,
        normalize_tag: Callable[[str], str] | None = None,
        provider_session: VisionLlmSession | None = None,
        provider: VisionLlmProvider | None = None,
        regenerate_tags_enabled: bool = True,
        regenerate_description_enabled: bool = True,
        regenerate_timeout_seconds: int = 300,
        regenerate_retry_count: int = 3,
        regenerate_max_resolution_mpx: float = 5.0,
        regenerate_model_name: str = "",
        regenerate_model_endpoint: str = "",
        regenerate_user_hint: str = "",
        merge_table_double_click_action_enabled: bool = True,
        merge_table_swipe_actions_enabled: bool = False,
        merge_table_horizontal_scroll_actions_enabled: bool = False,
        merge_table_horizontal_scroll_reverse_enabled: bool = False,
        merge_table_horizontal_scroll_stop_idle_seconds: float = 0.45,
        merge_table_horizontal_scroll_row_target_mode: int = 3,
        delete_image: Callable[[], tuple[bool, bool]] | None = None,
        confirm_delete: bool = True,
        allow_left_delete: bool = True,
        fixup_tag_keys: set[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window_title_base = title_text or "Fixup"
        self.setWindowTitle(self._window_title_base)
        self.resize(1280, 640)
        self._apply_annotations = apply_annotations
        self._clear_fixup = clear_fixup
        self._restore_fixup = restore_fixup
        self._resolved = False
        self._undo_available = False
        self._image_path = Path(image_path) if isinstance(image_path, str) else image_path
        self._global_key_filter_installed = False
        self._delete_image = delete_image
        self._confirm_delete = bool(confirm_delete)

        # ── Keyboard shortcuts ────────────────────────────────────────────
        self.remove_left_tag_action = QAction("Remove Selected Existing Tags", self)
        delete_shortcuts = [QKeySequence("Delete"), QKeySequence("Backspace")]
        self.remove_left_tag_action.setShortcuts(delete_shortcuts)
        self._delete_rows_shortcut_hint = native_shortcut_text(delete_shortcuts)
        self.addAction(self.remove_left_tag_action)

        self.merge_and_next_alt_action = QAction("Merge and Next", self)
        merge_next_shortcut_labels = platform_key_sequences(
            ["Alt+Enter", "Alt+Return"],
            ["Alt+Enter", "Alt+Return"],
        )
        self._merge_next_shortcut_hint = native_shortcut_text(merge_next_shortcut_labels)
        self.merge_and_next_alt_action.triggered.connect(self._merge_and_next)
        self.addAction(self.merge_and_next_alt_action)

        self.focus_tag_input_action = QAction("Focus Tag Input", self)
        focus_tag_shortcut = platform_key_sequence("Alt+T", "Alt+T")
        self.focus_tag_input_action.setShortcut(focus_tag_shortcut)
        self._focus_tag_input_shortcut_hint = native_shortcut_text(focus_tag_shortcut)
        self.addAction(self.focus_tag_input_action)

        self.undo_alt_action = QAction("Undo", self)
        undo_shortcuts = QKeySequence.keyBindings(QKeySequence.StandardKey.Undo)
        self.undo_alt_action.setShortcuts(undo_shortcuts)
        self._undo_shortcut_hint = native_shortcut_text(undo_shortcuts)
        self.undo_alt_action.triggered.connect(self._undo_merge)
        self.addAction(self.undo_alt_action)

        self.prev_actionable_row_action = QAction("Previous Actionable Row", self)
        prev_actionable_shortcut = platform_key_sequence("Alt+Up", "Alt+Up")
        self.prev_actionable_row_action.setShortcut(prev_actionable_shortcut)
        self._prev_actionable_shortcut_hint = native_shortcut_text(prev_actionable_shortcut)
        self.addAction(self.prev_actionable_row_action)

        self.next_actionable_row_action = QAction("Next Actionable Row", self)
        next_actionable_shortcut = platform_key_sequence("Alt+Down", "Alt+Down")
        self.next_actionable_row_action.setShortcut(next_actionable_shortcut)
        self._next_actionable_shortcut_hint = native_shortcut_text(next_actionable_shortcut)
        self.addAction(self.next_actionable_row_action)

        # ── RegeneratePanel (created first so is_regenerating lambda works)
        self._regen_panel = RegeneratePanel(
            provider=provider,
            provider_session=provider_session,
            image_path_getter=lambda: self._image_path,
            get_current_proposed=lambda: self._comparison_panel.get_current_proposed_for_regen(),
            normalize_tag=normalize_tag or (lambda text: text),
            normalize_annotation=normalize_annotation or (lambda text: text),
            existing_tags_getter=lambda: self._comparison_panel.get_existing_tags_for_regen(),
            regenerate_tags_enabled=regenerate_tags_enabled,
            regenerate_description_enabled=regenerate_description_enabled,
            regenerate_timeout_seconds=regenerate_timeout_seconds,
            regenerate_retry_count=regenerate_retry_count,
            regenerate_max_resolution_mpx=regenerate_max_resolution_mpx,
            regenerate_model_name=regenerate_model_name,
            regenerate_model_endpoint=regenerate_model_endpoint,
            regenerate_user_hint=regenerate_user_hint,
            parent=self,
        )
        self._regen_panel.proposed_annotations_ready.connect(self._on_regen_proposed_ready)
        self._regen_panel.regeneration_finished.connect(self._refresh_button_state)

        # ── ComparisonPanel ───────────────────────────────────────────────
        self._comparison_panel = ComparisonPanel(
            initial_annotations=current_tags,
            initial_description=fixup_data.corrected_description,
            initial_tags=fixup_data.corrected_tags,
            initial_search_matches=fixup_data.search_matches or [],
            initial_vision_tags=fixup_data.vision_tags or [],
            normalize_annotation=normalize_annotation,
            normalize_tag=normalize_tag,
            is_regenerating=lambda: self._regen_panel.is_regenerating,
            tag_suggestions=tag_suggestions,
            merge_table_double_click_action_enabled=merge_table_double_click_action_enabled,
            merge_table_swipe_actions_enabled=merge_table_swipe_actions_enabled,
            merge_table_horizontal_scroll_actions_enabled=merge_table_horizontal_scroll_actions_enabled,
            merge_table_horizontal_scroll_reverse_enabled=merge_table_horizontal_scroll_reverse_enabled,
            merge_table_horizontal_scroll_stop_idle_seconds=merge_table_horizontal_scroll_stop_idle_seconds,
            merge_table_horizontal_scroll_row_target_mode=merge_table_horizontal_scroll_row_target_mode,
            allow_left_delete=allow_left_delete,
            fixup_tag_keys=fixup_tag_keys,
            parent=self,
        )
        self._comparison_panel.state_changed.connect(self._refresh_button_state)
        self._comparison_panel.selection_changed.connect(self._refresh_button_state)
        self._comparison_panel.comparison_table.installEventFilter(self)
        self._comparison_panel.comparison_table.viewport().installEventFilter(self)
        self._comparison_panel.comparison_table.setToolTip(
            f"Keyboard: {self._prev_actionable_shortcut_hint}/{self._next_actionable_shortcut_hint} "
            "jump actionable rows, Left applies proposed rows, "
            f"Enter triggers current row action, {self._delete_rows_shortcut_hint} removes selected current rows."
        )
        self._comparison_panel.left_tag_input.setToolTip(
            f"Type a tag and press Enter. {self._focus_tag_input_shortcut_hint} clears and focuses this field."
        )
        self.remove_left_tag_action.triggered.connect(
            self._comparison_panel.remove_selected_left_items
        )
        self.focus_tag_input_action.triggered.connect(
            self._comparison_panel.focus_left_tag_input
        )
        self.prev_actionable_row_action.triggered.connect(
            self._comparison_panel.activate_previous_actionable_row
        )
        self.next_actionable_row_action.triggered.connect(
            self._comparison_panel.activate_next_actionable_row
        )

        # ── ImagePane ─────────────────────────────────────────────────────
        self._image_pane = ImagePane(
            image_path=self._image_path,
            confirm_delete=self._confirm_delete,
            delete_image=self._delete_image,
            regen_panel=self._regen_panel,
            parent=self,
        )
        self._image_pane.status_message.connect(self._regen_panel.set_status)
        self._image_pane.delete_result.connect(self._on_image_pane_delete_result)

        # ── Auxiliary widgets ─────────────────────────────────────────────
        self.issues_label = QLabel(fixup_data.issues or "No issue details provided.", self)
        self.issues_label.setWordWrap(True)
        self.issues_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.issues_label.setStyleSheet("border: 1px solid palette(mid); padding: 6px;")

        # ── Buttons ───────────────────────────────────────────────────────
        self.accept_button = QPushButton("&Accept", self)
        accept_shortcut = platform_key_sequence("Alt+A", "Alt+A")
        self.accept_button.setShortcut(accept_shortcut)
        self._accept_shortcut_hint = native_shortcut_text(accept_shortcut)
        self.accept_button.setToolTip(
            f"Accept all proposed rows and merge ({self._accept_shortcut_hint})"
        )
        self.accept_button.clicked.connect(self._accept_all_without_close)

        self.merge_button = QPushButton("Merge", self)
        self.merge_button.setToolTip("Apply current merged annotations")
        self.merge_button.clicked.connect(self._merge_without_close)

        self.undo_button = QPushButton("Undo", self)
        self.undo_button.setEnabled(False)
        self.undo_button.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        self.undo_button.setToolTip(
            f"Undo the last merge or local changes ({self._undo_shortcut_hint})"
        )
        self.undo_button.clicked.connect(self._undo_merge)

        self.merge_next_button = QPushButton("Merge and Next", self)
        self.merge_next_button.setShortcut(merge_next_shortcut_labels[0])
        self.merge_next_button.setToolTip(
            f"Apply current annotations and go to next item, even with no local edits ({self._merge_next_shortcut_hint})"
        )
        self.merge_next_button.clicked.connect(self._merge_and_next)

        self.prev_button = QPushButton("Prev", self)
        self.prev_button.setEnabled(can_navigate_prev)
        prev_nav_shortcut = platform_key_sequence("Alt+Left", "Meta+[")
        self.prev_button.setShortcut(prev_nav_shortcut)
        self._prev_nav_shortcut_hint = native_shortcut_text(prev_nav_shortcut)
        self.prev_button.setToolTip(f"Go to previous item ({self._prev_nav_shortcut_hint})")
        self.prev_button.clicked.connect(self._navigate_prev)

        self.next_button = QPushButton("Next", self)
        self.next_button.setEnabled(can_navigate_next)
        next_nav_shortcut = platform_key_sequence("Alt+Right", "Meta+]")
        self.next_button.setShortcut(next_nav_shortcut)
        self._next_nav_shortcut_hint = native_shortcut_text(next_nav_shortcut)
        self.next_button.setToolTip(f"Go to next item ({self._next_nav_shortcut_hint})")
        self.next_button.clicked.connect(self._navigate_next)

        for button in (
            self.accept_button,
            self.merge_button,
            self.undo_button,
            self.merge_next_button,
            self.prev_button,
            self.next_button,
        ):
            button.setAutoDefault(False)
            button.setDefault(False)

        # ── Layout ────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._comparison_panel)
        splitter.addWidget(self._image_pane)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([700, 500])

        button_row = QHBoxLayout()
        button_row.addWidget(self.prev_button)
        button_row.addWidget(self.accept_button)
        button_row.addWidget(self.merge_button)
        button_row.addWidget(self.undo_button)
        button_row.addStretch(1)
        button_row.addWidget(self.merge_next_button)
        button_row.addWidget(self.next_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.issues_label)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(button_row)

        # ── Initial render ────────────────────────────────────────────────
        self._last_merged_annotations = list(self._comparison_panel.initial_annotations)
        self._update_merge_next_button_presentation()
        self._refresh_button_state()
        self._comparison_panel.update_difference_highlights()
        self._comparison_panel.refresh_widget_item_sizes()

    def _update_merge_next_button_presentation(self) -> None:
        if self.next_button.isEnabled():
            self.merge_next_button.setText("Merge and Next")
            self.merge_next_button.setToolTip(
                f"Apply current annotations and go to next item, even with no local edits ({self._merge_next_shortcut_hint})"
            )
            self.merge_and_next_alt_action.setText("Merge and Next")
        else:
            self.merge_next_button.setText("Merge")
            self.merge_next_button.setToolTip(
                f"Apply current annotations and resolve this fixup ({self._merge_next_shortcut_hint})"
            )
            self.merge_and_next_alt_action.setText("Merge")

    def _refresh_button_state(self) -> None:
        has_acceptable_proposals = self._comparison_panel.has_acceptable_proposals()
        has_local_changes = self._has_local_changes()
        has_dialog_changes = self._has_dialog_state_changes()
        self._update_merge_next_button_presentation()
        self.accept_button.setEnabled(has_acceptable_proposals)
        self.merge_button.setEnabled(has_local_changes)
        self.undo_button.setEnabled(self._undo_available or has_dialog_changes)
        self.merge_next_button.setEnabled(True)
        self._comparison_panel.set_action_buttons_enabled(True)
        self._update_window_title_unsaved_marker(has_local_changes)

    def _update_window_title_unsaved_marker(self, has_unsaved_changes: bool) -> None:
        if not has_unsaved_changes:
            self.setWindowTitle(self._window_title_base)
            return

        # Keep the item-position suffix untouched: "... (x of y)".
        match = re.search(r"\s\(\d+\s+of\s+\d+\)$", self._window_title_base)
        if match is None:
            self.setWindowTitle(f"{self._window_title_base} *")
            return

        title_prefix = self._window_title_base[:match.start()]
        title_suffix = self._window_title_base[match.start():]
        self.setWindowTitle(f"{title_prefix} *{title_suffix}")

    def _has_local_changes(self) -> bool:
        return self._comparison_panel.has_local_changes_compared_to(self._last_merged_annotations)

    def _has_dialog_state_changes(self) -> bool:
        return self._has_local_changes() or self._comparison_panel.has_proposed_changes()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        comparison_table = getattr(self._comparison_panel, "comparison_table", None)
        left_tag_input = getattr(self._comparison_panel, "left_tag_input", None)
        if comparison_table is None or left_tag_input is None:
            return super().eventFilter(watched, event)

        def _is_plain_arrow_modifiers(modifiers: Qt.KeyboardModifier) -> bool:
            # Some macOS keyboards report arrow keys with KeypadModifier.
            return not bool(modifiers & ~Qt.KeyboardModifier.KeypadModifier)

        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            modifiers = event.modifiers()
            allowed_modifiers = Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.KeypadModifier
            has_alt_only = bool(modifiers & Qt.KeyboardModifier.AltModifier) and not bool(modifiers & ~allowed_modifiers)
            if has_alt_only:
                focused = self.focusWidget()
                table_cell_editing = comparison_table.state() == QAbstractItemView.State.EditingState
                editing_focus = isinstance(focused, (QLineEdit, QTextEdit)) and self.isAncestorOf(focused)
                left_tag_input_empty_focus = focused is left_tag_input and not left_tag_input.text().strip()
                if (not editing_focus or left_tag_input_empty_focus) and not table_cell_editing:
                    self._merge_and_next()
                    return True

        if watched in (comparison_table, comparison_table.viewport()) and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_F2 and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                if self._comparison_panel.begin_editing_comparison_row(comparison_table.currentRow()):
                    return True
            if watched is not comparison_table:
                return super().eventFilter(watched, event)
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() == Qt.KeyboardModifier.NoModifier and self._comparison_panel.trigger_action_for_current_row():
                    return True
            if event.key() == Qt.Key.Key_Home and _is_plain_arrow_modifiers(event.modifiers()):
                return self._comparison_panel.activate_first_comparison_row()
            if event.key() == Qt.Key.Key_End and _is_plain_arrow_modifiers(event.modifiers()):
                return self._comparison_panel.activate_last_comparison_row()
            if sys.platform == "darwin" and event.modifiers() == Qt.KeyboardModifier.MetaModifier:
                if event.key() == Qt.Key.Key_Up:
                    return self._comparison_panel.activate_first_comparison_row()
                if event.key() == Qt.Key.Key_Down:
                    return self._comparison_panel.activate_last_comparison_row()

        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Left
            and _is_plain_arrow_modifiers(event.modifiers())
        ):
            # This dialog installs an application-wide event filter so Left can work even when
            # the comparison table doesn't have focus. Ensure we only intercept Left for events
            # originating from within this dialog's widget tree.
            if isinstance(watched, QWidget) and not self.isAncestorOf(watched):
                return super().eventFilter(watched, event)
            if comparison_table.state() == QAbstractItemView.State.EditingState:
                _edit_focused = self.focusWidget()
                if _edit_focused is comparison_table or _edit_focused is comparison_table.viewport():
                    # Platform timing window: the table has entered EditingState (editItem was
                    # called from a deferred QTimer) but keyboard focus has not yet transferred
                    # to the in-cell editor — this is common on macOS where focus transfer is
                    # handled asynchronously by the window server.  If we just return False here
                    # the event reaches the table's own keyPressEvent, which is a no-op for Left
                    # in column 0 and never forwards to the editor, so the cursor never moves.
                    # Instead: locate the visible editor widget, redirect focus to it, then
                    # re-deliver the original event via sendEvent so the very first press works.
                    _editor = comparison_table.viewport().findChild(QTextEdit)
                    if _editor is not None and _editor.isVisible():
                        _editor.setFocus(Qt.FocusReason.OtherFocusReason)
                        QApplication.sendEvent(_editor, event)
                    return True
                return False
            focused = self.focusWidget()
            # When an in-cell editor is being activated, Qt may not yet report EditingState on the
            # table, but focus can already be inside the editor hosted under the table. In that
            # window, never steal Left-arrow cursor movement.
            if (
                focused is not None
                and comparison_table.isAncestorOf(focused)
                and focused is not comparison_table
                and focused is not comparison_table.viewport()
            ):
                return False
            if (
                isinstance(focused, (QLineEdit, QTextEdit))
                and self.isAncestorOf(focused)
            ):
                return False
            if focused is not None and comparison_table.isAncestorOf(focused):
                if isinstance(focused, (QLineEdit, QTextEdit, QAbstractSlider, QAbstractSpinBox, QComboBox)):
                    return False
            if focused is not None and not self.isAncestorOf(focused):
                return False
            if isinstance(focused, (QAbstractSlider, QAbstractSpinBox, QComboBox)):
                return False

            selected_rows = sorted({index.row() for index in comparison_table.selectedIndexes()})
            if self._comparison_panel.apply_proposed_rows_for_selected_rows():
                return True

        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            if not _is_plain_arrow_modifiers(event.modifiers()):
                return super().eventFilter(watched, event)

            step = -1 if event.key() == Qt.Key.Key_Up else 1
            left_input_empty = not left_tag_input.text().strip()

            focused = self.focusWidget()
            if focused is None:
                if self._comparison_panel.activate_adjacent_row_for_last_action(step):
                    return True
                if event.key() == Qt.Key.Key_Down and self._comparison_panel.activate_first_comparison_row():
                    return True
                return super().eventFilter(watched, event)

            if focused is left_tag_input:
                if left_input_empty and self._comparison_panel.activate_adjacent_row_for_last_action(step):
                    return True
                if event.key() == Qt.Key.Key_Down and left_input_empty and self._comparison_panel.activate_first_comparison_row():
                    return True
                return super().eventFilter(watched, event)

            if focused is comparison_table or comparison_table.isAncestorOf(focused):
                return super().eventFilter(watched, event)

            if isinstance(focused, (QLineEdit, QTextEdit, QAbstractItemView, QPushButton, QCheckBox, QSplitter)):
                return super().eventFilter(watched, event)

            if self.isAncestorOf(focused):
                if self._comparison_panel.activate_adjacent_row_for_last_action(step):
                    return True
                if event.key() == Qt.Key.Key_Down and self._comparison_panel.activate_first_comparison_row():
                    return True
                return True

        if event.type() == QEvent.Type.KeyPress:
            is_home_end = event.key() in (Qt.Key.Key_Home, Qt.Key.Key_End) and _is_plain_arrow_modifiers(event.modifiers())
            is_macos_cmd_home_end = (
                sys.platform == "darwin"
                and event.modifiers() == Qt.KeyboardModifier.MetaModifier
                and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down)
            )
            if not is_home_end and not is_macos_cmd_home_end:
                return super().eventFilter(watched, event)

            activate_row = (
                self._comparison_panel.activate_first_comparison_row
                if event.key() in (Qt.Key.Key_Home, Qt.Key.Key_Up)
                else self._comparison_panel.activate_last_comparison_row
            )
            left_input_empty = not left_tag_input.text().strip()

            focused = self.focusWidget()
            if focused is None:
                if activate_row():
                    return True
                return super().eventFilter(watched, event)

            if focused is left_tag_input:
                if left_input_empty and activate_row():
                    return True
                return super().eventFilter(watched, event)

            if focused is comparison_table or comparison_table.isAncestorOf(focused):
                return super().eventFilter(watched, event)

            if isinstance(focused, (QLineEdit, QTextEdit, QAbstractItemView, QPushButton, QCheckBox, QSplitter)):
                return super().eventFilter(watched, event)

            if self.isAncestorOf(focused):
                if activate_row():
                    return True

        return super().eventFilter(watched, event)

    def selected_annotations(self) -> list[str]:
        return self._comparison_panel.selected_annotations()

    def _on_regen_proposed_ready(self, description: str, tags: list[str], exact: bool) -> None:
        self._comparison_panel.set_proposed_annotations(description, tags, exact_match_only_for_tags=exact)
        self._comparison_panel.activate_first_comparison_row()

    def _enter_no_fixups_state_after_delete(self) -> None:
        self._regen_panel.cancel_regeneration(discard_result=True)
        self._image_pane.set_watched_image(None)
        self._image_path = None
        self._comparison_panel.enter_no_fixups_state()
        self._image_pane.clear_for_deleted()
        self.issues_label.setText("No fixup files remaining.")
        self._regen_panel.set_status("Current file was deleted. Press Esc to close.")
        self._regen_panel.set_all_controls_enabled(False)

        for widget in (
            self.accept_button,
            self.merge_button,
            self.undo_button,
            self.merge_next_button,
            self.prev_button,
            self.next_button,
        ):
            widget.setEnabled(False)

    def _on_image_pane_delete_result(self, has_fixups_remaining: bool) -> None:
        if has_fixups_remaining:
            self._navigate_next()
        else:
            self._enter_no_fixups_state_after_delete()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._global_key_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
                self._global_key_filter_installed = True
        self._comparison_panel.update_action_column_metrics()
        self._comparison_panel.update_difference_highlights()
        self._comparison_panel.resize_rows_to_contents()
        self._comparison_panel.refresh_widget_item_sizes()
        self._regen_panel.reposition_overlay()
        QTimer.singleShot(0, self._comparison_panel.apply_initial_table_focus)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._comparison_panel.resize_rows_to_contents()
        self._comparison_panel.refresh_widget_item_sizes()
        self._regen_panel.reposition_overlay()

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.ApplicationFontChange):
            self._comparison_panel.update_action_column_metrics()
            self._comparison_panel.update_difference_highlights()
            self._comparison_panel.resize_rows_to_contents()
            self._comparison_panel.refresh_widget_item_sizes()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._global_key_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._global_key_filter_installed = False
        self._regen_panel.cancel_regeneration(discard_result=True)
        self._image_pane.set_watched_image(None)
        super().closeEvent(event)

    def has_merged_changes(self) -> bool:
        return self._undo_available

    def _resolve_fixup(self) -> bool:
        if self._clear_fixup is None:
            return False
        if not self._clear_fixup():
            return False
        self._resolved = True
        self.undo_button.setEnabled(True)
        self._refresh_button_state()
        return True

    def _restore_initial_state(self) -> None:
        self._comparison_panel.restore_initial_state()

    def _merge_without_close(self) -> bool:
        if self._apply_annotations is None:
            return False
        current_annotations = list(self._comparison_panel.selected_annotations())
        self._apply_annotations(current_annotations, "Fixup merged + auto-saved")
        self._last_merged_annotations = current_annotations
        self._undo_available = True
        return self._resolve_fixup()

    def _accept_all_without_close(self) -> bool:
        self._comparison_panel.accept_all()
        return self._merge_without_close()


    def _undo_merge(self) -> None:
        if not self._undo_available and not self._has_dialog_state_changes():
            return

        if self._undo_available:
            if self._apply_annotations is None:
                return
            if self._restore_fixup is None or not self._restore_fixup():
                return
            self._apply_annotations(list(self._comparison_panel.initial_annotations), "Fixup undone + auto-saved")

        self._comparison_panel.restore_initial_state()
        self._last_merged_annotations = list(self._comparison_panel.initial_annotations)
        self._resolved = False
        self._undo_available = False
        self._regen_panel.set_status("Restored original fixup state.")
        self._refresh_button_state()

    def _navigate_prev(self) -> None:
        self._regen_panel.cancel_regeneration(discard_result=True)
        self.done(self.NAVIGATE_PREV_CODE)

    def _navigate_next(self) -> None:
        self._regen_panel.cancel_regeneration(discard_result=True)
        self.done(self.NAVIGATE_NEXT_CODE)

    def _merge_and_next(self) -> None:
        # Merge+Next must persist the Current column exactly as shown.
        # Proposed rows are only applied when explicitly accepted by the user.
        if self._merge_without_close() and self.next_button.isEnabled():
            self._navigate_next()