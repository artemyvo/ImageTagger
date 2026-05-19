"""ComparisonPanel — the merged-annotations comparison table widget.

Owns the left (current) list, right (proposed) list, comparison table,
tag input field, column header labels, and gesture handler.  The parent
dialog only orchestrates high-level merge operations (save, undo,
navigate); all annotation state and table rendering lives here.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Callable

from PyQt6.QtCore import QEvent, Qt, QSize, QStringListModel, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCompleter,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imagetagger.ui.item_action_widget import ItemActionWidget
from imagetagger.ui.diff_delegates import (
    DIFF_RANGES_ROLE,
    ITEM_TEXT_ROLE,
    IS_SEARCH_MATCH_ROLE,
    ITEM_BADGE_ROLE,
    DiffHighlightDelegate,
    EditableDiffDelegate,
    _danger_button_stylesheet,
    _diff_highlight_colors,
)
from imagetagger.ui.panels.comparison_gesture_handler import ComparisonGestureHandler
from imagetagger.utils.fixup_parser import strip_tag_list_prefix


class ComparisonPanel(QWidget):
    """Widget that owns the comparison table and all annotation state.

    Signals
    -------
    state_changed
        Emitted whenever the annotation data changes (left or right list
        content mutates).  The parent dialog connects this to its own
        button-state refresh.
    selection_changed
        Emitted whenever the table row selection changes.  The parent
        dialog connects this to its own button-state refresh.
    """

    state_changed = pyqtSignal()
    selection_changed = pyqtSignal()

    _PANE_HEADER_BOTTOM_SPACING = 4
    _PANE_HEADER_EXTRA_HEIGHT = 4
    _ROW_WIDGET_SELECTED_STYLE = "background-color: palette(highlight);"
    _ROW_WIDGET_UNSELECTED_STYLE = "background-color: transparent;"
    _TEXT_WIDGET_SELECTED_STYLE = "background-color: palette(highlight); color: palette(highlighted-text);"
    _TEXT_WIDGET_UNSELECTED_STYLE = "background-color: transparent; color: palette(text);"

    def __init__(
        self,
        initial_annotations: list[str],
        initial_description: str,
        initial_tags: list[str],
        initial_search_matches: list[str],
        initial_vision_tags: list[str] | None = None,
        normalize_annotation: Callable[[str], str] | None = None,
        normalize_tag: Callable[[str], str] | None = None,
        is_regenerating: Callable[[], bool] | None = None,
        tag_suggestions: list[str] | None = None,
        merge_table_double_click_action_enabled: bool = True,
        merge_table_swipe_actions_enabled: bool = False,
        merge_table_horizontal_scroll_actions_enabled: bool = False,
        merge_table_horizontal_scroll_reverse_enabled: bool = False,
        merge_table_horizontal_scroll_stop_idle_seconds: float = 0.45,
        merge_table_horizontal_scroll_row_target_mode: int = 3,
        allow_left_delete: bool = True,
        fixup_tag_keys: set[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # ── Injected callables ────────────────────────────────────────────
        self._normalize_annotation = normalize_annotation or (lambda text: text)
        self._normalize_tag = normalize_tag or (lambda text: text)
        self._is_regenerating = is_regenerating or (lambda: False)
        self._allow_left_delete: bool = allow_left_delete
        self._fixup_tag_keys: set[str] | None = fixup_tag_keys

        # ── Initial annotation state ──────────────────────────────────────
        self._initial_annotations: list[str] = [
            tag.strip() for tag in initial_annotations if tag.strip()
        ]
        self._initial_proposed_description: str = initial_description.strip()
        self._initial_proposed_tags: list[tuple[str, str]] = self._build_initial_proposed_tags(
            self._initial_proposed_description,
            [tag.strip() for tag in initial_tags if tag.strip()],
            [tag.strip() for tag in (initial_vision_tags or []) if tag.strip()],
        )
        self._initial_search_matches: list[str] = [
            q.strip() for q in initial_search_matches if q.strip()
        ]

        # ── Deduplication: tags that appear in both proposed and search matches ──
        # Show them once with 🔍 appended to the badge; exclude from standalone matches.
        _proposed_tag_keys: set[str] = {
            self._normalized_compare_key(clean)
            for clean, _ in self._initial_proposed_tags
        }
        _search_match_keys: set[str] = {
            self._normalized_compare_key(q) for q in self._initial_search_matches
        }
        self._initial_proposed_tags_effective: list[tuple[str, str]] = [
            (clean, badge + "🔍" if self._normalized_compare_key(clean) in _search_match_keys else badge)
            for clean, badge in self._initial_proposed_tags
        ]
        self._standalone_search_matches: list[str] = [
            q for q in self._initial_search_matches
            if self._normalized_compare_key(q) not in _proposed_tag_keys
        ]

        # ── Mutable annotation state ──────────────────────────────────────
        self._has_proposed_description: bool = bool(self._initial_proposed_description)
        self._exact_match_only_for_tags: bool = False
        self._protected_existing_keys: set[str] = set()
        if not self._has_proposed_description:
            self._protected_existing_keys = self._find_description_like_keys(
                self._initial_annotations
            )

        # ── Table rendering state ─────────────────────────────────────────
        self._updating_comparison_table: bool = False
        self._pending_edit_refresh: bool = False
        self._last_action_table_row: int | None = None
        self._initial_table_focus_applied: bool = False
        self._action_button_width: int = 24
        self._action_button_height: int = 22
        self._table_row_map: list[tuple[int | None, int | None]] = []
        self._table_row_action_callbacks: dict[int, Callable[[], None]] = {}
        self._table_row_action_symbols: dict[int, str] = {}
        self._active_comparison_editor: QTextEdit | None = None

        # ── Widgets ───────────────────────────────────────────────────────
        self.left_list = QListWidget(self)
        self.left_list.setItemDelegate(DiffHighlightDelegate(self.left_list))
        self.left_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.left_list.setWordWrap(True)
        self.left_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.left_list.setUniformItemSizes(False)
        self.left_list.hide()

        self.right_list = QListWidget(self)
        self.right_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.right_list.setWordWrap(True)
        self.right_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.right_list.setUniformItemSizes(False)
        self.right_list.hide()

        self.comparison_table = QTableWidget(self)
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.comparison_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.comparison_table.setWordWrap(True)
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._comparison_delegate = EditableDiffDelegate(self.comparison_table)
        self._comparison_delegate.editor_created.connect(self._on_comparison_editor_created)
        self._comparison_delegate.closeEditor.connect(self._on_comparison_editor_closed)
        self.comparison_table.setItemDelegateForColumn(0, self._comparison_delegate)
        self.comparison_table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.comparison_table.setMouseTracking(False)
        self.comparison_table.viewport().setMouseTracking(False)
        self.comparison_table.setShowGrid(False)
        self.comparison_table.setGridStyle(Qt.PenStyle.NoPen)
        self.comparison_table.setStyleSheet(self._comparison_table_base_stylesheet())
        self.comparison_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.comparison_table.customContextMenuRequested.connect(
            self._show_comparison_table_context_menu
        )
        self.comparison_table.itemChanged.connect(self._on_comparison_item_changed)
        self.comparison_table.itemSelectionChanged.connect(self._on_comparison_selection_changed)
        header = self.comparison_table.horizontalHeader()
        header.setVisible(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.comparison_left_label = self._create_pane_header_label("Current")
        self.comparison_right_label = self._create_pane_header_label("Proposed")
        self.comparison_header_action_spacer = QWidget(self)
        self.update_action_column_metrics()

        self.left_tag_input = QLineEdit(self)
        self.left_tag_input.setPlaceholderText("Type a tag and press Enter")
        self.left_tag_input.returnPressed.connect(self._add_left_tag_from_input)

        self._tag_suggestions_model = QStringListModel(tag_suggestions or [], self)
        _completer = QCompleter(self._tag_suggestions_model, self)
        _completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        _completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)
        _completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.left_tag_input.setCompleter(_completer)

        # ── Gesture handler ───────────────────────────────────────────────
        self._gesture_handler = ComparisonGestureHandler(
            self.comparison_table,
            swipe_enabled=merge_table_swipe_actions_enabled,
            hscroll_enabled=merge_table_horizontal_scroll_actions_enabled,
            hscroll_reverse=merge_table_horizontal_scroll_reverse_enabled,
            hscroll_stop_idle_seconds=merge_table_horizontal_scroll_stop_idle_seconds,
            hscroll_row_target_mode=merge_table_horizontal_scroll_row_target_mode,
            double_click_enabled=merge_table_double_click_action_enabled,
            on_select_row=self.select_comparison_row,
            on_remove_row=self._remove_value_for_table_row,
            on_apply_row=self._apply_proposed_value_for_table_row,
            on_begin_editing=self.begin_editing_comparison_row,
            on_trigger_row_action=self._trigger_action_for_table_row,
            on_move_row=self._move_table_row,
            parent=self,
        )

        # ── Layout ────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, self._PANE_HEADER_BOTTOM_SPACING)
        header_row.setSpacing(0)
        header_row.addWidget(self.comparison_left_label, stretch=1)
        header_row.addWidget(self.comparison_header_action_spacer, stretch=0)
        header_row.addWidget(self.comparison_right_label, stretch=1)
        layout.addLayout(header_row, stretch=0)
        layout.addWidget(self.comparison_table, stretch=1)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(8)
        input_row.addWidget(self.left_tag_input, stretch=1)
        input_row.addStretch(1)
        layout.addLayout(input_row, stretch=0)

        # ── Populate with initial data ────────────────────────────────────
        for tag in self._initial_annotations:
            self._add_left_item(tag)

        if self._initial_proposed_description:
            self._add_right_item(self._initial_proposed_description)
        for clean, badge in self._initial_proposed_tags_effective:
            self._add_right_item(clean, badge=badge)
        for search_query in self._standalone_search_matches:
            self._add_search_match_item(search_query)

        # Note: update_difference_highlights() is intentionally NOT called here.
        # The parent dialog must call it after it has finished its own __init__
        # (so that is_regenerating is available for action-button rendering).

    # ── Public read-only state ────────────────────────────────────────────

    @property
    def initial_annotations(self) -> list[str]:
        return self._initial_annotations

    @property
    def table_row_map(self) -> list[tuple[int | None, int | None]]:
        return self._table_row_map

    # ── Public API called by the parent dialog ────────────────────────────

    def selected_annotations(self) -> list[str]:
        return self._current_texts(self.left_list)

    def has_acceptable_proposals(self) -> bool:
        current_values = [
            value.strip()
            for value in self._current_texts(self.left_list)
            if value.strip()
        ]
        merged_values = self._merged_values_for_accept_all()
        current_normalized = [self._normalized_compare_text(v) for v in current_values]
        merged_normalized = [self._normalized_compare_text(v) for v in merged_values]
        return current_normalized != merged_normalized

    def has_proposed_changes(self) -> bool:
        current = [
            self._normalized_compare_text(text)
            for text in self._current_texts(self.right_list)
            if self._normalized_compare_text(text)
        ]
        initial_values: list[str] = []
        if self._normalized_compare_text(self._initial_proposed_description):
            initial_values.append(self._normalized_compare_text(self._initial_proposed_description))
        initial_values.extend(
            self._normalized_compare_text(clean)
            for clean, _badge in self._initial_proposed_tags_effective
            if self._normalized_compare_text(clean)
        )
        initial_values.extend(
            self._normalized_compare_text(text)
            for text in self._standalone_search_matches
            if self._normalized_compare_text(text)
        )
        return current != initial_values

    def has_local_changes_compared_to(self, last_merged: list[str]) -> bool:
        """Return True if the current left-list contents differ from *last_merged* after normalization."""
        current = [
            self._normalized_compare_text(text)
            for text in self.selected_annotations()
            if self._normalized_compare_text(text)
        ]
        normalized_merged = [
            self._normalized_compare_text(text)
            for text in last_merged
            if self._normalized_compare_text(text)
        ]
        return current != normalized_merged

    def get_existing_tags_for_regen(self) -> list[str] | None:
        """Return the current short (non-description-like) left-list entries as seed hints."""
        tags = [
            t.strip()
            for t in self._current_texts(self.left_list)
            if t.strip() and not self._is_description_like(t.strip())
        ]
        return tags if tags else None

    def get_current_proposed_for_regen(self) -> tuple[str, list[str], bool]:
        texts = [t.strip() for t in self._current_texts(self.right_list) if t.strip()]
        if self._has_proposed_description and texts:
            return texts[0], texts[1:], self._exact_match_only_for_tags
        return "", texts, self._exact_match_only_for_tags

    def set_proposed_annotations(
        self,
        description: str,
        tags: list[str],
        *,
        exact_match_only_for_tags: bool = False,
    ) -> None:
        self.right_list.clear()
        self._exact_match_only_for_tags = exact_match_only_for_tags
        normalized_description = description.strip()
        normalized_tags = self._drop_description_duplicate_tags(normalized_description, tags)

        self._has_proposed_description = bool(normalized_description)
        if not self._has_proposed_description:
            self._protected_existing_keys = self._find_description_like_keys(
                self._current_texts(self.left_list)
            )
        else:
            self._protected_existing_keys = set()

        if normalized_description:
            self._add_right_item(normalized_description)
        for tag in normalized_tags:
            self._add_right_item(tag)

        self._update_difference_highlights()
        self.state_changed.emit()

    def restore_initial_proposed_annotations(self) -> None:
        self.right_list.clear()
        self._exact_match_only_for_tags = False
        self._has_proposed_description = bool(self._initial_proposed_description)
        if not self._has_proposed_description:
            self._protected_existing_keys = self._find_description_like_keys(
                self._initial_annotations
            )
        else:
            self._protected_existing_keys = set()

        if self._initial_proposed_description:
            self._add_right_item(self._initial_proposed_description)
        for clean, badge in self._initial_proposed_tags_effective:
            self._add_right_item(clean, badge=badge)
        for search_query in self._standalone_search_matches:
            self._add_search_match_item(search_query)

    def restore_initial_state(self) -> None:
        self.left_list.clear()
        for text in self._initial_annotations:
            self._add_left_item(text)
        self.restore_initial_proposed_annotations()
        self._update_difference_highlights()
        self.state_changed.emit()

    def accept_all(self) -> None:
        merged_values = self._merged_values_for_accept_all()
        self.left_list.clear()
        for value in merged_values:
            self._add_left_item(value)
        self._update_difference_highlights()
        self.state_changed.emit()

    def set_action_buttons_enabled(self, enabled: bool) -> None:
        for row in range(self.comparison_table.rowCount()):
            action_host = self.comparison_table.cellWidget(row, 1)
            if action_host is None:
                continue
            for button in action_host.findChildren(QPushButton):
                button.setEnabled(enabled)

    def update_action_column_metrics(self) -> None:
        metrics = self.comparison_table.fontMetrics()
        symbol_width = max(metrics.horizontalAdvance("←"), metrics.horizontalAdvance("✕"))
        self._action_button_width = max(24, symbol_width + 16)
        self._action_button_height = max(22, metrics.height() + 8)

        action_col_width = self._action_button_width + 10
        self.comparison_table.setColumnWidth(1, action_col_width)
        if isinstance(self.comparison_header_action_spacer, QWidget):
            self.comparison_header_action_spacer.setFixedWidth(action_col_width)

    def refresh_widget_item_sizes(self) -> None:
        for i in range(self.right_list.count()):
            item = self.right_list.item(i)
            widget = self.right_list.itemWidget(item)
            if isinstance(widget, ItemActionWidget):
                height = self._estimate_row_height(widget.text, self.right_list, has_button=True)
                item.setSizeHint(QSize(0, height))

        for i in range(self.left_list.count()):
            item = self.left_list.item(i)
            widget = self.left_list.itemWidget(item)
            if isinstance(widget, ItemActionWidget):
                height = self._estimate_row_height(widget.text, self.left_list, has_button=True)
                item.setSizeHint(QSize(0, height))

    def resize_rows_to_contents(self) -> None:
        self.comparison_table.resizeRowsToContents()

    def update_difference_highlights(self) -> None:
        self._update_difference_highlights()

    def apply_initial_table_focus(self) -> None:
        if self._initial_table_focus_applied:
            return
        self._initial_table_focus_applied = True

        row_count = self.comparison_table.rowCount()
        if row_count > 0:
            for row in range(row_count):
                if self._row_needs_addressing(row):
                    self.comparison_table.setCurrentCell(row, 0)
                    self.comparison_table.selectRow(row)
                    break
            else:
                self.comparison_table.setCurrentCell(0, 0)
                self.comparison_table.selectRow(0)

        self.comparison_table.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    # Navigation — called by dialog's QActions and eventFilter

    def activate_previous_actionable_row(self) -> None:
        self._activate_adjacent_actionable_row(-1)

    def activate_next_actionable_row(self) -> None:
        self._activate_adjacent_actionable_row(1)

    def activate_first_comparison_row(self) -> bool:
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return False
        return self.select_comparison_row(0, Qt.FocusReason.ShortcutFocusReason)

    def activate_last_comparison_row(self) -> bool:
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return False
        return self.select_comparison_row(row_count - 1, Qt.FocusReason.ShortcutFocusReason)

    def activate_adjacent_row_for_last_action(self, step: int) -> bool:
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return False

        current_row = self.comparison_table.currentRow()
        if 0 <= current_row < row_count:
            target_row = current_row + step
            if 0 <= target_row < row_count:
                return self.select_comparison_row(target_row, Qt.FocusReason.ShortcutFocusReason)
            return False

        if self._last_action_table_row is None:
            return False

        if step < 0:
            target_row = min(max(0, self._last_action_table_row - 1), row_count - 1)
        else:
            target_row = min(max(0, self._last_action_table_row), row_count - 1)
        return self.select_comparison_row(target_row, Qt.FocusReason.ShortcutFocusReason)

    def select_comparison_row(self, row: int, focus_reason: Qt.FocusReason) -> bool:
        row_count = self.comparison_table.rowCount()
        if row < 0 or row >= row_count:
            return False

        preferred_column = self._preferred_focus_column_for_row(row)
        self.comparison_table.clearSelection()
        self.comparison_table.setCurrentCell(row, preferred_column)
        self.comparison_table.setFocus(focus_reason)
        self.comparison_table.selectRow(row)
        return True

    def begin_editing_comparison_row(self, row: int) -> bool:
        if row < 0 or row >= self.comparison_table.rowCount():
            return False

        editable_item = self.comparison_table.item(row, 0)
        if (
            editable_item is None
            or not bool(editable_item.flags() & Qt.ItemFlag.ItemIsEditable)
        ):
            return False

        self.comparison_table.setCurrentItem(editable_item)
        self.comparison_table.setFocus(Qt.FocusReason.MouseFocusReason)
        self.comparison_table.selectRow(row)

        def _open_editor() -> None:
            current_item = self.comparison_table.item(row, 0)
            if current_item is not editable_item:
                return
            self.comparison_table.editItem(current_item)

        QTimer.singleShot(0, _open_editor)
        return True

    def active_comparison_editor(self) -> QTextEdit | None:
        editor = self._active_comparison_editor
        if editor is None or not editor.isVisible():
            return None
        return editor

    def comparison_editor_owns(self, widget: QWidget | None) -> bool:
        editor = self.active_comparison_editor()
        if editor is None or widget is None:
            return False
        return widget is editor or editor.isAncestorOf(widget)

    def trigger_action_for_current_row(self) -> bool:
        row = self.comparison_table.currentRow()
        return self._trigger_action_for_table_row(row)

    def remember_last_action_table_row(self, row: int) -> None:
        if 0 <= row < len(self._table_row_map):
            self._last_action_table_row = row

    def apply_proposed_rows(self, rows: list[int]) -> None:
        self._apply_proposed_rows(rows)

    def apply_proposed_rows_for_selected_rows(self) -> bool:
        """Apply proposed (right) values for all currently selected actionable rows (Left-arrow action).

        Collects right-column indexes that have a visible '←' action button, applies them,
        updates highlights, emits state_changed, and repositions focus. Returns True if any
        rows were updated.
        """
        selected_rows = sorted({index.row() for index in self.comparison_table.selectedIndexes()})
        right_indexes: list[int] = []
        search_match_right_indexes: list[int] = []
        for row in selected_rows:
            if row < 0 or row >= len(self._table_row_map):
                continue
            _left_index, right_index = self._table_row_map[row]
            if right_index is None:
                continue
            cell_widget = self.comparison_table.cellWidget(row, 1)
            if cell_widget is not None:
                for btn in cell_widget.findChildren(QPushButton):
                    if btn.text() == "←":
                        right_item = self.right_list.item(right_index)
                        if right_item and right_item.data(IS_SEARCH_MATCH_ROLE):
                            search_match_right_indexes.append(right_index)
                        else:
                            right_indexes.append(right_index)
                        break
        if not right_indexes and not search_match_right_indexes:
            return False
        current_row = self.comparison_table.currentRow()
        self._remember_last_action_table_row(current_row)
        if right_indexes:
            self._apply_proposed_rows(right_indexes)
        if search_match_right_indexes:
            existing_keys = {
                self._normalized_compare_key(text)
                for text in self._current_texts(self.left_list)
                if self._normalized_compare_text(text)
            }
            for right_index in sorted(search_match_right_indexes, reverse=True):
                right_item = self.right_list.item(right_index)
                if right_item is None:
                    continue
                raw_text = right_item.data(ITEM_TEXT_ROLE) or right_item.text()
                normalized = self._normalize_tag(raw_text.strip())
                if normalized:
                    key = self._normalized_compare_key(normalized)
                    if key not in existing_keys:
                        self._add_left_item(normalized)
                        existing_keys.add(key)
                self.right_list.takeItem(right_index)
        self._update_difference_highlights()
        self.state_changed.emit()
        new_row_count = self.comparison_table.rowCount()
        if new_row_count > 0 and current_row >= 0:
            target_row = min(current_row, new_row_count - 1)
            self.select_comparison_row(target_row, Qt.FocusReason.ShortcutFocusReason)
        return True

    def focus_left_tag_input(self) -> None:
        self.left_tag_input.clear()
        self.left_tag_input.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def remove_selected_left_items(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.comparison_table.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows:
            return

        min_selected_row = selected_rows[-1]

        left_indexes_to_remove: list[int] = []
        for row in selected_rows:
            if row < 0 or row >= len(self._table_row_map):
                continue
            left_index, _ = self._table_row_map[row]
            if left_index is not None:
                left_indexes_to_remove.append(left_index)

        removed_any = False
        for left_index in sorted(set(left_indexes_to_remove), reverse=True):
            removed = self.left_list.takeItem(left_index)
            del removed
            removed_any = True

        if not removed_any:
            return

        self._update_difference_highlights()
        self.state_changed.emit()

        new_row_count = self.comparison_table.rowCount()
        if new_row_count > 0:
            target_row = min(min_selected_row, new_row_count - 1)
            self.select_comparison_row(target_row, Qt.FocusReason.OtherFocusReason)

    def enter_no_fixups_state(self) -> None:
        """Disable the panel for the 'no fixups remaining' state after a delete."""
        self.left_list.clear()
        self.right_list.clear()
        self._table_row_map = []
        self.comparison_table.clearContents()
        self.comparison_table.setRowCount(0)
        self.comparison_table.setEnabled(False)
        self.left_tag_input.clear()
        self.left_tag_input.setEnabled(False)

    def append_unique_to_left(self, values: list[str]) -> None:
        existing = {
            self._normalized_compare_key(value)
            for value in self._current_texts(self.left_list)
            if self._normalized_compare_text(value)
        }
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            key = self._normalized_compare_key(normalized)
            if key in existing:
                continue
            existing.add(key)
            self._add_left_item(normalized)

    # ── Private helpers ───────────────────────────────────────────────────

    def _create_pane_header_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setContentsMargins(0, 0, 0, 0)
        label.setMinimumHeight(label.fontMetrics().height() + self._PANE_HEADER_EXTRA_HEIGHT)
        return label

    def _add_left_item(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        item = QListWidgetItem(normalized)
        item.setData(DIFF_RANGES_ROLE, [])
        self.left_list.addItem(item)

    def _add_left_tag_from_input(self) -> None:
        new_tag = self._normalize_tag(self.left_tag_input.text())
        if not new_tag:
            return

        existing_tags = [text.strip() for text in self._current_texts(self.left_list)]
        normalized_new_key = self._normalized_compare_key(new_tag)
        existing_keys = {
            self._normalized_compare_key(text)
            for text in existing_tags
            if self._normalized_compare_text(text)
        }
        if normalized_new_key in existing_keys:
            self._select_existing_tag_row(normalized_new_key)
            self.left_tag_input.clear()
            QTimer.singleShot(0, self.left_tag_input.clear)
            QTimer.singleShot(0, self.left_tag_input.setFocus)
            return

        self._add_left_item(new_tag)
        self.left_tag_input.clear()
        QTimer.singleShot(0, self.left_tag_input.clear)
        QTimer.singleShot(0, self.left_tag_input.setFocus)
        self._update_difference_highlights()
        self.state_changed.emit()

    def _add_search_match_to_tags(self, search_query: str) -> None:
        normalized = self._normalize_tag(search_query)
        if not normalized:
            return

        current_tags = self._current_texts(self.left_list)
        normalized_key = self._normalized_compare_key(normalized)
        existing_keys = {self._normalized_compare_key(tag) for tag in current_tags}

        if normalized_key not in existing_keys:
            self._add_left_item(normalized)

        for i in range(self.right_list.count() - 1, -1, -1):
            item = self.right_list.item(i)
            if item and item.data(IS_SEARCH_MATCH_ROLE):
                item_text = item.text().strip()
                if self._normalized_compare_key(item_text) == normalized_key:
                    self.right_list.takeItem(i)
                    break

        self._update_difference_highlights()
        self.state_changed.emit()

    def _select_existing_tag_row(self, normalized_key: str) -> None:
        if not normalized_key:
            return

        target_left_index: int | None = None
        for index, text in enumerate(self._current_texts(self.left_list)):
            if self._normalized_compare_key(text) == normalized_key:
                target_left_index = index
                break

        if target_left_index is None:
            return

        target_row: int | None = None
        for row, (left_index, _right_index) in enumerate(self._table_row_map):
            if left_index == target_left_index:
                target_row = row
                break

        if target_row is None:
            return

        self.comparison_table.clearSelection()
        self.comparison_table.selectRow(target_row)
        self.comparison_table.setCurrentCell(target_row, 0)
        model_index = self.comparison_table.model().index(target_row, 0)
        self.comparison_table.scrollTo(
            model_index,
            QAbstractItemView.ScrollHint.PositionAtCenter,
        )

    def _add_right_item(self, text: str, *, badge: str = "") -> None:
        normalized = text.strip()
        if not normalized:
            return
        item = QListWidgetItem(normalized)
        item.setData(DIFF_RANGES_ROLE, [])
        item.setData(ITEM_TEXT_ROLE, normalized)
        if badge:
            item.setData(ITEM_BADGE_ROLE, badge)
        self.right_list.addItem(item)

    def _add_search_match_item(self, search_query: str) -> None:
        normalized = search_query.strip()
        if not normalized:
            return
        item = QListWidgetItem(normalized)
        item.setData(DIFF_RANGES_ROLE, [])
        item.setData(ITEM_TEXT_ROLE, normalized)
        item.setData(IS_SEARCH_MATCH_ROLE, True)
        self.right_list.addItem(item)

    def _current_texts(self, list_widget: QListWidget) -> list[str]:
        texts = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            widget = list_widget.itemWidget(item)
            if widget and hasattr(widget, "text"):
                # Prefer clean_value when set (right-list badged items)
                if hasattr(widget, "clean_value") and widget.clean_value is not None:
                    texts.append(widget.clean_value)
                else:
                    texts.append(widget.text)
            else:
                item_text = item.text().strip()
                if not item_text:
                    stored = item.data(ITEM_TEXT_ROLE)
                    if isinstance(stored, str):
                        item_text = stored.strip()
                texts.append(item_text)
        return texts

    def _normalized_compare_text(self, text: str) -> str:
        return self._normalize_annotation(text).strip()

    def _normalized_compare_key(self, text: str) -> str:
        return self._normalized_compare_text(text).casefold()

    @staticmethod
    def _is_description_like(text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        word_count = len(normalized.split())
        return word_count >= 5 or len(normalized) >= 40

    def _find_description_like_keys(self, values: list[str]) -> set[str]:
        candidates = [
            self._normalized_compare_text(value)
            for value in values
            if self._is_description_like(value)
        ]
        candidates = [value for value in candidates if value]
        if not candidates:
            return set()
        longest = max(candidates, key=len)
        return {self._normalized_compare_key(longest)}

    def _find_description_like_index(self, values: list[str]) -> int | None:
        best_index: int | None = None
        best_length = -1
        for index, value in enumerate(values):
            normalized = self._normalized_compare_text(value)
            if not self._is_description_like(normalized):
                continue
            if len(normalized) > best_length:
                best_length = len(normalized)
                best_index = index
        return best_index

    def _drop_description_duplicate_tags(self, description: str, tags: list[str]) -> list[str]:
        description_key = self._normalized_compare_key(description)
        if not description_key:
            return [tag for tag in tags if tag.strip()]
        filtered_tags: list[str] = []
        for tag in tags:
            normalized = tag.strip()
            if not normalized:
                continue
            if self._normalized_compare_key(normalized) == description_key:
                continue
            filtered_tags.append(normalized)
        return filtered_tags

    def _is_protected_existing_text(self, text: str) -> bool:
        if self._has_proposed_description:
            return False
        return self._normalized_compare_key(text) in self._protected_existing_keys

    def _is_fixup_deletion_candidate(self, text: str) -> bool:
        """Return True when fixup_tags is present and this left-side text was not validated."""
        if self._fixup_tag_keys is None:
            return False
        if not self._allow_left_delete:
            return False
        if self._is_description_like(text):
            return False
        return self._normalized_compare_key(text) not in self._fixup_tag_keys

    @staticmethod
    def _strip_tag_list_prefix(tag: str) -> str:
        return strip_tag_list_prefix(tag)

    def _build_initial_proposed_tags(
        self,
        description: str,
        tags: list[str],
        vision_tags: list[str],
    ) -> list[tuple[str, str]]:
        """Merge TAGS and VISIONTAGS into a deduped list of (clean_value, badge) pairs."""
        show_badges = bool(vision_tags)

        normal_keys = {self._normalized_compare_key(t) for t in tags}
        vision_keys = {self._normalized_compare_key(t) for t in vision_tags}

        # Build merged unique list: TAGS-order first, then VISIONTAGS-only additions
        merged_clean: list[str] = []
        seen_keys: set[str] = set()
        for t in tags:
            key = self._normalized_compare_key(t)
            if key not in seen_keys:
                seen_keys.add(key)
                merged_clean.append(t)
        for t in vision_tags:
            key = self._normalized_compare_key(t)
            if key not in seen_keys:
                seen_keys.add(key)
                merged_clean.append(t)

        filtered_clean = self._drop_description_duplicate_tags(description, merged_clean)

        result: list[tuple[str, str]] = []
        for t in filtered_clean:
            if not show_badges:
                result.append((t, ""))
                continue
            key = self._normalized_compare_key(t)
            in_normal = key in normal_keys
            in_vision = key in vision_keys
            if in_normal and in_vision:
                badge = "⚖️✨"
            elif in_vision:
                badge = "✨"
            else:
                badge = "⚖️"
            result.append((t, badge))
        return result

    def _normalize_proposed_text_for_merge(self, text: str, right_row: int) -> str:
        normalized = text.strip()
        if self._has_proposed_description and right_row == 0:
            return normalized
        return self._normalize_tag(self._strip_tag_list_prefix(normalized))

    def _apply_proposed_rows(self, rows: list[int]) -> None:
        if not rows:
            return

        left_texts = self._current_texts(self.left_list)
        right_texts = self._current_texts(self.right_list)
        _, right_matches, _ = self._compute_matches(left_texts, right_texts)

        existing_keys = {
            self._normalized_compare_key(text)
            for text in left_texts
            if self._normalized_compare_text(text)
        }

        for right_row in rows:
            if right_row < 0 or right_row >= len(right_texts):
                continue

            proposed_text = self._normalize_proposed_text_for_merge(right_texts[right_row], right_row)
            if not proposed_text:
                continue

            key = self._normalized_compare_key(proposed_text)

            matched_left_row = right_matches.get(right_row)
            if matched_left_row is not None and 0 <= matched_left_row < self.left_list.count():
                left_item = self.left_list.item(matched_left_row)
                if self.left_list.itemWidget(left_item) is not None:
                    self.left_list.setItemWidget(left_item, None)
                left_item.setText(proposed_text)
                left_item.setData(ITEM_TEXT_ROLE, proposed_text)
                existing_keys.add(key)
                continue

            if self._has_proposed_description and right_row == 0:
                replacement_index = self._find_description_like_index(left_texts)
                if replacement_index is None:
                    for left_index, left_text in enumerate(left_texts):
                        if self._normalized_compare_key(left_text) == key:
                            replacement_index = left_index
                            break

                if replacement_index is not None and 0 <= replacement_index < self.left_list.count():
                    replacement_item = self.left_list.item(replacement_index)
                    if self.left_list.itemWidget(replacement_item) is not None:
                        self.left_list.setItemWidget(replacement_item, None)
                    replacement_item.setText(proposed_text)
                    replacement_item.setData(ITEM_TEXT_ROLE, proposed_text)
                    existing_keys.add(key)
                    continue

            if key not in existing_keys:
                if self._has_proposed_description and right_row == 0:
                    existing_desc_index = self._find_description_like_index(left_texts)
                    if existing_desc_index is not None and 0 <= existing_desc_index < self.left_list.count():
                        desc_item = self.left_list.item(existing_desc_index)
                        if self.left_list.itemWidget(desc_item) is not None:
                            self.left_list.setItemWidget(desc_item, None)
                        desc_item.setText(proposed_text)
                        desc_item.setData(ITEM_TEXT_ROLE, proposed_text)
                    else:
                        item = QListWidgetItem(proposed_text)
                        item.setData(DIFF_RANGES_ROLE, [])
                        self.left_list.insertItem(0, item)
                else:
                    self._add_left_item(proposed_text)
                existing_keys.add(key)

    def _remember_last_action_table_row(self, row: int) -> None:
        if 0 <= row < len(self._table_row_map):
            self._last_action_table_row = row

    def _preferred_focus_column_for_row(self, row: int) -> int:
        left_item = self.comparison_table.item(row, 0)
        if left_item is not None and left_item.text().strip():
            return 0
        right_widget = self.comparison_table.cellWidget(row, 2)
        if right_widget is not None:
            return 2
        return 0

    def _row_needs_addressing(self, row: int) -> bool:
        if row < 0 or row >= self.comparison_table.rowCount():
            return False
        action_host = self.comparison_table.cellWidget(row, 1)
        if action_host is None:
            return False
        for btn in action_host.findChildren(QPushButton):
            if btn.text() in ("←", "✕"):
                return True
        return False

    def _activate_adjacent_actionable_row(self, step: int) -> bool:
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return False

        current_row = self.comparison_table.currentRow()
        if current_row < 0:
            current_row = -1 if step > 0 else row_count

        row = current_row + step
        while 0 <= row < row_count:
            if self._row_needs_addressing(row):
                return self.select_comparison_row(row, Qt.FocusReason.ShortcutFocusReason)
            row += step

        return False

    def _advance_to_next_actionable_from(self, start_row: int) -> None:
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return
        for row in range(start_row, row_count):
            if self._row_needs_addressing(row):
                self.select_comparison_row(row, Qt.FocusReason.ShortcutFocusReason)
                return
        self.select_comparison_row(
            min(start_row, row_count - 1), Qt.FocusReason.ShortcutFocusReason
        )

    def _action_button_for_row(self, row: int) -> QPushButton | None:
        if row < 0 or row >= self.comparison_table.rowCount():
            return None
        action_host = self.comparison_table.cellWidget(row, 1)
        if action_host is None:
            return None
        for button in action_host.findChildren(QPushButton):
            if button.isEnabled():
                return button
        return None

    @staticmethod
    def _action_context_menu_label(symbol: str) -> str:
        if symbol == "←":
            return "Apply Proposed Value"
        if symbol == "✕":
            return "Delete Current Value"
        if symbol == "🔍":
            return "Add Suggested Tag"
        return "Run Row Action"

    def _remove_value_for_table_row(self, row: int) -> bool:
        if row < 0 or row >= len(self._table_row_map):
            return False
        left_index, _right_index = self._table_row_map[row]
        if left_index is None:
            return False
        left_item = self.comparison_table.item(row, 0)
        if left_item is None:
            return False
        left_text = left_item.text().strip()
        if not left_text:
            return False
        before_count = self.left_list.count()
        self._remove_left_item_from_table(row, left_text)
        return self.left_list.count() < before_count

    def _apply_proposed_value_for_table_row(self, row: int) -> bool:
        if row < 0 or row >= len(self._table_row_map):
            return False
        _left_index, right_index = self._table_row_map[row]
        if right_index is None:
            return False
        button = self._action_button_for_row(row)
        if button is not None and button.text() in ("←", "🔍"):
            return self._trigger_action_for_table_row(row)
        self._remember_last_action_table_row(row)
        self._apply_proposed_rows([right_index])
        self._update_difference_highlights()
        self._advance_to_next_actionable_from(row + 1)
        self.state_changed.emit()
        return True

    def _trigger_action_for_table_row(self, row: int) -> bool:
        callback = self._table_row_action_callbacks.get(row)
        action_symbol = self._table_row_action_symbols.get(row, "")
        if callback is None:
            return False
        removes_row = action_symbol == "✕"
        callback()
        advance_from = row if removes_row else row + 1
        self._advance_to_next_actionable_from(advance_from)
        return True

    def _show_comparison_table_context_menu(self, position) -> None:
        row = self.comparison_table.rowAt(position.y())
        if row < 0:
            return

        self.select_comparison_row(row, Qt.FocusReason.MouseFocusReason)

        button = self._action_button_for_row(row)
        if button is None:
            return

        menu = QMenu(self)
        label = self._action_context_menu_label(button.text())
        row_action = menu.addAction(label)
        row_action.triggered.connect(
            lambda _checked=False, table_row=row: self._trigger_action_for_table_row(table_row)
        )
        menu.exec(self.comparison_table.viewport().mapToGlobal(position))

    def _merge_proposed_row_from_table(self, table_row: int, right_index: int) -> None:
        self._remember_last_action_table_row(table_row)
        self._apply_proposed_rows([right_index])
        self._update_difference_highlights()
        self.state_changed.emit()

    def _remove_left_item_from_table(self, table_row: int, text: str) -> None:
        self._remember_last_action_table_row(table_row)
        self._remove_left_item(text)

    def _remove_right_rows(self, rows: list[int]) -> None:
        for row in sorted(rows, reverse=True):
            item = self.right_list.takeItem(row)
            del item

    def _merged_values_for_accept_all(self) -> list[str]:
        proposed_values: list[str] = []
        for index, value in enumerate(self._current_texts(self.right_list)):
            if not value.strip():
                continue
            normalized = self._normalize_proposed_text_for_merge(value, index)
            if normalized:
                proposed_values.append(normalized)

        if self._fixup_tag_keys is None:
            # Vision-only scenario (no validation fixup): preserve all existing left-side
            # items and append proposed ones — do not replace.
            preserved_values = [
                value.strip()
                for value in self._current_texts(self.left_list)
                if value.strip()
            ]
        else:
            # Validation scenario: only keep protected items (e.g. description);
            # proposed tags from the fixup replace the existing tag set.
            preserved_values = [
                value.strip()
                for value in self._current_texts(self.left_list)
                if value.strip() and self._is_protected_existing_text(value)
            ]

        merged_values: list[str] = []
        seen: set[str] = set()
        for value in preserved_values + proposed_values:
            key = self._normalized_compare_key(value)
            if key in seen:
                continue
            seen.add(key)
            merged_values.append(value)
        return merged_values

    def _compute_matches(
        self,
        left_texts: list[str],
        right_texts: list[str],
        search_match_indexes: set[int] | None = None,
    ) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
        if search_match_indexes is None:
            search_match_indexes = set()

        left_matches: dict[int, int] = {}
        right_matches: dict[int, int] = {}
        right_match_kind: dict[int, str] = {}

        right_by_key: dict[str, list[int]] = {}
        for right_index, text in enumerate(right_texts):
            if right_index in search_match_indexes:
                continue
            right_by_key.setdefault(self._normalized_compare_key(text), []).append(right_index)

        protected_left_indexes: set[int] = {
            index
            for index, text in enumerate(left_texts)
            if self._is_protected_existing_text(text)
        }

        if self._has_proposed_description and right_texts:
            left_description_index = self._find_description_like_index(left_texts)
            if left_description_index is not None and left_description_index not in protected_left_indexes:
                description_right_index = 0
                if description_right_index not in search_match_indexes:
                    left_matches[left_description_index] = 0
                    right_matches[0] = left_description_index
                    left_description_norm = self._normalized_compare_text(
                        left_texts[left_description_index]
                    )
                    right_description_norm = self._normalized_compare_text(right_texts[0])
                    right_match_kind[0] = (
                        "exact"
                        if left_description_norm == right_description_norm
                        else "description"
                    )

        # Pass 1: exact matches
        for left_index, text in enumerate(left_texts):
            if left_index in protected_left_indexes:
                continue
            if left_index in left_matches:
                continue
            key = self._normalized_compare_key(text)
            candidates = right_by_key.get(key, [])
            while candidates and candidates[0] in right_matches:
                candidates.pop(0)
            if not candidates:
                continue
            right_index = candidates.pop(0)
            left_matches[left_index] = right_index
            right_matches[right_index] = left_index
            right_match_kind[right_index] = "exact"

        # Pass 2: fuzzy matches
        if self._exact_match_only_for_tags:
            return (left_matches, right_matches, right_match_kind)

        for left_index, left_text in enumerate(left_texts):
            if left_index in protected_left_indexes:
                continue
            if left_index in left_matches:
                continue
            best_right = -1
            best_ratio = 0.0
            for right_index, right_text in enumerate(right_texts):
                if right_index in search_match_indexes:
                    continue
                if right_index in right_matches:
                    continue
                ratio = SequenceMatcher(
                    None,
                    self._normalized_compare_text(left_text).casefold(),
                    self._normalized_compare_text(right_text).casefold(),
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_right = right_index

            if best_right >= 0 and best_ratio >= 0.75:
                left_matches[left_index] = best_right
                right_matches[best_right] = left_index
                right_match_kind[best_right] = "fuzzy"

        return (left_matches, right_matches, right_match_kind)

    def _update_action_widgets(
        self,
        left_texts: list[str],
        right_texts: list[str],
        left_matches: dict[int, int],
        right_match_kind: dict[int, str],
    ) -> None:
        for i in range(self.left_list.count()):
            item = self.left_list.item(i)
            item_text = left_texts[i] if i < len(left_texts) else item.text().strip()
            existing_widget = self.left_list.itemWidget(item)
            right_index = left_matches.get(i)
            is_fuzzy_match = right_index is not None and right_match_kind.get(right_index) == "fuzzy"
            should_show_x = self._allow_left_delete and (
                right_index is None
                or (not is_fuzzy_match and self._is_fixup_deletion_candidate(left_texts[i]))
            )

            if should_show_x:
                if not isinstance(existing_widget, ItemActionWidget):
                    def make_remove_callback(text: str) -> Callable[[], None]:
                        return lambda: self._remove_left_item(text)

                    widget = ItemActionWidget(
                        item_text, "✕", make_remove_callback(item_text), [], False
                    )
                    item.setData(ITEM_TEXT_ROLE, item_text)
                    item.setText("")
                    self.left_list.setItemWidget(item, widget)
            else:
                if existing_widget is not None:
                    restored = item.data(ITEM_TEXT_ROLE)
                    if isinstance(restored, str) and restored.strip():
                        item.setText(restored)
                    self.left_list.setItemWidget(item, None)

        for i in range(self.right_list.count()):
            item = self.right_list.item(i)
            text = right_texts[i] if i < len(right_texts) else ""
            ranges = item.data(DIFF_RANGES_ROLE)
            existing_widget = self.right_list.itemWidget(item)  # noqa: F841 (kept for parity)

            def make_accept_callback(value: str) -> Callable[[], None]:
                return lambda: self._move_item_from_right_to_left(value)

            show_arrow = right_match_kind.get(i) != "exact"
            button_text = "←" if show_arrow else ""
            callback = make_accept_callback(text) if show_arrow else None
            badge = item.data(ITEM_BADGE_ROLE) or ""
            widget = ItemActionWidget(
                text,
                button_text,
                callback,
                ranges if isinstance(ranges, list) else [],
                button_on_left=True,
                clean_value=text if badge else None,
                badge=badge,
            )
            item.setData(ITEM_TEXT_ROLE, text)
            item.setText("")
            self.right_list.setItemWidget(item, widget)

    def _estimate_row_height(self, text: str, list_widget: QListWidget, has_button: bool) -> int:
        viewport_width = max(80, list_widget.viewport().width())
        button_space = 40 if has_button else 0
        text_width = max(40, viewport_width - button_space - 12)
        metrics = list_widget.fontMetrics()
        rect = metrics.boundingRect(
            0,
            0,
            text_width,
            10000,
            int(Qt.TextFlag.TextWordWrap),
            text,
        )
        return max(24, rect.height() + 8)

    def _update_difference_highlights(self) -> None:
        left_items = [self.left_list.item(i) for i in range(self.left_list.count())]
        right_items = [self.right_list.item(i) for i in range(self.right_list.count())]

        left_texts = []
        for item in left_items:
            widget = self.left_list.itemWidget(item)
            if widget and hasattr(widget, "text"):
                left_texts.append(widget.text)
            else:
                left_texts.append(item.text())

        right_texts = []
        for item in right_items:
            widget = self.right_list.itemWidget(item)
            if widget and hasattr(widget, "text"):
                right_texts.append(widget.text)
            else:
                right_texts.append(item.text())

        search_match_indexes: set[int] = set()
        for right_index, item in enumerate(right_items):
            if item and item.data(IS_SEARCH_MATCH_ROLE):
                search_match_indexes.add(right_index)

        left_matches, right_matches, right_match_kind = self._compute_matches(
            left_texts, right_texts, search_match_indexes
        )

        def ranges_for_diff(
            left_text: str, right_text: str
        ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
            left_ranges: list[tuple[int, int]] = []
            right_ranges: list[tuple[int, int]] = []
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, left_text, right_text).get_opcodes():
                if tag == "equal":
                    continue
                if i2 > i1:
                    left_ranges.append((i1, i2))
                if j2 > j1:
                    right_ranges.append((j1, j2))
            return (left_ranges, right_ranges)

        for left_index, item in enumerate(left_items):
            ranges: list[tuple[int, int]] = []
            if left_index in left_matches:
                right_index = left_matches[left_index]
                ranges, _ = ranges_for_diff(left_texts[left_index], right_texts[right_index])
            else:
                if self._is_protected_existing_text(left_texts[left_index]):
                    ranges = []
                else:
                    ranges = []
            item.setData(DIFF_RANGES_ROLE, ranges)

        for right_index, item in enumerate(right_items):
            ranges = []
            if right_index in right_matches:
                left_index = right_matches[right_index]
                _, ranges = ranges_for_diff(left_texts[left_index], right_texts[right_index])
            else:
                ranges = []
            item.setData(DIFF_RANGES_ROLE, ranges)

        self._render_comparison_table(
            left_texts, right_texts, left_matches, right_matches, right_match_kind
        )

    def _build_highlighted_html(self, text: str, ranges: list[tuple[int, int]]) -> str:
        if not text:
            return ""
        if not ranges:
            return ItemActionWidget._escape_html(text)

        highlight_bg, highlight_text = _diff_highlight_colors(self.palette())
        highlight_bg_css = highlight_bg.name(QColor.NameFormat.HexRgb)
        highlight_text_css = highlight_text.name(QColor.NameFormat.HexRgb)
        html_parts: list[str] = []
        last_end = 0
        for start, end in sorted(ranges):
            if start > last_end:
                html_parts.append(ItemActionWidget._escape_html(text[last_end:start]))
            html_parts.append(
                f'<span style="background-color: {highlight_bg_css}; color: {highlight_text_css};">'
                f"{ItemActionWidget._escape_html(text[start:end])}</span>"
            )
            last_end = end
        if last_end < len(text):
            html_parts.append(ItemActionWidget._escape_html(text[last_end:]))
        return "".join(html_parts)

    def _make_text_cell_label(self, text: str, ranges: list[tuple[int, int]]) -> QLabel:
        label = QLabel(self)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setContentsMargins(0, 0, 0, 0)
        label.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        highlighted_html = self._build_highlighted_html(text, ranges)
        plain_html = ItemActionWidget._escape_html(text)
        label.setProperty("_highlighted_html", highlighted_html)
        label.setProperty("_plain_html", plain_html)
        label.setText(highlighted_html)
        return label

    def _make_badge_cell_widget(self, text_label: QLabel, badge: str) -> QWidget:
        container = QWidget(self.comparison_table)
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(4)
        layout.addWidget(text_label, stretch=1)
        badge_label = QLabel(badge, container)
        badge_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        badge_label.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout.addWidget(badge_label, stretch=0)
        container._text_label = text_label  # type: ignore[attr-defined]
        container._badge_label = badge_label  # type: ignore[attr-defined]
        return container

    @staticmethod
    def _comparison_table_base_stylesheet() -> str:
        return (
            "QTableWidget {"
            "  border: none;"
            "  outline: 0;"
            "  gridline-color: transparent;"
            "  selection-background-color: palette(highlight);"
            "  selection-color: palette(highlighted-text);"
            "}"
            "QTableWidget::item {"
            "  border: none;"
            "  margin: 0px;"
            "  padding: 0px;"
            "}"
            "QTableWidget::item:selected {"
            "  border: none;"
            "  outline: 0;"
            "}"
            "QTableWidget::item:focus {"
            "  border: none;"
            "  outline: 0;"
            "}"
            "QTableWidget::item:hover { background-color: transparent; }"
        )

    def _apply_comparison_row_widget_styles(
        self,
        row_selected: bool,
        action_widget: QWidget | None,
        left_widget: QWidget | None,
        right_widget: QWidget | None,
    ) -> None:
        if action_widget is not None:
            action_widget.setStyleSheet(
                self._ROW_WIDGET_SELECTED_STYLE if row_selected else self._ROW_WIDGET_UNSELECTED_STYLE
            )
        if isinstance(left_widget, QLabel):
            left_widget.setStyleSheet(
                self._TEXT_WIDGET_SELECTED_STYLE if row_selected else self._TEXT_WIDGET_UNSELECTED_STYLE
            )
        # right_widget may be a plain QLabel or a badge container QWidget
        text_label: QLabel | None = getattr(right_widget, "_text_label", None)
        badge_label: QLabel | None = getattr(right_widget, "_badge_label", None)
        if isinstance(text_label, QLabel):
            # Badge container path
            plain_html = text_label.property("_plain_html")
            highlighted_html = text_label.property("_highlighted_html")
            if isinstance(plain_html, str) and isinstance(highlighted_html, str):
                text_label.setText(plain_html if row_selected else highlighted_html)
            text_label.setStyleSheet(
                self._TEXT_WIDGET_SELECTED_STYLE if row_selected else self._TEXT_WIDGET_UNSELECTED_STYLE
            )
            if isinstance(badge_label, QLabel):
                badge_label.setStyleSheet(
                    self._TEXT_WIDGET_SELECTED_STYLE if row_selected else self._TEXT_WIDGET_UNSELECTED_STYLE
                )
            if right_widget is not None:
                right_widget.setStyleSheet(
                    self._ROW_WIDGET_SELECTED_STYLE if row_selected else self._ROW_WIDGET_UNSELECTED_STYLE
                )
        elif isinstance(right_widget, QLabel):
            plain_html = right_widget.property("_plain_html")
            highlighted_html = right_widget.property("_highlighted_html")
            if isinstance(plain_html, str) and isinstance(highlighted_html, str):
                right_widget.setText(plain_html if row_selected else highlighted_html)
            right_widget.setStyleSheet(
                self._TEXT_WIDGET_SELECTED_STYLE if row_selected else self._TEXT_WIDGET_UNSELECTED_STYLE
            )

    def _set_action_cell(
        self,
        row: int,
        action_text: str,
        callback: Callable[[], None] | None,
    ) -> None:
        if action_text and callback is not None:
            self._table_row_action_callbacks[row] = callback
            self._table_row_action_symbols[row] = action_text
        else:
            self._table_row_action_callbacks.pop(row, None)
            self._table_row_action_symbols.pop(row, None)

        action_host = QWidget(self.comparison_table)
        action_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        action_layout = QHBoxLayout(action_host)
        action_layout.setContentsMargins(0, 0, 0, 0)
        if action_text == "✕":
            action_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        elif action_text == "←":
            action_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        else:
            action_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if action_text and callback is not None:
            button = QPushButton(action_text, action_host)
            button.setFixedWidth(self._action_button_width)
            button.setFixedHeight(self._action_button_height)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setEnabled(not self._is_regenerating())
            if action_text == "✕":
                button.setStyleSheet(_danger_button_stylesheet(button.palette()))
            button.clicked.connect(lambda _checked=False, cb=callback: cb())
            action_layout.addWidget(button)
        self.comparison_table.setCellWidget(row, 1, action_host)

    def _on_comparison_selection_changed(self) -> None:
        self._sync_comparison_widget_selection_styles()
        self.selection_changed.emit()

    def _sync_comparison_widget_selection_styles(self) -> None:
        selected_rows = {index.row() for index in self.comparison_table.selectedIndexes()}
        for row in range(self.comparison_table.rowCount()):
            row_selected = row in selected_rows
            action_widget = self.comparison_table.cellWidget(row, 1)
            left_widget = self.comparison_table.cellWidget(row, 0)
            right_widget = self.comparison_table.cellWidget(row, 2)
            self._apply_comparison_row_widget_styles(
                row_selected, action_widget, left_widget, right_widget
            )

    def _render_comparison_table(
        self,
        left_texts: list[str],
        right_texts: list[str],
        left_matches: dict[int, int],
        right_matches: dict[int, int],
        right_match_kind: dict[int, str],
    ) -> None:
        scrollbar = self.comparison_table.verticalScrollBar()
        saved_scroll = scrollbar.value()

        self._updating_comparison_table = True
        try:
            self.comparison_table.setRowCount(0)
            self._table_row_map = []
            self._table_row_action_callbacks.clear()
            self._table_row_action_symbols.clear()

            consumed_left_indexes: set[int] = set()
            consumed_right_indexes: set[int] = set()

            if self._has_proposed_description and right_texts:
                description_left_index = right_matches.get(0)
                self._table_row_map.append((description_left_index, 0))
                consumed_right_indexes.add(0)
                if description_left_index is not None:
                    consumed_left_indexes.add(description_left_index)

            for left_index, left_text in enumerate(left_texts):
                if left_index in consumed_left_indexes:
                    continue
                right_index = left_matches.get(left_index)
                self._table_row_map.append((left_index, right_index))
                consumed_left_indexes.add(left_index)
                if right_index is not None:
                    consumed_right_indexes.add(right_index)

            for right_index in range(len(right_texts)):
                if right_index in consumed_right_indexes or right_index in right_matches:
                    continue
                self._table_row_map.append((None, right_index))

            self.comparison_table.setRowCount(len(self._table_row_map))

            for row, (left_index, right_index) in enumerate(self._table_row_map):
                if left_index is not None:
                    left_item = self.left_list.item(left_index)
                    left_text = left_texts[left_index]
                    left_ranges = left_item.data(DIFF_RANGES_ROLE)
                    table_item = QTableWidgetItem(left_text)
                    flags = (
                        Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsEditable
                    )
                    table_item.setFlags(flags)
                    table_item.setData(
                        DIFF_RANGES_ROLE,
                        left_ranges if isinstance(left_ranges, list) else [],
                    )
                    self.comparison_table.setItem(row, 0, table_item)
                else:
                    placeholder_item = QTableWidgetItem("")
                    placeholder_item.setFlags(
                        Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                    )
                    placeholder_item.setData(DIFF_RANGES_ROLE, [])
                    self.comparison_table.setItem(row, 0, placeholder_item)

                if right_index is not None:
                    right_item = self.right_list.item(right_index)
                    right_text = right_texts[right_index]
                    right_ranges = right_item.data(DIFF_RANGES_ROLE)
                    badge = right_item.data(ITEM_BADGE_ROLE) or ""
                    if right_item.data(IS_SEARCH_MATCH_ROLE):
                        badge = badge + "🔍"
                    text_label = self._make_text_cell_label(
                        right_text,
                        right_ranges if isinstance(right_ranges, list) else [],
                    )
                    right_widget: QLabel | QWidget = (
                        self._make_badge_cell_widget(text_label, badge) if badge else text_label
                    )
                    self.comparison_table.setCellWidget(row, 2, right_widget)
                else:
                    self.comparison_table.setCellWidget(
                        row, 2, self._make_text_cell_label("", [])
                    )

                action_text = ""
                callback: Callable[[], None] | None = None

                if right_index is not None:
                    right_item = self.right_list.item(right_index)
                    is_search_match = (
                        right_item.data(IS_SEARCH_MATCH_ROLE) if right_item else False
                    )
                    if is_search_match:
                        action_text = "←"
                        right_text = right_texts[right_index]
                        callback = lambda idx=right_index, text=right_text: self._add_search_match_to_tags(text)
                    elif left_index is not None:
                        if right_match_kind.get(right_index) != "exact":
                            action_text = "←"
                            callback = lambda table_row=row, idx=right_index: self._merge_proposed_row_from_table(table_row, idx)
                        elif self._is_fixup_deletion_candidate(left_texts[left_index]):
                            left_text = left_texts[left_index]
                            action_text = "✕"
                            callback = lambda table_row=row, value=left_text: self._remove_left_item_from_table(table_row, value)
                    else:
                        action_text = "←"
                        callback = lambda table_row=row, idx=right_index: self._merge_proposed_row_from_table(table_row, idx)
                elif left_index is not None and self._allow_left_delete:
                    left_text = left_texts[left_index]
                    action_text = "✕"
                    callback = lambda table_row=row, value=left_text: self._remove_left_item_from_table(table_row, value)

                self._set_action_cell(row, action_text, callback)
        finally:
            self._updating_comparison_table = False

        self.comparison_table.resizeRowsToContents()
        scrollbar.setValue(min(saved_scroll, scrollbar.maximum()))
        self._sync_comparison_widget_selection_styles()

    def _schedule_refresh_after_edit(self) -> None:
        if self._pending_edit_refresh:
            return

        self._pending_edit_refresh = True

        def _run_refresh() -> None:
            self._pending_edit_refresh = False
            saved_row = self.comparison_table.currentRow()
            self._update_difference_highlights()
            self.state_changed.emit()
            if saved_row >= 0:
                row_count = self.comparison_table.rowCount()
                if row_count > 0:
                    self.select_comparison_row(
                        min(saved_row, row_count - 1),
                        Qt.FocusReason.OtherFocusReason,
                    )

        QTimer.singleShot(0, _run_refresh)

    def _on_comparison_editor_created(self, editor: QTextEdit) -> None:
        self._active_comparison_editor = editor
        editor.destroyed.connect(lambda _obj=None: self._clear_active_comparison_editor(editor))

    def _on_comparison_editor_closed(self, editor, _hint) -> None:
        if isinstance(editor, QTextEdit):
            self._clear_active_comparison_editor(editor)

    def _clear_active_comparison_editor(self, editor: QTextEdit) -> None:
        if self._active_comparison_editor is editor:
            self._active_comparison_editor = None

    def _on_comparison_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_comparison_table:
            return
        if item.column() != 0:
            return

        row = item.row()
        if row < 0 or row >= len(self._table_row_map):
            return

        left_index, _right_index = self._table_row_map[row]
        if left_index is None or left_index < 0 or left_index >= self.left_list.count():
            return

        new_text = item.text().strip()
        if not new_text:
            removed = self.left_list.takeItem(left_index)
            del removed
        else:
            left_item = self.left_list.item(left_index)
            left_item.setText(new_text)
            left_item.setData(ITEM_TEXT_ROLE, new_text)

        self._schedule_refresh_after_edit()

    def _move_item_from_right_to_left(self, text: str) -> None:
        normalized = text.strip()
        normalized_key = self._normalized_compare_key(normalized)
        target_row = -1
        for i, value in enumerate(self._current_texts(self.right_list)):
            if self._normalized_compare_key(value) == normalized_key:
                target_row = i
                break
        if target_row < 0:
            return

        for table_row, (_left_index, right_index) in enumerate(self._table_row_map):
            if right_index == target_row:
                self._remember_last_action_table_row(table_row)
                break

        self._apply_proposed_rows([target_row])
        self._update_difference_highlights()
        self.state_changed.emit()

    def _remove_left_item(self, text: str) -> None:
        normalized = text.strip()
        for i in range(self.left_list.count()):
            item = self.left_list.item(i)
            widget = self.left_list.itemWidget(item)
            item_text = widget.text if widget else item.text()
            if item_text.strip() == normalized:
                self.left_list.takeItem(i)
                break
        self._update_difference_highlights()
        self.state_changed.emit()

    def _move_table_row(self, from_row: int, to_row: int) -> bool:
        """Reorder the left list so that the item at *from_row* moves to *to_row*.

        Only rows that own a left-list item can be reordered.  Returns True if
        the left list was actually modified.
        """
        if from_row == to_row:
            return False
        row_count = len(self._table_row_map)
        if from_row < 0 or from_row >= row_count or to_row < 0 or to_row >= row_count:
            return False

        from_left_idx, _ = self._table_row_map[from_row]
        if from_left_idx is None:
            return False  # Right-only rows cannot be reordered

        # Build the new table-row order after moving from_row to to_row.
        new_row_order = list(range(row_count))
        new_row_order.pop(from_row)
        new_row_order.insert(to_row, from_row)

        # Derive the new left-list item order from the new table-row order.
        left_count = self.left_list.count()
        new_left_order: list[int] = []
        seen: set[int] = set()
        for table_row in new_row_order:
            left_idx, _ = self._table_row_map[table_row]
            if left_idx is not None and left_idx not in seen:
                seen.add(left_idx)
                new_left_order.append(left_idx)
        # Include any left items not represented in the table (safety net).
        for i in range(left_count):
            if i not in seen:
                new_left_order.append(i)

        if new_left_order == list(range(left_count)):
            return False  # No effective change

        # Collect left-list text values before clearing.
        left_texts: list[str] = []
        for i in range(left_count):
            item = self.left_list.item(i)
            widget = self.left_list.itemWidget(item)
            if widget and hasattr(widget, "text"):
                left_texts.append(widget.text)
            else:
                text = item.text().strip()
                if not text:
                    stored = item.data(ITEM_TEXT_ROLE)
                    if isinstance(stored, str):
                        text = stored.strip()
                left_texts.append(text)

        self.left_list.clear()
        for idx in new_left_order:
            self._add_left_item(left_texts[idx])

        self._update_difference_highlights()
        self.state_changed.emit()
        return True
