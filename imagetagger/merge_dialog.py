from __future__ import annotations

from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Callable

from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, QSize, QStringListModel, QTimer
from PyQt6.QtGui import QAction, QColor, QKeySequence, QPixmap, QPainter, QPalette, QTextCharFormat, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCompleter,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class ItemActionWidget(QWidget):
    """Custom widget for list items with optional action button and diff highlighting."""
    
    def __init__(
        self,
        text: str,
        button_text: str = "",
        button_callback: Callable[[], None] | None = None,
        diff_ranges: list[tuple[int, int]] | None = None,
        button_on_left: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.text = text
        self.button_callback = button_callback
        self.diff_ranges = diff_ranges or []
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        
        # Create a label that will display highlighted text
        text_label = QLabel()
        text_label.setWordWrap(True)
        text_label.setMinimumHeight(20)
        text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Build HTML with highlighting for diff ranges
        html_text = self._build_highlighted_html(text, self.diff_ranges)
        text_label.setText(html_text)
        
        button: QPushButton | None = None
        if button_text and button_callback:
            button = QPushButton(button_text)
            button.setMaximumWidth(32)
            button.setMaximumHeight(20)
            button.clicked.connect(button_callback)
            if button_text == "✕":
                button.setStyleSheet("QPushButton { color: red; font-weight: bold; padding: 2px; }")

        if button is not None and button_on_left:
            layout.addWidget(button, stretch=0)
            layout.addWidget(text_label, stretch=1)
        else:
            layout.addWidget(text_label, stretch=1)
            if button is not None:
                layout.addWidget(button, stretch=0)
        
        self.setLayout(layout)
    
    def _build_highlighted_html(self, text: str, ranges: list[tuple[int, int]]) -> str:
        """Build HTML with diff highlighting for specified ranges."""
        if not ranges:
            return f"<span>{text}</span>"
        
        # Sort ranges to avoid overlaps
        sorted_ranges = sorted(ranges)
        
        html_parts = []
        last_end = 0
        
        for start, end in sorted_ranges:
            if start > last_end:
                html_parts.append(self._escape_html(text[last_end:start]))
            high_color = "#fff3b3"  # Light yellow
            html_parts.append(f'<span style="background-color: {high_color};">{self._escape_html(text[start:end])}</span>')
            last_end = end
        
        if last_end < len(text):
            html_parts.append(self._escape_html(text[last_end:]))
        
        return "".join(html_parts)
    
    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")


@dataclass
class FixupData:
    issues: str
    corrected_description: str
    corrected_tags: list[str]


DIFF_RANGES_ROLE = int(Qt.ItemDataRole.UserRole) + 1
ITEM_TEXT_ROLE = int(Qt.ItemDataRole.UserRole) + 2


class DiffHighlightDelegate(QStyledItemDelegate):
    def _normalized_ranges(self, text: str, raw_ranges: object) -> list[tuple[int, int]]:
        if not isinstance(raw_ranges, list):
            return []
        normalized: list[tuple[int, int]] = []
        text_len = len(text)
        for entry in raw_ranges:
            if not isinstance(entry, tuple) or len(entry) != 2:
                continue
            start, end = entry
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            bounded_start = max(0, min(text_len, start))
            bounded_end = max(bounded_start, min(text_len, end))
            if bounded_end > bounded_start:
                normalized.append((bounded_start, bounded_end))
        return normalized

    def _build_document(self, text: str, ranges: list[tuple[int, int]], option: QStyleOptionViewItem) -> QTextDocument:
        document = QTextDocument()
        document.setPlainText(text)
        document.setDefaultFont(option.font)

        if ranges:
            cursor = QTextCursor(document)
            char_format = QTextCharFormat()
            char_format.setBackground(QColor(255, 243, 179))
            for start, end in ranges:
                cursor.setPosition(start)
                cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(char_format)

        return document

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # type: ignore[override]
        style = option.widget.style() if option.widget is not None else None
        if style is None:
            style = QApplication.style()

        base = QStyleOptionViewItem(option)
        self.initStyleOption(base, index)
        text = base.text
        base.text = ""
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, base, painter, option.widget)

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            base,
            option.widget,
        )
        if text_rect.isEmpty() or not text:
            return

        ranges = self._normalized_ranges(text, index.data(DIFF_RANGES_ROLE))
        document = self._build_document(text, ranges, option)
        document.setTextWidth(float(max(10, text_rect.width())))

        painter.save()
        painter.translate(text_rect.topLeft())
        if option.state & QStyle.StateFlag.State_Selected:
            palette = option.palette
            painter.fillRect(
                0,
                0,
                text_rect.width(),
                text_rect.height(),
                palette.brush(QPalette.ColorRole.Highlight),
            )
        document.drawContents(painter)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> object:  # type: ignore[override]
        base = QStyleOptionViewItem(option)
        self.initStyleOption(base, index)
        text = base.text
        if not text:
            return super().sizeHint(option, index)

        width = option.rect.width()
        if width <= 0 and option.widget is not None:
            width = option.widget.viewport().width()
        width = max(40, width - 8)

        ranges = self._normalized_ranges(text, index.data(DIFF_RANGES_ROLE))
        document = self._build_document(text, ranges, option)
        document.setTextWidth(float(width))
        size = document.size().toSize()
        size.setHeight(max(size.height() + 6, 24))
        return size


class ScalableImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._original_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")

    def set_original_image(self, pixmap: QPixmap) -> None:
        self._original_pixmap = pixmap
        self._update_scaled_image()

    def _update_scaled_image(self) -> None:
        if self._original_pixmap is None:
            return
        
        available_width = self.width() - 4
        available_height = self.height() - 4
        
        if available_width <= 0 or available_height <= 0:
            return
        
        scaled = self._original_pixmap.scaled(
            QSize(available_width, available_height),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_scaled_image()


def parse_fixup_data(
    content: str,
    parse_tags: Callable[[str], list[str]],
    sanitize_annotation: Callable[[str], str],
) -> FixupData:
    sections: dict[str, list[str]] = {"issues": [], "tags": [], "description": []}
    current_section = "issues"

    for raw_line in content.splitlines():
        line = raw_line.strip()
        upper_line = line.upper()
        
        # Check if line starts with section header (handle both "HEADER:" and "HEADER: content")
        if upper_line.startswith("ISSUES:"):
            current_section = "issues"
            inline_content = line[7:].strip()  # Content after "ISSUES:"
            if inline_content:
                sections["issues"].append(inline_content)
            continue
        if upper_line.startswith("TAGS:"):
            current_section = "tags"
            inline_content = line[5:].strip()  # Content after "TAGS:"
            if inline_content:
                sections["tags"].append(inline_content)
            continue
        if upper_line.startswith("DESCRIPTION:"):
            current_section = "description"
            inline_content = line[12:].strip()  # Content after "DESCRIPTION:"
            if inline_content:
                sections["description"].append(inline_content)
            continue
        sections[current_section].append(raw_line.rstrip())

    issues = "\n".join(line for line in sections["issues"] if line.strip()).strip()
    corrected_description_raw = "\n".join(line for line in sections["description"] if line.strip()).strip()
    corrected_description = sanitize_annotation(corrected_description_raw)
    tags_text = "\n".join(line.strip() for line in sections["tags"] if line.strip())
    corrected_tags = parse_tags(tags_text)

    if not issues and not corrected_description and not corrected_tags:
        issues = content.strip()

    return FixupData(
        issues=issues,
        corrected_description=corrected_description,
        corrected_tags=corrected_tags,
    )


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
        initial_fixup_content: str = "",
        clear_fixup: Callable[[], bool] | None = None,
        restore_fixup: Callable[[str], bool] | None = None,
        can_navigate_prev: bool = False,
        can_navigate_next: bool = False,
        tag_suggestions: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title_text or "Fixup")
        self.resize(1280, 640)
        self._apply_annotations = apply_annotations
        self._initial_annotations = [tag.strip() for tag in current_tags if tag.strip()]
        self._initial_fixup_content = initial_fixup_content
        self._clear_fixup = clear_fixup
        self._restore_fixup = restore_fixup
        self._resolved = False
        self._undo_available = False
        self._has_proposed_description = bool(fixup_data.corrected_description)
        self._protected_existing_keys: set[str] = set()

        self.left_list = QListWidget(self)
        self.left_list.setItemDelegate(DiffHighlightDelegate(self.left_list))
        self.left_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.left_list.setWordWrap(True)
        self.left_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.left_list.setUniformItemSizes(False)

        self.remove_left_tag_action = QAction("Remove Selected Existing Tags", self)
        self.remove_left_tag_action.setShortcut("Delete")
        self.remove_left_tag_action.triggered.connect(self._remove_selected_left_items)
        self.left_list.addAction(self.remove_left_tag_action)

        self.left_tag_input = QLineEdit(self)
        self.left_tag_input.setPlaceholderText("Type a tag and press Enter")
        self.left_tag_input.returnPressed.connect(self._add_left_tag_from_input)

        self._tag_suggestions_model = QStringListModel(tag_suggestions or [], self)
        _completer = QCompleter(self._tag_suggestions_model, self)
        _completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        _completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)
        _completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.left_tag_input.setCompleter(_completer)

        self.right_list = QListWidget(self)
        self.right_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.right_list.setWordWrap(True)
        self.right_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.right_list.setUniformItemSizes(False)

        self.issues_label = QLabel(fixup_data.issues or "No issue details provided.", self)
        self.issues_label.setWordWrap(True)
        self.issues_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.issues_label.setStyleSheet("border: 1px solid #666; padding: 6px;")

        for tag in current_tags:
            self._add_left_item(tag)

        if not self._has_proposed_description:
            self._protected_existing_keys = self._find_description_like_keys(current_tags)

        if fixup_data.corrected_description:
            self._add_right_item(fixup_data.corrected_description)
        for tag in fixup_data.corrected_tags:
            self._add_right_item(tag)

        self.accept_button = QPushButton("Accept", self)
        self.accept_button.setShortcut(QKeySequence("Alt+A"))
        self.accept_button.setToolTip("Accept all proposed rows and merge (Alt+A)")
        self.accept_button.clicked.connect(self._accept_all_without_close)

        self.reject_button = QPushButton("Reject", self)
        self.reject_button.setShortcut(QKeySequence("Alt+R"))
        self.reject_button.setToolTip("Reject this fixup (Alt+R)")
        self.reject_button.clicked.connect(self._reject_without_close)

        self.merge_button = QPushButton("Merge", self)
        self.merge_button.setShortcut(QKeySequence("Alt+M"))
        self.merge_button.setToolTip("Apply current merged annotations (Alt+M)")
        self.merge_button.clicked.connect(self._merge_without_close)

        self.undo_button = QPushButton("Undo", self)
        self.undo_button.setEnabled(False)
        self.undo_button.setShortcut(QKeySequence("Alt+U"))
        self.undo_button.setToolTip("Undo the last merge or local changes (Alt+U)")
        self.undo_button.clicked.connect(self._undo_merge)

        self.accept_next_button = QPushButton("Accept and Next", self)
        self.accept_next_button.setShortcut(QKeySequence("Alt+Shift+A"))
        self.accept_next_button.setToolTip("Accept all proposed rows, merge, and go to next item (Alt+Shift+A)")
        self.accept_next_button.clicked.connect(self._accept_and_next)

        self.reject_next_button = QPushButton("Reject and Next", self)
        self.reject_next_button.setShortcut(QKeySequence("Alt+Shift+R"))
        self.reject_next_button.setToolTip("Reject this fixup and go to next item (Alt+Shift+R)")
        self.reject_next_button.clicked.connect(self._reject_and_next)

        self.merge_next_button = QPushButton("Merge and Next", self)
        self.merge_next_button.setShortcut(QKeySequence("Alt+Shift+M"))
        self.merge_next_button.setToolTip("Apply current merged annotations and go to next item (Alt+Shift+M)")
        self.merge_next_button.clicked.connect(self._merge_and_next)

        self.prev_button = QPushButton("Prev", self)
        self.prev_button.setEnabled(can_navigate_prev)
        self.prev_button.setShortcut(QKeySequence("Alt+Left"))
        self.prev_button.setToolTip("Go to previous item (Alt+Left)")
        self.prev_button.clicked.connect(self._navigate_prev)

        self.next_button = QPushButton("Next", self)
        self.next_button.setEnabled(can_navigate_next)
        self.next_button.setShortcut(QKeySequence("Alt+Right"))
        self.next_button.setToolTip("Go to next item (Alt+Right)")
        self.next_button.clicked.connect(self._navigate_next)

        for button in (
            self.accept_button,
            self.reject_button,
            self.merge_button,
            self.undo_button,
            self.accept_next_button,
            self.reject_next_button,
            self.merge_next_button,
            self.prev_button,
            self.next_button,
        ):
            button.setAutoDefault(False)
            button.setDefault(False)

        # Left pane
        left_pane = QWidget(self)
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Existing", self))
        left_layout.addWidget(self.left_list, stretch=1)
        left_layout.addWidget(self.left_tag_input, stretch=0)

        # Right pane
        right_pane = QWidget(self)
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Proposed", self))
        right_layout.addWidget(self.right_list, stretch=1)

        # Image pane
        image_pane = self._create_image_pane(image_path)

        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.addWidget(image_pane)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        splitter.setSizes([300, 300, 300])

        button_row = QHBoxLayout()
        button_row.addWidget(self.prev_button)
        button_row.addWidget(self.accept_button)
        button_row.addWidget(self.reject_button)
        button_row.addWidget(self.merge_button)
        button_row.addWidget(self.undo_button)
        button_row.addStretch(1)
        button_row.addWidget(self.accept_next_button)
        button_row.addWidget(self.reject_next_button)
        button_row.addWidget(self.merge_next_button)
        button_row.addWidget(self.next_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.issues_label)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(button_row)

        self._refresh_button_state()
        self._update_difference_highlights()
        self._refresh_widget_item_sizes()

    def _add_left_item(self, text: str) -> None:
        """Add item to left list with remove button if unmatched in right list."""
        normalized = text.strip()
        if not normalized:
            return
        item = QListWidgetItem(normalized)  # Set text on item
        item.setData(DIFF_RANGES_ROLE, [])
        self.left_list.addItem(item)

    def _add_left_tag_from_input(self) -> None:
        new_tag = self.left_tag_input.text().strip()
        if not new_tag:
            return

        existing_tags = [text.strip() for text in self._current_texts(self.left_list)]
        if new_tag in existing_tags:
            self.left_tag_input.selectAll()
            return

        self._add_left_item(new_tag)
        self.left_tag_input.clear()
        QTimer.singleShot(0, self.left_tag_input.clear)
        self._refresh_button_state()
        self._update_difference_highlights()

    def _add_right_item(self, text: str) -> None:
        """Add item to right list with accept button."""
        normalized = text.strip()
        if not normalized:
            return
        item = QListWidgetItem(normalized)
        item.setData(DIFF_RANGES_ROLE, [])
        item.setData(ITEM_TEXT_ROLE, normalized)
        self.right_list.addItem(item)

    def _current_texts(self, list_widget: QListWidget) -> list[str]:
        texts = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            widget = list_widget.itemWidget(item)
            if widget and hasattr(widget, 'text'):
                texts.append(widget.text)
            else:
                item_text = item.text().strip()
                if not item_text:
                    stored = item.data(ITEM_TEXT_ROLE)
                    if isinstance(stored, str):
                        item_text = stored.strip()
                texts.append(item_text)
        return texts

    @staticmethod
    def _is_description_like(text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        word_count = len(normalized.split())
        return word_count >= 5 or len(normalized) >= 40

    def _find_description_like_keys(self, values: list[str]) -> set[str]:
        candidates = [value.strip() for value in values if self._is_description_like(value)]
        if not candidates:
            return set()
        longest = max(candidates, key=len)
        return {longest.casefold()}

    def _is_protected_existing_text(self, text: str) -> bool:
        if self._has_proposed_description:
            return False
        return text.strip().casefold() in self._protected_existing_keys

    def _append_unique_to_left(self, values: list[str]) -> None:
        existing = {value.casefold() for value in self._current_texts(self.left_list)}
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in existing:
                continue
            existing.add(key)
            self._add_left_item(normalized)

    def _apply_proposed_rows(self, rows: list[int]) -> None:
        """Accept proposed rows by replacing matched existing items or appending new ones."""
        if not rows:
            return

        left_texts = self._current_texts(self.left_list)
        right_texts = self._current_texts(self.right_list)
        _, right_matches, _ = self._compute_matches(left_texts, right_texts)

        existing_keys = {text.strip().casefold() for text in left_texts if text.strip()}

        for right_row in rows:
            if right_row < 0 or right_row >= len(right_texts):
                continue

            proposed_text = right_texts[right_row].strip()
            if not proposed_text:
                continue

            matched_left_row = right_matches.get(right_row)
            if matched_left_row is not None and 0 <= matched_left_row < self.left_list.count():
                left_item = self.left_list.item(matched_left_row)
                # Ensure text is visible even if this row previously had a widget.
                if self.left_list.itemWidget(left_item) is not None:
                    self.left_list.setItemWidget(left_item, None)
                left_item.setText(proposed_text)
                left_item.setData(ITEM_TEXT_ROLE, proposed_text)
                existing_keys.add(proposed_text.casefold())
                continue

            key = proposed_text.casefold()
            if key not in existing_keys:
                self._add_left_item(proposed_text)
                existing_keys.add(key)

    def _remove_right_rows(self, rows: list[int]) -> None:
        for row in sorted(rows, reverse=True):
            item = self.right_list.takeItem(row)
            del item

    def _refresh_button_state(self) -> None:
        has_right_items = self.right_list.count() > 0
        is_resolved = self._resolved
        has_local_changes = self._has_local_changes()
        can_navigate_next = self.next_button.isEnabled()
        self.accept_button.setEnabled(has_right_items and not is_resolved)
        self.reject_button.setEnabled(not is_resolved)
        self.merge_button.setEnabled(has_local_changes and not is_resolved)
        self.undo_button.setEnabled(self._undo_available or has_local_changes)
        self.accept_next_button.setEnabled(has_right_items and not is_resolved and can_navigate_next)
        self.reject_next_button.setEnabled(not is_resolved and can_navigate_next)
        self.merge_next_button.setEnabled(has_local_changes and not is_resolved and can_navigate_next)

    def _has_local_changes(self) -> bool:
        current = [text.strip() for text in self.selected_annotations() if text.strip()]
        return current != self._initial_annotations

    def _compute_matches(
        self,
        left_texts: list[str],
        right_texts: list[str],
    ) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
        left_matches: dict[int, int] = {}
        right_matches: dict[int, int] = {}
        right_match_kind: dict[int, str] = {}

        right_by_key: dict[str, list[int]] = {}
        for right_index, text in enumerate(right_texts):
            right_by_key.setdefault(text.strip().casefold(), []).append(right_index)

        protected_left_indexes: set[int] = {
            index
            for index, text in enumerate(left_texts)
            if self._is_protected_existing_text(text)
        }

        # Pass 1: exact matches
        for left_index, text in enumerate(left_texts):
            if left_index in protected_left_indexes:
                continue
            key = text.strip().casefold()
            candidates = right_by_key.get(key, [])
            if not candidates:
                continue
            right_index = candidates.pop(0)
            left_matches[left_index] = right_index
            right_matches[right_index] = left_index
            right_match_kind[right_index] = "exact"

        # Pass 2: fuzzy matches
        for left_index, left_text in enumerate(left_texts):
            if left_index in protected_left_indexes:
                continue
            if left_index in left_matches:
                continue
            best_right = -1
            best_ratio = 0.0
            for right_index, right_text in enumerate(right_texts):
                if right_index in right_matches:
                    continue
                ratio = SequenceMatcher(None, left_text.casefold(), right_text.casefold()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_right = right_index

            if best_right >= 0 and best_ratio >= 0.25:
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
        # Existing list: show X only for truly unmatched items.
        for i in range(self.left_list.count()):
            item = self.left_list.item(i)
            item_text = left_texts[i] if i < len(left_texts) else item.text().strip()
            existing_widget = self.left_list.itemWidget(item)

            should_show_x = i not in left_matches and not self._is_protected_existing_text(item_text)

            if should_show_x:
                if not isinstance(existing_widget, ItemActionWidget):
                    def make_remove_callback(text: str) -> Callable[[], None]:
                        return lambda: self._remove_left_item(text)

                    widget = ItemActionWidget(item_text, "✕", make_remove_callback(item_text), [], False)
                    item.setData(ITEM_TEXT_ROLE, item_text)
                    item.setText("")
                    self.left_list.setItemWidget(item, widget)
            else:
                if existing_widget is not None:
                    restored = item.data(ITEM_TEXT_ROLE)
                    if isinstance(restored, str) and restored.strip():
                        item.setText(restored)
                    self.left_list.setItemWidget(item, None)

        # Proposed list: show arrow only when applicable (not exact-match rows).
        for i in range(self.right_list.count()):
            item = self.right_list.item(i)
            text = right_texts[i] if i < len(right_texts) else ""
            ranges = item.data(DIFF_RANGES_ROLE)
            existing_widget = self.right_list.itemWidget(item)

            def make_accept_callback(value: str) -> Callable[[], None]:
                return lambda: self._move_item_from_right_to_left(value)

            show_arrow = right_match_kind.get(i) != "exact"
            button_text = "←" if show_arrow else ""
            callback = make_accept_callback(text) if show_arrow else None
            widget = ItemActionWidget(
                text,
                button_text,
                callback,
                ranges if isinstance(ranges, list) else [],
                button_on_left=True,
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

    def _refresh_widget_item_sizes(self) -> None:
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

    def _update_difference_highlights(self) -> None:
        left_items = [self.left_list.item(i) for i in range(self.left_list.count())]
        right_items = [self.right_list.item(i) for i in range(self.right_list.count())]
        
        # Get text from items (handling widgets)
        left_texts = []
        for item in left_items:
            widget = self.left_list.itemWidget(item)
            if widget and hasattr(widget, 'text'):
                left_texts.append(widget.text)
            else:
                left_texts.append(item.text())
        
        right_texts = []
        for item in right_items:
            widget = self.right_list.itemWidget(item)
            if widget and hasattr(widget, 'text'):
                right_texts.append(widget.text)
            else:
                right_texts.append(item.text())

        left_matches, right_matches, right_match_kind = self._compute_matches(left_texts, right_texts)

        def ranges_for_diff(left_text: str, right_text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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

        # Update left items with diff ranges
        for left_index, item in enumerate(left_items):
            ranges = []
            if left_index in left_matches:
                right_index = left_matches[left_index]
                ranges, _ = ranges_for_diff(left_texts[left_index], right_texts[right_index])
            else:
                if self._is_protected_existing_text(left_texts[left_index]):
                    ranges = []
                else:
                    full = [(0, len(left_texts[left_index]))] if left_texts[left_index] else []
                    ranges = full
            
            # Store ranges for delegate rendering or widget display
            item.setData(DIFF_RANGES_ROLE, ranges)
            
            # Left widgets are refreshed centrally in _update_action_widgets.

        # Update right items with diff ranges  
        for right_index, item in enumerate(right_items):
            ranges = []
            if right_index in right_matches:
                left_index = right_matches[right_index]
                _, ranges = ranges_for_diff(left_texts[left_index], right_texts[right_index])
            else:
                full = [(0, len(right_texts[right_index]))] if right_texts[right_index] else []
                ranges = full
            
            item.setData(DIFF_RANGES_ROLE, ranges)
            # Right widgets are refreshed centrally in _update_action_widgets.

        self._update_action_widgets(left_texts, right_texts, left_matches, right_match_kind)

        self._refresh_widget_item_sizes()

    def accept_all(self) -> None:
        proposed_values = [value.strip() for value in self._current_texts(self.right_list) if value.strip()]
        preserved_values = [
            value.strip()
            for value in self._current_texts(self.left_list)
            if value.strip() and self._is_protected_existing_text(value)
        ]

        merged_values: list[str] = []
        seen: set[str] = set()

        for value in preserved_values + proposed_values:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged_values.append(value)

        self.left_list.clear()
        for value in merged_values:
            self._add_left_item(value)

        self._refresh_button_state()
        self._update_difference_highlights()

    def _move_item_from_right_to_left(self, text: str) -> None:
        """Move a specific item from right to left list."""
        normalized = text.strip()
        target_row = -1
        # Find corresponding row in right list
        for i in range(self.right_list.count()):
            item_widget = self.right_list.itemWidget(self.right_list.item(i))
            if item_widget and item_widget.text.strip() == normalized:
                target_row = i
                break
        if target_row < 0:
            return

        self._apply_proposed_rows([target_row])
        self._refresh_button_state()
        self._update_difference_highlights()

    def _remove_left_item(self, text: str) -> None:
        """Remove a specific item from left list."""
        normalized = text.strip()
        if self._is_protected_existing_text(normalized):
            return
        for i in range(self.left_list.count()):
            item = self.left_list.item(i)
            # Get text from widget if set, otherwise from item
            widget = self.left_list.itemWidget(item)
            item_text = widget.text if widget else item.text()
            if item_text.strip() == normalized:
                self.left_list.takeItem(i)
                break
        self._refresh_button_state()
        self._update_difference_highlights()

    def _remove_selected_left_items(self) -> None:
        selected = self.left_list.selectedItems()
        if not selected:
            return

        removed_any = False
        for item in selected:
            widget = self.left_list.itemWidget(item)
            item_text = widget.text if widget else item.text()
            if self._is_protected_existing_text(item_text):
                continue
            row = self.left_list.row(item)
            removed = self.left_list.takeItem(row)
            del removed
            removed_any = True

        if not removed_any:
            return

        self._refresh_button_state()
        self._update_difference_highlights()

    def selected_annotations(self) -> list[str]:
        return self._current_texts(self.left_list)

    def _create_image_pane(self, image_path: Path | None) -> QWidget:
        pane = QWidget(self)
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.addWidget(QLabel("Image", self))

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        image_label = ScalableImageLabel(self)

        image_loaded = False
        if image_path:
            # Ensure path is a Path object
            if isinstance(image_path, str):
                image_path = Path(image_path)
            
            if image_path.exists() and image_path.is_file():
                try:
                    pixmap = QPixmap(str(image_path))
                    if not pixmap.isNull():
                        image_label.set_original_image(pixmap)
                        image_loaded = True
                    else:
                        image_label.setText(f"Unsupported format:\n{image_path.name}")
                except Exception as e:
                    image_label.setText(f"Failed to load:\n{str(e)[:50]}")
            else:
                image_label.setText(f"File not found:\n{image_path.name if image_path else 'unknown'}")
        
        if not image_loaded and image_label.text() == "":
            image_label.setText("No image path provided")

        scroll.setWidget(image_label)
        pane_layout.addWidget(scroll, stretch=1)
        return pane

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.right_list.itemSelectionChanged.connect(self._refresh_button_state)
        self.left_list.itemSelectionChanged.connect(self._refresh_button_state)
        self._refresh_widget_item_sizes()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_widget_item_sizes()

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
        self.left_list.clear()
        for text in self._initial_annotations:
            self._add_left_item(text)

        self._refresh_button_state()
        self._update_difference_highlights()

    def _merge_without_close(self) -> bool:
        if self._apply_annotations is None:
            return False
        self._apply_annotations(self.selected_annotations(), "Fixup merged + auto-saved")
        self._undo_available = True
        return self._resolve_fixup()

    def _accept_all_without_close(self) -> bool:
        self.accept_all()
        return self._merge_without_close()

    def _reject_without_close(self) -> bool:
        if self._resolve_fixup():
            self._undo_available = True
            self._refresh_button_state()
            return True
        return False

    def _undo_merge(self) -> None:
        if not self._undo_available and not self._has_local_changes():
            return

        if self._undo_available:
            if self._apply_annotations is None:
                return
            if self._restore_fixup is None or not self._restore_fixup(self._initial_fixup_content):
                return
            self._apply_annotations(list(self._initial_annotations), "Fixup undone + auto-saved")

        self._restore_initial_state()
        self._resolved = False
        self._undo_available = False
        self._refresh_button_state()

    def _navigate_prev(self) -> None:
        self.done(self.NAVIGATE_PREV_CODE)

    def _navigate_next(self) -> None:
        self.done(self.NAVIGATE_NEXT_CODE)

    def _accept_and_next(self) -> None:
        if self._accept_all_without_close():
            self._navigate_next()

    def _reject_and_next(self) -> None:
        if self._reject_without_close():
            self._navigate_next()

    def _merge_and_next(self) -> None:
        if self._merge_without_close():
            self._navigate_next()