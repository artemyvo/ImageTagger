from __future__ import annotations

from difflib import SequenceMatcher
from dataclasses import dataclass, field
import sys
import time
from typing import Callable

from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, Qt, QSize, QStringListModel, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDoubleValidator, QIntValidator, QKeySequence, QPixmap, QPainter, QPalette, QTextCharFormat, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QCompleter,
    QDialog,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imagetagger.ollama import (
    OllamaCancellation,
    OllamaCancelled,
    OllamaError,
    configure_runtime,
    consume_resize_warning,
    generate_description,
    generate_tags,
)
from imagetagger.external_editors import (
    ExternalEditor,
    discover_graphics_editors,
    launch_image_in_editor,
    launch_image_in_system_default,
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
    search_matches: list[str] = field(default_factory=list)


class RegenerateWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, task: Callable[[Callable[[str], None]], object]) -> None:
        super().__init__()
        self.task = task

    def run(self) -> None:
        try:
            result = self.task(self.progress.emit)
        except OllamaCancelled as exc:
            self.cancelled.emit(str(exc))
            return
        except OllamaError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"Unexpected Ollama error: {exc}")
            return
        self.finished.emit(result)


DIFF_RANGES_ROLE = int(Qt.ItemDataRole.UserRole) + 1
ITEM_TEXT_ROLE = int(Qt.ItemDataRole.UserRole) + 2
IS_SEARCH_MATCH_ROLE = int(Qt.ItemDataRole.UserRole) + 3


def strip_tag_list_prefix(tag: str) -> str:
    """Remove one or more leading markdown list markers from a tag value."""
    cleaned = tag.strip()
    while cleaned.startswith("- "):
        cleaned = cleaned[2:].lstrip()
    return cleaned


def _normalize_fixup_section_entry(value: str) -> str:
    """Normalize entry by stripping whitespace and leading dash."""
    text = value.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    return text


def _normalize_search_match_entry(value: str, sanitize_annotation: Callable[[str], str]) -> str:
    normalized = sanitize_annotation(_normalize_fixup_section_entry(value)).strip()
    return normalized.lower()


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
        base.state &= ~QStyle.StateFlag.State_HasFocus
        text = base.text
        base.text = ""
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, base, painter, option.widget)

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            base,
            option.widget,
        )
        if text_rect.isEmpty():
            return

        if option.state & QStyle.StateFlag.State_Selected:
            palette = option.palette
            painter.save()
            painter.fillRect(
                text_rect,
                palette.brush(QPalette.ColorRole.Highlight),
            )
            painter.restore()

        if not text:
            return

        # When selected, keep text plain for readability on highlight background.
        ranges = [] if (option.state & QStyle.StateFlag.State_Selected) else self._normalized_ranges(text, index.data(DIFF_RANGES_ROLE))
        document = self._build_document(text, ranges, option)
        document.setTextWidth(float(max(10, text_rect.width())))

        painter.save()
        painter.translate(text_rect.topLeft())
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


class EditableDiffDelegate(DiffHighlightDelegate):
    _CLOSED_BY_DELEGATE_PROP = "_closed_by_editable_diff_delegate"

    def _resize_editor(self, editor: QTextEdit) -> None:
        margins = editor.contentsMargins()
        document = editor.document()
        document.setTextWidth(max(20.0, editor.viewport().width()))
        doc_height = int(document.size().height())
        frame = editor.frameWidth() * 2
        vertical_margins = margins.top() + margins.bottom()
        min_height = editor.fontMetrics().lineSpacing() + 10
        editor.setFixedHeight(max(min_height, doc_height + frame + vertical_margins + 4))

    def createEditor(self, parent, option, index):  # type: ignore[override]
        editor = QTextEdit(parent)
        editor.setAcceptRichText(False)
        editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor.setFrameStyle(0)
        editor.installEventFilter(self)
        editor.textChanged.connect(lambda: self._resize_editor(editor))
        return editor

    def setEditorData(self, editor, index) -> None:  # type: ignore[override]
        if isinstance(editor, QTextEdit):
            editor.setPlainText(index.data(Qt.ItemDataRole.EditRole) or "")
            self._resize_editor(editor)
            editor.selectAll()

    def setModelData(self, editor, model, index) -> None:  # type: ignore[override]
        if isinstance(editor, QTextEdit):
            model.setData(index, editor.toPlainText(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index) -> None:  # type: ignore[override]
        if isinstance(editor, QTextEdit):
            editor.setGeometry(option.rect)
            self._resize_editor(editor)

    def eventFilter(self, editor, event) -> bool:
        if isinstance(editor, QTextEdit):
            if event.type() == QEvent.Type.KeyPress:
                key = event.key()
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    editor.setProperty(self._CLOSED_BY_DELEGATE_PROP, True)
                    self.commitData.emit(editor)
                    self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
                    return True
                if key == Qt.Key.Key_Escape:
                    editor.setProperty(self._CLOSED_BY_DELEGATE_PROP, True)
                    self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.RevertModelCache)
                    return True
            if event.type() == QEvent.Type.FocusOut:
                if bool(editor.property(self._CLOSED_BY_DELEGATE_PROP)):
                    return True
                editor.setProperty(self._CLOSED_BY_DELEGATE_PROP, True)
                self.commitData.emit(editor)
                self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
                return True
        return super().eventFilter(editor, event)


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
    sections: dict[str, list[str]] = {"issues": [], "tags": [], "description": [], "ai_find": []}
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
        if upper_line.startswith("AI_FIND_MATCHES:"):
            current_section = "ai_find"
            inline_content = line[16:].strip()  # Content after "AI_FIND_MATCHES:"
            if inline_content:
                sections["ai_find"].append(inline_content)
            continue
        sections[current_section].append(raw_line.rstrip())

    issues = "\n".join(line for line in sections["issues"] if line.strip()).strip()
    corrected_description_raw = "\n".join(line for line in sections["description"] if line.strip()).strip()
    corrected_description = sanitize_annotation(corrected_description_raw)
    tags_text = "\n".join(line.strip() for line in sections["tags"] if line.strip())
    corrected_tags = [
        cleaned
        for tag in parse_tags(tags_text)
        if (cleaned := strip_tag_list_prefix(tag))
    ]
    
    search_matches = []
    seen_search_matches: set[str] = set()
    for line in sections["ai_find"]:
        normalized_match = _normalize_search_match_entry(line, sanitize_annotation)
        if not normalized_match:
            continue
        if normalized_match in seen_search_matches:
            continue
        seen_search_matches.add(normalized_match)
        search_matches.append(normalized_match)

    if not issues and not corrected_description and not corrected_tags:
        issues = content.strip()

    return FixupData(
        issues=issues,
        corrected_description=corrected_description,
        corrected_tags=corrected_tags,
        search_matches=search_matches,
    )


class FixupDialog(QDialog):
    NAVIGATE_PREV_CODE = 2
    NAVIGATE_NEXT_CODE = 3
    _PANE_HEADER_BOTTOM_SPACING = 4
    _PANE_HEADER_EXTRA_HEIGHT = 4
    _ROW_WIDGET_SELECTED_STYLE = "background-color: palette(highlight);"
    _ROW_WIDGET_UNSELECTED_STYLE = "background-color: transparent;"
    _TEXT_WIDGET_SELECTED_STYLE = "background-color: palette(highlight); color: palette(highlighted-text);"
    _TEXT_WIDGET_UNSELECTED_STYLE = "background-color: transparent; color: palette(text);"

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
        normalize_annotation: Callable[[str], str] | None = None,
        ollama_server_url: str = "",
        ollama_model_name: str = "",
        regenerate_tags_enabled: bool = True,
        regenerate_description_enabled: bool = True,
        regenerate_timeout_seconds: int = 300,
        regenerate_retry_count: int = 3,
        regenerate_max_resolution_mpx: float = 5.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title_text or "Fixup")
        self.resize(1280, 640)
        self._apply_annotations = apply_annotations
        self._normalize_annotation = normalize_annotation or (lambda text: text)
        self._initial_annotations = [tag.strip() for tag in current_tags if tag.strip()]
        self._last_merged_annotations = list(self._initial_annotations)
        self._initial_proposed_description = fixup_data.corrected_description.strip()
        self._initial_proposed_tags = self._drop_description_duplicate_tags(
            self._initial_proposed_description,
            [tag.strip() for tag in fixup_data.corrected_tags if tag.strip()],
        )
        self._initial_search_matches = [query.strip() for query in fixup_data.search_matches if query.strip()]
        self._initial_fixup_content = initial_fixup_content
        self._clear_fixup = clear_fixup
        self._restore_fixup = restore_fixup
        self._resolved = False
        self._undo_available = False
        self._has_proposed_description = bool(fixup_data.corrected_description)
        self._exact_match_only_for_tags = False
        self._protected_existing_keys: set[str] = set()
        self._updating_comparison_table = False
        self._pending_edit_refresh = False
        self._last_action_table_row: int | None = None
        self._image_path = image_path
        self._ollama_server_url = ollama_server_url.strip()
        self._ollama_model_name = ollama_model_name.strip()
        self._regenerate_thread: QThread | None = None
        self._regenerate_worker: RegenerateWorker | None = None
        self._regenerate_cancel: OllamaCancellation | None = None
        self._discard_regenerate_result = False
        self._global_key_filter_installed = False
        self._search_matches = fixup_data.search_matches or []
        self._detected_external_editors: list[ExternalEditor] | None = None

        self.left_list = QListWidget(self)
        self.left_list.setItemDelegate(DiffHighlightDelegate(self.left_list))
        self.left_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.left_list.setWordWrap(True)
        self.left_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.left_list.setUniformItemSizes(False)
        self.left_list.hide()

        self.remove_left_tag_action = QAction("Remove Selected Existing Tags", self)
        self.remove_left_tag_action.setShortcut("Delete")
        self.remove_left_tag_action.triggered.connect(self._remove_selected_left_items)
        self.left_list.addAction(self.remove_left_tag_action)
        self.addAction(self.remove_left_tag_action)

        self.merge_and_next_alt_action = QAction("Merge and Next", self)
        self.merge_and_next_alt_action.setShortcuts(
            [QKeySequence("Alt+Enter"), QKeySequence("Alt+Return")]
        )
        self.merge_and_next_alt_action.triggered.connect(self._merge_and_next)
        self.addAction(self.merge_and_next_alt_action)

        self.focus_tag_input_action = QAction("Focus Tag Input", self)
        self.focus_tag_input_action.setShortcut(QKeySequence("Alt+T"))
        self.focus_tag_input_action.triggered.connect(self._focus_left_tag_input)
        self.addAction(self.focus_tag_input_action)

        self.undo_alt_action = QAction("Undo", self)
        self.undo_alt_action.setShortcut(QKeySequence("Alt+U"))
        self.undo_alt_action.triggered.connect(self._undo_merge)
        self.addAction(self.undo_alt_action)

        self.prev_actionable_row_action = QAction("Previous Actionable Row", self)
        self.prev_actionable_row_action.setShortcut(QKeySequence("Alt+Up"))
        self.prev_actionable_row_action.triggered.connect(self._activate_previous_actionable_row)
        self.addAction(self.prev_actionable_row_action)

        self.next_actionable_row_action = QAction("Next Actionable Row", self)
        self.next_actionable_row_action.setShortcut(QKeySequence("Alt+Down"))
        self.next_actionable_row_action.triggered.connect(self._activate_next_actionable_row)
        self.addAction(self.next_actionable_row_action)

        self.left_tag_input = QLineEdit(self)
        self.left_tag_input.setPlaceholderText("Type a tag and press Enter")
        self.left_tag_input.setToolTip("Type a tag and press Enter. Alt+T clears and focuses this field.")
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
        self.comparison_table.setItemDelegateForColumn(0, EditableDiffDelegate(self.comparison_table))
        self.comparison_table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.comparison_table.setMouseTracking(False)
        self.comparison_table.viewport().setMouseTracking(False)
        self.comparison_table.setShowGrid(False)
        self.comparison_table.setGridStyle(Qt.PenStyle.NoPen)
        self.comparison_table.setStyleSheet(self._comparison_table_base_stylesheet())
        self.comparison_table.setToolTip(
            "Keyboard: Alt+Up/Alt+Down jump actionable rows, Left applies proposed rows, "
            "Enter triggers current row action, Delete removes selected current rows."
        )
        self.comparison_table.itemChanged.connect(self._on_comparison_item_changed)
        self.comparison_table.itemSelectionChanged.connect(self._on_comparison_selection_changed)
        self.comparison_table.installEventFilter(self)
        header = self.comparison_table.horizontalHeader()
        header.setVisible(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.comparison_left_label = self._create_pane_header_label("Current")
        self.comparison_right_label = self._create_pane_header_label("Proposed")

        self.comparison_header_action_spacer = QWidget(self)
        self._action_button_width = 24
        self._action_button_height = 22
        self._update_action_column_metrics()
        self._table_row_map: list[tuple[int | None, int | None]] = []

        self.regenerate_tags_checkbox = QCheckBox("Tags", self)
        self.regenerate_tags_checkbox.setChecked(regenerate_tags_enabled)
        self.regenerate_tags_checkbox.checkStateChanged.connect(lambda _state: self._update_regenerate_controls())

        self.regenerate_description_checkbox = QCheckBox("Description", self)
        self.regenerate_description_checkbox.setChecked(regenerate_description_enabled)
        self.regenerate_description_checkbox.checkStateChanged.connect(lambda _state: self._update_regenerate_controls())

        self.regenerate_timeout_input = QLineEdit(self)
        self.regenerate_timeout_input.setValidator(QIntValidator(1, 86400, self))
        self.regenerate_timeout_input.setText(str(max(1, int(regenerate_timeout_seconds))))
        self.regenerate_timeout_input.setMaximumWidth(90)

        self.regenerate_retry_input = QLineEdit(self)
        self.regenerate_retry_input.setValidator(QIntValidator(0, 10, self))
        self.regenerate_retry_input.setText(str(max(0, int(regenerate_retry_count))))
        self.regenerate_retry_input.setMaximumWidth(60)

        self.regenerate_max_resolution_input = QLineEdit(self)
        max_resolution_validator = QDoubleValidator(0.01, 1000.0, 3, self)
        max_resolution_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.regenerate_max_resolution_input.setValidator(max_resolution_validator)
        try:
            max_resolution_value = float(regenerate_max_resolution_mpx)
            if max_resolution_value <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            max_resolution_value = 5.0
        self.regenerate_max_resolution_input.setText(self._format_mpx(max_resolution_value))
        self.regenerate_max_resolution_input.setMaximumWidth(80)

        self.regenerate_button = QPushButton("&Regenerate", self)
        self.regenerate_button.setShortcut(QKeySequence("Alt+R"))
        self.regenerate_button.setToolTip("Regenerate proposed annotations (Alt+R)")
        self.regenerate_button.clicked.connect(self._regenerate_proposed_annotations)

        self.regenerate_status_label = QLabel(self)
        self.regenerate_status_label.setWordWrap(True)
        self.regenerate_status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

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
        for search_query in fixup_data.search_matches:
            self._add_search_match_item(search_query)

        self.accept_button = QPushButton("Accept", self)
        self.accept_button.setToolTip("Accept all proposed rows and merge")
        self.accept_button.clicked.connect(self._accept_all_without_close)

        self.merge_button = QPushButton("Merge", self)
        self.merge_button.setToolTip("Apply current merged annotations")
        self.merge_button.clicked.connect(self._merge_without_close)

        self.undo_button = QPushButton("Undo", self)
        self.undo_button.setEnabled(False)
        self.undo_button.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        self.undo_button.setToolTip("Undo the last merge or local changes (Alt+U)")
        self.undo_button.clicked.connect(self._undo_merge)

        self.merge_next_button = QPushButton("Merge and Next", self)
        self.merge_next_button.setShortcut(QKeySequence("Alt+Enter"))
        self.merge_next_button.setToolTip("Apply current annotations and go to next item, even with no local edits (Alt+Enter)")
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
            self.regenerate_button,
            self.accept_button,
            self.merge_button,
            self.undo_button,
            self.merge_next_button,
            self.prev_button,
            self.next_button,
        ):
            button.setAutoDefault(False)
            button.setDefault(False)

        # Table pane
        table_pane = QWidget(self)
        table_layout = QVBoxLayout(table_pane)
        table_layout.setContentsMargins(0, 0, 0, 0)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, self._PANE_HEADER_BOTTOM_SPACING)
        header_row.setSpacing(0)
        header_row.addWidget(self.comparison_left_label, stretch=1)
        header_row.addWidget(self.comparison_header_action_spacer, stretch=0)
        header_row.addWidget(self.comparison_right_label, stretch=1)
        table_layout.addLayout(header_row, stretch=0)
        table_layout.addWidget(self.comparison_table, stretch=1)
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(8)
        input_row.addWidget(self.left_tag_input, stretch=1)
        input_row.addStretch(1)
        table_layout.addLayout(input_row, stretch=0)

        # Image pane
        image_pane = self._create_image_pane(image_path)

        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(table_pane)
        splitter.addWidget(image_pane)
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

        self._update_merge_next_button_presentation()
        self._refresh_button_state()
        self._update_difference_highlights()
        self._refresh_widget_item_sizes()
        self._update_regenerate_controls()

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
        normalized_new_key = self._normalized_compare_key(new_tag)
        existing_keys = {
            self._normalized_compare_key(text)
            for text in existing_tags
            if self._normalized_compare_text(text)
        }
        if normalized_new_key in existing_keys:
            self._select_existing_tag_row(normalized_new_key)
            self.left_tag_input.selectAll()
            return

        self._add_left_item(new_tag)
        self.left_tag_input.clear()
        QTimer.singleShot(0, self.left_tag_input.clear)
        QTimer.singleShot(0, self.left_tag_input.setFocus)
        self._refresh_button_state()
        self._update_difference_highlights()

    def _add_search_match_to_tags(self, search_query: str) -> None:
        """Add search query to current tags (left list) and remove from search matches."""
        normalized = search_query.strip()
        if not normalized:
            return
        
        current_tags = self._current_texts(self.left_list)
        normalized_key = self._normalized_compare_key(normalized)
        existing_keys = {self._normalized_compare_key(tag) for tag in current_tags}
        
        if normalized_key not in existing_keys:
            self._add_left_item(normalized)
        
        # Remove the search match item from right list
        for i in range(self.right_list.count() - 1, -1, -1):
            item = self.right_list.item(i)
            if item and item.data(IS_SEARCH_MATCH_ROLE):
                item_text = item.text().strip()
                if self._normalized_compare_key(item_text) == normalized_key:
                    self.right_list.takeItem(i)
                    break
        
        self._refresh_button_state()
        self._update_difference_highlights()

    def _focus_left_tag_input(self) -> None:
        self.left_tag_input.clear()
        self.left_tag_input.setFocus(Qt.FocusReason.ShortcutFocusReason)

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

    def _add_right_item(self, text: str) -> None:
        """Add item to right list with accept button."""
        normalized = text.strip()
        if not normalized:
            return
        item = QListWidgetItem(normalized)
        item.setData(DIFF_RANGES_ROLE, [])
        item.setData(ITEM_TEXT_ROLE, normalized)
        self.right_list.addItem(item)

    def _add_search_match_item(self, search_query: str) -> None:
        """Add search match item to right list (marked as search match)."""
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

    def _append_unique_to_left(self, values: list[str]) -> None:
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

    @staticmethod
    def _strip_tag_list_prefix(tag: str) -> str:
        """Remove a leading markdown list marker from a tag value."""
        return strip_tag_list_prefix(tag)

    def _normalize_proposed_text_for_merge(self, text: str, right_row: int) -> str:
        normalized = text.strip()
        # Row 0 is reserved for description when present; other proposed rows are tags.
        if self._has_proposed_description and right_row == 0:
            return normalized
        return self._strip_tag_list_prefix(normalized)

    def _apply_proposed_rows(self, rows: list[int]) -> None:
        """Accept proposed rows by replacing matched existing items or appending new ones."""
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

            matched_left_row = right_matches.get(right_row)
            if matched_left_row is not None and 0 <= matched_left_row < self.left_list.count():
                left_item = self.left_list.item(matched_left_row)
                # Ensure text is visible even if this row previously had a widget.
                if self.left_list.itemWidget(left_item) is not None:
                    self.left_list.setItemWidget(left_item, None)
                left_item.setText(proposed_text)
                left_item.setData(ITEM_TEXT_ROLE, proposed_text)
                existing_keys.add(self._normalized_compare_key(proposed_text))
                continue

            key = self._normalized_compare_key(proposed_text)
            if key not in existing_keys:
                if self._has_proposed_description and right_row == 0:
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

    def _select_comparison_row(self, row: int, focus_reason: Qt.FocusReason) -> bool:
        row_count = self.comparison_table.rowCount()
        if row < 0 or row >= row_count:
            return False

        preferred_column = self._preferred_focus_column_for_row(row)

        # Set current cell BEFORE setFocus so Qt's focusInEvent does not
        # auto-select row 0 (its fallback when no current index exists).
        self.comparison_table.clearSelection()
        self.comparison_table.setCurrentCell(row, preferred_column)
        self.comparison_table.setFocus(focus_reason)
        self.comparison_table.selectRow(row)
        return True

    def _activate_adjacent_row_for_last_action(self, step: int) -> bool:
        if self._last_action_table_row is None:
            return False

        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return False

        if step < 0:
            target_row = min(max(0, self._last_action_table_row - 1), row_count - 1)
        else:
            target_row = min(max(0, self._last_action_table_row), row_count - 1)
        return self._select_comparison_row(target_row, Qt.FocusReason.ShortcutFocusReason)

    def _activate_last_comparison_row(self) -> bool:
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return False

        return self._select_comparison_row(row_count - 1, Qt.FocusReason.ShortcutFocusReason)

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
                return self._select_comparison_row(row, Qt.FocusReason.ShortcutFocusReason)
            row += step

        return False

    def _activate_previous_actionable_row(self) -> None:
        self._activate_adjacent_actionable_row(-1)

    def _activate_next_actionable_row(self) -> None:
        self._activate_adjacent_actionable_row(1)

    def _advance_to_next_actionable_from(self, start_row: int) -> None:
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return
        # Try to find an actionable row at or after start_row.
        for row in range(start_row, row_count):
            if self._row_needs_addressing(row):
                self._select_comparison_row(row, Qt.FocusReason.ShortcutFocusReason)
                return
        # No actionable row ahead: select the next row after the acted one,
        # clamped to the last row if start_row is now past the end.
        self._select_comparison_row(min(start_row, row_count - 1), Qt.FocusReason.ShortcutFocusReason)

    def _trigger_action_for_current_row(self) -> bool:
        row = self.comparison_table.currentRow()
        if row < 0:
            return False

        action_host = self.comparison_table.cellWidget(row, 1)
        if action_host is None:
            return False

        for button in action_host.findChildren(QPushButton):
            if button.isEnabled():
                # ✕ removes the row so rows below shift up by 1: scan from
                # start_row in the rebuilt table (it now points to the old
                # start_row+1).  ← / 🔍 keep the row so we skip past it.
                removes_row = button.text() == "✕"
                button.click()
                advance_from = row if removes_row else row + 1
                self._advance_to_next_actionable_from(advance_from)
                return True
        return False

    def _merge_proposed_row_from_table(self, table_row: int, right_index: int) -> None:
        self._remember_last_action_table_row(table_row)
        self._apply_proposed_rows([right_index])
        self._refresh_button_state()
        self._update_difference_highlights()

    def _remove_left_item_from_table(self, table_row: int, text: str) -> None:
        self._remember_last_action_table_row(table_row)
        self._remove_left_item(text)

    def _remove_right_rows(self, rows: list[int]) -> None:
        for row in sorted(rows, reverse=True):
            item = self.right_list.takeItem(row)
            del item

    def _update_merge_next_button_presentation(self) -> None:
        if self.next_button.isEnabled():
            self.merge_next_button.setText("Merge and Next")
            self.merge_next_button.setToolTip(
                "Apply current annotations and go to next item, even with no local edits (Alt+Enter)"
            )
            self.merge_and_next_alt_action.setText("Merge and Next")
        else:
            self.merge_next_button.setText("Merge")
            self.merge_next_button.setToolTip("Apply current annotations and resolve this fixup (Alt+Enter)")
            self.merge_and_next_alt_action.setText("Merge")

    def _refresh_button_state(self) -> None:
        has_right_items = self.right_list.count() > 0
        is_resolved = self._resolved
        has_local_changes = self._has_local_changes()
        has_dialog_changes = self._has_dialog_state_changes()
        regenerate_in_progress = self._regenerate_thread is not None
        self._update_merge_next_button_presentation()
        self.accept_button.setEnabled(has_right_items and not is_resolved and not regenerate_in_progress)
        self.merge_button.setEnabled(has_local_changes and not regenerate_in_progress)
        self.undo_button.setEnabled((self._undo_available or has_dialog_changes) and not regenerate_in_progress)
        self.merge_next_button.setEnabled(not regenerate_in_progress)

    def _has_local_changes(self) -> bool:
        current = [
            self._normalized_compare_text(text)
            for text in self.selected_annotations()
            if self._normalized_compare_text(text)
        ]
        last_merged = [
            self._normalized_compare_text(text)
            for text in self._last_merged_annotations
            if self._normalized_compare_text(text)
        ]
        return current != last_merged

    def _has_proposed_changes(self) -> bool:
        current = [
            self._normalized_compare_text(text)
            for text in self._current_texts(self.right_list)
            if self._normalized_compare_text(text)
        ]

        initial_values = []
        if self._normalized_compare_text(self._initial_proposed_description):
            initial_values.append(self._normalized_compare_text(self._initial_proposed_description))
        initial_values.extend(
            self._normalized_compare_text(text)
            for text in self._initial_proposed_tags
            if self._normalized_compare_text(text)
        )
        initial_values.extend(
            self._normalized_compare_text(text)
            for text in self._initial_search_matches
            if self._normalized_compare_text(text)
        )
        return current != initial_values

    def _has_dialog_state_changes(self) -> bool:
        return self._has_local_changes() or self._has_proposed_changes()

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
            # Skip search match items from being matched
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
                # Don't match description with search match items
                description_right_index = 0
                if description_right_index not in search_match_indexes:
                    left_matches[left_description_index] = 0
                    right_matches[0] = left_description_index
                    left_description_key = self._normalized_compare_key(left_texts[left_description_index])
                    right_description_key = self._normalized_compare_key(right_texts[0])
                    right_match_kind[0] = "exact" if left_description_key == right_description_key else "description"

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
                # Skip search match items from being matched
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

        # Identify search match items to exclude from matching
        search_match_indexes: set[int] = set()
        for right_index, item in enumerate(right_items):
            if item and item.data(IS_SEARCH_MATCH_ROLE):
                search_match_indexes.add(right_index)

        left_matches, right_matches, right_match_kind = self._compute_matches(left_texts, right_texts, search_match_indexes)

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

        self._render_comparison_table(left_texts, right_texts, left_matches, right_matches, right_match_kind)

    def _build_highlighted_html(self, text: str, ranges: list[tuple[int, int]]) -> str:
        if not text:
            return ""
        if not ranges:
            return ItemActionWidget._escape_html(text)

        html_parts: list[str] = []
        last_end = 0
        for start, end in sorted(ranges):
            if start > last_end:
                html_parts.append(ItemActionWidget._escape_html(text[last_end:start]))
            html_parts.append(
                f'<span style="background-color: #fff3b3;">{ItemActionWidget._escape_html(text[start:end])}</span>'
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

        if isinstance(right_widget, QLabel):
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
            if action_text == "✕":
                button.setStyleSheet("QPushButton { color: red; font-weight: bold; padding: 2px; }")
            button.clicked.connect(lambda _checked=False, cb=callback: cb())
            action_layout.addWidget(button)
        self.comparison_table.setCellWidget(row, 1, action_host)

    def _on_comparison_selection_changed(self) -> None:
        self._refresh_button_state()
        self._sync_comparison_widget_selection_styles()

    def _sync_comparison_widget_selection_styles(self) -> None:
        selected_rows = {index.row() for index in self.comparison_table.selectedIndexes()}
        for row in range(self.comparison_table.rowCount()):
            row_selected = row in selected_rows
            action_widget = self.comparison_table.cellWidget(row, 1)
            left_widget = self.comparison_table.cellWidget(row, 0)
            right_widget = self.comparison_table.cellWidget(row, 2)
            self._apply_comparison_row_widget_styles(row_selected, action_widget, left_widget, right_widget)

    def _update_action_column_metrics(self) -> None:
        metrics = self.comparison_table.fontMetrics()
        symbol_width = max(metrics.horizontalAdvance("←"), metrics.horizontalAdvance("✕"))
        self._action_button_width = max(24, symbol_width + 16)
        self._action_button_height = max(22, metrics.height() + 8)

        action_col_width = self._action_button_width + 10
        self.comparison_table.setColumnWidth(1, action_col_width)
        header_spacer = getattr(self, "comparison_header_action_spacer", None)
        if isinstance(header_spacer, QWidget):
            header_spacer.setFixedWidth(action_col_width)

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
                    flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                    if not self._is_protected_existing_text(left_text):
                        flags |= Qt.ItemFlag.ItemIsEditable
                    table_item.setFlags(flags)
                    table_item.setData(
                        DIFF_RANGES_ROLE,
                        left_ranges if isinstance(left_ranges, list) else [],
                    )
                    self.comparison_table.setItem(row, 0, table_item)
                else:
                    placeholder_item = QTableWidgetItem("")
                    placeholder_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                    placeholder_item.setData(DIFF_RANGES_ROLE, [])
                    self.comparison_table.setItem(row, 0, placeholder_item)

                if right_index is not None:
                    right_item = self.right_list.item(right_index)
                    right_text = right_texts[right_index]
                    right_ranges = right_item.data(DIFF_RANGES_ROLE)
                    right_widget = self._make_text_cell_label(
                        right_text,
                        right_ranges if isinstance(right_ranges, list) else [],
                    )
                    self.comparison_table.setCellWidget(row, 2, right_widget)
                else:
                    self.comparison_table.setCellWidget(row, 2, self._make_text_cell_label("", []))

                action_text = ""
                callback: Callable[[], None] | None = None
                
                # Check if right item is a search match and use magnifying glass
                if right_index is not None:
                    right_item = self.right_list.item(right_index)
                    is_search_match = right_item.data(IS_SEARCH_MATCH_ROLE) if right_item else False
                    
                    if is_search_match:
                        # For search matches, add magnifying glass button to add to current tags
                        action_text = "🔍"
                        right_text = right_texts[right_index]
                        callback = lambda idx=right_index, text=right_text: self._add_search_match_to_tags(text)
                    elif left_index is not None:
                        if right_match_kind.get(right_index) != "exact":
                            action_text = "←"
                            callback = lambda table_row=row, idx=right_index: self._merge_proposed_row_from_table(table_row, idx)
                    else:
                        # Right-only item (not exact match): use arrow to import
                        action_text = "←"
                        callback = lambda table_row=row, idx=right_index: self._merge_proposed_row_from_table(table_row, idx)
                elif left_index is not None:
                    left_text = left_texts[left_index]
                    if not self._is_protected_existing_text(left_text):
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
            self._refresh_button_state()
            saved_row = self.comparison_table.currentRow()
            self._update_difference_highlights()
            if saved_row >= 0:
                row_count = self.comparison_table.rowCount()
                if row_count > 0:
                    self._select_comparison_row(
                        min(saved_row, row_count - 1),
                        Qt.FocusReason.OtherFocusReason,
                    )

        QTimer.singleShot(0, _run_refresh)

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

    def accept_all(self) -> None:
        proposed_values: list[str] = []
        for index, value in enumerate(self._current_texts(self.right_list)):
            if not value.strip():
                continue
            normalized = self._normalize_proposed_text_for_merge(value, index)
            if normalized:
                proposed_values.append(normalized)
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

        self.left_list.clear()
        for value in merged_values:
            self._add_left_item(value)

        self._refresh_button_state()
        self._update_difference_highlights()

    def _move_item_from_right_to_left(self, text: str) -> None:
        """Move a specific item from right to left list."""
        normalized = text.strip()
        normalized_key = self._normalized_compare_key(normalized)
        target_row = -1
        # Find corresponding row in right list
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

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        comparison_table = getattr(self, "comparison_table", None)
        left_tag_input = getattr(self, "left_tag_input", None)
        if comparison_table is None or left_tag_input is None:
            return super().eventFilter(watched, event)

        if watched is comparison_table and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() == Qt.KeyboardModifier.NoModifier and self._trigger_action_for_current_row():
                    return True
            if event.key() == Qt.Key.Key_Left:
                selected_rows = sorted({index.row() for index in comparison_table.selectedIndexes()})
                right_indexes: list[int] = []
                for row in selected_rows:
                    if row < 0 or row >= len(self._table_row_map):
                        continue
                    _left_index, right_index = self._table_row_map[row]
                    if right_index is None:
                        continue
                    cell_widget = comparison_table.cellWidget(row, 1)
                    if cell_widget is not None:
                        for btn in cell_widget.findChildren(QPushButton):
                            if btn.text() == "←":
                                right_indexes.append(right_index)
                                break
                if right_indexes:
                    current_row = comparison_table.currentRow()
                    self._remember_last_action_table_row(current_row)
                    self._apply_proposed_rows(right_indexes)
                    self._refresh_button_state()
                    self._update_difference_highlights()
                    new_row_count = comparison_table.rowCount()
                    if new_row_count > 0 and current_row >= 0:
                        target_row = min(current_row, new_row_count - 1)
                        self._select_comparison_row(target_row, Qt.FocusReason.ShortcutFocusReason)
                    return True
            if event.key() == Qt.Key.Key_Home and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                return self._activate_first_comparison_row()
            if event.key() == Qt.Key.Key_End and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                return self._activate_last_comparison_row()

        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            if event.modifiers() != Qt.KeyboardModifier.NoModifier:
                return super().eventFilter(watched, event)

            step = -1 if event.key() == Qt.Key.Key_Up else 1
            left_input_empty = not left_tag_input.text().strip()

            focused = self.focusWidget()
            if focused is None:
                if self._activate_adjacent_row_for_last_action(step):
                    return True
                if event.key() == Qt.Key.Key_Down and self._activate_first_comparison_row():
                    return True
                return super().eventFilter(watched, event)

            if focused is left_tag_input:
                if left_input_empty and self._activate_adjacent_row_for_last_action(step):
                    return True
                if event.key() == Qt.Key.Key_Down and left_input_empty and self._activate_first_comparison_row():
                    return True
                return super().eventFilter(watched, event)

            if focused is comparison_table or comparison_table.isAncestorOf(focused):
                return super().eventFilter(watched, event)

            if isinstance(focused, (QLineEdit, QTextEdit, QAbstractItemView, QPushButton, QCheckBox)):
                return super().eventFilter(watched, event)

            if self.isAncestorOf(focused):
                if self._activate_adjacent_row_for_last_action(step):
                    return True
                if event.key() == Qt.Key.Key_Down and self._activate_first_comparison_row():
                    return True

        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Home, Qt.Key.Key_End):
            if event.modifiers() != Qt.KeyboardModifier.NoModifier:
                return super().eventFilter(watched, event)

            activate_row = self._activate_first_comparison_row if event.key() == Qt.Key.Key_Home else self._activate_last_comparison_row
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

            if isinstance(focused, (QLineEdit, QTextEdit, QAbstractItemView, QPushButton, QCheckBox)):
                return super().eventFilter(watched, event)

            if self.isAncestorOf(focused):
                if activate_row():
                    return True

        return super().eventFilter(watched, event)

    def _remove_selected_left_items(self) -> None:
        selected_rows = sorted({index.row() for index in self.comparison_table.selectedIndexes()}, reverse=True)
        if not selected_rows:
            return

        # Remember the topmost selected row so we can restore focus after rebuild.
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
            item = self.left_list.item(left_index)
            item_text = item.text()
            if self._is_protected_existing_text(item_text):
                continue
            removed = self.left_list.takeItem(left_index)
            del removed
            removed_any = True

        if not removed_any:
            return

        self._refresh_button_state()
        self._update_difference_highlights()

        # After the table is rebuilt, activate the row that followed the deleted
        # selection (or the new last row if everything deleted was at the end).
        new_row_count = self.comparison_table.rowCount()
        if new_row_count > 0:
            target_row = min(min_selected_row, new_row_count - 1)
            self.comparison_table.setCurrentCell(target_row, 0)

    def _activate_first_comparison_row(self) -> bool:
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return False

        return self._select_comparison_row(0, Qt.FocusReason.ShortcutFocusReason)

    def selected_annotations(self) -> list[str]:
        return self._current_texts(self.left_list)

    def _regenerate_timeout_seconds(self) -> float:
        raw_value = self.regenerate_timeout_input.text().strip()
        if not raw_value:
            QMessageBox.warning(self, "Invalid timeout", "Enter timeout in seconds.")
            raise OllamaError("Enter timeout in seconds.")

        try:
            timeout = int(raw_value)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid timeout", "Timeout must be a whole number of seconds.")
            raise OllamaError("Timeout must be a whole number of seconds.") from exc

        if timeout < 1:
            QMessageBox.warning(self, "Invalid timeout", "Timeout must be at least 1 second.")
            raise OllamaError("Timeout must be at least 1 second.")

        return float(timeout)

    def _regenerate_retry_count(self) -> int:
        raw_value = self.regenerate_retry_input.text().strip()
        try:
            retries = int(raw_value)
        except ValueError:
            return 0
        return max(0, retries)

    @staticmethod
    def _format_mpx(value: float) -> str:
        normalized = f"{float(value):.3f}".rstrip("0").rstrip(".")
        if "." not in normalized:
            normalized += ".0"
        return normalized

    def _regenerate_max_resolution_mpx(self) -> float:
        raw_value = self.regenerate_max_resolution_input.text().strip()
        if not raw_value:
            QMessageBox.warning(self, "Invalid query downscale", "Enter query downscale in megapixels.")
            raise OllamaError("Enter query downscale in megapixels.")

        try:
            value = float(raw_value)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid query downscale", "Query downscale must be a number.")
            raise OllamaError("Query downscale must be a number.") from exc

        if value <= 0:
            QMessageBox.warning(self, "Invalid query downscale", "Query downscale must be greater than 0.")
            raise OllamaError("Query downscale must be greater than 0.")

        return value

    def _update_regenerate_controls(self) -> None:
        connected = bool(self._ollama_model_name)
        working = self._regenerate_thread is not None
        options_selected = (
            self.regenerate_tags_checkbox.isChecked()
            or self.regenerate_description_checkbox.isChecked()
        )

        self.regenerate_tags_checkbox.setEnabled(not working)
        self.regenerate_description_checkbox.setEnabled(not working)
        self.regenerate_timeout_input.setEnabled(not working)
        self.regenerate_retry_input.setEnabled(not working)
        self.regenerate_max_resolution_input.setEnabled(not working)

        if working:
            self.regenerate_button.setEnabled(False)
            self.regenerate_button.setText("Regenerating...")
            return

        self.regenerate_button.setText("&Regenerate")
        self.regenerate_button.setEnabled(connected and options_selected and self._image_path is not None)
        if not connected:
            self.regenerate_status_label.setText("Regenerate is disabled until an Ollama model is connected in the main window.")

    def _set_proposed_annotations(
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
            self._protected_existing_keys = self._find_description_like_keys(self._current_texts(self.left_list))
        else:
            self._protected_existing_keys = set()

        if normalized_description:
            self._add_right_item(normalized_description)
        for tag in normalized_tags:
            self._add_right_item(tag)

        self._refresh_button_state()
        self._update_difference_highlights()

    def _restore_initial_proposed_annotations(self) -> None:
        self.right_list.clear()
        self._exact_match_only_for_tags = False
        self._has_proposed_description = bool(self._initial_proposed_description)
        if not self._has_proposed_description:
            self._protected_existing_keys = self._find_description_like_keys(self._initial_annotations)
        else:
            self._protected_existing_keys = set()

        if self._initial_proposed_description:
            self._add_right_item(self._initial_proposed_description)
        for tag in self._initial_proposed_tags:
            self._add_right_item(tag)
        for search_query in self._initial_search_matches:
            self._add_search_match_item(search_query)

    def _create_regenerate_frame(self) -> QWidget:
        frame = QFrame(self)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Sunken)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.addWidget(self.regenerate_tags_checkbox)
        controls_row.addWidget(self.regenerate_description_checkbox)
        controls_row.addSpacing(12)
        controls_row.addWidget(QLabel("Timeout", self))
        controls_row.addWidget(self.regenerate_timeout_input)
        controls_row.addSpacing(8)
        controls_row.addWidget(QLabel("Retries", self))
        controls_row.addWidget(self.regenerate_retry_input)
        controls_row.addSpacing(8)
        controls_row.addWidget(QLabel("Query downscale", self))
        controls_row.addWidget(self.regenerate_max_resolution_input)
        controls_row.addWidget(QLabel("(MPx)", self))
        controls_row.addStretch(1)

        layout.addLayout(controls_row)
        layout.addWidget(self.regenerate_button, stretch=0)
        layout.addWidget(self.regenerate_status_label, stretch=0)
        return frame

    def _regenerate_proposed_annotations(self) -> None:
        if self._regenerate_thread is not None:
            return
        if self._image_path is None:
            return
        if not self.regenerate_tags_checkbox.isChecked() and not self.regenerate_description_checkbox.isChecked():
            QMessageBox.information(self, "Nothing selected", "Enable Tags or Description before regenerating.")
            return
        if not self._ollama_model_name:
            return

        try:
            max_resolution_mpx = self._regenerate_max_resolution_mpx()
        except OllamaError:
            return
        configure_runtime(max_image_pixels=max(1, int(max_resolution_mpx * 1_000_000)))

        resize_warning = consume_resize_warning()
        if resize_warning:
            QMessageBox.warning(self, "Image resize disabled", resize_warning)

        cancel_token = OllamaCancellation()
        self._regenerate_cancel = cancel_token
        self._discard_regenerate_result = False

        def task(report_progress: Callable[[str], None]) -> object:
            timeout = self._regenerate_timeout_seconds()
            retry_count = self._regenerate_retry_count()
            image_name = self._image_path.name
            last_error: OllamaError | None = None

            for attempt in range(retry_count + 1):
                cancel_token.raise_if_cancelled()
                attempt_start = time.monotonic()
                if attempt == 0:
                    report_progress(f"Regenerating {image_name}...")
                else:
                    report_progress(f"Regenerating {image_name} (retry {attempt}/{retry_count})...")

                def remaining_timeout() -> float:
                    elapsed = time.monotonic() - attempt_start
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise OllamaError(
                            f"Timed out after {int(timeout)} seconds while regenerating annotations for {image_name}."
                        )
                    return remaining

                try:
                    description = ""
                    tags: list[str] = []
                    if self.regenerate_description_checkbox.isChecked():
                        description = self._normalize_annotation(
                            generate_description(
                                self._ollama_server_url,
                                self._ollama_model_name,
                                self._image_path,
                                timeout=remaining_timeout(),
                                cancellation=cancel_token,
                            ).strip()
                        )
                    if self.regenerate_tags_checkbox.isChecked():
                        tags = self._dedupe_preserve_order(
                            self._parse_regenerated_tags(
                                generate_tags(
                                    self._ollama_server_url,
                                    self._ollama_model_name,
                                    self._image_path,
                                    timeout=remaining_timeout(),
                                    cancellation=cancel_token,
                                )
                            )
                        )

                    if description or tags:
                        return {"description": description, "tags": tags}
                    last_error = OllamaError("Ollama returned no annotations.")
                except OllamaCancelled:
                    raise
                except OllamaError as exc:
                    last_error = exc

            if last_error is not None:
                raise last_error
            raise OllamaError("Ollama returned no annotations.")

        self._regenerate_thread = QThread(self)
        self._regenerate_worker = RegenerateWorker(task)
        self._regenerate_worker.moveToThread(self._regenerate_thread)
        self._regenerate_thread.started.connect(self._regenerate_worker.run)
        self._regenerate_worker.progress.connect(self.regenerate_status_label.setText)
        self._regenerate_worker.finished.connect(self._on_regenerate_finished)
        self._regenerate_worker.cancelled.connect(self._on_regenerate_cancelled)
        self._regenerate_worker.failed.connect(self._on_regenerate_failed)
        self._regenerate_worker.finished.connect(self._regenerate_thread.quit)
        self._regenerate_worker.cancelled.connect(self._regenerate_thread.quit)
        self._regenerate_worker.failed.connect(self._regenerate_thread.quit)
        self._regenerate_thread.finished.connect(self._cleanup_regenerate_task)
        self._update_regenerate_controls()
        self._regenerate_thread.start()

    def _parse_regenerated_tags(self, text: str) -> list[str]:
        normalized = text.replace("\r", "").replace("\n", ",")
        tags: list[str] = []
        for part in normalized.split(","):
            cleaned = self._normalize_annotation(part).strip()
            if cleaned:
                tags.append(cleaned.lower())
        return tags

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        unique_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = self._normalized_compare_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            unique_values.append(value.strip())
        return unique_values

    def _on_regenerate_finished(self, payload: object) -> None:
        if self._discard_regenerate_result:
            self.regenerate_status_label.setText("Regeneration discarded.")
            return

        data = payload if isinstance(payload, dict) else {}
        description = str(data.get("description", "")).strip()
        raw_tags = data.get("tags")
        tags = [str(tag).strip() for tag in raw_tags] if isinstance(raw_tags, list) else []
        tags = [tag for tag in tags if tag]

        current_proposed = [text.strip() for text in self._current_texts(self.right_list) if text.strip()]
        current_description = ""
        current_tags: list[str] = []
        if self._has_proposed_description and current_proposed:
            current_description = current_proposed[0]
            current_tags = current_proposed[1:]
        else:
            current_tags = current_proposed

        regenerate_description = self.regenerate_description_checkbox.isChecked()
        regenerate_tags = self.regenerate_tags_checkbox.isChecked()

        final_description = description if regenerate_description else current_description
        if regenerate_tags:
            existing_keys = {
                self._normalized_compare_key(tag)
                for tag in current_tags
                if self._normalized_compare_key(tag)
            }
            appended_tags: list[str] = []
            for tag in tags:
                key = self._normalized_compare_key(tag)
                if not key or key in existing_keys:
                    continue
                existing_keys.add(key)
                appended_tags.append(tag)
            final_tags = current_tags + appended_tags
        else:
            final_tags = current_tags
        exact_only_for_tags = True if regenerate_tags else self._exact_match_only_for_tags

        self._set_proposed_annotations(
            final_description,
            final_tags,
            exact_match_only_for_tags=exact_only_for_tags,
        )
        if description or tags:
            self.regenerate_status_label.setText("Regenerated proposed annotations.")
        else:
            self.regenerate_status_label.setText("Ollama returned no annotations.")

    def _on_regenerate_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Regenerate failed", message)
        self.regenerate_status_label.setText("Regenerate failed.")

    def _on_regenerate_cancelled(self, message: str) -> None:
        if self._discard_regenerate_result:
            self.regenerate_status_label.setText("Regeneration discarded.")
            return
        self.regenerate_status_label.setText(message or "Regeneration stopped.")

    def _cancel_regenerate(self, *, discard_result: bool = True) -> None:
        if discard_result:
            self._discard_regenerate_result = True
        if self._regenerate_cancel is not None:
            self._regenerate_cancel.cancel()

    def _cleanup_regenerate_task(self) -> None:
        if self._regenerate_worker is not None:
            self._regenerate_worker.deleteLater()
        if self._regenerate_thread is not None:
            self._regenerate_thread.deleteLater()
        self._regenerate_worker = None
        self._regenerate_thread = None
        self._regenerate_cancel = None
        self._update_regenerate_controls()
        self._refresh_button_state()

    def _create_image_pane(self, image_path: Path | None) -> QWidget:
        pane = QWidget(self)
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.addWidget(self._create_pane_header_label("Image"))
        pane_layout.addSpacing(self._PANE_HEADER_BOTTOM_SPACING)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        image_label = ScalableImageLabel(self)
        image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        image_label.customContextMenuRequested.connect(self._show_image_context_menu)

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
        pane_layout.addWidget(self._create_regenerate_frame(), stretch=0)
        return pane

    def _create_pane_header_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setContentsMargins(0, 0, 0, 0)
        label.setMinimumHeight(label.fontMetrics().height() + self._PANE_HEADER_EXTRA_HEIGHT)
        return label

    def _show_image_context_menu(self, position) -> None:
        image_path = self._image_path
        if image_path is None:
            return

        menu = QMenu(self)
        open_default_action = menu.addAction("Open in Default App")
        open_default_action.triggered.connect(self._open_image_in_default_app)

        open_with_menu = menu.addMenu("Open With")
        editors = self._get_detected_external_editors(refresh=False)
        if editors:
            for editor in editors:
                action = open_with_menu.addAction(editor.display_name)
                action.triggered.connect(
                    lambda _checked=False, selected_editor=editor: self._open_image_with_editor(selected_editor)
                )
        else:
            unavailable = open_with_menu.addAction("No common editors detected")
            unavailable.setEnabled(False)

        open_with_menu.addSeparator()
        choose_action = open_with_menu.addAction("Choose executable...")
        choose_action.triggered.connect(self._open_image_with_custom_editor)

        source_widget = self.sender()
        if isinstance(source_widget, QWidget):
            global_position = source_widget.mapToGlobal(position)
        else:
            global_position = self.mapToGlobal(position)
        menu.exec(global_position)

    def _open_image_in_default_app(self) -> None:
        image_path = self._image_path
        if image_path is None:
            return

        try:
            launch_image_in_system_default(image_path)
        except OSError as exc:
            QMessageBox.warning(self, "Open image failed", f"Could not open image in default app:\n{exc}")
            return
        except Exception as exc:
            QMessageBox.warning(self, "Open image failed", f"Could not open image in default app:\n{exc}")
            return

        self.regenerate_status_label.setText(f"Opened {image_path.name} in default app")

    def _open_image_with_editor(self, editor: ExternalEditor) -> None:
        image_path = self._image_path
        if image_path is None:
            return

        try:
            launch_image_in_editor(editor, image_path)
        except OSError as exc:
            QMessageBox.warning(self, "Open editor failed", f"Could not open image with {editor.display_name}:\n{exc}")
            return
        except Exception as exc:
            QMessageBox.warning(self, "Open editor failed", f"Could not open image with {editor.display_name}:\n{exc}")
            return

        self.regenerate_status_label.setText(f"Opened {image_path.name} with {editor.display_name}")

    def _open_image_with_custom_editor(self) -> None:
        if self._image_path is None:
            return

        if sys.platform.startswith("win"):
            file_filter = "Applications (*.exe);;All files (*)"
        elif sys.platform == "darwin":
            file_filter = "Applications (*.app);;All files (*)"
        else:
            file_filter = "All files (*)"

        selected_path, _ = QFileDialog.getOpenFileName(self, "Choose graphics editor", "", file_filter)
        if not selected_path:
            return

        selected = Path(selected_path)
        if not selected.exists():
            QMessageBox.warning(self, "Editor not found", "Selected editor path does not exist.")
            return

        launch_kind = "mac_app" if sys.platform == "darwin" and selected.suffix.lower() == ".app" else "executable"
        custom_editor = ExternalEditor(
            id="custom",
            display_name=selected.stem or selected.name,
            launch_target=str(selected),
            launch_kind=launch_kind,
        )
        self._open_image_with_editor(custom_editor)

    def _get_detected_external_editors(self, refresh: bool = False) -> list[ExternalEditor]:
        if not refresh and self._detected_external_editors is not None:
            return list(self._detected_external_editors)

        try:
            editors = discover_graphics_editors()
        except Exception:
            editors = []
        self._detected_external_editors = editors
        return list(editors)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._global_key_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
                self._global_key_filter_installed = True
        self._update_action_column_metrics()
        self._update_difference_highlights()
        self.comparison_table.resizeRowsToContents()
        self._refresh_widget_item_sizes()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.comparison_table.resizeRowsToContents()
        self._refresh_widget_item_sizes()

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.ApplicationFontChange):
            self._update_action_column_metrics()
            self._update_difference_highlights()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._global_key_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._global_key_filter_installed = False
        self._cancel_regenerate(discard_result=True)
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
        self.left_list.clear()
        for text in self._initial_annotations:
            self._add_left_item(text)

        self._restore_initial_proposed_annotations()

        self._refresh_button_state()
        self._update_difference_highlights()

    def _merge_without_close(self) -> bool:
        if self._apply_annotations is None:
            return False
        current_annotations = list(self.selected_annotations())
        self._apply_annotations(current_annotations, "Fixup merged + auto-saved")
        self._last_merged_annotations = current_annotations
        self._undo_available = True
        return self._resolve_fixup()

    def _accept_all_without_close(self) -> bool:
        self.accept_all()
        return self._merge_without_close()

    def _undo_merge(self) -> None:
        if not self._undo_available and not self._has_dialog_state_changes():
            return

        if self._undo_available:
            if self._apply_annotations is None:
                return
            if self._restore_fixup is None or not self._restore_fixup(self._initial_fixup_content):
                return
            self._apply_annotations(list(self._initial_annotations), "Fixup undone + auto-saved")

        self._restore_initial_state()
        self._last_merged_annotations = list(self._initial_annotations)
        self._resolved = False
        self._undo_available = False
        self.regenerate_status_label.setText("Restored original fixup state.")
        self._refresh_button_state()

    def _navigate_prev(self) -> None:
        self._cancel_regenerate(discard_result=True)
        self.done(self.NAVIGATE_PREV_CODE)

    def _navigate_next(self) -> None:
        self._cancel_regenerate(discard_result=True)
        self.done(self.NAVIGATE_NEXT_CODE)

    def _merge_and_next(self) -> None:
        if self._merge_without_close() and self.next_button.isEnabled():
            self._navigate_next()