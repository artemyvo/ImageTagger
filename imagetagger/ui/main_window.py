from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageCms, UnidentifiedImageError

from PyQt6.QtCore import QEvent, QModelIndex, QObject, QRect, QStringListModel, QThread, Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QImage, QImageReader, QKeySequence, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QStyle,
    QStyledItemDelegate,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imagetagger import config as _config
from imagetagger.utils.annotations import parse_tags_text, sanitize_annotation_text, sanitize_description_text, sanitize_tag_text
from imagetagger.utils.image_prep import configure_image_preparation, consume_image_preparation_warning
from imagetagger.ui.image_reload_helper import ImageReloadHelper
from imagetagger.utils.input_validators import InputValidator
from imagetagger.utils.validators import (
    create_max_resolution_validator,
    create_retry_validator,
    create_threads_validator,
    create_timeout_validator,
)
from imagetagger.utils.io_utils import atomic_write_text
from imagetagger.utils.sidecar import SidecarData, get_sidecar_json_path, read_sidecar_data, write_sidecar_data
from imagetagger.providers.llm_provider import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_VISION_PROVIDER,
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
    VisionLlmSession,
)
from imagetagger.ui.merge_actions import (
    clear_fixup_sidecar,
    clear_validation_fields_sidecar,
    delete_sidecar_for_image,
    open_fixup_dialog_for_image,
    record_ai_find_match_for_image,
    record_refine_result_for_image,
    write_fixup_sidecar,
)
from imagetagger.utils.external_editors import (
    ExternalEditor,
    discover_graphics_editors,
    launch_image_in_editor,
    launch_image_in_system_default,
)
from imagetagger.utils.llm_queries import (
    active_prompt_for_kind,
    clear_prompt_override,
    format_annotations_for_validation,
    get_default_prompt,
    load_prompt_for_kind,
    parse_refine_response,
    parse_vision_response,
    parse_yes_no_response,
    prepare_description_query,
    prepare_refine_query,
    prepare_search_query,
    prepare_tagging_query,
    prepare_validation_query,
    prepare_vision_query,
    prompt_source_for_kind,
    reset_prompt_to_default,
    save_prompt_for_kind,
    set_prompt_override,
)
from imagetagger.ui.workers import (
    IMAGE_EXTENSIONS,
    THUMB_SIZE,
    MIN_FONT_POINT_SIZE,
    MAX_FONT_POINT_SIZE,
    FolderLoadWorker,
    LlmTaskWorker,
    RegenerateWorker,
    TagPurgeWorker,
)
from imagetagger.ui.shortcuts import platform_key_sequence
from imagetagger.ui.server_settings_frame import create_server_settings_frame
from imagetagger.utils.theme_colors import danger_accent_color, danger_text_on_accent_color, info_accent_color, info_text_on_accent_color, success_accent_color, success_text_on_accent_color





from imagetagger.utils.filter_parser import (
    FilterSyntaxError,
    _FilterNode,
    _FilterRuntime,
    _parse_filter_expression,
)
from imagetagger.ui.models import ImageRecord




class TagListWidget(QListWidget):
    tags_reordered = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        super().dropEvent(event)
        self.tags_reordered.emit()


class GlobalTagListWidget(QListWidget):
    delete_requested = pyqtSignal(list)  # list[str]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            tags = [
                item.data(Qt.ItemDataRole.UserRole)
                for item in self.selectedItems()
                if item.data(Qt.ItemDataRole.UserRole)
            ]
            if tags:
                self.delete_requested.emit(tags)
            return
        super().keyPressEvent(event)


class WrappedTagItemDelegate(QStyledItemDelegate):
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
                    self.commitData.emit(editor)
                    self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
                    return True
                if key == Qt.Key.Key_Escape:
                    self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.RevertModelCache)
                    return True
            if event.type() == QEvent.Type.FocusOut:
                self.commitData.emit(editor)
                self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
        return super().eventFilter(editor, event)


@dataclass(frozen=True)
class _ListItemBadgeSpec:
    text: str
    background: QColor
    foreground: QColor


# Item data roles used by _ImageRowDelegate.
_ROLE_PIXMAP = Qt.ItemDataRole.UserRole          # QPixmap | None — pre-scaled thumbnail
_ROLE_BADGES = Qt.ItemDataRole.UserRole + 1      # frozenset[str] — active badge symbols


class _ImageRowDelegate(QStyledItemDelegate):
    """Paints image list rows directly with QPainter — no per-row widget trees.

    This replaces the previous setItemWidget approach which created ~8 QObjects
    per row and became prohibitively expensive at large folder sizes (≥8 k images).
    """

    _BADGE_COL_W = 64
    _LEFT_MARGIN = 4
    _RIGHT_MARGIN = 6
    _V_MARGIN = 3
    _SPACING = 8

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window.list_widget)
        self._window = window

    def sizeHint(self, option: object, index: QModelIndex) -> QSize:  # type: ignore[override]
        return QSize(0, THUMB_SIZE.height() + 10)

    def paint(self, painter: QPainter, option: object, index: QModelIndex) -> None:  # type: ignore[override]
        from PyQt6.QtWidgets import QStyleOptionViewItem  # local import avoids circular
        opt = option  # type: ignore[assignment]
        painter.save()

        is_selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        if is_selected:
            painter.fillRect(opt.rect, opt.palette.highlight())
            text_color = opt.palette.highlightedText().color()
        else:
            painter.fillRect(opt.rect, opt.palette.base())
            text_color = opt.palette.text().color()

        x = opt.rect.left() + self._LEFT_MARGIN
        row_h = opt.rect.height()

        # Thumbnail (pre-scaled; no per-paint scaling needed).
        pixmap: QPixmap | None = index.data(_ROLE_PIXMAP)
        if pixmap is not None and not pixmap.isNull():
            py = opt.rect.top() + max(0, (row_h - pixmap.height()) // 2)
            painter.drawPixmap(x, py, pixmap)
        x += THUMB_SIZE.width() + self._SPACING

        # Badges.
        active_badges: frozenset[str] | None = index.data(_ROLE_BADGES)
        w = self._window
        if active_badges:
            # Ensure palette color cache is warm (cheap if already cached).
            if w._cached_danger_color is None:
                w._badge_specs_from_active_set(frozenset())
            danger_col: QColor = w._cached_danger_color  # type: ignore[assignment]
            info_col: QColor = w._cached_info_color      # type: ignore[assignment]
            success_col: QColor = w._cached_success_color  # type: ignore[assignment]
            slot_count = len(w._IMAGE_ROW_BADGE_SLOT_ORDER)
            slot_h = row_h / slot_count
            for i, symbol in enumerate(w._IMAGE_ROW_BADGE_SLOT_ORDER):
                if symbol in active_badges:
                    if symbol == "\u2696\ufe0f":
                        color = danger_col
                    elif symbol == "\u2705":
                        color = success_col
                    else:
                        color = info_col
                    badge_rect = QRect(
                        x,
                        opt.rect.top() + round(i * slot_h),
                        self._BADGE_COL_W,
                        round(slot_h),
                    )
                    painter.setPen(color)
                    painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, symbol)
        x += self._BADGE_COL_W + self._SPACING

        # Title text.
        title = index.data(Qt.ItemDataRole.DisplayRole) or ""
        text_rect = QRect(
            x,
            opt.rect.top() + self._V_MARGIN,
            opt.rect.right() - self._RIGHT_MARGIN - x,
            row_h - 2 * self._V_MARGIN,
        )
        painter.setPen(text_color)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            title,
        )

        painter.restore()

    def helpEvent(  # type: ignore[override]
        self,
        event: object,
        view: object,
        option: object,
        index: QModelIndex,
    ) -> bool:
        """Lazily build the item tooltip on first hover so load time is unaffected."""
        from PyQt6.QtCore import QEvent as _QEvent
        from PyQt6.QtWidgets import QAbstractItemView as _QAbstractItemView
        ev = event  # type: ignore[assignment]
        lw = view   # type: ignore[assignment]
        if ev.type() == _QEvent.Type.ToolTip:
            item = lw.item(index.row())
            if item is not None and not item.toolTip():
                row = index.row()
                window = self._window
                if 0 <= row < len(window.records):
                    record = window.records[row]
                    active: frozenset[str] = index.data(_ROLE_BADGES) or frozenset()
                    badge_specs = window._badge_specs_from_active_set(active)
                    item.setToolTip(window._build_list_item_tooltip(record, badge_specs=badge_specs))
        return super().helpEvent(ev, lw, option, index)  # type: ignore[arg-type]


class MainWindow(QMainWindow):
    _IMAGE_ROW_BADGE_SLOT_ORDER = ("⚖️", "✨", "🔍", "✅")

    def __init__(self) -> None:
        super().__init__()
        self.resize(1400, 860)

        self.records: List[ImageRecord] = []
        self.current_index: int = -1
        self._loader_thread: QThread | None = None
        self._loader_worker: FolderLoadWorker | None = None
        self._llm_provider = DEFAULT_VISION_PROVIDER
        self._llm_thread: QThread | None = None
        self._llm_worker: LlmTaskWorker | None = None
        self.known_tags: set[str] = set()
        self.tag_counts: Counter[str] = Counter()
        self._updating_tag_list = False
        self._ignore_selection_sync = False
        self._generate_batch_total = 0
        self._generate_batch_processed = 0
        self._generate_batch_updated = 0
        self._generate_batch_vision_updated = 0
        self._generate_batch_new_annotations = 0
        self._generate_batch_started_at: float | None = None
        self._generate_batch_retry_images = 0
        self._validate_batch_total = 0
        self._validate_batch_processed = 0
        self._validate_batch_clean = 0
        self._validate_batch_issues = 0
        self._validate_batch_skipped = 0
        self._validate_batch_started_at: float | None = None
        self._validate_batch_retry_images = 0
        self._validate_batch_llm_disobeyed = 0
        self._validate_pending_indices: set[int] = set()
        self._ai_find_batch_total = 0
        self._ai_find_batch_processed = 0
        self._ai_find_batch_matched = 0
        self._ai_find_batch_started_at: float | None = None
        self._ai_find_batch_retry_images = 0
        self._llm_action_name: str | None = None
        self._llm_cancel: LlmRequestCancellation | None = None
        self._llm_threads_auto_mode = False
        self._llm_threads_current = 0
        self._icc_warning_paths: list[str] = []
        self._right_splitter_initialized = False
        self.prompt_editors: dict[str, QTextEdit] = {}
        self.prompt_status_labels: dict[str, QLabel] = {}
        self.prompt_role_inputs: dict[str, QLineEdit] = {}
        self._vision_updating = False
        self._vision_loaded_description = ""
        self._vision_loaded_reasoning = ""

        self.open_action: QAction | None = None
        self.refresh_action: QAction | None = None
        self.save_action: QAction | None = None
        self.increase_font_action: QAction | None = None
        self.decrease_font_action: QAction | None = None
        self.llm_endpoint = self._llm_provider.default_endpoint
        self.llm_model_name = ""
        self.status_connection_label: QLabel | None = None
        self._pending_selection_path: Path | None = None
        self._root_directory: Path | None = None
        self._detected_external_editors: list[ExternalEditor] | None = None
        self._directory_loading_active = False
        self._directory_load_cancel_requested = False
        self._directory_load_cancelled_by_user = False

        # Image reload detection for external editor changes
        self._image_reload_helper = ImageReloadHelper(self, self._on_image_reload)
        self._prev_selected_rows: set[int] = set()

        # Cached full-resolution pixmap for the currently displayed image.
        # Avoids re-reading from disk on resize events and rapid re-selections.
        self._cached_pixmap: QPixmap | None = None
        self._cached_pixmap_path: Path | None = None

        # Cached palette-derived badge colors; invalidated on PaletteChange.
        self._cached_danger_color: QColor | None = None
        self._cached_danger_fg_color: QColor | None = None
        self._cached_info_color: QColor | None = None
        self._cached_info_fg_color: QColor | None = None
        self._cached_success_color: QColor | None = None
        self._cached_success_fg_color: QColor | None = None

        # Debounce timer for resize events to avoid re-scaling on every pixel.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(60)  # ms
        self._resize_timer.timeout.connect(self._on_resize_timer)

        self._cfg = _config.load()

        self._build_ui()
        self._build_menu()
        self._apply_tag_list_height()
        self._apply_config()
        self._update_llm_controls()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.currentRowChanged.connect(self.on_selection_changed)
        self.list_widget.setItemDelegate(_ImageRowDelegate(self))

        self.filter_input = QLineEdit(self)
        self.filter_input.setPlaceholderText("Filter (fixup, vision, \"tag\", 'text', &, |, parentheses)")
        self.filter_input.textChanged.connect(self._apply_image_filter)

        self.filter_help_button = QPushButton("?", self)
        self.filter_help_button.setFixedWidth(26)
        self.filter_help_button.setToolTip("Show filter syntax help")
        self.filter_help_button.clicked.connect(self._show_filter_rules_dialog)

        filter_row = QWidget(self)
        filter_row_layout = QHBoxLayout(filter_row)
        filter_row_layout.setContentsMargins(0, 0, 0, 0)
        filter_row_layout.setSpacing(6)
        filter_row_layout.addWidget(self.filter_input, stretch=1)
        filter_row_layout.addWidget(self.filter_help_button, stretch=0)

        self.left_panel = QWidget(self)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(filter_row, stretch=0)
        left_layout.addWidget(self.list_widget, stretch=1)

        self.select_all_images_action = QAction("Select All Images", self)
        self.select_all_images_action.setShortcuts(
            QKeySequence.keyBindings(QKeySequence.StandardKey.SelectAll)
        )
        self.select_all_images_action.triggered.connect(self._select_all_images)
        self.list_widget.addAction(self.select_all_images_action)

        self.jump_first_fixup_action = QAction("Jump to First Fixup", self)
        self.jump_first_fixup_action.setShortcut(platform_key_sequence("Alt+F", "Alt+F"))
        self.jump_first_fixup_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.jump_first_fixup_action.triggered.connect(self._jump_to_first_fixup)
        self.addAction(self.jump_first_fixup_action)

        self.jump_last_fixup_action = QAction("Jump to Last Fixup", self)
        self.jump_last_fixup_action.setShortcut(platform_key_sequence("Alt+L", "Alt+L"))
        self.jump_last_fixup_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.jump_last_fixup_action.triggered.connect(self._jump_to_last_fixup)
        self.addAction(self.jump_last_fixup_action)

        self.select_prev_image_action = QAction("Select Previous Image", self)
        self.select_prev_image_action.setShortcut(platform_key_sequence("Alt+Up", "Alt+Up"))
        self.select_prev_image_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.select_prev_image_action.triggered.connect(lambda: self._move_image_selection(-1))
        self.addAction(self.select_prev_image_action)

        self.select_next_image_action = QAction("Select Next Image", self)
        self.select_next_image_action.setShortcut(platform_key_sequence("Alt+Down", "Alt+Down"))
        self.select_next_image_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.select_next_image_action.triggered.connect(lambda: self._move_image_selection(1))
        self.addAction(self.select_next_image_action)

        self.center_panel = QWidget(self)
        center_layout = QVBoxLayout(self.center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("No image selected", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumWidth(500)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.image_label.setStyleSheet(
            "background-color: palette(base); color: palette(text); border: 1px solid palette(mid);"
        )
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self._show_image_context_menu)
        center_layout.addWidget(self.image_label)

        self.right_panel = QWidget(self)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Top-right panel tabs: Tags + Vision
        self.tag_tabs = QTabWidget(self)

        tags_panel = QWidget(self)
        tags_layout = QVBoxLayout(tags_panel)
        tags_layout.setContentsMargins(6, 6, 6, 6)
        tags_layout.setSpacing(6)

        self.tag_input = QLineEdit(self)
        self.tag_input.setPlaceholderText("Type a tag and press Enter")
        self.tag_input.returnPressed.connect(self._add_tag_from_input)
        self.tag_input.installEventFilter(self)

        self.tag_suggestions_model = QStringListModel(self)
        self.tag_completer = QCompleter(self.tag_suggestions_model, self)
        self.tag_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.tag_completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)
        self.tag_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.tag_input.setCompleter(self.tag_completer)

        self.tag_list = TagListWidget(self)
        self.tag_list.setItemDelegate(WrappedTagItemDelegate(self.tag_list))
        self.tag_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tag_list.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.tag_list.setWordWrap(True)
        self.tag_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.tag_list.setUniformItemSizes(False)
        self.tag_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tag_list.itemChanged.connect(self._on_tag_item_changed)
        self.tag_list.tags_reordered.connect(self._on_tags_reordered)

        self.remove_tag_action = QAction("Remove Selected Tag", self)
        self.remove_tag_action.setShortcuts([QKeySequence("Delete"), QKeySequence("Backspace")])
        self.remove_tag_action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self.remove_tag_action.triggered.connect(self._remove_selected_tags)
        self.tag_list.addAction(self.remove_tag_action)

        tags_layout.addWidget(self.tag_input, stretch=0)
        tags_layout.addWidget(self.tag_list, stretch=1)
        self.tag_tabs.addTab(tags_panel, "Tags")

        vision_panel = QWidget(self)
        vision_layout = QVBoxLayout(vision_panel)
        vision_layout.setContentsMargins(6, 6, 6, 6)
        vision_layout.setSpacing(6)

        vision_layout.addWidget(QLabel("Description", self))
        self.vision_description = QTextEdit(self)
        self.vision_description.setAcceptRichText(False)
        self.vision_description.textChanged.connect(self._update_vision_dirty_state)
        vision_layout.addWidget(self.vision_description, stretch=1)

        vision_layout.addWidget(QLabel("CoT", self))
        self.vision_reasoning = QTextEdit(self)
        self.vision_reasoning.setAcceptRichText(False)
        self.vision_reasoning.textChanged.connect(self._update_vision_dirty_state)
        vision_layout.addWidget(self.vision_reasoning, stretch=1)

        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self.vision_save_button = QPushButton("Save", self)
        self.vision_save_button.setEnabled(False)
        self.vision_save_button.clicked.connect(self._save_vision_for_current_image)
        save_row.addWidget(self.vision_save_button)
        vision_layout.addLayout(save_row)

        self.tag_tabs.addTab(vision_panel, "Vision")
        # No image selected on startup; keep vision inputs disabled until a record is active.
        self._clear_vision_fields()

        controls_panel = QWidget(self)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        self.controls_tabs = QTabWidget(self)
        autotag_tab = QWidget(self)
        autotag_layout = QVBoxLayout(autotag_tab)
        autotag_layout.setContentsMargins(6, 6, 6, 6)
        autotag_layout.setSpacing(6)

        self.llm_endpoint_input = QLineEdit(self)
        self.llm_endpoint_input.setPlaceholderText("http://127.0.0.1:11434 (Ollama) or :8000 (OpenAI-compatible)")
        self.llm_endpoint_input.setText(self.llm_endpoint)

        self.llm_fetch_button = QPushButton("Fetch models", self)
        self.llm_fetch_button.clicked.connect(self.fetch_provider_models)

        self.llm_model_combo = QComboBox(self)
        self.llm_model_combo.setEditable(False)

        self.llm_use_button = QPushButton("Use", self)
        self.llm_use_button.clicked.connect(self.use_selected_provider_model)

        self.llm_timeout_input = QLineEdit(self)
        self.llm_timeout_input.setValidator(create_timeout_validator(self))
        self.llm_timeout_input.setText(str(int(DEFAULT_LLM_TIMEOUT)))
        self.llm_timeout_input.setMaximumWidth(90)

        self.llm_retry_input = QLineEdit(self)
        self.llm_retry_input.setValidator(create_retry_validator(self))
        self.llm_retry_input.setText("3")
        self.llm_retry_input.setMaximumWidth(60)

        self.llm_max_resolution_input = QLineEdit(self)
        self.llm_max_resolution_input.setValidator(create_max_resolution_validator(self))
        self.llm_max_resolution_input.setText("5.0")
        self.llm_max_resolution_input.setMaximumWidth(80)

        self.llm_threads_input = QLineEdit(self)
        self.llm_threads_input.setValidator(create_threads_validator(self))
        self.llm_threads_input.setText("1")
        self.llm_threads_input.setMaximumWidth(50)
        self.llm_threads_input.setToolTip("0 = auto")

        self.generate_tags_checkbox = QCheckBox("Tags", self)
        self.generate_tags_checkbox.setChecked(True)
        self.generate_tags_checkbox.checkStateChanged.connect(lambda _state: self._update_llm_controls())
        self.generate_description_checkbox = QCheckBox("Description", self)
        self.generate_description_checkbox.setChecked(True)
        self.generate_description_checkbox.checkStateChanged.connect(lambda _state: self._update_llm_controls())
        self.generate_vision_checkbox = QCheckBox("Vision", self)
        self.generate_vision_checkbox.setChecked(False)
        self.generate_vision_checkbox.checkStateChanged.connect(lambda _state: self._update_llm_controls())
        self.generate_refine_checkbox = QCheckBox("Refine", self)
        self.generate_refine_checkbox.setChecked(False)
        self.generate_refine_checkbox.checkStateChanged.connect(lambda _state: self._update_llm_controls())

        server_settings_frame = create_server_settings_frame(
            parent=self,
            endpoint_input=self.llm_endpoint_input,
            fetch_button=self.llm_fetch_button,
            model_combo=self.llm_model_combo,
            use_button=self.llm_use_button,
            include_tags_checkbox=self.generate_tags_checkbox,
            include_description_checkbox=self.generate_description_checkbox,
            include_vision_checkbox=self.generate_vision_checkbox,
            include_refine_checkbox=self.generate_refine_checkbox,
            timeout_input=self.llm_timeout_input,
            retry_input=self.llm_retry_input,
            max_resolution_input=self.llm_max_resolution_input,
            threads_input=self.llm_threads_input,
        )

        gen_row = QHBoxLayout()
        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.generate_with_llm)
        gen_row.addWidget(self.generate_button)

        buttons_row = QHBoxLayout()
        self.validate_button = QPushButton("Validate", self)
        self.validate_button.clicked.connect(self.validate_tags_with_llm)
        self.fixup_button = QPushButton("Fixup", self)
        self.fixup_button.clicked.connect(self.open_fixup_dialog)
        buttons_row.addWidget(self.validate_button)
        buttons_row.addWidget(self.fixup_button)

        ai_find_row = QHBoxLayout()
        self.ai_find_input = QLineEdit(self)
        self.ai_find_input.setPlaceholderText("Find concept in selected images (e.g. raven)")
        self.ai_find_input.textChanged.connect(lambda _text: self._update_llm_controls())
        self.ai_find_button = QPushButton("AI Find", self)
        self.ai_find_button.clicked.connect(self.ai_find_with_llm)
        ai_find_row.addWidget(self.ai_find_input, stretch=1)
        ai_find_row.addWidget(self.ai_find_button)

        self.stop_loading_button = QPushButton("Stop loading", self)
        self.stop_loading_button.clicked.connect(self._request_stop_directory_loading)
        self.stop_loading_button.setVisible(False)

        autotag_layout.addWidget(server_settings_frame)
        autotag_layout.addLayout(gen_row)
        autotag_layout.addLayout(buttons_row)
        autotag_layout.addLayout(ai_find_row)
        autotag_layout.addWidget(self.stop_loading_button)
        autotag_layout.addStretch(1)

        self.controls_tabs.addTab(autotag_tab, "AutoTag")
        self._add_prompt_tab("description", "Description")
        self._add_prompt_tab("tagging", "Tagging")
        self._add_prompt_tab("vision", "Vision")
        self._add_prompt_tab("refine", "Refine")
        self._add_prompt_tab("validation", "Validation")
        self._add_prompt_tab("search", "AI Search")
        self._add_known_tags_tab()
        controls_layout.addWidget(self.controls_tabs)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.right_splitter.addWidget(self.tag_tabs)
        self.right_splitter.addWidget(controls_panel)
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.setStretchFactor(0, 2)
        self.right_splitter.setStretchFactor(1, 1)

        right_layout.addWidget(self.right_splitter)

        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.center_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([340, 720, 340])

        content = QWidget(self)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.splitter)

        root_layout.addWidget(content, stretch=1)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar(self))
        self.status_connection_label = QLabel("no model", self)
        self.status_connection_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.statusBar().addPermanentWidget(self.status_connection_label)
        self._update_window_title(self._active_directory())

    def _add_prompt_tab(self, kind: str, title: str) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        role_row = QHBoxLayout()
        role_label = QLabel("Role:", self)
        role_input = QLineEdit(self)
        role_input.setPlaceholderText('e.g. "You are an ornithology expert."')
        saved_role = self._cfg.get("agent_roles", {}).get(kind, "")
        role_input.setText(saved_role)
        self.prompt_role_inputs[kind] = role_input
        set_role_button = QPushButton("Set", self)
        set_role_button.setMaximumWidth(50)
        set_role_button.clicked.connect(lambda _checked=False, prompt_kind=kind: self._set_agent_role(prompt_kind))
        role_row.addWidget(role_label)
        role_row.addWidget(role_input, stretch=1)
        role_row.addWidget(set_role_button)

        editor = QTextEdit(self)
        editor.setAcceptRichText(False)
        editor.setPlainText(load_prompt_for_kind(kind))
        self.prompt_editors[kind] = editor
        editor.textChanged.connect(lambda prompt_kind=kind: self._update_prompt_status(prompt_kind, edited=True))

        status_label = QLabel(self)
        self.prompt_status_labels[kind] = status_label

        buttons_row = QHBoxLayout()
        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(lambda _checked=False, prompt_kind=kind: self._apply_prompt_override(prompt_kind))
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(lambda _checked=False, prompt_kind=kind: self._save_prompt_to_file(prompt_kind))
        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(lambda _checked=False, prompt_kind=kind: self._reset_prompt_to_default(prompt_kind))
        test_button = QPushButton("Test", self)
        test_button.clicked.connect(lambda _checked=False, prompt_kind=kind: self._test_prompt(prompt_kind))
        buttons_row.addWidget(apply_button)
        buttons_row.addWidget(save_button)
        buttons_row.addWidget(reset_button)
        buttons_row.addWidget(test_button)
        buttons_row.addStretch(1)

        layout.addLayout(role_row)
        layout.addWidget(editor, stretch=1)
        layout.addWidget(status_label, stretch=0)
        layout.addLayout(buttons_row)

        self.controls_tabs.addTab(tab, title)
        self._update_prompt_status(kind)

    def _add_known_tags_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.known_tags_filter = QLineEdit(self)
        self.known_tags_filter.setPlaceholderText("Filter tags…")
        self.known_tags_filter.setClearButtonEnabled(True)
        self.known_tags_filter.textChanged.connect(self._refresh_known_tags_list)
        layout.addWidget(self.known_tags_filter)

        self.known_tags_list = GlobalTagListWidget(self)
        self.known_tags_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.known_tags_list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.known_tags_list.delete_requested.connect(self._delete_global_tag)

        layout.addWidget(self.known_tags_list, stretch=1)

        self.controls_tabs.addTab(tab, "Tags")
        self._refresh_known_tags_list()

    def _update_prompt_status(self, kind: str, edited: bool = False) -> None:
        label = self.prompt_status_labels.get(kind)
        editor = self.prompt_editors.get(kind)
        if label is None or editor is None:
            return

        source = prompt_source_for_kind(kind)
        source_text = "Default (code)"
        if source == "memory":
            source_text = "Applied (memory override)"
        elif source == "file":
            source_text = "File"

        suffix = ""
        if edited:
            current_text = editor.toPlainText().strip()
            try:
                active_text = active_prompt_for_kind(kind)
            except LlmProviderError:
                active_text = load_prompt_for_kind(kind)

            if current_text != active_text:
                suffix = " | Edited (not applied)"

        if source == "file":
            file_text = load_prompt_for_kind(kind)
            if file_text == get_default_prompt(kind):
                source_text = "File (default content)"

        label.setText(f"Active source: {source_text}{suffix}")

    def _prompt_title(self, kind: str) -> str:
        if kind == "description":
            return "Description"
        if kind == "tagging":
            return "Tagging"
        if kind == "vision":
            return "Vision"
        if kind == "refine":
            return "Refine"
        if kind == "validation":
            return "Validation"
        if kind == "search":
            return "AI Search"
        return kind

    def _prompt_editor_text(self, kind: str) -> str:
        editor = self.prompt_editors.get(kind)
        if editor is None:
            raise LlmProviderError(f"Prompt editor for {kind} is not available.")
        return editor.toPlainText().strip()

    def _test_prompt(self, kind: str) -> None:
        if self._llm_thread is not None:
            QMessageBox.information(self, "LLM busy", "Wait for the current LLM task to finish before testing a prompt.")
            return

        session = self._active_provider_session()
        if session is None:
            QMessageBox.warning(self, "No model selected", f"Connect to {self._llm_provider.display_name} and choose a model first.")
            return

        if self.current_index < 0 or self.current_index >= len(self.records):
            QMessageBox.information(self, "No image selected", "Select an image before testing the prompt.")
            return

        record = self.records[self.current_index]

        try:
            editor_text = self._prompt_editor_text(kind)
        except LlmProviderError as exc:
            QMessageBox.warning(self, "Prompt error", str(exc))
            return

        if kind == "vision":
            tags_lines = self._parse_annotations_for_tag_list(record.text)
            tags_text = "\n".join(tags_lines).strip()
            prompt = editor_text.replace("{tags}", tags_text)
        elif kind == "validation":
            prompt = editor_text.replace("{tags}", format_annotations_for_validation(record.text))
        elif kind == "search":
            query = " ".join(self.ai_find_input.text().split())
            if not query:
                QMessageBox.information(self, "Missing search text", "Enter text in the AI Find box to test the search prompt.")
                return
            prompt = editor_text.replace("{query}", query)
        elif kind == "refine":
            vision_data = read_sidecar_data(record.image_path)
            if vision_data.description:
                parts.append(f'description: "{vision_data.description}"')
            if vision_data.reasoning:
                parts.append(f'reasoning: "{vision_data.reasoning}"')
            vision_text = "\n".join(parts) if parts else "(no vision data available for this image)"
            prompt = editor_text.replace("{vision_data}", vision_text)
        else:
            prompt = editor_text

        try:
            self._apply_query_downscale_setting()
        except LlmProviderError:
            return

        timeout = self._llm_timeout_seconds()
        image_path = record.image_path
        title = self._prompt_title(kind)
        cancel_token = LlmRequestCancellation()

        def test_task(report_progress: Callable[[str], None]) -> str:
            return session.generate(
                image_path,
                prompt,
                timeout=timeout,
                cancellation=cancel_token,
            )

        test_thread = QThread(self)
        test_worker = RegenerateWorker(test_task)
        test_worker.moveToThread(test_thread)
        test_thread.started.connect(test_worker.run)
        test_worker.finished.connect(test_thread.quit)
        test_worker.failed.connect(test_thread.quit)
        test_worker.cancelled.connect(test_thread.quit)

        def _on_test_finished(result: object) -> None:
            test_thread.deleteLater()
            test_worker.deleteLater()
            from imagetagger.ui.llm_test_result_dialog import LlmTestResultDialog
            display_text = (
                "=== PROMPT ===\n"
                + prompt
                + "\n\n=== RESPONSE ===\n"
                + str(result)
            )
            dlg = LlmTestResultDialog(f"Test — {title}", display_text, self)
            dlg.exec()

        def _on_test_failed(error: str) -> None:
            test_thread.deleteLater()
            test_worker.deleteLater()
            QMessageBox.warning(self, f"Test failed — {title}", error)

        test_worker.finished.connect(_on_test_finished)
        test_worker.failed.connect(_on_test_failed)
        test_worker.cancelled.connect(_on_test_failed)

        self.statusBar().showMessage(f"Testing {title} prompt with {self._llm_provider.display_name}…")
        test_thread.start()

    def _set_agent_role(self, kind: str) -> None:
        role_input = self.prompt_role_inputs.get(kind)
        role = role_input.text().strip() if role_input is not None else ""
        if "agent_roles" not in self._cfg or not isinstance(self._cfg["agent_roles"], dict):
            self._cfg["agent_roles"] = {}
        if role:
            self._cfg["agent_roles"][kind] = role
        else:
            self._cfg["agent_roles"].pop(kind, None)
        _config.save(self._cfg)
        self.statusBar().showMessage(
            f"Role for {self._prompt_title(kind)} {'set' if role else 'cleared'}"
        )

    def _apply_prompt_override(self, kind: str) -> None:
        try:
            set_prompt_override(kind, self._prompt_editor_text(kind))
        except LlmProviderError as exc:
            QMessageBox.critical(self, "Apply prompt failed", str(exc))
            return
        self._update_prompt_status(kind)
        self.statusBar().showMessage(f"Applied {self._prompt_title(kind)} prompt in memory")

    def _save_prompt_to_file(self, kind: str) -> None:
        try:
            saved_text = save_prompt_for_kind(kind, self._prompt_editor_text(kind))
        except LlmProviderError as exc:
            QMessageBox.critical(self, "Save prompt failed", str(exc))
            return

        editor = self.prompt_editors.get(kind)
        if editor is not None:
            editor.setPlainText(saved_text)
        self._update_prompt_status(kind)
        self.statusBar().showMessage(f"Saved {self._prompt_title(kind)} prompt file")

    def _reset_prompt_to_default(self, kind: str) -> None:
        try:
            default_text = reset_prompt_to_default(kind)
            clear_prompt_override(kind)
        except LlmProviderError as exc:
            QMessageBox.critical(self, "Reset prompt failed", str(exc))
            return

        editor = self.prompt_editors.get(kind)
        if editor is not None:
            editor.setPlainText(default_text)
        self._update_prompt_status(kind)
        self.statusBar().showMessage(
            f"Reset {self._prompt_title(kind)} prompt to code default"
        )

    def _apply_config(self) -> None:
        font_point_size = self._cfg.get("font_point_size", 0)
        if isinstance(font_point_size, int) and font_point_size > 0:
            self._apply_app_font_size(font_point_size, persist=False)

        self._apply_main_window_geometry_from_config()

        server = self._cfg.get("llm_endpoint", self._cfg.get("ollama_server", "")).strip()
        model = self._cfg.get("llm_model", self._cfg.get("ollama_model", "")).strip()

        max_resolution_mpx = self._cfg.get("llm_max_resolution_mpx", 5)
        try:
            max_resolution_value = float(max_resolution_mpx)
            if max_resolution_value <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            max_resolution_value = 5.0
        self.llm_max_resolution_input.setText(self._format_mpx(max_resolution_value))
        max_pixels = max(1, int(max_resolution_value * 1_000_000))
        configure_image_preparation(max_image_pixels=max_pixels)

        raw_threads = self._cfg.get("llm_threads", self._cfg.get("ollama_threads", 1))
        try:
            thread_count = int(raw_threads)
            if thread_count < 0:
                raise ValueError()
        except (TypeError, ValueError):
            thread_count = 1
        self.llm_threads_input.setText(str(thread_count))

        if server:
            self.llm_endpoint = server
            self.llm_endpoint_input.setText(server)
        if model:
            self.llm_model_name = model
            self.llm_model_combo.addItem(model)
            self.llm_model_combo.setCurrentIndex(0)

        # Load last used directory if it exists
        last_dir = self._cfg.get("last_open_directory", "").strip()
        if last_dir:
            folder = Path(last_dir)
            if folder.exists() and folder.is_dir():
                last_image_str = self._cfg.get("last_selected_image", "").strip()
                restore_path: Path | None = None
                if last_image_str:
                    candidate = Path(last_image_str)
                    if candidate.exists() and candidate.is_file():
                        try:
                            candidate.relative_to(folder)
                            restore_path = candidate
                        except ValueError:
                            restore_path = None
                self.load_directory(folder, restore_selection=restore_path)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("File")

        self.open_action = QAction("Open Folder", self)
        self.open_action.setShortcut(platform_key_sequence("Ctrl+L", "Meta+O"))
        self.open_action.triggered.connect(self.open_folder)

        self.refresh_action = QAction("Refresh Folder", self)
        self.refresh_action.setShortcut(platform_key_sequence("Ctrl+R", "Meta+R"))
        self.refresh_action.triggered.connect(self.refresh_directory)

        self.exit_action = QAction("Exit", self)
        quit_shortcuts = [platform_key_sequence("Alt+F4", "Meta+Q")]
        for shortcut in QKeySequence.keyBindings(QKeySequence.StandardKey.Quit):
            if shortcut not in quit_shortcuts:
                quit_shortcuts.append(shortcut)
        self.exit_action.setShortcuts(quit_shortcuts)
        self.exit_action.triggered.connect(self.close)

        menu.addAction(self.open_action)
        menu.addAction(self.refresh_action)
        menu.addSeparator()
        menu.addAction(self.exit_action)

        edit_menu = self.menuBar().addMenu("Edit")

        self.increase_font_action = QAction("Increase Font", self)
        self.increase_font_action.setShortcuts(QKeySequence.keyBindings(QKeySequence.StandardKey.ZoomIn))
        self.increase_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.increase_font_action.triggered.connect(self.increase_font_size)

        self.decrease_font_action = QAction("Decrease Font", self)
        self.decrease_font_action.setShortcuts(QKeySequence.keyBindings(QKeySequence.StandardKey.ZoomOut))
        self.decrease_font_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.decrease_font_action.triggered.connect(self.decrease_font_size)

        edit_menu.addAction(self.increase_font_action)
        edit_menu.addAction(self.decrease_font_action)

    def _apply_main_window_geometry_from_config(self) -> None:
        raw = self._cfg.get("main_window_geometry")
        if not isinstance(raw, dict):
            return

        x = raw.get("x")
        y = raw.get("y")
        width = raw.get("width")
        height = raw.get("height")
        if not all(isinstance(value, int) for value in (x, y, width, height)):
            return
        if width <= 0 or height <= 0:
            return

        self.setGeometry(x, y, width, height)

    def _save_main_window_geometry(self) -> None:
        geometry = self.geometry()
        width = int(geometry.width())
        height = int(geometry.height())
        if width <= 0 or height <= 0:
            return

        self._cfg["main_window_geometry"] = {
            "x": int(geometry.x()),
            "y": int(geometry.y()),
            "width": width,
            "height": height,
        }
        _config.save(self._cfg)

    def _current_app_font_size(self) -> int:
        app = QApplication.instance()
        if app is None:
            return int(self.font().pointSize())

        point_size = int(app.font().pointSize())
        if point_size <= 0:
            point_size = int(self.font().pointSize())
        if point_size <= 0:
            point_size = 10
        return point_size

    def _apply_app_font_size(self, point_size: int, persist: bool = True) -> None:
        clamped = max(MIN_FONT_POINT_SIZE, min(MAX_FONT_POINT_SIZE, int(point_size)))
        app = QApplication.instance()
        if app is not None:
            font = QFont(app.font())
            font.setPointSize(clamped)
            app.setFont(font)

        # Recalculate tag row heights after font size changes.
        self._update_tag_item_heights()

        if persist:
            self._cfg["font_point_size"] = clamped
            _config.save(self._cfg)

    def increase_font_size(self) -> None:
        self._apply_app_font_size(self._current_app_font_size() + 1)

    def decrease_font_size(self) -> None:
        self._apply_app_font_size(self._current_app_font_size() - 1)

    def fetch_provider_models(self) -> None:
        server = self.llm_endpoint_input.text().strip()
        self.llm_fetch_button.setEnabled(False)
        try:
            model_names = self._llm_provider.fetch_models(server, timeout=self._llm_timeout_seconds())
            normalized_server = self._llm_provider.normalize_endpoint(server)
        except LlmProviderError as exc:
            QMessageBox.warning(self, f"{self._llm_provider.display_name} connection failed", str(exc))
            return
        finally:
            self.llm_fetch_button.setEnabled(True)

        self.llm_endpoint_input.setText(normalized_server)
        self.llm_model_combo.clear()
        self.llm_model_combo.addItems(model_names)
        if model_names:
            self.llm_model_combo.setCurrentIndex(0)
            self.statusBar().showMessage(f"Fetched {len(model_names)} model(s) from {normalized_server}")
        else:
            self.statusBar().showMessage(f"No models found at {normalized_server}")
            QMessageBox.information(self, "No models found", f"The {self._llm_provider.display_name} server returned no models.")

    def use_selected_provider_model(self) -> None:
        model_name = self.llm_model_combo.currentText().strip()
        if not model_name:
            QMessageBox.warning(self, "No model selected", "Fetch models and choose one before using it.")
            return

        try:
            normalized_server = self._llm_provider.normalize_endpoint(self.llm_endpoint_input.text())
        except LlmProviderError as exc:
            QMessageBox.warning(self, "Invalid server", str(exc))
            return

        self.llm_endpoint = normalized_server
        self.llm_model_name = model_name
        self.llm_endpoint_input.setText(normalized_server)
        self._cfg["llm_endpoint"] = normalized_server
        self._cfg["llm_model"] = model_name
        try:
            self._cfg["llm_threads"] = self._llm_thread_count(show_message=False)
        except LlmProviderError:
            self._cfg["llm_threads"] = 1
        _config.save(self._cfg)
        self._update_llm_controls()
        self.statusBar().showMessage(f"{self._llm_provider.display_name} model selected: {self.llm_model_name}")

    def _active_provider_session(self) -> VisionLlmSession | None:
        if not self.llm_model_name.strip():
            return None
        return self._llm_provider.create_session(self.llm_endpoint, self.llm_model_name)

    def _llm_timeout_seconds(self) -> float:
        def show_error(msg: str) -> None:
            QMessageBox.warning(self, "Invalid timeout", msg)
        return InputValidator.parse_timeout_seconds(self.llm_timeout_input.text(), show_error)

    def _llm_retry_count(self) -> int:
        return InputValidator.parse_retry_count(self.llm_retry_input.text())

    @staticmethod
    def _format_mpx(value: float) -> str:
        return InputValidator.format_megapixels(value)

    def _llm_max_resolution_mpx_value(self, show_message: bool = True) -> float:
        def show_error(msg: str) -> None:
            if show_message:
                QMessageBox.warning(self, "Invalid query downscale", msg)
        return InputValidator.parse_max_resolution_mpx(self.llm_max_resolution_input.text(), show_error)

    def _apply_query_downscale_setting(self) -> float:
        max_resolution_mpx = self._llm_max_resolution_mpx_value()
        max_pixels = max(1, int(max_resolution_mpx * 1_000_000))
        configure_image_preparation(max_image_pixels=max_pixels)
        return max_resolution_mpx

    def _llm_thread_count(self, show_message: bool = True) -> int:
        raw_value = self.llm_threads_input.text().strip()
        if not raw_value:
            if show_message:
                QMessageBox.warning(self, "Invalid threads", "Enter thread count (0 for auto).")
            raise LlmProviderError("Enter thread count.")

        try:
            thread_count = int(raw_value)
        except ValueError as exc:
            if show_message:
                QMessageBox.warning(self, "Invalid threads", "Thread count must be a whole number.")
            raise LlmProviderError("Thread count must be a whole number.") from exc

        if thread_count < 0:
            if show_message:
                QMessageBox.warning(self, "Invalid threads", "Thread count must be 0 or greater.")
            raise LlmProviderError("Thread count must be 0 or greater.")

        return thread_count

    def _update_llm_controls(self) -> None:
        connected = bool(self.llm_model_name.strip())
        if connected:
            text = f"{self.llm_model_name} @ {self.llm_endpoint}"
        else:
            text = "no model"
        if self.status_connection_label is not None:
            self.status_connection_label.setText(text)
        if self._llm_thread is not None and self._llm_action_name == "Generate":
            self.generate_button.setText("Stop generation")
            self.generate_button.setEnabled(True)
            self.validate_button.setText("Validate")
            self.validate_button.setEnabled(False)
            self.ai_find_button.setText("AI Find")
            self.ai_find_button.setEnabled(False)
            self.ai_find_input.setEnabled(False)
            self._update_fixup_button_state()
            return
        if self._llm_thread is not None and self._llm_action_name == "Validate":
            self.generate_button.setText("Generate")
            self.generate_button.setEnabled(False)
            self.validate_button.setText("Stop validation")
            self.validate_button.setEnabled(True)
            self.ai_find_button.setText("AI Find")
            self.ai_find_button.setEnabled(False)
            self.ai_find_input.setEnabled(False)
            self._update_fixup_button_state()
            return
        if self._llm_thread is not None and self._llm_action_name == "AI Find":
            self.generate_button.setText("Generate")
            self.generate_button.setEnabled(False)
            self.validate_button.setText("Validate")
            self.validate_button.setEnabled(False)
            self.ai_find_button.setText("Stop AI Find")
            self.ai_find_button.setEnabled(True)
            self.ai_find_input.setEnabled(False)
            self._update_fixup_button_state()
            return

        self.generate_button.setText("Generate")
        self.validate_button.setText("Validate")
        self.ai_find_button.setText("AI Find")
        active = connected and self._llm_thread is None
        self.generate_button.setEnabled(
            active
            and (
                self.generate_tags_checkbox.isChecked()
                or self.generate_description_checkbox.isChecked()
                or self.generate_vision_checkbox.isChecked()
                or self.generate_refine_checkbox.isChecked()
            )
        )
        self.validate_button.setEnabled(active)
        self.ai_find_input.setEnabled(active)
        self.ai_find_button.setEnabled(active and bool(self.ai_find_input.text().strip()))
        self._update_fixup_button_state()

    def _current_record(self) -> ImageRecord | None:
        if 0 <= self.current_index < len(self.records):
            return self.records[self.current_index]
        return None

    def _clear_vision_fields(self) -> None:
        self._vision_updating = True
        try:
            if hasattr(self, "vision_description"):
                self.vision_description.setPlainText("")
            if hasattr(self, "vision_reasoning"):
                self.vision_reasoning.setPlainText("")
        finally:
            self._vision_updating = False
        self._vision_loaded_description = ""
        self._vision_loaded_reasoning = ""
        if hasattr(self, "vision_save_button"):
            self.vision_save_button.setEnabled(False)
        if hasattr(self, "vision_description"):
            self.vision_description.setEnabled(False)
        if hasattr(self, "vision_reasoning"):
            self.vision_reasoning.setEnabled(False)
        if hasattr(self, "image_label"):
            self.image_label.setToolTip("")

    def _load_vision_for_current_image(self) -> None:
        record = self._current_record()
        if record is None:
            self._clear_vision_fields()
            return

        vision_data = read_sidecar_data(record.image_path)

        self._vision_updating = True
        try:
            self.vision_description.setPlainText(vision_data.description)
            self.vision_reasoning.setPlainText(vision_data.reasoning)
        finally:
            self._vision_updating = False

        self._vision_loaded_description = vision_data.description
        self._vision_loaded_reasoning = vision_data.reasoning
        self.vision_description.setEnabled(True)
        self.vision_reasoning.setEnabled(True)
        self._update_vision_dirty_state()
        self._update_image_label_tooltip(vision_data)

    def _update_vision_dirty_state(self) -> None:
        if self._vision_updating:
            return
        record = self._current_record()
        if record is None:
            self.vision_save_button.setEnabled(False)
            return
        current_description = self.vision_description.toPlainText()
        current_reasoning = self.vision_reasoning.toPlainText()
        dirty = (
            current_description != self._vision_loaded_description
            or current_reasoning != self._vision_loaded_reasoning
        )
        self.vision_save_button.setEnabled(dirty)

    def _save_vision_for_current_image(self) -> None:
        record = self._current_record()
        if record is None:
            return

        vision_data = read_sidecar_data(record.image_path)
        vision_data.description = self.vision_description.toPlainText()
        vision_data.reasoning = self.vision_reasoning.toPlainText()
        try:
            write_sidecar_data(record.image_path, vision_data)
        except Exception as exc:
            path = get_sidecar_json_path(record.image_path)
            QMessageBox.warning(self, "Save failed", f"Could not write {path.name}:\n\n{exc}")
            return

        self._vision_loaded_description = vision_data.description
        self._vision_loaded_reasoning = vision_data.reasoning
        self._update_vision_dirty_state()
        path = get_sidecar_json_path(record.image_path)
        self.statusBar().showMessage(f"Saved vision metadata: {path.name}")

    def _selected_record_indexes(self) -> list[int]:
        indexes = sorted(
            item.row()
            for item in self.list_widget.selectedIndexes()
            if self.list_widget.item(item.row()) is not None and not self.list_widget.item(item.row()).isHidden()
        )
        if indexes:
            return indexes
        if 0 <= self.current_index < len(self.records):
            current_item = self.list_widget.item(self.current_index)
            if current_item is not None and current_item.isHidden():
                return []
            return [self.current_index]
        return []

    def _select_all_images(self) -> None:
        if self.list_widget.count() == 0:
            return
        self._ignore_selection_sync = True
        self.list_widget.clearSelection()
        first_visible_row = -1
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is None or item.isHidden():
                continue
            item.setSelected(True)
            if first_visible_row < 0:
                first_visible_row = row
        if self.current_index < 0 and first_visible_row >= 0:
            self.list_widget.setCurrentRow(first_visible_row)
        self._ignore_selection_sync = False

    def _record_matches_filter(self, record: ImageRecord) -> bool:
        expression = self.filter_input.text().strip()
        if not expression:
            return True

        try:
            parsed = _parse_filter_expression(expression)
        except FilterSyntaxError:
            return True

        if parsed is None:
            return True

        runtime = self._build_filter_runtime()
        return parsed.evaluate(record, runtime)

    def _show_filter_rules_dialog(self) -> None:
        rules_text = (
            "Filter rules:\n\n"
            "- fixup: show images with fixup files\n"
            "- untagged: show images with no annotation file\n"
            "- vision: show images with a sidecar .json (vision data)\n"
            "- resolution <, >, <=, >=: compare resolution in megapixels\n"
            "- \"tag\": match an exact tag\n"
            "- 'text': match free text inside annotation content\n"
            "- ! or ~: NOT operator\n"
            "- &: AND operator\n"
            "- |: OR operator\n"
            "- ( ... ): group expressions\n\n"
            "Precedence (highest to lowest): NOT, AND, OR\n\n"
            "Examples:\n"
            "- !fixup\n"
            "- resolution < 1.0\n"
            "- (resolution > 5) & 'landscape'\n"
            "- untagged | fixup\n"
            "- fixup & \"landscape\"\n"
            "- !\"portrait\" | 'sunset'\n"
            "- (fixup & \"animal\") | ~'night'"
        )
        QMessageBox.information(self, "Filter rules", rules_text)

    def _get_image_resolution_mpx(self, record: ImageRecord) -> float | None:
        """Return image resolution in megapixels, caching the result on the record."""
        if record._resolution_mpx is not None:
            return record._resolution_mpx
        try:
            with Image.open(record.image_path) as image:
                width, height = image.size
                record._resolution_mpx = (width * height) / 1_000_000.0
                return record._resolution_mpx
        except Exception:
            return None

    def _build_filter_runtime(self, tag_cache: dict[Path, set[str]] | None = None) -> _FilterRuntime:
        named_filters: dict[str, Callable[[ImageRecord], bool]] = {
            "fixup": lambda record: record.has_pending_fixup,
            "untagged": lambda record: not record.text_path.exists(),
            "vision": lambda record: get_sidecar_json_path(record.image_path).exists(),
        }
        return _FilterRuntime(
            named_filters=named_filters,
            tag_filter=lambda record, tag: self._record_has_tag(record, tag, tag_cache=tag_cache),
            freetext_filter=lambda record, text: self._record_contains_freetext(record, text),
            get_resolution_mpx=self._get_image_resolution_mpx,
        )

    def _record_has_tag(self, record: ImageRecord, tag: str, tag_cache: dict[Path, set[str]] | None = None) -> bool:
        normalized = self._sanitize_annotation_text(tag).casefold()
        if not normalized:
            return False

        if tag_cache is not None:
            cached = tag_cache.get(record.image_path)
            if cached is None:
                cached = {item.casefold() for item in self._parse_tags(record.text)}
                tag_cache[record.image_path] = cached
            return normalized in cached

        return normalized in {item.casefold() for item in self._parse_tags(record.text)}

    def _record_contains_freetext(self, record: ImageRecord, text: str) -> bool:
        query = text.casefold().strip()
        if not query:
            return False
        return query in record.text.casefold()

    def _first_visible_row(self) -> int:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None and not item.isHidden():
                return row
        return -1

    def _visible_record_count(self) -> int:
        if not self.filter_input.text().strip():
            return len(self.records)
        visible = 0
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None and not item.isHidden():
                visible += 1
        return visible

    def _visible_position_for_row(self, row: int) -> int:
        if row < 0:
            return -1
        if not self.filter_input.text().strip():
            return row + 1

        position = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item is None or item.isHidden():
                continue
            position += 1
            if i == row:
                return position
        return -1

    def _filtered_total_status_text(self, selected_row: int | None = None) -> str:
        visible = self._visible_record_count()
        total = len(self.records)
        if selected_row is None:
            return f"({visible} of {total})"

        position = self._visible_position_for_row(selected_row)
        if position > 0:
            return f"({position} of {visible}, total {total})"
        return f"({visible} of {total})"

    def _apply_image_filter(self, _text: str | None = None) -> None:
        selected_path: Path | None = None
        if 0 <= self.current_index < len(self.records):
            selected_path = self.records[self.current_index].image_path

        expression = self.filter_input.text().strip()
        parsed: _FilterNode | None = None
        runtime: _FilterRuntime | None = None

        if expression:
            try:
                parsed = _parse_filter_expression(expression)
            except FilterSyntaxError as exc:
                self.statusBar().showMessage(f"Invalid filter: {exc}")
            else:
                tag_cache: dict[Path, set[str]] = {}
                runtime = self._build_filter_runtime(tag_cache=tag_cache)

        for row, record in enumerate(self.records):
            item = self.list_widget.item(row)
            if item is None:
                continue
            is_match = True
            if parsed is not None and runtime is not None:
                is_match = parsed.evaluate(record, runtime)
            item.setHidden(not is_match)

        if selected_path is not None:
            for row, record in enumerate(self.records):
                if record.image_path != selected_path:
                    continue
                item = self.list_widget.item(row)
                if item is not None and not item.isHidden():
                    self.list_widget.setCurrentRow(row)
                    return

        first_visible = self._first_visible_row()
        if first_visible >= 0:
            self.list_widget.setCurrentRow(first_visible)
            return

        self.list_widget.clearSelection()
        self.current_index = -1
        self._update_fixup_button_state()
        self.statusBar().showMessage(f"{self._filtered_total_status_text()} no images match filter")

    def _on_fixup_state_changed(self, image_path: Path | None = None) -> None:
        if image_path is None:
            for record in self.records:
                record._sidecar_has_pending_fixup = None
            # Defer the bulk visual refresh to the next event-loop iteration so
            # _apply_image_filter and _update_fixup_button_state run first and
            # the GUI stays responsive during large batch operations.
            QTimer.singleShot(0, self._refresh_all_list_item_previews)
        else:
            for index, record in enumerate(self.records):
                if record.image_path == image_path:
                    record._sidecar_has_pending_fixup = None
                    self._update_list_item_preview(index)
                    break
        self._apply_image_filter()
        self._update_fixup_button_state()
        # Refresh the image preview tooltip if the changed image is currently selected.
        current = self._current_record()
        if current is not None and (image_path is None or current.image_path == image_path):
            self._update_image_label_tooltip()

    def _update_fixup_button_state(self) -> None:
        record = self._current_record()
        enabled = record is not None and bool(self._list_item_badge_specs(record))
        if enabled:
            item = self.list_widget.item(self.current_index)
            if item is None or item.isHidden():
                enabled = False
        if self._llm_thread is not None:
            if self._llm_action_name != "Validate" or self.current_index in self._validate_pending_indices:
                enabled = False
        self.fixup_button.setEnabled(enabled)

    def _find_adjacent_fixup_index(self, start_index: int, direction: int) -> int | None:
        if direction not in (-1, 1):
            return None

        index = start_index + direction
        while 0 <= index < len(self.records):
            item = self.list_widget.item(index)
            if item is not None and item.isHidden():
                index += direction
                continue

            if index in self._validate_pending_indices:
                index += direction
                continue

            record = self.records[index]
            if record.has_pending_fixup:
                return index

            index += direction

        return None

    def _find_fixup_index(self, reverse: bool) -> int | None:
        if not self.records:
            return None

        indices = range(len(self.records) - 1, -1, -1) if reverse else range(len(self.records))
        for index in indices:
            if index in self._validate_pending_indices:
                continue
            item = self.list_widget.item(index)
            if item is not None and item.isHidden():
                continue
            record = self.records[index]
            if record.has_pending_fixup:
                return index
        return None

    def _jump_to_first_fixup(self) -> None:
        index = self._find_fixup_index(reverse=False)
        if index is None:
            self.statusBar().showMessage("No fixup image found in the current list")
            return
        self.list_widget.setCurrentRow(index)

    def _jump_to_last_fixup(self) -> None:
        index = self._find_fixup_index(reverse=True)
        if index is None:
            self.statusBar().showMessage("No fixup image found in the current list")
            return
        self.list_widget.setCurrentRow(index)

    def _move_image_selection(self, direction: int) -> bool:
        if direction not in (-1, 1):
            return False

        if not self.records:
            return False

        focus_tag_list = (
            self.tag_list.hasFocus()
            and len(self.tag_list.selectedItems()) > 0
        )

        if 0 <= self.current_index < len(self.records):
            index = self.current_index + direction
        else:
            index = 0 if direction > 0 else len(self.records) - 1

        while 0 <= index < len(self.records):
            item = self.list_widget.item(index)
            if item is not None and not item.isHidden():
                self.list_widget.setCurrentRow(index)
                if focus_tag_list and self.tag_list.count() > 0:
                    self.tag_list.setCurrentRow(0)
                    self.tag_list.setFocus()
                return True
            index += direction

        return False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if watched is self.tag_input and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if not self.tag_input.text().strip():
                if key == Qt.Key.Key_Up:
                    self._move_image_selection(-1)
                    return True
                if key == Qt.Key.Key_Down:
                    self._move_image_selection(1)
                    return True
        return super().eventFilter(watched, event)

    def _show_image_context_menu(self, position) -> None:
        record = self._current_record()
        if record is None:
            return

        menu = QMenu(self)

        open_default_action = menu.addAction("Open in Default App")
        open_default_action.triggered.connect(self._open_current_image_in_default_app)

        open_with_menu = menu.addMenu("Open With")

        editors = self._get_detected_external_editors(refresh=False)
        if editors:
            for editor in editors:
                action = open_with_menu.addAction(editor.display_name)
                action.triggered.connect(
                    lambda _checked=False, selected_editor=editor: self._open_current_image_with_editor(selected_editor)
                )
        else:
            unavailable = open_with_menu.addAction("No common editors detected")
            unavailable.setEnabled(False)

        open_with_menu.addSeparator()
        choose_action = open_with_menu.addAction("Choose executable...")
        choose_action.triggered.connect(self._open_current_image_with_custom_editor)

        refresh_action = open_with_menu.addAction("Refresh detected editors")
        refresh_action.triggered.connect(lambda _checked=False: self._get_detected_external_editors(refresh=True))

        menu.addSeparator()
        delete_action = menu.addAction("Delete file")
        delete_action.triggered.connect(self._delete_current_image_from_context_menu)

        menu.exec(self.image_label.mapToGlobal(position))

    def _confirm_on_delete_enabled(self) -> bool:
        value = self._cfg.get("confirm_on_delete", True)
        return value if isinstance(value, bool) else True

    def _delete_image_and_related_files(self, image_path: Path, *, confirm: bool = True) -> tuple[bool, bool]:
        record_index = self._record_index_for_image_path(image_path)
        if record_index < 0:
            return (False, any(item.has_pending_fixup for item in self.records))

        record = self.records[record_index]
        if confirm and self._confirm_on_delete_enabled():
            answer = QMessageBox.question(
                self,
                "Delete file",
                (
                    f"Delete this image and related files?\n\n"
                    f"Image: {record.image_path.name}\n"
                    f"Also deletes: {record.text_path.name} and sidecar .json"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return (False, any(item.has_pending_fixup for item in self.records))

        errors: list[str] = []
        try:
            record.image_path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{record.image_path.name}: {exc}")

        try:
            record.text_path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{record.text_path.name}: {exc}")

        delete_sidecar_for_image(record.image_path)

        if errors:
            QMessageBox.warning(
                self,
                "Delete failed",
                "Could not delete one or more files:\n\n" + "\n".join(errors),
            )
            return (False, any(item.has_pending_fixup for item in self.records))

        was_current = self.current_index == record_index
        self.records.pop(record_index)
        deleted_item = self.list_widget.takeItem(record_index)
        del deleted_item

        self._rebuild_known_tags_from_records()
        self._refresh_tag_completions()

        if not self.records:
            self.current_index = -1
            self.list_widget.clearSelection()
            self._set_watched_image(None)
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("No image selected")
            self.tag_input.clear()
            self.tag_list.clear()
            self._clear_vision_fields()
            self._update_fixup_button_state()
            self.statusBar().showMessage("Deleted file; no images remain")
            return (True, False)

        if was_current:
            next_index = record_index if record_index < len(self.records) else len(self.records) - 1
            self.list_widget.setCurrentRow(next_index)
        elif self.current_index > record_index:
            self.current_index -= 1

        has_fixups_remaining = any(item.has_pending_fixup for item in self.records)
        self.statusBar().showMessage(f"Deleted {record.image_path.name}")
        return (True, has_fixups_remaining)

    def _delete_current_image_from_context_menu(self) -> None:
        record = self._current_record()
        if record is None:
            return
        self._delete_image_and_related_files(record.image_path, confirm=True)

    def _current_image_path(self) -> Path | None:
        record = self._current_record()
        if record is None:
            return None
        return record.image_path

    @staticmethod
    def _normalized_path_for_compare(path: Path) -> str:
        return os.path.normcase(str(path.resolve(strict=False)))

    def _set_watched_image(self, image_path: Path | None) -> None:
        self._image_reload_helper.set_watched_image(image_path)

    def _on_image_reload(self, image_path: Path) -> None:
        """Callback invoked when watched image is reloaded."""
        current_record = self._current_record()
        if current_record is None:
            return
        if self._normalized_path_for_compare(current_record.image_path) != self._normalized_path_for_compare(image_path):
            return

        # Skip transient save states where the file is temporarily unavailable.
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return

        # Invalidate the pixmap cache so _show_image re-reads the updated file.
        if self._cached_pixmap_path == image_path:
            self._cached_pixmap = None
            self._cached_pixmap_path = None

        self._show_image(image_path)

        record_index = self._record_index_for_image_path(image_path)
        if record_index >= 0:
            self._update_list_item_preview(record_index)

        self.statusBar().showMessage(f"Reloaded image: {image_path.name}")

    def _open_current_image_in_default_app(self) -> None:
        image_path = self._current_image_path()
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

        self._set_watched_image(image_path)
        self.statusBar().showMessage(f"Opened {image_path.name} in default app")

    def _open_current_image_with_editor(self, editor: ExternalEditor) -> None:
        image_path = self._current_image_path()
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

        self._set_watched_image(image_path)
        self.statusBar().showMessage(f"Opened {image_path.name} with {editor.display_name}")

    def _open_current_image_with_custom_editor(self) -> None:
        image_path = self._current_image_path()
        if image_path is None:
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
        self._open_current_image_with_editor(custom_editor)

    def _get_detected_external_editors(self, refresh: bool = False) -> list[ExternalEditor]:
        if not refresh and self._detected_external_editors is not None:
            return list(self._detected_external_editors)

        try:
            editors = discover_graphics_editors()
        except Exception:
            editors = []
        self._detected_external_editors = editors
        return list(editors)

    def _record_index_for_image_path(self, image_path: Path) -> int:
        for index, record in enumerate(self.records):
            if record.image_path == image_path:
                return index
        return -1

    def _set_tags_for_image_path(self, image_path: Path, tags: list[str], status_prefix: str = "Auto-saved") -> None:
        record_index = self._record_index_for_image_path(image_path)
        if record_index < 0:
            return

        normalized_tags = [tag for tag in (item.strip() for item in tags) if tag]
        record = self.records[record_index]
        record.text = self._serialize_tags(normalized_tags)

        if self.current_index == record_index:
            self._populate_tag_list(normalized_tags)

        self._update_list_item_preview(record_index)
        self._rebuild_known_tags_from_records()
        self._refresh_tag_completions()
        self._write_record_text(record, status_prefix=status_prefix)

    def _merge_dialog_geometry_from_config(self) -> dict[str, int] | None:
        raw = self._cfg.get("merge_dialog_geometry")
        if not isinstance(raw, dict):
            return None

        x = raw.get("x")
        y = raw.get("y")
        width = raw.get("width")
        height = raw.get("height")
        if not all(isinstance(value, int) for value in (x, y, width, height)):
            return None
        if width <= 0 or height <= 0:
            return None

        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

    def _save_merge_dialog_geometry(self, geometry: dict[str, int]) -> None:
        x = geometry.get("x")
        y = geometry.get("y")
        width = geometry.get("width")
        height = geometry.get("height")
        if not all(isinstance(value, int) for value in (x, y, width, height)):
            return
        if width <= 0 or height <= 0:
            return

        self._cfg["merge_dialog_geometry"] = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
        _config.save(self._cfg)

    def _display_image_path(self, image_path: Path) -> str:
        if self._root_directory is None:
            return image_path.name
        try:
            relative_path = image_path.relative_to(self._root_directory)
        except ValueError:
            return image_path.name
        relative_text = relative_path.as_posix()
        return relative_text or image_path.name

    def open_fixup_dialog(self) -> None:
        record = self._current_record()
        if record is None:
            return

        try:
            regenerate_max_resolution_mpx = self._apply_query_downscale_setting()
        except LlmProviderError:
            return

        initial_fixup_record_indices = [
            i for i, item in enumerate(self.records)
            if item.has_pending_fixup
        ]
        initial_fixup_total = len(initial_fixup_record_indices)
        if initial_fixup_total <= 0:
            return

        regenerate_tags_enabled = self.generate_tags_checkbox.isChecked()
        regenerate_description_enabled = self.generate_description_checkbox.isChecked()
        timeout_text = self.llm_timeout_input.text().strip()
        retry_text = self.llm_retry_input.text().strip()
        try:
            regenerate_timeout_seconds = int(timeout_text) if timeout_text else int(DEFAULT_LLM_TIMEOUT)
        except ValueError:
            regenerate_timeout_seconds = int(DEFAULT_LLM_TIMEOUT)
        try:
            regenerate_retry_count = int(retry_text) if retry_text else 0
        except ValueError:
            regenerate_retry_count = 0
        
        # Persistent settings across image navigation
        regenerate_model_name = ""
        regenerate_model_endpoint = ""
        regenerate_user_hint = ""

        def _capture_regenerate_settings(values: dict[str, int | float | bool | str]) -> None:
            nonlocal regenerate_tags_enabled
            nonlocal regenerate_description_enabled
            nonlocal regenerate_timeout_seconds
            nonlocal regenerate_retry_count
            nonlocal regenerate_max_resolution_mpx
            nonlocal regenerate_model_name
            nonlocal regenerate_model_endpoint
            nonlocal regenerate_user_hint

            tags_enabled = values.get("tags_enabled")
            description_enabled = values.get("description_enabled")
            timeout_seconds = values.get("timeout_seconds")
            retry_count = values.get("retry_count")
            max_resolution_mpx = values.get("max_resolution_mpx")
            model_name = values.get("model_name")
            model_endpoint = values.get("model_endpoint")
            user_hint = values.get("user_hint")

            if isinstance(tags_enabled, bool):
                regenerate_tags_enabled = tags_enabled
            if isinstance(description_enabled, bool):
                regenerate_description_enabled = description_enabled
            if isinstance(timeout_seconds, int):
                regenerate_timeout_seconds = max(1, timeout_seconds)
            if isinstance(retry_count, int):
                regenerate_retry_count = max(0, retry_count)
            if isinstance(max_resolution_mpx, (int, float)) and max_resolution_mpx > 0:
                regenerate_max_resolution_mpx = float(max_resolution_mpx)
                self.llm_max_resolution_input.setText(self._format_mpx(regenerate_max_resolution_mpx))
                configure_image_preparation(max_image_pixels=max(1, int(regenerate_max_resolution_mpx * 1_000_000)))
            if isinstance(model_name, str):
                regenerate_model_name = model_name
            if isinstance(model_endpoint, str):
                regenerate_model_endpoint = model_endpoint
            if isinstance(user_hint, str):
                regenerate_user_hint = user_hint

        while True:
            record = self._current_record()
            if record is None:
                return

            try:
                display_index = initial_fixup_record_indices.index(self.current_index) + 1
            except ValueError:
                display_index = 1

            title_path = self._display_image_path(record.image_path)

            dialog_title = f"Fixup - {title_path} ({display_index} of {initial_fixup_total})"

            prev_fixup_index = self._find_adjacent_fixup_index(self.current_index, -1)
            next_fixup_index = self._find_adjacent_fixup_index(self.current_index, 1)
            mouse_actions_cfg = self._cfg.get("merge_table_mouse_actions", {})
            if not isinstance(mouse_actions_cfg, dict):
                mouse_actions_cfg = {}

            outcome = open_fixup_dialog_for_image(
                parent=self,
                image_path=record.image_path,
                current_annotations=self._current_tags(),
                title_text=dialog_title,
                parse_tags=self._parse_tags,
                sanitize_annotation=self._sanitize_annotation_text,
                apply_annotations=lambda tags, status_prefix, image_path=record.image_path: self._set_tags_for_image_path(
                    image_path,
                    tags,
                    status_prefix=status_prefix,
                ),
                show_status=self.statusBar().showMessage,
                refresh_fixup_state=self._on_fixup_state_changed,
                initial_geometry=self._merge_dialog_geometry_from_config(),
                save_geometry=self._save_merge_dialog_geometry,
                can_navigate_prev=prev_fixup_index is not None,
                can_navigate_next=next_fixup_index is not None,
                tag_suggestions=self._sorted_tag_suggestions(),
                provider_session=self._active_provider_session(),
                provider=self._llm_provider,
                regenerate_tags_enabled=regenerate_tags_enabled,
                regenerate_description_enabled=regenerate_description_enabled,
                regenerate_timeout_seconds=regenerate_timeout_seconds,
                regenerate_retry_count=regenerate_retry_count,
                regenerate_max_resolution_mpx=regenerate_max_resolution_mpx,
                regenerate_model_name=regenerate_model_name,
                regenerate_model_endpoint=regenerate_model_endpoint,
                regenerate_user_hint=regenerate_user_hint,
                merge_table_double_click_action_enabled=bool(
                    mouse_actions_cfg.get("double_click_action_enabled", True)
                ),
                merge_table_swipe_actions_enabled=bool(
                    mouse_actions_cfg.get("swipe_actions_enabled", False)
                ),
                merge_table_horizontal_scroll_actions_enabled=bool(
                    mouse_actions_cfg.get("horizontal_scroll_actions_enabled", False)
                ),
                merge_table_horizontal_scroll_reverse_enabled=bool(
                    mouse_actions_cfg.get("horizontal_scroll_reverse_enabled", False)
                ),
                merge_table_horizontal_scroll_stop_idle_seconds=float(
                    mouse_actions_cfg.get("horizontal_scroll_stop_idle_seconds", 0.45)
                ),
                merge_table_horizontal_scroll_row_target_mode=int(
                    mouse_actions_cfg.get(
                        "horizontal_scroll_row_target_mode",
                        _config.MERGE_TABLE_HSCROLL_TARGET_POINTER_ON_SELECTED,
                    )
                ),
                delete_image=lambda image_path=record.image_path: self._delete_image_and_related_files(
                    image_path,
                    confirm=False,
                ),
                confirm_delete=self._confirm_on_delete_enabled(),
                save_regenerate_settings=_capture_regenerate_settings,
            )

            if outcome == "prev":
                target = self._find_adjacent_fixup_index(self.current_index, -1)
                if target is not None:
                    self.list_widget.setCurrentRow(target)
                    continue
                return

            if outcome == "next":
                # Deletion from merge dialog can change list indices while the dialog is open.
                # Recompute target from current state instead of using stale pre-dialog indices.
                current_record = self._current_record()
                if current_record is not None and current_record.has_pending_fixup:
                    continue

                target = self._find_adjacent_fixup_index(self.current_index, 1)
                if target is None:
                    target = self._find_adjacent_fixup_index(self.current_index, -1)
                if target is None:
                    target = self._find_fixup_index(reverse=False)
                if target is not None:
                    self.list_widget.setCurrentRow(target)
                    continue
            return

    def open_folder(self) -> None:
        if self._loader_thread and self._loader_thread.isRunning():
            self.statusBar().showMessage("A folder is already loading")
            return

        start_dir = self._cfg.get("last_open_directory", "") or ""
        folder = QFileDialog.getExistingDirectory(self, "Select image folder", start_dir)
        if not folder:
            return

        self.load_directory(Path(folder))

    def refresh_directory(self) -> None:
        if self._loader_thread and self._loader_thread.isRunning():
            self.statusBar().showMessage("A folder is already loading")
            return

        folder = self._root_directory
        if folder is None:
            QMessageBox.information(self, "No folder selected", "Open a folder before refreshing it.")
            return

        current_record = self._current_record()
        restore_selection = current_record.image_path if current_record is not None else None
        self.load_directory(folder, restore_selection=restore_selection)

    def _active_directory(self) -> Path | None:
        record = self._current_record()
        if record is not None:
            return record.image_path.parent

        if self.records:
            return self.records[0].image_path.parent

        folder = str(self._cfg.get("last_open_directory", "")).strip()
        if not folder:
            return None

        return Path(folder)

    def _update_window_title(self, folder: Path | None) -> None:
        base_title = "ImageTagger"
        if folder is None:
            self.setWindowTitle(base_title)
            return

        name = folder.name.strip() or str(folder)
        self.setWindowTitle(f"{base_title} - {name}")

    def _find_image_name_collision(self, folder: Path) -> tuple[Path, Path] | None:
        """Return the first pair of image files that would map to the same .txt path."""
        try:
            image_paths = sorted(
                [
                    p
                    for p in folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda p: p.relative_to(folder).as_posix().lower(),
            )
        except OSError as exc:
            QMessageBox.critical(self, "Folder load failed", f"Failed to read folder: {exc}")
            return None

        seen_by_txt_path: dict[str, Path] = {}
        for image_path in image_paths:
            txt_key = str(image_path.with_suffix(".txt")).casefold()
            existing = seen_by_txt_path.get(txt_key)
            if existing is None:
                seen_by_txt_path[txt_key] = image_path
                continue

            if existing.suffix.lower() != image_path.suffix.lower():
                return existing, image_path

        return None

    def load_directory(self, folder: Path, restore_selection: Path | None = None) -> None:
        self._directory_loading_active = True
        self._directory_load_cancel_requested = False
        self._directory_load_cancelled_by_user = False
        self._set_loading_state(True)
        self.statusBar().showMessage("Scanning folder...")
        self._pending_selection_path = restore_selection
        self._detected_external_editors = None

        try:
            max_thread_cap = int(self._cfg.get("directory_loader_max_threads", 8))
        except (TypeError, ValueError):
            max_thread_cap = 8
        max_thread_cap = max(1, max_thread_cap)

        self._loader_thread = QThread(self)
        self._loader_worker = FolderLoadWorker(folder, max_thread_cap=max_thread_cap)
        self._loader_worker.moveToThread(self._loader_thread)

        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.scan_progress.connect(self._on_scan_progress)
        self._loader_worker.scan_ready.connect(self._on_scan_ready)
        self._loader_worker.collision_detected.connect(self._on_collision_detected)
        self._loader_worker.item_loaded.connect(self._on_item_loaded)
        self._loader_worker.progress.connect(self._on_load_progress)
        self._loader_worker.icc_warning.connect(self._on_icc_warning_detected)
        self._loader_worker.finished.connect(self._on_load_finished)
        self._loader_worker.failed.connect(self._on_load_failed)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.failed.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._cleanup_loader)

        self._loader_thread.start()

    def _on_scan_ready(self, total: int) -> None:
        if self._directory_load_cancel_requested:
            return

        folder = self._loader_worker.folder if self._loader_worker is not None else None
        if folder is None:
            return

        self._root_directory = folder
        self._update_window_title(folder)
        self._clear_loaded_directory_data(reset_root=False)

        # _clear_loaded_directory_data already calls _refresh_tag_completions() on
        # the now-empty known_tags — no need to call it again here.

        self.statusBar().showMessage(f"Loading {total} images...")
        self._loader_worker.allow_processing()

    def _on_scan_progress(self, files: int, directories: int, images: int) -> None:
        if self._directory_load_cancel_requested:
            return
        self.statusBar().showMessage(
            f"Scanning folder... {files} files, {directories} directories, {images} images"
        )

    def _on_collision_detected(self, first: str, second: str) -> None:
        self._directory_loading_active = False
        self.statusBar().showMessage("Duplicate images detected")
        self._set_loading_state(False)
        if self._loader_worker is not None:
            self._loader_worker.cancel()
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._loader_thread.quit()

        QMessageBox.warning(
            self,
            "Duplicate image names detected",
            "Two image files have the same name with different extensions, which would "
            "collide on the same .txt description file.\n\n"
            f"Filename: {Path(first).stem}\n"
            f"Files:\n- {first}\n- {second}",
        )

    def _add_list_item(
        self,
        record: ImageRecord,
        thumbnail: QImage | None = None,
        active_badges: frozenset[str] | None = None,
        is_visible: bool | None = None,
    ) -> None:
        if active_badges is None:
            badge_specs = self._list_item_badge_specs(record)
            active_badges = frozenset(s.text for s in badge_specs)

        # Pre-scale thumbnail once so the delegate draws it without per-paint work.
        pixmap: QPixmap | None = None
        if thumbnail is not None and not thumbnail.isNull():
            raw = QPixmap.fromImage(thumbnail)
            if not raw.isNull():
                pixmap = raw.scaled(
                    THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

        item = QListWidgetItem()
        item.setData(_ROLE_PIXMAP, pixmap)
        item.setData(_ROLE_BADGES, active_badges)
        item.setData(Qt.ItemDataRole.DisplayRole, self._build_list_item_title(record))
        # Tooltip is built lazily on first hover by _ImageRowDelegate.helpEvent.
        item.setSizeHint(QSize(0, THUMB_SIZE.height() + 10))
        # Use pre-computed visibility when provided (avoids re-parsing the filter per item).
        visible = is_visible if is_visible is not None else self._record_matches_filter(record)
        item.setHidden(not visible)
        self.list_widget.addItem(item)

    def _set_loading_state(self, loading: bool) -> None:
        if self.open_action is not None:
            self.open_action.setEnabled(not loading)
        if self.refresh_action is not None:
            self.refresh_action.setEnabled(not loading)
        if self.save_action is not None:
            self.save_action.setEnabled(not loading)
        self.list_widget.setEnabled(not loading)
        self.tag_input.setEnabled(not loading)
        self.tag_list.setEnabled(not loading)
        self.llm_endpoint_input.setEnabled(not loading and self._llm_thread is None)
        self.llm_fetch_button.setEnabled(not loading and self._llm_thread is None)
        self.llm_model_combo.setEnabled(not loading and self._llm_thread is None)
        self.llm_timeout_input.setEnabled(not loading and self._llm_thread is None)
        self.llm_retry_input.setEnabled(not loading and self._llm_thread is None)
        self.llm_max_resolution_input.setEnabled(not loading and self._llm_thread is None)
        self.llm_use_button.setEnabled(not loading and self._llm_thread is None)
        self.ai_find_input.setEnabled(not loading and self._llm_thread is None)
        self.ai_find_button.setEnabled(not loading and self._llm_thread is None and bool(self.llm_model_name.strip()))
        self.fixup_button.setEnabled(False)
        self.stop_loading_button.setVisible(loading)
        self.stop_loading_button.setEnabled(loading and self._directory_loading_active)
        if loading:
            self.generate_button.setEnabled(False)
            self.validate_button.setEnabled(False)
            self.ai_find_button.setEnabled(False)
        else:
            self._update_llm_controls()

    def _clear_loaded_directory_data(self, reset_root: bool) -> None:
        self._icc_warning_paths = []
        self.records = []
        self.known_tags.clear()
        self.tag_counts.clear()
        self.list_widget.clear()
        self.image_label.setText("No image selected")
        self.tag_input.clear()
        self.tag_list.clear()
        self._clear_vision_fields()
        self.current_index = -1
        self._set_watched_image(None)
        if reset_root:
            self._root_directory = None
            self._update_window_title(self._active_directory())
        self._refresh_tag_completions()

    def _request_stop_directory_loading(self) -> None:
        if not self._directory_loading_active:
            return

        self._directory_load_cancel_requested = True
        self._directory_load_cancelled_by_user = True

        if self._loader_worker is not None:
            self._loader_worker.cancel()

        # Aborted loads should not auto-resume on next app start.
        self._cfg["last_open_directory"] = ""
        self._cfg.pop("last_selected_image", None)
        _config.save(self._cfg)

        self._clear_loaded_directory_data(reset_root=True)
        self.statusBar().showMessage("Stopping folder load and discarding partial results...")
        self.stop_loading_button.setEnabled(False)

    def _apply_tag_list_height(self) -> None:
        if self._right_splitter_initialized:
            return

        total_height = max(420, self.right_panel.height() if self.right_panel.height() > 0 else self.height())
        top_height = max(180, int(total_height * 0.6))
        bottom_height = max(180, total_height - top_height)
        self.right_splitter.setSizes([top_height, bottom_height])
        self._right_splitter_initialized = True

    def _on_item_loaded(self, payload: object) -> None:
        if self._directory_load_cancel_requested:
            return

        # payload is a list[dict] batch emitted per chunk from FolderLoadWorker.
        items = payload if isinstance(payload, list) else []
        if not items:
            return

        # Parse the filter expression once per batch instead of once per item.
        expression = self.filter_input.text().strip()
        batch_parsed: _FilterNode | None = None
        batch_runtime: _FilterRuntime | None = None
        if expression:
            try:
                batch_parsed = _parse_filter_expression(expression)
                if batch_parsed is not None:
                    batch_runtime = self._build_filter_runtime()
            except FilterSyntaxError:
                pass

        self.list_widget.setUpdatesEnabled(False)
        try:
            for data in items:
                if self._directory_load_cancel_requested:
                    break

                if not isinstance(data, dict):
                    continue
                image_path_str = str(data.get("image_path", "")).strip()
                text_path_str = str(data.get("text_path", "")).strip()
                text = str(data.get("text", ""))
                thumb_payload = data.get("thumbnail")
                thumb_image: QImage | None = None
                if isinstance(thumb_payload, dict):
                    try:
                        b = thumb_payload.get("bytes")
                        width = int(thumb_payload.get("width", 0))
                        height = int(thumb_payload.get("height", 0))
                        bytes_per_line = int(thumb_payload.get("bytes_per_line", width * 4))
                        has_alpha = bool(thumb_payload.get("has_alpha", True))
                        if b is not None and width > 0 and height > 0 and bytes_per_line > 0:
                            fmt = QImage.Format.Format_RGBA8888 if has_alpha else QImage.Format.Format_RGB888
                            qimage = QImage(
                                b,
                                width,
                                height,
                                bytes_per_line,
                                fmt,
                            )
                            # Detach from the underlying bytes buffer to avoid lifetime issues.
                            thumb_image = qimage.copy()
                    except Exception:
                        thumb_image = None

                if not image_path_str or not text_path_str:
                    continue

                image_path = Path(image_path_str)
                text_path = Path(text_path_str)

                # active_badges and has_pending_fixup were computed in the worker thread
                # alongside the thumbnail; no main-thread sidecar I/O needed here.
                active_badges_raw = data.get("active_badges")
                active_badges: frozenset[str] | None = (
                    active_badges_raw
                    if isinstance(active_badges_raw, frozenset)
                    else None
                )
                has_pending_fixup_raw = data.get("has_pending_fixup")

                record = ImageRecord(image_path=image_path, text_path=text_path, text=text)
                # Pre-populate the lazy sidecar cache on the record so the first
                # access to record.has_pending_fixup never hits the filesystem.
                if isinstance(has_pending_fixup_raw, bool):
                    record._sidecar_has_pending_fixup = has_pending_fixup_raw
                self.records.append(record)

                # Compute visibility with the pre-parsed filter (avoids O(n) re-parses).
                if batch_parsed is not None and batch_runtime is not None:
                    is_visible: bool | None = batch_parsed.evaluate(record, batch_runtime)
                elif expression:
                    is_visible = True  # parse failed → show everything
                else:
                    is_visible = True
                self._add_list_item(record, thumb_image, active_badges=active_badges, is_visible=is_visible)

                # Accumulate tag counts incrementally so _on_load_finished can skip
                # the full O(n) rebuild over all records.
                parsed_tags = parse_tags_text(text)
                self.tag_counts.update(parsed_tags)
                self.known_tags.update(parsed_tags)
        finally:
            self.list_widget.setUpdatesEnabled(True)

    def _on_load_progress(self, processed: int, total: int, percent: int) -> None:
        if self._directory_load_cancel_requested:
            return

        self.statusBar().showMessage(
            f"Processed {processed} images of {total}, {percent}% done"
        )

    def _on_icc_warning_detected(self, image_path: str) -> None:
        normalized = image_path.strip()
        if not normalized:
            return
        if normalized not in self._icc_warning_paths:
            self._icc_warning_paths.append(normalized)
        self.statusBar().showMessage(f"Warning: invalid ICC profile in {Path(normalized).name}")

    def _on_load_finished(self, total: int, folder: str) -> None:
        self._directory_loading_active = False
        self._set_loading_state(False)
        if self._directory_load_cancel_requested:
            self._pending_selection_path = None
            self._icc_warning_paths = []
            self.statusBar().showMessage("Folder loading stopped")
            return

        # Persist startup folder only for completed (non-aborted) loads.
        self._cfg["last_open_directory"] = folder
        _config.save(self._cfg)

        # Tag counts were built incrementally in _on_item_loaded; just refresh
        # the sorted completion list and known-tags UI without re-iterating records.
        self._refresh_tag_completions()
        self._apply_tag_list_height()
        self._restore_selection_after_load()
        if self.records:
            self.statusBar().showMessage(f"Loaded {len(self.records)} images from {folder}")
            if self._icc_warning_paths:
                affected = len(self._icc_warning_paths)
                preview_lines = [Path(path).name for path in self._icc_warning_paths[:10]]
                details = "\n".join(preview_lines)
                if affected > 10:
                    details += f"\n... and {affected - 10} more"
                QMessageBox.warning(
                    self,
                    "Invalid ICC profiles detected",
                    "Some images contain invalid ICC profiles.\n"
                    "These images may trigger stderr warnings and should be fixed before downstream use.\n\n"
                    f"Affected images ({affected}):\n{details}",
                )
        else:
            self.statusBar().showMessage("No supported images found in selected folder")
        self._icc_warning_paths = []

    def _on_load_failed(self, message: str) -> None:
        self._directory_loading_active = False
        self._set_loading_state(False)
        self._pending_selection_path = None
        self._icc_warning_paths = []
        QMessageBox.critical(self, "Folder load failed", message)
        self.statusBar().showMessage("Folder load failed")

    def _cleanup_loader(self) -> None:
        was_active = self._directory_loading_active
        if self._loader_worker is not None:
            self._loader_worker.deleteLater()
        if self._loader_thread is not None:
            self._loader_thread.deleteLater()
        self._loader_worker = None
        self._loader_thread = None

        if was_active:
            self._directory_loading_active = False
            self._set_loading_state(False)
            if self._directory_load_cancelled_by_user:
                self.statusBar().showMessage("Folder loading stopped")

    def _restore_selection_after_load(self) -> None:
        if not self.records:
            self._pending_selection_path = None
            return

        target_path = self._pending_selection_path
        self._pending_selection_path = None
        if target_path is not None:
            for index, record in enumerate(self.records):
                if record.image_path == target_path:
                    item = self.list_widget.item(index)
                    if item is not None and not item.isHidden():
                        self.list_widget.setCurrentRow(index)
                        return

        first_visible = self._first_visible_row()
        if first_visible >= 0:
            self.list_widget.setCurrentRow(first_visible)

    @staticmethod
    def _load_normalized_pixmap(image_path: Path) -> QPixmap:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return pixmap
        # Normalize high-DPI asset naming semantics (for example "@2x") so
        # previews and thumbnails use consistent logical sizing across formats.
        pixmap.setDevicePixelRatio(1.0)
        return pixmap

    def _build_thumbnail_icon(self, image_path: Path) -> QIcon:
        pixmap = self._load_normalized_pixmap(image_path)
        if pixmap.isNull():
            return QIcon()

        thumb = pixmap.scaled(
            THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(thumb)

    def _badge_specs_from_active_set(self, active_badges: frozenset[str]) -> list[_ListItemBadgeSpec]:
        """Build badge specs from a pre-computed set of active symbols using cached palette colors."""
        # Ensure palette-derived colors are cached.
        if self._cached_danger_color is None:
            palette = self.palette()
            self._cached_danger_color = danger_accent_color(palette)
            self._cached_danger_fg_color = danger_text_on_accent_color(palette)
            self._cached_info_color = info_accent_color(palette)
            self._cached_info_fg_color = info_text_on_accent_color(palette)
            self._cached_success_color = success_accent_color(palette)
            self._cached_success_fg_color = success_text_on_accent_color(palette)
        danger_color: QColor = self._cached_danger_color  # type: ignore[assignment]
        danger_fg: QColor = self._cached_danger_fg_color  # type: ignore[assignment]
        info_color: QColor = self._cached_info_color  # type: ignore[assignment]
        info_fg: QColor = self._cached_info_fg_color  # type: ignore[assignment]
        success_color: QColor = self._cached_success_color  # type: ignore[assignment]
        success_fg: QColor = self._cached_success_fg_color  # type: ignore[assignment]

        specs: list[_ListItemBadgeSpec] = []
        for symbol in self._IMAGE_ROW_BADGE_SLOT_ORDER:
            if symbol not in active_badges:
                continue
            if symbol == "⚖️":
                specs.append(_ListItemBadgeSpec(text=symbol, background=danger_color, foreground=danger_fg))
            elif symbol == "✅":
                specs.append(_ListItemBadgeSpec(text=symbol, background=success_color, foreground=success_fg))
            else:
                specs.append(_ListItemBadgeSpec(text=symbol, background=info_color, foreground=info_fg))
        return specs

    def _list_item_badge_specs(self, record: ImageRecord) -> list[_ListItemBadgeSpec]:
        sidecar = read_sidecar_data(record.image_path)
        active: set[str] = set()
        if sidecar.fixup_issues or sidecar.fixup_tags or sidecar.fixup_description:
            active.add("⚖️")
        if sidecar.vision_tags or (sidecar.vision_caption or "").strip():
            active.add("✨")
        if sidecar.ai_find_matches:
            active.add("🔍")
        if sidecar.validated is not None:
            active.add("✅")
        return self._badge_specs_from_active_set(frozenset(active))

    def _build_list_item_thumbnail(self, record: ImageRecord, thumbnail: QImage | None = None) -> QPixmap:
        if thumbnail is not None and not thumbnail.isNull():
            return QPixmap.fromImage(thumbnail)
        base_icon = self._build_thumbnail_icon(record.image_path)
        return base_icon.pixmap(THUMB_SIZE)

    @staticmethod
    def _badge_label_for_symbol(symbol: str) -> str:
        if symbol == "⚖️":
            return "fixup"
        if symbol == "✨":
            return "vision"
        if symbol == "🔍":
            return "search"
        if symbol == "✅":
            return "validated"
        return symbol

    def _create_badge_chip_label(self, badge: _ListItemBadgeSpec, parent: QWidget) -> QLabel:
        label = QLabel(badge.text, parent)
        label.setObjectName("imageRowBadgeLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        accent = badge.background.name(QColor.NameFormat.HexRgb)
        label.setStyleSheet(
            "QLabel#imageRowBadgeLabel {"
            "background-color: transparent;"
            f"color: {accent};"
            "border-radius: 6px;"
            "padding: 1px 8px;"
            "font-weight: 600;"
            "}"
        )
        return label

    def _create_badge_placeholder_label(self, parent: QWidget) -> QLabel:
        placeholder = QLabel(" ", parent)
        placeholder.setObjectName("imageRowBadgePlaceholder")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        placeholder.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        placeholder.setStyleSheet(
            "QLabel#imageRowBadgePlaceholder {"
            "background-color: transparent;"
            "border: 1px solid transparent;"
            "border-radius: 6px;"
            "padding: 1px 8px;"
            "}"
        )
        return placeholder

    def _set_image_list_row_widget(self, index: int, thumbnail: QImage | None = None) -> None:
        item = self.list_widget.item(index)
        if item is None or index < 0 or index >= len(self.records):
            return
        record = self.records[index]
        badge_specs = self._list_item_badge_specs(record)
        active_badges = frozenset(s.text for s in badge_specs)
        item.setData(_ROLE_BADGES, active_badges)
        item.setData(Qt.ItemDataRole.DisplayRole, self._build_list_item_title(record))
        item.setToolTip(self._build_list_item_tooltip(record, badge_specs=badge_specs))
        if thumbnail is not None and not thumbnail.isNull():
            raw = QPixmap.fromImage(thumbnail)
            if not raw.isNull():
                item.setData(
                    _ROLE_PIXMAP,
                    raw.scaled(
                        THUMB_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    ),
                )

    def _build_list_item_tooltip(self, record: ImageRecord, badge_specs: list | None = None) -> str:
        parts = [str(record.image_path)]
        badges = badge_specs if badge_specs is not None else self._list_item_badge_specs(record)
        if badges:
            active_symbols = {badge.text for badge in badges}
            legend = [f"{badge.text} {self._badge_label_for_symbol(badge.text)}" for badge in badges]
            parts.append(f"Badges: {', '.join(legend)}")
            if "✅" in active_symbols:
                sidecar = read_sidecar_data(record.image_path)
                if sidecar.validated is not None:
                    date_str = sidecar.validated.split("T")[0] if "T" in sidecar.validated else sidecar.validated
                    by = sidecar.validated_by or "unknown"
                    parts.append(f"Validated by {by} on {date_str}")
        return "\n".join(parts)

    def _update_image_label_tooltip(self, sidecar: object | None = None) -> None:
        """Set or clear the image preview tooltip based on the validated state of the current image."""
        from imagetagger.utils.sidecar import SidecarData
        if sidecar is None:
            record = self._current_record()
            if record is None:
                self.image_label.setToolTip("")
                return
            sidecar = read_sidecar_data(record.image_path)
        if not isinstance(sidecar, SidecarData) or sidecar.validated is None:
            self.image_label.setToolTip("")
            return
        date_str = sidecar.validated.split("T")[0] if "T" in sidecar.validated else sidecar.validated
        by = sidecar.validated_by or "unknown"
        self.image_label.setToolTip(f"Validated by {by} on {date_str}")

    def _build_preview_text(self, text: str, max_chars: int = 42) -> str:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return "(no description)"
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1] + "..."

    def _build_list_item_title(self, record: ImageRecord) -> str:
        preview = self._build_preview_text(record.text)
        return f"{record.image_path.name}\n{preview}"

    def on_selection_changed(self, index: int) -> None:
        if self._ignore_selection_sync:
            return
        if index < 0 or index >= len(self.records):
            self.current_index = -1
            self._set_watched_image(None)
            self._update_fixup_button_state()
            if hasattr(self, "vision_description"):
                self._clear_vision_fields()
            if self.records:
                self.statusBar().showMessage(self._filtered_total_status_text())
            return

        self.current_index = index
        record = self.records[index]
        self._set_watched_image(record.image_path)

        self._show_image(record.image_path)

        tags = self._parse_annotations_for_tag_list(record.text)
        self._populate_tag_list(tags)
        self.tag_input.clear()
        self._load_vision_for_current_image()
        self._update_fixup_button_state()
        self.statusBar().showMessage(f"{self._filtered_total_status_text(selected_row=index)} {record.image_path}")

    def _show_image(self, image_path: Path) -> None:
        # Re-use the cached pixmap when the same image is requested (e.g. during
        # a window resize) to avoid a disk read on every resize event.
        if self._cached_pixmap_path != image_path or self._cached_pixmap is None:
            pixmap = self._load_normalized_pixmap(image_path)
            if pixmap.isNull():
                self.image_label.setText(f"Unable to load image:\n{image_path.name}")
                self.image_label.setPixmap(QPixmap())
                self._cached_pixmap = None
                self._cached_pixmap_path = None
                return
            self._cached_pixmap = pixmap
            self._cached_pixmap_path = image_path
        else:
            pixmap = self._cached_pixmap

        target_size = self.image_label.size()
        if target_size.width() < 50 or target_size.height() < 50:
            return

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _on_resize_timer(self) -> None:
        """Called after the resize debounce delay to re-scale the current image."""
        if 0 <= self.current_index < len(self.records):
            self._show_image(self.records[self.current_index].image_path)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_tag_list_height()
        self._update_tag_item_heights()
        if 0 <= self.current_index < len(self.records):
            # Debounce: reschedule the timer so a burst of resize events only
            # triggers a single re-scale after the user stops dragging.
            self._resize_timer.start()

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            # Invalidate cached badge colors so they are recomputed from the new palette.
            self._cached_danger_color = None
            self._cached_danger_fg_color = None
            self._cached_info_color = None
            self._cached_info_fg_color = None
            self._cached_success_color = None
            self._cached_success_fg_color = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._llm_cancel is not None:
            self._llm_cancel.cancel()
        if self._loader_worker is not None:
            self._loader_worker.cancel()

        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(2000)
        if self._llm_thread is not None and self._llm_thread.isRunning():
            self._llm_thread.quit()
            self._llm_thread.wait(2000)

        self._save_main_window_geometry()
        record = self._current_record()
        if record is not None:
            self._cfg["last_selected_image"] = str(record.image_path)
        else:
            self._cfg.pop("last_selected_image", None)

        configured_downscale = self._cfg.get("llm_max_resolution_mpx", 5)
        try:
            fallback_value = float(configured_downscale)
            if fallback_value <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            fallback_value = 5.0

        try:
            self._cfg["llm_max_resolution_mpx"] = self._llm_max_resolution_mpx_value(show_message=False)
        except LlmProviderError:
            self._cfg["llm_max_resolution_mpx"] = fallback_value

        configured_threads = self._cfg.get("llm_threads", self._cfg.get("ollama_threads", 1))
        try:
            fallback_threads = int(configured_threads)
            if fallback_threads < 0:
                raise ValueError()
        except (TypeError, ValueError):
            fallback_threads = 1

        try:
            self._cfg["llm_threads"] = self._llm_thread_count(show_message=False)
        except LlmProviderError:
            self._cfg["llm_threads"] = fallback_threads

        _config.save(self._cfg)
        super().closeEvent(event)

    def _populate_tag_list(self, tags: list[str]) -> None:
        self._updating_tag_list = True
        self.tag_list.clear()
        for tag in tags:
            item = QListWidgetItem(tag)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.tag_list.addItem(item)
        self._update_tag_item_heights()
        self._updating_tag_list = False

    def _parse_tags(self, text: str) -> list[str]:
        return parse_tags_text(text)

    def _parse_annotations_for_tag_list(self, text: str) -> list[str]:
        parts = self._split_record_annotations(text)
        if not parts:
            return []

        description_index = self._find_description_like_index(parts)
        description_text = ""
        if description_index is not None:
            description_text = sanitize_description_text(parts[description_index])

        parsed: list[str] = []
        if description_text:
            parsed.append(description_text)

        description_key = sanitize_annotation_text(description_text).casefold() if description_text else ""
        seen: set[str] = set()
        for idx, value in enumerate(parts):
            if idx == description_index:
                continue
            normalized_tag = sanitize_tag_text(value)
            if not normalized_tag:
                continue
            if description_key and sanitize_annotation_text(normalized_tag).casefold() == description_key:
                continue
            key = normalized_tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            parsed.append(normalized_tag)

        return parsed

    def _sanitize_annotation_text(self, text: str) -> str:
        return sanitize_annotation_text(text)

    def _serialize_tags(self, tags: list[str]) -> str:
        return ", ".join(tags)

    def _current_tags(self) -> list[str]:
        return [self.tag_list.item(i).text() for i in range(self.tag_list.count())]

    def _set_current_tags(self, tags: list[str], status_prefix: str = "Auto-saved") -> None:
        if self.current_index < 0 or self.current_index >= len(self.records):
            return
        self._populate_tag_list(tags)
        self._sync_record_from_tag_list(status_prefix=status_prefix, rebuild_completions=False)

    def _sync_record_from_tag_list(self, status_prefix: str = "Auto-saved", rebuild_completions: bool = True) -> None:
        if self.current_index < 0 or self.current_index >= len(self.records):
            return

        tags = self._current_tags()
        record = self.records[self.current_index]
        new_text = self._serialize_tags(tags)
        old_parsed = self._parse_tags(record.text)
        new_parsed = self._parse_tags(new_text)
        # Ignore format-only churn so loading/selection does not trigger unsolicited saves.
        if old_parsed == new_parsed:
            return

        record.text = new_text
        self._update_list_item_preview(self.current_index)
        if rebuild_completions:
            # Incremental update: subtract old tags, add new tags — avoids O(n)
            # full rebuild on every single-image tag edit.
            self.tag_counts.subtract(old_parsed)
            self.tag_counts.update(new_parsed)
            # Remove zeroed/negative entries produced by subtract().
            zero_keys = [k for k, v in self.tag_counts.items() if v <= 0]
            for k in zero_keys:
                del self.tag_counts[k]
            self.known_tags = set(self.tag_counts)
            self._refresh_tag_completions()
        self._write_record_text(record, status_prefix=status_prefix)

    def _rebuild_known_tags_from_records(self) -> None:
        counts: Counter[str] = Counter()
        for record in self.records:
            counts.update(self._parse_tags(record.text))
        self.tag_counts = counts
        self.known_tags = set(counts)

    def _sorted_tag_suggestions(self) -> list[str]:
        return sorted(
            self.known_tags,
            key=lambda tag: (-self.tag_counts.get(tag, 0), tag.lower(), tag),
        )

    def _refresh_tag_completions(self) -> None:
        suggestions = self._sorted_tag_suggestions()
        self.tag_suggestions_model.setStringList(suggestions)
        self._refresh_known_tags_list()

    def _refresh_known_tags_list(self) -> None:
        if not hasattr(self, "known_tags_list"):
            return

        filter_text = ""
        if hasattr(self, "known_tags_filter"):
            filter_text = self.known_tags_filter.text().strip().casefold()

        self.known_tags_list.setUpdatesEnabled(False)
        try:
            self.known_tags_list.clear()
            for tag in self._sorted_tag_suggestions():
                if filter_text and filter_text not in tag.casefold():
                    continue
                count = self.tag_counts.get(tag, 0)
                item = QListWidgetItem(f"{tag} ({count})")
                item.setData(Qt.ItemDataRole.UserRole, tag)
                self.known_tags_list.addItem(item)
        finally:
            self.known_tags_list.setUpdatesEnabled(True)

    def _delete_global_tag(self, tags_to_delete: list[str]) -> None:
        tags_set = set(tags_to_delete)
        affected = [r for r in self.records if tags_set & set(self._parse_tags(r.text))]
        if not affected:
            return
        tag_count = len(tags_set)
        tag_label = f'"{next(iter(tags_set))}"' if tag_count == 1 else f"{tag_count} tags"
        reply = QMessageBox.question(
            self,
            "Remove tags from all images",
            f'Remove {tag_label} from {len(affected)} image(s)?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Compute new text for every affected record on the main thread (fast, no I/O).
        jobs: list[tuple] = []
        for record in affected:
            new_tags = [t for t in self._parse_tags(record.text) if t not in tags_set]
            new_text = self._serialize_tags(new_tags)
            record.text = new_text
            jobs.append((record.text_path, new_text))

        # Refresh UI immediately so the user sees the change right away.
        self._rebuild_known_tags_from_records()
        self._refresh_tag_completions()
        if self.current_index >= 0 and self.current_index < len(self.records):
            record = self.records[self.current_index]
            if record in affected:
                self._populate_tag_list(self._parse_tags(record.text))

        total = len(jobs)
        self.statusBar().showMessage(f'Removing {tag_label} — writing 0 / {total}…')

        worker = TagPurgeWorker(jobs)
        thread = QThread(self)
        worker.moveToThread(thread)

        def on_progress(done: int, total: int) -> None:
            self.statusBar().showMessage(f'Removing {tag_label} — writing {done} / {total}…')

        def on_finished() -> None:
            self.statusBar().showMessage(f'Removed {tag_label} from {total} file(s).')
            thread.quit()

        def on_failed(msg: str) -> None:
            QMessageBox.critical(self, "Save failed", f"Could not write some files:\n{msg}")
            thread.quit()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # Keep references so Python doesn't GC the thread/worker before they finish.
        self._purge_thread = thread
        self._purge_worker = worker

        thread.start()

    def _add_tag_from_input(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.records):
            return

        new_tag = sanitize_tag_text(self.tag_input.text())
        if not new_tag:
            return

        existing_keys = {
            sanitize_tag_text(existing_tag)
            for existing_tag in self._current_tags()
            if sanitize_tag_text(existing_tag)
        }
        if new_tag in existing_keys:
            self.statusBar().showMessage(f"Tag already exists: {new_tag}")
            self.tag_input.selectAll()
            return

        item = QListWidgetItem(new_tag)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.tag_list.addItem(item)
        self.tag_input.clear()
        QTimer.singleShot(0, self.tag_input.clear)
        self._update_tag_item_heights()
        self._sync_record_from_tag_list()

    def _on_tag_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating_tag_list:
            return

        new_text = item.text().strip()
        row = self.tag_list.row(item)

        if not new_text:
            removed = self.tag_list.takeItem(row)
            del removed
            self._update_tag_item_heights()
            self._sync_record_from_tag_list()
            return

        self._updating_tag_list = True
        item.setText(new_text)
        self._updating_tag_list = False
        self._update_tag_item_heights()
        self._sync_record_from_tag_list()

    def _remove_selected_tags(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.records):
            return

        selected = self.tag_list.selectedItems()
        if not selected:
            return

        first_row = min(self.tag_list.row(item) for item in selected)

        for item in selected:
            row = self.tag_list.row(item)
            removed = self.tag_list.takeItem(row)
            del removed

        self._update_tag_item_heights()
        self._sync_record_from_tag_list()

        count = self.tag_list.count()
        if count > 0:
            next_row = min(first_row, count - 1)
            self.tag_list.setCurrentRow(next_row)

    def _on_tags_reordered(self) -> None:
        self._update_tag_item_heights()
        self._sync_record_from_tag_list()

    def _update_tag_item_heights(self) -> None:
        viewport_width = max(60, self.tag_list.viewport().width() - 10)
        fm = self.tag_list.fontMetrics()
        for i in range(self.tag_list.count()):
            item = self.tag_list.item(i)
            text_rect = fm.boundingRect(
                QRect(0, 0, viewport_width, 10000),
                int(Qt.TextFlag.TextWordWrap),
                item.text(),
            )
            item.setSizeHint(QSize(viewport_width, max(24, text_rect.height() + 8)))

    def _update_list_item_preview(self, index: int) -> None:
        item = self.list_widget.item(index)
        if not item:
            return

        record = self.records[index]
        item.setToolTip(self._build_list_item_tooltip(record))
        self._set_image_list_row_widget(index)

    def _refresh_all_list_item_previews(self) -> None:
        for index in range(len(self.records)):
            self._update_list_item_preview(index)

    def save_current_text(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.records):
            QMessageBox.information(self, "Nothing to save", "No image selected.")
            return

        record = self.records[self.current_index]
        tags = self._current_tags()
        text = self._serialize_tags(tags)
        record.text = text

        if self._write_record_text(record, status_prefix="Saved"):
            self._update_list_item_preview(self.current_index)

    def _write_record_text(self, record: ImageRecord, status_prefix: str) -> bool:
        try:
            atomic_write_text(record.text_path, record.text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save text file:\n{exc}")
            return False

        self.statusBar().showMessage(f"{status_prefix}: {record.text_path.name}")
        return True

    def _llm_threads_status_suffix(self) -> str:
        if self._llm_action_name is None:
            return ""
        current = max(1, int(self._llm_threads_current))
        if self._llm_threads_auto_mode:
            return f" | threads: {current} auto"
        return f" | threads: {current}"

    def _run_parallel_llm_jobs(
        self,
        jobs: list[object],
        requested_threads: int,
        action_prefix: str,
        cancel_token: LlmRequestCancellation,
        report_progress: Callable[[str], None],
        report_item: Callable[[object], None],
        process_one: Callable[[int, object], dict],
    ) -> None:
        total = len(jobs)
        if total <= 0:
            return

        auto_mode = requested_threads == 0

        def cfg_int(key: str, default: int, minimum: int, maximum: int) -> int:
            raw_value = self._cfg.get(key, default)
            try:
                parsed = int(raw_value)
            except (TypeError, ValueError):
                parsed = default
            return max(minimum, min(maximum, parsed))

        def cfg_float(key: str, default: float, minimum: float, maximum: float) -> float:
            raw_value = self._cfg.get(key, default)
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                parsed = default
            return max(minimum, min(maximum, parsed))

        auto_max_threads = cfg_int("llm_auto_max_threads", int(self._cfg.get("ollama_auto_max_threads", 32)), 1, 512)
        if auto_mode:
            max_threads = min(total, auto_max_threads)
            target_parallelism = 1
        else:
            max_threads = min(total, requested_threads)
            target_parallelism = max_threads

        self._llm_threads_auto_mode = auto_mode
        self._llm_threads_current = target_parallelism

        # Auto mode uses a conservative AIMD-like controller with latency/retry guardrails.
        # Scale up by +1 every `scale_up_every` consecutive clean completions after warmup.
        # A retry resets the streak and steps down by 1; a timeout halves immediately.
        adaptive_warmup_items = cfg_int("llm_auto_warmup_items", int(self._cfg.get("ollama_auto_warmup_items", 4)), 1, 1000)
        scale_up_every = cfg_int("llm_auto_scale_up_every", int(self._cfg.get("ollama_auto_scale_up_every", 3)), 1, 100)
        consecutive_clean = 0
        backoff_epoch = 0

        def log_thread_change(previous: int, current: int, reason: str, image_name: str = "") -> None:
            if previous == current:
                return
            message = (
                f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                f"llm_threads_change action={action_prefix.lower()} from={previous} to={current} reason={reason}"
            )
            if image_name:
                message += f" image={image_name}"
            print(message, flush=True)

        executor = ThreadPoolExecutor(max_workers=max_threads)
        in_flight: set = set()
        future_backoff_epochs: dict = {}
        next_job_index = 0
        completed = 0

        try:
            while next_job_index < total and len(in_flight) < target_parallelism:
                future = executor.submit(process_one, next_job_index + 1, jobs[next_job_index])
                in_flight.add(future)
                future_backoff_epochs[future] = backoff_epoch
                next_job_index += 1

            while in_flight:
                cancel_token.raise_if_cancelled()
                finished = next(as_completed(in_flight))
                in_flight.remove(finished)
                finished_backoff_epoch = int(future_backoff_epochs.pop(finished, backoff_epoch))

                payload = finished.result()
                completed += 1

                retried = bool(payload.get("retried")) if isinstance(payload, dict) else False
                image_name = ""
                if isinstance(payload, dict):
                    image_name = str(payload.get("image_name", "")).strip()
                if auto_mode:
                    timed_out = bool(payload.get("timed_out")) if isinstance(payload, dict) else False
                    if timed_out:
                        # Only the first negative signal from a given submission epoch
                        # should trigger backoff. Other in-flight failures from the same
                        # overload window are stale signals and should not keep shrinking.
                        if finished_backoff_epoch == backoff_epoch:
                            previous_parallelism = target_parallelism
                            target_parallelism = max(1, target_parallelism // 2)
                            backoff_epoch += 1
                            consecutive_clean = 0
                            log_thread_change(
                                previous_parallelism,
                                target_parallelism,
                                f"timeout_backoff completed={completed}/{total} epoch={backoff_epoch}",
                                image_name=image_name,
                            )
                    elif retried:
                        if finished_backoff_epoch == backoff_epoch:
                            previous_parallelism = target_parallelism
                            target_parallelism = max(1, target_parallelism - 1)
                            backoff_epoch += 1
                            consecutive_clean = 0
                            log_thread_change(
                                previous_parallelism,
                                target_parallelism,
                                f"retry_backoff completed={completed}/{total} epoch={backoff_epoch}",
                                image_name=image_name,
                            )
                    elif completed >= adaptive_warmup_items:
                        consecutive_clean += 1
                        # Require scale_up_every * current_parallelism clean items before
                        # adding a thread. This naturally slows ramp-up at higher concurrency
                        # and prevents runaway scaling when the server is under load.
                        threshold = scale_up_every * max(1, target_parallelism)
                        if consecutive_clean >= threshold and target_parallelism < max_threads:
                            previous_parallelism = target_parallelism
                            target_parallelism += 1
                            consecutive_clean = 0
                            log_thread_change(
                                previous_parallelism,
                                target_parallelism,
                                f"clean_scale_up completed={completed}/{total} threshold={threshold}",
                                image_name=image_name,
                            )
                    self._llm_threads_current = target_parallelism

                if image_name:
                    report_progress(f"{action_prefix}: processing {completed}/{total} - {image_name}")
                else:
                    report_progress(f"{action_prefix}: processing {completed}/{total}")

                report_item(payload)

                while next_job_index < total and len(in_flight) < target_parallelism:
                    future = executor.submit(process_one, next_job_index + 1, jobs[next_job_index])
                    in_flight.add(future)
                    future_backoff_epochs[future] = backoff_epoch
                    next_job_index += 1
        except Exception:
            cancel_token.cancel()
            for future in in_flight:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def generate_with_llm(self) -> None:
        if self._llm_thread is not None and self._llm_action_name == "Generate":
            self._request_stop_generation()
            return

        try:
            self._apply_query_downscale_setting()
        except LlmProviderError:
            return

        try:
            thread_count = self._llm_thread_count()
        except LlmProviderError:
            return

        selected_indexes = self._selected_record_indexes()
        if not selected_indexes:
            QMessageBox.information(self, "No image selected", "Select an image before generating annotations.")
            return
        include_tags = self.generate_tags_checkbox.isChecked()
        include_description = self.generate_description_checkbox.isChecked()
        include_vision = self.generate_vision_checkbox.isChecked()
        include_refine = self.generate_refine_checkbox.isChecked()
        if not include_tags and not include_description and not include_vision and not include_refine:
            QMessageBox.information(self, "Nothing selected", "Enable Tags, Description, Vision, or Refine before generating.")
            return

        cancel_token = LlmRequestCancellation()
        session = self._active_provider_session()
        if session is None:
            QMessageBox.warning(self, "No model selected", f"Choose a {self._llm_provider.display_name} model first.")
            return

        def generate_task(
            report_progress: Callable[[str], None],
            report_item: Callable[[object], None],
        ) -> object:
            timeout = self._llm_timeout_seconds()
            retry_count = self._llm_retry_count()
            total = len(selected_indexes)
            vision_enabled = include_vision
            refine_enabled = include_refine
            agent_roles = self._cfg.get("agent_roles") or {}
            description_role = agent_roles.get("description") or None
            tagging_role = agent_roles.get("tagging") or None

            def process_one(position: int, record_index: int) -> dict:
                record = self.records[record_index]
                image_name = self._display_image_path(record.image_path)

                # Existing short tags guide LLM generation when the user pre-seeds 1-2 tags.
                existing_parts = self._split_record_annotations(record.text)
                existing_tags_hint = [
                    part for part in existing_parts
                    if not self._is_description_like_annotation(part)
                ] or None
                description_query = prepare_description_query(existing_tags=existing_tags_hint, agent_role=description_role) if include_description else None
                tags_query = prepare_tagging_query(existing_tags=existing_tags_hint, agent_role=tagging_role) if include_tags else None
                generated_description = ""
                generated_tags: list[str] = []
                vision_reasoning = ""
                vision_description = ""
                refine_tags: list[str] = []
                refine_caption = ""
                image_retried = False
                image_timed_out = False
                image_started_at = time.monotonic()

                for attempt in range(retry_count + 1):
                    cancel_token.raise_if_cancelled()
                    if attempt > 0:
                        image_retried = True

                    attempt_start = time.monotonic()

                    if attempt == 0:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_start image={image_name}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_retry image={image_name} retry={attempt}/{retry_count}",
                            flush=True,
                        )

                    def remaining_timeout(_start=attempt_start) -> float:
                        elapsed = time.monotonic() - _start
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise LlmProviderError(
                                f"Timed out after {int(timeout)} seconds while generating annotations for {record.image_path.name}."
                            )
                        return remaining

                    attempt_description = ""
                    attempt_tags: list[str] = []
                    attempt_vision_reasoning = ""
                    attempt_vision_description = ""
                    attempt_refine_tags: list[str] = []
                    attempt_refine_caption = ""
                    retry_needed = False
                    try:
                        if description_query is not None:
                            description = sanitize_description_text(
                                session.generate(
                                    record.image_path,
                                    description_query.prompt,
                                    timeout=remaining_timeout(),
                                    cancellation=cancel_token,
                                ).strip()
                            )
                            if description:
                                attempt_description = description

                        if tags_query is not None:
                            attempt_tags.extend(
                                [
                                    tag.lower()
                                    for tag in self._parse_tags(
                                        session.generate(
                                            record.image_path,
                                            tags_query.prompt,
                                            timeout=remaining_timeout(),
                                            cancellation=cancel_token,
                                        )
                                    )
                                ]
                            )

                        if vision_enabled:
                            # Vision prompt consumes the outputs of description + tags.
                            tags_input_lines: list[str] = []
                            if attempt_description:
                                tags_input_lines.append(attempt_description)
                            if attempt_tags:
                                tags_input_lines.extend(attempt_tags)
                            if not tags_input_lines:
                                # Vision-only generation should still treat existing annotations as ground truth.
                                tags_input_lines = self._parse_annotations_for_tag_list(record.text)
                            tags_input = "\n".join(tags_input_lines).strip()
                            vision_query = prepare_vision_query(tags_text=tags_input, user_hint=None)
                            vision_raw = session.generate(
                                record.image_path,
                                vision_query.prompt,
                                timeout=remaining_timeout(),
                                cancellation=cancel_token,
                            )
                            attempt_vision_reasoning, attempt_vision_description = parse_vision_response(vision_raw)

                            if attempt_vision_reasoning or attempt_vision_description:
                                # Persist immediately so multi-image batch can be applied without UI focus.
                                try:
                                    vision_data = read_sidecar_data(record.image_path)
                                    vision_data.description = attempt_vision_description
                                    vision_data.reasoning = attempt_vision_reasoning
                                    write_sidecar_data(record.image_path, vision_data)
                                except OSError as exc:
                                    raise LlmProviderError(f"Could not write {record.image_path.with_suffix('.json').name}: {exc}") from exc

                        if refine_enabled:
                            # Refine reads the sidecar .json written by Vision (or a pre-existing one).
                            sidecar = read_sidecar_data(record.image_path)
                            if sidecar.description or sidecar.reasoning:
                                refine_query = prepare_refine_query(
                                    description=sidecar.description,
                                    reasoning=sidecar.reasoning,
                                )
                                refine_raw = session.generate(
                                    record.image_path,
                                    refine_query.prompt,
                                    timeout=remaining_timeout(),
                                    cancellation=cancel_token,
                                )
                                attempt_refine_tags, attempt_refine_caption = parse_refine_response(refine_raw)

                        if not attempt_description and not attempt_tags and not (attempt_vision_reasoning or attempt_vision_description) and not attempt_refine_tags and not attempt_refine_caption:
                            retry_needed = True
                    except LlmProviderCancelled:
                        raise
                    except LlmProviderError as exc:
                        if "Timed out" in str(exc):
                            image_timed_out = True
                        retry_needed = True

                    elapsed_seconds = time.monotonic() - attempt_start
                    if not retry_needed:
                        generated_description = attempt_description
                        generated_tags = attempt_tags
                        vision_reasoning = attempt_vision_reasoning
                        vision_description = attempt_vision_description
                        refine_tags = attempt_refine_tags
                        refine_caption = attempt_refine_caption
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_done image={image_name} elapsed_s={elapsed_seconds:.2f}",
                            flush=True,
                        )
                        break

                    print(
                        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_failed image={image_name} attempt={attempt + 1}/{retry_count + 1} elapsed_s={elapsed_seconds:.2f}",
                        flush=True,
                    )

                return {
                    "kind": "generate_item",
                    "index": record_index,
                    "description": generated_description,
                    "tags": generated_tags,
                    "vision_description": vision_description,
                    "vision_reasoning": vision_reasoning,
                    "refine_tags": refine_tags,
                    "refine_caption": refine_caption,
                    "retried": image_retried,
                    "timed_out": image_timed_out,
                    "elapsed_s": max(0.0, time.monotonic() - image_started_at),
                    "position": position,
                    "total": total,
                    "image_name": image_name,
                }

            self._run_parallel_llm_jobs(
                jobs=[int(index) for index in selected_indexes],
                requested_threads=thread_count,
                action_prefix="Generate",
                cancel_token=cancel_token,
                report_progress=report_progress,
                report_item=report_item,
                process_one=lambda position, job: process_one(position, int(job)),
            )

            report_progress(f"Generate: finalizing {total}/{total}")
            return {"batch": True, "streamed": True, "total": total}

        self._generate_batch_total = len(selected_indexes)
        self._generate_batch_processed = 0
        self._generate_batch_updated = 0
        self._generate_batch_vision_updated = 0
        self._generate_batch_refine_updated = 0
        self._generate_batch_new_annotations = 0
        self._generate_batch_started_at = time.monotonic()
        self._generate_batch_retry_images = 0

        self._start_llm_task(
            task=generate_task,
            action_name="Generate",
            empty_message="Ollama returned no annotations.",
            cancel_token=cancel_token,
            merge_with_existing=True,
        )

    def validate_tags_with_llm(self) -> None:
        if self._llm_thread is not None and self._llm_action_name == "Validate":
            self._request_stop_validation()
            return

        try:
            self._apply_query_downscale_setting()
        except LlmProviderError:
            return

        try:
            thread_count = self._llm_thread_count()
        except LlmProviderError:
            return

        selected_indexes = self._selected_record_indexes()
        if not selected_indexes:
            QMessageBox.information(self, "No image selected", "Select an image before validating tags.")
            return

        annotated_records: list[tuple[int, str]] = []
        skipped_without_annotations = 0
        for record_index in selected_indexes:
            if record_index < 0 or record_index >= len(self.records):
                continue
            annotations = self.records[record_index].text
            if not annotations.strip():
                skipped_without_annotations += 1
                continue
            annotated_records.append((record_index, annotations))

        if not annotated_records:
            QMessageBox.information(self, "No annotations to validate", "Add tags or a description before validating.")
            return

        cancel_token = LlmRequestCancellation()
        session = self._active_provider_session()
        if session is None:
            QMessageBox.warning(self, "No model selected", f"Choose a {self._llm_provider.display_name} model first.")
            return

        def validate_task(
            report_progress: Callable[[str], None],
            report_item: Callable[[object], None],
        ) -> object:
            timeout = self._llm_timeout_seconds()
            retry_count = self._llm_retry_count()
            total = len(annotated_records)

            def process_one(position: int, record_index: int, annotations: str) -> dict:
                record = self.records[record_index]
                image_name = self._display_image_path(record.image_path)
                image_retried = False
                validation_result = ""
                image_timed_out = False
                image_started_at = time.monotonic()

                for attempt in range(retry_count + 1):
                    cancel_token.raise_if_cancelled()
                    if attempt > 0:
                        image_retried = True

                    attempt_start = time.monotonic()

                    if attempt == 0:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] validation_start image={image_name}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] validation_retry image={image_name} retry={attempt}/{retry_count}",
                            flush=True,
                        )

                    def remaining_timeout(_start=attempt_start) -> float:
                        elapsed = time.monotonic() - _start
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise LlmProviderError(
                                f"Timed out after {int(timeout)} seconds while validating annotations for {record.image_path.name}."
                            )
                        return remaining

                    try:
                        validation_result = session.generate(
                            record.image_path,
                            prepare_validation_query(annotations).prompt,
                            timeout=remaining_timeout(),
                            cancellation=cancel_token,
                        )

                        # Detect format hallucination and trigger retry
                        if validation_result.strip().upper() != "OK":
                            from imagetagger.utils.fixup_parser import has_fixup_section_headers
                            if not has_fixup_section_headers(validation_result):
                                raise LlmProviderError("Model hallucinated output format (missing headers).")
                    except LlmProviderCancelled:
                        raise
                    except LlmProviderError as exc:
                        if "Timed out" in str(exc):
                            image_timed_out = True
                        elapsed_seconds = time.monotonic() - attempt_start
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] validation_failed image={image_name} attempt={attempt + 1}/{retry_count + 1} elapsed_s={elapsed_seconds:.2f}",
                            flush=True,
                        )
                        if attempt >= retry_count:
                            raise
                        continue

                    elapsed_seconds = time.monotonic() - attempt_start
                    print(
                        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] validation_done image={image_name} elapsed_s={elapsed_seconds:.2f}",
                        flush=True,
                    )
                    break

                return {
                    "kind": "validate_item",
                    "index": record_index,
                    "result": validation_result,
                    "retried": image_retried,
                    "timed_out": image_timed_out,
                    "elapsed_s": max(0.0, time.monotonic() - image_started_at),
                    "position": position,
                    "total": total,
                    "image_name": image_name,
                }

            self._run_parallel_llm_jobs(
                jobs=[(int(record_index), str(annotations)) for record_index, annotations in annotated_records],
                requested_threads=thread_count,
                action_prefix="Validate",
                cancel_token=cancel_token,
                report_progress=report_progress,
                report_item=report_item,
                process_one=lambda position, job: process_one(position, int(job[0]), str(job[1])),
            )

            report_progress(f"Validate: finalizing {total}/{total}")
            return {
                "batch": True,
                "streamed": True,
                "mode": "validate",
                "total": total,
                "skipped": skipped_without_annotations,
            }

        self._validate_batch_total = len(annotated_records)
        self._validate_batch_processed = 0
        self._validate_batch_clean = 0
        self._validate_batch_issues = 0
        self._validate_batch_skipped = skipped_without_annotations
        self._validate_batch_started_at = time.monotonic()
        self._validate_batch_retry_images = 0
        self._validate_batch_llm_disobeyed = 0
        self._validate_pending_indices = {idx for idx, _ in annotated_records}

        self._start_llm_task(
            task=validate_task,
            action_name="Validate",
            empty_message="Ollama returned no validation result.",
            cancel_token=cancel_token,
            validation_report=True,
        )

    def ai_find_with_llm(self) -> None:
        if self._llm_thread is not None and self._llm_action_name == "AI Find":
            self._request_stop_ai_find()
            return

        try:
            self._apply_query_downscale_setting()
        except LlmProviderError:
            return

        try:
            thread_count = self._llm_thread_count()
        except LlmProviderError:
            return

        query = " ".join(self.ai_find_input.text().split())
        if not query:
            QMessageBox.information(self, "Missing search text", "Enter text to search for before running AI Find.")
            return

        selected_indexes = self._selected_record_indexes()
        if not selected_indexes:
            QMessageBox.information(self, "No image selected", "Select one or more images before running AI Find.")
            return

        cancel_token = LlmRequestCancellation()
        session = self._active_provider_session()
        if session is None:
            QMessageBox.warning(self, "No model selected", f"Choose a {self._llm_provider.display_name} model first.")
            return
        search_query = prepare_search_query(query)

        def find_task(
            report_progress: Callable[[str], None],
            report_item: Callable[[object], None],
        ) -> object:
            timeout = self._llm_timeout_seconds()
            retry_count = self._llm_retry_count()
            total = len(selected_indexes)

            def process_one(position: int, record_index: int) -> dict:
                record = self.records[record_index]
                image_name = record.image_path.name
                image_retried = False
                matched = False
                image_timed_out = False
                image_started_at = time.monotonic()

                for attempt in range(retry_count + 1):
                    cancel_token.raise_if_cancelled()
                    if attempt > 0:
                        image_retried = True

                    attempt_start = time.monotonic()
                    if attempt == 0:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] ai_find_start image={image_name} query={query}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] ai_find_retry image={image_name} retry={attempt}/{retry_count}",
                            flush=True,
                        )

                    def remaining_timeout(_start=attempt_start) -> float:
                        elapsed = time.monotonic() - _start
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise LlmProviderError(
                                f"Timed out after {int(timeout)} seconds while searching {record.image_path.name}."
                            )
                        return remaining

                    try:
                        matched = parse_yes_no_response(
                            session.generate(
                                record.image_path,
                                search_query.prompt,
                                timeout=remaining_timeout(),
                                cancellation=cancel_token,
                            ),
                            context="AI Find",
                        )
                    except LlmProviderCancelled:
                        raise
                    except LlmProviderError as exc:
                        if "Timed out" in str(exc):
                            image_timed_out = True
                        elapsed_seconds = time.monotonic() - attempt_start
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] ai_find_failed image={image_name} attempt={attempt + 1}/{retry_count + 1} elapsed_s={elapsed_seconds:.2f}",
                            flush=True,
                        )
                        if attempt >= retry_count:
                            raise
                        continue

                    elapsed_seconds = time.monotonic() - attempt_start
                    print(
                        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] ai_find_done image={image_name} matched={matched} elapsed_s={elapsed_seconds:.2f}",
                        flush=True,
                    )
                    break

                return {
                    "kind": "ai_find_item",
                    "index": record_index,
                    "matched": matched,
                    "retried": image_retried,
                    "timed_out": image_timed_out,
                    "elapsed_s": max(0.0, time.monotonic() - image_started_at),
                    "position": position,
                    "total": total,
                    "query": query,
                    "image_name": image_name,
                }

            self._run_parallel_llm_jobs(
                jobs=[int(index) for index in selected_indexes],
                requested_threads=thread_count,
                action_prefix="AI Find",
                cancel_token=cancel_token,
                report_progress=report_progress,
                report_item=report_item,
                process_one=lambda position, job: process_one(position, int(job)),
            )

            report_progress(f"AI Find: finalizing {total}/{total}")
            return {
                "batch": True,
                "streamed": True,
                "mode": "ai_find",
                "total": total,
                "query": query,
            }

        self._ai_find_batch_total = len(selected_indexes)
        self._ai_find_batch_processed = 0
        self._ai_find_batch_matched = 0
        self._ai_find_batch_started_at = time.monotonic()
        self._ai_find_batch_retry_images = 0

        self._start_llm_task(
            task=find_task,
            action_name="AI Find",
            empty_message="No matching images were found.",
            cancel_token=cancel_token,
        )

    def _start_llm_task(
        self,
        task: Callable[[Callable[[str], None], Callable[[object], None]], object],
        action_name: str,
        empty_message: str,
        cancel_token: LlmRequestCancellation | None = None,
        result_as_single: bool = False,
        merge_with_existing: bool = False,
        validation_report: bool = False,
    ) -> None:
        if not self.llm_model_name.strip():
            QMessageBox.warning(self, "No model selected", f"Connect to {self._llm_provider.display_name} and choose a model first.")
            return
        if self._llm_thread is not None:
            return

        resize_warning = consume_image_preparation_warning()
        if resize_warning:
            QMessageBox.warning(self, "Image resize disabled", resize_warning)

        self.statusBar().showMessage(f"{action_name} with {self._llm_provider.display_name}...")
        self.validate_button.setEnabled(False)
        self.llm_endpoint_input.setEnabled(False)
        self.llm_fetch_button.setEnabled(False)
        self.llm_model_combo.setEnabled(False)
        self.llm_timeout_input.setEnabled(False)
        self.llm_retry_input.setEnabled(False)
        self.llm_max_resolution_input.setEnabled(False)
        self.llm_threads_input.setEnabled(False)
        self.llm_use_button.setEnabled(False)
        self._llm_action_name = action_name
        self._llm_cancel = cancel_token
        self._update_llm_controls()

        self._llm_thread = QThread(self)
        self._llm_worker = LlmTaskWorker(task)
        self._llm_worker.moveToThread(self._llm_thread)
        self._llm_thread.started.connect(self._llm_worker.run)
        self._llm_worker.finished.connect(
            lambda result: self._on_llm_task_finished(
                result,
                action_name,
                empty_message,
                result_as_single,
                merge_with_existing,
                validation_report,
            )
        )
        self._llm_worker.progress.connect(self._on_llm_task_progress)
        self._llm_worker.item_ready.connect(self._on_llm_task_item_ready)
        self._llm_worker.cancelled.connect(self._on_llm_task_cancelled)
        self._llm_worker.failed.connect(self._on_llm_task_failed)
        self._llm_worker.finished.connect(self._llm_thread.quit)
        self._llm_worker.cancelled.connect(self._llm_thread.quit)
        self._llm_worker.failed.connect(self._llm_thread.quit)
        self._llm_thread.finished.connect(self._cleanup_llm_task)
        self._update_llm_controls()
        self._llm_thread.start()

    def _request_stop_generation(self) -> None:
        if self._llm_action_name != "Generate" or self._llm_cancel is None:
            return
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Stopping generation...")
        self.statusBar().showMessage("Stopping generation...")
        self._llm_cancel.cancel()

    def _request_stop_validation(self) -> None:
        if self._llm_action_name != "Validate" or self._llm_cancel is None:
            return
        self.validate_button.setEnabled(False)
        self.validate_button.setText("Stopping validation...")
        self.statusBar().showMessage("Stopping validation...")
        self._llm_cancel.cancel()

    def _request_stop_ai_find(self) -> None:
        if self._llm_action_name != "AI Find" or self._llm_cancel is None:
            return
        self.ai_find_button.setEnabled(False)
        self.ai_find_button.setText("Stopping AI Find...")
        self.statusBar().showMessage("Stopping AI Find...")
        self._llm_cancel.cancel()

    def _format_duration(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _batch_progress_details(
        self,
        processed: int,
        total: int,
        started_at: float | None,
        retry_images: int,
    ) -> str:
        if started_at is None:
            return f" | elapsed --:-- | est --:-- | retried images {retry_images}"

        elapsed = max(0.0, time.monotonic() - started_at)
        if processed > 0:
            remaining = max(0, total - processed)
            average_per_image = elapsed / processed
            estimated_remaining = average_per_image * remaining
            estimated_text = self._format_duration(estimated_remaining)
        else:
            estimated_text = "--:--"

        return (
            f" | elapsed {self._format_duration(elapsed)}"
            f" | est {estimated_text}"
            f" | retried images {retry_images}"
        )

    def _on_llm_task_progress(self, message: str) -> None:
        if self._llm_action_name == "Generate" and (
            message.startswith("Generate: processing")
            or message.startswith("Generate: retry")
            or message.startswith("Generate: finalizing")
        ):
            details = self._batch_progress_details(
                self._generate_batch_processed,
                self._generate_batch_total,
                self._generate_batch_started_at,
                self._generate_batch_retry_images,
            )
            self.statusBar().showMessage(f"{message}{details}{self._llm_threads_status_suffix()}")
            return

        if self._llm_action_name == "Validate" and (
            message.startswith("Validate: processing")
            or message.startswith("Validate: retry")
            or message.startswith("Validate: finalizing")
        ):
            details = self._batch_progress_details(
                self._validate_batch_processed,
                self._validate_batch_total,
                self._validate_batch_started_at,
                self._validate_batch_retry_images,
            )
            issues_text = f" | invalid: {self._validate_batch_issues}"
            disobeyed_text = f" | LLM disobeyed: {self._validate_batch_llm_disobeyed}" if self._validate_batch_llm_disobeyed > 0 else ""
            self.statusBar().showMessage(f"{message}{details}{issues_text}{disobeyed_text}{self._llm_threads_status_suffix()}")
            return

        if self._llm_action_name == "AI Find" and (
            message.startswith("AI Find: processing")
            or message.startswith("AI Find: retry")
            or message.startswith("AI Find: finalizing")
        ):
            details = self._batch_progress_details(
                self._ai_find_batch_processed,
                self._ai_find_batch_total,
                self._ai_find_batch_started_at,
                self._ai_find_batch_retry_images,
            )
            found_text = f" | found images: {self._ai_find_batch_matched}"
            self.statusBar().showMessage(f"{message}{details}{found_text}{self._llm_threads_status_suffix()}")
            return

        self.statusBar().showMessage(f"{message}{self._llm_threads_status_suffix()}")

    def _on_llm_task_cancelled(self, message: str) -> None:
        action = self._llm_action_name or "Request"
        if action == "Validate":
            total = self._validate_batch_total
            processed = self._validate_batch_processed
            if total > 0:
                self.statusBar().showMessage(
                    f"Validation stopped after {processed}/{total} image{'s' if total != 1 else ''}."
                )
            else:
                self.statusBar().showMessage(message or "Validation stopped.")
            return

        if action == "AI Find":
            total = self._ai_find_batch_total
            processed = self._ai_find_batch_processed
            matched = self._ai_find_batch_matched
            if total > 0:
                self.statusBar().showMessage(
                    f"AI Find stopped after {processed}/{total} image{'s' if total != 1 else ''} (found images: {matched})."
                )
            else:
                self.statusBar().showMessage(message or "AI Find stopped.")
            return

        self.statusBar().showMessage(message or f"{action} stopped.")

    @staticmethod
    def _is_description_like_annotation(text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if normalized[0].isupper() and normalized.endswith(".") and (" " in normalized or len(normalized) >= 8):
            return True
        word_count = len(normalized.split())
        return word_count >= 5 or len(normalized) >= 40

    def _find_description_like_index(self, values: list[str]) -> int | None:
        best_index: int | None = None
        best_length = -1
        for index, value in enumerate(values):
            normalized = value.strip()
            if not self._is_description_like_annotation(normalized):
                continue
            if len(normalized) > best_length:
                best_length = len(normalized)
                best_index = index
        return best_index

    def _split_record_annotations(self, text: str) -> list[str]:
        raw = text.replace("\r", "").replace("\n", ",")
        return [part.strip() for part in raw.split(",") if part.strip()]

    def _apply_generated_items_to_record(
        self,
        record_index: int,
        description: str,
        tags: list[str],
    ) -> tuple[bool, int]:
        if record_index < 0 or record_index >= len(self.records):
            return (False, 0)

        record = self.records[record_index]
        existing_annotations = self._split_record_annotations(record.text)
        existing_description_index = self._find_description_like_index(existing_annotations)
        existing_description = (
            existing_annotations[existing_description_index]
            if existing_description_index is not None
            else ""
        )

        existing_tags = [
            value
            for idx, value in enumerate(existing_annotations)
            if idx != existing_description_index
        ]
        seen = {
            sanitize_tag_text(tag).casefold()
            for tag in existing_tags
            if sanitize_tag_text(tag)
        }
        merged_tags = list(existing_tags)
        added = 0

        for item in tags:
            normalized_item = sanitize_tag_text(item)
            if not normalized_item:
                continue
            key = normalized_item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged_tags.append(normalized_item)
            added += 1

        next_description = description.strip() or existing_description
        if next_description:
            description_key = sanitize_annotation_text(next_description).casefold()
            before_filter_count = len(merged_tags)
            merged_tags = [
                tag for tag in merged_tags if sanitize_annotation_text(tag).casefold() != description_key
            ]
            removed_count = before_filter_count - len(merged_tags)
            if removed_count > 0:
                added = max(0, added - removed_count)

        description_changed = bool(next_description) and (
            sanitize_annotation_text(next_description).casefold()
            != sanitize_annotation_text(existing_description).casefold()
        )
        if description_changed:
            added += 1

        final_annotations: list[str] = []
        if next_description:
            final_annotations.append(next_description)
        final_annotations.extend(merged_tags)

        new_text = self._serialize_tags(final_annotations)
        if not new_text or new_text == record.text:
            return (False, 0)

        record.text = new_text
        if self._write_record_text(record, status_prefix="Generate + auto-saved"):
            self._update_list_item_preview(record_index)

            if self.current_index == record_index and not self.tag_input.hasFocus():
                if self.tag_list.state() != QAbstractItemView.State.EditingState:
                    self._populate_tag_list(self._parse_annotations_for_tag_list(record.text))

            return (True, added)

        return (False, 0)

    def _on_llm_task_item_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        kind = payload.get("kind")
        if kind == "generate_item":
            raw_index = payload.get("index")
            raw_description = payload.get("description")
            raw_tags = payload.get("tags")
            raw_vision_description = payload.get("vision_description")
            raw_vision_reasoning = payload.get("vision_reasoning")
            raw_retried = payload.get("retried")
            raw_position = payload.get("position")
            raw_total = payload.get("total")

            if not isinstance(raw_index, int):
                return
            if raw_index < 0 or raw_index >= len(self.records):
                return
            description = str(raw_description).strip() if isinstance(raw_description, str) else ""
            tags = [str(item).strip() for item in raw_tags] if isinstance(raw_tags, list) else []
            tags = [item for item in tags if item]

            updated, added = self._apply_generated_items_to_record(raw_index, description, tags)
            self._generate_batch_processed += 1
            if isinstance(raw_retried, bool) and raw_retried:
                self._generate_batch_retry_images += 1
            if updated:
                self._generate_batch_updated += 1
                self._generate_batch_new_annotations += added
                self._rebuild_known_tags_from_records()
                self._refresh_tag_completions()

            if isinstance(raw_position, int) and isinstance(raw_total, int) and raw_total > 0:
                details = self._batch_progress_details(
                    self._generate_batch_processed,
                    self._generate_batch_total,
                    self._generate_batch_started_at,
                    self._generate_batch_retry_images,
                )
                self.statusBar().showMessage(
                    f"Generate: applied {raw_position}/{raw_total} - {self.records[raw_index].image_path.name}{details}{self._llm_threads_status_suffix()}"
                )
            vision_desc = str(raw_vision_description).strip() if isinstance(raw_vision_description, str) else ""
            vision_reason = str(raw_vision_reasoning).strip() if isinstance(raw_vision_reasoning, str) else ""
            if vision_desc or vision_reason:
                self._generate_batch_vision_updated += 1
                if self.current_index == raw_index:
                    self._load_vision_for_current_image()

            raw_refine_tags = payload.get("refine_tags")
            raw_refine_caption = payload.get("refine_caption")
            refine_tags = [str(t).strip() for t in raw_refine_tags if str(t).strip()] if isinstance(raw_refine_tags, list) else []
            refine_caption = str(raw_refine_caption).strip() if isinstance(raw_refine_caption, str) else ""
            if refine_tags or refine_caption:
                try:
                    record_refine_result_for_image(
                        self.records[raw_index].image_path,
                        refine_tags,
                        refine_caption,
                    )
                except OSError:
                    pass
                else:
                    self._generate_batch_refine_updated += 1
                    self._on_fixup_state_changed(self.records[raw_index].image_path)
            return

        if kind == "ai_find_item":
            raw_index = payload.get("index")
            raw_matched = payload.get("matched")
            raw_retried = payload.get("retried")
            raw_position = payload.get("position")
            raw_total = payload.get("total")
            raw_query = payload.get("query")

            if not isinstance(raw_index, int):
                return
            if raw_index < 0 or raw_index >= len(self.records):
                return

            matched = bool(raw_matched)
            query = " ".join(str(raw_query or "").split())

            self._ai_find_batch_processed += 1
            if isinstance(raw_retried, bool) and raw_retried:
                self._ai_find_batch_retry_images += 1

            if matched and query:
                try:
                    record_ai_find_match_for_image(
                        self.records[raw_index].image_path,
                        query,
                        normalize_annotation=self._sanitize_annotation_text,
                    )
                except OSError:
                    pass
                else:
                    self._ai_find_batch_matched += 1
                    self._on_fixup_state_changed(self.records[raw_index].image_path)

            if isinstance(raw_position, int) and isinstance(raw_total, int) and raw_total > 0:
                details = self._batch_progress_details(
                    self._ai_find_batch_processed,
                    self._ai_find_batch_total,
                    self._ai_find_batch_started_at,
                    self._ai_find_batch_retry_images,
                )
                result_text = "match" if matched else "no match"
                found_text = f" | found images: {self._ai_find_batch_matched}"
                self.statusBar().showMessage(
                    f"AI Find: applied {raw_position}/{raw_total} - {self.records[raw_index].image_path.name} ({result_text}){details}{found_text}{self._llm_threads_status_suffix()}"
                )
            return

        if kind != "validate_item":
            return

        raw_index = payload.get("index")
        raw_result = payload.get("result")
        raw_retried = payload.get("retried")
        raw_position = payload.get("position")
        raw_total = payload.get("total")

        if not isinstance(raw_index, int):
            return
        if raw_index < 0 or raw_index >= len(self.records):
            return

        outcome, llm_violated_no_commas = self._apply_validation_result_to_record(raw_index, str(raw_result or ""))
        self._validate_pending_indices.discard(raw_index)
        self._validate_batch_processed += 1
        if isinstance(raw_retried, bool) and raw_retried:
            self._validate_batch_retry_images += 1
        if outcome == "clean":
            self._validate_batch_clean += 1
        elif outcome == "issues":
            self._validate_batch_issues += 1
        if llm_violated_no_commas:
            self._validate_batch_llm_disobeyed += 1

        if isinstance(raw_position, int) and isinstance(raw_total, int) and raw_total > 0:
            details = self._batch_progress_details(
                self._validate_batch_processed,
                self._validate_batch_total,
                self._validate_batch_started_at,
                self._validate_batch_retry_images,
            )
            self.statusBar().showMessage(
                f"Validate: processed {self._validate_batch_processed}/{raw_total}{details}{self._llm_threads_status_suffix()}"
            )

    def _on_llm_task_finished(
        self,
        result: object,
        action_name: str,
        empty_message: str,
        result_as_single: bool = False,
        merge_with_existing: bool = False,
        validation_report: bool = False,
    ) -> None:
        if (
            isinstance(result, dict)
            and result.get("batch") is True
            and result.get("streamed") is True
            and result.get("mode") == "validate"
        ):
            skipped = self._validate_batch_skipped
            total_checked = self._validate_batch_clean + self._validate_batch_issues

            if total_checked == 0:
                QMessageBox.information(self, f"{action_name} finished", empty_message)
                self.statusBar().showMessage(f"{action_name} finished")
            else:
                parts = [
                    f"{total_checked} image{'s' if total_checked != 1 else ''} checked",
                    f"{self._validate_batch_clean} clean",
                    f"{self._validate_batch_issues} fixup file{'s' if self._validate_batch_issues != 1 else ''}",
                ]
                if skipped:
                    parts.append(f"{skipped} skipped")
                self.statusBar().showMessage(
                    f"{action_name} complete via {self._llm_provider.display_name} ({', '.join(parts)})"
                )

            self._validate_batch_total = 0
            self._validate_batch_processed = 0
            self._validate_batch_clean = 0
            self._validate_batch_issues = 0
            self._validate_batch_skipped = 0
            return

        if (
            isinstance(result, dict)
            and result.get("batch") is True
            and result.get("streamed") is True
            and result.get("mode") == "ai_find"
        ):
            query = " ".join(str(result.get("query") or "").split())
            total = self._ai_find_batch_total
            matched = self._ai_find_batch_matched
            if matched <= 0:
                QMessageBox.information(self, f"{action_name} finished", empty_message)
                self.statusBar().showMessage(f"{action_name} complete (found images: 0 of {total} for '{query}')")
            else:
                self.statusBar().showMessage(
                    f"{action_name} complete (found images: {matched} of {total} for '{query}')"
                )

            self._ai_find_batch_total = 0
            self._ai_find_batch_processed = 0
            self._ai_find_batch_matched = 0
            self._ai_find_batch_retry_images = 0
            return

        if isinstance(result, dict) and result.get("batch") is True and result.get("streamed") is True:
            # Check if any output was generated (tags/description or vision)
            has_annotations = self._generate_batch_updated > 0
            has_vision = self._generate_batch_vision_updated > 0
            has_refine = self._generate_batch_refine_updated > 0

            if not has_annotations and not has_vision and not has_refine:
                QMessageBox.information(self, f"{action_name} finished", empty_message)
                self.statusBar().showMessage(f"{action_name} finished")
            else:
                parts = []
                if has_annotations:
                    parts.append(f"{self._generate_batch_updated} image{'s' if self._generate_batch_updated != 1 else ''}, {self._generate_batch_new_annotations} new annotation{'s' if self._generate_batch_new_annotations != 1 else ''}")
                if has_vision:
                    parts.append(f"{self._generate_batch_vision_updated} vision update{'s' if self._generate_batch_vision_updated != 1 else ''}")
                if has_refine:
                    parts.append(f"{self._generate_batch_refine_updated} refine fixup{'s' if self._generate_batch_refine_updated != 1 else ''}")
                summary = ", ".join(parts)
                self.statusBar().showMessage(
                    f"{action_name} complete via {self._llm_provider.display_name} ({summary})"
                )
            self._generate_batch_total = 0
            self._generate_batch_processed = 0
            self._generate_batch_updated = 0
            self._generate_batch_vision_updated = 0
            self._generate_batch_refine_updated = 0
            self._generate_batch_new_annotations = 0
            return

        if isinstance(result, dict) and result.get("batch") is True:
            batch_results = result.get("results")
            entries = batch_results if isinstance(batch_results, list) else []
            updated = 0
            with_new = 0

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                raw_index = entry.get("index")
                if not isinstance(raw_index, int):
                    continue
                if raw_index < 0 or raw_index >= len(self.records):
                    continue

                raw_items = entry.get("items")
                items = [str(item).strip() for item in raw_items] if isinstance(raw_items, list) else []
                items = [item for item in items if item]

                record = self.records[raw_index]
                existing_tags = self._parse_tags(record.text)
                seen = {tag.casefold() for tag in existing_tags}
                merged_tags = list(existing_tags)
                added = 0

                for item in items:
                    key = item.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    merged_tags.append(item)
                    added += 1

                if added == 0:
                    continue

                record.text = self._serialize_tags(merged_tags)
                if self._write_record_text(record, status_prefix="Generate + auto-saved"):
                    self._update_list_item_preview(raw_index)
                    updated += 1
                    with_new += added

            if updated == 0:
                QMessageBox.information(self, f"{action_name} finished", empty_message)
                self.statusBar().showMessage(f"{action_name} finished")
                return

            self._rebuild_known_tags_from_records()
            self._refresh_tag_completions()

            if 0 <= self.current_index < len(self.records):
                self._populate_tag_list(self._parse_annotations_for_tag_list(self.records[self.current_index].text))

            self.statusBar().showMessage(
                f"{action_name} complete via {self._llm_provider.display_name} ({updated} image{'s' if updated != 1 else ''}, {with_new} new annotation{'s' if with_new != 1 else ''})"
            )
            return

        if validation_report:
            cleaned = str(result).strip()
            if not cleaned:
                QMessageBox.information(self, f"{action_name} finished", empty_message)
                self.statusBar().showMessage(f"{action_name} finished")
                return

            if cleaned.casefold() == "ok":
                self.statusBar().showMessage("Validate complete: no issues found")
                record = self._current_record()
                if record is not None:
                    clear_validation_fields_sidecar(
                        record.image_path,
                        model=self.llm_model_name,
                        date=datetime.now().astimezone().isoformat(timespec="seconds"),
                    )
                    self._on_fixup_state_changed()
                return

            if self.current_index < 0 or self.current_index >= len(self.records):
                QMessageBox.warning(self, "Validate failed", "No selected image to write a fixup file.")
                self.statusBar().showMessage("Validate failed")
                return

            record = self.records[self.current_index]
            try:
                from imagetagger.utils.fixup_parser import parse_fixup_data
                parsed_fixup = parse_fixup_data(cleaned, self._parse_tags, self._sanitize_annotation_text)
                write_fixup_sidecar(
                    record.image_path,
                    parsed_fixup.issues or None,
                    parsed_fixup.corrected_tags or None,
                    parsed_fixup.corrected_description_raw or None,
                    model=self.llm_model_name,
                    date=datetime.now().astimezone().isoformat(timespec="seconds"),
                )
            except OSError as exc:
                QMessageBox.warning(self, "Fixup write failed", f"Could not write sidecar:\n{exc}")
                self.statusBar().showMessage("Validate failed: could not write sidecar")
                return

            self.statusBar().showMessage("Validate found issues: saved to sidecar")
            self._on_fixup_state_changed()
            return

        if isinstance(result, list):
            tags = [str(item).strip() for item in result if str(item).strip()]
        else:
            text_result = str(result)
            tags = [text_result.strip()] if result_as_single else self._parse_tags(text_result)
        if not tags:
            QMessageBox.information(self, f"{action_name} finished", empty_message)
            self.statusBar().showMessage(f"{action_name} finished")
            return

        if merge_with_existing:
            existing_tags = self._current_tags()
            seen = {tag.casefold() for tag in existing_tags}
            merged_tags = list(existing_tags)
            added_count = 0

            for tag in tags:
                normalized = tag.strip()
                if not normalized:
                    continue
                key = normalized.casefold()
                if key in seen:
                    continue
                seen.add(key)
                merged_tags.append(normalized)
                added_count += 1

            if added_count == 0:
                self.statusBar().showMessage(f"{action_name} finished (no new tags added)")
                return

            self._set_current_tags(merged_tags, status_prefix=f"{action_name} + auto-saved")
            self.statusBar().showMessage(
                f"{action_name} complete via {self._llm_provider.display_name} ({added_count} tag{'s' if added_count != 1 else ''} added)"
            )
            return

        self._set_current_tags(tags, status_prefix=f"{action_name} + auto-saved")
        self.statusBar().showMessage(f"{action_name} complete via {self._llm_provider.display_name}")

    def _on_llm_task_failed(self, message: str) -> None:
        QMessageBox.warning(self, f"{self._llm_provider.display_name} request failed", message)
        self.statusBar().showMessage(f"{self._llm_provider.display_name} request failed")

    def _apply_validation_result_to_record(self, record_index: int, result: str) -> tuple[str, bool]:
        if record_index < 0 or record_index >= len(self.records):
            return ("invalid", False)

        cleaned = result.strip()
        if not cleaned:
            return ("empty", False)

        record = self.records[record_index]
        validation_model = self.llm_model_name
        validation_date = datetime.now().astimezone().isoformat(timespec="seconds")
        if cleaned.casefold() == "ok":
            clear_validation_fields_sidecar(record.image_path, model=validation_model, date=validation_date)
            self._on_fixup_state_changed(record.image_path)
            return ("clean", False)

        # Check if LLM violated the "no commas" rule in the description
        llm_violated_no_commas = False
        parsed = None
        try:
            from imagetagger.utils.fixup_parser import parse_fixup_data
            parsed = parse_fixup_data(cleaned, self._parse_tags, self._sanitize_annotation_text)
            if parsed.corrected_description and "," in parsed.corrected_description_raw:
                llm_violated_no_commas = True
        except Exception:
            pass  # Silently ignore parsing errors, just track nothing

        try:
            if parsed is not None:
                write_fixup_sidecar(
                    record.image_path,
                    parsed.issues or None,
                    parsed.corrected_tags or None,
                    parsed.corrected_description_raw or None,
                    model=validation_model,
                    date=validation_date,
                )
            else:
                write_fixup_sidecar(record.image_path, cleaned, None, None, model=validation_model, date=validation_date)
        except OSError:
            return ("error", llm_violated_no_commas)

        self._on_fixup_state_changed(record.image_path)
        return ("issues", llm_violated_no_commas)

    def _cleanup_llm_task(self) -> None:
        if self._llm_worker is not None:
            self._llm_worker.deleteLater()
        if self._llm_thread is not None:
            self._llm_thread.deleteLater()
        self._llm_worker = None
        self._llm_thread = None
        self._generate_batch_total = 0
        self._generate_batch_processed = 0
        self._generate_batch_updated = 0
        self._generate_batch_vision_updated = 0
        self._generate_batch_new_annotations = 0
        self._generate_batch_started_at = None
        self._generate_batch_retry_images = 0
        self._validate_batch_total = 0
        self._validate_batch_processed = 0
        self._validate_batch_clean = 0
        self._validate_batch_issues = 0
        self._validate_batch_skipped = 0
        self._validate_batch_started_at = None
        self._validate_batch_retry_images = 0
        self._validate_pending_indices = set()
        self._ai_find_batch_total = 0
        self._ai_find_batch_processed = 0
        self._ai_find_batch_matched = 0
        self._ai_find_batch_started_at = None
        self._ai_find_batch_retry_images = 0
        self._llm_action_name = None
        self._llm_cancel = None
        self._llm_threads_auto_mode = False
        self._llm_threads_current = 0
        self._update_llm_controls()
        self.llm_endpoint_input.setEnabled(True)
        self.llm_fetch_button.setEnabled(True)
        self.llm_model_combo.setEnabled(True)
        self.llm_timeout_input.setEnabled(True)
        self.llm_retry_input.setEnabled(True)
        self.llm_max_resolution_input.setEnabled(True)
        self.llm_threads_input.setEnabled(True)
        self.llm_use_button.setEnabled(True)
        self._update_fixup_button_state()


def main() -> None:
    QImageReader.setAllocationLimit(1024)

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
