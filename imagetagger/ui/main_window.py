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
from imagetagger.utils.io_utils import atomic_write_text, bg_write_text
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
    render_prompt_with_agent_role,
    render_prompt_with_existing_tags,
    render_prompt_with_user_hint,
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
from imagetagger.ui.tag_controller import TagController
from imagetagger.ui.llm_controller import LlmController
from imagetagger.ui.directory_controller import DirectoryController
from imagetagger.ui.fixup_controller import FixupController
from imagetagger.ui.image_view_controller import ImageViewController
from imagetagger.utils.theme_colors import danger_accent_color, danger_text_on_accent_color, info_accent_color, info_text_on_accent_color, success_accent_color, success_text_on_accent_color





from imagetagger.utils.filter_parser import (
    FilterSyntaxError,
    _FilterNode,
    _FilterRuntime,
    _parse_filter_expression,
)
from imagetagger.ui.models import ImageRecord, _UNKNOWN




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
        self._record_index_by_path: dict[Path, int] = {}
        self.current_index: int = -1
        self._llm_provider = DEFAULT_VISION_PROVIDER
        self.known_tags: set[str] = set()
        self.tag_counts: Counter[str] = Counter()
        self._ignore_selection_sync = False
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
        self._detected_external_editors: list[ExternalEditor] | None = None

        # Image reload detection for external editor changes
        self._image_reload_helper = ImageReloadHelper(self, self._on_image_reload)
        self._prev_selected_rows: set[int] = set()

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

        self.directory_controller = DirectoryController(self)
        self.llm_controller = LlmController(self)
        self._build_ui()
        self.tag_controller = TagController(
            self,
            self.tag_input,
            self.tag_list,
            self.tag_suggestions_model,
            self.known_tags_list,
            self.known_tags_filter,
        )
        self.fixup_controller = FixupController(self)
        self.image_view_controller = ImageViewController(self, self.image_label)
        self._build_menu()
        self._apply_tag_list_height()
        self._apply_config()
        self._update_llm_controls()

    # ------------------------------------------------------------------
    # Properties delegating LLM state to LlmController (used throughout
    # MainWindow for backwards-compatible access)
    # ------------------------------------------------------------------

    @property
    def _cached_pixmap(self) -> "QPixmap | None":
        return self.image_view_controller._cached_pixmap

    @_cached_pixmap.setter
    def _cached_pixmap(self, value: "QPixmap | None") -> None:
        self.image_view_controller._cached_pixmap = value

    @property
    def _cached_pixmap_path(self) -> "Path | None":
        return self.image_view_controller._cached_pixmap_path

    @_cached_pixmap_path.setter
    def _cached_pixmap_path(self, value: "Path | None") -> None:
        self.image_view_controller._cached_pixmap_path = value

    @property
    def _fixup_navigating(self) -> bool:
        return self.fixup_controller._fixup_navigating

    @_fixup_navigating.setter
    def _fixup_navigating(self, value: bool) -> None:
        self.fixup_controller._fixup_navigating = value

    @property
    def _llm_thread(self):
        return self.llm_controller._llm_thread

    @property
    def _llm_action_name(self):
        return self.llm_controller._llm_action_name

    @property
    def _llm_cancel(self):
        return self.llm_controller._llm_cancel

    @property
    def _validate_pending_indices(self) -> set:
        return self.llm_controller._validate_pending_indices

    # ------------------------------------------------------------------
    # Properties delegating directory state to DirectoryController
    # (used in closeEvent, _display_image_path, and other retained methods)
    # ------------------------------------------------------------------

    @property
    def _root_directory(self) -> Path | None:
        return self.directory_controller._root_directory

    @property
    def _loader_thread(self):
        return self.directory_controller._loader_thread

    @property
    def _loader_worker(self):
        return self.directory_controller._loader_worker

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
        self.list_widget.installEventFilter(self)

        self.filter_input = QLineEdit(self)
        self.filter_input.setPlaceholderText("Filter (fixup, vision, validated, \"tag\", 'text', &, |, parentheses)")
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

        self.focus_tag_input_action = QAction("Focus Tag Input", self)
        self.focus_tag_input_action.setShortcut(platform_key_sequence("Alt+T", "Alt+T"))
        self.focus_tag_input_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.focus_tag_input_action.triggered.connect(self._focus_tag_input_when_autotag)
        self.addAction(self.focus_tag_input_action)

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
        self.generate_button = QPushButton("&Generate", self)
        self.generate_button.clicked.connect(self.generate_with_llm)
        gen_row.addWidget(self.generate_button)

        buttons_row = QHBoxLayout()
        self.validate_button = QPushButton("&Validate", self)
        self.validate_button.clicked.connect(self.validate_tags_with_llm)
        self.fixup_button = QPushButton("Fi&xup", self)
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
        self.known_tags_filter.installEventFilter(self)
        layout.addWidget(self.known_tags_filter)

        self.known_tags_list = GlobalTagListWidget(self)
        self.known_tags_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.known_tags_list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.known_tags_list.delete_requested.connect(self._delete_global_tag)

        layout.addWidget(self.known_tags_list, stretch=1)

        global_tag_buttons_row = QHBoxLayout()
        global_tag_buttons_row.setSpacing(4)
        self.bump_tag_button = QPushButton("Bump", self)
        self.bump_tag_button.setToolTip("Move selected tag to first position (after description) in all images that contain it")
        self.bump_tag_button.clicked.connect(self._bump_selected_tag)
        global_tag_buttons_row.addWidget(self.bump_tag_button)
        global_tag_buttons_row.addStretch(1)
        layout.addLayout(global_tag_buttons_row)

        self.controls_tabs.addTab(tab, "Tags")
        self._refresh_known_tags_list()

    def _update_prompt_status(self, kind: str, edited: bool = False) -> None:
        self.llm_controller._update_prompt_status(kind, edited)

    def _prompt_title(self, kind: str) -> str:
        return self.llm_controller._prompt_title(kind)

    def _prompt_editor_text(self, kind: str) -> str:
        return self.llm_controller._prompt_editor_text(kind)

    def _test_prompt(self, kind: str) -> None:
        self.llm_controller._test_prompt(kind)

    def _set_agent_role(self, kind: str) -> None:
        self.llm_controller._set_agent_role(kind)

    def _apply_prompt_override(self, kind: str) -> None:
        self.llm_controller._apply_prompt_override(kind)

    def _save_prompt_to_file(self, kind: str) -> None:
        self.llm_controller._save_prompt_to_file(kind)

    def _reset_prompt_to_default(self, kind: str) -> None:
        self.llm_controller._reset_prompt_to_default(kind)

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
        self.llm_controller.fetch_provider_models()

    def use_selected_provider_model(self) -> None:
        self.llm_controller.use_selected_provider_model()

    def _active_provider_session(self) -> VisionLlmSession | None:
        return self.llm_controller._active_provider_session()

    def _llm_timeout_seconds(self) -> float:
        return self.llm_controller._llm_timeout_seconds()

    def _llm_retry_count(self) -> int:
        return self.llm_controller._llm_retry_count()

    @staticmethod
    def _format_mpx(value: float) -> str:
        return InputValidator.format_megapixels(value)

    def _llm_max_resolution_mpx_value(self, show_message: bool = True) -> float:
        def show_error(msg: str) -> None:
            if show_message:
                QMessageBox.warning(self, "Invalid query downscale", msg)
        return InputValidator.parse_max_resolution_mpx(self.llm_max_resolution_input.text(), show_error)

    def _apply_query_downscale_setting(self) -> float:
        return self.llm_controller._apply_query_downscale_setting()

    def _llm_thread_count(self, show_message: bool = True) -> int:
        return self.llm_controller._llm_thread_count(show_message=show_message)

    def _update_llm_controls(self) -> None:
        if not hasattr(self, "llm_controller"):
            return
        self.llm_controller._update_llm_controls()

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

        record._sidecar_validated = _UNKNOWN
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
            "- validated: show images that have passed validation (✅ badge)\n"
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
            "- !validated\n"
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
            "validated": lambda record: record.sidecar_validated is not None,
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
        self.fixup_controller._on_fixup_state_changed(image_path)

    def _update_fixup_button_state(self) -> None:
        self.fixup_controller._update_fixup_button_state()

    def _find_adjacent_fixup_index(self, start_index: int, direction: int) -> int | None:
        return self.fixup_controller._find_adjacent_fixup_index(start_index, direction)

    def _find_fixup_index(self, reverse: bool) -> int | None:
        return self.fixup_controller._find_fixup_index(reverse)

    def _jump_to_first_fixup(self) -> None:
        self.fixup_controller._jump_to_first_fixup()

    def _jump_to_last_fixup(self) -> None:
        self.fixup_controller._jump_to_last_fixup()

    def _focus_tag_input_when_autotag(self) -> None:
        if self.controls_tabs.currentIndex() == 0:
            self.tag_input.setFocus()

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
        if watched is self.list_widget and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            modifiers = event.modifiers()
            plain = not bool(modifiers & ~Qt.KeyboardModifier.KeypadModifier)
            if plain and key == Qt.Key.Key_Home:
                first = next(
                    (i for i in range(self.list_widget.count())
                     if not self.list_widget.item(i).isHidden()),
                    None,
                )
                if first is not None:
                    self.list_widget.setCurrentRow(first)
                return True
            if plain and key == Qt.Key.Key_End:
                last = next(
                    (i for i in range(self.list_widget.count() - 1, -1, -1)
                     if not self.list_widget.item(i).isHidden()),
                    None,
                )
                if last is not None:
                    self.list_widget.setCurrentRow(last)
                return True
        if hasattr(self, "tag_input") and watched is self.tag_input and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if not self.tag_input.text().strip():
                if key == Qt.Key.Key_Up:
                    self._move_image_selection(-1)
                    return True
                if key == Qt.Key.Key_Down:
                    self._move_image_selection(1)
                    return True
        if hasattr(self, "known_tags_filter") and watched is self.known_tags_filter and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Down:
                self.known_tags_list.setFocus()
                if self.known_tags_list.currentItem() is None and self.known_tags_list.count() > 0:
                    self.known_tags_list.setCurrentRow(0)
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
        # Keep the index dict consistent: remove the deleted entry and
        # shift down all indices that were above it.
        del self._record_index_by_path[image_path]
        for p, idx in self._record_index_by_path.items():
            if idx > record_index:
                self._record_index_by_path[p] = idx - 1
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
        self.image_view_controller._set_watched_image(image_path)

    def _on_image_reload(self, image_path: Path) -> None:
        self.image_view_controller._on_image_reload(image_path)

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
        return self._record_index_by_path.get(image_path, -1)

    def _set_tags_for_image_path(self, image_path: Path, tags: list[str], status_prefix: str = "Auto-saved") -> None:
        record_index = self._record_index_for_image_path(image_path)
        if record_index < 0:
            return

        normalized_tags = [tag for tag in (item.strip() for item in tags) if tag]
        record = self.records[record_index]
        old_tags = self._parse_tags(record.text)
        record.text = self._serialize_tags(normalized_tags)

        if self.current_index == record_index:
            self._populate_tag_list(normalized_tags)

        self._update_list_item_preview(record_index)
        self._update_tag_counts_incremental(old_tags, normalized_tags)
        self._refresh_tag_completions()
        # Use background write — in-memory state is already authoritative above.
        bg_write_text(record.text_path, record.text, encoding="utf-8")
        self.statusBar().showMessage(f"{status_prefix}: {record.text_path.name}")

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
        self.fixup_controller.open_fixup_dialog()

    def open_folder(self) -> None:
        self.directory_controller.open_folder()

    def refresh_directory(self) -> None:
        self.directory_controller.refresh_directory()

    def _active_directory(self) -> Path | None:
        return self.directory_controller._active_directory()

    def _update_window_title(self, folder: Path | None) -> None:
        self.directory_controller._update_window_title(folder)

    def _find_image_name_collision(self, folder: Path) -> tuple[Path, Path] | None:
        return self.directory_controller._find_image_name_collision(folder)

    def load_directory(self, folder: Path, restore_selection: Path | None = None) -> None:
        self.directory_controller.load_directory(folder, restore_selection=restore_selection)

    def _on_scan_ready(self, total: int) -> None:
        self.directory_controller._on_scan_ready(total)

    def _on_scan_progress(self, files: int, directories: int, images: int) -> None:
        self.directory_controller._on_scan_progress(files, directories, images)

    def _on_collision_detected(self, first: str, second: str) -> None:
        self.directory_controller._on_collision_detected(first, second)

    def _set_loading_state(self, loading: bool) -> None:
        self.directory_controller._set_loading_state(loading)

    def _clear_loaded_directory_data(self, reset_root: bool) -> None:
        self.directory_controller._clear_loaded_directory_data(reset_root)

    def _request_stop_directory_loading(self) -> None:
        self.directory_controller._request_stop_directory_loading()

    def _apply_tag_list_height(self) -> None:
        if self._right_splitter_initialized:
            return

        total_height = max(420, self.right_panel.height() if self.right_panel.height() > 0 else self.height())
        top_height = max(180, int(total_height * 0.6))
        bottom_height = max(180, total_height - top_height)
        self.right_splitter.setSizes([top_height, bottom_height])
        self._right_splitter_initialized = True

    def _on_item_loaded(self, payload: object) -> None:
        self.directory_controller._on_item_loaded(payload)

    def _on_load_progress(self, processed: int, total: int, percent: int) -> None:
        self.directory_controller._on_load_progress(processed, total, percent)

    def _on_icc_warning_detected(self, image_path: str) -> None:
        self.directory_controller._on_icc_warning_detected(image_path)

    def _on_load_finished(self, total: int, folder: str) -> None:
        self.directory_controller._on_load_finished(total, folder)

    def _on_load_failed(self, message: str) -> None:
        self.directory_controller._on_load_failed(message)

    def _cleanup_loader(self) -> None:
        self.directory_controller._cleanup_loader()

    def _restore_selection_after_load(self) -> None:
        self.directory_controller._restore_selection_after_load()

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

        if not self._fixup_navigating:
            self._show_image(record.image_path)

        tags = self._parse_annotations_for_tag_list(record.text)
        self._populate_tag_list(tags)
        self.tag_input.clear()
        if not self._fixup_navigating:
            self._load_vision_for_current_image()
        self._update_fixup_button_state()
        self.statusBar().showMessage(f"{self._filtered_total_status_text(selected_row=index)} {record.image_path}")

    def _show_image(self, image_path: Path) -> None:
        self.image_view_controller._show_image(image_path)

    def _on_resize_timer(self) -> None:
        self.image_view_controller._on_resize_timer()

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
        self.tag_controller._populate_tag_list(tags)

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
        self.tag_controller._rebuild_known_tags_from_records()

    def _update_tag_counts_incremental(self, old_tags: list[str], new_tags: list[str]) -> None:
        self.tag_controller._update_tag_counts_incremental(old_tags, new_tags)

    def _sorted_tag_suggestions(self) -> list[str]:
        return self.tag_controller._sorted_tag_suggestions()

    def _refresh_tag_completions(self) -> None:
        self.tag_controller._refresh_tag_completions()

    def _refresh_known_tags_list(self) -> None:
        if not hasattr(self, "tag_controller"):
            return
        self.tag_controller._refresh_known_tags_list()

    def _delete_global_tag(self, tags_to_delete: list[str]) -> None:
        self.tag_controller._delete_global_tag(tags_to_delete)

    def _add_tag_from_input(self) -> None:
        self.tag_controller._add_tag_from_input()

    def _on_tag_item_changed(self, item: QListWidgetItem) -> None:
        self.tag_controller._on_tag_item_changed(item)

    def _remove_selected_tags(self) -> None:
        self.tag_controller._remove_selected_tags()

    def _on_tags_reordered(self) -> None:
        self.tag_controller._on_tags_reordered()

    def _bump_selected_tag(self) -> None:
        self.tag_controller._bump_selected_tag()

    def _update_tag_item_heights(self) -> None:
        self.tag_controller._update_tag_item_heights()

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
        return self.llm_controller._llm_threads_status_suffix()

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
        self.llm_controller._run_parallel_llm_jobs(
            jobs=jobs,
            requested_threads=requested_threads,
            action_prefix=action_prefix,
            cancel_token=cancel_token,
            report_progress=report_progress,
            report_item=report_item,
            process_one=process_one,
        )
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
                perf_retry = bool(payload.get("perf_retry", retried)) if isinstance(payload, dict) else retried
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
                    elif perf_retry:
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
        self.llm_controller.generate_with_llm()

    def validate_tags_with_llm(self) -> None:
        self.llm_controller.validate_tags_with_llm()

    def ai_find_with_llm(self) -> None:
        self.llm_controller.ai_find_with_llm()

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
        self.llm_controller._start_llm_task(
            task=task,
            action_name=action_name,
            empty_message=empty_message,
            cancel_token=cancel_token,
            result_as_single=result_as_single,
            merge_with_existing=merge_with_existing,
            validation_report=validation_report,
        )
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
        self.llm_controller._request_stop_generation()

    def _request_stop_validation(self) -> None:
        self.llm_controller._request_stop_validation()

    def _request_stop_ai_find(self) -> None:
        self.llm_controller._request_stop_ai_find()

    def _format_duration(self, seconds: float) -> str:
        return self.llm_controller._format_duration(seconds)

    def _batch_progress_details(
        self,
        processed: int,
        total: int,
        started_at: float | None,
        retry_images: int,
    ) -> str:
        return self.llm_controller._batch_progress_details(
            processed=processed,
            total=total,
            started_at=started_at,
            retry_images=retry_images,
        )
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
        self.llm_controller._on_llm_task_progress(message)

    def _on_llm_task_cancelled(self, message: str) -> None:
        self.llm_controller._on_llm_task_cancelled(message)

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
        self.llm_controller._on_llm_task_item_ready(payload)

    def _on_llm_task_finished(
        self,
        result: object,
        action_name: str,
        empty_message: str,
        result_as_single: bool = False,
        merge_with_existing: bool = False,
        validation_report: bool = False,
    ) -> None:
        self.llm_controller._on_llm_task_finished(
            result=result,
            action_name=action_name,
            empty_message=empty_message,
            result_as_single=result_as_single,
            merge_with_existing=merge_with_existing,
            validation_report=validation_report,
        )
        if (
            isinstance(result, dict)
            and result.get("batch") is True
            and result.get("streamed") is True
            and result.get("mode") == "validate"
        ):
            skipped = self._validate_batch_skipped
            total_checked = self._validate_batch_clean + self._validate_batch_issues
            errors = self._validate_batch_errors

            if total_checked == 0 and errors == 0:
                QMessageBox.information(self, f"{action_name} finished", empty_message)
                self.statusBar().showMessage(f"{action_name} finished")
            else:
                parts = [
                    f"{total_checked} image{'s' if total_checked != 1 else ''} checked",
                    f"{self._validate_batch_clean} clean",
                    f"{self._validate_batch_issues} fixup file{'s' if self._validate_batch_issues != 1 else ''}",
                ]
                if errors:
                    parts.append(f"{errors} error{'s' if errors != 1 else ''}")
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
            self._validate_batch_errors = 0
            ctx_exhausted = self._validate_batch_context_exhausted
            self._validate_batch_context_exhausted = 0
            if ctx_exhausted > 0:
                n = ctx_exhausted
                QMessageBox.warning(
                    self,
                    "Context window exhausted",
                    f"{n} image{'s' if n != 1 else ''} failed because the model exhausted its "
                    f"context window — thinking (CoT) tokens consumed all available space, leaving "
                    f"no room for a response.\n\n"
                    f"To fix this, increase num_ctx on your Ollama server. For example, pull the "
                    f"model with a higher context size or set num_ctx in the model options "
                    f"(e.g. 32768 or higher).",
                )
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
                if not parsed_fixup.issues and not parsed_fixup.corrected_tags and not parsed_fixup.corrected_description_raw:
                    clear_validation_fields_sidecar(
                        record.image_path,
                        model=self.llm_model_name,
                        date=datetime.now().astimezone().isoformat(timespec="seconds"),
                    )
                    self.statusBar().showMessage("Validate complete: no issues found")
                    self._on_fixup_state_changed()
                    return
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
        self.llm_controller._on_llm_task_failed(message)

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
                if not parsed.issues and not parsed.corrected_tags and not parsed.corrected_description_raw:
                    clear_validation_fields_sidecar(record.image_path, model=validation_model, date=validation_date)
                    self._on_fixup_state_changed(record.image_path)
                    return ("clean", llm_violated_no_commas)
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
        self.llm_controller._cleanup_llm_task()


def main() -> None:
    QImageReader.setAllocationLimit(1024)

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
