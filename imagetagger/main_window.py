from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
import time
from typing import Callable, List

from PIL import Image, ImageCms, UnidentifiedImageError

from PyQt6.QtCore import QEvent, QObject, QRect, QStringListModel, QThread, Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QImage, QImageReader, QIntValidator, QKeySequence, QPainter, QPixmap
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
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QStyledItemDelegate,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imagetagger import config as _config
from imagetagger.annotations import parse_tags_text, sanitize_annotation_text
from imagetagger.merge_actions import (
    clear_fixup_files_for_image,
    existing_fixup_path_for_image,
    open_fixup_dialog_for_image,
    write_fixup_for_image,
)
from imagetagger.ollama import (
    active_prompt_for_kind,
    clear_prompt_override,
    configure_runtime,
    consume_resize_warning,
    DEFAULT_TIMEOUT,
    DEFAULT_OLLAMA_SERVER,
    get_default_prompt,
    load_prompt_for_kind,
    OllamaCancelled,
    OllamaCancellation,
    OllamaError,
    fetch_models,
    generate_description,
    generate_tags,
    normalize_server_url,
    prompt_source_for_kind,
    save_prompt_for_kind,
    set_prompt_override,
    validate_tags,
    reset_prompt_to_default,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
THUMB_SIZE = QSize(96, 96)
MIN_FONT_POINT_SIZE = 8
MAX_FONT_POINT_SIZE = 40


@dataclass
class ImageRecord:
    image_path: Path
    text_path: Path
    text: str


class FolderLoadWorker(QObject):
    progress = pyqtSignal(int, int, int)
    item_loaded = pyqtSignal(object)
    finished = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    icc_warning = pyqtSignal(str)

    def __init__(self, folder: Path) -> None:
        super().__init__()
        self.folder = folder

    @staticmethod
    def _has_invalid_icc_profile(image_path: Path) -> bool:
        try:
            with Image.open(image_path) as image:
                raw_profile = image.info.get("icc_profile")
                if not raw_profile:
                    return False
                if isinstance(raw_profile, str):
                    raw_profile = raw_profile.encode("utf-8", errors="ignore")
                if not isinstance(raw_profile, (bytes, bytearray)):
                    return False

                ImageCms.ImageCmsProfile(BytesIO(bytes(raw_profile)))
                return False
        except (OSError, ValueError, UnidentifiedImageError):
            return False
        except ImageCms.PyCMSError:
            return True

    def run(self) -> None:
        try:
            image_paths = sorted(
                [
                    p
                    for p in self.folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda p: p.relative_to(self.folder).as_posix().lower(),
            )
        except OSError as exc:
            self.failed.emit(f"Failed to read folder: {exc}")
            return

        total = len(image_paths)
        self.progress.emit(0, total, 0)

        for index, image_path in enumerate(image_paths, start=1):
            if self._has_invalid_icc_profile(image_path):
                self.icc_warning.emit(str(image_path))

            text_path = image_path.with_suffix(".txt")
            text = ""
            if text_path.exists():
                try:
                    text = text_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = ""

            thumb = QImage()
            reader = QImageReader(str(image_path))
            reader.setAutoTransform(True)
            image = reader.read()
            if not image.isNull():
                thumb = image.scaled(
                    THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

            self.item_loaded.emit(
                {
                    "image_path": str(image_path),
                    "text_path": str(text_path),
                    "text": text,
                    "thumbnail": thumb,
                }
            )

            percent = int((index / total) * 100) if total else 100
            self.progress.emit(index, total, percent)

        self.finished.emit(total, str(self.folder))


class OllamaTaskWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    progress = pyqtSignal(str)
    item_ready = pyqtSignal(object)

    def __init__(self, task: Callable[[Callable[[str], None], Callable[[object], None]], object]) -> None:
        super().__init__()
        self.task = task

    def run(self) -> None:
        try:
            result = self.task(self.progress.emit, self.item_ready.emit)
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ImageTagger")
        self.resize(1400, 860)

        self.records: List[ImageRecord] = []
        self.current_index: int = -1
        self._loader_thread: QThread | None = None
        self._loader_worker: FolderLoadWorker | None = None
        self._ollama_thread: QThread | None = None
        self._ollama_worker: OllamaTaskWorker | None = None
        self.known_tags: set[str] = set()
        self.tag_counts: Counter[str] = Counter()
        self._updating_tag_list = False
        self._ignore_selection_sync = False
        self._generate_batch_total = 0
        self._generate_batch_processed = 0
        self._generate_batch_updated = 0
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
        self._ollama_action_name: str | None = None
        self._ollama_cancel: OllamaCancellation | None = None
        self._icc_warning_paths: list[str] = []
        self._right_splitter_initialized = False
        self.prompt_editors: dict[str, QTextEdit] = {}
        self.prompt_status_labels: dict[str, QLabel] = {}

        self.open_action: QAction | None = None
        self.refresh_action: QAction | None = None
        self.save_action: QAction | None = None
        self.increase_font_action: QAction | None = None
        self.decrease_font_action: QAction | None = None
        self.ollama_server_url = DEFAULT_OLLAMA_SERVER
        self.ollama_model_name = ""
        self.status_connection_label: QLabel | None = None
        self._pending_selection_path: Path | None = None
        self._root_directory: Path | None = None

        self._cfg = _config.load()

        self._build_ui()
        self._build_menu()
        self._apply_tag_list_height()
        self._apply_config()
        self._update_ollama_controls()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)

        self.list_widget = QListWidget(self)
        self.list_widget.setIconSize(THUMB_SIZE)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.currentRowChanged.connect(self.on_selection_changed)

        self.filter_input = QLineEdit(self)
        self.filter_input.setPlaceholderText("Filter (type 'fixup')")
        self.filter_input.textChanged.connect(self._apply_image_filter)

        self.left_panel = QWidget(self)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.filter_input, stretch=0)
        left_layout.addWidget(self.list_widget, stretch=1)

        self.select_all_images_action = QAction("Select All Images", self)
        self.select_all_images_action.setShortcut("Ctrl+A")
        self.select_all_images_action.triggered.connect(self._select_all_images)
        self.list_widget.addAction(self.select_all_images_action)

        self.jump_first_fixup_action = QAction("Jump to First Fixup", self)
        self.jump_first_fixup_action.setShortcut("Alt+F")
        self.jump_first_fixup_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.jump_first_fixup_action.triggered.connect(self._jump_to_first_fixup)
        self.list_widget.addAction(self.jump_first_fixup_action)

        self.jump_last_fixup_action = QAction("Jump to Last Fixup", self)
        self.jump_last_fixup_action.setShortcut("Alt+L")
        self.jump_last_fixup_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.jump_last_fixup_action.triggered.connect(self._jump_to_last_fixup)
        self.list_widget.addAction(self.jump_last_fixup_action)

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
        self.image_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #333;")
        center_layout.addWidget(self.image_label)

        self.right_panel = QWidget(self)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        tags_panel = QWidget(self)
        tags_layout = QVBoxLayout(tags_panel)
        tags_layout.setContentsMargins(0, 0, 0, 0)
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
        self.remove_tag_action.setShortcut("Delete")
        self.remove_tag_action.triggered.connect(self._remove_selected_tags)
        self.tag_list.addAction(self.remove_tag_action)

        tags_layout.addWidget(self.tag_input, stretch=0)
        tags_layout.addWidget(self.tag_list, stretch=1)

        controls_panel = QWidget(self)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        self.controls_tabs = QTabWidget(self)
        autotag_tab = QWidget(self)
        autotag_layout = QVBoxLayout(autotag_tab)
        autotag_layout.setContentsMargins(6, 6, 6, 6)
        autotag_layout.setSpacing(6)

        self.ollama_server_input = QLineEdit(self)
        self.ollama_server_input.setPlaceholderText("http://127.0.0.1:11434")
        self.ollama_server_input.setText(self.ollama_server_url)

        self.ollama_fetch_button = QPushButton("Fetch models", self)
        self.ollama_fetch_button.clicked.connect(self.fetch_ollama_models)

        self.ollama_model_combo = QComboBox(self)
        self.ollama_model_combo.setEditable(False)

        self.ollama_use_button = QPushButton("Use", self)
        self.ollama_use_button.clicked.connect(self.use_selected_ollama_model)

        self.ollama_timeout_input = QLineEdit(self)
        self.ollama_timeout_input.setValidator(QIntValidator(1, 86400, self))
        self.ollama_timeout_input.setText(str(int(DEFAULT_TIMEOUT)))
        self.ollama_timeout_input.setMaximumWidth(90)

        self.ollama_retry_input = QLineEdit(self)
        self.ollama_retry_input.setValidator(QIntValidator(0, 10, self))
        self.ollama_retry_input.setText("3")
        self.ollama_retry_input.setMaximumWidth(60)

        server_row = QHBoxLayout()
        server_row.addWidget(self.ollama_server_input, stretch=1)
        server_row.addWidget(self.ollama_fetch_button)

        model_row = QHBoxLayout()
        model_row.addWidget(self.ollama_model_combo, stretch=1)
        model_row.addWidget(self.ollama_use_button)

        generate_options_row = QHBoxLayout()
        self.generate_tags_checkbox = QCheckBox("Tags", self)
        self.generate_tags_checkbox.setChecked(True)
        self.generate_tags_checkbox.checkStateChanged.connect(lambda _state: self._update_ollama_controls())
        self.generate_description_checkbox = QCheckBox("Description", self)
        self.generate_description_checkbox.setChecked(True)
        self.generate_description_checkbox.checkStateChanged.connect(lambda _state: self._update_ollama_controls())
        generate_options_row.addWidget(self.generate_tags_checkbox)
        generate_options_row.addWidget(self.generate_description_checkbox)
        generate_options_row.addSpacing(12)
        generate_options_row.addWidget(QLabel("Timeout", self))
        generate_options_row.addWidget(self.ollama_timeout_input)
        generate_options_row.addSpacing(8)
        generate_options_row.addWidget(QLabel("Retries", self))
        generate_options_row.addWidget(self.ollama_retry_input)
        generate_options_row.addStretch(1)

        gen_row = QHBoxLayout()
        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.generate_with_ollama)
        gen_row.addWidget(self.generate_button)

        buttons_row = QHBoxLayout()
        self.validate_button = QPushButton("Validate", self)
        self.validate_button.clicked.connect(self.validate_tags_with_ollama)
        self.fixup_button = QPushButton("Fixup", self)
        self.fixup_button.clicked.connect(self.open_fixup_dialog)
        buttons_row.addWidget(self.validate_button)
        buttons_row.addWidget(self.fixup_button)

        autotag_layout.addLayout(server_row)
        autotag_layout.addLayout(model_row)
        autotag_layout.addLayout(generate_options_row)
        autotag_layout.addLayout(gen_row)
        autotag_layout.addLayout(buttons_row)
        autotag_layout.addStretch(1)

        self.controls_tabs.addTab(autotag_tab, "AutoTag")
        self._add_prompt_tab("description", "Description")
        self._add_prompt_tab("tagging", "Tagging")
        self._add_prompt_tab("validation", "Validation")
        controls_layout.addWidget(self.controls_tabs)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.right_splitter.addWidget(tags_panel)
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

    def _add_prompt_tab(self, kind: str, title: str) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

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
        buttons_row.addWidget(apply_button)
        buttons_row.addWidget(save_button)
        buttons_row.addWidget(reset_button)
        buttons_row.addStretch(1)

        layout.addWidget(editor, stretch=1)
        layout.addWidget(status_label, stretch=0)
        layout.addLayout(buttons_row)

        self.controls_tabs.addTab(tab, title)
        self._update_prompt_status(kind)

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
            except OllamaError:
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
        if kind == "validation":
            return "Validation"
        return kind

    def _prompt_editor_text(self, kind: str) -> str:
        editor = self.prompt_editors.get(kind)
        if editor is None:
            raise OllamaError(f"Prompt editor for {kind} is not available.")
        return editor.toPlainText().strip()

    def _apply_prompt_override(self, kind: str) -> None:
        try:
            set_prompt_override(kind, self._prompt_editor_text(kind))
        except OllamaError as exc:
            QMessageBox.critical(self, "Apply prompt failed", str(exc))
            return
        self._update_prompt_status(kind)
        self.statusBar().showMessage(f"Applied {self._prompt_title(kind)} prompt in memory")

    def _save_prompt_to_file(self, kind: str) -> None:
        try:
            saved_text = save_prompt_for_kind(kind, self._prompt_editor_text(kind))
        except OllamaError as exc:
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
        except OllamaError as exc:
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

        server = self._cfg.get("ollama_server", "").strip()
        model = self._cfg.get("ollama_model", "").strip()

        max_resolution_mpx = self._cfg.get("ollama_max_resolution_mpx", 5)
        try:
            max_pixels = int(float(max_resolution_mpx) * 1_000_000)
        except (TypeError, ValueError):
            max_pixels = 5_000_000
        configure_runtime(max_image_pixels=max_pixels)

        if server:
            self.ollama_server_url = server
            self.ollama_server_input.setText(server)
        if model:
            self.ollama_model_name = model
            self.ollama_model_combo.addItem(model)
            self.ollama_model_combo.setCurrentIndex(0)

        # Load last used directory if it exists
        last_dir = self._cfg.get("last_open_directory", "").strip()
        if last_dir:
            folder = Path(last_dir)
            if folder.exists() and folder.is_dir():
                last_image_str = self._cfg.get("last_selected_image", "").strip()
                restore_path: Path | None = None
                if last_image_str:
                    candidate = Path(last_image_str)
                    if candidate.exists() and candidate.is_file() and candidate.parent == folder:
                        restore_path = candidate
                self.load_directory(folder, restore_selection=restore_path)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("File")

        self.open_action = QAction("Open Folder", self)
        self.open_action.setShortcut("Ctrl+L")
        self.open_action.triggered.connect(self.open_folder)

        self.refresh_action = QAction("Refresh Folder", self)
        self.refresh_action.setShortcut("Ctrl+R")
        self.refresh_action.triggered.connect(self.refresh_directory)

        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut("Alt+F4")
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

    def fetch_ollama_models(self) -> None:
        server = self.ollama_server_input.text().strip()
        self.ollama_fetch_button.setEnabled(False)
        try:
            model_names = fetch_models(server, timeout=self._ollama_timeout_seconds())
            normalized_server = normalize_server_url(server)
        except OllamaError as exc:
            QMessageBox.warning(self, "Ollama connection failed", str(exc))
            return
        finally:
            self.ollama_fetch_button.setEnabled(True)

        self.ollama_server_input.setText(normalized_server)
        self.ollama_model_combo.clear()
        self.ollama_model_combo.addItems(model_names)
        if model_names:
            self.ollama_model_combo.setCurrentIndex(0)
            self.statusBar().showMessage(f"Fetched {len(model_names)} model(s) from {normalized_server}")
        else:
            self.statusBar().showMessage(f"No models found at {normalized_server}")
            QMessageBox.information(self, "No models found", "The Ollama server returned no models.")

    def use_selected_ollama_model(self) -> None:
        model_name = self.ollama_model_combo.currentText().strip()
        if not model_name:
            QMessageBox.warning(self, "No model selected", "Fetch models and choose one before using it.")
            return

        try:
            normalized_server = normalize_server_url(self.ollama_server_input.text())
        except OllamaError as exc:
            QMessageBox.warning(self, "Invalid server", str(exc))
            return

        self.ollama_server_url = normalized_server
        self.ollama_model_name = model_name
        self.ollama_server_input.setText(normalized_server)
        self._cfg["ollama_server"] = normalized_server
        self._cfg["ollama_model"] = model_name
        _config.save(self._cfg)
        self._update_ollama_controls()
        self.statusBar().showMessage(f"Ollama model selected: {self.ollama_model_name}")

    def _ollama_timeout_seconds(self) -> float:
        raw_value = self.ollama_timeout_input.text().strip()
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

    def _ollama_retry_count(self) -> int:
        raw_value = self.ollama_retry_input.text().strip()
        try:
            retries = int(raw_value)
        except ValueError:
            return 0
        return max(0, retries)

    def _update_ollama_controls(self) -> None:
        connected = bool(self.ollama_model_name.strip())
        if connected:
            text = f"{self.ollama_model_name} @ {self.ollama_server_url}"
        else:
            text = "no model"
        if self.status_connection_label is not None:
            self.status_connection_label.setText(text)
        if self._ollama_thread is not None and self._ollama_action_name == "Generate":
            self.generate_button.setText("Stop generation")
            self.generate_button.setEnabled(True)
            self.validate_button.setText("Validate")
            self.validate_button.setEnabled(False)
            self._update_fixup_button_state()
            return
        if self._ollama_thread is not None and self._ollama_action_name == "Validate":
            self.generate_button.setText("Generate")
            self.generate_button.setEnabled(False)
            self.validate_button.setText("Stop validation")
            self.validate_button.setEnabled(True)
            self._update_fixup_button_state()
            return

        self.generate_button.setText("Generate")
        self.validate_button.setText("Validate")
        active = connected and self._ollama_thread is None
        self.generate_button.setEnabled(
            active
            and (self.generate_tags_checkbox.isChecked() or self.generate_description_checkbox.isChecked())
        )
        self.validate_button.setEnabled(active)
        self._update_fixup_button_state()

    def _current_record(self) -> ImageRecord | None:
        if 0 <= self.current_index < len(self.records):
            return self.records[self.current_index]
        return None

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

    def _is_fixup_filter_enabled(self) -> bool:
        return self.filter_input.text().strip().casefold() == "fixup"

    def _record_matches_filter(self, record: ImageRecord) -> bool:
        if not self._is_fixup_filter_enabled():
            return True
        return existing_fixup_path_for_image(record.image_path) is not None

    def _first_visible_row(self) -> int:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None and not item.isHidden():
                return row
        return -1

    def _apply_image_filter(self, _text: str | None = None) -> None:
        selected_path: Path | None = None
        if 0 <= self.current_index < len(self.records):
            selected_path = self.records[self.current_index].image_path

        for row, record in enumerate(self.records):
            item = self.list_widget.item(row)
            if item is None:
                continue
            item.setHidden(not self._record_matches_filter(record))

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

    def _on_fixup_state_changed(self, image_path: Path | None = None) -> None:
        if image_path is None:
            self._refresh_all_list_item_previews()
        else:
            for index, record in enumerate(self.records):
                if record.image_path == image_path:
                    self._update_list_item_preview(index)
                    break
        if self._is_fixup_filter_enabled():
            self._apply_image_filter()
        self._update_fixup_button_state()

    def _update_fixup_button_state(self) -> None:
        record = self._current_record()
        enabled = record is not None and existing_fixup_path_for_image(record.image_path) is not None
        if enabled:
            item = self.list_widget.item(self.current_index)
            if item is None or item.isHidden():
                enabled = False
        if self._ollama_thread is not None:
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

            record = self.records[index]
            if existing_fixup_path_for_image(record.image_path) is not None:
                return index

            index += direction

        return None

    def _find_fixup_index(self, reverse: bool = False) -> int | None:
        if not self.records:
            return None

        indices = range(len(self.records) - 1, -1, -1) if reverse else range(len(self.records))
        for index in indices:
            item = self.list_widget.item(index)
            if item is not None and item.isHidden():
                continue
            record = self.records[index]
            if existing_fixup_path_for_image(record.image_path) is not None:
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

        if 0 <= self.current_index < len(self.records):
            index = self.current_index + direction
        else:
            index = 0 if direction > 0 else len(self.records) - 1

        while 0 <= index < len(self.records):
            item = self.list_widget.item(index)
            if item is not None and not item.isHidden():
                self.list_widget.setCurrentRow(index)
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

    def open_fixup_dialog(self) -> None:
        record = self._current_record()
        if record is None:
            return

        initial_fixup_record_indices = [
            i for i, item in enumerate(self.records)
            if existing_fixup_path_for_image(item.image_path) is not None
        ]
        initial_fixup_total = len(initial_fixup_record_indices)
        if initial_fixup_total <= 0:
            return

        regenerate_tags_enabled = self.generate_tags_checkbox.isChecked()
        regenerate_description_enabled = self.generate_description_checkbox.isChecked()
        timeout_text = self.ollama_timeout_input.text().strip()
        retry_text = self.ollama_retry_input.text().strip()
        try:
            regenerate_timeout_seconds = int(timeout_text) if timeout_text else int(DEFAULT_TIMEOUT)
        except ValueError:
            regenerate_timeout_seconds = int(DEFAULT_TIMEOUT)
        try:
            regenerate_retry_count = int(retry_text) if retry_text else 0
        except ValueError:
            regenerate_retry_count = 0

        def _capture_regenerate_settings(values: dict[str, int | bool]) -> None:
            nonlocal regenerate_tags_enabled
            nonlocal regenerate_description_enabled
            nonlocal regenerate_timeout_seconds
            nonlocal regenerate_retry_count

            tags_enabled = values.get("tags_enabled")
            description_enabled = values.get("description_enabled")
            timeout_seconds = values.get("timeout_seconds")
            retry_count = values.get("retry_count")

            if isinstance(tags_enabled, bool):
                regenerate_tags_enabled = tags_enabled
            if isinstance(description_enabled, bool):
                regenerate_description_enabled = description_enabled
            if isinstance(timeout_seconds, int):
                regenerate_timeout_seconds = max(1, timeout_seconds)
            if isinstance(retry_count, int):
                regenerate_retry_count = max(0, retry_count)

        while True:
            record = self._current_record()
            if record is None:
                return

            try:
                display_index = initial_fixup_record_indices.index(self.current_index) + 1
            except ValueError:
                display_index = 1
            dialog_title = f"Fixup - {record.image_path.name} ({display_index} of {initial_fixup_total})"

            prev_fixup_index = self._find_adjacent_fixup_index(self.current_index, -1)
            next_fixup_index = self._find_adjacent_fixup_index(self.current_index, 1)

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
                ollama_server_url=self.ollama_server_url,
                ollama_model_name=self.ollama_model_name,
                regenerate_tags_enabled=regenerate_tags_enabled,
                regenerate_description_enabled=regenerate_description_enabled,
                regenerate_timeout_seconds=regenerate_timeout_seconds,
                regenerate_retry_count=regenerate_retry_count,
                save_regenerate_settings=_capture_regenerate_settings,
            )

            if outcome == "prev" and prev_fixup_index is not None:
                self.list_widget.setCurrentRow(prev_fixup_index)
                continue
            if outcome == "next" and next_fixup_index is not None:
                self.list_widget.setCurrentRow(next_fixup_index)
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

        self._cfg["last_open_directory"] = folder
        _config.save(self._cfg)
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
        collision = self._find_image_name_collision(folder)
        if collision is not None:
            first, second = collision
            QMessageBox.warning(
                self,
                "Duplicate image names detected",
                "Two image files have the same name with different extensions, which would "
                "collide on the same .txt description file.\n\n"
                f"Filename: {first.stem}\n"
                f"Files:\n- {first}\n- {second}",
            )
            return

        self._root_directory = folder
        self._pending_selection_path = restore_selection
        self._icc_warning_paths = []
        self.records = []
        self.known_tags.clear()
        self.tag_counts.clear()
        self.list_widget.clear()
        self.image_label.setText("No image selected")
        self.tag_input.clear()
        self.tag_list.clear()
        self.current_index = -1
        self._refresh_tag_completions()

        self._set_loading_state(True)

        self._loader_thread = QThread(self)
        self._loader_worker = FolderLoadWorker(folder)
        self._loader_worker.moveToThread(self._loader_thread)

        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.item_loaded.connect(self._on_item_loaded)
        self._loader_worker.progress.connect(self._on_load_progress)
        self._loader_worker.icc_warning.connect(self._on_icc_warning_detected)
        self._loader_worker.finished.connect(self._on_load_finished)
        self._loader_worker.failed.connect(self._on_load_failed)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.failed.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._cleanup_loader)

        self._loader_thread.start()

    def _add_list_item(self, record: ImageRecord, thumbnail: QImage | None = None) -> None:
        title = self._build_list_item_title(record)

        item = QListWidgetItem(title)
        item.setToolTip(str(record.image_path))
        item.setIcon(self._build_list_item_icon(record, thumbnail))
        item.setHidden(not self._record_matches_filter(record))
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
        self.ollama_server_input.setEnabled(not loading and self._ollama_thread is None)
        self.ollama_fetch_button.setEnabled(not loading and self._ollama_thread is None)
        self.ollama_model_combo.setEnabled(not loading and self._ollama_thread is None)
        self.ollama_timeout_input.setEnabled(not loading and self._ollama_thread is None)
        self.ollama_retry_input.setEnabled(not loading and self._ollama_thread is None)
        self.ollama_use_button.setEnabled(not loading and self._ollama_thread is None)
        self.fixup_button.setEnabled(False)
        if loading:
            self.generate_button.setEnabled(False)
            self.validate_button.setEnabled(False)
        else:
            self._update_ollama_controls()

    def _apply_tag_list_height(self) -> None:
        if self._right_splitter_initialized:
            return

        total_height = max(420, self.right_panel.height() if self.right_panel.height() > 0 else self.height())
        top_height = max(180, int(total_height * 0.6))
        bottom_height = max(180, total_height - top_height)
        self.right_splitter.setSizes([top_height, bottom_height])
        self._right_splitter_initialized = True

    def _on_item_loaded(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        image_path_str = str(data.get("image_path", "")).strip()
        text_path_str = str(data.get("text_path", "")).strip()
        text = str(data.get("text", ""))
        thumbnail = data.get("thumbnail")
        thumb_image = thumbnail if isinstance(thumbnail, QImage) else None

        if not image_path_str or not text_path_str:
            return

        image_path = Path(image_path_str)
        text_path = Path(text_path_str)

        record = ImageRecord(image_path=image_path, text_path=text_path, text=text)
        self.records.append(record)
        parsed_tags = self._parse_tags(text)
        self.known_tags.update(parsed_tags)
        self.tag_counts.update(parsed_tags)
        self._add_list_item(record, thumb_image)

    def _on_load_progress(self, processed: int, total: int, percent: int) -> None:
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
        self._set_loading_state(False)
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
        self._set_loading_state(False)
        self._pending_selection_path = None
        self._icc_warning_paths = []
        QMessageBox.critical(self, "Folder load failed", message)
        self.statusBar().showMessage("Folder load failed")

    def _cleanup_loader(self) -> None:
        if self._loader_worker is not None:
            self._loader_worker.deleteLater()
        if self._loader_thread is not None:
            self._loader_thread.deleteLater()
        self._loader_worker = None
        self._loader_thread = None

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

    def _build_thumbnail_icon(self, image_path: Path) -> QIcon:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return QIcon()

        thumb = pixmap.scaled(
            THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(thumb)

    def _draw_fixup_badge(self, thumbnail: QPixmap) -> QPixmap:
        if thumbnail.isNull():
            return thumbnail

        badge_ready = QPixmap(thumbnail)
        painter = QPainter(badge_ready)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        diameter = max(14, min(26, min(badge_ready.width(), badge_ready.height()) // 2))
        x = 2
        y = 2
        badge_rect = QRect(x, y, diameter, diameter)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(220, 45, 45))
        painter.drawEllipse(badge_rect)

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(badge_rect, int(Qt.AlignmentFlag.AlignCenter), "!")
        painter.end()
        return badge_ready

    def _build_list_item_icon(self, record: ImageRecord, thumbnail: QImage | None = None) -> QIcon:
        if thumbnail is not None and not thumbnail.isNull():
            base = QPixmap.fromImage(thumbnail)
        else:
            base_icon = self._build_thumbnail_icon(record.image_path)
            base = base_icon.pixmap(THUMB_SIZE)

        if existing_fixup_path_for_image(record.image_path) is not None:
            base = self._draw_fixup_badge(base)
        return QIcon(base)

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
            self._update_fixup_button_state()
            return

        self.current_index = index
        record = self.records[index]

        self._show_image(record.image_path)

        tags = self._parse_tags(record.text)
        self._populate_tag_list(tags)
        self.tag_input.clear()
        self._update_fixup_button_state()
        self.statusBar().showMessage(f"({index + 1} of {len(self.records)}) {record.image_path}")

    def _show_image(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText(f"Unable to load image:\n{image_path.name}")
            self.image_label.setPixmap(QPixmap())
            return

        target_size = self.image_label.size()
        if target_size.width() < 50 or target_size.height() < 50:
            return

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_tag_list_height()
        self._update_tag_item_heights()
        if 0 <= self.current_index < len(self.records):
            self._show_image(self.records[self.current_index].image_path)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_main_window_geometry()
        record = self._current_record()
        if record is not None:
            self._cfg["last_selected_image"] = str(record.image_path)
        else:
            self._cfg.pop("last_selected_image", None)
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
        # Ignore format-only churn so loading/selection does not trigger unsolicited saves.
        if self._parse_tags(record.text) == self._parse_tags(new_text):
            return

        record.text = new_text
        self._update_list_item_preview(self.current_index)
        if rebuild_completions:
            self._rebuild_known_tags_from_records()
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

    def _add_tag_from_input(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.records):
            return

        new_tag = self.tag_input.text().strip()
        if not new_tag:
            return

        existing_tags = self._current_tags()
        if new_tag in existing_tags:
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

        for item in selected:
            row = self.tag_list.row(item)
            removed = self.tag_list.takeItem(row)
            del removed

        self._update_tag_item_heights()
        self._sync_record_from_tag_list()

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
        item.setText(self._build_list_item_title(record))
        item.setIcon(self._build_list_item_icon(record))

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
            record.text_path.write_text(record.text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save text file:\n{exc}")
            return False

        self.statusBar().showMessage(f"{status_prefix}: {record.text_path.name}")
        return True

    def generate_with_ollama(self) -> None:
        if self._ollama_thread is not None and self._ollama_action_name == "Generate":
            self._request_stop_generation()
            return

        selected_indexes = self._selected_record_indexes()
        if not selected_indexes:
            QMessageBox.information(self, "No image selected", "Select an image before generating annotations.")
            return
        include_tags = self.generate_tags_checkbox.isChecked()
        include_description = self.generate_description_checkbox.isChecked()
        if not include_tags and not include_description:
            QMessageBox.information(self, "Nothing selected", "Enable Tags or Description before generating.")
            return

        cancel_token = OllamaCancellation()

        def generate_task(
            report_progress: Callable[[str], None],
            report_item: Callable[[object], None],
        ) -> object:
            timeout = self._ollama_timeout_seconds()
            retry_count = self._ollama_retry_count()
            total = len(selected_indexes)

            for position, record_index in enumerate(selected_indexes, start=1):
                record = self.records[record_index]
                image_name = record.image_path.name
                generated_items: list[str] = []
                image_retried = False

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
                        report_progress(
                            f"Generate: processing {position}/{total} - {record.image_path.name}"
                        )
                    else:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_retry image={image_name} attempt={attempt + 1}/{retry_count + 1}",
                            flush=True,
                        )
                        report_progress(
                            f"Generate: retry {attempt}/{retry_count} - {position}/{total} - {record.image_path.name}"
                        )

                    def remaining_timeout(_start=attempt_start) -> float:
                        elapsed = time.monotonic() - _start
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise OllamaError(
                                f"Timed out after {int(timeout)} seconds while generating annotations for {record.image_path.name}."
                            )
                        return remaining

                    attempt_items: list[str] = []
                    retry_needed = False
                    try:
                        if include_description:
                            description = self._sanitize_annotation_text(
                                generate_description(
                                    self.ollama_server_url,
                                    self.ollama_model_name,
                                    record.image_path,
                                    timeout=remaining_timeout(),
                                    cancellation=cancel_token,
                                ).strip()
                            )
                            if description:
                                attempt_items.append(description)

                        if include_tags:
                            attempt_items.extend(
                                self._parse_tags(
                                    generate_tags(
                                        self.ollama_server_url,
                                        self.ollama_model_name,
                                        record.image_path,
                                        timeout=remaining_timeout(),
                                        cancellation=cancel_token,
                                    )
                                )
                            )

                        if not attempt_items:
                            retry_needed = True
                    except OllamaCancelled:
                        raise
                    except OllamaError:
                        retry_needed = True

                    elapsed_seconds = time.monotonic() - attempt_start
                    if not retry_needed:
                        generated_items = attempt_items
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_done image={image_name} elapsed_s={elapsed_seconds:.2f}",
                            flush=True,
                        )
                        break
                    else:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_failed image={image_name} attempt={attempt + 1}/{retry_count + 1} elapsed_s={elapsed_seconds:.2f}",
                            flush=True,
                        )

                report_item(
                    {
                        "kind": "generate_item",
                        "index": record_index,
                        "items": generated_items,
                        "retried": image_retried,
                        "position": position,
                        "total": total,
                    }
                )

            report_progress(f"Generate: finalizing {total}/{total}")
            return {"batch": True, "streamed": True, "total": total}

        self._generate_batch_total = len(selected_indexes)
        self._generate_batch_processed = 0
        self._generate_batch_updated = 0
        self._generate_batch_new_annotations = 0
        self._generate_batch_started_at = time.monotonic()
        self._generate_batch_retry_images = 0

        self._start_ollama_task(
            task=generate_task,
            action_name="Generate",
            empty_message="Ollama returned no annotations.",
            cancel_token=cancel_token,
            merge_with_existing=True,
        )

    def validate_tags_with_ollama(self) -> None:
        if self._ollama_thread is not None and self._ollama_action_name == "Validate":
            self._request_stop_validation()
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
            annotations = self._serialize_tags(self._parse_tags(self.records[record_index].text))
            if not annotations:
                skipped_without_annotations += 1
                continue
            annotated_records.append((record_index, annotations))

        if not annotated_records:
            QMessageBox.information(self, "No annotations to validate", "Add tags or a description before validating.")
            return

        cancel_token = OllamaCancellation()

        def validate_task(
            report_progress: Callable[[str], None],
            report_item: Callable[[object], None],
        ) -> object:
            timeout = self._ollama_timeout_seconds()
            retry_count = self._ollama_retry_count()
            total = len(annotated_records)

            for position, (record_index, annotations) in enumerate(annotated_records, start=1):
                record = self.records[record_index]
                image_name = record.image_path.name
                image_retried = False
                validation_result = ""

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
                        report_progress(
                            f"Validate: processing {position}/{total} - {record.image_path.name}"
                        )
                    else:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] validation_retry image={image_name} attempt={attempt + 1}/{retry_count + 1}",
                            flush=True,
                        )
                        report_progress(
                            f"Validate: retry {attempt}/{retry_count} - {position}/{total} - {record.image_path.name}"
                        )

                    def remaining_timeout(_start=attempt_start) -> float:
                        elapsed = time.monotonic() - _start
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise OllamaError(
                                f"Timed out after {int(timeout)} seconds while validating annotations for {record.image_path.name}."
                            )
                        return remaining

                    try:
                        validation_result = validate_tags(
                            self.ollama_server_url,
                            self.ollama_model_name,
                            record.image_path,
                            annotations,
                            timeout=remaining_timeout(),
                            cancellation=cancel_token,
                        )
                    except OllamaCancelled:
                        raise
                    except OllamaError:
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

                report_item(
                    {
                        "kind": "validate_item",
                        "index": record_index,
                        "result": validation_result,
                        "retried": image_retried,
                        "position": position,
                        "total": total,
                    }
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

        self._start_ollama_task(
            task=validate_task,
            action_name="Validate",
            empty_message="Ollama returned no validation result.",
            cancel_token=cancel_token,
            validation_report=True,
        )

    def _start_ollama_task(
        self,
        task: Callable[[Callable[[str], None], Callable[[object], None]], object],
        action_name: str,
        empty_message: str,
        cancel_token: OllamaCancellation | None = None,
        result_as_single: bool = False,
        merge_with_existing: bool = False,
        validation_report: bool = False,
    ) -> None:
        if not self.ollama_model_name.strip():
            QMessageBox.warning(self, "No Ollama model", "Connect to an Ollama server and choose a model first.")
            return
        if self._ollama_thread is not None:
            return

        resize_warning = consume_resize_warning()
        if resize_warning:
            QMessageBox.warning(self, "Image resize disabled", resize_warning)

        self.statusBar().showMessage(f"{action_name} with Ollama...")
        self.validate_button.setEnabled(False)
        self.ollama_server_input.setEnabled(False)
        self.ollama_fetch_button.setEnabled(False)
        self.ollama_model_combo.setEnabled(False)
        self.ollama_timeout_input.setEnabled(False)
        self.ollama_retry_input.setEnabled(False)
        self.ollama_use_button.setEnabled(False)
        self._ollama_action_name = action_name
        self._ollama_cancel = cancel_token
        self._update_ollama_controls()

        self._ollama_thread = QThread(self)
        self._ollama_worker = OllamaTaskWorker(task)
        self._ollama_worker.moveToThread(self._ollama_thread)
        self._ollama_thread.started.connect(self._ollama_worker.run)
        self._ollama_worker.finished.connect(
            lambda result: self._on_ollama_task_finished(
                result,
                action_name,
                empty_message,
                result_as_single,
                merge_with_existing,
                validation_report,
            )
        )
        self._ollama_worker.progress.connect(self._on_ollama_task_progress)
        self._ollama_worker.item_ready.connect(self._on_ollama_task_item_ready)
        self._ollama_worker.cancelled.connect(self._on_ollama_task_cancelled)
        self._ollama_worker.failed.connect(self._on_ollama_task_failed)
        self._ollama_worker.finished.connect(self._ollama_thread.quit)
        self._ollama_worker.cancelled.connect(self._ollama_thread.quit)
        self._ollama_worker.failed.connect(self._ollama_thread.quit)
        self._ollama_thread.finished.connect(self._cleanup_ollama_task)
        self._update_ollama_controls()
        self._ollama_thread.start()

    def _request_stop_generation(self) -> None:
        if self._ollama_action_name != "Generate" or self._ollama_cancel is None:
            return
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Stopping generation...")
        self.statusBar().showMessage("Stopping generation...")
        self._ollama_cancel.cancel()

    def _request_stop_validation(self) -> None:
        if self._ollama_action_name != "Validate" or self._ollama_cancel is None:
            return
        self.validate_button.setEnabled(False)
        self.validate_button.setText("Stopping validation...")
        self.statusBar().showMessage("Stopping validation...")
        self._ollama_cancel.cancel()

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

    def _on_ollama_task_progress(self, message: str) -> None:
        if self._ollama_action_name == "Generate" and (
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
            self.statusBar().showMessage(f"{message}{details}")
            return

        if self._ollama_action_name == "Validate" and (
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
            self.statusBar().showMessage(f"{message}{details}")
            return

        self.statusBar().showMessage(message)

    def _on_ollama_task_cancelled(self, message: str) -> None:
        action = self._ollama_action_name or "Request"
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

        self.statusBar().showMessage(message or f"{action} stopped.")

    def _apply_generated_items_to_record(self, record_index: int, items: list[str]) -> tuple[bool, int]:
        if record_index < 0 or record_index >= len(self.records):
            return (False, 0)

        record = self.records[record_index]
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
            return (False, 0)

        record.text = self._serialize_tags(merged_tags)
        if self._write_record_text(record, status_prefix="Generate + auto-saved"):
            self._update_list_item_preview(record_index)

            if self.current_index == record_index and not self.tag_input.hasFocus():
                if self.tag_list.state() != QAbstractItemView.State.EditingState:
                    self._populate_tag_list(self._parse_tags(record.text))

            return (True, added)

        return (False, 0)

    def _on_ollama_task_item_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        kind = payload.get("kind")
        if kind == "generate_item":
            raw_index = payload.get("index")
            raw_items = payload.get("items")
            raw_retried = payload.get("retried")
            raw_position = payload.get("position")
            raw_total = payload.get("total")

            if not isinstance(raw_index, int):
                return
            if raw_index < 0 or raw_index >= len(self.records):
                return
            items = [str(item).strip() for item in raw_items] if isinstance(raw_items, list) else []
            items = [item for item in items if item]

            updated, added = self._apply_generated_items_to_record(raw_index, items)
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
                    f"Generate: applied {raw_position}/{raw_total} - {self.records[raw_index].image_path.name}{details}"
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

        outcome = self._apply_validation_result_to_record(raw_index, str(raw_result or ""))
        self._validate_batch_processed += 1
        if isinstance(raw_retried, bool) and raw_retried:
            self._validate_batch_retry_images += 1
        if outcome == "clean":
            self._validate_batch_clean += 1
        elif outcome == "issues":
            self._validate_batch_issues += 1

        if isinstance(raw_position, int) and isinstance(raw_total, int) and raw_total > 0:
            details = self._batch_progress_details(
                self._validate_batch_processed,
                self._validate_batch_total,
                self._validate_batch_started_at,
                self._validate_batch_retry_images,
            )
            self.statusBar().showMessage(
                f"Validate: applied {raw_position}/{raw_total} - {self.records[raw_index].image_path.name}{details}"
            )

    def _on_ollama_task_finished(
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
                    f"{action_name} complete via Ollama ({', '.join(parts)})"
                )

            self._validate_batch_total = 0
            self._validate_batch_processed = 0
            self._validate_batch_clean = 0
            self._validate_batch_issues = 0
            self._validate_batch_skipped = 0
            return

        if isinstance(result, dict) and result.get("batch") is True and result.get("streamed") is True:
            if self._generate_batch_updated == 0:
                QMessageBox.information(self, f"{action_name} finished", empty_message)
                self.statusBar().showMessage(f"{action_name} finished")
            else:
                self.statusBar().showMessage(
                    f"{action_name} complete via Ollama ({self._generate_batch_updated} image{'s' if self._generate_batch_updated != 1 else ''}, {self._generate_batch_new_annotations} new annotation{'s' if self._generate_batch_new_annotations != 1 else ''})"
                )
            self._generate_batch_total = 0
            self._generate_batch_processed = 0
            self._generate_batch_updated = 0
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
                self._populate_tag_list(self._parse_tags(self.records[self.current_index].text))

            self.statusBar().showMessage(
                f"{action_name} complete via Ollama ({updated} image{'s' if updated != 1 else ''}, {with_new} new annotation{'s' if with_new != 1 else ''})"
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
                    clear_fixup_files_for_image(record.image_path)
                    self._on_fixup_state_changed()
                return

            if self.current_index < 0 or self.current_index >= len(self.records):
                QMessageBox.warning(self, "Validate failed", "No selected image to write a fixup file.")
                self.statusBar().showMessage("Validate failed")
                return

            record = self.records[self.current_index]
            try:
                fixup_path = write_fixup_for_image(record.image_path, cleaned)
            except OSError as exc:
                QMessageBox.warning(self, "Fixup write failed", f"Could not write fixup file:\n{exc}")
                self.statusBar().showMessage("Validate failed: could not write .fixup")
                return

            self.statusBar().showMessage(f"Validate found issues: wrote {fixup_path.name}")
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
                f"{action_name} complete via Ollama ({added_count} tag{'s' if added_count != 1 else ''} added)"
            )
            return

        self._set_current_tags(tags, status_prefix=f"{action_name} + auto-saved")
        self.statusBar().showMessage(f"{action_name} complete via Ollama")

    def _on_ollama_task_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Ollama request failed", message)
        self.statusBar().showMessage("Ollama request failed")

    def _apply_validation_result_to_record(self, record_index: int, result: str) -> str:
        if record_index < 0 or record_index >= len(self.records):
            return "invalid"

        cleaned = result.strip()
        if not cleaned:
            return "empty"

        record = self.records[record_index]
        if cleaned.casefold() == "ok":
            clear_fixup_files_for_image(record.image_path)
            self._on_fixup_state_changed(record.image_path)
            return "clean"

        try:
            write_fixup_for_image(record.image_path, cleaned)
        except OSError:
            return "error"

        self._on_fixup_state_changed(record.image_path)
        return "issues"

    def _cleanup_ollama_task(self) -> None:
        if self._ollama_worker is not None:
            self._ollama_worker.deleteLater()
        if self._ollama_thread is not None:
            self._ollama_thread.deleteLater()
        self._ollama_worker = None
        self._ollama_thread = None
        self._generate_batch_total = 0
        self._generate_batch_processed = 0
        self._generate_batch_updated = 0
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
        self._ollama_action_name = None
        self._ollama_cancel = None
        self._update_ollama_controls()
        self.ollama_server_input.setEnabled(True)
        self.ollama_fetch_button.setEnabled(True)
        self.ollama_model_combo.setEnabled(True)
        self.ollama_timeout_input.setEnabled(True)
        self.ollama_retry_input.setEnabled(True)
        self.ollama_use_button.setEnabled(True)
        self._update_fixup_button_state()


def main() -> None:
    QImageReader.setAllocationLimit(1024)

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
