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

from PyQt6.QtCore import QEvent, QObject, QRect, QStringListModel, QThread, Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDoubleValidator, QFont, QIcon, QImage, QImageReader, QIntValidator, QKeySequence, QPainter, QPixmap
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
    QStyledItemDelegate,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imagetagger import config as _config
from imagetagger.annotations import parse_tags_text, sanitize_annotation_text
from imagetagger.io_utils import atomic_write_text
from imagetagger.merge_actions import (
    clear_fixup_files_for_image,
    existing_fixup_path_for_image,
    open_fixup_dialog_for_image,
    record_ai_find_match_for_image,
    write_fixup_for_image,
)
from imagetagger.external_editors import (
    ExternalEditor,
    discover_graphics_editors,
    launch_image_in_editor,
    launch_image_in_system_default,
)
from imagetagger.ollama import (
    active_prompt_for_kind,
    clear_prompt_override,
    configure_runtime,
    consume_resize_warning,
    DEFAULT_TIMEOUT,
    DEFAULT_OLLAMA_SERVER,
    get_default_prompt,
    image_matches_query,
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


class FilterSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class _FilterToken:
    kind: str
    value: str
    position: int


class _FilterNode:
    def evaluate(self, record: ImageRecord, runtime: "_FilterRuntime") -> bool:
        raise NotImplementedError()


@dataclass(frozen=True)
class _NamedFilterNode(_FilterNode):
    name: str

    def evaluate(self, record: ImageRecord, runtime: "_FilterRuntime") -> bool:
        predicate = runtime.named_filters.get(self.name.casefold())
        if predicate is None:
            return False
        return predicate(record)


@dataclass(frozen=True)
class _TagFilterNode(_FilterNode):
    tag: str

    def evaluate(self, record: ImageRecord, runtime: "_FilterRuntime") -> bool:
        return runtime.tag_filter(record, self.tag)


@dataclass(frozen=True)
class _FreetextFilterNode(_FilterNode):
    text: str

    def evaluate(self, record: ImageRecord, runtime: "_FilterRuntime") -> bool:
        return runtime.freetext_filter(record, self.text)


@dataclass(frozen=True)
class _AndFilterNode(_FilterNode):
    left: _FilterNode
    right: _FilterNode

    def evaluate(self, record: ImageRecord, runtime: "_FilterRuntime") -> bool:
        return self.left.evaluate(record, runtime) and self.right.evaluate(record, runtime)


@dataclass(frozen=True)
class _OrFilterNode(_FilterNode):
    left: _FilterNode
    right: _FilterNode

    def evaluate(self, record: ImageRecord, runtime: "_FilterRuntime") -> bool:
        return self.left.evaluate(record, runtime) or self.right.evaluate(record, runtime)


@dataclass
class _FilterRuntime:
    named_filters: dict[str, Callable[[ImageRecord], bool]]
    tag_filter: Callable[[ImageRecord, str], bool]
    freetext_filter: Callable[[ImageRecord, str], bool]


def _tokenize_filter_expression(expression: str) -> list[_FilterToken]:
    tokens: list[_FilterToken] = []
    index = 0
    length = len(expression)

    while index < length:
        char = expression[index]

        if char.isspace():
            index += 1
            continue

        if char in "&|()":
            tokens.append(_FilterToken(kind=char, value=char, position=index))
            index += 1
            continue

        if char == '"':
            start = index
            index += 1
            value_chars: list[str] = []
            while index < length:
                current = expression[index]
                if current == "\\":
                    index += 1
                    if index >= length:
                        raise FilterSyntaxError(f"Unfinished escape sequence at position {start + 1}.")
                    value_chars.append(expression[index])
                    index += 1
                    continue
                if current == '"':
                    index += 1
                    break
                value_chars.append(current)
                index += 1
            else:
                raise FilterSyntaxError(f"Missing closing quote for tag at position {start + 1}.")

            tokens.append(_FilterToken(kind="STRING", value="".join(value_chars), position=start))
            continue

        if char == "'":
            start = index
            index += 1
            value_chars: list[str] = []
            while index < length:
                current = expression[index]
                if current == "\\":
                    index += 1
                    if index >= length:
                        raise FilterSyntaxError(f"Unfinished escape sequence at position {start + 1}.")
                    value_chars.append(expression[index])
                    index += 1
                    continue
                if current == "'":
                    index += 1
                    break
                value_chars.append(current)
                index += 1
            else:
                raise FilterSyntaxError(f"Missing closing quote for freetext at position {start + 1}.")

            tokens.append(_FilterToken(kind="FREETEXT", value="".join(value_chars), position=start))
            continue

        start = index
        while index < length and (not expression[index].isspace()) and expression[index] not in "&|()\"":
            index += 1

        value = expression[start:index]
        if not value:
            raise FilterSyntaxError(f"Unexpected character at position {start + 1}.")
        tokens.append(_FilterToken(kind="NAME", value=value, position=start))

    return tokens


def _parse_filter_expression(expression: str) -> _FilterNode | None:
    tokens = _tokenize_filter_expression(expression)
    if not tokens:
        return None

    position = 0

    def _peek() -> _FilterToken | None:
        if position >= len(tokens):
            return None
        return tokens[position]

    def _consume(expected_kind: str | None = None) -> _FilterToken:
        nonlocal position
        token = _peek()
        if token is None:
            raise FilterSyntaxError("Unexpected end of filter expression.")
        if expected_kind is not None and token.kind != expected_kind:
            raise FilterSyntaxError(
                f"Expected '{expected_kind}' at position {token.position + 1}, got '{token.value}'."
            )
        position += 1
        return token

    def _parse_primary() -> _FilterNode:
        token = _peek()
        if token is None:
            raise FilterSyntaxError("Unexpected end of filter expression.")

        if token.kind == "(":
            _consume("(")
            nested = _parse_or_expression()
            closing = _peek()
            if closing is None or closing.kind != ")":
                at = token.position + 1 if closing is None else closing.position + 1
                raise FilterSyntaxError(f"Missing ')' for group near position {at}.")
            _consume(")")
            return nested

        if token.kind == "NAME":
            _consume("NAME")
            return _NamedFilterNode(name=token.value)

        if token.kind == "STRING":
            _consume("STRING")
            return _TagFilterNode(tag=token.value)

        if token.kind == "FREETEXT":
            _consume("FREETEXT")
            return _FreetextFilterNode(text=token.value)

        raise FilterSyntaxError(f"Unexpected token '{token.value}' at position {token.position + 1}.")

    def _parse_and_expression() -> _FilterNode:
        node = _parse_primary()
        while True:
            token = _peek()
            if token is None or token.kind != "&":
                break
            _consume("&")
            node = _AndFilterNode(left=node, right=_parse_primary())
        return node

    def _parse_or_expression() -> _FilterNode:
        node = _parse_and_expression()
        while True:
            token = _peek()
            if token is None or token.kind != "|":
                break
            _consume("|")
            node = _OrFilterNode(left=node, right=_parse_and_expression())
        return node

    parsed = _parse_or_expression()
    trailing = _peek()
    if trailing is not None:
        raise FilterSyntaxError(
            f"Unexpected token '{trailing.value}' at position {trailing.position + 1}."
        )
    return parsed


class FolderLoadWorker(QObject):
    progress = pyqtSignal(int, int, int)
    item_loaded = pyqtSignal(object)
    finished = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    icc_warning = pyqtSignal(str)
    scan_ready = pyqtSignal(int)
    collision_detected = pyqtSignal(str, str)

    def __init__(self, folder: Path, max_thread_cap: int = 8) -> None:
        super().__init__()
        self.folder = folder
        self._max_thread_cap = max(1, int(max_thread_cap))
        self._cancelled = False
        self._allow_processing = threading.Event()

    def cancel(self) -> None:
        self._cancelled = True
        self._allow_processing.set()

    def allow_processing(self) -> None:
        self._allow_processing.set()

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

    def _scan_folder(self) -> tuple[list[Path], tuple[Path, Path] | None]:
        """
        Scan the folder once, returning sorted image paths and the first collision (if any).

        Collision definition: two image files that would map to the same `.txt` file
        (same stem after suffix replacement), but with different image extensions.
        """
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
            raise exc

        seen_by_txt_path: dict[str, Path] = {}
        for image_path in image_paths:
            txt_key = str(image_path.with_suffix(".txt")).casefold()
            existing = seen_by_txt_path.get(txt_key)
            if existing is None:
                seen_by_txt_path[txt_key] = image_path
                continue

            if existing.suffix.lower() != image_path.suffix.lower():
                return image_paths, (existing, image_path)

        return image_paths, None

    @staticmethod
    def _thumbnail_rgba_bytes(image_path: Path) -> tuple[dict | None, bool]:
        """
        Build a small RGBA thumbnail using Pillow.

        Returns: (thumbnail_payload, icc_invalid)
        where thumbnail_payload is dict(width, height, bytes, bytes_per_line) suitable for
        reconstructing a QImage on the GUI thread.
        """
        try:
            img = Image.open(image_path)
        except (OSError, UnidentifiedImageError):
            return None, False

        with img:
            # Validate ICC profile using the raw ICC bytes (if present).
            raw_profile = img.info.get("icc_profile")
            icc_invalid = False
            if raw_profile:
                if isinstance(raw_profile, str):
                    raw_profile = raw_profile.encode("utf-8", errors="ignore")
                if isinstance(raw_profile, (bytes, bytearray)):
                    try:
                        ImageCms.ImageCmsProfile(BytesIO(bytes(raw_profile)))
                    except ImageCms.PyCMSError:
                        icc_invalid = True

            # Pillow doesn't auto-apply EXIF orientation, so we transpose to match Qt behavior.
            try:
                from PIL import ImageOps

                img = ImageOps.exif_transpose(img)
            except Exception:
                # If exif_transpose fails, fall back to the original decoded orientation.
                pass

            # Downscale aggressively to 96x96-ish so we don't decode full-size into memory.
            thumb = img.convert("RGBA")
            thumb.thumbnail((THUMB_SIZE.width(), THUMB_SIZE.height()), resample=Image.Resampling.LANCZOS)

            rgba_bytes = thumb.tobytes()
            width, height = thumb.size
            bytes_per_line = width * 4
            return (
                {
                    "width": width,
                    "height": height,
                    "bytes": rgba_bytes,
                    "bytes_per_line": bytes_per_line,
                },
                icc_invalid,
            )

    def run(self) -> None:
        try:
            image_paths, collision = self._scan_folder()
        except OSError as exc:
            self.failed.emit(f"Failed to read folder: {exc}")
            return
        if self._cancelled:
            return

        if collision is not None:
            first, second = collision
            self.collision_detected.emit(str(first), str(second))
            return

        total = len(image_paths)
        self.scan_ready.emit(total)
        self.progress.emit(0, total, 0)

        # Wait until the GUI thread resets its state and is ready for results.
        while not self._allow_processing.is_set():
            if self._cancelled:
                return
            self._allow_processing.wait(timeout=0.05)

        if total == 0:
            self.finished.emit(0, str(self.folder))
            return

        processed = 0

        # Use a bounded worker pool to utilize multiple cores while avoiding huge memory spikes.
        max_workers = max(1, (os.cpu_count() or 1) - 1)
        # Cap to keep decoding and UI payloads bounded.
        max_workers = min(max_workers, self._max_thread_cap)
        chunk_size = 64

        def process_one(image_path: Path) -> dict:
            text_path = image_path.with_suffix(".txt")
            try:
                text = (
                    text_path.read_text(encoding="utf-8", errors="replace")
                    if text_path.exists()
                    else ""
                )
            except OSError:
                text = ""

            thumb_payload, icc_invalid = self._thumbnail_rgba_bytes(image_path)
            return {
                "image_path": str(image_path),
                "text_path": str(text_path),
                "text": text,
                "thumbnail": thumb_payload,
                "icc_invalid": icc_invalid,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                for chunk_start in range(0, total, chunk_size):
                    if self._cancelled:
                        break

                    chunk = image_paths[chunk_start : chunk_start + chunk_size]
                    for result in executor.map(process_one, chunk):
                        if self._cancelled:
                            break
                        processed += 1

                        if result.get("icc_invalid"):
                            self.icc_warning.emit(result["image_path"])

                        # Keep emitted payload small; thumbnail bytes are only ~96x96 RGBA.
                        self.item_loaded.emit(
                            {
                                "image_path": result["image_path"],
                                "text_path": result["text_path"],
                                "text": result["text"],
                                "thumbnail": result["thumbnail"],
                            }
                        )

                        percent = int((processed / total) * 100) if total else 100
                        self.progress.emit(processed, total, percent)
            except Exception as exc:
                self.failed.emit(f"Failed while processing folder: {exc}")
                return

        self.finished.emit(processed, str(self.folder))


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
        self._ai_find_batch_total = 0
        self._ai_find_batch_processed = 0
        self._ai_find_batch_matched = 0
        self._ai_find_batch_started_at: float | None = None
        self._ai_find_batch_retry_images = 0
        self._ollama_action_name: str | None = None
        self._ollama_cancel: OllamaCancellation | None = None
        self._ollama_threads_auto_mode = False
        self._ollama_threads_current = 0
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
        self._detected_external_editors: list[ExternalEditor] | None = None

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
        self.filter_input.setPlaceholderText("Filter (fixup, \"tag\", 'text', &, |, parentheses)")
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
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self._show_image_context_menu)
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

        self.ollama_max_resolution_input = QLineEdit(self)
        max_resolution_validator = QDoubleValidator(0.01, 1000.0, 3, self)
        max_resolution_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.ollama_max_resolution_input.setValidator(max_resolution_validator)
        self.ollama_max_resolution_input.setText("5.0")
        self.ollama_max_resolution_input.setMaximumWidth(80)

        self.ollama_threads_input = QLineEdit(self)
        self.ollama_threads_input.setValidator(QIntValidator(0, 128, self))
        self.ollama_threads_input.setText("1")
        self.ollama_threads_input.setMaximumWidth(50)
        self.ollama_threads_input.setToolTip("0 = auto")

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
        generate_options_row.addSpacing(8)
        generate_options_row.addWidget(QLabel("Downscale", self))
        generate_options_row.addWidget(self.ollama_max_resolution_input)
        generate_options_row.addSpacing(8)
        generate_options_row.addWidget(QLabel("Threads", self))
        generate_options_row.addWidget(self.ollama_threads_input)
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

        ai_find_row = QHBoxLayout()
        self.ai_find_input = QLineEdit(self)
        self.ai_find_input.setPlaceholderText("Find concept in selected images (e.g. raven)")
        self.ai_find_input.textChanged.connect(lambda _text: self._update_ollama_controls())
        self.ai_find_button = QPushButton("AI Find", self)
        self.ai_find_button.clicked.connect(self.ai_find_with_ollama)
        ai_find_row.addWidget(self.ai_find_input, stretch=1)
        ai_find_row.addWidget(self.ai_find_button)

        autotag_layout.addLayout(server_row)
        autotag_layout.addLayout(model_row)
        autotag_layout.addLayout(generate_options_row)
        autotag_layout.addLayout(gen_row)
        autotag_layout.addLayout(buttons_row)
        autotag_layout.addLayout(ai_find_row)
        autotag_layout.addStretch(1)

        self.controls_tabs.addTab(autotag_tab, "AutoTag")
        self._add_prompt_tab("description", "Description")
        self._add_prompt_tab("tagging", "Tagging")
        self._add_prompt_tab("validation", "Validation")
        self._add_prompt_tab("search", "AI Search")
        self._add_known_tags_tab()
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
        self._update_window_title(self._active_directory())

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

    def _add_known_tags_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.known_tags_list = QListWidget(self)
        self.known_tags_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.known_tags_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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
        if kind == "search":
            return "AI Search"
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
            max_resolution_value = float(max_resolution_mpx)
            if max_resolution_value <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            max_resolution_value = 5.0
        self.ollama_max_resolution_input.setText(self._format_mpx(max_resolution_value))
        max_pixels = max(1, int(max_resolution_value * 1_000_000))
        configure_runtime(max_image_pixels=max_pixels)

        raw_threads = self._cfg.get("ollama_threads", 1)
        try:
            thread_count = int(raw_threads)
            if thread_count < 0:
                raise ValueError()
        except (TypeError, ValueError):
            thread_count = 1
        self.ollama_threads_input.setText(str(thread_count))

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
        try:
            self._cfg["ollama_threads"] = self._ollama_thread_count(show_message=False)
        except OllamaError:
            self._cfg["ollama_threads"] = 1
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

    @staticmethod
    def _format_mpx(value: float) -> str:
        normalized = f"{float(value):.3f}".rstrip("0").rstrip(".")
        if "." not in normalized:
            normalized += ".0"
        return normalized

    def _ollama_max_resolution_mpx_value(self, show_message: bool = True) -> float:
        raw_value = self.ollama_max_resolution_input.text().strip()
        if not raw_value:
            if show_message:
                QMessageBox.warning(self, "Invalid query downscale", "Enter query downscale in megapixels.")
            raise OllamaError("Enter query downscale in megapixels.")

        try:
            value = float(raw_value)
        except ValueError as exc:
            if show_message:
                QMessageBox.warning(self, "Invalid query downscale", "Query downscale must be a number.")
            raise OllamaError("Query downscale must be a number.") from exc

        if value <= 0:
            if show_message:
                QMessageBox.warning(self, "Invalid query downscale", "Query downscale must be greater than 0.")
            raise OllamaError("Query downscale must be greater than 0.")

        return value

    def _apply_query_downscale_setting(self) -> float:
        max_resolution_mpx = self._ollama_max_resolution_mpx_value()
        max_pixels = max(1, int(max_resolution_mpx * 1_000_000))
        configure_runtime(max_image_pixels=max_pixels)
        return max_resolution_mpx

    def _ollama_thread_count(self, show_message: bool = True) -> int:
        raw_value = self.ollama_threads_input.text().strip()
        if not raw_value:
            if show_message:
                QMessageBox.warning(self, "Invalid threads", "Enter thread count (0 for auto).")
            raise OllamaError("Enter thread count.")

        try:
            thread_count = int(raw_value)
        except ValueError as exc:
            if show_message:
                QMessageBox.warning(self, "Invalid threads", "Thread count must be a whole number.")
            raise OllamaError("Thread count must be a whole number.") from exc

        if thread_count < 0:
            if show_message:
                QMessageBox.warning(self, "Invalid threads", "Thread count must be 0 or greater.")
            raise OllamaError("Thread count must be 0 or greater.")

        return thread_count

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
            self.ai_find_button.setText("AI Find")
            self.ai_find_button.setEnabled(False)
            self.ai_find_input.setEnabled(False)
            self._update_fixup_button_state()
            return
        if self._ollama_thread is not None and self._ollama_action_name == "Validate":
            self.generate_button.setText("Generate")
            self.generate_button.setEnabled(False)
            self.validate_button.setText("Stop validation")
            self.validate_button.setEnabled(True)
            self.ai_find_button.setText("AI Find")
            self.ai_find_button.setEnabled(False)
            self.ai_find_input.setEnabled(False)
            self._update_fixup_button_state()
            return
        if self._ollama_thread is not None and self._ollama_action_name == "AI Find":
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
        active = connected and self._ollama_thread is None
        self.generate_button.setEnabled(
            active
            and (self.generate_tags_checkbox.isChecked() or self.generate_description_checkbox.isChecked())
        )
        self.validate_button.setEnabled(active)
        self.ai_find_input.setEnabled(active)
        self.ai_find_button.setEnabled(active and bool(self.ai_find_input.text().strip()))
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
            "- \"tag\": match an exact tag\n"
            "- 'text': match free text inside annotation content\n"
            "- &: AND operator\n"
            "- |: OR operator\n"
            "- ( ... ): group expressions\n\n"
            "Examples:\n"
            "- fixup & \"landscape\"\n"
            "- \"portrait\" | 'sunset'\n"
            "- (fixup & \"animal\") | 'night'"
        )
        QMessageBox.information(self, "Filter rules", rules_text)

    def _build_filter_runtime(self, tag_cache: dict[Path, set[str]] | None = None) -> _FilterRuntime:
        named_filters: dict[str, Callable[[ImageRecord], bool]] = {
            "fixup": lambda record: existing_fixup_path_for_image(record.image_path) is not None,
        }
        return _FilterRuntime(
            named_filters=named_filters,
            tag_filter=lambda record, tag: self._record_has_tag(record, tag, tag_cache=tag_cache),
            freetext_filter=lambda record, text: self._record_contains_freetext(record, text),
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
        visible = 0
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None and not item.isHidden():
                visible += 1
        return visible

    def _visible_position_for_row(self, row: int) -> int:
        if row < 0:
            return -1

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
            self._refresh_all_list_item_previews()
        else:
            for index, record in enumerate(self.records):
                if record.image_path == image_path:
                    self._update_list_item_preview(index)
                    break
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

        menu.exec(self.image_label.mapToGlobal(position))

    def _current_image_path(self) -> Path | None:
        record = self._current_record()
        if record is None:
            return None
        return record.image_path

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

    def open_fixup_dialog(self) -> None:
        record = self._current_record()
        if record is None:
            return

        try:
            regenerate_max_resolution_mpx = self._apply_query_downscale_setting()
        except OllamaError:
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

        def _capture_regenerate_settings(values: dict[str, int | float | bool]) -> None:
            nonlocal regenerate_tags_enabled
            nonlocal regenerate_description_enabled
            nonlocal regenerate_timeout_seconds
            nonlocal regenerate_retry_count
            nonlocal regenerate_max_resolution_mpx

            tags_enabled = values.get("tags_enabled")
            description_enabled = values.get("description_enabled")
            timeout_seconds = values.get("timeout_seconds")
            retry_count = values.get("retry_count")
            max_resolution_mpx = values.get("max_resolution_mpx")

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
                self.ollama_max_resolution_input.setText(self._format_mpx(regenerate_max_resolution_mpx))
                configure_runtime(max_image_pixels=max(1, int(regenerate_max_resolution_mpx * 1_000_000)))

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
                regenerate_max_resolution_mpx=regenerate_max_resolution_mpx,
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
        folder = self._loader_worker.folder if self._loader_worker is not None else None
        if folder is None:
            return

        self._root_directory = folder
        self._update_window_title(folder)
        self._icc_warning_paths = []
        self.records = []
        self.known_tags.clear()
        self.tag_counts.clear()
        self.list_widget.clear()
        self.image_label.setText("No image selected")
        self.tag_input.clear()
        self.tag_list.clear()
        self.current_index = -1

        # Tag completions depend on known tags, which we compute after load settles.
        self._refresh_tag_completions()

        self.statusBar().showMessage(f"Loading {total} images...")
        self._loader_worker.allow_processing()

    def _on_collision_detected(self, first: str, second: str) -> None:
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
        self.ollama_max_resolution_input.setEnabled(not loading and self._ollama_thread is None)
        self.ollama_use_button.setEnabled(not loading and self._ollama_thread is None)
        self.ai_find_input.setEnabled(not loading and self._ollama_thread is None)
        self.ai_find_button.setEnabled(not loading and self._ollama_thread is None and bool(self.ollama_model_name.strip()))
        self.fixup_button.setEnabled(False)
        if loading:
            self.generate_button.setEnabled(False)
            self.validate_button.setEnabled(False)
            self.ai_find_button.setEnabled(False)
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
        thumb_payload = data.get("thumbnail")
        thumb_image: QImage | None = None
        if isinstance(thumb_payload, dict):
            try:
                b = thumb_payload.get("bytes")
                width = int(thumb_payload.get("width", 0))
                height = int(thumb_payload.get("height", 0))
                bytes_per_line = int(thumb_payload.get("bytes_per_line", width * 4))
                if b is not None and width > 0 and height > 0 and bytes_per_line > 0:
                    qimage = QImage(
                        b,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format.Format_RGBA8888,
                    )
                    # Detach from the underlying bytes buffer to avoid lifetime issues.
                    thumb_image = qimage.copy()
            except Exception:
                thumb_image = None

        if not image_path_str or not text_path_str:
            return

        image_path = Path(image_path_str)
        text_path = Path(text_path_str)

        record = ImageRecord(image_path=image_path, text_path=text_path, text=text)
        self.records.append(record)
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
        # Compute tag completions after the list has populated (tags settle after load).
        self._rebuild_known_tags_from_records()
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
            if self.records:
                self.statusBar().showMessage(self._filtered_total_status_text())
            return

        self.current_index = index
        record = self.records[index]

        self._show_image(record.image_path)

        tags = self._parse_tags(record.text)
        self._populate_tag_list(tags)
        self.tag_input.clear()
        self._update_fixup_button_state()
        self.statusBar().showMessage(f"{self._filtered_total_status_text(selected_row=index)} {record.image_path}")

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
        if self._ollama_cancel is not None:
            self._ollama_cancel.cancel()
        if self._loader_worker is not None:
            self._loader_worker.cancel()

        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(2000)
        if self._ollama_thread is not None and self._ollama_thread.isRunning():
            self._ollama_thread.quit()
            self._ollama_thread.wait(2000)

        self._save_main_window_geometry()
        record = self._current_record()
        if record is not None:
            self._cfg["last_selected_image"] = str(record.image_path)
        else:
            self._cfg.pop("last_selected_image", None)

        configured_downscale = self._cfg.get("ollama_max_resolution_mpx", 5)
        try:
            fallback_value = float(configured_downscale)
            if fallback_value <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            fallback_value = 5.0

        try:
            self._cfg["ollama_max_resolution_mpx"] = self._ollama_max_resolution_mpx_value(show_message=False)
        except OllamaError:
            self._cfg["ollama_max_resolution_mpx"] = fallback_value

        configured_threads = self._cfg.get("ollama_threads", 1)
        try:
            fallback_threads = int(configured_threads)
            if fallback_threads < 0:
                raise ValueError()
        except (TypeError, ValueError):
            fallback_threads = 1

        try:
            self._cfg["ollama_threads"] = self._ollama_thread_count(show_message=False)
        except OllamaError:
            self._cfg["ollama_threads"] = fallback_threads

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
        self._refresh_known_tags_list()

    def _refresh_known_tags_list(self) -> None:
        if not hasattr(self, "known_tags_list"):
            return

        self.known_tags_list.clear()
        for tag in self._sorted_tag_suggestions():
            count = self.tag_counts.get(tag, 0)
            self.known_tags_list.addItem(f"{tag} ({count})")

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
            atomic_write_text(record.text_path, record.text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save text file:\n{exc}")
            return False

        self.statusBar().showMessage(f"{status_prefix}: {record.text_path.name}")
        return True

    def _ollama_threads_status_suffix(self) -> str:
        if self._ollama_action_name is None:
            return ""
        current = max(1, int(self._ollama_threads_current))
        if self._ollama_threads_auto_mode:
            return f" | threads: {current} auto"
        return f" | threads: {current}"

    def _run_parallel_ollama_jobs(
        self,
        jobs: list[object],
        requested_threads: int,
        action_prefix: str,
        cancel_token: OllamaCancellation,
        report_progress: Callable[[str], None],
        report_item: Callable[[object], None],
        process_one: Callable[[int, object], dict],
    ) -> None:
        total = len(jobs)
        if total <= 0:
            return

        auto_mode = requested_threads == 0
        if auto_mode:
            max_threads = min(total, 16)
            if max_threads < 1:
                max_threads = 1
            target_parallelism = 1
        else:
            max_threads = min(total, requested_threads)
            target_parallelism = max_threads

        self._ollama_threads_auto_mode = auto_mode
        self._ollama_threads_current = target_parallelism

        executor = ThreadPoolExecutor(max_workers=max_threads)
        in_flight: set = set()
        next_job_index = 0
        completed = 0

        try:
            while next_job_index < total and len(in_flight) < target_parallelism:
                future = executor.submit(process_one, next_job_index + 1, jobs[next_job_index])
                in_flight.add(future)
                next_job_index += 1

            while in_flight:
                cancel_token.raise_if_cancelled()
                finished = next(as_completed(in_flight))
                in_flight.remove(finished)

                payload = finished.result()
                completed += 1

                retried = bool(payload.get("retried")) if isinstance(payload, dict) else False
                if auto_mode:
                    if retried:
                        target_parallelism = max(1, target_parallelism - 1)
                    elif target_parallelism < max_threads:
                        target_parallelism += 1
                    self._ollama_threads_current = target_parallelism

                image_name = ""
                if isinstance(payload, dict):
                    image_name = str(payload.get("image_name", "")).strip()
                if image_name:
                    report_progress(f"{action_prefix}: processing {completed}/{total} - {image_name}")
                else:
                    report_progress(f"{action_prefix}: processing {completed}/{total}")

                report_item(payload)

                while next_job_index < total and len(in_flight) < target_parallelism:
                    future = executor.submit(process_one, next_job_index + 1, jobs[next_job_index])
                    in_flight.add(future)
                    next_job_index += 1
        except Exception:
            cancel_token.cancel()
            for future in in_flight:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def generate_with_ollama(self) -> None:
        if self._ollama_thread is not None and self._ollama_action_name == "Generate":
            self._request_stop_generation()
            return

        try:
            self._apply_query_downscale_setting()
        except OllamaError:
            return

        try:
            thread_count = self._ollama_thread_count()
        except OllamaError:
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

            def process_one(position: int, record_index: int) -> dict:
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
                    else:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_retry image={image_name} attempt={attempt + 1}/{retry_count + 1}",
                            flush=True,
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

                    print(
                        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] generation_failed image={image_name} attempt={attempt + 1}/{retry_count + 1} elapsed_s={elapsed_seconds:.2f}",
                        flush=True,
                    )

                return {
                    "kind": "generate_item",
                    "index": record_index,
                    "items": generated_items,
                    "retried": image_retried,
                    "position": position,
                    "total": total,
                    "image_name": image_name,
                }

            self._run_parallel_ollama_jobs(
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

        try:
            self._apply_query_downscale_setting()
        except OllamaError:
            return

        try:
            thread_count = self._ollama_thread_count()
        except OllamaError:
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

            def process_one(position: int, record_index: int, annotations: str) -> dict:
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
                    else:
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] validation_retry image={image_name} attempt={attempt + 1}/{retry_count + 1}",
                            flush=True,
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

                return {
                    "kind": "validate_item",
                    "index": record_index,
                    "result": validation_result,
                    "retried": image_retried,
                    "position": position,
                    "total": total,
                    "image_name": image_name,
                }

            self._run_parallel_ollama_jobs(
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

        self._start_ollama_task(
            task=validate_task,
            action_name="Validate",
            empty_message="Ollama returned no validation result.",
            cancel_token=cancel_token,
            validation_report=True,
        )

    def ai_find_with_ollama(self) -> None:
        if self._ollama_thread is not None and self._ollama_action_name == "AI Find":
            self._request_stop_ai_find()
            return

        try:
            self._apply_query_downscale_setting()
        except OllamaError:
            return

        try:
            thread_count = self._ollama_thread_count()
        except OllamaError:
            return

        query = " ".join(self.ai_find_input.text().split())
        if not query:
            QMessageBox.information(self, "Missing search text", "Enter text to search for before running AI Find.")
            return

        selected_indexes = self._selected_record_indexes()
        if not selected_indexes:
            QMessageBox.information(self, "No image selected", "Select one or more images before running AI Find.")
            return

        cancel_token = OllamaCancellation()

        def find_task(
            report_progress: Callable[[str], None],
            report_item: Callable[[object], None],
        ) -> object:
            timeout = self._ollama_timeout_seconds()
            retry_count = self._ollama_retry_count()
            total = len(selected_indexes)

            def process_one(position: int, record_index: int) -> dict:
                record = self.records[record_index]
                image_name = record.image_path.name
                image_retried = False
                matched = False

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
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] ai_find_retry image={image_name} attempt={attempt + 1}/{retry_count + 1}",
                            flush=True,
                        )

                    def remaining_timeout(_start=attempt_start) -> float:
                        elapsed = time.monotonic() - _start
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise OllamaError(
                                f"Timed out after {int(timeout)} seconds while searching {record.image_path.name}."
                            )
                        return remaining

                    try:
                        matched = image_matches_query(
                            self.ollama_server_url,
                            self.ollama_model_name,
                            record.image_path,
                            query,
                            timeout=remaining_timeout(),
                            cancellation=cancel_token,
                        )
                    except OllamaCancelled:
                        raise
                    except OllamaError:
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
                    "position": position,
                    "total": total,
                    "query": query,
                    "image_name": image_name,
                }

            self._run_parallel_ollama_jobs(
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

        self._start_ollama_task(
            task=find_task,
            action_name="AI Find",
            empty_message="No matching images were found.",
            cancel_token=cancel_token,
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
        self.ollama_max_resolution_input.setEnabled(False)
        self.ollama_threads_input.setEnabled(False)
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

    def _request_stop_ai_find(self) -> None:
        if self._ollama_action_name != "AI Find" or self._ollama_cancel is None:
            return
        self.ai_find_button.setEnabled(False)
        self.ai_find_button.setText("Stopping AI Find...")
        self.statusBar().showMessage("Stopping AI Find...")
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
            self.statusBar().showMessage(f"{message}{details}{self._ollama_threads_status_suffix()}")
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
            issues_text = f" | invalid: {self._validate_batch_issues}"
            self.statusBar().showMessage(f"{message}{details}{issues_text}{self._ollama_threads_status_suffix()}")
            return

        if self._ollama_action_name == "AI Find" and (
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
            self.statusBar().showMessage(f"{message}{details}{found_text}{self._ollama_threads_status_suffix()}")
            return

        self.statusBar().showMessage(f"{message}{self._ollama_threads_status_suffix()}")

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
                    f"Generate: applied {raw_position}/{raw_total} - {self.records[raw_index].image_path.name}{details}{self._ollama_threads_status_suffix()}"
                )
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
                    f"AI Find: applied {raw_position}/{raw_total} - {self.records[raw_index].image_path.name} ({result_text}){details}{found_text}{self._ollama_threads_status_suffix()}"
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
                f"Validate: processed {self._validate_batch_processed}/{raw_total}{details}{self._ollama_threads_status_suffix()}"
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
        self._ai_find_batch_total = 0
        self._ai_find_batch_processed = 0
        self._ai_find_batch_matched = 0
        self._ai_find_batch_started_at = None
        self._ai_find_batch_retry_images = 0
        self._ollama_action_name = None
        self._ollama_cancel = None
        self._ollama_threads_auto_mode = False
        self._ollama_threads_current = 0
        self._update_ollama_controls()
        self.ollama_server_input.setEnabled(True)
        self.ollama_fetch_button.setEnabled(True)
        self.ollama_model_combo.setEnabled(True)
        self.ollama_timeout_input.setEnabled(True)
        self.ollama_retry_input.setEnabled(True)
        self.ollama_max_resolution_input.setEnabled(True)
        self.ollama_threads_input.setEnabled(True)
        self.ollama_use_button.setEnabled(True)
        self._update_fixup_button_state()


def main() -> None:
    QImageReader.setAllocationLimit(1024)

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
