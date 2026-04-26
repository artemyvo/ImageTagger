from __future__ import annotations

from difflib import SequenceMatcher
from dataclasses import dataclass, field
import os
import re
import sys
import time
from typing import Callable

from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, Qt, QSize, QStringListModel, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeySequence, QPixmap, QPainter, QPalette, QTextCharFormat, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QFileDialog,
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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from imagetagger.utils.annotations import normalize_description_text, remove_commas_from_description
from imagetagger import config as _config
from imagetagger.utils.image_prep import configure_image_preparation, consume_image_preparation_warning
from imagetagger.utils.image_reload_helper import ImageReloadHelper
from imagetagger.utils.input_validators import InputValidator
from imagetagger.utils.validators import (
    create_max_resolution_validator,
    create_retry_validator,
    create_timeout_validator,
)
from imagetagger.providers.llm_provider import (
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
    VisionLlmProvider,
    VisionLlmSession,
)
from imagetagger.utils.llm_queries import (
    active_prompt_for_kind,
    render_prompt_with_user_hint,
)
from imagetagger.utils.external_editors import (
    ExternalEditor,
    discover_graphics_editors,
    launch_image_in_editor,
    launch_image_in_system_default,
)
from imagetagger.ui.shortcuts import native_shortcut_text, platform_key_sequence, platform_key_sequences
from imagetagger.ui.item_action_widget import ItemActionWidget
from imagetagger.ui.workers import RegenerateWorker
from imagetagger.ui.diff_delegates import (
    DIFF_RANGES_ROLE,
    ITEM_TEXT_ROLE,
    IS_SEARCH_MATCH_ROLE,
    DiffHighlightDelegate,
    EditableDiffDelegate,
    _danger_button_stylesheet,
    _diff_highlight_colors,
    strip_tag_list_prefix,
    _normalize_search_match_entry,
)
from imagetagger.ui.scalable_image_label import ScalableImageLabel
from imagetagger.ui.server_settings_frame import create_server_settings_frame
from imagetagger.utils.theme_colors import danger_accent_color





@dataclass
class FixupData:
    issues: str
    corrected_description: str
    corrected_description_raw: str
    corrected_tags: list[str]
    search_matches: list[str] = field(default_factory=list)
    has_headers: bool = False











def parse_fixup_data(
    content: str,
    parse_tags: Callable[[str], list[str]],
    sanitize_annotation: Callable[[str], str],
) -> FixupData:
    sections: dict[str, list[str]] = {"issues": [], "tags": [], "description": [], "ai_find": []}
    current_section = "issues"
    has_headers = False

    # Robust regex for headers like "ISSUES:", "### Tags :", "**Description**:", etc.
    # CRITICAL: Must require ':' after the keyword to avoid matching content lines like
    # "Description incorrectly..." which would incorrectly be treated as a DESCRIPTION header.
    header_pattern = re.compile(r"^[#*_\s>\-]*(ISSUES|TAGS|DESCRIPTION|AI_FIND_MATCHES)\b\s*:", re.IGNORECASE)

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = header_pattern.match(line)
        if match:
            has_headers = True
            header_keyword = match.group(1).upper()
            if header_keyword == "ISSUES":
                current_section = "issues"
            elif header_keyword == "TAGS":
                current_section = "tags"
            elif header_keyword == "DESCRIPTION":
                current_section = "description"
            elif header_keyword == "AI_FIND_MATCHES":
                current_section = "ai_find"

            # Handle content on the same line after the header (after colon or match end)
            sep_idx = line.find(":")
            content_start = sep_idx + 1 if sep_idx != -1 else match.end()
            inline_content = line[content_start:].strip()
            if inline_content:
                sections[current_section].append(inline_content)
            continue

        sections[current_section].append(raw_line.rstrip())

    issues = "\n".join(line for line in sections["issues"] if line.strip()).strip()
    corrected_description_raw = "\n".join(line for line in sections["description"] if line.strip()).strip()
    corrected_description = normalize_description_text(corrected_description_raw)
    # Remove any commas that may have snuck into the description from the fixup file
    corrected_description = remove_commas_from_description(corrected_description)
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
        corrected_description_raw=corrected_description_raw,
        corrected_tags=corrected_tags,
        search_matches=search_matches,
        has_headers=has_headers,
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
    _SWIPE_MIN_DISTANCE_PX = 90
    _SWIPE_MAX_VERTICAL_DRIFT_PX = 48
    _SWIPE_HORIZONTAL_BIAS = 1.2
    _HSCROLL_TRACKPAD_THRESHOLD_PX = 84.0
    _HSCROLL_MOUSE_NOTCH_EQUIVALENT_PX = 96.0
    _HSCROLL_DEFAULT_STOP_IDLE_SECONDS = 0.45
    _HSCROLL_TARGET_POINTER_ROW = 1
    _HSCROLL_TARGET_SELECTED_ROW = 2
    _HSCROLL_TARGET_POINTER_ON_SELECTED = 3

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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window_title_base = title_text or "Fixup"
        self.setWindowTitle(self._window_title_base)
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
        self._merge_table_double_click_action_enabled = bool(merge_table_double_click_action_enabled)
        self._merge_table_swipe_actions_enabled = bool(merge_table_swipe_actions_enabled)
        self._merge_table_horizontal_scroll_actions_enabled = bool(merge_table_horizontal_scroll_actions_enabled)
        self._merge_table_horizontal_scroll_reverse_enabled = bool(merge_table_horizontal_scroll_reverse_enabled)
        self._merge_table_horizontal_scroll_stop_idle_seconds = max(
            0.0,
            float(merge_table_horizontal_scroll_stop_idle_seconds),
        )
        if merge_table_horizontal_scroll_row_target_mode in (
            self._HSCROLL_TARGET_POINTER_ROW,
            self._HSCROLL_TARGET_SELECTED_ROW,
            self._HSCROLL_TARGET_POINTER_ON_SELECTED,
        ):
            self._merge_table_horizontal_scroll_row_target_mode = int(merge_table_horizontal_scroll_row_target_mode)
        else:
            self._merge_table_horizontal_scroll_row_target_mode = self._HSCROLL_TARGET_POINTER_ON_SELECTED
        self._comparison_swipe_drag_active = False
        self._comparison_swipe_start_pos: tuple[float, float] | None = None
        self._comparison_swipe_row = -1
        self._comparison_hscroll_accumulator_x = 0.0
        self._comparison_hscroll_row = -1
        self._comparison_hscroll_wait_for_stop = False
        self._comparison_hscroll_rearm_after = 0.0
        self._image_path = Path(image_path) if isinstance(image_path, str) else image_path
        self._image_label: ScalableImageLabel | None = None
        self._image_header_label: QLabel | None = None
        self._provider_session = provider_session
        self._llm_provider = provider
        self._llm_endpoint = regenerate_model_endpoint or ""
        self._llm_model_name = regenerate_model_name or ""
        self._regenerate_user_hint_value = regenerate_user_hint or ""
        self._regenerate_selected_model = regenerate_model_name or ""
        self._regenerate_thread: QThread | None = None
        self._regenerate_worker: RegenerateWorker | None = None
        self._regenerate_cancel: LlmRequestCancellation | None = None
        self._regenerate_started_at: float | None = None
        self._discard_regenerate_result = False
        self._global_key_filter_installed = False
        self._initial_table_focus_applied = False
        self._search_matches = fixup_data.search_matches or []
        self._delete_image = delete_image
        self._confirm_delete = bool(confirm_delete)
        self._detected_external_editors: list[ExternalEditor] | None = None

        # Image reload detection for external editor changes
        self._image_reload_helper = ImageReloadHelper(self, self._on_image_reload)

        self.left_list = QListWidget(self)
        self.left_list.setItemDelegate(DiffHighlightDelegate(self.left_list))
        self.left_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.left_list.setWordWrap(True)
        self.left_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.left_list.setUniformItemSizes(False)
        self.left_list.hide()

        self.remove_left_tag_action = QAction("Remove Selected Existing Tags", self)
        delete_shortcuts = [QKeySequence("Delete"), QKeySequence("Backspace")]
        self.remove_left_tag_action.setShortcuts(delete_shortcuts)
        self._delete_rows_shortcut_hint = native_shortcut_text(delete_shortcuts)
        self.remove_left_tag_action.triggered.connect(self._remove_selected_left_items)
        self.left_list.addAction(self.remove_left_tag_action)
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
        self.focus_tag_input_action.triggered.connect(self._focus_left_tag_input)
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
        self.prev_actionable_row_action.triggered.connect(self._activate_previous_actionable_row)
        self.addAction(self.prev_actionable_row_action)

        self.next_actionable_row_action = QAction("Next Actionable Row", self)
        next_actionable_shortcut = platform_key_sequence("Alt+Down", "Alt+Down")
        self.next_actionable_row_action.setShortcut(next_actionable_shortcut)
        self._next_actionable_shortcut_hint = native_shortcut_text(next_actionable_shortcut)
        self.next_actionable_row_action.triggered.connect(self._activate_next_actionable_row)
        self.addAction(self.next_actionable_row_action)

        self.clear_hint_action = QAction("Clear User Hint", self)
        clear_hint_shortcut = platform_key_sequence("Alt+H", "Alt+H")
        self.clear_hint_action.setShortcut(clear_hint_shortcut)
        self._clear_hint_shortcut_hint = native_shortcut_text(clear_hint_shortcut)
        self.clear_hint_action.triggered.connect(self._clear_and_focus_hint)
        self.addAction(self.clear_hint_action)

        self.left_tag_input = QLineEdit(self)
        self.left_tag_input.setPlaceholderText("Type a tag and press Enter")
        self.left_tag_input.setToolTip(
            f"Type a tag and press Enter. {self._focus_tag_input_shortcut_hint} clears and focuses this field."
        )
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
            f"Keyboard: {self._prev_actionable_shortcut_hint}/{self._next_actionable_shortcut_hint} "
            "jump actionable rows, Left applies proposed rows, "
            f"Enter triggers current row action, {self._delete_rows_shortcut_hint} removes selected current rows."
        )
        self.comparison_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.comparison_table.customContextMenuRequested.connect(self._show_comparison_table_context_menu)
        self.comparison_table.itemChanged.connect(self._on_comparison_item_changed)
        self.comparison_table.itemSelectionChanged.connect(self._on_comparison_selection_changed)
        self.comparison_table.installEventFilter(self)
        self.comparison_table.viewport().installEventFilter(self)
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
        self.regenerate_timeout_input.setValidator(create_timeout_validator(self))
        self.regenerate_timeout_input.setText(str(max(1, int(regenerate_timeout_seconds))))
        self.regenerate_timeout_input.setMaximumWidth(90)

        self.regenerate_retry_input = QLineEdit(self)
        self.regenerate_retry_input.setValidator(create_retry_validator(self))
        self.regenerate_retry_input.setText(str(max(0, int(regenerate_retry_count))))
        self.regenerate_retry_input.setMaximumWidth(60)

        self.regenerate_max_resolution_input = QLineEdit(self)
        self.regenerate_max_resolution_input.setValidator(create_max_resolution_validator(self))
        try:
            max_resolution_value = float(regenerate_max_resolution_mpx)
            if max_resolution_value <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            max_resolution_value = 5.0
        self.regenerate_max_resolution_input.setText(self._format_mpx(max_resolution_value))
        self.regenerate_max_resolution_input.setMaximumWidth(80)

        # Server controls for model selection during merge
        self.llm_endpoint_input = QLineEdit(self)
        self.llm_endpoint_input.setPlaceholderText("http://127.0.0.1:11434 (Ollama) or :8000 (OpenAI-compatible)")
        if self._llm_provider is not None:
            self.llm_endpoint_input.setText(self._llm_provider.default_endpoint)

        self.llm_fetch_button = QPushButton("Fetch models", self)
        self.llm_fetch_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.llm_fetch_button.clicked.connect(self._fetch_provider_models)

        self.llm_model_combo = QComboBox(self)
        self.llm_model_combo.setEditable(False)

        self.llm_use_button = QPushButton("Use", self)
        self.llm_use_button.clicked.connect(self._use_selected_provider_model)

        self.regenerate_user_hint_input = QTextEdit(self)
        self.regenerate_user_hint_input.setAcceptRichText(False)
        self._regenerate_user_hint_placeholder = "User hint (optional). Example: The cat is not a Maine Coon."
        self.regenerate_user_hint_input.setPlaceholderText("")
        self.regenerate_user_hint_input.setToolTip(
            "Optional guidance used only for this regenerate run."
        )
        user_hint_height = (self.regenerate_user_hint_input.fontMetrics().lineSpacing() * 2) + 14
        self.regenerate_user_hint_input.setFixedHeight(user_hint_height)
        self.regenerate_user_hint_input.textChanged.connect(self._update_regenerate_user_hint_visibility)
        self.regenerate_user_hint_overlay_label = QLabel(
            self._regenerate_user_hint_placeholder,
            self.regenerate_user_hint_input.viewport(),
        )
        self.regenerate_user_hint_overlay_label.setWordWrap(True)
        self.regenerate_user_hint_overlay_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.regenerate_user_hint_overlay_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.regenerate_user_hint_overlay_label.setStyleSheet("color: palette(mid);")

        self.regenerate_user_hint_clear_button = QPushButton("Clear", self)
        self.regenerate_user_hint_clear_button.setToolTip("Clear the current user hint.")
        self.regenerate_user_hint_clear_button.clicked.connect(self._clear_regenerate_user_hint)

        self.regenerate_button = QPushButton("Regenerate", self)
        regenerate_shortcut = platform_key_sequence("Alt+R", "Alt+R")
        self._regenerate_shortcut_hint = native_shortcut_text(regenerate_shortcut)
        self.regenerate_button.setToolTip(
            f"Regenerate proposed annotations with the selected model ({self._regenerate_shortcut_hint})"
        )
        self.regenerate_button.clicked.connect(self._regenerate_proposed_annotations)

        self.regenerate_alt_action = QAction("Regenerate", self)
        self.regenerate_alt_action.setShortcut(regenerate_shortcut)
        self.regenerate_alt_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.regenerate_alt_action.triggered.connect(self._regenerate_proposed_annotations)
        self.addAction(self.regenerate_alt_action)

        self.regenerate_status_label = QLabel(self)
        self.regenerate_status_label.setWordWrap(True)
        self.regenerate_status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.regenerate_status_label.setMinimumWidth(0)
        self.regenerate_status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        self.issues_label = QLabel(fixup_data.issues or "No issue details provided.", self)
        self.issues_label.setWordWrap(True)
        self.issues_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.issues_label.setStyleSheet("border: 1px solid palette(mid); padding: 6px;")

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
        self._initialize_provider_controls()
        self._update_regenerate_controls()
        self._set_watched_image(self._image_path)

    def _set_watched_image(self, image_path: Path | None) -> None:
        self._image_reload_helper.set_watched_image(image_path)

    def _on_image_reload(self, image_path: Path) -> None:
        """Callback invoked when watched image is reloaded."""
        if not self._load_dialog_image_preview(image_path):
            return
        self.regenerate_status_label.setText(f"Reloaded image: {image_path.name}")

    @staticmethod
    def _load_normalized_pixmap(image_path: Path) -> QPixmap:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return pixmap
        # Normalize high-DPI asset naming semantics (for example "@2x") so
        # preview sizing is consistent across file names and formats.
        pixmap.setDevicePixelRatio(1.0)
        return pixmap

    def _load_dialog_image_preview(self, image_path: Path | None) -> bool:
        image_label = self._image_label
        if image_label is None:
            return False

        if image_path is None:
            self._set_image_header_text(None, None)
            image_label.clear_original_image("No image path provided")
            return False

        if not image_path.exists() or not image_path.is_file():
            self._set_image_header_text(None, None)
            image_label.clear_original_image(f"File not found:\n{image_path.name}")
            return False

        pixmap = self._load_normalized_pixmap(image_path)
        if pixmap.isNull():
            self._set_image_header_text(None, None)
            image_label.clear_original_image(f"Unsupported format:\n{image_path.name}")
            return False

        self._set_image_header_text(pixmap.width(), pixmap.height())
        image_label.setText("")
        image_label.set_original_image(pixmap)
        return True

    def _set_image_header_text(self, width: int | None, height: int | None) -> None:
        label = self._image_header_label
        if label is None:
            return
        if width is None or height is None or width <= 0 or height <= 0:
            label.setText("Image")
            return
        megapixels = (float(width) * float(height)) / 1_000_000.0
        label.setText(f"Image: {width}x{height} - {megapixels:0.1f} MPx")

    def _apply_pending_image_reload(self) -> None:
        if not self._image_reload_pending:
            return

        self._image_reload_pending = False
        image_path = self._watched_image_path
        if image_path is None:
            return

        # External editors may save with atomic replace, so ensure watcher is re-attached.
        self._ensure_watched_image_subscription()

        if not self._load_dialog_image_preview(image_path):
            return

        self._watched_image_mtime_ns = self._image_mtime_ns(image_path)
        self.regenerate_status_label.setText(f"Reloaded image: {image_path.name}")

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
            self.left_tag_input.clear()
            self.left_tag_input.setFocus(Qt.FocusReason.OtherFocusReason)
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
                # Special handling for descriptions: find existing description and replace it in-place
                # rather than always inserting at position 0 (which could reorder existing items)
                if self._has_proposed_description and right_row == 0:
                    existing_desc_index = self._find_description_like_index(left_texts)
                    if existing_desc_index is not None and 0 <= existing_desc_index < self.left_list.count():
                        # Replace existing description in its current position
                        desc_item = self.left_list.item(existing_desc_index)
                        if self.left_list.itemWidget(desc_item) is not None:
                            self.left_list.setItemWidget(desc_item, None)
                        desc_item.setText(proposed_text)
                        desc_item.setData(ITEM_TEXT_ROLE, proposed_text)
                    else:
                        # No existing description found, insert new one at position 0
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
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return False

        current_row = self.comparison_table.currentRow()
        if 0 <= current_row < row_count:
            target_row = current_row + step
            if 0 <= target_row < row_count:
                return self._select_comparison_row(target_row, Qt.FocusReason.ShortcutFocusReason)
            return False

        if self._last_action_table_row is None:
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

    def _apply_initial_table_focus(self) -> None:
        if self._initial_table_focus_applied:
            return
        self._initial_table_focus_applied = True

        self.comparison_table.setFocus()
        row_count = self.comparison_table.rowCount()
        if row_count <= 0:
            return

        for row in range(row_count):
            if self._row_needs_addressing(row):
                self._select_comparison_row(row, Qt.FocusReason.ActiveWindowFocusReason)
                return

        self._select_comparison_row(0, Qt.FocusReason.ActiveWindowFocusReason)

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

    def _swipe_min_distance_px(self) -> int:
        return max(self._SWIPE_MIN_DISTANCE_PX, QApplication.startDragDistance() * 6)

    def _reset_comparison_swipe_tracking(self) -> None:
        self._comparison_swipe_drag_active = False
        self._comparison_swipe_start_pos = None
        self._comparison_swipe_row = -1

    def _reset_comparison_hscroll_tracking(self) -> None:
        self._comparison_hscroll_accumulator_x = 0.0
        self._comparison_hscroll_row = -1

    def _reset_comparison_hscroll_blocking(self) -> None:
        self._comparison_hscroll_wait_for_stop = False
        self._comparison_hscroll_rearm_after = 0.0

    def _horizontal_scroll_delta_from_wheel(self, event) -> float:
        pixel_delta = event.pixelDelta()  # type: ignore[attr-defined]
        if not pixel_delta.isNull():
            return float(pixel_delta.x())

        angle_delta = event.angleDelta()  # type: ignore[attr-defined]
        if angle_delta.x() == 0:
            return 0.0

        # Typical wheel notches report angleDelta.x() of +/-120.
        # Convert notches to a px-like scale so threshold logic can be shared.
        return (float(angle_delta.x()) / 120.0) * self._HSCROLL_MOUSE_NOTCH_EQUIVALENT_PX

    @staticmethod
    def _wheel_is_mostly_horizontal(event) -> bool:
        pixel_delta = event.pixelDelta()  # type: ignore[attr-defined]
        if not pixel_delta.isNull():
            return abs(pixel_delta.x()) > abs(pixel_delta.y()) * 1.1

        angle_delta = event.angleDelta()  # type: ignore[attr-defined]
        if angle_delta.x() == 0:
            return False
        return abs(angle_delta.x()) > abs(angle_delta.y()) * 1.1

    def _handle_comparison_table_horizontal_scroll(self, row: int, delta_x: float) -> bool:
        now = time.monotonic()
        if self._comparison_hscroll_wait_for_stop:
            stop_idle_seconds = self._merge_table_horizontal_scroll_stop_idle_seconds
            if now < self._comparison_hscroll_rearm_after:
                self._comparison_hscroll_rearm_after = now + stop_idle_seconds
                return True
            # Scrolling was idle long enough: arm the next gesture.
            self._reset_comparison_hscroll_blocking()
            self._reset_comparison_hscroll_tracking()

        if row != self._comparison_hscroll_row:
            self._comparison_hscroll_row = row
            self._comparison_hscroll_accumulator_x = 0.0

        self._comparison_hscroll_accumulator_x += delta_x
        if abs(self._comparison_hscroll_accumulator_x) < self._HSCROLL_TRACKPAD_THRESHOLD_PX:
            return True

        self._select_comparison_row(row, Qt.FocusReason.MouseFocusReason)
        handled_action = False
        direction_is_right = self._comparison_hscroll_accumulator_x > 0
        if self._merge_table_horizontal_scroll_reverse_enabled:
            direction_is_right = not direction_is_right

        if direction_is_right:
            handled_action = self._remove_value_for_table_row(row)
        else:
            handled_action = self._apply_proposed_value_for_table_row(row)

        self._comparison_hscroll_accumulator_x = 0.0
        self._comparison_hscroll_row = -1
        if handled_action:
            if self._merge_table_horizontal_scroll_stop_idle_seconds > 0.0:
                self._comparison_hscroll_wait_for_stop = True
                self._comparison_hscroll_rearm_after = (
                    now + self._merge_table_horizontal_scroll_stop_idle_seconds
                )
            return True
        return False

    def _horizontal_scroll_target_row(self, pointer_row: int) -> int | None:
        mode = self._merge_table_horizontal_scroll_row_target_mode
        selected_row = self.comparison_table.currentRow()

        if mode == self._HSCROLL_TARGET_POINTER_ROW:
            return pointer_row if pointer_row >= 0 else None

        if mode == self._HSCROLL_TARGET_SELECTED_ROW:
            return selected_row if selected_row >= 0 else None

        # Default/safest mode: only act when pointer is over the selected row.
        if selected_row >= 0 and pointer_row == selected_row:
            return selected_row
        return None

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

        # Prefer the existing row action button path so swipe behavior
        # stays in sync with button and Enter-triggered actions.
        button = self._action_button_for_row(row)
        if button is not None and button.text() in ("←", "🔍"):
            return self._trigger_action_for_table_row(row)

        self._remember_last_action_table_row(row)
        self._apply_proposed_rows([right_index])
        self._refresh_button_state()
        self._update_difference_highlights()
        self._advance_to_next_actionable_from(row + 1)
        return True

    def _handle_comparison_table_swipe(self, row: int, delta_x: float, delta_y: float) -> bool:
        min_distance = float(self._swipe_min_distance_px())
        max_vertical_drift = max(float(self._SWIPE_MAX_VERTICAL_DRIFT_PX), min_distance * 0.6)

        if abs(delta_x) < min_distance:
            return False
        if abs(delta_y) > max_vertical_drift:
            return False
        if abs(delta_x) < abs(delta_y) * self._SWIPE_HORIZONTAL_BIAS:
            return False

        self._select_comparison_row(row, Qt.FocusReason.MouseFocusReason)
        if delta_x > 0:
            return self._remove_value_for_table_row(row)
        return self._apply_proposed_value_for_table_row(row)

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

    def _trigger_action_for_table_row(self, row: int) -> bool:
        button = self._action_button_for_row(row)
        if button is None:
            return False

        # ✕ removes the row so rows below shift up by 1: scan from
        # start_row in the rebuilt table (it now points to the old
        # start_row+1).  ← / 🔍 keep the row so we skip past it.
        removes_row = button.text() == "✕"
        button.click()
        advance_from = row if removes_row else row + 1
        self._advance_to_next_actionable_from(advance_from)
        return True

    def _trigger_action_for_current_row(self) -> bool:
        row = self.comparison_table.currentRow()
        return self._trigger_action_for_table_row(row)

    def _show_comparison_table_context_menu(self, position) -> None:
        row = self.comparison_table.rowAt(position.y())
        if row < 0:
            return

        self._select_comparison_row(row, Qt.FocusReason.MouseFocusReason)

        button = self._action_button_for_row(row)
        if button is None:
            return

        menu = QMenu(self)
        label = self._action_context_menu_label(button.text())
        row_action = menu.addAction(label)
        row_action.triggered.connect(lambda _checked=False, table_row=row: self._trigger_action_for_table_row(table_row))
        menu.exec(self.comparison_table.viewport().mapToGlobal(position))

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
        has_acceptable_proposals = self._has_acceptable_proposals()
        has_local_changes = self._has_local_changes()
        has_dialog_changes = self._has_dialog_state_changes()
        regenerate_in_progress = self._regenerate_thread is not None
        self._update_merge_next_button_presentation()
        self.accept_button.setEnabled(has_acceptable_proposals and not regenerate_in_progress)
        self.merge_button.setEnabled(has_local_changes and not regenerate_in_progress)
        self.undo_button.setEnabled((self._undo_available or has_dialog_changes) and not regenerate_in_progress)
        self.merge_next_button.setEnabled(not regenerate_in_progress)
        self._set_comparison_action_buttons_enabled(not regenerate_in_progress)
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

    def _set_comparison_action_buttons_enabled(self, enabled: bool) -> None:
        for row in range(self.comparison_table.rowCount()):
            action_host = self.comparison_table.cellWidget(row, 1)
            if action_host is None:
                continue
            for button in action_host.findChildren(QPushButton):
                button.setEnabled(enabled)

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

    def _merged_values_for_accept_all(self) -> list[str]:
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
        return merged_values

    def _has_acceptable_proposals(self) -> bool:
        current_values = [
            value.strip()
            for value in self._current_texts(self.left_list)
            if value.strip()
        ]
        merged_values = self._merged_values_for_accept_all()

        current_normalized = [self._normalized_compare_text(value) for value in current_values]
        merged_normalized = [self._normalized_compare_text(value) for value in merged_values]
        return current_normalized != merged_normalized

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
            button.setEnabled(self._regenerate_thread is None)
            if action_text == "✕":
                button.setStyleSheet(_danger_button_stylesheet(button.palette()))
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
        merged_values = self._merged_values_for_accept_all()

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

        def _is_plain_arrow_modifiers(modifiers: Qt.KeyboardModifier) -> bool:
            # Some macOS keyboards report arrow keys with KeypadModifier.
            return not bool(modifiers & ~Qt.KeyboardModifier.KeypadModifier)

        if self._merge_table_swipe_actions_enabled and watched is comparison_table.viewport():
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
                row = comparison_table.rowAt(int(event.position().y()))  # type: ignore[attr-defined]
                if row >= 0:
                    position = event.position()  # type: ignore[attr-defined]
                    self._comparison_swipe_drag_active = True
                    self._comparison_swipe_start_pos = (float(position.x()), float(position.y()))
                    self._comparison_swipe_row = row
                else:
                    self._reset_comparison_swipe_tracking()
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
                handled_swipe = False
                if (
                    self._comparison_swipe_drag_active
                    and self._comparison_swipe_start_pos is not None
                    and self._comparison_swipe_row >= 0
                ):
                    start_x, start_y = self._comparison_swipe_start_pos
                    position = event.position()  # type: ignore[attr-defined]
                    row = comparison_table.rowAt(int(position.y()))
                    if row == self._comparison_swipe_row:
                        delta_x = float(position.x()) - start_x
                        delta_y = float(position.y()) - start_y
                        handled_swipe = self._handle_comparison_table_swipe(row, delta_x, delta_y)
                self._reset_comparison_swipe_tracking()
                if handled_swipe:
                    return True
            elif event.type() == QEvent.Type.Leave:
                self._reset_comparison_swipe_tracking()

        if self._merge_table_horizontal_scroll_actions_enabled and watched is comparison_table.viewport():
            if event.type() == QEvent.Type.Wheel:
                if not self._wheel_is_mostly_horizontal(event):
                    return super().eventFilter(watched, event)
                pointer_row = comparison_table.rowAt(int(event.position().y()))  # type: ignore[attr-defined]
                row = self._horizontal_scroll_target_row(pointer_row)
                if row is None:
                    self._reset_comparison_hscroll_tracking()
                    self._reset_comparison_hscroll_blocking()
                    return super().eventFilter(watched, event)
                delta_x = self._horizontal_scroll_delta_from_wheel(event)
                if delta_x == 0.0:
                    return super().eventFilter(watched, event)
                if self._handle_comparison_table_horizontal_scroll(row, delta_x):
                    return True
            elif event.type() == QEvent.Type.Leave:
                self._reset_comparison_hscroll_tracking()
                self._reset_comparison_hscroll_blocking()

        if (
            self._merge_table_double_click_action_enabled
            and watched is comparison_table.viewport()
            and event.type() == QEvent.Type.MouseButtonDblClick
            and event.button() == Qt.MouseButton.LeftButton  # type: ignore[attr-defined]
        ):
            row = comparison_table.rowAt(int(event.position().y()))  # type: ignore[attr-defined]
            if row >= 0:
                comparison_table.setCurrentCell(row, max(comparison_table.currentColumn(), 0))
                if self._trigger_action_for_current_row():
                    return True

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

        if watched is comparison_table and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() == Qt.KeyboardModifier.NoModifier and self._trigger_action_for_current_row():
                    return True
            if event.key() == Qt.Key.Key_Home and _is_plain_arrow_modifiers(event.modifiers()):
                return self._activate_first_comparison_row()
            if event.key() == Qt.Key.Key_End and _is_plain_arrow_modifiers(event.modifiers()):
                return self._activate_last_comparison_row()
            if sys.platform == "darwin" and event.modifiers() == Qt.KeyboardModifier.MetaModifier:
                if event.key() == Qt.Key.Key_Up:
                    return self._activate_first_comparison_row()
                if event.key() == Qt.Key.Key_Down:
                    return self._activate_last_comparison_row()

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

        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            if not _is_plain_arrow_modifiers(event.modifiers()):
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

            if isinstance(focused, (QLineEdit, QTextEdit, QAbstractItemView, QPushButton, QCheckBox, QSplitter)):
                return super().eventFilter(watched, event)

            if self.isAncestorOf(focused):
                if self._activate_adjacent_row_for_last_action(step):
                    return True
                if event.key() == Qt.Key.Key_Down and self._activate_first_comparison_row():
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
                self._activate_first_comparison_row
                if event.key() in (Qt.Key.Key_Home, Qt.Key.Key_Up)
                else self._activate_last_comparison_row
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
        def show_error(msg: str) -> None:
            QMessageBox.warning(self, "Invalid timeout", msg)
        return InputValidator.parse_timeout_seconds(self.regenerate_timeout_input.text(), show_error)

    def _regenerate_retry_count(self) -> int:
        return InputValidator.parse_retry_count(self.regenerate_retry_input.text())

    @staticmethod
    def _format_mpx(value: float) -> str:
        return InputValidator.format_megapixels(value)

    def _regenerate_max_resolution_mpx(self) -> float:
        def show_error(msg: str) -> None:
            QMessageBox.warning(self, "Invalid query downscale", msg)
        return InputValidator.parse_max_resolution_mpx(self.regenerate_max_resolution_input.text(), show_error)

    def _update_regenerate_controls(self) -> None:
        connected = self._active_regenerate_session(show_errors=False) is not None
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
        self.llm_endpoint_input.setEnabled(not working)
        self.llm_fetch_button.setEnabled(not working and self._llm_provider is not None)
        self.llm_model_combo.setEnabled(not working)
        self.llm_use_button.setEnabled(not working and self.llm_model_combo.count() > 0 and self._llm_provider is not None)
        self.regenerate_user_hint_input.setEnabled(not working)
        self.regenerate_user_hint_clear_button.setEnabled(not working)
        self._update_regenerate_user_hint_visibility()

        if working:
            self.regenerate_button.setEnabled(False)
            self.regenerate_button.setText("Regenerating...")
            self.regenerate_alt_action.setEnabled(False)
            return

        regenerate_enabled = connected and options_selected and self._image_path is not None
        self.regenerate_button.setText("Regenerate")
        self.regenerate_button.setEnabled(regenerate_enabled)
        self.regenerate_alt_action.setEnabled(regenerate_enabled)
        if not connected:
            self.regenerate_status_label.setText(
                "Regenerate is disabled until a model is selected in this dialog or in the main window."
            )

    def _active_regenerate_session(self, *, show_errors: bool) -> VisionLlmSession | None:
        if self._llm_provider is not None and self._llm_model_name.strip():
            endpoint = self._llm_endpoint.strip() or self.llm_endpoint_input.text().strip()
            if not endpoint:
                if show_errors:
                    QMessageBox.warning(self, "Server required", "Enter a server endpoint before regenerating.")
                return None
            try:
                normalized_endpoint = self._llm_provider.normalize_endpoint(endpoint)
            except LlmProviderError as exc:
                if show_errors:
                    QMessageBox.warning(self, "Invalid server", str(exc))
                return None
            self._llm_endpoint = normalized_endpoint
            self.llm_endpoint_input.setText(normalized_endpoint)
            return self._llm_provider.create_session(normalized_endpoint, self._llm_model_name)
        return self._provider_session

    def _initialize_provider_controls(self) -> None:
        if self._provider_session is None:
            return

        session_endpoint = str(getattr(self._provider_session, "endpoint", "")).strip()
        session_model = str(getattr(self._provider_session, "model_name", "")).strip()

        if session_endpoint:
            # Only set endpoint if one wasn't already provided via persistence
            if not self._llm_endpoint.strip():
                self._llm_endpoint = session_endpoint
                self.llm_endpoint_input.setText(session_endpoint)

        if session_model:
            # Only set model if one wasn't already provided via persistence
            if not self._llm_model_name.strip():
                self._llm_model_name = session_model
                if self.llm_model_combo.findText(session_model) == -1:
                    self.llm_model_combo.addItem(session_model)
                self.llm_model_combo.setCurrentText(session_model)

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
        return create_server_settings_frame(
            parent=self,
            endpoint_input=self.llm_endpoint_input,
            fetch_button=self.llm_fetch_button,
            model_combo=self.llm_model_combo,
            use_button=self.llm_use_button,
            include_tags_checkbox=self.regenerate_tags_checkbox,
            include_description_checkbox=self.regenerate_description_checkbox,
            timeout_input=self.regenerate_timeout_input,
            retry_input=self.regenerate_retry_input,
            max_resolution_input=self.regenerate_max_resolution_input,
        )

    def _create_regenerate_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._create_regenerate_frame(), stretch=0)

        user_hint_row = QHBoxLayout()
        user_hint_row.setContentsMargins(0, 0, 0, 0)
        user_hint_row.setSpacing(6)
        user_hint_row.addWidget(self.regenerate_user_hint_input, stretch=1)
        user_hint_row.addWidget(self.regenerate_user_hint_clear_button, stretch=0)
        layout.addLayout(user_hint_row)

        self._position_regenerate_user_hint_overlay()

        return panel

    def _regenerate_user_hint_text(self) -> str:
        return self.regenerate_user_hint_input.toPlainText().strip()

    def _position_regenerate_user_hint_overlay(self) -> None:
        overlay_margins = int(self.regenerate_user_hint_input.document().documentMargin())
        x_pos = 8
        y_pos = overlay_margins + 2
        viewport_width = self.regenerate_user_hint_input.viewport().width()
        viewport_height = self.regenerate_user_hint_input.viewport().height()
        width = max(20, viewport_width - (x_pos * 2))
        height = max(20, viewport_height - y_pos)
        self.regenerate_user_hint_overlay_label.setGeometry(x_pos, y_pos, width, height)

    def _update_regenerate_user_hint_visibility(self) -> None:
        working = self._regenerate_thread is not None
        has_text = bool(self._regenerate_user_hint_text())
        self.regenerate_user_hint_overlay_label.setVisible((not working) and (not has_text))

    def _clear_regenerate_user_hint(self) -> None:
        if self.regenerate_user_hint_input.toPlainText():
            self.regenerate_user_hint_input.clear()
            self.regenerate_status_label.setText("User hint cleared.")

    def _clear_and_focus_hint(self) -> None:
        """Clear hint input and focus it for user to type a new hint."""
        self.regenerate_user_hint_input.clear()
        self.regenerate_user_hint_input.setFocus()
        self.regenerate_status_label.setText("User hint cleared. Type a new hint.")

    def _regenerate_proposed_annotations(self) -> None:
        if self._regenerate_thread is not None:
            return
        if self._image_path is None:
            return
        if not self.regenerate_tags_checkbox.isChecked() and not self.regenerate_description_checkbox.isChecked():
            QMessageBox.information(self, "Nothing selected", "Enable Tags or Description before regenerating.")
            return
        active_session = self._active_regenerate_session(show_errors=True)
        if active_session is None:
            return

        try:
            max_resolution_mpx = self._regenerate_max_resolution_mpx()
        except LlmProviderError:
            return
        configure_image_preparation(max_image_pixels=max(1, int(max_resolution_mpx * 1_000_000)))

        resize_warning = consume_image_preparation_warning()
        if resize_warning:
            QMessageBox.warning(self, "Image resize disabled", resize_warning)

        cancel_token = LlmRequestCancellation()
        self._regenerate_cancel = cancel_token
        self._discard_regenerate_result = False
        user_hint = self._regenerate_user_hint_text()
        description_prompt = (
            render_prompt_with_user_hint(active_prompt_for_kind("description"), user_hint)
            if self.regenerate_description_checkbox.isChecked()
            else None
        )
        tags_prompt = (
            render_prompt_with_user_hint(active_prompt_for_kind("tagging"), user_hint)
            if self.regenerate_tags_checkbox.isChecked()
            else None
        )

        debug_prompts = bool(_config.load().get("debug_regenerate_prompt_console", False))
        if debug_prompts:
            print("[merge-regenerate] final prompts begin", flush=True)
            print(f"[merge-regenerate] image={self._image_path.name}", flush=True)
            if description_prompt is not None:
                print("[merge-regenerate] description prompt:", flush=True)
                print(description_prompt, flush=True)
            if tags_prompt is not None:
                print("[merge-regenerate] tags prompt:", flush=True)
                print(tags_prompt, flush=True)
            print("[merge-regenerate] final prompts end", flush=True)

        def task(report_progress: Callable[[str], None]) -> object:
            timeout = self._regenerate_timeout_seconds()
            retry_count = self._regenerate_retry_count()
            image_name = self._image_path.name
            last_error: LlmProviderError | None = None
            has_user_hint = bool(user_hint)

            for attempt in range(retry_count + 1):
                cancel_token.raise_if_cancelled()
                attempt_start = time.monotonic()
                if attempt == 0:
                    if has_user_hint:
                        report_progress(f"Regenerating {image_name} with user hint...")
                    else:
                        report_progress(f"Regenerating {image_name}...")
                else:
                    if has_user_hint:
                        report_progress(
                            f"Regenerating {image_name} with user hint (retry {attempt}/{retry_count})..."
                        )
                    else:
                        report_progress(f"Regenerating {image_name} (retry {attempt}/{retry_count})...")

                def remaining_timeout() -> float:
                    elapsed = time.monotonic() - attempt_start
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise LlmProviderError(
                            f"Timed out after {int(timeout)} seconds while regenerating annotations for {image_name}."
                        )
                    return remaining

                try:
                    description = ""
                    tags: list[str] = []
                    if description_prompt is not None:
                        description = normalize_description_text(
                            active_session.generate(
                                self._image_path,
                                description_prompt,
                                timeout=remaining_timeout(),
                                cancellation=cancel_token,
                                thread_count=1,
                            ).strip()
                        )
                        # Remove any commas from regenerated description (critical for .txt file format)
                        description = remove_commas_from_description(description)
                    if tags_prompt is not None:
                        tags = self._dedupe_preserve_order(
                            self._parse_regenerated_tags(
                                active_session.generate(
                                    self._image_path,
                                    tags_prompt,
                                    timeout=remaining_timeout(),
                                    cancellation=cancel_token,
                                    thread_count=1,
                                )
                            )
                        )

                    if description or tags:
                        return {"description": description, "tags": tags}
                    last_error = LlmProviderError("Model returned no annotations.")
                except LlmProviderCancelled:
                    raise
                except LlmProviderError as exc:
                    last_error = exc

            if last_error is not None:
                raise last_error
            raise LlmProviderError("Model returned no annotations.")

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
        self._regenerate_started_at = time.monotonic()
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
        self._activate_first_comparison_row()
        elapsed_seconds: int | None = None
        if self._regenerate_started_at is not None:
            elapsed_seconds = max(1, int(time.monotonic() - self._regenerate_started_at))
        if description or tags:
            if elapsed_seconds is not None:
                self.regenerate_status_label.setText(
                    f"Regenerated proposed annotations in {elapsed_seconds} seconds."
                )
            else:
                self.regenerate_status_label.setText("Regenerated proposed annotations.")
        else:
            self.regenerate_status_label.setText("Model returned no annotations.")

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
        self._regenerate_started_at = None
        self._update_regenerate_controls()
        self._refresh_button_state()

    def _create_image_pane(self, image_path: Path | None) -> QWidget:
        pane = QWidget(self)
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        self._image_header_label = self._create_pane_header_label("Image")
        pane_layout.addWidget(self._image_header_label)
        pane_layout.addSpacing(self._PANE_HEADER_BOTTOM_SPACING)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        image_label = ScalableImageLabel(self)
        self._image_label = image_label
        image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        image_label.customContextMenuRequested.connect(self._show_image_context_menu)

        self._load_dialog_image_preview(image_path)

        scroll.setWidget(image_label)
        pane_layout.addWidget(scroll, stretch=1)
        pane_layout.addWidget(self._create_regenerate_panel(), stretch=0)
        pane_layout.addWidget(self.regenerate_button, stretch=0)
        pane_layout.addWidget(self.regenerate_status_label, stretch=0)
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

        menu.addSeparator()
        delete_action = menu.addAction("Delete file")
        delete_action.triggered.connect(self._delete_file_from_context_menu)

        source_widget = self.sender()
        if isinstance(source_widget, QWidget):
            global_position = source_widget.mapToGlobal(position)
        else:
            global_position = self.mapToGlobal(position)
        menu.exec(global_position)

    def _delete_file_from_context_menu(self) -> None:
        image_path = self._image_path
        if image_path is None:
            return

        if self._delete_image is None:
            QMessageBox.warning(self, "Delete failed", "Delete handler is not available.")
            return

        if self._confirm_delete:
            answer = QMessageBox.question(
                self,
                "Delete file",
                (
                    f"Delete this image and related files?\n\n"
                    f"Image: {image_path.name}\n"
                    "Also deletes matching .txt and .fixup files"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        deleted, has_fixups_remaining = self._delete_image()
        if not deleted:
            return

        if has_fixups_remaining:
            self._navigate_next()
            return

        self._enter_no_fixups_state_after_delete()

    def _enter_no_fixups_state_after_delete(self) -> None:
        self._cancel_regenerate(discard_result=True)
        self._set_watched_image(None)
        self._image_path = None

        self.left_list.clear()
        self.right_list.clear()
        self._table_row_map = []
        self.comparison_table.clearContents()
        self.comparison_table.setRowCount(0)
        self.comparison_table.setEnabled(False)
        self.left_tag_input.clear()
        self.left_tag_input.setEnabled(False)

        if self._image_label is not None:
            self._image_label.clear_original_image("No fixup files remaining")
            self._image_label.setEnabled(False)

        self._set_image_header_text(None, None)
        self.issues_label.setText("No fixup files remaining.")
        self.regenerate_status_label.setText("Current file was deleted. Press Esc to close.")

        for widget in (
            self.regenerate_tags_checkbox,
            self.regenerate_description_checkbox,
            self.regenerate_timeout_input,
            self.regenerate_retry_input,
            self.regenerate_max_resolution_input,
            self.llm_endpoint_input,
            self.llm_fetch_button,
            self.llm_model_combo,
            self.llm_use_button,
            self.regenerate_user_hint_input,
            self.regenerate_user_hint_clear_button,
            self.regenerate_button,
            self.accept_button,
            self.merge_button,
            self.undo_button,
            self.merge_next_button,
            self.prev_button,
            self.next_button,
        ):
            widget.setEnabled(False)

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

        self._set_watched_image(image_path)
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

        self._set_watched_image(image_path)
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

    def _fetch_provider_models(self) -> None:
        """Fetch available models from the configured server endpoint."""
        if self._llm_provider is None:
            QMessageBox.warning(self, "No provider", "No LLM provider is configured.")
            return

        server = self.llm_endpoint_input.text().strip()
        if not server:
            QMessageBox.warning(self, "Server required", "Enter a server endpoint before fetching models.")
            return

        self.llm_fetch_button.setEnabled(False)
        try:
            model_names = self._llm_provider.fetch_models(server, timeout=10.0)
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
        if model_names:
            self.regenerate_status_label.setText(f"Fetched {len(model_names)} model(s)")
        else:
            self.regenerate_status_label.setText(f"No models found at {normalized_server}")
            QMessageBox.information(self, "No models found", f"The {self._llm_provider.display_name} server returned no models.")
        self._update_regenerate_controls()

    def _use_selected_provider_model(self) -> None:
        """Use the selected model for regenerate operations in this dialog."""
        if self._llm_provider is None:
            QMessageBox.warning(self, "No provider", "No LLM provider is configured.")
            return

        model_name = self.llm_model_combo.currentText().strip()
        if not model_name:
            QMessageBox.warning(self, "No model selected", "Fetch models and choose one before using it.")
            return

        try:
            normalized_server = self._llm_provider.normalize_endpoint(self.llm_endpoint_input.text())
        except LlmProviderError as exc:
            QMessageBox.warning(self, "Invalid server", str(exc))
            return

        self._llm_endpoint = normalized_server
        self._llm_model_name = model_name
        self.llm_endpoint_input.setText(normalized_server)
        self.regenerate_status_label.setText(f"Model selected: {self._llm_model_name}")
        self._update_regenerate_controls()

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
        
        # Restore persistent endpoint and model from previous dialog
        if self._llm_endpoint.strip():
            self.llm_endpoint_input.setText(self._llm_endpoint)
        
        # Restore model selection - add to combo if not present, then set as current
        # AND update the instance variable so it persists even if user doesn't click "Use"
        if self._llm_model_name.strip():
            current_count = self.llm_model_combo.count()
            # Add model to combo if it's not already there
            if self.llm_model_combo.findText(self._llm_model_name) == -1:
                self.llm_model_combo.addItem(self._llm_model_name)
            # Set as current item
            index = self.llm_model_combo.findText(self._llm_model_name)
            if index >= 0:
                self.llm_model_combo.setCurrentIndex(index)
            self.regenerate_status_label.setText(f"Model selected: {self._llm_model_name}")
        
        # Restore persistent hint text from previous dialog
        if self._regenerate_user_hint_value.strip():
            self.regenerate_user_hint_input.setPlainText(self._regenerate_user_hint_value)
        
        self._position_regenerate_user_hint_overlay()
        self._update_regenerate_user_hint_visibility()
        QTimer.singleShot(0, self._apply_initial_table_focus)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.comparison_table.resizeRowsToContents()
        self._refresh_widget_item_sizes()
        self._position_regenerate_user_hint_overlay()

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
        self._set_watched_image(None)
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
        # Merge+Next must persist the Current column exactly as shown.
        # Proposed rows are only applied when explicitly accepted by the user.
        if self._merge_without_close() and self.next_button.isEnabled():
            self._navigate_next()