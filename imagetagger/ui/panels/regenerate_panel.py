from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imagetagger import config as _config
from imagetagger.utils.annotations import sanitize_description_text
from imagetagger.utils.image_prep import (
    configure_image_preparation,
    consume_image_preparation_warning,
    is_pillow_image_resize_available,
)
from imagetagger.utils.input_validators import InputValidator
from imagetagger.utils.validators import (
    create_max_resolution_validator,
    create_retry_validator,
    create_timeout_validator,
)
from imagetagger.utils.llm_queries import active_prompt_for_kind, render_prompt_with_agent_role, render_prompt_with_existing_tags, render_prompt_with_user_hint
from imagetagger.providers.llm_provider import (
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
    VisionLlmProvider,
    VisionLlmSession,
)
from imagetagger.ui.workers import RegenerateWorker
from imagetagger.ui.server_settings_frame import create_server_settings_frame
from imagetagger.ui.shortcuts import native_shortcut_text, platform_key_sequence


class RegeneratePanel(QWidget):
    """Self-contained regeneration controls extracted from FixupDialog.

    Signals
    -------
    proposed_annotations_ready(description, tags, exact_match_only_for_tags)
        Emitted after a successful regeneration run.  The dialog connects this
        to ``_set_proposed_annotations`` and calls ``_activate_first_comparison_row``.
    regeneration_started()
        Emitted when a regeneration worker thread is launched.
    regeneration_finished()
        Emitted when the worker thread cleans up (success, failure, or cancel).
        The dialog connects this to ``_refresh_button_state``.
    model_selection_changed(endpoint, model_name)
        Emitted when the user explicitly clicks "Use" to adopt a fetched model.
    """

    proposed_annotations_ready = pyqtSignal(str, list, bool)
    regeneration_started = pyqtSignal()
    regeneration_finished = pyqtSignal()
    model_selection_changed = pyqtSignal(str, str)

    def __init__(
        self,
        *,
        provider: VisionLlmProvider | None,
        provider_session: VisionLlmSession | None,
        image_path_getter: Callable[[], Path | None],
        get_current_proposed: Callable[[], tuple[str, list[str], bool]],
        normalize_tag: Callable[[str], str],
        normalize_annotation: Callable[[str], str],
        existing_tags_getter: Callable[[], list[str] | None] | None = None,
        regenerate_tags_enabled: bool = True,
        regenerate_description_enabled: bool = True,
        regenerate_timeout_seconds: int = 300,
        regenerate_retry_count: int = 3,
        regenerate_max_resolution_mpx: float = 5.0,
        regenerate_model_name: str = "",
        regenerate_model_endpoint: str = "",
        regenerate_user_hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._image_path_getter = image_path_getter
        self._get_current_proposed = get_current_proposed
        self._normalize_tag = normalize_tag
        self._normalize_annotation = normalize_annotation
        self._existing_tags_getter = existing_tags_getter
        self._llm_provider = provider
        self._provider_session = provider_session
        self._llm_endpoint = regenerate_model_endpoint or ""
        self._llm_model_name = regenerate_model_name or ""

        self._regenerate_thread: QThread | None = None
        self._regenerate_worker: RegenerateWorker | None = None
        self._regenerate_cancel: LlmRequestCancellation | None = None
        self._regenerate_started_at: float | None = None
        self._discard_regenerate_result = False

        # ── Widgets ────────────────────────────────────────────────────────
        self.regenerate_tags_checkbox = QCheckBox("Tags", self)
        self.regenerate_tags_checkbox.setChecked(regenerate_tags_enabled)
        self.regenerate_tags_checkbox.checkStateChanged.connect(
            lambda _state: self._update_regenerate_controls()
        )

        self.regenerate_description_checkbox = QCheckBox("Description", self)
        self.regenerate_description_checkbox.setChecked(regenerate_description_enabled)
        self.regenerate_description_checkbox.checkStateChanged.connect(
            lambda _state: self._update_regenerate_controls()
        )

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

        self.llm_endpoint_input = QLineEdit(self)
        self.llm_endpoint_input.setPlaceholderText(
            "http://127.0.0.1:11434 (Ollama) or :8000 (OpenAI-compatible)"
        )
        if self._llm_provider is not None:
            self.llm_endpoint_input.setText(self._llm_provider.default_endpoint)

        self.llm_fetch_button = QPushButton("Fetch models", self)
        self.llm_fetch_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.llm_fetch_button.clicked.connect(self._fetch_provider_models)

        self.llm_model_combo = QComboBox(self)
        self.llm_model_combo.setEditable(False)

        self.llm_use_button = QPushButton("Use", self)
        self.llm_use_button.clicked.connect(self._use_selected_provider_model)

        _clear_hint_shortcut = platform_key_sequence("Alt+H", "Alt+H")
        self._clear_hint_shortcut_hint = native_shortcut_text(_clear_hint_shortcut)

        self.regenerate_user_hint_input = QTextEdit(self)
        self.regenerate_user_hint_input.setAcceptRichText(False)
        self._regenerate_user_hint_placeholder = (
            "User hint (optional). Example: The cat is not a Maine Coon."
        )
        self.regenerate_user_hint_input.setPlaceholderText("")
        self.regenerate_user_hint_input.setToolTip(
            "Optional guidance used only for this regenerate run."
        )
        _user_hint_height = (
            self.regenerate_user_hint_input.fontMetrics().lineSpacing() * 2
        ) + 14
        self.regenerate_user_hint_input.setFixedHeight(_user_hint_height)
        self.regenerate_user_hint_input.textChanged.connect(
            self._update_regenerate_user_hint_visibility
        )

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

        _regenerate_shortcut = platform_key_sequence("Alt+R", "Alt+R")
        self._regenerate_shortcut_hint = native_shortcut_text(_regenerate_shortcut)

        self.regenerate_button = QPushButton("Regenerate", self)
        self.regenerate_button.setAutoDefault(False)
        self.regenerate_button.setDefault(False)
        self.regenerate_button.setToolTip(
            f"Regenerate proposed annotations with the selected model"
            f" ({self._regenerate_shortcut_hint})"
        )
        self.regenerate_button.clicked.connect(self._regenerate_proposed_annotations)

        self.regenerate_alt_action = QAction("Regenerate", self)
        self.regenerate_alt_action.setShortcut(_regenerate_shortcut)
        self.regenerate_alt_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.regenerate_alt_action.triggered.connect(self._regenerate_proposed_annotations)
        self.addAction(self.regenerate_alt_action)

        self.clear_hint_action = QAction("Clear User Hint", self)
        self.clear_hint_action.setShortcut(_clear_hint_shortcut)
        self.clear_hint_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.clear_hint_action.triggered.connect(self._clear_and_focus_hint)
        self.addAction(self.clear_hint_action)

        self.regenerate_status_label = QLabel(self)
        self.regenerate_status_label.setWordWrap(True)
        self.regenerate_status_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.regenerate_status_label.setMinimumWidth(0)
        self.regenerate_status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        # ── Layout ─────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._create_settings_frame(), stretch=0)

        user_hint_row = QHBoxLayout()
        user_hint_row.setContentsMargins(0, 0, 0, 0)
        user_hint_row.setSpacing(6)
        user_hint_row.addWidget(self.regenerate_user_hint_input, stretch=1)
        user_hint_row.addWidget(self.regenerate_user_hint_clear_button, stretch=0)
        layout.addLayout(user_hint_row)

        layout.addWidget(self.regenerate_button, stretch=0)
        layout.addWidget(self.regenerate_status_label, stretch=0)

        # ── Initial state ───────────────────────────────────────────────────
        self._initialize_provider_controls()

        # Apply constructor-provided model name / endpoint / hint that may not
        # come from a live provider_session (e.g. persisted across dialogs).
        if self._llm_endpoint.strip():
            self.llm_endpoint_input.setText(self._llm_endpoint)
        if self._llm_model_name.strip():
            if self.llm_model_combo.findText(self._llm_model_name) == -1:
                self.llm_model_combo.addItem(self._llm_model_name)
            idx = self.llm_model_combo.findText(self._llm_model_name)
            if idx >= 0:
                self.llm_model_combo.setCurrentIndex(idx)
        if regenerate_user_hint:
            self.regenerate_user_hint_input.setPlainText(regenerate_user_hint)

        self._update_regenerate_controls()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def is_regenerating(self) -> bool:
        """True while a background regeneration worker is running."""
        return self._regenerate_thread is not None

    @property
    def current_endpoint(self) -> str:
        return self._llm_endpoint

    @property
    def current_model_name(self) -> str:
        return self._llm_model_name

    @property
    def current_user_hint(self) -> str:
        return self.regenerate_user_hint_input.toPlainText().strip()

    def cancel_regeneration(self, *, discard_result: bool = True) -> None:
        """Cancel any in-progress regeneration worker."""
        if discard_result:
            self._discard_regenerate_result = True
        if self._regenerate_cancel is not None:
            self._regenerate_cancel.cancel()

    def reposition_overlay(self) -> None:
        """Reposition the placeholder overlay and refresh hint visibility.

        Must be called from ``showEvent`` and ``resizeEvent`` of the parent
        dialog because the viewport size is only accurate after layout.
        """
        self._position_regenerate_user_hint_overlay()
        self._update_regenerate_user_hint_visibility()

    def set_status(self, text: str) -> None:
        """Set the status label text (used by the dialog for image-reload etc.)."""
        self.regenerate_status_label.setText(text)

    def set_all_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable all regenerate controls as a group."""
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
        ):
            widget.setEnabled(enabled)

    # ── Private layout helpers ──────────────────────────────────────────────

    def _create_settings_frame(self) -> QWidget:
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

    # ── Provider helpers (moved from FixupDialog) ───────────────────────────

    def _initialize_provider_controls(self) -> None:
        if self._provider_session is None:
            return

        session_endpoint = str(getattr(self._provider_session, "endpoint", "")).strip()
        session_model = str(getattr(self._provider_session, "model_name", "")).strip()

        if session_endpoint and not self._llm_endpoint.strip():
            self._llm_endpoint = session_endpoint
            self.llm_endpoint_input.setText(session_endpoint)

        if session_model and not self._llm_model_name.strip():
            self._llm_model_name = session_model
            if self.llm_model_combo.findText(session_model) == -1:
                self.llm_model_combo.addItem(session_model)
            self.llm_model_combo.setCurrentText(session_model)

    def _active_regenerate_session(self, *, show_errors: bool) -> VisionLlmSession | None:
        if self._llm_provider is not None and self._llm_model_name.strip():
            # Prefer the text field (what the user sees and can edit) over the
            # stored _llm_endpoint so that typing a different server without
            # clicking "Use" is still honoured for regeneration.
            endpoint = self.llm_endpoint_input.text().strip() or self._llm_endpoint.strip()
            if not endpoint:
                if show_errors:
                    QMessageBox.warning(
                        self, "Server required", "Enter a server endpoint before regenerating."
                    )
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

    def _fetch_provider_models(self) -> None:
        if self._llm_provider is None:
            QMessageBox.warning(self, "No provider", "No LLM provider is configured.")
            return

        server = self.llm_endpoint_input.text().strip()
        if not server:
            QMessageBox.warning(
                self, "Server required", "Enter a server endpoint before fetching models."
            )
            return

        self.llm_fetch_button.setEnabled(False)
        try:
            model_names = self._llm_provider.fetch_models(server, timeout=10.0)
            normalized_server = self._llm_provider.normalize_endpoint(server)
        except LlmProviderError as exc:
            QMessageBox.warning(
                self, f"{self._llm_provider.display_name} connection failed", str(exc)
            )
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
            QMessageBox.information(
                self,
                "No models found",
                f"The {self._llm_provider.display_name} server returned no models.",
            )
        self._update_regenerate_controls()

    def _use_selected_provider_model(self) -> None:
        if self._llm_provider is None:
            QMessageBox.warning(self, "No provider", "No LLM provider is configured.")
            return

        model_name = self.llm_model_combo.currentText().strip()
        if not model_name:
            QMessageBox.warning(
                self, "No model selected", "Fetch models and choose one before using it."
            )
            return

        try:
            normalized_server = self._llm_provider.normalize_endpoint(
                self.llm_endpoint_input.text()
            )
        except LlmProviderError as exc:
            QMessageBox.warning(self, "Invalid server", str(exc))
            return

        self._llm_endpoint = normalized_server
        self._llm_model_name = model_name
        self.llm_endpoint_input.setText(normalized_server)
        status_parts = [f"Model selected: {self._llm_model_name}"]
        if not is_pillow_image_resize_available():
            status_parts.append(
                "Warning: Pillow is not installed — query downscaling is disabled and "
                "full-resolution images are sent to the model. Install Pillow to enable downscaling."
            )
        self.regenerate_status_label.setText("\n\n".join(status_parts))
        self.model_selection_changed.emit(self._llm_endpoint, self._llm_model_name)
        self._update_regenerate_controls()

    # ── Controls state ──────────────────────────────────────────────────────

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
        self.llm_use_button.setEnabled(
            not working
            and self.llm_model_combo.count() > 0
            and self._llm_provider is not None
        )
        self.regenerate_user_hint_input.setEnabled(not working)
        self.regenerate_user_hint_clear_button.setEnabled(not working)
        self._update_regenerate_user_hint_visibility()

        if working:
            self.regenerate_button.setEnabled(False)
            self.regenerate_button.setText("Regenerating...")
            self.regenerate_alt_action.setEnabled(False)
            return

        image_path = self._image_path_getter()
        regenerate_enabled = connected and options_selected and image_path is not None
        self.regenerate_button.setText("Regenerate")
        self.regenerate_button.setEnabled(regenerate_enabled)
        self.regenerate_alt_action.setEnabled(regenerate_enabled)
        if not connected:
            self.regenerate_status_label.setText(
                "Regenerate is disabled until a model is selected in this dialog"
                " or in the main window."
            )

    # ── User-hint overlay ───────────────────────────────────────────────────

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
        self.regenerate_user_hint_input.clear()
        self.regenerate_user_hint_input.setFocus()
        self.regenerate_status_label.setText("User hint cleared. Type a new hint.")

    # ── Validation helpers ──────────────────────────────────────────────────

    def _regenerate_timeout_seconds(self) -> float:
        def show_error(msg: str) -> None:
            QMessageBox.warning(self, "Invalid timeout", msg)

        return InputValidator.parse_timeout_seconds(
            self.regenerate_timeout_input.text(), show_error
        )

    def _regenerate_retry_count(self) -> int:
        return InputValidator.parse_retry_count(self.regenerate_retry_input.text())

    @staticmethod
    def _format_mpx(value: float) -> str:
        return InputValidator.format_megapixels(value)

    def _regenerate_max_resolution_mpx(self) -> float:
        def show_error(msg: str) -> None:
            QMessageBox.warning(self, "Invalid query downscale", msg)

        return InputValidator.parse_max_resolution_mpx(
            self.regenerate_max_resolution_input.text(), show_error
        )

    # ── Tag parsing helpers ─────────────────────────────────────────────────

    def _normalized_compare_key(self, text: str) -> str:
        return self._normalize_annotation(text).strip().casefold()

    def _parse_regenerated_tags(self, text: str) -> list[str]:
        normalized = text.replace("\r", "").replace("\n", ",")
        tags: list[str] = []
        for part in normalized.split(","):
            cleaned = self._normalize_tag(part)
            if cleaned:
                tags.append(cleaned)
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

    # ── Regeneration worker ─────────────────────────────────────────────────

    def _regenerate_proposed_annotations(self) -> None:
        if self._regenerate_thread is not None:
            return
        image_path = self._image_path_getter()
        if image_path is None:
            return
        if (
            not self.regenerate_tags_checkbox.isChecked()
            and not self.regenerate_description_checkbox.isChecked()
        ):
            QMessageBox.information(
                self, "Nothing selected", "Enable Tags or Description before regenerating."
            )
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
        existing_tags = self._existing_tags_getter() if self._existing_tags_getter is not None else None
        agent_roles = _config.load().get("agent_roles") or {}
        description_prompt = (
            render_prompt_with_user_hint(
                render_prompt_with_existing_tags(
                    render_prompt_with_agent_role(active_prompt_for_kind("description"), agent_roles.get("description")),
                    existing_tags,
                ),
                user_hint,
            )
            if self.regenerate_description_checkbox.isChecked()
            else None
        )
        tags_prompt = (
            render_prompt_with_user_hint(
                render_prompt_with_existing_tags(
                    render_prompt_with_agent_role(active_prompt_for_kind("tagging"), agent_roles.get("tagging")),
                    existing_tags,
                ),
                user_hint,
            )
            if self.regenerate_tags_checkbox.isChecked()
            else None
        )

        debug_prompts = bool(_config.load().get("debug_regenerate_prompt_console", False))
        if debug_prompts:
            print("[merge-regenerate] final prompts begin", flush=True)
            print(f"[merge-regenerate] image={image_path.name}", flush=True)
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
            image_name = image_path.name
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
                            f"Regenerating {image_name} with user hint"
                            f" (retry {attempt}/{retry_count})..."
                        )
                    else:
                        report_progress(
                            f"Regenerating {image_name} (retry {attempt}/{retry_count})..."
                        )

                def remaining_timeout() -> float:
                    elapsed = time.monotonic() - attempt_start
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise LlmProviderError(
                            f"Timed out after {int(timeout)} seconds while regenerating"
                            f" annotations for {image_name}."
                        )
                    return remaining

                try:
                    description = ""
                    tags: list[str] = []
                    if description_prompt is not None:
                        description = sanitize_description_text(
                            active_session.generate(
                                image_path,
                                description_prompt,
                                timeout=remaining_timeout(),
                                cancellation=cancel_token,
                                thread_count=1,
                            ).strip()
                        )
                    if tags_prompt is not None:
                        tags = self._dedupe_preserve_order(
                            self._parse_regenerated_tags(
                                active_session.generate(
                                    image_path,
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
        self.regeneration_started.emit()
        self._update_regenerate_controls()
        self._regenerate_thread.start()

    def _on_regenerate_finished(self, payload: object) -> None:
        if self._discard_regenerate_result:
            self.regenerate_status_label.setText("Regeneration discarded.")
            return

        data = payload if isinstance(payload, dict) else {}
        description = str(data.get("description", "")).strip()
        raw_tags = data.get("tags")
        tags = [str(tag).strip() for tag in raw_tags] if isinstance(raw_tags, list) else []
        tags = [tag for tag in tags if tag]

        current_description, current_tags, exact_match_only_for_tags = self._get_current_proposed()

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
        exact_only = True if regenerate_tags else exact_match_only_for_tags

        self.proposed_annotations_ready.emit(final_description, final_tags, exact_only)

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
        self.regeneration_finished.emit()
