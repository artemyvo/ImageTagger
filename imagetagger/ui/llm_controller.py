from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
from typing import TYPE_CHECKING, Callable

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QMessageBox

from imagetagger.utils.annotations import sanitize_annotation_text, sanitize_tag_text
from imagetagger.utils.image_prep import configure_image_preparation, consume_image_preparation_warning
from imagetagger.utils.input_validators import InputValidator
from imagetagger.utils.sidecar import read_sidecar_data, write_sidecar_data
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
from imagetagger.ui.merge_actions import (
    clear_validation_fields_sidecar,
    record_ai_find_match_for_image,
    record_refine_result_for_image,
    write_fixup_sidecar,
)
from imagetagger.providers.llm_provider import (
    LlmProviderCancelled,
    LlmProviderError,
    LlmRequestCancellation,
    VisionLlmSession,
)
from imagetagger.ui.workers import LlmTaskWorker, RegenerateWorker

if TYPE_CHECKING:
    from imagetagger.ui.main_window import MainWindow


class LlmController:
    """Owns all LLM orchestration logic extracted from MainWindow (step 4.2).

    All UI widget access and shared-state access goes through ``self._window``.
    """

    def __init__(self, window: "MainWindow") -> None:
        self._window = window

        # State fields moved off MainWindow
        self._llm_thread: QThread | None = None
        self._llm_worker: LlmTaskWorker | None = None
        self._llm_cancel: LlmRequestCancellation | None = None
        self._llm_action_name: str | None = None
        self._llm_threads_auto_mode: bool = False
        self._llm_threads_current: int = 0

        self._generate_batch_total: int = 0
        self._generate_batch_processed: int = 0
        self._generate_batch_updated: int = 0
        self._generate_batch_vision_updated: int = 0
        self._generate_batch_new_annotations: int = 0
        self._generate_batch_started_at: float | None = None
        self._generate_batch_retry_images: int = 0
        self._generate_batch_refine_updated: int = 0

        self._validate_batch_total: int = 0
        self._validate_batch_processed: int = 0
        self._validate_batch_clean: int = 0
        self._validate_batch_issues: int = 0
        self._validate_batch_skipped: int = 0
        self._validate_batch_started_at: float | None = None
        self._validate_batch_retry_images: int = 0
        self._validate_batch_llm_disobeyed: int = 0
        self._validate_batch_errors: int = 0
        self._validate_batch_context_exhausted: int = 0
        self._validate_pending_indices: set[int] = set()

        self._ai_find_batch_total: int = 0
        self._ai_find_batch_processed: int = 0
        self._ai_find_batch_matched: int = 0
        self._ai_find_batch_started_at: float | None = None
        self._ai_find_batch_retry_images: int = 0

    # ------------------------------------------------------------------
    # Public read-only properties used by MainWindow
    # ------------------------------------------------------------------

    @property
    def llm_thread(self) -> QThread | None:
        return self._llm_thread

    @property
    def llm_action_name(self) -> str | None:
        return self._llm_action_name

    @property
    def llm_cancel(self) -> LlmRequestCancellation | None:
        return self._llm_cancel

    @property
    def validate_pending_indices(self) -> set[int]:
        return self._validate_pending_indices

    # ------------------------------------------------------------------
    # Prompt helpers (kept here because they are LLM-only)
    # ------------------------------------------------------------------

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
        from imagetagger.providers.llm_provider import LlmProviderError
        editor = self._window.prompt_editors.get(kind)
        if editor is None:
            raise LlmProviderError(f"Prompt editor for {kind} is not available.")
        return editor.toPlainText().strip()

    def _update_prompt_status(self, kind: str, edited: bool = False) -> None:
        w = self._window
        label = w.prompt_status_labels.get(kind)
        editor = w.prompt_editors.get(kind)
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

    def _set_agent_role(self, kind: str) -> None:
        from imagetagger import config as _config
        w = self._window
        role_input = w.prompt_role_inputs.get(kind)
        role = role_input.text().strip() if role_input is not None else ""
        if "agent_roles" not in w._cfg or not isinstance(w._cfg["agent_roles"], dict):
            w._cfg["agent_roles"] = {}
        if role:
            w._cfg["agent_roles"][kind] = role
        else:
            w._cfg["agent_roles"].pop(kind, None)
        _config.save(w._cfg)
        w.statusBar().showMessage(
            f"Role for {self._prompt_title(kind)} {'set' if role else 'cleared'}"
        )

    def _apply_prompt_override(self, kind: str) -> None:
        w = self._window
        try:
            set_prompt_override(kind, self._prompt_editor_text(kind))
        except LlmProviderError as exc:
            QMessageBox.critical(w, "Apply prompt failed", str(exc))
            return
        self._update_prompt_status(kind)
        w.statusBar().showMessage(f"Applied {self._prompt_title(kind)} prompt in memory")

    def _save_prompt_to_file(self, kind: str) -> None:
        w = self._window
        try:
            saved_text = save_prompt_for_kind(kind, self._prompt_editor_text(kind))
        except LlmProviderError as exc:
            QMessageBox.critical(w, "Save prompt failed", str(exc))
            return

        editor = w.prompt_editors.get(kind)
        if editor is not None:
            editor.setPlainText(saved_text)
        self._update_prompt_status(kind)
        w.statusBar().showMessage(f"Saved {self._prompt_title(kind)} prompt file")

    def _reset_prompt_to_default(self, kind: str) -> None:
        w = self._window
        try:
            default_text = reset_prompt_to_default(kind)
            clear_prompt_override(kind)
        except LlmProviderError as exc:
            QMessageBox.critical(w, "Reset prompt failed", str(exc))
            return

        editor = w.prompt_editors.get(kind)
        if editor is not None:
            editor.setPlainText(default_text)
        self._update_prompt_status(kind)
        w.statusBar().showMessage(
            f"Reset {self._prompt_title(kind)} prompt to code default"
        )

    # ------------------------------------------------------------------
    # Provider / connection helpers
    # ------------------------------------------------------------------

    def fetch_provider_models(self) -> None:
        w = self._window
        server = w.llm_endpoint_input.text().strip()
        w.llm_fetch_button.setEnabled(False)
        try:
            model_names = w._llm_provider.fetch_models(server, timeout=self._llm_timeout_seconds())
            normalized_server = w._llm_provider.normalize_endpoint(server)
        except LlmProviderError as exc:
            QMessageBox.warning(w, f"{w._llm_provider.display_name} connection failed", str(exc))
            return
        finally:
            w.llm_fetch_button.setEnabled(True)

        w.llm_endpoint_input.setText(normalized_server)
        w.llm_model_combo.clear()
        w.llm_model_combo.addItems(model_names)
        if model_names:
            w.llm_model_combo.setCurrentIndex(0)
            w.statusBar().showMessage(f"Fetched {len(model_names)} model(s) from {normalized_server}")
        else:
            w.statusBar().showMessage(f"No models found at {normalized_server}")
            QMessageBox.information(w, "No models found", f"The {w._llm_provider.display_name} server returned no models.")

    def use_selected_provider_model(self) -> None:
        w = self._window
        model_name = w.llm_model_combo.currentText().strip()
        if not model_name:
            QMessageBox.warning(w, "No model selected", "Fetch models and choose one before using it.")
            return

        try:
            normalized_server = w._llm_provider.normalize_endpoint(w.llm_endpoint_input.text())
        except LlmProviderError as exc:
            QMessageBox.warning(w, "Invalid server", str(exc))
            return

        w.llm_endpoint = normalized_server
        w.llm_model_name = model_name
        w.llm_endpoint_input.setText(normalized_server)
        w._cfg["llm_endpoint"] = normalized_server
        w._cfg["llm_model"] = model_name
        try:
            w._cfg["llm_threads"] = self._llm_thread_count(show_message=False)
        except LlmProviderError:
            w._cfg["llm_threads"] = 1
        from imagetagger import config as _config
        _config.save(w._cfg)
        self._update_llm_controls()
        w.statusBar().showMessage(f"{w._llm_provider.display_name} model selected: {w.llm_model_name}")

    def _active_provider_session(self) -> VisionLlmSession | None:
        w = self._window
        if not w.llm_model_name.strip():
            return None
        return w._llm_provider.create_session(w.llm_endpoint, w.llm_model_name)

    def _llm_timeout_seconds(self) -> float:
        def show_error(msg: str) -> None:
            QMessageBox.warning(self._window, "Invalid timeout", msg)
        return InputValidator.parse_timeout_seconds(self._window.llm_timeout_input.text(), show_error)

    def _llm_retry_count(self) -> int:
        return InputValidator.parse_retry_count(self._window.llm_retry_input.text())

    def _llm_max_resolution_mpx_value(self, show_message: bool = True) -> float:
        def show_error(msg: str) -> None:
            if show_message:
                QMessageBox.warning(self._window, "Invalid query downscale", msg)
        return InputValidator.parse_max_resolution_mpx(self._window.llm_max_resolution_input.text(), show_error)

    def _apply_query_downscale_setting(self) -> float:
        max_resolution_mpx = self._llm_max_resolution_mpx_value()
        max_pixels = max(1, int(max_resolution_mpx * 1_000_000))
        configure_image_preparation(max_image_pixels=max_pixels)
        return max_resolution_mpx

    def _llm_thread_count(self, show_message: bool = True) -> int:
        w = self._window
        raw_value = w.llm_threads_input.text().strip()
        if not raw_value:
            if show_message:
                QMessageBox.warning(w, "Invalid threads", "Enter thread count (0 for auto).")
            raise LlmProviderError("Enter thread count.")
        try:
            thread_count = int(raw_value)
        except ValueError as exc:
            if show_message:
                QMessageBox.warning(w, "Invalid threads", "Thread count must be a whole number.")
            raise LlmProviderError("Thread count must be a whole number.") from exc
        if thread_count < 0:
            if show_message:
                QMessageBox.warning(w, "Invalid threads", "Thread count must be 0 or greater.")
            raise LlmProviderError("Thread count must be 0 or greater.")
        return thread_count

    def _llm_threads_status_suffix(self) -> str:
        if self._llm_action_name is None:
            return ""
        current = max(1, int(self._llm_threads_current))
        if self._llm_threads_auto_mode:
            return f" | threads: {current} auto"
        return f" | threads: {current}"

    # ------------------------------------------------------------------
    # UI state
    # ------------------------------------------------------------------

    def _update_llm_controls(self) -> None:
        w = self._window
        connected = bool(w.llm_model_name.strip())
        if connected:
            text = f"{w.llm_model_name} @ {w.llm_endpoint}"
        else:
            text = "no model"
        if w.status_connection_label is not None:
            w.status_connection_label.setText(text)

        if self._llm_thread is not None and self._llm_action_name == "Generate":
            w.generate_button.setText("&Stop generation")
            w.generate_button.setEnabled(True)
            w.validate_button.setText("&Validate")
            w.validate_button.setEnabled(False)
            w.ai_find_button.setText("AI Find")
            w.ai_find_button.setEnabled(False)
            w.ai_find_input.setEnabled(False)
            w._update_fixup_button_state()
            return
        if self._llm_thread is not None and self._llm_action_name == "Validate":
            w.generate_button.setText("&Generate")
            w.generate_button.setEnabled(False)
            w.validate_button.setText("&Stop validation")
            w.validate_button.setEnabled(True)
            w.ai_find_button.setText("AI Find")
            w.ai_find_button.setEnabled(False)
            w.ai_find_input.setEnabled(False)
            w._update_fixup_button_state()
            return
        if self._llm_thread is not None and self._llm_action_name == "AI Find":
            w.generate_button.setText("&Generate")
            w.generate_button.setEnabled(False)
            w.validate_button.setText("&Validate")
            w.validate_button.setEnabled(False)
            w.ai_find_button.setText("Stop AI Find")
            w.ai_find_button.setEnabled(True)
            w.ai_find_input.setEnabled(False)
            w._update_fixup_button_state()
            return

        w.generate_button.setText("&Generate")
        w.validate_button.setText("&Validate")
        w.ai_find_button.setText("AI Find")
        active = connected and self._llm_thread is None
        w.generate_button.setEnabled(
            active
            and (
                w.generate_tags_checkbox.isChecked()
                or w.generate_description_checkbox.isChecked()
                or w.generate_vision_checkbox.isChecked()
                or w.generate_refine_checkbox.isChecked()
            )
        )
        w.validate_button.setEnabled(active)
        w.ai_find_input.setEnabled(active)
        w.ai_find_button.setEnabled(active and bool(w.ai_find_input.text().strip()))
        w._update_fixup_button_state()

    # ------------------------------------------------------------------
    # Test prompt
    # ------------------------------------------------------------------

    def _test_prompt(self, kind: str) -> None:
        w = self._window
        if self._llm_thread is not None:
            QMessageBox.information(w, "LLM busy", "Wait for the current LLM task to finish before testing a prompt.")
            return

        session = self._active_provider_session()
        if session is None:
            QMessageBox.warning(w, "No model selected", f"Connect to {w._llm_provider.display_name} and choose a model first.")
            return

        if w.current_index < 0 or w.current_index >= len(w.records):
            QMessageBox.information(w, "No image selected", "Select an image before testing the prompt.")
            return

        record = w.records[w.current_index]

        try:
            editor_text = self._prompt_editor_text(kind)
        except LlmProviderError as exc:
            QMessageBox.warning(w, "Prompt error", str(exc))
            return

        agent_role = w._cfg.get("agent_roles", {}).get(kind) or None

        if kind == "vision":
            tags_lines = w._parse_annotations_for_tag_list(record.text)
            tags_text = "\n".join(tags_lines).strip()
            prompt = editor_text.replace("{tags}", tags_text)
            prompt = render_prompt_with_agent_role(prompt, agent_role)
            prompt = render_prompt_with_user_hint(prompt)
        elif kind == "validation":
            prompt = editor_text.replace("{tags}", format_annotations_for_validation(record.text))
            prompt = render_prompt_with_agent_role(prompt, agent_role)
            prompt = render_prompt_with_user_hint(prompt)
        elif kind == "search":
            query = " ".join(w.ai_find_input.text().split())
            if not query:
                QMessageBox.information(w, "Missing search text", "Enter text in the AI Find box to test the search prompt.")
                return
            prompt = editor_text.replace("{query}", query)
            prompt = render_prompt_with_agent_role(prompt, agent_role)
            prompt = render_prompt_with_user_hint(prompt)
        elif kind == "refine":
            vision_data = read_sidecar_data(record.image_path)
            parts: list[str] = []
            if vision_data.description:
                parts.append(f'description: "{vision_data.description}"')
            if vision_data.reasoning:
                parts.append(f'reasoning: "{vision_data.reasoning}"')
            vision_text = "\n".join(parts) if parts else "(no vision data available for this image)"
            prompt = editor_text.replace("{vision_data}", vision_text)
            prompt = render_prompt_with_agent_role(prompt, agent_role)
            prompt = render_prompt_with_user_hint(prompt)
        else:
            existing_tags = w._parse_annotations_for_tag_list(record.text)
            prompt = render_prompt_with_agent_role(editor_text, agent_role)
            prompt = render_prompt_with_existing_tags(prompt, existing_tags)
            prompt = render_prompt_with_user_hint(prompt)

        try:
            self._apply_query_downscale_setting()
        except LlmProviderError:
            return

        timeout = self._llm_timeout_seconds()
        image_path = record.image_path
        title = w._prompt_title(kind)
        cancel_token = LlmRequestCancellation()

        def test_task(report_progress: Callable[[str], None]) -> str:
            return session.generate(
                image_path,
                prompt,
                timeout=timeout,
                cancellation=cancel_token,
            )

        test_thread = QThread(w)
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
            dlg = LlmTestResultDialog(f"Test — {title}", display_text, w)
            dlg.exec()

        def _on_test_failed(error: str) -> None:
            test_thread.deleteLater()
            test_worker.deleteLater()
            QMessageBox.warning(w, f"Test failed — {title}", error)

        test_worker.finished.connect(_on_test_finished)
        test_worker.failed.connect(_on_test_failed)
        test_worker.cancelled.connect(_on_test_failed)

        w.statusBar().showMessage(f"Testing {title} prompt with {w._llm_provider.display_name}…")
        test_thread.start()

    # ------------------------------------------------------------------
    # Parallel job runner
    # ------------------------------------------------------------------

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

        w = self._window
        auto_mode = requested_threads == 0

        def cfg_int(key: str, default: int, minimum: int, maximum: int) -> int:
            raw_value = w._cfg.get(key, default)
            try:
                parsed = int(raw_value)
            except (TypeError, ValueError):
                parsed = default
            return max(minimum, min(maximum, parsed))

        auto_max_threads = cfg_int("llm_auto_max_threads", int(w._cfg.get("ollama_auto_max_threads", 32)), 1, 512)
        if auto_mode:
            max_threads = min(total, auto_max_threads)
            target_parallelism = 1
        else:
            max_threads = min(total, requested_threads)
            target_parallelism = max_threads

        self._llm_threads_auto_mode = auto_mode
        self._llm_threads_current = target_parallelism

        adaptive_warmup_items = cfg_int("llm_auto_warmup_items", int(w._cfg.get("ollama_auto_warmup_items", 4)), 1, 1000)
        scale_up_every = cfg_int("llm_auto_scale_up_every", int(w._cfg.get("ollama_auto_scale_up_every", 3)), 1, 100)
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

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate_with_llm(self) -> None:
        w = self._window
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

        selected_indexes = w._selected_record_indexes()
        if not selected_indexes:
            QMessageBox.information(w, "No image selected", "Select an image before generating annotations.")
            return
        include_tags = w.generate_tags_checkbox.isChecked()
        include_description = w.generate_description_checkbox.isChecked()
        include_vision = w.generate_vision_checkbox.isChecked()
        include_refine = w.generate_refine_checkbox.isChecked()
        if not include_tags and not include_description and not include_vision and not include_refine:
            QMessageBox.information(w, "Nothing selected", "Enable Tags, Description, Vision, or Refine before generating.")
            return

        cancel_token = LlmRequestCancellation()
        session = self._active_provider_session()
        if session is None:
            QMessageBox.warning(w, "No model selected", f"Choose a {w._llm_provider.display_name} model first.")
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
            agent_roles = w._cfg.get("agent_roles") or {}
            description_role = agent_roles.get("description") or None
            tagging_role = agent_roles.get("tagging") or None

            def process_one(position: int, record_index: int) -> dict:
                record = w.records[record_index]
                image_name = w._display_image_path(record.image_path)

                existing_parts = w._split_record_annotations(record.text)
                existing_tags_hint = [
                    part for part in existing_parts
                    if not w._is_description_like_annotation(part)
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
                image_perf_retry = False
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
                            from imagetagger.utils.annotations import sanitize_description_text
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
                                    for tag in w._parse_tags(
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
                            tags_input_lines: list[str] = []
                            if attempt_description:
                                tags_input_lines.append(attempt_description)
                            if attempt_tags:
                                tags_input_lines.extend(attempt_tags)
                            if not tags_input_lines:
                                tags_input_lines = w._parse_annotations_for_tag_list(record.text)
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
                                try:
                                    vision_data = read_sidecar_data(record.image_path)
                                    vision_data.description = attempt_vision_description
                                    vision_data.reasoning = attempt_vision_reasoning
                                    write_sidecar_data(record.image_path, vision_data)
                                except OSError as exc:
                                    raise LlmProviderError(f"Could not write {record.image_path.with_suffix('.json').name}: {exc}") from exc

                        if refine_enabled:
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
                        if not getattr(exc, "no_backoff", False):
                            image_perf_retry = True
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
                    "perf_retry": image_perf_retry,
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

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate_tags_with_llm(self) -> None:
        w = self._window
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

        selected_indexes = w._selected_record_indexes()
        if not selected_indexes:
            QMessageBox.information(w, "No image selected", "Select an image before validating tags.")
            return

        annotated_records: list[tuple[int, str]] = []
        skipped_without_annotations = 0
        for record_index in selected_indexes:
            if record_index < 0 or record_index >= len(w.records):
                continue
            annotations = w.records[record_index].text
            if not annotations.strip():
                skipped_without_annotations += 1
                continue
            annotated_records.append((record_index, annotations))

        if not annotated_records:
            QMessageBox.information(w, "No annotations to validate", "Add tags or a description before validating.")
            return

        cancel_token = LlmRequestCancellation()
        session = self._active_provider_session()
        if session is None:
            QMessageBox.warning(w, "No model selected", f"Choose a {w._llm_provider.display_name} model first.")
            return

        def validate_task(
            report_progress: Callable[[str], None],
            report_item: Callable[[object], None],
        ) -> object:
            timeout = self._llm_timeout_seconds()
            retry_count = self._llm_retry_count()
            total = len(annotated_records)

            def process_one(position: int, record_index: int, annotations: str) -> dict:
                record = w.records[record_index]
                image_name = w._display_image_path(record.image_path)
                image_retried = False
                image_perf_retry = False
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

                        if validation_result.strip().upper() != "OK":
                            from imagetagger.utils.fixup_parser import has_fixup_section_headers
                            if not has_fixup_section_headers(validation_result):
                                raise LlmProviderError("Model hallucinated output format (missing headers).")
                    except LlmProviderCancelled:
                        raise
                    except LlmProviderError as exc:
                        if "Timed out" in str(exc):
                            image_timed_out = True
                        if not getattr(exc, "no_backoff", False):
                            image_perf_retry = True
                        elapsed_seconds = time.monotonic() - attempt_start
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] validation_failed image={image_name} attempt={attempt + 1}/{retry_count + 1} elapsed_s={elapsed_seconds:.2f}",
                            flush=True,
                        )
                        if attempt >= retry_count:
                            return {
                                "kind": "validate_item_error",
                                "index": record_index,
                                "error": str(exc),
                                "timed_out": image_timed_out,
                                "context_exhausted": getattr(exc, "context_exhausted", False),
                                "position": position,
                                "total": total,
                                "image_name": image_name,
                            }
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
                    "perf_retry": image_perf_retry,
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
        self._validate_batch_errors = 0
        self._validate_batch_context_exhausted = 0
        self._validate_pending_indices = {idx for idx, _ in annotated_records}

        self._start_llm_task(
            task=validate_task,
            action_name="Validate",
            empty_message="Ollama returned no validation result.",
            cancel_token=cancel_token,
            validation_report=True,
        )

    # ------------------------------------------------------------------
    # AI Find
    # ------------------------------------------------------------------

    def ai_find_with_llm(self) -> None:
        w = self._window
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

        query = " ".join(w.ai_find_input.text().split())
        if not query:
            QMessageBox.information(w, "Missing search text", "Enter text to search for before running AI Find.")
            return

        selected_indexes = w._selected_record_indexes()
        if not selected_indexes:
            QMessageBox.information(w, "No image selected", "Select one or more images before running AI Find.")
            return

        cancel_token = LlmRequestCancellation()
        session = self._active_provider_session()
        if session is None:
            QMessageBox.warning(w, "No model selected", f"Choose a {w._llm_provider.display_name} model first.")
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
                record = w.records[record_index]
                image_name = record.image_path.name
                image_retried = False
                image_perf_retry = False
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
                        if not getattr(exc, "no_backoff", False):
                            image_perf_retry = True
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
                    "perf_retry": image_perf_retry,
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

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

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
        w = self._window
        if not w.llm_model_name.strip():
            QMessageBox.warning(w, "No model selected", f"Connect to {w._llm_provider.display_name} and choose a model first.")
            return
        if self._llm_thread is not None:
            return

        resize_warning = consume_image_preparation_warning()
        if resize_warning:
            QMessageBox.warning(w, "Image resize disabled", resize_warning)

        w.statusBar().showMessage(f"{action_name} with {w._llm_provider.display_name}...")
        w.validate_button.setEnabled(False)
        w.llm_endpoint_input.setEnabled(False)
        w.llm_fetch_button.setEnabled(False)
        w.llm_model_combo.setEnabled(False)
        w.llm_timeout_input.setEnabled(False)
        w.llm_retry_input.setEnabled(False)
        w.llm_max_resolution_input.setEnabled(False)
        w.llm_threads_input.setEnabled(False)
        w.llm_use_button.setEnabled(False)
        self._llm_action_name = action_name
        self._llm_cancel = cancel_token
        self._update_llm_controls()

        self._llm_thread = QThread(w)
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
        w = self._window
        if self._llm_action_name != "Generate" or self._llm_cancel is None:
            return
        w.generate_button.setEnabled(False)
        w.generate_button.setText("Stopping generation...")
        w.statusBar().showMessage("Stopping generation...")
        self._llm_cancel.cancel()

    def _request_stop_validation(self) -> None:
        w = self._window
        if self._llm_action_name != "Validate" or self._llm_cancel is None:
            return
        w.validate_button.setEnabled(False)
        w.validate_button.setText("Stopping validation...")
        w.statusBar().showMessage("Stopping validation...")
        self._llm_cancel.cancel()

    def _request_stop_ai_find(self) -> None:
        w = self._window
        if self._llm_action_name != "AI Find" or self._llm_cancel is None:
            return
        w.ai_find_button.setEnabled(False)
        w.ai_find_button.setText("Stopping AI Find...")
        w.statusBar().showMessage("Stopping AI Find...")
        self._llm_cancel.cancel()

    def _cleanup_llm_task(self) -> None:
        w = self._window
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
        self._validate_batch_errors = 0
        self._validate_batch_context_exhausted = 0
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
        w.llm_endpoint_input.setEnabled(True)
        w.llm_fetch_button.setEnabled(True)
        w.llm_model_combo.setEnabled(True)
        w.llm_timeout_input.setEnabled(True)
        w.llm_retry_input.setEnabled(True)
        w.llm_max_resolution_input.setEnabled(True)
        w.llm_threads_input.setEnabled(True)
        w.llm_use_button.setEnabled(True)
        w._update_fixup_button_state()

    # ------------------------------------------------------------------
    # Progress / item / finished / failed callbacks
    # ------------------------------------------------------------------

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
        w = self._window
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
            w.statusBar().showMessage(f"{message}{details}{self._llm_threads_status_suffix()}")
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
            errors_text = f" | errors: {self._validate_batch_errors}" if self._validate_batch_errors > 0 else ""
            disobeyed_text = f" | LLM disobeyed: {self._validate_batch_llm_disobeyed}" if self._validate_batch_llm_disobeyed > 0 else ""
            w.statusBar().showMessage(f"{message}{details}{issues_text}{errors_text}{disobeyed_text}{self._llm_threads_status_suffix()}")
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
            w.statusBar().showMessage(f"{message}{details}{found_text}{self._llm_threads_status_suffix()}")
            return

        w.statusBar().showMessage(f"{message}{self._llm_threads_status_suffix()}")

    def _on_llm_task_cancelled(self, message: str) -> None:
        w = self._window
        action = self._llm_action_name or "Request"
        if action == "Validate":
            total = self._validate_batch_total
            processed = self._validate_batch_processed
            if total > 0:
                w.statusBar().showMessage(
                    f"Validation stopped after {processed}/{total} image{'s' if total != 1 else ''}."
                )
            else:
                w.statusBar().showMessage(message or "Validation stopped.")
            return

        if action == "AI Find":
            total = self._ai_find_batch_total
            processed = self._ai_find_batch_processed
            matched = self._ai_find_batch_matched
            if total > 0:
                w.statusBar().showMessage(
                    f"AI Find stopped after {processed}/{total} image{'s' if total != 1 else ''} (found images: {matched})."
                )
            else:
                w.statusBar().showMessage(message or "AI Find stopped.")
            return

        w.statusBar().showMessage(message or f"{action} stopped.")

    def _on_llm_task_item_ready(self, payload: object) -> None:
        w = self._window
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
            if raw_index < 0 or raw_index >= len(w.records):
                return
            description = str(raw_description).strip() if isinstance(raw_description, str) else ""
            tags = [str(item).strip() for item in raw_tags] if isinstance(raw_tags, list) else []
            tags = [item for item in tags if item]

            updated, added = w._apply_generated_items_to_record(raw_index, description, tags)
            self._generate_batch_processed += 1
            if isinstance(raw_retried, bool) and raw_retried:
                self._generate_batch_retry_images += 1
            if updated:
                self._generate_batch_updated += 1
                self._generate_batch_new_annotations += added
                w._rebuild_known_tags_from_records()
                w._refresh_tag_completions()

            if isinstance(raw_position, int) and isinstance(raw_total, int) and raw_total > 0:
                details = self._batch_progress_details(
                    self._generate_batch_processed,
                    self._generate_batch_total,
                    self._generate_batch_started_at,
                    self._generate_batch_retry_images,
                )
                w.statusBar().showMessage(
                    f"Generate: applied {raw_position}/{raw_total} - {w.records[raw_index].image_path.name}{details}{self._llm_threads_status_suffix()}"
                )
            vision_desc = str(raw_vision_description).strip() if isinstance(raw_vision_description, str) else ""
            vision_reason = str(raw_vision_reasoning).strip() if isinstance(raw_vision_reasoning, str) else ""
            if vision_desc or vision_reason:
                self._generate_batch_vision_updated += 1
                if w.current_index == raw_index:
                    w._load_vision_for_current_image()

            raw_refine_tags = payload.get("refine_tags")
            raw_refine_caption = payload.get("refine_caption")
            refine_tags = [str(t).strip() for t in raw_refine_tags if str(t).strip()] if isinstance(raw_refine_tags, list) else []
            refine_caption = str(raw_refine_caption).strip() if isinstance(raw_refine_caption, str) else ""
            if refine_tags or refine_caption:
                try:
                    record_refine_result_for_image(
                        w.records[raw_index].image_path,
                        refine_tags,
                        refine_caption,
                    )
                except OSError:
                    pass
                else:
                    self._generate_batch_refine_updated += 1
                    w._on_fixup_state_changed(w.records[raw_index].image_path)
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
            if raw_index < 0 or raw_index >= len(w.records):
                return

            matched = bool(raw_matched)
            query = " ".join(str(raw_query or "").split())

            self._ai_find_batch_processed += 1
            if isinstance(raw_retried, bool) and raw_retried:
                self._ai_find_batch_retry_images += 1

            if matched and query:
                try:
                    record_ai_find_match_for_image(
                        w.records[raw_index].image_path,
                        query,
                        normalize_annotation=w._sanitize_annotation_text,
                    )
                except OSError:
                    pass
                else:
                    self._ai_find_batch_matched += 1
                    w._on_fixup_state_changed(w.records[raw_index].image_path)

            if isinstance(raw_position, int) and isinstance(raw_total, int) and raw_total > 0:
                details = self._batch_progress_details(
                    self._ai_find_batch_processed,
                    self._ai_find_batch_total,
                    self._ai_find_batch_started_at,
                    self._ai_find_batch_retry_images,
                )
                result_text = "match" if matched else "no match"
                found_text = f" | found images: {self._ai_find_batch_matched}"
                w.statusBar().showMessage(
                    f"AI Find: applied {raw_position}/{raw_total} - {w.records[raw_index].image_path.name} ({result_text}){details}{found_text}{self._llm_threads_status_suffix()}"
                )
            return

        if kind == "validate_item_error":
            raw_index = payload.get("index")
            raw_error = str(payload.get("error") or "")
            raw_image_name = str(payload.get("image_name") or "").strip()
            raw_timed_out = bool(payload.get("timed_out"))
            raw_context_exhausted = bool(payload.get("context_exhausted"))

            if isinstance(raw_index, int) and 0 <= raw_index < len(w.records):
                self._validate_pending_indices.discard(raw_index)
            self._validate_batch_processed += 1
            self._validate_batch_errors += 1
            if raw_timed_out:
                self._validate_batch_retry_images += 1
            if raw_context_exhausted:
                self._validate_batch_context_exhausted += 1

            print(
                f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] validation_error image={raw_image_name} error={raw_error!r}",
                flush=True,
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
        if raw_index < 0 or raw_index >= len(w.records):
            return

        outcome, llm_violated_no_commas = w._apply_validation_result_to_record(raw_index, str(raw_result or ""))
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
            w.statusBar().showMessage(
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
        w = self._window
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
                QMessageBox.information(w, f"{action_name} finished", empty_message)
                w.statusBar().showMessage(f"{action_name} finished")
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
                w.statusBar().showMessage(
                    f"{action_name} complete via {w._llm_provider.display_name} ({', '.join(parts)})"
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
                    w,
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
                QMessageBox.information(w, f"{action_name} finished", empty_message)
                w.statusBar().showMessage(f"{action_name} complete (found images: 0 of {total} for '{query}')")
            else:
                w.statusBar().showMessage(
                    f"{action_name} complete (found images: {matched} of {total} for '{query}')"
                )

            self._ai_find_batch_total = 0
            self._ai_find_batch_processed = 0
            self._ai_find_batch_matched = 0
            self._ai_find_batch_retry_images = 0
            return

        if isinstance(result, dict) and result.get("batch") is True and result.get("streamed") is True:
            has_annotations = self._generate_batch_updated > 0
            has_vision = self._generate_batch_vision_updated > 0
            has_refine = self._generate_batch_refine_updated > 0

            if not has_annotations and not has_vision and not has_refine:
                QMessageBox.information(w, f"{action_name} finished", empty_message)
                w.statusBar().showMessage(f"{action_name} finished")
            else:
                parts = []
                if has_annotations:
                    parts.append(f"{self._generate_batch_updated} image{'s' if self._generate_batch_updated != 1 else ''}, {self._generate_batch_new_annotations} new annotation{'s' if self._generate_batch_new_annotations != 1 else ''}")
                if has_vision:
                    parts.append(f"{self._generate_batch_vision_updated} vision update{'s' if self._generate_batch_vision_updated != 1 else ''}")
                if has_refine:
                    parts.append(f"{self._generate_batch_refine_updated} refine fixup{'s' if self._generate_batch_refine_updated != 1 else ''}")
                summary = ", ".join(parts)
                w.statusBar().showMessage(
                    f"{action_name} complete via {w._llm_provider.display_name} ({summary})"
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
                if raw_index < 0 or raw_index >= len(w.records):
                    continue

                raw_items = entry.get("items")
                items = [str(item).strip() for item in raw_items] if isinstance(raw_items, list) else []
                items = [item for item in items if item]

                record = w.records[raw_index]
                existing_tags = w._parse_tags(record.text)
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

                record.text = w._serialize_tags(merged_tags)
                if w._write_record_text(record, status_prefix="Generate + auto-saved"):
                    w._update_list_item_preview(raw_index)
                    updated += 1
                    with_new += added

            if updated == 0:
                QMessageBox.information(w, f"{action_name} finished", empty_message)
                w.statusBar().showMessage(f"{action_name} finished")
                return

            w._rebuild_known_tags_from_records()
            w._refresh_tag_completions()

            if 0 <= w.current_index < len(w.records):
                w._populate_tag_list(w._parse_annotations_for_tag_list(w.records[w.current_index].text))

            w.statusBar().showMessage(
                f"{action_name} complete via {w._llm_provider.display_name} ({updated} image{'s' if updated != 1 else ''}, {with_new} new annotation{'s' if with_new != 1 else ''})"
            )
            return

        if validation_report:
            cleaned = str(result).strip()
            if not cleaned:
                QMessageBox.information(w, f"{action_name} finished", empty_message)
                w.statusBar().showMessage(f"{action_name} finished")
                return

            if cleaned.casefold() == "ok":
                w.statusBar().showMessage("Validate complete: no issues found")
                record = w._current_record()
                if record is not None:
                    clear_validation_fields_sidecar(
                        record.image_path,
                        model=w.llm_model_name,
                        date=datetime.now().astimezone().isoformat(timespec="seconds"),
                    )
                    w._on_fixup_state_changed()
                return

            if w.current_index < 0 or w.current_index >= len(w.records):
                QMessageBox.warning(w, "Validate failed", "No selected image to write a fixup file.")
                w.statusBar().showMessage("Validate failed")
                return

            record = w.records[w.current_index]
            try:
                from imagetagger.utils.fixup_parser import parse_fixup_data
                parsed_fixup = parse_fixup_data(cleaned, w._parse_tags, w._sanitize_annotation_text)
                if not parsed_fixup.issues and not parsed_fixup.corrected_tags and not parsed_fixup.corrected_description_raw:
                    clear_validation_fields_sidecar(
                        record.image_path,
                        model=w.llm_model_name,
                        date=datetime.now().astimezone().isoformat(timespec="seconds"),
                    )
                    w.statusBar().showMessage("Validate complete: no issues found")
                    w._on_fixup_state_changed()
                    return
                write_fixup_sidecar(
                    record.image_path,
                    parsed_fixup.issues or None,
                    parsed_fixup.corrected_tags or None,
                    parsed_fixup.corrected_description_raw or None,
                    model=w.llm_model_name,
                    date=datetime.now().astimezone().isoformat(timespec="seconds"),
                )
            except OSError as exc:
                QMessageBox.warning(w, "Fixup write failed", f"Could not write sidecar:\n{exc}")
                w.statusBar().showMessage("Validate failed: could not write sidecar")
                return

            w.statusBar().showMessage("Validate found issues: saved to sidecar")
            w._on_fixup_state_changed()
            return

        if isinstance(result, list):
            tags = [str(item).strip() for item in result if str(item).strip()]
        else:
            text_result = str(result)
            tags = [text_result.strip()] if result_as_single else w._parse_tags(text_result)
        if not tags:
            QMessageBox.information(w, f"{action_name} finished", empty_message)
            w.statusBar().showMessage(f"{action_name} finished")
            return

        if merge_with_existing:
            existing_tags = w._current_tags()
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
                w.statusBar().showMessage(f"{action_name} finished (no new tags added)")
                return

            w._set_current_tags(merged_tags, status_prefix=f"{action_name} + auto-saved")
            w.statusBar().showMessage(
                f"{action_name} complete via {w._llm_provider.display_name} ({added_count} tag{'s' if added_count != 1 else ''} added)"
            )
            return

        w._set_current_tags(tags, status_prefix=f"{action_name} + auto-saved")
        w.statusBar().showMessage(f"{action_name} complete via {w._llm_provider.display_name}")

    def _on_llm_task_failed(self, message: str) -> None:
        w = self._window
        QMessageBox.warning(w, f"{w._llm_provider.display_name} request failed", message)
        w.statusBar().showMessage(f"{w._llm_provider.display_name} request failed")
