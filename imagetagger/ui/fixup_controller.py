from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThreadPool, QTimer, pyqtSlot

from imagetagger.providers.llm_provider import LlmProviderError
from imagetagger.ui.merge_actions import open_fixup_dialog_for_image
from imagetagger.ui.workers import _SimpleRunnable
from imagetagger.ui.models import _UNKNOWN

if TYPE_CHECKING:
    from imagetagger.ui.main_window import MainWindow


class FixupController:
    """Owns all fixup-navigation and fixup-button logic extracted from MainWindow (step 4.4).

    All UI widget access and shared-state access goes through ``self._window``.
    """

    def __init__(self, window: "MainWindow") -> None:
        self._window = window

        # State moved off MainWindow
        self._fixup_navigating: bool = False

    # ------------------------------------------------------------------
    # Fixup button state
    # ------------------------------------------------------------------

    def _update_fixup_button_state(self) -> None:
        w = self._window
        record = w._current_record()
        enabled = record is not None
        if enabled:
            item = w.list_widget.item(w.current_index)
            if item is None or item.isHidden():
                enabled = False
        if w._llm_thread is not None:
            if w._llm_action_name != "Validate" or w.current_index in w._validate_pending_indices:
                enabled = False
        w.fixup_button.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Fixup state change (called after LLM validate/generate batch ops)
    # ------------------------------------------------------------------

    def _on_fixup_state_changed(self, image_path: Path | None = None) -> None:
        w = self._window
        if image_path is None:
            # Bulk invalidation: reset all cached values, then warm the cache
            # in a background thread so sidecar reads don't block the main thread.
            for record in w.records:
                record._sidecar_has_pending_fixup = None
                record._sidecar_validated = _UNKNOWN
            self._async_warm_fixup_cache()
        else:
            # Single-image path: one sidecar read is fine on the main thread.
            record_index = w._record_index_by_path.get(image_path, -1)
            if record_index >= 0:
                record = w.records[record_index]
                record._sidecar_has_pending_fixup = None
                record._sidecar_validated = _UNKNOWN
                w._update_list_item_preview(record_index)
            # Only re-evaluate the filter when one is actually active; with no
            # filter there is nothing to hide/show and calling _apply_image_filter
            # would redundantly iterate all n list items and re-set the selection.
            if w.filter_input.text().strip():
                w._apply_image_filter()
            self._update_fixup_button_state()
            current = w._current_record()
            if current is not None and current.image_path == image_path:
                w._update_image_label_tooltip()

    # Step 5.2 — background cache warm
    def _async_warm_fixup_cache(self) -> None:
        records = list(self._window.records)  # snapshot to avoid mutation during iteration

        def _warm() -> None:
            from imagetagger.utils.sidecar import read_sidecar_data
            for record in records:
                sidecar = read_sidecar_data(record.image_path)
                record._sidecar_has_pending_fixup = sidecar.has_pending_fixup
                record._sidecar_validated = sidecar.validated

        worker = _SimpleRunnable(_warm)
        worker.finished.connect(self._on_fixup_cache_warmed)
        QThreadPool.globalInstance().start(worker)

    # Step 5.3 — slot called on the main thread once the cache is warm
    @pyqtSlot()
    def _on_fixup_cache_warmed(self) -> None:
        w = self._window
        QTimer.singleShot(0, w._refresh_all_list_item_previews)
        w._apply_image_filter()
        self._update_fixup_button_state()
        current = w._current_record()
        if current is not None:
            w._update_image_label_tooltip()

    # ------------------------------------------------------------------
    # Fixup index navigation helpers
    # ------------------------------------------------------------------

    def _find_adjacent_fixup_index(self, start_index: int, direction: int) -> int | None:
        w = self._window
        if direction not in (-1, 1):
            return None

        index = start_index + direction
        while 0 <= index < len(w.records):
            item = w.list_widget.item(index)
            if item is not None and item.isHidden():
                index += direction
                continue

            if index in w._validate_pending_indices:
                index += direction
                continue

            record = w.records[index]
            if record.has_pending_fixup:
                return index

            index += direction

        return None

    def _find_fixup_index(self, reverse: bool) -> int | None:
        w = self._window
        if not w.records:
            return None

        indices = range(len(w.records) - 1, -1, -1) if reverse else range(len(w.records))
        for index in indices:
            if index in w._validate_pending_indices:
                continue
            item = w.list_widget.item(index)
            if item is not None and item.isHidden():
                continue
            record = w.records[index]
            if record.has_pending_fixup:
                return index
        return None

    def _jump_to_first_fixup(self) -> None:
        w = self._window
        index = self._find_fixup_index(reverse=False)
        if index is None:
            w.statusBar().showMessage("No fixup image found in the current list")
            return
        w.list_widget.setCurrentRow(index)

    def _jump_to_last_fixup(self) -> None:
        w = self._window
        index = self._find_fixup_index(reverse=True)
        if index is None:
            w.statusBar().showMessage("No fixup image found in the current list")
            return
        w.list_widget.setCurrentRow(index)

    # ------------------------------------------------------------------
    # Fixup dialog
    # ------------------------------------------------------------------

    def open_fixup_dialog(self) -> None:
        w = self._window
        record = w._current_record()
        if record is None:
            return

        try:
            regenerate_max_resolution_mpx = w._apply_query_downscale_setting()
        except LlmProviderError:
            return

        initial_fixup_record_indices = [
            i for i, item in enumerate(w.records)
            if item.has_pending_fixup
        ]
        initial_fixup_total = len(initial_fixup_record_indices)

        regenerate_tags_enabled = w.generate_tags_checkbox.isChecked()
        regenerate_description_enabled = w.generate_description_checkbox.isChecked()
        timeout_text = w.llm_timeout_input.text().strip()
        retry_text = w.llm_retry_input.text().strip()
        try:
            regenerate_timeout_seconds = int(timeout_text) if timeout_text else int(w._llm_timeout_seconds())
        except ValueError:
            regenerate_timeout_seconds = int(w._llm_timeout_seconds())
        try:
            regenerate_retry_count = int(retry_text) if retry_text else 0
        except ValueError:
            regenerate_retry_count = 0

        # Persistent settings across image navigation
        regenerate_model_name = ""
        regenerate_model_endpoint = ""
        regenerate_user_hint = ""

        def _capture_regenerate_settings(values: dict) -> None:
            nonlocal regenerate_tags_enabled
            nonlocal regenerate_description_enabled
            nonlocal regenerate_timeout_seconds
            nonlocal regenerate_retry_count
            nonlocal regenerate_max_resolution_mpx
            nonlocal regenerate_model_name
            nonlocal regenerate_model_endpoint
            nonlocal regenerate_user_hint

            from imagetagger.utils.image_prep import configure_image_preparation

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
                w.llm_max_resolution_input.setText(w._format_mpx(regenerate_max_resolution_mpx))
                configure_image_preparation(max_image_pixels=max(1, int(regenerate_max_resolution_mpx * 1_000_000)))
            if isinstance(model_name, str):
                regenerate_model_name = model_name
            if isinstance(model_endpoint, str):
                regenerate_model_endpoint = model_endpoint
            if isinstance(user_hint, str):
                regenerate_user_hint = user_hint

        from imagetagger import config as _config

        _deferred_refresh_needed = False
        while True:
            record = w._current_record()
            if record is None:
                return

            try:
                display_index = initial_fixup_record_indices.index(w.current_index) + 1
            except ValueError:
                display_index = 1

            title_path = w._display_image_path(record.image_path)

            if initial_fixup_total > 0:
                dialog_title = f"Fixup - {title_path} ({display_index} of {initial_fixup_total})"
            else:
                dialog_title = f"Fixup - {title_path}"

            prev_fixup_index = self._find_adjacent_fixup_index(w.current_index, -1)
            next_fixup_index = self._find_adjacent_fixup_index(w.current_index, 1)
            mouse_actions_cfg = w._cfg.get("merge_table_mouse_actions", {})
            if not isinstance(mouse_actions_cfg, dict):
                mouse_actions_cfg = {}

            outcome = open_fixup_dialog_for_image(
                parent=w,
                image_path=record.image_path,
                current_annotations=w._current_tags(),
                title_text=dialog_title,
                parse_tags=w._parse_tags,
                sanitize_annotation=w._sanitize_annotation_text,
                apply_annotations=lambda tags, status_prefix, image_path=record.image_path: w._set_tags_for_image_path(
                    image_path,
                    tags,
                    status_prefix=status_prefix,
                ),
                show_status=w.statusBar().showMessage,
                refresh_fixup_state=self._on_fixup_state_changed,
                initial_geometry=w._merge_dialog_geometry_from_config(),
                save_geometry=w._save_merge_dialog_geometry,
                can_navigate_prev=prev_fixup_index is not None,
                can_navigate_next=next_fixup_index is not None,
                tag_suggestions=w._sorted_tag_suggestions(),
                provider_session=w._active_provider_session(),
                provider=w._llm_provider,
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
                delete_image=lambda image_path=record.image_path: w._delete_image_and_related_files(
                    image_path,
                    confirm=False,
                ),
                confirm_delete=w._confirm_on_delete_enabled(),
                save_regenerate_settings=_capture_regenerate_settings,
                reasoning_lines=int(w._cfg.get("merge_dialog_reasoning_lines", 5)),
            )

            if outcome == "prev":
                target = self._find_adjacent_fixup_index(w.current_index, -1)
                if target is not None:
                    self._fixup_navigating = True
                    try:
                        w.list_widget.setCurrentRow(target)
                    finally:
                        self._fixup_navigating = False
                    _deferred_refresh_needed = True
                    continue
                break

            if outcome == "next":
                # Deletion from merge dialog can change list indices while the dialog is open.
                # Recompute target from current state instead of using stale pre-dialog indices.
                current_record = w._current_record()
                if current_record is not None and current_record.has_pending_fixup:
                    continue

                target = self._find_adjacent_fixup_index(w.current_index, 1)
                if target is None:
                    target = self._find_adjacent_fixup_index(w.current_index, -1)
                if target is None:
                    target = self._find_fixup_index(reverse=False)
                if target is not None:
                    self._fixup_navigating = True
                    try:
                        w.list_widget.setCurrentRow(target)
                    finally:
                        self._fixup_navigating = False
                    _deferred_refresh_needed = True
                    continue
            break

        if _deferred_refresh_needed:
            record = w._current_record()
            if record is not None:
                w._show_image(record.image_path)
                w._load_vision_for_current_image()
