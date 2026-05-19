from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize, QThread, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox

from imagetagger import config as _config
from imagetagger.utils.annotations import parse_tags_text
from imagetagger.utils.filter_parser import (
    FilterSyntaxError,
    _FilterNode,
    _FilterRuntime,
    _parse_filter_expression,
)
from imagetagger.ui.models import ImageRecord, _UNKNOWN
from imagetagger.ui.workers import IMAGE_EXTENSIONS, THUMB_SIZE, FolderLoadWorker

if TYPE_CHECKING:
    from imagetagger.ui.main_window import MainWindow

# Mirror the item-data roles defined in main_window — same Qt enum values.
_ROLE_PIXMAP = Qt.ItemDataRole.UserRole          # QPixmap | None — pre-scaled thumbnail
_ROLE_BADGES = Qt.ItemDataRole.UserRole + 1      # frozenset[str] — active badge symbols


class DirectoryController:
    """Owns all folder-loading logic extracted from MainWindow (step 4.3).

    All UI widget access and shared-state access goes through ``self._window``.
    """

    def __init__(self, window: "MainWindow") -> None:
        self._window = window

        # State fields moved off MainWindow
        self._loader_thread: QThread | None = None
        self._loader_worker: FolderLoadWorker | None = None
        self._root_directory: Path | None = None
        self._directory_loading_active: bool = False
        self._directory_load_cancel_requested: bool = False
        self._directory_load_cancelled_by_user: bool = False
        self._pending_selection_path: Path | None = None
        self._icc_warning_paths: list[str] = []

    # ------------------------------------------------------------------
    # Public entry points (called from MainWindow / menu actions)
    # ------------------------------------------------------------------

    def open_folder(self) -> None:
        if self._loader_thread and self._loader_thread.isRunning():
            self._window.statusBar().showMessage("A folder is already loading")
            return

        start_dir = self._window._cfg.get("last_open_directory", "") or ""
        folder = QFileDialog.getExistingDirectory(self._window, "Select image folder", start_dir)
        if not folder:
            return

        self.load_directory(Path(folder))

    def refresh_directory(self) -> None:
        if self._loader_thread and self._loader_thread.isRunning():
            self._window.statusBar().showMessage("A folder is already loading")
            return

        folder = self._root_directory
        if folder is None:
            QMessageBox.information(self._window, "No folder selected", "Open a folder before refreshing it.")
            return

        w = self._window
        current_record = w._current_record()
        restore_selection = current_record.image_path if current_record is not None else None
        self.load_directory(folder, restore_selection=restore_selection)

    def load_directory(self, folder: Path, restore_selection: Path | None = None) -> None:
        self._directory_loading_active = True
        self._directory_load_cancel_requested = False
        self._directory_load_cancelled_by_user = False
        self._set_loading_state(True)
        self._window.statusBar().showMessage("Scanning folder...")
        self._pending_selection_path = restore_selection
        self._window._detected_external_editors = None

        try:
            max_thread_cap = int(self._window._cfg.get("directory_loader_max_threads", 8))
        except (TypeError, ValueError):
            max_thread_cap = 8
        max_thread_cap = max(1, max_thread_cap)

        self._loader_thread = QThread(self._window)
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

    # ------------------------------------------------------------------
    # Window-title helper
    # ------------------------------------------------------------------

    def _active_directory(self) -> Path | None:
        w = self._window
        record = w._current_record()
        if record is not None:
            return record.image_path.parent

        if w.records:
            return w.records[0].image_path.parent

        folder = str(w._cfg.get("last_open_directory", "")).strip()
        if not folder:
            return None

        return Path(folder)

    def _update_window_title(self, folder: Path | None) -> None:
        base_title = "ImageTagger"
        if folder is None:
            self._window.setWindowTitle(base_title)
            return

        name = folder.name.strip() or str(folder)
        self._window.setWindowTitle(f"{base_title} - {name}")

    # ------------------------------------------------------------------
    # Pre-load collision check
    # ------------------------------------------------------------------

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
            QMessageBox.critical(self._window, "Folder load failed", f"Failed to read folder: {exc}")
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

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

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

        self._window.statusBar().showMessage(f"Loading {total} images...")
        self._loader_worker.allow_processing()

    def _on_scan_progress(self, files: int, directories: int, images: int) -> None:
        if self._directory_load_cancel_requested:
            return
        self._window.statusBar().showMessage(
            f"Scanning folder... {files} files, {directories} directories, {images} images"
        )

    def _on_collision_detected(self, first: str, second: str) -> None:
        self._directory_loading_active = False
        self._window.statusBar().showMessage("Duplicate images detected")
        self._set_loading_state(False)
        if self._loader_worker is not None:
            self._loader_worker.cancel()
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._loader_thread.quit()

        QMessageBox.warning(
            self._window,
            "Duplicate image names detected",
            "Two image files have the same name with different extensions, which would "
            "collide on the same .txt description file.\n\n"
            f"Filename: {Path(first).stem}\n"
            f"Files:\n- {first}\n- {second}",
        )

    def _on_item_loaded(self, payload: object) -> None:
        if self._directory_load_cancel_requested:
            return

        # payload is a list[dict] batch emitted per chunk from FolderLoadWorker.
        items = payload if isinstance(payload, list) else []
        if not items:
            return

        w = self._window

        # Parse the filter expression once per batch instead of once per item.
        expression = w.filter_input.text().strip()
        batch_parsed: _FilterNode | None = None
        batch_runtime: _FilterRuntime | None = None
        if expression:
            try:
                batch_parsed = _parse_filter_expression(expression)
                if batch_parsed is not None:
                    batch_runtime = w._build_filter_runtime()
            except FilterSyntaxError:
                pass

        w.list_widget.setUpdatesEnabled(False)
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
                # Pre-populate the validated cache so filter evaluations never
                # need to hit the filesystem for this record.
                validated_raw = data.get("validated", _UNKNOWN)
                record._sidecar_validated = validated_raw
                w.records.append(record)
                w._record_index_by_path[image_path] = len(w.records) - 1

                # Compute visibility with the pre-parsed filter (avoids O(n) re-parses).
                if batch_parsed is not None and batch_runtime is not None:
                    is_visible: bool | None = batch_parsed.evaluate(record, batch_runtime)
                elif expression:
                    is_visible = True  # parse failed → show everything
                else:
                    is_visible = True
                w._add_list_item(record, thumb_image, active_badges=active_badges, is_visible=is_visible)

                # Accumulate tag counts incrementally so _on_load_finished can skip
                # the full O(n) rebuild over all records.
                parsed_tags = parse_tags_text(text)
                w.tag_counts.update(parsed_tags)
                w.known_tags.update(parsed_tags)
        finally:
            w.list_widget.setUpdatesEnabled(True)

    def _on_load_progress(self, processed: int, total: int, percent: int) -> None:
        if self._directory_load_cancel_requested:
            return

        self._window.statusBar().showMessage(
            f"Processed {processed} images of {total}, {percent}% done"
        )

    def _on_icc_warning_detected(self, image_path: str) -> None:
        normalized = image_path.strip()
        if not normalized:
            return
        if normalized not in self._icc_warning_paths:
            self._icc_warning_paths.append(normalized)
        self._window.statusBar().showMessage(f"Warning: invalid ICC profile in {Path(normalized).name}")

    def _on_load_finished(self, total: int, folder: str) -> None:
        w = self._window
        self._directory_loading_active = False
        self._set_loading_state(False)
        if self._directory_load_cancel_requested:
            self._pending_selection_path = None
            self._icc_warning_paths = []
            w.statusBar().showMessage("Folder loading stopped")
            return

        # Persist startup folder only for completed (non-aborted) loads.
        w._cfg["last_open_directory"] = folder
        _config.save(w._cfg)

        # Tag counts were built incrementally in _on_item_loaded; just refresh
        # the sorted completion list and known-tags UI without re-iterating records.
        w._refresh_tag_completions()
        w._apply_tag_list_height()
        self._restore_selection_after_load()
        if w.records:
            w.statusBar().showMessage(f"Loaded {len(w.records)} images from {folder}")
            if self._icc_warning_paths:
                affected = len(self._icc_warning_paths)
                preview_lines = [Path(path).name for path in self._icc_warning_paths[:10]]
                details = "\n".join(preview_lines)
                if affected > 10:
                    details += f"\n... and {affected - 10} more"
                QMessageBox.warning(
                    w,
                    "Invalid ICC profiles detected",
                    "Some images contain invalid ICC profiles.\n"
                    "These images may trigger stderr warnings and should be fixed before downstream use.\n\n"
                    f"Affected images ({affected}):\n{details}",
                )
        else:
            w.statusBar().showMessage("No supported images found in selected folder")
        self._icc_warning_paths = []

    def _on_load_failed(self, message: str) -> None:
        w = self._window
        self._directory_loading_active = False
        self._set_loading_state(False)
        self._pending_selection_path = None
        self._icc_warning_paths = []
        QMessageBox.critical(w, "Folder load failed", message)
        w.statusBar().showMessage("Folder load failed")

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
                self._window.statusBar().showMessage("Folder loading stopped")

    # ------------------------------------------------------------------
    # Loading-state and data helpers
    # ------------------------------------------------------------------

    def _set_loading_state(self, loading: bool) -> None:
        w = self._window
        if w.open_action is not None:
            w.open_action.setEnabled(not loading)
        if w.refresh_action is not None:
            w.refresh_action.setEnabled(not loading)
        if w.save_action is not None:
            w.save_action.setEnabled(not loading)
        w.list_widget.setEnabled(not loading)
        w.tag_input.setEnabled(not loading)
        w.tag_list.setEnabled(not loading)
        w.llm_endpoint_input.setEnabled(not loading and w._llm_thread is None)
        w.llm_fetch_button.setEnabled(not loading and w._llm_thread is None)
        w.llm_model_combo.setEnabled(not loading and w._llm_thread is None)
        w.llm_timeout_input.setEnabled(not loading and w._llm_thread is None)
        w.llm_retry_input.setEnabled(not loading and w._llm_thread is None)
        w.llm_max_resolution_input.setEnabled(not loading and w._llm_thread is None)
        w.llm_use_button.setEnabled(not loading and w._llm_thread is None)
        w.ai_find_input.setEnabled(not loading and w._llm_thread is None)
        w.ai_find_button.setEnabled(not loading and w._llm_thread is None and bool(w.llm_model_name.strip()))
        w.fixup_button.setEnabled(False)
        w.stop_loading_button.setVisible(loading)
        w.stop_loading_button.setEnabled(loading and self._directory_loading_active)
        if loading:
            w.generate_button.setEnabled(False)
            w.validate_button.setEnabled(False)
            w.ai_find_button.setEnabled(False)
        else:
            w._update_llm_controls()

    def _clear_loaded_directory_data(self, reset_root: bool) -> None:
        w = self._window
        self._icc_warning_paths = []
        w.records = []
        w._record_index_by_path.clear()
        w.known_tags.clear()
        w.tag_counts.clear()
        w.list_widget.clear()
        w.image_label.setText("No image selected")
        w.tag_input.clear()
        w.tag_list.clear()
        w._clear_vision_fields()
        w.current_index = -1
        w._set_watched_image(None)
        if reset_root:
            self._root_directory = None
            self._update_window_title(self._active_directory())
        w._refresh_tag_completions()

    def _request_stop_directory_loading(self) -> None:
        if not self._directory_loading_active:
            return

        self._directory_load_cancel_requested = True
        self._directory_load_cancelled_by_user = True

        if self._loader_worker is not None:
            self._loader_worker.cancel()

        w = self._window
        # Aborted loads should not auto-resume on next app start.
        w._cfg["last_open_directory"] = ""
        w._cfg.pop("last_selected_image", None)
        _config.save(w._cfg)

        self._clear_loaded_directory_data(reset_root=True)
        w.statusBar().showMessage("Stopping folder load and discarding partial results...")
        w.stop_loading_button.setEnabled(False)

    def _restore_selection_after_load(self) -> None:
        w = self._window
        if not w.records:
            self._pending_selection_path = None
            return

        target_path = self._pending_selection_path
        self._pending_selection_path = None
        if target_path is not None:
            for index, record in enumerate(w.records):
                if record.image_path == target_path:
                    item = w.list_widget.item(index)
                    if item is not None and not item.isHidden():
                        w.list_widget.setCurrentRow(index)
                        return

        first_visible = w._first_visible_row()
        if first_visible >= 0:
            w.list_widget.setCurrentRow(first_visible)
