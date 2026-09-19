"""Image pane widget for FixupDialog.

Encapsulates image display, file-watching, the right-click context menu for
the image, and the "Fix ratio" crop mode.  Emits signals instead of calling
back into the dialog.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from imagetagger.ui.crop_overlay import CropOverlay
from imagetagger.ui.image_reload_helper import ImageReloadHelper
from imagetagger.ui.scalable_image_label import ScalableImageLabel
from imagetagger.utils.aspect_ratio import AspectRatio, CropCandidate, crop_candidates
from imagetagger.utils.external_editors import (
    ExternalEditor,
    get_graphics_editors,
    launch_image_in_editor,
    launch_image_in_system_default,
)
from imagetagger.utils.image_crop import ImageCropError, crop_image_file


class _ImageLoadSignaller(QObject):
    """Carries image-load result signals back to the main thread."""

    loaded = pyqtSignal(object, object, int)  # (Path, QImage, load generation)
    failed = pyqtSignal(object, int)          # (Path, load generation)


class _ImageLoadRunnable(QRunnable):
    """Loads a QImage on a thread-pool thread and signals completion."""

    def __init__(self, image_path: Path, signaller: _ImageLoadSignaller, generation: int) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._image_path = image_path
        self._signaller = signaller
        self._generation = generation

    def run(self) -> None:
        image = QImage(str(self._image_path))
        if image.isNull():
            self._signaller.failed.emit(self._image_path, self._generation)
        else:
            self._signaller.loaded.emit(self._image_path, image, self._generation)


class ImagePane(QWidget):
    """Image display pane with file-watching and context menu.

    Signals:
        status_message(str): Emitted when a status update should be shown in
            the regeneration panel (or any status bar).
        delete_result(bool): Emitted after a context-menu delete succeeds.
            ``True``  — has more fixup files; caller should navigate to next.
            ``False`` — no more fixup files; caller should enter no-fixups state.
        dimensions_changed(int, int): Emitted with the loaded image size, or
            ``(0, 0)`` while no image is available.
        crop_mode_changed(bool): Emitted when the "Fix ratio" crop mode is
            entered (``True``) or left (``False``).  The dialog treats crop
            mode as modal and parks its other controls meanwhile.
    """

    status_message = pyqtSignal(str)
    delete_result = pyqtSignal(bool)
    dimensions_changed = pyqtSignal(int, int)
    crop_mode_changed = pyqtSignal(bool)

    def __init__(
        self,
        image_path: Path | None,
        confirm_delete: bool,
        delete_image: Callable[[], tuple[bool, bool]] | None,
        regen_panel: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._current_image_path = image_path
        self._confirm_delete = confirm_delete
        self._delete_image = delete_image
        self._async_load_signaller: _ImageLoadSignaller | None = None
        self._pending_reload_status: Path | None = None
        self._image_size: tuple[int, int] | None = None
        # Bumped by every load_image(); results of superseded loads are dropped
        # so the pixmap, _image_size and the crop frame always describe one file state.
        self._load_generation = 0
        self._crop_mode_active = False
        self._crop_candidates: list[CropCandidate] = []
        # mtime of the file right after our own crop, so the file watcher does
        # not report that write as an external edit.
        self._self_written_mtime_ns: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.image_label = ScalableImageLabel(self)
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self._show_image_context_menu)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, stretch=1)

        self._crop_overlay = CropOverlay(self.image_label)
        self._crop_bar = self._build_crop_bar()
        self._crop_bar.hide()
        layout.addWidget(self._crop_bar, stretch=0)

        layout.addWidget(regen_panel, stretch=0)

        self._image_reload_helper = ImageReloadHelper(self, self._on_image_reload)

        # Load initial image and start watching for external changes.
        self.load_image(image_path)
        self._image_reload_helper.set_watched_image(image_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, image_path: Path | None) -> bool:
        """Load and display *image_path* asynchronously.

        Returns ``False`` immediately if the path is absent or invalid;
        returns ``True`` once background loading has been dispatched.
        The image label shows a placeholder until the load completes.
        """
        # The frame refers to the pixels that were on screen; never carry it
        # over to a reloaded or different image.
        self.cancel_crop_mode()
        self._image_size = None
        self._load_generation += 1

        if image_path is None:
            self._current_image_path = None
            self._set_header_text(None, None)
            self.image_label.clear_original_image("No image path provided")
            return False

        if not image_path.exists() or not image_path.is_file():
            self._current_image_path = image_path
            self._set_header_text(None, None)
            self.image_label.clear_original_image(f"File not found:\n{image_path.name}")
            return False

        # Update current path so context-menu actions work immediately,
        # then show a placeholder and start the background load.
        self._current_image_path = image_path
        self._set_header_text(None, None)
        self.image_label.clear_original_image("Loading\u2026")

        signaller = _ImageLoadSignaller()
        signaller.loaded.connect(self._on_async_image_loaded)
        signaller.failed.connect(self._on_async_image_failed)
        self._async_load_signaller = signaller  # keep reference alive until signal fires
        QThreadPool.globalInstance().start(
            _ImageLoadRunnable(image_path, signaller, self._load_generation)
        )
        return True

    def set_watched_image(self, image_path: Path | None) -> None:
        """Set the image path to watch for external changes."""
        self._image_reload_helper.set_watched_image(image_path)

    def clear_for_deleted(self) -> None:
        """Show 'no fixup files remaining' state after the current file is deleted."""
        self.cancel_crop_mode()
        self._image_size = None
        self._current_image_path = None
        self.image_label.clear_original_image("No fixup files remaining")
        self.image_label.setEnabled(False)
        self._set_header_text(None, None)

    def image_size(self) -> tuple[int, int] | None:
        """Pixel size of the image currently shown, or ``None`` while loading."""
        return self._image_size

    def is_crop_mode_active(self) -> bool:
        return self._crop_mode_active

    # ------------------------------------------------------------------
    # Fix ratio (crop mode)
    # ------------------------------------------------------------------

    def begin_ratio_fix(self, allowed_ratios: list[AspectRatio]) -> bool:
        """Enter crop mode with the closest allowed ratio preselected."""
        if self._crop_mode_active:
            return True
        if self._current_image_path is None or self._image_size is None:
            return False

        width, height = self._image_size
        # A candidate that keeps every pixel is not a fix.
        candidates = [
            candidate
            for candidate in crop_candidates(width, height, allowed_ratios)
            if (candidate.width, candidate.height) != (width, height)
        ]
        if not candidates:
            self.status_message.emit("Fix ratio: no allowed ratio can be applied to this image.")
            return False

        self._crop_candidates = candidates
        self._crop_ratio_combo.blockSignals(True)
        self._crop_ratio_combo.clear()
        for candidate in candidates:
            self._crop_ratio_combo.addItem(
                f"{candidate.ratio.label}  \u2192  {candidate.width}\u00d7{candidate.height}"
                f"  (keeps {candidate.kept_fraction:.1%})"
            )
        self._crop_ratio_combo.setCurrentIndex(0)  # sorted: most pixels kept first
        self._crop_ratio_combo.blockSignals(False)

        self._crop_mode_active = True
        self._crop_bar.show()
        self._crop_overlay.begin(width, height)
        self._on_crop_ratio_changed(0)
        self.crop_mode_changed.emit(True)
        self.status_message.emit(
            "Fix ratio: drag the frame (or use the arrow keys) to choose what to keep. "
            "Enter applies, Esc cancels."
        )
        return True

    def cancel_crop_mode(self) -> None:
        """Leave crop mode without touching the image file."""
        self._end_crop_mode()

    def handle_crop_key(self, event: QKeyEvent) -> bool:
        """Crop-mode keyboard handling; returns ``True`` when the key is consumed.

        Called from the dialog's key filter so the merge table's arrow/Enter
        behaviour and the dialog's Esc-to-close never fire while cropping.
        """
        if not self._crop_mode_active:
            return False

        key = event.key()
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier

        if key == Qt.Key.Key_Escape:
            if not event.isAutoRepeat():
                self._end_crop_mode()
                self.status_message.emit("Fix ratio cancelled.")
            return True

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if modifiers == Qt.KeyboardModifier.NoModifier and not event.isAutoRepeat():
                self._apply_crop()
            return True

        steps = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        if key in steps:
            if not (modifiers & ~Qt.KeyboardModifier.ShiftModifier):
                step_x, step_y = steps[key]
                self._crop_overlay.nudge(
                    step_x,
                    step_y,
                    big=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
                )
            return True

        return False

    def _build_crop_bar(self) -> QWidget:
        bar = QWidget(self)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._crop_ratio_combo = QComboBox(bar)
        self._crop_ratio_combo.setToolTip(
            "Target ratio. Allowed ratios are listed closest first (most pixels kept)."
        )
        self._crop_ratio_combo.currentIndexChanged.connect(self._on_crop_ratio_changed)

        self.crop_apply_button = QPushButton("Apply", bar)
        self.crop_apply_button.setToolTip("Crop the image file to the frame (Enter)")
        self.crop_apply_button.clicked.connect(self._apply_crop)

        self.crop_cancel_button = QPushButton("Cancel", bar)
        self.crop_cancel_button.setToolTip("Leave the image unchanged (Esc)")
        self.crop_cancel_button.clicked.connect(self._cancel_crop_clicked)

        for button in (self.crop_apply_button, self.crop_cancel_button):
            button.setAutoDefault(False)
            button.setDefault(False)
        # Keep keyboard focus on the frame so the arrow keys always move it.
        for widget in (self._crop_ratio_combo, self.crop_apply_button, self.crop_cancel_button):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        row.addWidget(QLabel("Ratio", bar), stretch=0)
        row.addWidget(self._crop_ratio_combo, stretch=0)
        row.addStretch(1)
        row.addWidget(self.crop_apply_button, stretch=0)
        row.addWidget(self.crop_cancel_button, stretch=0)
        return bar

    def _current_crop_candidate(self) -> CropCandidate | None:
        index = self._crop_ratio_combo.currentIndex()
        if 0 <= index < len(self._crop_candidates):
            return self._crop_candidates[index]
        return None

    def _on_crop_ratio_changed(self, _index: int) -> None:
        candidate = self._current_crop_candidate()
        if candidate is None or not self._crop_mode_active:
            return
        self._crop_overlay.set_crop_size(candidate.width, candidate.height)

    def _cancel_crop_clicked(self) -> None:
        self._end_crop_mode()
        self.status_message.emit("Fix ratio cancelled.")

    def _end_crop_mode(self) -> None:
        if not self._crop_mode_active:
            return
        self._crop_mode_active = False
        self._crop_candidates = []
        self._crop_overlay.end()
        self._crop_bar.hide()
        self.crop_mode_changed.emit(False)

    def _apply_crop(self) -> None:
        image_path = self._current_image_path
        candidate = self._current_crop_candidate()
        if not self._crop_mode_active or image_path is None or candidate is None:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            new_width, new_height = crop_image_file(
                image_path,
                self._crop_overlay.crop_box(),
                # The size the frame was laid out on: refuses the crop if the
                # file on disk is no longer the image the user framed.
                expected_size=self._crop_overlay.image_size(),
            )
        except ImageCropError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Fix ratio failed", f"Could not crop {image_path.name}:\n{exc}")
            return
        QApplication.restoreOverrideCursor()

        self._end_crop_mode()
        try:
            self._self_written_mtime_ns = image_path.stat().st_mtime_ns
        except OSError:
            self._self_written_mtime_ns = None
        self.load_image(image_path)
        # Re-arm the watcher on the replaced file with the new mtime as baseline.
        self.set_watched_image(image_path)
        self.status_message.emit(
            f"Cropped {image_path.name} to {new_width}\u00d7{new_height} ({candidate.ratio.label})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_image_reload(self, image_path: Path) -> None:
        if self._self_written_mtime_ns is not None:
            try:
                unchanged_since_crop = image_path.stat().st_mtime_ns == self._self_written_mtime_ns
            except OSError:
                unchanged_since_crop = False
            if unchanged_since_crop:
                return  # late watcher event for our own crop; already reloaded
            self._self_written_mtime_ns = None
        self._pending_reload_status = image_path
        if not self.load_image(image_path):
            self._pending_reload_status = None
        # Status message is emitted by _on_async_image_loaded once load completes.

    def _on_async_image_loaded(self, image_path: object, image: object, generation: int) -> None:
        """Called on the main thread when the background QImage load succeeds."""
        path: Path = image_path  # type: ignore[assignment]
        if generation != self._load_generation or path != self._current_image_path:
            return  # stale result — a newer load was requested

        pixmap = QPixmap.fromImage(image)  # type: ignore[arg-type]
        pixmap.setDevicePixelRatio(1.0)
        if pixmap.isNull():
            self._set_header_text(None, None)
            self.image_label.clear_original_image(f"Unsupported format:\n{path.name}")
        else:
            self._image_size = (pixmap.width(), pixmap.height())
            self._set_header_text(pixmap.width(), pixmap.height())
            self.image_label.set_original_image(pixmap)

        if self._pending_reload_status == path:
            self._pending_reload_status = None
            self.status_message.emit(f"Reloaded image: {path.name}")

    def _on_async_image_failed(self, image_path: object, generation: int) -> None:
        """Called on the main thread when the background QImage load fails."""
        path: Path = image_path  # type: ignore[assignment]
        if generation != self._load_generation or path != self._current_image_path:
            return  # stale result
        self._set_header_text(None, None)
        self.image_label.clear_original_image(f"Unsupported format:\n{path.name}")
        if self._pending_reload_status == path:
            self._pending_reload_status = None

    @staticmethod
    def _load_normalized_pixmap(image_path: Path) -> QPixmap:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return pixmap
        # Normalize high-DPI asset naming so preview sizing is consistent.
        pixmap.setDevicePixelRatio(1.0)
        return pixmap

    def _set_header_text(self, width: int | None, height: int | None) -> None:
        if width is not None and height is not None and width > 0 and height > 0:
            self.dimensions_changed.emit(width, height)
        else:
            self.dimensions_changed.emit(0, 0)

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_image_context_menu(self, position) -> None:
        menu = QMenu(self)

        open_default_action = menu.addAction("Open in Default App")
        open_default_action.triggered.connect(self._open_image_in_default_app)

        open_with_menu = menu.addMenu("Open With")
        editors = self._get_detected_external_editors(refresh=False)
        if editors:
            for editor in editors:
                action = open_with_menu.addAction(editor.display_name)
                action.triggered.connect(
                    lambda _checked=False, e=editor: self._open_image_with_editor(e)
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
        if self._current_image_path is None:
            return

        if self._delete_image is None:
            QMessageBox.warning(self, "Delete failed", "Delete handler is not available.")
            return

        if self._confirm_delete:
            confirm = QMessageBox(self)
            confirm.setWindowTitle("Delete file")
            confirm.setText(
                f"Delete this image and related files?\n\n"
                f"Image: {self._current_image_path.name}\n"
                "Also deletes matching .txt and .fixup files"
            )
            confirm.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            confirm.setDefaultButton(QMessageBox.StandardButton.No)
            confirm.raise_()
            confirm.activateWindow()
            if confirm.exec() != QMessageBox.StandardButton.Yes:
                return

        deleted, has_fixups_remaining = self._delete_image()
        if not deleted:
            return

        self.delete_result.emit(has_fixups_remaining)

    def _open_image_in_default_app(self) -> None:
        image_path = self._current_image_path
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

        self.set_watched_image(image_path)
        self.status_message.emit(f"Opened {image_path.name} in default app")

    def _open_image_with_editor(self, editor: ExternalEditor) -> None:
        image_path = self._current_image_path
        if image_path is None:
            return

        try:
            launch_image_in_editor(editor, image_path)
        except OSError as exc:
            QMessageBox.warning(
                self, "Open editor failed",
                f"Could not open image with {editor.display_name}:\n{exc}",
            )
            return
        except Exception as exc:
            QMessageBox.warning(
                self, "Open editor failed",
                f"Could not open image with {editor.display_name}:\n{exc}",
            )
            return

        self.set_watched_image(image_path)
        self.status_message.emit(f"Opened {image_path.name} with {editor.display_name}")

    def _open_image_with_custom_editor(self) -> None:
        if self._current_image_path is None:
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

        launch_kind = (
            "mac_app"
            if sys.platform == "darwin" and selected.suffix.lower() == ".app"
            else "executable"
        )
        custom_editor = ExternalEditor(
            id="custom",
            display_name=selected.stem or selected.name,
            launch_target=str(selected),
            launch_kind=launch_kind,
        )
        self._open_image_with_editor(custom_editor)

    def _get_detected_external_editors(self, refresh: bool = False) -> list[ExternalEditor]:
        return get_graphics_editors(refresh=refresh)
