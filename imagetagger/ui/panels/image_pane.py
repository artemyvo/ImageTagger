"""Image pane widget for FixupDialog.

Encapsulates image display, file-watching, and the right-click context menu
for the image.  Emits signals instead of calling back into the dialog.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from imagetagger.ui.image_reload_helper import ImageReloadHelper
from imagetagger.ui.scalable_image_label import ScalableImageLabel
from imagetagger.utils.external_editors import (
    ExternalEditor,
    discover_graphics_editors,
    launch_image_in_editor,
    launch_image_in_system_default,
)


class ImagePane(QWidget):
    """Image display pane with file-watching and context menu.

    Signals:
        status_message(str): Emitted when a status update should be shown in
            the regeneration panel (or any status bar).
        delete_result(bool): Emitted after a context-menu delete succeeds.
            ``True``  — has more fixup files; caller should navigate to next.
            ``False`` — no more fixup files; caller should enter no-fixups state.
    """

    status_message = pyqtSignal(str)
    delete_result = pyqtSignal(bool)

    _PANE_HEADER_BOTTOM_SPACING = 4
    _PANE_HEADER_EXTRA_HEIGHT = 4

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
        self._detected_external_editors: list[ExternalEditor] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_header_label = QLabel("Image", self)
        self._image_header_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._image_header_label.setContentsMargins(0, 0, 0, 0)
        self._image_header_label.setMinimumHeight(
            self._image_header_label.fontMetrics().height() + self._PANE_HEADER_EXTRA_HEIGHT
        )
        layout.addWidget(self._image_header_label)
        layout.addSpacing(self._PANE_HEADER_BOTTOM_SPACING)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        self.image_label = ScalableImageLabel(self)
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self._show_image_context_menu)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, stretch=1)

        layout.addWidget(regen_panel, stretch=0)

        self._image_reload_helper = ImageReloadHelper(self, self._on_image_reload)

        # Load initial image and start watching for external changes.
        self.load_image(image_path)
        self._image_reload_helper.set_watched_image(image_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, image_path: Path | None) -> bool:
        """Load and display *image_path*.  Returns ``True`` on success."""
        if image_path is None:
            self._set_header_text(None, None)
            self.image_label.clear_original_image("No image path provided")
            return False

        if not image_path.exists() or not image_path.is_file():
            self._set_header_text(None, None)
            self.image_label.clear_original_image(f"File not found:\n{image_path.name}")
            return False

        pixmap = self._load_normalized_pixmap(image_path)
        if pixmap.isNull():
            self._set_header_text(None, None)
            self.image_label.clear_original_image(f"Unsupported format:\n{image_path.name}")
            return False

        self._set_header_text(pixmap.width(), pixmap.height())
        self.image_label.setText("")
        self.image_label.set_original_image(pixmap)
        return True

    def set_watched_image(self, image_path: Path | None) -> None:
        """Set the image path to watch for external changes."""
        self._image_reload_helper.set_watched_image(image_path)

    def clear_for_deleted(self) -> None:
        """Show 'no fixup files remaining' state after the current file is deleted."""
        self._current_image_path = None
        self.image_label.clear_original_image("No fixup files remaining")
        self.image_label.setEnabled(False)
        self._set_header_text(None, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_image_reload(self, image_path: Path) -> None:
        if not self.load_image(image_path):
            return
        self.status_message.emit(f"Reloaded image: {image_path.name}")

    @staticmethod
    def _load_normalized_pixmap(image_path: Path) -> QPixmap:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return pixmap
        # Normalize high-DPI asset naming so preview sizing is consistent.
        pixmap.setDevicePixelRatio(1.0)
        return pixmap

    def _set_header_text(self, width: int | None, height: int | None) -> None:
        label = self._image_header_label
        if width is None or height is None or width <= 0 or height <= 0:
            label.setText("Image")
            return
        megapixels = (float(width) * float(height)) / 1_000_000.0
        label.setText(f"Image: {width}x{height} - {megapixels:0.1f} MPx")

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
            answer = QMessageBox.question(
                self,
                "Delete file",
                (
                    f"Delete this image and related files?\n\n"
                    f"Image: {self._current_image_path.name}\n"
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
        if not refresh and self._detected_external_editors is not None:
            return list(self._detected_external_editors)

        try:
            editors = discover_graphics_editors()
        except Exception:
            editors = []
        self._detected_external_editors = editors
        return list(editors)
