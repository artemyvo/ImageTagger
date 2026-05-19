from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel

if TYPE_CHECKING:
    from imagetagger.ui.main_window import MainWindow


class ImageViewController:
    """Owns image display, pixmap cache, and file-watch reload logic (step 4.5).

    Receives the image label widget via constructor injection.
    All reads/writes to shared MainWindow state go through ``self._window``.
    """

    def __init__(self, window: "MainWindow", image_label: QLabel) -> None:
        self._window = window
        self._image_label = image_label

        # Cached full-resolution pixmap for the currently displayed image.
        # Avoids re-reading from disk on resize events and rapid re-selections.
        self._cached_pixmap: QPixmap | None = None
        self._cached_pixmap_path: Path | None = None

    # ------------------------------------------------------------------
    # File-watch helpers
    # ------------------------------------------------------------------

    def _set_watched_image(self, image_path: Path | None) -> None:
        self._window._image_reload_helper.set_watched_image(image_path)

    def _on_image_reload(self, image_path: Path) -> None:
        """Callback invoked when the watched image is modified by an external editor."""
        w = self._window
        current_record = w._current_record()
        if current_record is None:
            return
        if self._normalized_path_for_compare(current_record.image_path) != self._normalized_path_for_compare(image_path):
            return

        # Skip transient save states where the file is temporarily unavailable.
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return

        # Invalidate the pixmap cache so _show_image re-reads the updated file.
        if self._cached_pixmap_path == image_path:
            self._cached_pixmap = None
            self._cached_pixmap_path = None

        self._show_image(image_path)

        record_index = w._record_index_for_image_path(image_path)
        if record_index >= 0:
            w._update_list_item_preview(record_index)

        w.statusBar().showMessage(f"Reloaded image: {image_path.name}")

    @staticmethod
    def _normalized_path_for_compare(path: Path) -> str:
        return os.path.normcase(str(path.resolve(strict=False)))

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def _show_image(self, image_path: Path) -> None:
        w = self._window
        # Re-use the cached pixmap when the same image is requested (e.g. during
        # a window resize) to avoid a disk read on every resize event.
        if self._cached_pixmap_path != image_path or self._cached_pixmap is None:
            pixmap = w._load_normalized_pixmap(image_path)
            if pixmap.isNull():
                self._image_label.setText(f"Unable to load image:\n{image_path.name}")
                self._image_label.setPixmap(QPixmap())
                self._cached_pixmap = None
                self._cached_pixmap_path = None
                return
            self._cached_pixmap = pixmap
            self._cached_pixmap_path = image_path
        else:
            pixmap = self._cached_pixmap

        target_size = self._image_label.size()
        if target_size.width() < 50 or target_size.height() < 50:
            return

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def _on_resize_timer(self) -> None:
        """Called after the resize debounce delay to re-scale the current image."""
        w = self._window
        if 0 <= w.current_index < len(w.records):
            self._show_image(w.records[w.current_index].image_path)
