from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QWidget

class ScalableImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._original_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: palette(base); border: 1px solid palette(mid);")

    def set_original_image(self, pixmap: QPixmap) -> None:
        self._original_pixmap = pixmap
        self._update_scaled_image()

    def clear_original_image(self, text: str = "") -> None:
        self._original_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def _update_scaled_image(self) -> None:
        if self._original_pixmap is None:
            return
        
        available_width = self.width() - 4
        available_height = self.height() - 4
        
        if available_width <= 0 or available_height <= 0:
            return
        
        scaled = self._original_pixmap.scaled(
            QSize(available_width, available_height),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_scaled_image()
