from __future__ import annotations

from typing import Callable
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
from imagetagger.ui.diff_delegates import _diff_highlight_colors, _danger_button_stylesheet

class ItemActionWidget(QWidget):
    """Custom widget for list items with optional action button and diff highlighting."""
    
    def __init__(
        self,
        text: str,
        button_text: str = "",
        button_callback: Callable[[], None] | None = None,
        diff_ranges: list[tuple[int, int]] | None = None,
        button_on_left: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.text = text
        self.button_callback = button_callback
        self.diff_ranges = diff_ranges or []
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        
        # Create a label that will display highlighted text
        text_label = QLabel()
        text_label.setWordWrap(True)
        text_label.setMinimumHeight(20)
        text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Build HTML with highlighting for diff ranges
        html_text = self._build_highlighted_html(text, self.diff_ranges)
        text_label.setText(html_text)
        
        button: QPushButton | None = None
        if button_text and button_callback:
            button = QPushButton(button_text)
            button.setMaximumWidth(32)
            button.setMaximumHeight(20)
            button.clicked.connect(button_callback)
            if button_text == "✕":
                button.setStyleSheet(_danger_button_stylesheet(button.palette()))

        if button is not None and button_on_left:
            layout.addWidget(button, stretch=0)
            layout.addWidget(text_label, stretch=1)
        else:
            layout.addWidget(text_label, stretch=1)
            if button is not None:
                layout.addWidget(button, stretch=0)
        
        self.setLayout(layout)
    
    def _build_highlighted_html(self, text: str, ranges: list[tuple[int, int]]) -> str:
        """Build HTML with diff highlighting for specified ranges."""
        if not ranges:
            return f"<span>{self._escape_html(text)}</span>"
        
        # Sort ranges to avoid overlaps
        sorted_ranges = sorted(ranges)
        
        html_parts = []
        last_end = 0
        highlight_bg, highlight_text = _diff_highlight_colors(self.palette())
        highlight_bg_css = highlight_bg.name(QColor.NameFormat.HexRgb)
        highlight_text_css = highlight_text.name(QColor.NameFormat.HexRgb)
        
        for start, end in sorted_ranges:
            if start > last_end:
                html_parts.append(self._escape_html(text[last_end:start]))
            html_parts.append(
                f'<span style="background-color: {highlight_bg_css}; color: {highlight_text_css};">'
                f"{self._escape_html(text[start:end])}</span>"
            )
            last_end = end
        
        if last_end < len(text):
            html_parts.append(self._escape_html(text[last_end:]))
        
        return "".join(html_parts)
    
    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
