from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton

class FilterPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        self.filter_input = QLineEdit(self)
        self.filter_input.setPlaceholderText("Filter (fixup, \"tag\", 'text', &, |, parentheses)")
        
        self.filter_help_button = QPushButton("?", self)
        self.filter_help_button.setFixedWidth(26)
        self.filter_help_button.setToolTip("Show filter syntax help")
        
        layout.addWidget(self.filter_input, stretch=1)
        layout.addWidget(self.filter_help_button, stretch=0)
