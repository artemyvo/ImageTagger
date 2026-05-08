from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton

class VisionPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Description", self))
        self.description = QTextEdit(self)
        self.description.setAcceptRichText(False)
        layout.addWidget(self.description, stretch=1)

        layout.addWidget(QLabel("CoT", self))
        self.reasoning = QTextEdit(self)
        self.reasoning.setAcceptRichText(False)
        layout.addWidget(self.reasoning, stretch=1)

        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self.save_button = QPushButton("Save", self)
        self.save_button.setEnabled(False)
        save_row.addWidget(self.save_button)
        layout.addLayout(save_row)
