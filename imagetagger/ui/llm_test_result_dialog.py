from __future__ import annotations

from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class LlmTestResultDialog(QDialog):
    """Dialog that displays a raw LLM response for prompt testing."""

    def __init__(self, title: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 480)
        self.setMinimumSize(300, 200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._output = QPlainTextEdit(self)
        self._output.setReadOnly(True)
        self._output.setPlainText(text)
        # Intentionally use the system monospace font at its default size rather than
        # the user's configured app font size.  This dialog exists purely for quick
        # copy/paste of raw LLM query logs, where the fixed-width typeface at a
        # comfortable reading size matters more than matching the UI theme.
        mono_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._output.setFont(mono_font)

        button_row = QHBoxLayout()
        copy_button = QPushButton("Copy to clipboard", self)
        copy_button.clicked.connect(self._copy_to_clipboard)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)

        button_row.addWidget(copy_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)

        layout.addWidget(self._output, stretch=1)
        layout.addLayout(button_row)

    def _copy_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._output.toPlainText())
