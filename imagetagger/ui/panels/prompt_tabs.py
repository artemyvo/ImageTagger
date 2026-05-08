from __future__ import annotations

from typing import Callable
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QPushButton

def create_prompt_tab(
    parent: QWidget,
    kind: str,
    initial_text: str,
    on_text_changed: Callable[[str], None],
    on_apply: Callable[[str], None],
    on_save: Callable[[str], None],
    on_reset: Callable[[str], None],
) -> tuple[QWidget, QTextEdit, QLabel]:
    tab = QWidget(parent)
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(6)

    editor = QTextEdit(parent)
    editor.setAcceptRichText(False)
    editor.setPlainText(initial_text)
    editor.textChanged.connect(lambda: on_text_changed(kind))

    status_label = QLabel(parent)

    buttons_row = QHBoxLayout()
    apply_button = QPushButton("Apply", parent)
    apply_button.clicked.connect(lambda _checked=False: on_apply(kind))
    save_button = QPushButton("Save", parent)
    save_button.clicked.connect(lambda _checked=False: on_save(kind))
    reset_button = QPushButton("Reset", parent)
    reset_button.clicked.connect(lambda _checked=False: on_reset(kind))
    
    buttons_row.addWidget(apply_button)
    buttons_row.addWidget(save_button)
    buttons_row.addWidget(reset_button)
    buttons_row.addStretch(1)

    layout.addWidget(editor, stretch=1)
    layout.addWidget(status_label, stretch=0)
    layout.addLayout(buttons_row)

    return tab, editor, status_label
