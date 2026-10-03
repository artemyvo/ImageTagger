"""macOS shortcuts use Command ("Ctrl+" in Qt), never Control ("Meta+")."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence

import imagetagger.ui.main_window as main_window_module
import imagetagger.ui.shortcuts as shortcuts_module


def test_macos_main_window_shortcuts_use_command(make_main_window, monkeypatch):
    monkeypatch.setattr(shortcuts_module, "is_macos", lambda: True)
    monkeypatch.setattr(main_window_module, "is_macos", lambda: True, raising=False)
    window = make_main_window()

    using_control = []
    for action in window.findChildren(QAction):
        for sequence in action.shortcuts():
            for index in range(sequence.count()):
                if sequence[index].keyboardModifiers() & Qt.KeyboardModifier.MetaModifier:
                    using_control.append((action.text(), sequence.toString()))
    assert using_control == []

    assert window.open_action.shortcut() == QKeySequence("Ctrl+O")
    assert window.refresh_action.shortcut() == QKeySequence("Ctrl+R")
