from __future__ import annotations

import sys

from PyQt6.QtGui import QKeySequence


def is_macos() -> bool:
    return sys.platform == "darwin"


def platform_key_sequence(default: str, macos: str | None = None) -> QKeySequence:
    if is_macos() and macos:
        return QKeySequence(macos)
    return QKeySequence(default)


def platform_key_sequences(default: list[str], macos: list[str] | None = None) -> list[QKeySequence]:
    specs = macos if is_macos() and macos else default
    return [QKeySequence(spec) for spec in specs]


def native_shortcut_text(shortcuts: QKeySequence | list[QKeySequence]) -> str:
    values = shortcuts if isinstance(shortcuts, list) else [shortcuts]
    labels: list[str] = []
    for shortcut in values:
        text = shortcut.toString(QKeySequence.SequenceFormat.NativeText)
        if text and text not in labels:
            labels.append(text)
    return " / ".join(labels)
