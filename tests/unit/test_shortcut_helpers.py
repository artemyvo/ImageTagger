"""Platform shortcut helpers: macOS specs only on macOS, native text joined without duplicates."""
from __future__ import annotations

import pytest
from PyQt6.QtGui import QKeySequence

import imagetagger.ui.shortcuts as shortcuts
from imagetagger.ui.shortcuts import native_shortcut_text, platform_key_sequence, platform_key_sequences


def _native(spec: str) -> str:
    return QKeySequence(spec).toString(QKeySequence.SequenceFormat.NativeText)


@pytest.fixture(params=[True, False], ids=["macos", "other"])
def on_macos(request, monkeypatch) -> bool:
    monkeypatch.setattr(shortcuts, "is_macos", lambda: request.param)
    return request.param


def test_is_macos_follows_sys_platform(monkeypatch):
    monkeypatch.setattr(shortcuts.sys, "platform", "darwin")
    assert shortcuts.is_macos() is True
    monkeypatch.setattr(shortcuts.sys, "platform", "linux")
    assert shortcuts.is_macos() is False


def test_platform_key_sequence_picks_macos_spec_only_on_macos(on_macos):
    expected = "Ctrl+Shift+Z" if on_macos else "Ctrl+Y"
    assert platform_key_sequence("Ctrl+Y", "Ctrl+Shift+Z") == QKeySequence(expected)


@pytest.mark.parametrize("macos", [None, ""])
def test_platform_key_sequence_without_macos_spec_uses_default(on_macos, macos):
    assert platform_key_sequence("Alt+F", macos) == QKeySequence("Alt+F")


def test_platform_key_sequences_picks_macos_specs_only_on_macos(on_macos):
    result = platform_key_sequences(["Delete", "Backspace"], ["Ctrl+Backspace"])
    expected = ["Ctrl+Backspace"] if on_macos else ["Delete", "Backspace"]
    assert result == [QKeySequence(spec) for spec in expected]


@pytest.mark.parametrize("macos", [None, []])
def test_platform_key_sequences_without_macos_specs_uses_default(on_macos, macos):
    assert platform_key_sequences(["F5", "Ctrl+R"], macos) == [QKeySequence("F5"), QKeySequence("Ctrl+R")]


def test_platform_key_sequences_returns_fresh_list(on_macos):
    first = platform_key_sequences(["F5"], ["F6"])
    first.append(QKeySequence("F7"))
    assert len(platform_key_sequences(["F5"], ["F6"])) == 1


def test_native_shortcut_text_single_sequence():
    assert native_shortcut_text(QKeySequence("Ctrl+O")) == _native("Ctrl+O")
    assert native_shortcut_text(QKeySequence("Ctrl+O")) != ""


def test_native_shortcut_text_joins_in_order():
    text = native_shortcut_text([QKeySequence("F5"), QKeySequence("Ctrl+R")])
    assert text == f"{_native('F5')} / {_native('Ctrl+R')}"


def test_native_shortcut_text_dedupes_and_skips_empty():
    sequences = [QKeySequence("Ctrl+O"), QKeySequence(), QKeySequence("Ctrl+O"), QKeySequence("Ctrl+P")]
    assert native_shortcut_text(sequences) == f"{_native('Ctrl+O')} / {_native('Ctrl+P')}"


@pytest.mark.parametrize("value", [[], [QKeySequence()], QKeySequence()])
def test_native_shortcut_text_empty(value):
    assert native_shortcut_text(value) == ""
