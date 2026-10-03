"""atomic_write_text: content, newlines, and no debris after success or failure."""
from __future__ import annotations

import os

import pytest

from imagetagger.utils import io_utils
from imagetagger.utils.io_utils import atomic_write_text


def _names(folder) -> list:
    return sorted(p.name for p in folder.iterdir())


def test_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_replaces_existing_content(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("old content that is longer", encoding="utf-8")
    atomic_write_text(path, "new")
    assert path.read_text(encoding="utf-8") == "new"


@pytest.mark.parametrize(
    "content",
    ["line1\nline2\n", "line1\r\nline2\r\n", "mixed\r\nand\nbare\rcr", "", "no newline"],
    ids=["lf", "crlf", "mixed", "empty", "no-newline"],
)
def test_newlines_are_written_exactly(tmp_path, content):
    path = tmp_path / "a.txt"
    atomic_write_text(path, content)
    assert path.read_bytes() == content.encode("utf-8")


def test_unicode_round_trip(tmp_path):
    path = tmp_path / "a.txt"
    content = "café, 日本語, emoji \U0001F600, ümlaut"
    atomic_write_text(path, content)
    assert path.read_bytes() == content.encode("utf-8")


def test_honours_encoding(tmp_path):
    path = tmp_path / "a.txt"
    atomic_write_text(path, "café", encoding="latin-1")
    assert path.read_bytes() == "café".encode("latin-1")


def test_no_temp_file_left_after_success(tmp_path):
    path = tmp_path / "a.txt"
    atomic_write_text(path, "one")
    atomic_write_text(path, "two")
    assert _names(tmp_path) == ["a.txt"]


def test_failure_while_writing_leaves_original_and_no_temp_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(path, "not ascii: é", encoding="ascii")
    assert path.read_text(encoding="utf-8") == "original"
    assert _names(tmp_path) == ["a.txt"]


def test_failure_in_fsync_leaves_original_and_no_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"
    path.write_text("original", encoding="utf-8")

    def failing_fsync(fd):
        raise OSError(5, "Input/output error")

    with monkeypatch.context() as patch, pytest.raises(OSError):
        patch.setattr(io_utils.os, "fsync", failing_fsync)
        atomic_write_text(path, "new")
    assert path.read_text(encoding="utf-8") == "original"
    assert _names(tmp_path) == ["a.txt"]


def test_failure_in_replace_leaves_original_and_no_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"
    path.write_text("original", encoding="utf-8")
    calls = []

    def failing_replace(src, dst):
        calls.append((src, dst))
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch, pytest.raises(OSError, match="No space"):
        patch.setattr(io_utils.os, "replace", failing_replace)
        atomic_write_text(path, "new")

    assert len(calls) == 1
    assert calls[0][1] == path
    assert path.read_text(encoding="utf-8") == "original"
    assert _names(tmp_path) == ["a.txt"]


def test_failure_when_target_did_not_exist_leaves_nothing(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"

    def failing_replace(src, dst):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch, pytest.raises(OSError):
        patch.setattr(io_utils.os, "replace", failing_replace)
        atomic_write_text(path, "new")
    assert _names(tmp_path) == []


def test_temp_file_is_created_next_to_the_target(tmp_path, monkeypatch):
    """Same directory, so os.replace is a rename on one filesystem."""
    path = tmp_path / "sub" / "a.txt"
    seen = []
    real_replace = os.replace

    def spy_replace(src, dst):
        seen.append(src)
        real_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(io_utils.os, "replace", spy_replace)
        atomic_write_text(path, "x")
    (src,) = seen
    src = os.fspath(src)
    assert os.path.dirname(src) == str(path.parent)
    assert os.path.basename(src).startswith(".a.txt.")
    assert src.endswith(".tmp")
