"""Folder loading, refresh, deletes and the image filter (DirectoryController + MainWindow)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from imagetagger import config as _config
from imagetagger.utils.sidecar import SidecarData, get_sidecar_json_path

_ROLE_BADGES = Qt.ItemDataRole.UserRole + 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_image(path: Path, hue: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = QImage(16, 16, QImage.Format.Format_RGB32)
    image.fill(QColor.fromHsv(hue % 360, 200, 200))
    assert image.save(str(path)), path


def _load(qtbot, window, folder: Path, restore_selection: Optional[Path] = None) -> None:
    window.load_directory(folder, restore_selection=restore_selection)
    _wait_loaded(qtbot, window)


def _wait_loaded(qtbot, window) -> None:
    qtbot.waitUntil(
        lambda: window._loader_thread is None
        and not window.directory_controller._directory_loading_active,
        timeout=10_000,
    )


def _refresh(qtbot, window) -> None:
    window.refresh_directory()
    _wait_loaded(qtbot, window)


def _names(window) -> list:
    return [record.image_path.name for record in window.records]


def _visible_names(window) -> list:
    return [
        window.records[row].image_path.name
        for row in range(window.list_widget.count())
        if not window.list_widget.item(row).isHidden()
    ]


def _current_name(window) -> Optional[str]:
    record = window._current_record()
    return None if record is None else record.image_path.name


def _tag_list(window) -> list:
    return [window.tag_list.item(i).text() for i in range(window.tag_list.count())]


def _assert_consistent(window) -> None:
    assert window._record_index_by_path == {r.image_path: i for i, r in enumerate(window.records)}
    assert window.list_widget.count() == len(window.records)
    for row, record in enumerate(window.records):
        title = window.list_widget.item(row).data(Qt.ItemDataRole.DisplayRole)
        assert title.split("\n")[0] == record.image_path.name
    record = window._current_record()
    if record is None:
        assert window.current_index == -1
    else:
        assert window.list_widget.currentRow() == window.current_index
        assert _tag_list(window) == window._parse_annotations_for_tag_list(record.text)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_reads_supported_images_sorted_with_texts(qtbot, make_main_window, tmp_path):
    folder = tmp_path / "mixed"
    _save_image(folder / "b.png", 10)
    (folder / "b.txt").write_text("tag_b", encoding="utf-8")
    _save_image(folder / "A.jpg", 20)
    (folder / "A.txt").write_text("tag_upper_a, second", encoding="utf-8")
    _save_image(folder / "c.BMP", 30)  # no .txt
    _save_image(folder / "sub" / "a2.png", 40)
    (folder / "sub" / "a2.txt").write_text("nested", encoding="utf-8")
    (folder / "notes.txt").write_text("orphan text", encoding="utf-8")
    (folder / "d.tiff").write_bytes(b"not a supported extension")
    (folder / "readme.md").write_text("x", encoding="utf-8")

    window = make_main_window()
    _load(qtbot, window, folder)

    # Case-insensitive order of the path relative to the folder; recursive.
    assert _names(window) == ["A.jpg", "b.png", "c.BMP", "a2.png"]
    assert [r.text for r in window.records] == ["tag_upper_a, second", "tag_b", "", "nested"]
    assert [r.text_path for r in window.records] == [
        folder / "A.txt", folder / "b.txt", folder / "c.txt", folder / "sub" / "a2.txt",
    ]
    assert not (folder / "c.txt").exists()  # loading never creates text files
    assert window.records[0]._image_size == (16, 16)
    _assert_consistent(window)

    # The first image is selected and shown.
    assert window.current_index == 0
    assert _tag_list(window) == ["tag_upper_a", "second"]

    assert window.tag_counts == {"tag_upper_a": 1, "second": 1, "tag_b": 1, "nested": 1}
    assert window.known_tags == {"tag_upper_a", "second", "tag_b", "nested"}
    assert window._root_directory == folder
    assert window.windowTitle() == "ImageTagger - mixed"
    assert _config.load()["last_open_directory"] == str(folder)
    assert window.list_widget.isEnabled() and window.tag_input.isEnabled()


def test_load_empty_folder(qtbot, make_main_window, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    (folder / "only.txt").write_text("text", encoding="utf-8")
    window = make_main_window()
    _load(qtbot, window, folder)

    assert window.records == []
    assert window.current_index == -1
    _assert_consistent(window)
    assert window.statusBar().currentMessage() == "No supported images found in selected folder"


def test_load_passes_sidecar_badges_and_validated_state(qtbot, make_main_window, image_folder):
    folder = image_folder(
        {"a": "tag_a", "b": "tag_b", "c": "tag_c"},
        sidecars={
            "a": SidecarData(validated="2026-01-02T03:04:05+00:00", validated_by="user"),
            "b": SidecarData(vision_caption="a caption", ai_find_matches=["red"]),
        },
    )
    window = make_main_window(folder)

    badges = [window.list_widget.item(row).data(_ROLE_BADGES) for row in range(3)]
    assert badges == [frozenset({"✅"}), frozenset({"✨", "🔍"}), frozenset()]

    a, b, c = window.records
    # Loader-provided caches: no main-thread sidecar reads needed.
    assert a._sidecar_validated == "2026-01-02T03:04:05+00:00"
    assert b._sidecar_validated is None
    assert c._sidecar_validated is None
    assert a._sidecar_has_pending_fixup is False
    assert b._sidecar_has_pending_fixup is True
    assert c._sidecar_has_pending_fixup is False


def test_load_with_name_collision_warns_and_keeps_previous_folder(
    qtbot, make_main_window, image_folder, tmp_path, message_boxes
):
    window = make_main_window(image_folder({"a": "tag_a", "b": "tag_b"}))
    first_root = window._root_directory
    bad = tmp_path / "collide"
    _save_image(bad / "x.png")
    _save_image(bad / "x.jpg")

    _load(qtbot, window, bad)

    assert "Duplicate image names detected" in message_boxes.titles("warning")
    assert _names(window) == ["a.png", "b.png"]
    assert window._root_directory == first_root
    _assert_consistent(window)
    assert window.list_widget.isEnabled()


def test_loading_a_second_folder_replaces_everything(qtbot, main_window, image_folder):
    window = main_window
    window.list_widget.setCurrentRow(2)
    assert _current_name(window) == "c.png"
    old_paths = set(window._record_index_by_path)

    other = image_folder({"x": "tag_x, shared", "y": "shared"}, folder_name="other")
    _load(qtbot, window, other)

    assert _names(window) == ["x.png", "y.png"]
    assert all(other in r.image_path.parents for r in window.records)
    assert not (old_paths & set(window._record_index_by_path))
    assert window.current_index == 0
    assert _current_name(window) == "x.png"
    assert window.tag_counts == {"tag_x": 1, "shared": 2}
    assert window.known_tags == {"tag_x", "shared"}
    assert window._root_directory == other
    assert window.windowTitle() == "ImageTagger - other"
    _assert_consistent(window)


def test_open_folder_uses_dialog_and_last_directory(qtbot, make_main_window, image_folder, monkeypatch):
    folder = image_folder({"a": "tag_a"})
    window = make_main_window()
    window._cfg["last_open_directory"] = "/start/here"
    asked = []

    def fake_dialog(parent, title, start_dir):
        asked.append(start_dir)
        return str(folder)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(fake_dialog))
    window.open_folder()
    _wait_loaded(qtbot, window)

    assert asked == ["/start/here"]
    assert _names(window) == ["a.png"]

    # Cancelling the dialog leaves everything alone.
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a: ""))
    window.open_folder()
    assert window._loader_thread is None
    assert _names(window) == ["a.png"]


def test_last_selected_image_is_restored_on_startup(qtbot, make_main_window, image_folder):
    folder = image_folder({"a": "tag_a", "b": "tag_b", "c": "tag_c"})
    _config.save({"last_open_directory": str(folder), "last_selected_image": str(folder / "b.png")})

    window = make_main_window()  # _apply_config starts the load
    qtbot.waitUntil(lambda: window._loader_thread is None and len(window.records) == 3, timeout=10_000)

    assert _current_name(window) == "b.png"
    assert _tag_list(window) == ["tag_b"]
    _assert_consistent(window)


def test_last_selected_image_outside_folder_is_ignored(qtbot, make_main_window, image_folder, tmp_path):
    folder = image_folder({"a": "tag_a", "b": "tag_b"})
    outside = tmp_path / "elsewhere" / "b.png"
    _save_image(outside)
    _config.save({"last_open_directory": str(folder), "last_selected_image": str(outside)})

    window = make_main_window()
    qtbot.waitUntil(lambda: window._loader_thread is None and len(window.records) == 2, timeout=10_000)

    assert _current_name(window) == "a.png"


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


def test_refresh_restores_selection_and_rereads_texts(qtbot, main_window):
    window = main_window
    folder = window._root_directory
    window.list_widget.setCurrentRow(1)
    assert _current_name(window) == "b.png"

    (folder / "b.txt").write_text("changed_b", encoding="utf-8")
    _save_image(folder / "0first.png")
    (folder / "0first.txt").write_text("new_one", encoding="utf-8")
    (folder / "c.png").unlink()
    (folder / "c.txt").unlink()

    _refresh(qtbot, window)

    assert _names(window) == ["0first.png", "a.png", "b.png"]
    assert _current_name(window) == "b.png"
    assert window.current_index == 2
    assert _tag_list(window) == ["changed_b"]
    assert window.tag_counts == {"new_one": 1, "tag_a": 1, "changed_b": 1}
    _assert_consistent(window)


def test_refresh_finds_a_sidecar_another_program_created(qtbot, main_window):
    window = main_window
    record = window.records[0]
    assert not record.has_pending_fixup  # caches "no sidecar"
    get_sidecar_json_path(record.image_path).write_text(
        '{"description": "", "reasoning": "", "fixup_tags": ["x"]}', encoding="utf-8"
    )

    _refresh(qtbot, window)

    assert window.records[0].has_pending_fixup


def test_refresh_falls_back_to_first_row_when_selection_is_gone(qtbot, main_window):
    window = main_window
    folder = window._root_directory
    window.list_widget.setCurrentRow(2)
    (folder / "c.png").unlink()

    _refresh(qtbot, window)

    assert _names(window) == ["a.png", "b.png"]
    assert _current_name(window) == "a.png"
    _assert_consistent(window)


def test_refresh_falls_back_to_first_visible_row_when_selection_is_filtered_out(qtbot, main_window):
    window = main_window
    window.list_widget.setCurrentRow(0)
    window.filter_input.setText('!"tag_a"')
    assert _current_name(window) == "b.png"
    # After the refresh b matches no more; a hidden row must not be selected.
    (window._root_directory / "b.txt").write_text("tag_a", encoding="utf-8")

    _refresh(qtbot, window)

    assert _visible_names(window) == ["c.png"]
    assert _current_name(window) == "c.png"
    _assert_consistent(window)


def test_refresh_without_folder_informs(make_main_window, message_boxes):
    window = make_main_window()
    window.refresh_directory()
    assert message_boxes.titles("information") == ["No folder selected"]
    assert window._loader_thread is None


# ---------------------------------------------------------------------------
# Deletes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current, deleted, expected_names, expected_current",
    [
        ("a", "a", ["b.png", "c.png"], "b.png"),   # first, current
        ("b", "b", ["a.png", "c.png"], "c.png"),   # middle, current
        ("c", "c", ["a.png", "b.png"], "b.png"),   # last, current
        ("c", "a", ["b.png", "c.png"], "c.png"),   # above the current row
        ("a", "c", ["a.png", "b.png"], "a.png"),   # below the current row
        ("a", "b", ["a.png", "c.png"], "a.png"),
    ],
)
def test_delete_image(main_window, current, deleted, expected_names, expected_current, message_boxes):
    window = main_window
    folder = window._root_directory
    window.list_widget.setCurrentRow(["a", "b", "c"].index(current))
    get_sidecar_json_path(folder / f"{deleted}.png").write_text("{}", encoding="utf-8")

    ok, _ = window._delete_image_and_related_files(folder / f"{deleted}.png")

    assert ok
    assert message_boxes.titles("question") == ["Delete file"]
    assert not (folder / f"{deleted}.png").exists()
    assert not (folder / f"{deleted}.txt").exists()
    assert not (folder / f"{deleted}.json").exists()
    assert _names(window) == expected_names
    assert _current_name(window) == expected_current
    assert f"tag_{deleted}" not in window.known_tags
    assert window.tag_counts == {f"tag_{n[0]}": 1 for n in expected_names}
    _assert_consistent(window)


def test_delete_current_from_context_menu(main_window):
    window = main_window
    window.list_widget.setCurrentRow(1)
    window._delete_current_image_from_context_menu()
    assert _names(window) == ["a.png", "c.png"]
    assert _current_name(window) == "c.png"
    _assert_consistent(window)


def test_delete_only_image(qtbot, make_main_window, image_folder):
    window = make_main_window(image_folder({"a": "tag_a"}))
    folder = window._root_directory

    ok, fixups_left = window._delete_image_and_related_files(folder / "a.png")

    assert (ok, fixups_left) == (True, False)
    assert window.records == []
    assert window.current_index == -1
    assert window.tag_list.count() == 0
    assert window.known_tags == set()
    assert window.image_label.text() == "No image selected"
    _assert_consistent(window)


def test_delete_all_images_one_by_one(main_window):
    window = main_window
    folder = window._root_directory
    for name in ("b", "a", "c"):
        assert window._delete_image_and_related_files(folder / f"{name}.png", confirm=False)[0]
        _assert_consistent(window)
    assert window.records == []


def test_delete_declined_changes_nothing(main_window, monkeypatch, message_boxes):
    window = main_window
    folder = window._root_directory
    window.list_widget.setCurrentRow(1)
    asked = []

    def answer_no(parent, title, text, *args, **kwargs):
        asked.append(title)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer_no))
    ok, _ = window._delete_image_and_related_files(folder / "b.png")

    assert not ok
    assert asked == ["Delete file"]
    assert (folder / "b.png").exists() and (folder / "b.txt").exists()
    assert _names(window) == ["a.png", "b.png", "c.png"]
    assert _current_name(window) == "b.png"
    _assert_consistent(window)


def test_delete_without_confirmation_when_disabled(main_window, message_boxes):
    window = main_window
    window._cfg["confirm_on_delete"] = False
    ok, _ = window._delete_image_and_related_files(window._root_directory / "a.png")
    assert ok
    assert message_boxes.titles("question") == []


def test_delete_unknown_path_is_a_no_op(main_window, tmp_path):
    ok, _ = main_window._delete_image_and_related_files(tmp_path / "nope.png")
    assert not ok
    assert _names(main_window) == ["a.png", "b.png", "c.png"]


def test_delete_current_skips_rows_hidden_by_filter(main_window):
    window = main_window
    folder = window._root_directory
    window.filter_input.setText('!"tag_b"')
    window.list_widget.setCurrentRow(0)

    assert window._delete_image_and_related_files(folder / "a.png", confirm=False)[0]

    assert _names(window) == ["b.png", "c.png"]
    assert _visible_names(window) == ["c.png"]
    assert _current_name(window) == "c.png"
    _assert_consistent(window)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_hides_rows_and_moves_selection(main_window):
    window = main_window
    window.list_widget.setCurrentRow(0)

    window.filter_input.setText('"tag_b" | "TAG_C"')
    assert _visible_names(window) == ["b.png", "c.png"]
    assert _current_name(window) == "b.png"  # a is hidden, first visible row taken

    window.filter_input.setText("'tag_'")
    assert _visible_names(window) == ["a.png", "b.png", "c.png"]
    assert _current_name(window) == "b.png"  # still visible, kept

    window.filter_input.setText("")
    assert _visible_names(window) == ["a.png", "b.png", "c.png"]
    _assert_consistent(window)


def test_filter_with_no_match_clears_selection(main_window):
    window = main_window
    window.filter_input.setText('"missing"')
    assert _visible_names(window) == []
    assert window.current_index == -1
    assert window._selected_record_indexes() == []
    assert "no images match filter" in window.statusBar().currentMessage()


def test_invalid_filter_shows_everything(main_window):
    window = main_window
    window.filter_input.setText('"tag_a"')
    window.filter_input.setText('("tag_a"')
    assert _visible_names(window) == ["a.png", "b.png", "c.png"]
    assert window.statusBar().currentMessage().startswith("Invalid filter")


def test_selected_record_indexes_ignores_hidden_rows(main_window):
    window = main_window
    window.filter_input.setText('!"tag_b"')
    window.list_widget.selectAll()  # Qt also selects the hidden row
    assert window._selected_record_indexes() == [0, 2]

    window.list_widget.clearSelection()
    window._select_all_images()
    assert window._selected_record_indexes() == [0, 2]
    assert not window.list_widget.item(1).isSelected()


def test_selected_record_indexes_falls_back_to_current(main_window):
    window = main_window
    window.list_widget.setCurrentRow(2)
    window.list_widget.clearSelection()
    assert window._selected_record_indexes() == [2]


def test_untagged_and_validated_filters(qtbot, make_main_window, image_folder):
    folder = image_folder(
        {"a": "tag_a", "b": "tag_b", "c": "tag_c"},
        sidecars={"c": SidecarData(validated="2026-01-01T00:00:00+00:00")},
    )
    (folder / "b.txt").unlink()
    window = make_main_window(folder)

    window.filter_input.setText("untagged")
    assert _visible_names(window) == ["b.png"]
    window.filter_input.setText("validated")
    assert _visible_names(window) == ["c.png"]
    window.filter_input.setText("!validated & !untagged")
    assert _visible_names(window) == ["a.png"]


def test_filter_applies_while_loading(qtbot, main_window, image_folder):
    window = main_window
    window.filter_input.setText('"shared"')
    other = image_folder({"x": "shared", "y": "tag_y", "z": "shared, tag_z"}, folder_name="other")

    _load(qtbot, window, other)

    assert _names(window) == ["x.png", "y.png", "z.png"]
    assert _visible_names(window) == ["x.png", "z.png"]
    assert _current_name(window) == "x.png"
    _assert_consistent(window)
