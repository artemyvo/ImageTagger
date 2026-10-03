"""Tag editing on the current image and global tag operations (TagController)."""
from __future__ import annotations

from collections import Counter

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QInputDialog, QMessageBox

DESCRIPTION = "A small red square on a plain background."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tag_list(window) -> list:
    return [window.tag_list.item(i).text() for i in range(window.tag_list.count())]


def _add(window, text: str) -> None:
    window.tag_input.setText(text)
    window.tag_input.returnPressed.emit()


def _read(record) -> str:
    return record.text_path.read_text(encoding="utf-8")


def _rebuilt_counts(window) -> Counter:
    counts: Counter = Counter()
    for record in window.records:
        counts.update(window._parse_tags(record.text))
    return counts


def _assert_counts_match_rebuild(window) -> None:
    expected = _rebuilt_counts(window)
    assert dict(window.tag_counts) == dict(expected)
    assert window.known_tags == set(expected)
    assert window.tag_suggestions_model.stringList() == sorted(
        expected, key=lambda tag: (-expected[tag], tag.lower(), tag)
    )


def _known_tags_rows(window) -> list:
    window.tag_controller._do_refresh_known_tags_list()  # skip the debounce
    return [window.known_tags_list.item(i).text() for i in range(window.known_tags_list.count())]


def _select_known_tags(window, *tags: str) -> None:
    window.tag_controller._do_refresh_known_tags_list()
    window.known_tags_list.clearSelection()
    for i in range(window.known_tags_list.count()):
        item = window.known_tags_list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) in tags:
            item.setSelected(True)


@pytest.fixture
def confirm_answer(monkeypatch):
    """Answer the QMessageBox(...).exec() confirmations of the global tag operations."""
    state = {"answer": QMessageBox.StandardButton.Yes, "asked": []}

    def fake_exec(box):
        state["asked"].append(box.text())
        return state["answer"]

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    return state


def _wait_status(qtbot, window, prefix: str) -> None:
    qtbot.waitUntil(lambda: window.statusBar().currentMessage().startswith(prefix), timeout=5_000)


@pytest.fixture
def tagged_window(make_main_window, image_folder):
    """a: shared, tag_a, extra   b: tag_b, shared   c: tag_c."""
    return make_main_window(
        image_folder({"a": "shared, tag_a, extra", "b": "tag_b, shared", "c": "tag_c"})
    )


# ---------------------------------------------------------------------------
# Adding
# ---------------------------------------------------------------------------


def test_add_tag_from_input_saves_and_updates_counts(qtbot, main_window, drain_writes):
    window = main_window
    record = window.records[0]

    _add(window, "  New   Tag. ")

    assert _tag_list(window) == ["tag_a", "new tag"]
    assert record.text == "tag_a, new tag"
    drain_writes()
    assert _read(record) == "tag_a, new tag"
    assert window.tag_input.text() == ""
    assert window.tag_counts["new tag"] == 1
    assert "new tag" in window.tag_suggestions_model.stringList()
    title = window.list_widget.item(0).data(Qt.ItemDataRole.DisplayRole)
    assert title == "a.png\ntag_a, new tag"
    _assert_counts_match_rebuild(window)


@pytest.mark.parametrize("text", ["TAG_A", " tag_a. ", "(Tag_A)"])
def test_add_duplicate_is_rejected_case_insensitively(main_window, text):
    window = main_window
    record = window.records[0]

    _add(window, text)

    assert _tag_list(window) == ["tag_a"]
    assert record.text == "tag_a"
    assert _read(record) == "tag_a"
    assert window.statusBar().currentMessage() == "Tag already exists: tag_a"
    assert window.tag_input.text() == text  # kept (selected) for correction


@pytest.mark.parametrize("text", ["", "   ", " , . ( ) "])
def test_add_blank_tag_is_ignored(main_window, text):
    window = main_window
    _add(window, text)
    assert _tag_list(window) == ["tag_a"]
    assert window.records[0].text == "tag_a"


def test_add_tag_without_current_image_is_ignored(main_window):
    window = main_window
    window.filter_input.setText('"nothing"')
    assert window.current_index == -1
    _add(window, "orphan")
    assert "orphan" not in window.known_tags
    assert [r.text for r in window.records] == ["tag_a", "tag_b", "tag_c"]


def test_add_tag_to_image_without_text_file_creates_it(qtbot, main_window, drain_writes):
    window = main_window
    record = window.records[1]
    record.text_path.unlink()
    record.text = ""
    window.list_widget.setCurrentRow(1)
    assert _tag_list(window) == []

    _add(window, "first")
    drain_writes()

    assert _read(record) == "first"


def test_add_keeps_description_first_and_unchanged(qtbot, make_main_window, image_folder, drain_writes):
    window = make_main_window(image_folder({"a": f"tag_a, {DESCRIPTION}"}))
    record = window.records[0]
    assert _tag_list(window) == [DESCRIPTION, "tag_a"]

    _add(window, "new")
    drain_writes()

    assert _read(record) == f"{DESCRIPTION}, tag_a, new"
    _assert_counts_match_rebuild(window)


def test_selecting_an_image_does_not_rewrite_its_file(qtbot, make_main_window, image_folder, drain_writes):
    """Loading a record into the tag list reorders/dedupes for display only."""
    original = f"tag_a, {DESCRIPTION}, TAG_A"
    window = make_main_window(image_folder({"a": original, "b": "tag_b"}))
    window.list_widget.setCurrentRow(1)
    window.list_widget.setCurrentRow(0)
    drain_writes()
    assert _tag_list(window) == [DESCRIPTION, "tag_a"]
    assert _read(window.records[0]) == original


# ---------------------------------------------------------------------------
# Removing, editing, reordering
# ---------------------------------------------------------------------------


def test_remove_selected_tags(qtbot, tagged_window, drain_writes):
    window = tagged_window
    record = window.records[0]
    window.tag_list.item(0).setSelected(True)
    window.tag_list.item(2).setSelected(True)

    window.remove_tag_action.trigger()
    drain_writes()

    assert _tag_list(window) == ["tag_a"]
    assert _read(record) == "tag_a"
    assert window.tag_list.currentRow() == 0
    assert window.tag_counts["shared"] == 1
    assert "extra" not in window.known_tags
    _assert_counts_match_rebuild(window)


def test_remove_last_tag_leaves_empty_file(qtbot, main_window, drain_writes):
    window = main_window
    window.tag_list.item(0).setSelected(True)
    window._remove_selected_tags()
    drain_writes()
    assert _tag_list(window) == []
    assert _read(window.records[0]) == ""
    assert "tag_a" not in window.known_tags
    _assert_counts_match_rebuild(window)


def test_remove_with_nothing_selected_is_a_no_op(tagged_window):
    window = tagged_window
    window.tag_list.clearSelection()
    window._remove_selected_tags()
    assert _tag_list(window) == ["shared", "tag_a", "extra"]


def test_edit_tag_in_place(qtbot, tagged_window, drain_writes):
    window = tagged_window
    record = window.records[0]

    window.tag_list.item(1).setText("  renamed  ")  # itemChanged -> _on_tag_item_changed
    drain_writes()

    assert _tag_list(window) == ["shared", "renamed", "extra"]
    assert _read(record) == "shared, renamed, extra"
    assert "tag_a" not in window.known_tags
    assert window.tag_counts["renamed"] == 1
    _assert_counts_match_rebuild(window)


def test_editing_a_tag_to_blank_removes_it(qtbot, tagged_window, drain_writes):
    window = tagged_window
    window.tag_list.item(0).setText("   ")
    drain_writes()
    assert _tag_list(window) == ["tag_a", "extra"]
    assert _read(window.records[0]) == "tag_a, extra"
    assert window.tag_counts["shared"] == 1
    _assert_counts_match_rebuild(window)


def test_editing_a_tag_into_an_existing_one_does_not_duplicate_it(tagged_window, drain_writes):
    window = tagged_window
    window.tag_list.item(2).setText("tag_a")  # "extra" -> "tag_a", already on the image
    drain_writes()
    assert window._parse_tags(window.records[0].text).count("tag_a") == 1


def test_case_only_edit_of_description_is_saved(qtbot, make_main_window, image_folder, drain_writes):
    window = make_main_window(image_folder({"a": f"{DESCRIPTION}, tag_a"}))
    edited = "A small RED square on a plain background."
    window.tag_list.item(0).setText(edited)
    drain_writes()
    assert window.records[0].text == f"{edited}, tag_a"
    assert _read(window.records[0]) == f"{edited}, tag_a"


def test_reorder_saves_new_order(qtbot, tagged_window, drain_writes):
    window = tagged_window
    record = window.records[0]
    counts_before = dict(window.tag_counts)

    # What a drag and drop does: move the row, then TagListWidget emits tags_reordered.
    item = window.tag_list.takeItem(2)
    window.tag_list.insertItem(0, item)
    window.tag_list.tags_reordered.emit()
    drain_writes()

    assert _tag_list(window) == ["extra", "shared", "tag_a"]
    assert _read(record) == "extra, shared, tag_a"
    assert dict(window.tag_counts) == counts_before


def test_tag_edits_stay_on_their_own_image(qtbot, tagged_window, drain_writes):
    window = tagged_window
    _add(window, "for_a")
    window.list_widget.setCurrentRow(1)
    assert _tag_list(window) == ["tag_b", "shared"]
    _add(window, "for_b")
    window.list_widget.setCurrentRow(0)
    drain_writes()

    assert _tag_list(window) == ["shared", "tag_a", "extra", "for_a"]
    assert _read(window.records[0]) == "shared, tag_a, extra, for_a"
    assert _read(window.records[1]) == "tag_b, shared, for_b"
    assert _read(window.records[2]) == "tag_c"


# ---------------------------------------------------------------------------
# Known tags, counts and completions
# ---------------------------------------------------------------------------


def test_counts_and_completions_after_load(tagged_window):
    window = tagged_window
    assert dict(window.tag_counts) == {"shared": 2, "tag_a": 1, "extra": 1, "tag_b": 1, "tag_c": 1}
    # Most used first, then alphabetical.
    assert window.tag_suggestions_model.stringList() == ["shared", "extra", "tag_a", "tag_b", "tag_c"]
    _assert_counts_match_rebuild(window)


def test_incremental_counts_match_full_rebuild(qtbot, tagged_window, drain_writes):
    window = tagged_window
    _add(window, "tag_b")                       # a now shares tag_b
    window.tag_list.item(0).setText("renamed")  # shared -> renamed on a
    window.list_widget.setCurrentRow(1)
    window.tag_list.item(1).setSelected(True)   # remove shared from b
    window._remove_selected_tags()
    window._set_tags_for_image_path(window.records[2].image_path, ["tag_c", "tag_b"])
    drain_writes()

    assert "shared" not in window.known_tags
    assert window.tag_counts["tag_b"] == 3
    _assert_counts_match_rebuild(window)


def test_known_tags_list_shows_counts_and_filters(qtbot, tagged_window):
    window = tagged_window
    assert _known_tags_rows(window) == [
        "shared (2)", "extra (1)", "tag_a (1)", "tag_b (1)", "tag_c (1)",
    ]

    window.known_tags_filter.setText("TAG_")
    assert _known_tags_rows(window) == ["tag_a (1)", "tag_b (1)", "tag_c (1)"]

    window.known_tags_filter.setText("")
    _add(window, "tag_c")
    # The debounced rebuild catches up by itself.
    qtbot.waitUntil(
        lambda: [window.known_tags_list.item(i).text() for i in range(window.known_tags_list.count())]
        == ["shared (2)", "tag_c (2)", "extra (1)", "tag_a (1)", "tag_b (1)"],
        timeout=2_000,
    )


# ---------------------------------------------------------------------------
# Global tag delete
# ---------------------------------------------------------------------------


def test_global_delete_via_delete_key(qtbot, tagged_window, confirm_answer, drain_writes):
    window = tagged_window
    _select_known_tags(window, "shared")

    qtbot.keyClick(window.known_tags_list, Qt.Key.Key_Delete)
    _wait_status(qtbot, window, "Removed")

    assert confirm_answer["asked"] == ['Remove "shared" from 2 image(s)?']
    a, b, c = window.records
    assert (a.text, b.text, c.text) == ("tag_a, extra", "tag_b", "tag_c")
    assert (_read(a), _read(b), _read(c)) == ("tag_a, extra", "tag_b", "tag_c")
    assert _tag_list(window) == ["tag_a", "extra"]  # current image refreshed
    assert "shared" not in window.known_tags
    _assert_counts_match_rebuild(window)


def test_global_delete_several_tags(qtbot, tagged_window, confirm_answer):
    window = tagged_window
    window.known_tags_list.delete_requested.emit(["extra", "tag_c"])
    _wait_status(qtbot, window, "Removed")

    assert confirm_answer["asked"] == ["Remove 2 tags from 2 image(s)?"]
    assert [_read(r) for r in window.records] == ["shared, tag_a", "tag_b, shared", ""]
    _assert_counts_match_rebuild(window)


def test_global_delete_declined_changes_nothing(qtbot, tagged_window, confirm_answer):
    window = tagged_window
    confirm_answer["answer"] = QMessageBox.StandardButton.No
    window.known_tags_list.delete_requested.emit(["shared"])
    qtbot.wait(20)

    assert confirm_answer["asked"] == ['Remove "shared" from 2 image(s)?']
    assert [r.text for r in window.records] == ["shared, tag_a, extra", "tag_b, shared", "tag_c"]
    assert [_read(r) for r in window.records] == ["shared, tag_a, extra", "tag_b, shared", "tag_c"]
    assert window.tag_counts["shared"] == 2


def test_global_delete_of_unused_tag_asks_nothing(tagged_window, confirm_answer):
    tagged_window.known_tags_list.delete_requested.emit(["nowhere"])
    assert confirm_answer["asked"] == []


def test_global_delete_keeps_descriptions(qtbot, make_main_window, image_folder, confirm_answer):
    window = make_main_window(image_folder({"a": f"{DESCRIPTION}, shared", "b": "shared, tag_b"}))
    window.known_tags_list.delete_requested.emit(["shared"])
    _wait_status(qtbot, window, "Removed")
    assert _read(window.records[0]) == DESCRIPTION


def test_global_delete_refreshes_list_previews(qtbot, tagged_window, confirm_answer):
    window = tagged_window
    window.known_tags_list.delete_requested.emit(["shared"])
    _wait_status(qtbot, window, "Removed")
    assert window.list_widget.item(1).data(Qt.ItemDataRole.DisplayRole) == "b.png\ntag_b"


def test_global_delete_is_not_undone_by_an_earlier_queued_save(
    qtbot, tagged_window, confirm_answer, stalled_write_queue, drain_writes
):
    window = tagged_window
    b = window.records[1]
    # An auto-save for b (still with "shared") is queued behind a slow disk.
    window._set_tags_for_image_path(b.image_path, ["tag_b", "shared", "more"])

    window.known_tags_list.delete_requested.emit(["shared"])
    assert b.text == "tag_b, more"  # memory changes at once; the file waits its turn
    qtbot.wait(200)  # time for any write that skips the queue to land
    assert _read(b) == "tag_b, shared"
    stalled_write_queue.set()
    _wait_status(qtbot, window, "Removed")
    drain_writes()

    assert b.text == "tag_b, more"
    assert _read(b) == "tag_b, more"


# ---------------------------------------------------------------------------
# Global rename and bump
# ---------------------------------------------------------------------------


def test_global_rename_merges_into_existing_tag(qtbot, tagged_window, confirm_answer, monkeypatch):
    window = tagged_window
    _select_known_tags(window, "extra")
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Shared", True)))

    window.rename_tag_button.click()
    _wait_status(qtbot, window, "Renamed")

    assert len(confirm_answer["asked"]) == 1 and "already exists" in confirm_answer["asked"][0]
    assert _read(window.records[0]) == "shared, tag_a"
    assert _tag_list(window) == ["shared", "tag_a"]
    _assert_counts_match_rebuild(window)


def test_global_rename_cancelled(tagged_window, monkeypatch):
    window = tagged_window
    _select_known_tags(window, "extra")
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("new", False)))
    window._rename_selected_tag()
    assert window.records[0].text == "shared, tag_a, extra"


def test_global_bump_moves_tag_first(qtbot, tagged_window):
    window = tagged_window
    window.list_widget.setCurrentRow(1)  # b: tag_b, shared
    _select_known_tags(window, "shared")

    window.bump_tag_button.click()
    _wait_status(qtbot, window, "Bumped")

    assert _read(window.records[0]) == "shared, tag_a, extra"  # already first, untouched
    assert _read(window.records[1]) == "shared, tag_b"
    assert _tag_list(window) == ["shared", "tag_b"]
    assert window.tag_list.currentItem().text() == "shared"


def test_global_rename_keeps_descriptions_and_refreshes_rows(
    qtbot, make_main_window, image_folder, confirm_answer, monkeypatch, drain_writes
):
    window = make_main_window(image_folder({"a": f"{DESCRIPTION}, old", "b": "old, tag_b"}))
    _select_known_tags(window, "old")
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("new", True)))

    window.rename_tag_button.click()
    _wait_status(qtbot, window, "Renamed")
    drain_writes()

    assert _read(window.records[0]) == f"{DESCRIPTION}, new"
    assert _read(window.records[1]) == "new, tag_b"
    assert window.list_widget.item(1).data(Qt.ItemDataRole.DisplayRole) == "b.png\nnew, tag_b"


def test_global_bump_refreshes_rows(qtbot, tagged_window):
    window = tagged_window
    _select_known_tags(window, "shared")
    window.bump_tag_button.click()
    _wait_status(qtbot, window, "Bumped")
    assert window.list_widget.item(1).data(Qt.ItemDataRole.DisplayRole) == "b.png\nshared, tag_b"


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("NEW Tag", "shared, tag_a, new tag"),  # normalized like the tag input
        ("TAG_A", "shared, tag_a"),  # same as another tag once normalized
    ],
)
def test_edited_tag_is_normalized(tagged_window, drain_writes, typed, expected):
    window = tagged_window
    window.tag_list.item(2).setText(typed)  # "extra"
    drain_writes()
    assert _read(window.records[0]) == expected
