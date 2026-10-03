"""The Fixup (merge) dialog flow, driven through MainWindow.open_fixup_dialog.

FixupDialog.exec is replaced by a scripted interaction: each opened dialog
runs the next script step, which calls the dialog's own slots (the ones its
buttons are connected to), and exec returns the dialog's result code without
ever blocking.
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

import pytest
from PyQt6.QtWidgets import QDialog

from imagetagger.ui.main_window import _ROLE_BADGES
from imagetagger.ui.merge_dialog import FixupDialog
from imagetagger.utils.sidecar import SidecarData, get_sidecar_json_path, read_sidecar_data

_UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _fixup(*tags: str, **fields) -> SidecarData:
    return SidecarData(fixup_issues="issue", fixup_tags=list(tags), **fields)


class Driver:
    """Runs one script step per opened dialog and records what each dialog showed."""

    def __init__(self) -> None:
        self.steps: List[Callable[[FixupDialog], Optional[int]]] = []
        self.seen: List[dict] = []

    def script(self, *steps: Callable[[FixupDialog], Optional[int]]) -> "Driver":
        self.steps.extend(steps)
        return self

    def exec(self, dialog: FixupDialog) -> int:
        panel = dialog._comparison_panel
        self.seen.append(
            {
                "title": dialog.windowTitle(),
                "image": dialog._image_path.stem,
                "current": panel.selected_annotations(),
                "proposed": panel._current_texts(panel.right_list),
                "prev": dialog.prev_button.isEnabled(),
                "next": dialog.next_button.isEnabled(),
            }
        )
        assert len(self.seen) <= 10, "dialog reopened too often"
        if self.steps:
            code = self.steps.pop(0)(dialog)
            if code is not None:
                dialog.done(code)
        return dialog.result()

    @property
    def images(self) -> List[str]:
        return [entry["image"] for entry in self.seen]


@pytest.fixture
def driver(monkeypatch) -> Driver:
    drv = Driver()
    monkeypatch.setattr(FixupDialog, "exec", lambda self: drv.exec(self))
    return drv


def _text(record) -> str:
    return record.text_path.read_text(encoding="utf-8")


def _sidecar_on_disk(record) -> dict:
    return json.loads(get_sidecar_json_path(record.image_path).read_text(encoding="utf-8"))


def _record(window, stem):
    return next(record for record in window.records if record.image_path.stem == stem)


def _row_action(dialog: FixupDialog, text: str) -> str:
    """Run the action of the comparison row showing *text* (current or proposed side)."""
    panel = dialog._comparison_panel
    left = panel._current_texts(panel.left_list)
    right = panel._current_texts(panel.right_list)
    for row, (left_index, right_index) in enumerate(panel.table_row_map):
        texts = []
        if left_index is not None:
            texts.append(left[left_index])
        if right_index is not None:
            texts.append(right[right_index])
        if text in texts:
            symbol = panel._table_row_action_symbols.get(row, "")
            assert panel._trigger_action_for_table_row(row), f"row {text!r} has no action"
            return symbol
    raise AssertionError(f"no comparison row shows {text!r}")


def _row_symbols(dialog: FixupDialog) -> dict:
    panel = dialog._comparison_panel
    left = panel._current_texts(panel.left_list)
    right = panel._current_texts(panel.right_list)
    result = {}
    for row, (left_index, right_index) in enumerate(panel.table_row_map):
        key = left[left_index] if left_index is not None else right[right_index]
        result[key] = panel._table_row_action_symbols.get(row, "")
    return result


@pytest.fixture
def window(make_main_window, image_folder):
    """a and c have pending fixups; b does not."""
    folder = image_folder(
        {"a": "old, keep", "b": "tag_b", "c": "tag_c"},
        sidecars={
            "a": _fixup("keep", "new", fixup_model="model-x", fixup_date="2026-01-01"),
            "c": _fixup("tag_c", "extra"),
        },
    )
    win = make_main_window(folder)
    assert win.current_index == 0
    return win


def _assert_resolved_by_user(record):
    data = read_sidecar_data(record.image_path)
    assert not data.has_pending_fixup
    assert data.validated_by == "user"
    assert _UTC_STAMP.match(data.validated)


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def test_dialog_shows_current_and_proposed_annotations(window, driver):
    window.open_fixup_dialog()
    assert len(driver.seen) == 1
    shown = driver.seen[0]
    assert shown["title"].endswith("a.png (1 of 2)")
    assert shown["current"] == ["old", "keep"]
    assert shown["proposed"] == ["keep", "new"]
    assert (shown["prev"], shown["next"]) == (False, True)


def test_cancel_changes_nothing(window, driver, drain_writes):
    driver.script(lambda dialog: QDialog.DialogCode.Rejected)
    window.open_fixup_dialog()
    drain_writes()
    a = _record(window, "a")
    assert driver.images == ["a"]
    assert _text(a) == "old, keep"
    assert _sidecar_on_disk(a)["fixup_tags"] == ["keep", "new"]
    assert a.has_pending_fixup
    assert window.current_index == 0


def test_rows_offer_delete_for_unvalidated_and_apply_for_proposed(window, driver):
    symbols = {}
    driver.script(lambda dialog: symbols.update(_row_symbols(dialog)))
    window.open_fixup_dialog()
    assert symbols == {"old": "✕", "keep": "", "new": "←"}


# ---------------------------------------------------------------------------
# Accept all / reject all / per-row decisions
# ---------------------------------------------------------------------------


def test_accept_all_and_merge_writes_text_and_resolves(window, driver, drain_writes):
    def accept_all(dialog):
        assert dialog.accept_button.isEnabled()
        assert not dialog.merge_button.isEnabled()  # nothing edited yet
        dialog._accept_all_without_close()
        assert dialog.undo_button.isEnabled()
        assert not dialog.accept_button.isEnabled()

    driver.script(accept_all)
    window.open_fixup_dialog()
    drain_writes()

    a = _record(window, "a")
    assert a.text == "keep, new"
    assert _text(a) == "keep, new"
    _assert_resolved_by_user(a)
    on_disk = _sidecar_on_disk(a)
    assert "fixup_tags" not in on_disk and "fixup_issues" not in on_disk
    assert on_disk["fixup_model"] == "model-x"
    assert not a.has_pending_fixup
    assert window.list_widget.item(0).data(_ROLE_BADGES) == frozenset({"✅"})
    assert window._current_tags() == ["keep", "new"]


def test_merge_without_accepting_keeps_current_text_and_resolves(window, driver, drain_writes):
    """Rejecting every proposal: Merge and Next keeps the Current column as it is."""
    driver.script(lambda dialog: dialog._merge_and_next())
    window.open_fixup_dialog()
    drain_writes()

    a = _record(window, "a")
    assert _text(a) == "old, keep"
    _assert_resolved_by_user(a)
    assert driver.images == ["a", "c"]
    assert driver.seen[1]["title"].endswith("c.png (2 of 2)")
    assert driver.seen[1]["prev"] is False  # a no longer needs a fixup


def test_per_row_accept_one_proposal_keep_the_rest(window, driver, drain_writes):
    def decide(dialog):
        assert _row_action(dialog, "new") == "←"
        assert dialog.merge_button.isEnabled()
        assert dialog.windowTitle().endswith("a.png * (1 of 2)")  # unsaved marker
        dialog._merge_without_close()

    driver.script(decide)
    window.open_fixup_dialog()
    drain_writes()

    a = _record(window, "a")
    assert _text(a) == "old, keep, new"
    _assert_resolved_by_user(a)


def test_per_row_delete_one_current_tag(window, driver, drain_writes):
    def decide(dialog):
        assert _row_action(dialog, "old") == "✕"
        assert dialog._comparison_panel.selected_annotations() == ["keep"]
        dialog._merge_and_next()

    driver.script(decide)
    window.open_fixup_dialog()
    drain_writes()

    assert _text(_record(window, "a")) == "keep"
    _assert_resolved_by_user(_record(window, "a"))


def test_merge_and_next_walks_all_fixups(window, driver, drain_writes):
    def accept_and_next(dialog):
        dialog._accept_all_without_close()
        dialog._merge_and_next()

    driver.script(accept_and_next, accept_and_next)
    window.open_fixup_dialog()
    drain_writes()

    assert driver.images == ["a", "c"]
    assert _text(_record(window, "a")) == "keep, new"
    assert _text(_record(window, "c")) == "tag_c, extra"
    assert _text(_record(window, "b")) == "tag_b"
    for stem in "ac":
        _assert_resolved_by_user(_record(window, stem))
    assert not get_sidecar_json_path(_record(window, "b").image_path).exists()
    assert window.current_index == 2


def test_merge_on_last_fixup_stays_open(window, driver):
    """Without a next fixup, Merge and Next is plain Merge and the dialog stays."""
    states = []

    def on_c(dialog):
        states.append(dialog.merge_next_button.text())
        dialog._merge_and_next()
        states.append(dialog.result())

    window.list_widget.setCurrentRow(2)
    driver.script(on_c)
    window.open_fixup_dialog()
    assert states == ["Merge", 0]
    _assert_resolved_by_user(_record(window, "c"))


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def test_undo_restores_text_and_pending_fixup(window, driver, drain_writes):
    def merge_then_undo(dialog):
        dialog._accept_all_without_close()
        assert not read_sidecar_data(dialog._image_path).has_pending_fixup
        dialog._undo_merge()
        assert not dialog.undo_button.isEnabled()
        assert dialog._comparison_panel.selected_annotations() == ["old", "keep"]

    driver.script(merge_then_undo)
    window.open_fixup_dialog()
    drain_writes()

    a = _record(window, "a")
    assert _text(a) == "old, keep"
    data = read_sidecar_data(a.image_path)
    assert data.has_pending_fixup
    assert (data.fixup_issues, data.fixup_tags) == ("issue", ["keep", "new"])
    assert data.validated is None
    assert a.has_pending_fixup
    assert "⚖️" in window.list_widget.item(0).data(_ROLE_BADGES)


def test_undo_restores_the_whole_sidecar(make_main_window, image_folder, driver, drain_writes):
    original = _fixup(
        "keep", "new",
        fixup_model="model-x", fixup_date="2026-01-01",
        validated="2025-12-31T00:00:00Z", validated_by="model-x",
    )
    window = make_main_window(image_folder({"a": "old, keep"}, sidecars={"a": original}))

    def merge_then_undo(dialog):
        dialog._accept_all_without_close()
        dialog._undo_merge()

    driver.script(merge_then_undo)
    window.open_fixup_dialog()
    drain_writes()

    assert read_sidecar_data(window.records[0].image_path) == original


def test_undo_of_local_edits_only_writes_nothing(window, driver, drain_writes):
    def edit_then_undo(dialog):
        _row_action(dialog, "new")
        dialog._undo_merge()
        assert dialog._comparison_panel.selected_annotations() == ["old", "keep"]

    driver.script(edit_then_undo)
    window.open_fixup_dialog()
    drain_writes()

    a = _record(window, "a")
    assert _text(a) == "old, keep"
    assert read_sidecar_data(a.image_path).validated is None
    assert a.has_pending_fixup


# ---------------------------------------------------------------------------
# Prev / Next navigation
# ---------------------------------------------------------------------------


@pytest.fixture
def nav_window(make_main_window, image_folder):
    """Fixups on a, c and e; b and d have none."""
    folder = image_folder(
        {name: f"tag_{name}" for name in "abcde"},
        sidecars={stem: _fixup(f"tag_{stem}", "more") for stem in "ace"},
    )
    return make_main_window(folder)


def test_next_and_prev_visit_only_fixup_images(nav_window, driver, drain_writes):
    driver.script(
        lambda dialog: dialog._skip_to_next(),
        lambda dialog: dialog._skip_to_next(),
        lambda dialog: dialog._navigate_prev(),
        lambda dialog: QDialog.DialogCode.Rejected,
    )
    nav_window.open_fixup_dialog()
    drain_writes()

    assert driver.images == ["a", "c", "e", "c"]
    assert [(s["prev"], s["next"]) for s in driver.seen] == [
        (False, True), (True, True), (True, False), (True, True)
    ]
    assert [s["title"][-8:] for s in driver.seen] == ["(1 of 3)", "(2 of 3)", "(3 of 3)", "(2 of 3)"]
    assert nav_window.current_index == 2
    for record in nav_window.records:
        assert _text(record) == f"tag_{record.image_path.stem}"
    assert sum(record.has_pending_fixup for record in nav_window.records) == 3


def test_next_skips_hidden_rows(nav_window, driver):
    nav_window.list_widget.item(2).setHidden(True)
    driver.script(lambda dialog: dialog._skip_to_next())
    nav_window.open_fixup_dialog()
    assert driver.images == ["a", "e"]


def test_skip_from_last_fixup_wraps_back(nav_window, driver):
    """Next on the last fixup goes to the previous one (the list has no further fixup)."""
    nav_window.list_widget.setCurrentRow(4)
    driver.script(lambda dialog: dialog._skip_to_next())
    nav_window.open_fixup_dialog()
    assert driver.images == ["e", "c"]


# ---------------------------------------------------------------------------
# Deleting the image from the dialog
# ---------------------------------------------------------------------------


def _delete_from_dialog(dialog):
    dialog._image_pane._confirm_delete = False
    dialog._image_pane._delete_file_from_context_menu()


def test_delete_moves_on_to_the_next_fixup(window, driver, drain_writes):
    a = _record(window, "a")
    driver.script(_delete_from_dialog)
    window.open_fixup_dialog()
    drain_writes()

    assert driver.images == ["a", "c"]
    assert not a.image_path.exists()
    assert not a.text_path.exists()
    assert not get_sidecar_json_path(a.image_path).exists()
    assert [record.image_path.stem for record in window.records] == ["b", "c"]
    assert window._current_record().image_path.stem == "c"


def test_delete_last_fixup_enters_no_fixups_state(make_main_window, image_folder, driver, drain_writes):
    window = make_main_window(image_folder({"a": "old", "b": "tag_b"}, sidecars={"a": _fixup("new")}))
    states = {}

    def delete(dialog):
        _delete_from_dialog(dialog)
        states["buttons"] = [
            button.isEnabled()
            for button in (dialog.accept_button, dialog.merge_button, dialog.merge_next_button,
                           dialog.prev_button, dialog.next_button)
        ]
        states["issues"] = dialog.issues_label.toPlainText()

    driver.script(delete)
    window.open_fixup_dialog()
    drain_writes()

    assert driver.images == ["a"]
    assert states == {"buttons": [False] * 5, "issues": "No fixup files remaining."}
    assert [record.image_path.stem for record in window.records] == ["b"]


def test_left_delete_only_offered_for_validation_fixups(make_main_window, image_folder, driver):
    """Vision/AI Find-only proposals never offer to delete current tags."""
    window = make_main_window(
        image_folder({"a": "old, keep"}, sidecars={"a": SidecarData(vision_tags=["keep", "new"])})
    )
    symbols = {}
    driver.script(lambda dialog: symbols.update(_row_symbols(dialog)))
    window.open_fixup_dialog()
    assert "✕" not in symbols.values()
    assert symbols["new"] == "←"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_first_close_saves_geometry_to_config(window, driver, isolated_config):
    window.open_fixup_dialog()
    saved = window._cfg["merge_dialog_geometry"]
    assert saved["width"] > 0 and saved["height"] > 0
    assert json.loads(isolated_config.read_text(encoding="utf-8"))["merge_dialog_geometry"] == saved


def test_unmoved_dialog_keeps_saved_geometry(window, driver, monkeypatch):
    geometry = {"x": 50, "y": 60, "width": 500, "height": 300}
    window._cfg["merge_dialog_geometry"] = dict(geometry)
    saves = []
    monkeypatch.setattr(window, "_save_merge_dialog_geometry", lambda g: saves.append(g))
    window.open_fixup_dialog()
    assert saves == []
    assert window._cfg["merge_dialog_geometry"] == geometry


def test_user_move_is_saved_relative_to_the_saved_geometry(window, driver, qtbot, isolated_config):
    window._cfg["merge_dialog_geometry"] = {"x": 50, "y": 60, "width": 500, "height": 300}
    requested = {}

    def show_and_move(dialog):
        requested["rect"] = dialog.geometry()
        dialog.show()
        qtbot.wait(400)  # past the settle delay
        dialog.move(dialog.x() + 40, dialog.y() + 30)
        dialog.resize(dialog.width() + 20, dialog.height())
        qtbot.wait(10)

    driver.script(show_and_move)
    window.open_fixup_dialog()

    rect = requested["rect"]
    expected = {"x": rect.x() + 40, "y": rect.y() + 30, "width": rect.width() + 20, "height": rect.height()}
    assert window._cfg["merge_dialog_geometry"] == expected
    assert json.loads(isolated_config.read_text(encoding="utf-8"))["merge_dialog_geometry"] == expected
