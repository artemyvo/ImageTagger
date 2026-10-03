"""BulkFixupDialog: global per-tag decisions applied across every pending image.

The dialog is built directly (as MainWindow.open_bulk_fixup_dialog does) and
never shown; decisions are made by clicking each row's segment buttons and
Apply runs through the Apply button.
"""
from __future__ import annotations

import json

import pytest
from PyQt6.QtWidgets import QDialog

from imagetagger.ui.bulk_fixup_dialog import BulkFixupDialog, _DecisionWidget
from imagetagger.ui.main_window import _ROLE_BADGES
from imagetagger.utils.sidecar import SidecarData, get_sidecar_json_path, read_sidecar_data

_DESCRIPTION = "A red car parked on a quiet street"


def _fixup(*tags: str) -> SidecarData:
    return SidecarData(fixup_issues="issue", fixup_tags=list(tags), fixup_model="m")


def _dialog(window) -> BulkFixupDialog:
    pending = [record for record in window.records if record.has_pending_fixup]
    return BulkFixupDialog(
        records=pending,
        all_records=list(window.records),
        parse_tags=window._parse_tags,
        is_description_like=window._is_description_like_annotation,
        parent=window,
    )


def _decide(dialog: BulkFixupDialog, tag: str, kind: str, state: str) -> None:
    index = dialog._row_keys.index((tag, kind))
    widget = dialog._decision_widgets[index]
    button = {
        _DecisionWidget.ACCEPT: widget._btn_accept,
        _DecisionWidget.REJECT: widget._btn_reject,
        _DecisionWidget.UNRESOLVED: widget._btn_skip,
    }[state]
    button.click()
    assert widget.state == state


def _apply(dialog: BulkFixupDialog, drain_writes) -> None:
    dialog._apply_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    drain_writes()


def _record(window, stem):
    return next(record for record in window.records if record.image_path.stem == stem)


def _text(window, stem) -> str:
    return _record(window, stem).text_path.read_text(encoding="utf-8")


def _disk(window, stem) -> dict:
    path = get_sidecar_json_path(_record(window, stem).image_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_resolved_by_user(window, stem):
    on_disk = _disk(window, stem)
    assert "fixup_tags" not in on_disk and "fixup_issues" not in on_disk
    assert on_disk["validated_by"] == "user"
    assert on_disk["validated"]
    assert not _record(window, stem).has_pending_fixup


@pytest.fixture
def window(make_main_window, image_folder):
    """a: add blue, delete car; b: add blue; c: no fixup; d: fixup without tags."""
    folder = image_folder(
        {"a": "red, car", "b": "red", "c": "green", "d": "tag_d"},
        sidecars={
            "a": _fixup("red", "blue"),
            "b": _fixup("red", "blue"),
            "d": SidecarData(fixup_description="Only a better description here"),
        },
    )
    return make_main_window(folder)


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


def test_proposals_aggregate_pending_changes(window):
    dialog = _dialog(window)
    assert [record.image_path.stem for record in dialog._fixup_records] == ["a", "b"]
    assert sorted(dialog._row_keys) == [("blue", "add"), ("car", "del")]
    table = dialog._table
    values = {
        dialog._row_keys[row]: [table.item(row, col).text() for col in (1, 2, 3)]
        for row in range(table.rowCount())
    }
    # Resolves / Current / Result
    assert values == {("blue", "add"): ["1", "0", "2"], ("car", "del"): ["0", "1", "0"]}
    assert dialog._resolve_label.text() == "Images to resolve: 0 of 2"
    _decide(dialog, "blue", "add", _DecisionWidget.ACCEPT)
    assert dialog._resolve_label.text() == "Images to resolve: 1 of 2"
    _decide(dialog, "car", "del", _DecisionWidget.ACCEPT)
    assert dialog._resolve_label.text() == "Images to resolve: 2 of 2"


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def test_no_decision_writes_nothing(window, drain_writes, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_set_tags_for_image_path", lambda *args: calls.append(args))
    before = {stem: _disk(window, stem) for stem in "abd"}
    _apply(_dialog(window), drain_writes)
    assert calls == []
    assert {stem: _disk(window, stem) for stem in "abd"} == before
    assert [_text(window, stem) for stem in "abcd"] == ["red, car", "red", "green", "tag_d"]


def test_accept_everything_resolves_every_image(window, drain_writes):
    dialog = _dialog(window)
    _decide(dialog, "blue", "add", _DecisionWidget.ACCEPT)
    _decide(dialog, "car", "del", _DecisionWidget.ACCEPT)
    _apply(dialog, drain_writes)

    assert _text(window, "a") == "blue, red"
    assert _text(window, "b") == "blue, red"
    assert _record(window, "a").text == "blue, red"
    for stem in "ab":
        _assert_resolved_by_user(window, stem)
    assert _text(window, "c") == "green"
    assert not get_sidecar_json_path(_record(window, "c").image_path).exists()
    assert _disk(window, "d")["fixup_description"] == "Only a better description here"
    assert window.list_widget.item(0).data(_ROLE_BADGES) == frozenset({"✅"})


def test_partial_accept_keeps_remaining_fixup_tags(window, drain_writes):
    dialog = _dialog(window)
    _decide(dialog, "blue", "add", _DecisionWidget.ACCEPT)
    _apply(dialog, drain_writes)

    # a still has "car" to delete: its fixup stays pending, blue now applied.
    assert _text(window, "a") == "blue, car, red"
    on_disk = _disk(window, "a")
    assert sorted(on_disk["fixup_tags"]) == ["blue", "red"]
    assert on_disk["fixup_issues"] == "issue"
    assert "validated" not in on_disk
    assert _record(window, "a").has_pending_fixup
    # b only had the add, so it is fully resolved.
    assert _text(window, "b") == "blue, red"
    _assert_resolved_by_user(window, "b")


def test_reject_add_drops_it_from_fixup_tags(window, drain_writes):
    dialog = _dialog(window)
    _decide(dialog, "blue", "add", _DecisionWidget.REJECT)
    _apply(dialog, drain_writes)

    assert _text(window, "a") == "car, red"
    assert _disk(window, "a")["fixup_tags"] == ["red"]
    assert _record(window, "a").has_pending_fixup  # the car deletion is undecided
    assert _text(window, "b") == "red"
    _assert_resolved_by_user(window, "b")


def test_reject_delete_keeps_tag_and_adds_it_to_fixup_tags(window, drain_writes):
    dialog = _dialog(window)
    _decide(dialog, "car", "del", _DecisionWidget.REJECT)
    _apply(dialog, drain_writes)

    assert _text(window, "a") == "car, red"
    assert _disk(window, "a")["fixup_tags"] == ["blue", "car", "red"]
    assert _record(window, "a").has_pending_fixup
    # b has no pending "car" deletion: untouched.
    assert _text(window, "b") == "red"
    assert _disk(window, "b")["fixup_tags"] == ["red", "blue"]


def test_mixed_decisions_resolve_image(window, drain_writes):
    dialog = _dialog(window)
    _decide(dialog, "blue", "add", _DecisionWidget.REJECT)
    _decide(dialog, "car", "del", _DecisionWidget.ACCEPT)
    _apply(dialog, drain_writes)

    assert _text(window, "a") == "red"
    assert _text(window, "b") == "red"
    for stem in "ab":
        _assert_resolved_by_user(window, stem)


def test_decision_reverted_to_unresolved_counts_as_none(window, drain_writes):
    dialog = _dialog(window)
    _decide(dialog, "blue", "add", _DecisionWidget.ACCEPT)
    _decide(dialog, "blue", "add", _DecisionWidget.UNRESOLVED)
    _apply(dialog, drain_writes)
    assert _text(window, "a") == "red, car"
    assert _disk(window, "a")["fixup_tags"] == ["red", "blue"]


# ---------------------------------------------------------------------------
# Description-like entries
# ---------------------------------------------------------------------------


@pytest.fixture
def described(make_main_window, image_folder):
    folder = image_folder(
        {"a": f"{_DESCRIPTION}, red, car"},
        sidecars={"a": _fixup("A red car parked on a quiet street at night", "red", "blue")},
    )
    return make_main_window(folder)


def test_description_entries_are_not_proposals(described):
    dialog = _dialog(described)
    assert sorted(dialog._row_keys) == [("blue", "add"), ("car", "del")]


def test_description_like_fixup_entry_kept_on_partial_update(described, drain_writes):
    dialog = _dialog(described)
    _decide(dialog, "car", "del", _DecisionWidget.ACCEPT)
    _apply(dialog, drain_writes)

    on_disk = _disk(described, "a")
    assert on_disk["fixup_tags"] == ["A red car parked on a quiet street at night", "blue", "red"]
    text = _text(described, "a")
    assert text.split(", ")[1:] == ["red"]
    assert text.split(", ")[0].casefold() == _DESCRIPTION.casefold()


def test_description_kept_verbatim_in_text(described, drain_writes):
    dialog = _dialog(described)
    _decide(dialog, "car", "del", _DecisionWidget.ACCEPT)
    _apply(dialog, drain_writes)
    assert _text(described, "a") == f"{_DESCRIPTION}, red"


# ---------------------------------------------------------------------------
# From the main window
# ---------------------------------------------------------------------------


def test_main_window_passes_only_listed_pending_records(window, monkeypatch):
    seen = []

    def fake_exec(dialog):
        seen.append([record.image_path.stem for record in dialog._fixup_records])
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(BulkFixupDialog, "exec", fake_exec)
    window.open_bulk_fixup_dialog()
    window.list_widget.item(1).setHidden(True)
    window.open_bulk_fixup_dialog()
    assert seen == [["a", "b"], ["a"]]
    assert read_sidecar_data(_record(window, "a").image_path).fixup_tags == ["red", "blue"]
