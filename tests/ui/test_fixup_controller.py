"""FixupController: fixup state refresh, button state, fixup navigation and list badges."""
from __future__ import annotations

import json

import pytest
from PyQt6.QtGui import QColor, QImage

from imagetagger.ui.main_window import _ROLE_BADGES
from imagetagger.ui.models import _UNKNOWN
from imagetagger.utils.sidecar import SidecarData, write_sidecar_data

_FIXUP = SidecarData(fixup_issues="issue", fixup_tags=["x"])


def _badges(window, index):
    return window.list_widget.item(index).data(_ROLE_BADGES) or frozenset()


def _row(window, stem):
    return next(i for i, record in enumerate(window.records) if record.image_path.stem == stem)


def _save_png(path, width, height):
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(path))


@pytest.fixture
def five(make_main_window, image_folder):
    """Images a..e; b, c and e have a pending fixup."""
    folder = image_folder(
        {name: f"tag_{name}" for name in "abcde"},
        sidecars={"b": _FIXUP, "c": _FIXUP, "e": _FIXUP},
    )
    return make_main_window(folder)


# ---------------------------------------------------------------------------
# _on_fixup_state_changed
# ---------------------------------------------------------------------------


def test_single_path_refresh_rereads_sidecar_and_badge(main_window):
    record = main_window.records[0]
    assert not record.has_pending_fixup  # cached from here on
    write_sidecar_data(record.image_path, _FIXUP)
    assert not record.has_pending_fixup  # still the cached value

    main_window._on_fixup_state_changed(record.image_path)

    assert record.has_pending_fixup
    assert "⚖️" in _badges(main_window, 0)
    assert main_window.bulk_fixup_button.isEnabled()


def test_single_path_refresh_leaves_other_records_cached(main_window):
    a, b = main_window.records[0], main_window.records[1]
    assert not a.has_pending_fixup and not b.has_pending_fixup
    write_sidecar_data(a.image_path, _FIXUP)
    write_sidecar_data(b.image_path, _FIXUP)

    main_window._on_fixup_state_changed(a.image_path)

    assert a.has_pending_fixup
    assert b._sidecar_has_pending_fixup is False  # only the given image is re-read


def test_single_path_refresh_for_unknown_image_is_harmless(main_window, tmp_path):
    main_window._on_fixup_state_changed(tmp_path / "elsewhere.png")
    assert main_window.fixup_button.isEnabled()


def test_single_path_refresh_reapplies_an_active_filter(main_window):
    main_window.filter_input.setText("fixup")
    assert all(main_window.list_widget.item(i).isHidden() for i in range(3))

    record = main_window.records[1]
    write_sidecar_data(record.image_path, _FIXUP)
    main_window._on_fixup_state_changed(record.image_path)

    hidden = [main_window.list_widget.item(i).isHidden() for i in range(3)]
    assert hidden == [True, False, True]
    assert main_window.current_index == 1


def test_single_path_refresh_updates_validated_tooltip_of_current_image(main_window):
    record = main_window.records[0]
    assert main_window.current_index == 0
    write_sidecar_data(record.image_path, SidecarData(validated="2026-01-02T03:04:05Z", validated_by="user"))
    main_window._on_fixup_state_changed(record.image_path)
    assert main_window.image_label.toolTip() == "Validated by user on 2026-01-02"


def test_bulk_refresh_invalidates_every_record_then_warms(main_window, monkeypatch):
    for record in main_window.records:
        record._sidecar_has_pending_fixup = True
        record._sidecar_validated = "stale"
    warms = []
    monkeypatch.setattr(main_window.fixup_controller, "_async_warm_fixup_cache", lambda: warms.append(1))

    main_window._on_fixup_state_changed(None)

    assert warms == [1]
    for record in main_window.records:
        assert record._sidecar_has_pending_fixup is None
        assert record._sidecar_validated is _UNKNOWN


def test_bulk_refresh_warms_cache_and_refreshes_rows_and_buttons(qtbot, main_window):
    assert not main_window.bulk_fixup_button.isEnabled()
    a, b, c = main_window.records
    write_sidecar_data(b.image_path, _FIXUP)
    write_sidecar_data(c.image_path, SidecarData(validated="2026-01-01T00:00:00Z", validated_by="m"))

    main_window._on_fixup_state_changed(None)

    qtbot.waitUntil(lambda: "⚖️" in _badges(main_window, 1) and "✅" in _badges(main_window, 2), timeout=5_000)
    assert b._sidecar_has_pending_fixup is True
    assert a._sidecar_has_pending_fixup is False
    assert c._sidecar_validated == "2026-01-01T00:00:00Z"
    assert a._sidecar_validated is None
    assert main_window.bulk_fixup_button.isEnabled()
    assert _badges(main_window, 0) == frozenset()


# ---------------------------------------------------------------------------
# Fixup button state
# ---------------------------------------------------------------------------


def test_fixup_button_enabled_for_a_listed_image_even_without_fixup(main_window):
    main_window._update_fixup_button_state()
    assert main_window.fixup_button.isEnabled()
    assert not main_window.bulk_fixup_button.isEnabled()


def test_fixup_button_disabled_without_a_record(main_window):
    main_window.current_index = -1
    main_window._update_fixup_button_state()
    assert not main_window.fixup_button.isEnabled()


def test_fixup_button_disabled_when_current_row_is_hidden(main_window):
    main_window.list_widget.item(main_window.current_index).setHidden(True)
    main_window._update_fixup_button_state()
    assert not main_window.fixup_button.isEnabled()


@pytest.mark.parametrize("action", ["Generate", "AI Find", None])
def test_fixup_button_disabled_while_other_llm_task_runs(main_window, monkeypatch, action):
    monkeypatch.setattr(main_window.llm_controller, "_llm_thread", object())
    monkeypatch.setattr(main_window.llm_controller, "_llm_action_name", action)
    main_window._update_fixup_button_state()
    assert not main_window.fixup_button.isEnabled()


def test_fixup_button_during_validate_depends_on_current_image_pending(main_window, monkeypatch):
    llm = main_window.llm_controller
    monkeypatch.setattr(llm, "_llm_thread", object())
    monkeypatch.setattr(llm, "_llm_action_name", "Validate")
    monkeypatch.setattr(llm, "_validate_pending_paths", {main_window.records[1].image_path})

    main_window._update_fixup_button_state()
    assert main_window.fixup_button.isEnabled()  # current image a is not waiting

    llm._validate_pending_paths.add(main_window.records[0].image_path)
    main_window._update_fixup_button_state()
    assert not main_window.fixup_button.isEnabled()


def test_bulk_button_ignores_hidden_fixup_rows(five):
    five._update_fixup_button_state()
    assert five.bulk_fixup_button.isEnabled()
    for stem in "bce":
        five.list_widget.item(_row(five, stem)).setHidden(True)
    five._update_fixup_button_state()
    assert not five.bulk_fixup_button.isEnabled()


# ---------------------------------------------------------------------------
# Fixup navigation
# ---------------------------------------------------------------------------


def test_adjacent_fixup_index_forward_and_backward(five):
    find = five._find_adjacent_fixup_index
    assert find(0, 1) == 1
    assert find(1, 1) == 2
    assert find(2, 1) == 4
    assert find(4, 1) is None
    assert find(4, -1) == 2
    assert find(1, -1) is None
    assert find(3, -1) == 2


@pytest.mark.parametrize("direction", [0, 2, -2])
def test_adjacent_fixup_index_rejects_bad_direction(five, direction):
    assert five._find_adjacent_fixup_index(0, direction) is None


def test_adjacent_fixup_index_skips_hidden_and_validate_pending(five, monkeypatch):
    five.list_widget.item(1).setHidden(True)
    monkeypatch.setattr(five.llm_controller, "_validate_pending_paths", {five.records[2].image_path})
    assert five._find_adjacent_fixup_index(0, 1) == 4
    assert five._find_adjacent_fixup_index(4, -1) is None


def test_find_fixup_index_first_and_last(five, monkeypatch):
    assert five._find_fixup_index(reverse=False) == 1
    assert five._find_fixup_index(reverse=True) == 4

    five.list_widget.item(1).setHidden(True)
    monkeypatch.setattr(five.llm_controller, "_validate_pending_paths", {five.records[4].image_path})
    assert five._find_fixup_index(reverse=False) == 2
    assert five._find_fixup_index(reverse=True) == 2

    five.list_widget.item(2).setHidden(True)
    assert five._find_fixup_index(reverse=False) is None


def test_find_fixup_index_without_records(main_window):
    main_window.records.clear()
    assert main_window._find_fixup_index(reverse=False) is None
    assert main_window._find_fixup_index(reverse=True) is None


def test_jump_to_first_and_last_fixup(five):
    five._jump_to_last_fixup()
    assert five.current_index == 4
    five._jump_to_first_fixup()
    assert five.current_index == 1

    five.list_widget.item(1).setHidden(True)
    five.list_widget.setCurrentRow(3)
    five._jump_to_first_fixup()
    assert five.current_index == 2


def test_jump_without_fixups_reports_and_stays(main_window):
    main_window.list_widget.setCurrentRow(1)
    for jump in (main_window._jump_to_first_fixup, main_window._jump_to_last_fixup):
        main_window.statusBar().clearMessage()
        jump()
        assert main_window.current_index == 1
        assert main_window.statusBar().currentMessage() == "No fixup image found in the current list"


# ---------------------------------------------------------------------------
# _record_needs_fixup and the aspect-ratio fixup
# ---------------------------------------------------------------------------


@pytest.fixture
def wide_folder(image_folder):
    """a.png is 30x10 (3:1, not a default allowed ratio); b.png is 16x16."""
    folder = image_folder({"a": "tag_a", "b": "tag_b"})
    _save_png(folder / "a.png", 30, 10)
    return folder


def _write_config(path, **values):
    path.write_text(json.dumps(values), encoding="utf-8")


def test_ratio_outside_allowed_ratios_needs_fixup(make_main_window, wide_folder):
    window = make_main_window(wide_folder)
    a, b = window.records
    assert not a.has_pending_fixup
    assert window._record_needs_fixup(a)
    assert not window._record_needs_fixup(b)
    assert "✂️" in _badges(window, 0)
    assert "✂️" not in _badges(window, 1)
    assert window._find_fixup_index(reverse=False) == 0
    assert window._find_adjacent_fixup_index(1, -1) == 0


def test_ratio_in_configured_allowed_ratios_needs_no_fixup(make_main_window, wide_folder, isolated_config):
    _write_config(isolated_config, allowed_ratios="3:1")
    window = make_main_window(wide_folder)
    a, b = window.records
    assert not window._record_needs_fixup(a)
    assert window._record_needs_fixup(b)  # 1:1 is no longer allowed
    assert window._find_fixup_index(reverse=False) == 1


def test_empty_allowed_ratios_disables_the_ratio_fixup(make_main_window, wide_folder, isolated_config):
    _write_config(isolated_config, allowed_ratios="")
    window = make_main_window(wide_folder)
    assert not any(window._record_needs_fixup(record) for record in window.records)
    assert "✂️" not in _badges(window, 0)
    assert window._find_fixup_index(reverse=False) is None
    assert not window.status_ratio_label.isVisible()


def test_pending_sidecar_fixup_needs_fixup_without_ratio_check(make_main_window, image_folder, isolated_config):
    _write_config(isolated_config, allowed_ratios="")
    window = make_main_window(image_folder({"a": "tag_a", "b": "tag_b"}, sidecars={"b": _FIXUP}))
    assert [window._record_needs_fixup(record) for record in window.records] == [False, True]


# ---------------------------------------------------------------------------
# List badges
# ---------------------------------------------------------------------------


def test_list_badges_reflect_sidecar_fields(make_main_window, image_folder):
    folder = image_folder(
        {name: f"tag_{name}" for name in "abcdef"},
        sidecars={
            "b": SidecarData(fixup_description="better"),
            "c": SidecarData(vision_caption="a car"),
            "d": SidecarData(ai_find_matches=["red car"]),
            "e": SidecarData(validated="2026-01-01T00:00:00Z", validated_by="user"),
            "f": SidecarData(fixup_tags=["t"], vision_tags=["v"], ai_find_matches=["q"],
                             validated="2026-01-01T00:00:00Z"),
        },
    )
    window = make_main_window(folder)
    assert [_badges(window, i) for i in range(6)] == [
        frozenset(),
        frozenset({"⚖️"}),
        frozenset({"✨"}),
        frozenset({"🔍"}),
        frozenset({"✅"}),
        frozenset({"⚖️", "✨", "🔍", "✅"}),
    ]
    window._update_list_item_preview(5)
    tooltip = window.list_widget.item(5).toolTip()
    assert "Badges: ⚖️ fixup, ✨ vision, 🔍 search, ✅ validated" in tooltip
    assert "Validated by unknown on 2026-01-01" in tooltip


def test_update_list_item_preview_follows_sidecar_changes(main_window):
    path = main_window.records[0].image_path
    write_sidecar_data(path, SidecarData(fixup_tags=["t"], vision_tags=["v"], ai_find_matches=["q"]))
    main_window._update_list_item_preview(0)
    assert _badges(main_window, 0) == frozenset({"⚖️", "✨", "🔍"})

    write_sidecar_data(path, SidecarData(validated="2026-01-01T00:00:00Z", validated_by="user"))
    main_window._update_list_item_preview(0)
    assert _badges(main_window, 0) == frozenset({"✅"})
    assert "Validated by user on 2026-01-01" in main_window.list_widget.item(0).toolTip()


def test_update_list_item_preview_out_of_range_is_ignored(main_window):
    main_window._update_list_item_preview(99)
