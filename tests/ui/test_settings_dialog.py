"""SettingsDialog (merge_table_mouse_actions) and MainWindow.open_settings_dialog.

The dialog is built directly and driven through its widgets and buttons; the
main window's modal exec is replaced by a scripted step that never blocks.
"""
from __future__ import annotations

import json

import pytest
from PyQt6.QtWidgets import QDialog, QDialogButtonBox

from imagetagger import config as _config
from imagetagger.ui import fixup_controller
from imagetagger.ui.settings_dialog import SettingsDialog

KEYS = (
    "double_click_action_enabled",
    "swipe_actions_enabled",
    "horizontal_scroll_actions_enabled",
    "horizontal_scroll_reverse_enabled",
    "horizontal_scroll_stop_idle_seconds",
    "horizontal_scroll_row_target_mode",
)

CUSTOM = {
    "double_click_action_enabled": False,
    "swipe_actions_enabled": True,
    "horizontal_scroll_actions_enabled": True,
    "horizontal_scroll_reverse_enabled": True,
    "horizontal_scroll_stop_idle_seconds": 1.25,
    "horizontal_scroll_row_target_mode": _config.MERGE_TABLE_HSCROLL_TARGET_SELECTED_ROW,
}


@pytest.fixture
def make_dialog(qtbot):
    def _make(cfg: dict) -> SettingsDialog:
        dialog = SettingsDialog(cfg)
        qtbot.addWidget(dialog)
        return dialog

    return _make


def _shown(dialog: SettingsDialog) -> dict:
    return {
        "double_click_action_enabled": dialog.double_click_checkbox.isChecked(),
        "swipe_actions_enabled": dialog.swipe_checkbox.isChecked(),
        "horizontal_scroll_actions_enabled": dialog.hscroll_checkbox.isChecked(),
        "horizontal_scroll_reverse_enabled": dialog.hscroll_reverse_checkbox.isChecked(),
        "horizontal_scroll_stop_idle_seconds": round(dialog.hscroll_pause_spinbox.value(), 2),
        "horizontal_scroll_row_target_mode": dialog.hscroll_target_combo.currentData(),
    }


def _set(dialog: SettingsDialog, values: dict) -> None:
    dialog.double_click_checkbox.setChecked(values["double_click_action_enabled"])
    dialog.swipe_checkbox.setChecked(values["swipe_actions_enabled"])
    dialog.hscroll_checkbox.setChecked(values["horizontal_scroll_actions_enabled"])
    dialog.hscroll_reverse_checkbox.setChecked(values["horizontal_scroll_reverse_enabled"])
    dialog.hscroll_pause_spinbox.setValue(values["horizontal_scroll_stop_idle_seconds"])
    dialog.hscroll_target_combo.setCurrentIndex(
        dialog.hscroll_target_combo.findData(values["horizontal_scroll_row_target_mode"])
    )


def _button(dialog: SettingsDialog, which):
    box = dialog.findChild(QDialogButtonBox)
    return box.button(which)


# ---------------------------------------------------------------------------
# Showing the current values
# ---------------------------------------------------------------------------


def test_empty_config_shows_defaults(make_dialog):
    dialog = make_dialog({})
    assert _shown(dialog) == _config.default_merge_table_mouse_actions()
    assert not dialog._hscroll_options.isEnabled()  # horizontal scroll is off by default


def test_shows_config_values(make_dialog):
    dialog = make_dialog({"merge_table_mouse_actions": dict(CUSTOM)})
    assert _shown(dialog) == CUSTOM
    assert dialog._hscroll_options.isEnabled()


def test_loaded_config_is_shown(make_dialog, isolated_config):
    isolated_config.write_text(json.dumps({"merge_table_mouse_actions": CUSTOM}), encoding="utf-8")
    assert _shown(make_dialog(_config.load())) == CUSTOM


def test_partial_settings_fill_in_defaults(make_dialog):
    dialog = make_dialog({"merge_table_mouse_actions": {"swipe_actions_enabled": True}})
    expected = {**_config.default_merge_table_mouse_actions(), "swipe_actions_enabled": True}
    assert _shown(dialog) == expected


@pytest.mark.parametrize("value", [None, "yes", 3, ["swipe_actions_enabled"]])
def test_non_dict_settings_show_defaults(make_dialog, value):
    dialog = make_dialog({"merge_table_mouse_actions": value})
    assert _shown(dialog) == _config.default_merge_table_mouse_actions()


def test_unknown_row_target_mode_selects_the_first_choice(make_dialog):
    dialog = make_dialog({"merge_table_mouse_actions": {"horizontal_scroll_row_target_mode": 99}})
    assert dialog.hscroll_target_combo.currentIndex() == 0
    assert dialog.hscroll_target_combo.currentData() == _config.MERGE_TABLE_HSCROLL_TARGET_POINTER_ON_SELECTED


def test_pause_outside_the_spinbox_range_is_clamped(make_dialog):
    dialog = make_dialog({"merge_table_mouse_actions": {"horizontal_scroll_stop_idle_seconds": 60}})
    assert dialog.hscroll_pause_spinbox.value() == 5.0
    cfg: dict = {}
    dialog.apply_to(cfg)
    assert cfg["merge_table_mouse_actions"]["horizontal_scroll_stop_idle_seconds"] == 5.0


def test_every_row_target_mode_is_offered(make_dialog):
    dialog = make_dialog({})
    combo = dialog.hscroll_target_combo
    assert {combo.itemData(i) for i in range(combo.count())} == _config.MERGE_TABLE_HSCROLL_TARGET_ALLOWED_MODES


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_horizontal_scroll_checkbox_enables_its_options(make_dialog):
    dialog = make_dialog({})
    dialog.hscroll_checkbox.setChecked(True)
    assert dialog._hscroll_options.isEnabled()
    assert dialog.hscroll_pause_spinbox.isEnabled() and dialog.hscroll_target_combo.isEnabled()
    dialog.hscroll_checkbox.setChecked(False)
    assert not dialog.hscroll_pause_spinbox.isEnabled()


def test_restore_defaults_shows_defaults_without_writing(make_dialog):
    cfg = {"merge_table_mouse_actions": dict(CUSTOM)}
    dialog = make_dialog(cfg)
    dialog.open()
    _button(dialog, QDialogButtonBox.StandardButton.RestoreDefaults).click()
    assert _shown(dialog) == _config.default_merge_table_mouse_actions()
    assert not dialog._hscroll_options.isEnabled()
    assert cfg["merge_table_mouse_actions"] == CUSTOM  # saved only on OK
    assert dialog.isVisible()  # the dialog stays open


def test_pause_is_rounded_to_two_decimals(make_dialog):
    dialog = make_dialog({})
    dialog.hscroll_pause_spinbox.setValue(0.456)
    cfg: dict = {}
    dialog.apply_to(cfg)
    assert cfg["merge_table_mouse_actions"]["horizontal_scroll_stop_idle_seconds"] == 0.46


def test_pause_below_zero_is_clamped(make_dialog):
    dialog = make_dialog({})
    dialog.hscroll_pause_spinbox.setValue(-3)
    assert dialog.hscroll_pause_spinbox.value() == 0.0


# ---------------------------------------------------------------------------
# apply_to and the OK / Cancel buttons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_each_control_round_trips(make_dialog, key):
    defaults = _config.default_merge_table_mouse_actions()
    changed = dict(defaults)
    changed[key] = CUSTOM[key]
    assert changed != defaults
    dialog = make_dialog({})
    _set(dialog, changed)
    cfg: dict = {"other_setting": 7}
    dialog.apply_to(cfg)
    assert cfg["merge_table_mouse_actions"] == changed
    assert cfg["other_setting"] == 7  # other settings are left alone
    # ... and back through a save/load and a new dialog.
    _config.save({**_config.load(), **cfg})
    assert _config.load()["merge_table_mouse_actions"] == changed
    assert _shown(make_dialog(_config.load())) == changed


def test_apply_writes_plain_types(make_dialog):
    dialog = make_dialog({"merge_table_mouse_actions": dict(CUSTOM)})
    cfg: dict = {}
    dialog.apply_to(cfg)
    values = cfg["merge_table_mouse_actions"]
    assert set(values) == set(KEYS)
    assert type(values["horizontal_scroll_row_target_mode"]) is int
    assert type(values["horizontal_scroll_stop_idle_seconds"]) is float
    json.dumps(cfg)


def test_ok_and_cancel_buttons_set_the_result(make_dialog):
    dialog = make_dialog({})
    dialog.open()
    _button(dialog, QDialogButtonBox.StandardButton.Ok).click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog = make_dialog({})
    dialog.open()
    _button(dialog, QDialogButtonBox.StandardButton.Cancel).click()
    assert dialog.result() == QDialog.DialogCode.Rejected


# ---------------------------------------------------------------------------
# MainWindow.open_settings_dialog
# ---------------------------------------------------------------------------


class Script:
    """Replaces SettingsDialog.exec: set widgets, then press OK or Cancel."""

    def __init__(self, values=None, button=QDialogButtonBox.StandardButton.Ok) -> None:
        self.values = values
        self.button = button
        self.dialogs = []

    def exec(self, dialog: SettingsDialog) -> int:
        self.dialogs.append(_shown(dialog))
        dialog.open()
        if self.values is not None:
            _set(dialog, self.values)
        _button(dialog, self.button).click()
        return dialog.result()


@pytest.fixture
def script(monkeypatch):
    def _install(values=None, button=QDialogButtonBox.StandardButton.Ok) -> Script:
        drv = Script(values, button)
        monkeypatch.setattr(SettingsDialog, "exec", lambda self: drv.exec(self))
        return drv

    return _install


def _saved(isolated_config) -> dict:
    return json.loads(isolated_config.read_text(encoding="utf-8"))


def test_main_window_ok_saves_the_settings(main_window, script, isolated_config):
    drv = script(CUSTOM)
    main_window.settings_action.trigger()
    assert drv.dialogs == [_config.default_merge_table_mouse_actions()]
    assert main_window._cfg["merge_table_mouse_actions"] == CUSTOM
    assert _saved(isolated_config)["merge_table_mouse_actions"] == CUSTOM
    assert main_window.statusBar().currentMessage() == "Settings saved"


def test_main_window_reopens_with_the_saved_values(main_window, script):
    script(CUSTOM)
    main_window.open_settings_dialog()
    drv = script(None)
    main_window.open_settings_dialog()
    assert drv.dialogs == [CUSTOM]


def test_main_window_cancel_changes_nothing(main_window, script, isolated_config):
    before_cfg = json.loads(json.dumps(main_window._cfg))
    before_file = isolated_config.read_bytes() if isolated_config.exists() else None
    main_window.statusBar().showMessage("unchanged")
    script(CUSTOM, QDialogButtonBox.StandardButton.Cancel)
    main_window.open_settings_dialog()
    assert main_window._cfg == before_cfg
    after_file = isolated_config.read_bytes() if isolated_config.exists() else None
    assert after_file == before_file
    assert main_window.statusBar().currentMessage() == "unchanged"


def test_main_window_ok_keeps_other_settings(main_window, script, isolated_config):
    main_window._cfg["confirm_on_delete"] = False
    main_window._cfg["allowed_ratios"] = "1:1, 3:1"
    script(CUSTOM)
    main_window.open_settings_dialog()
    saved = _saved(isolated_config)
    assert saved["confirm_on_delete"] is False
    assert saved["allowed_ratios"] == "1:1, 3:1"


def test_fixup_dialog_uses_the_new_settings(main_window, script, monkeypatch):
    script(CUSTOM)
    main_window.open_settings_dialog()

    captured = []

    def _fake_open(**kwargs):
        captured.append(kwargs)
        return "close"

    monkeypatch.setattr(fixup_controller, "open_fixup_dialog_for_image", _fake_open)
    main_window.open_fixup_dialog()
    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["merge_table_double_click_action_enabled"] is False
    assert kwargs["merge_table_swipe_actions_enabled"] is True
    assert kwargs["merge_table_horizontal_scroll_actions_enabled"] is True
    assert kwargs["merge_table_horizontal_scroll_reverse_enabled"] is True
    assert kwargs["merge_table_horizontal_scroll_stop_idle_seconds"] == 1.25
    assert kwargs["merge_table_horizontal_scroll_row_target_mode"] == _config.MERGE_TABLE_HSCROLL_TARGET_SELECTED_ROW
