"""config.json: defaults, round trip, and normalization of hand-edited or legacy values."""
from __future__ import annotations

import copy
import json

import pytest

from imagetagger import config
from imagetagger.config import _DEFAULTS, _normalize_loaded_config


def _write(path, payload) -> None:
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def test_config_path_is_isolated(isolated_config, tmp_path_factory):
    real = config.Path(config.__file__).resolve().parent.parent / "config.json"
    assert config._CONFIG_PATH == isolated_config
    assert config._CONFIG_PATH != real
    assert tmp_path_factory.getbasetemp() in config._CONFIG_PATH.parents


def test_load_without_file_gives_defaults(isolated_config):
    assert not isolated_config.exists()
    assert config.load() == _DEFAULTS


def test_save_load_round_trip(isolated_config):
    cfg = config.load()
    cfg.update(
        last_open_directory="/data/images",
        main_window_geometry={"x": -10, "y": 20, "width": 800, "height": 600},
        llm_endpoint="http://localhost:11434",
        llm_model="llava",
        merge_dialog_tags_temperature=1.5,
        merge_dialog_description_temperature=0.0,
        llm_think_tags=True,
        llm_think_description=True,
        llm_max_resolution_mpx=2.5,
        llm_threads=0,
        llm_auto_max_threads=4,
        llm_auto_warmup_items=0,
        llm_auto_scale_up_every=2,
        merge_dialog_geometry={"x": 1, "y": 2, "width": 3, "height": 4},
        font_point_size=13,
        directory_loader_max_threads=2,
        confirm_on_delete=False,
        last_selected_image="/data/images/a.png",
        debug_regenerate_prompt_console=True,
        debug_prompts=True,
        merge_table_mouse_actions={
            "double_click_action_enabled": False,
            "swipe_actions_enabled": True,
            "horizontal_scroll_actions_enabled": True,
            "horizontal_scroll_reverse_enabled": True,
            "horizontal_scroll_stop_idle_seconds": 1.25,
            "horizontal_scroll_row_target_mode": config.MERGE_TABLE_HSCROLL_TARGET_POINTER_ROW,
        },
        agent_roles={"tags": "tagger"},
        merge_dialog_reasoning_lines=9,
        allowed_ratios="1:1, 16:9",
    )
    config.save(cfg)
    assert isolated_config.exists()
    assert config.load() == cfg


def test_saved_file_is_readable_json_with_unicode(isolated_config):
    cfg = config.load()
    cfg["last_open_directory"] = "/data/café"
    config.save(cfg)
    text = isolated_config.read_text(encoding="utf-8")
    assert "café" in text
    assert json.loads(text)["last_open_directory"] == "/data/café"


def test_save_omits_empty_last_selected_image(isolated_config):
    config.save(config.load())
    assert "last_selected_image" not in json.loads(isolated_config.read_text(encoding="utf-8"))
    assert config.load()["last_selected_image"] == ""


def test_save_drops_unknown_keys(isolated_config):
    cfg = config.load()
    cfg["no_such_setting"] = 1
    config.save(cfg)
    assert "no_such_setting" not in json.loads(isolated_config.read_text(encoding="utf-8"))


def test_save_does_not_mutate_the_callers_dict(isolated_config):
    cfg = config.load()
    cfg["no_such_setting"] = 1
    before = json.loads(json.dumps(cfg))
    config.save(cfg)
    assert cfg == before


def test_save_swallows_os_errors(isolated_config, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_CONFIG_PATH", tmp_path / "missing-dir" / "config.json")
    config.save(config.load())  # does not raise
    assert not (tmp_path / "missing-dir").exists()


@pytest.mark.parametrize("text", ["", "{not json", "[1, 2]", '"x"', "null", "42"])
def test_corrupt_or_non_dict_file_gives_defaults(isolated_config, text):
    _write(isolated_config, text)
    assert config.load() == _DEFAULTS


def test_non_utf8_file_gives_defaults(isolated_config):
    isolated_config.write_bytes(b'{"llm_model": "\xff\xfe"}')
    assert config.load() == _DEFAULTS


def test_mutating_loaded_defaults_does_not_leak_into_later_loads(isolated_config, monkeypatch):
    # Work on a private copy so the leak, while it exists, cannot reach other tests.
    monkeypatch.setattr(config, "_DEFAULTS", copy.deepcopy(config._DEFAULTS))
    cfg = config.load()
    cfg["agent_roles"]["tags"] = "tagger"  # as LlmController does when a role is picked
    cfg["merge_table_mouse_actions"]["swipe_actions_enabled"] = True
    fresh = config.load()
    assert fresh["agent_roles"] == {}
    assert fresh["merge_table_mouse_actions"] == config.default_merge_table_mouse_actions()


def test_default_merge_table_mouse_actions_is_a_fresh_copy():
    first = config.default_merge_table_mouse_actions()
    first["swipe_actions_enabled"] = True
    assert config.default_merge_table_mouse_actions()["swipe_actions_enabled"] is False


# ---------------------------------------------------------------------------
# _normalize_loaded_config
# ---------------------------------------------------------------------------


def test_normalize_empty_dict_gives_defaults():
    assert _normalize_loaded_config({}) == _DEFAULTS


def test_normalize_drops_unknown_keys():
    assert "mystery" not in _normalize_loaded_config({"mystery": 1})


@pytest.mark.parametrize(
    "key, value",
    [
        ("last_open_directory", 5),
        ("last_open_directory", None),
        ("llm_endpoint", ["http://x"]),
        ("llm_model", 3),
        ("last_selected_image", {}),
        ("merge_dialog_tags_temperature", "0.5"),
        ("merge_dialog_tags_temperature", -0.1),
        ("merge_dialog_tags_temperature", 2.01),
        ("merge_dialog_tags_temperature", True),
        ("merge_dialog_description_temperature", 3),
        ("llm_think_tags", 1),
        ("llm_think_description", "yes"),
        ("llm_max_resolution_mpx", 0),
        ("llm_max_resolution_mpx", -1),
        ("llm_max_resolution_mpx", "5"),
        ("llm_threads", -1),
        ("llm_threads", 2.0),
        ("llm_threads", True),
        ("llm_auto_max_threads", 0),
        ("llm_auto_warmup_items", -1),
        ("llm_auto_scale_up_every", 0),
        ("font_point_size", -1),
        ("font_point_size", "12"),
        ("directory_loader_max_threads", 0),
        ("confirm_on_delete", 0),
        ("debug_regenerate_prompt_console", "true"),
        ("debug_prompts", None),
        ("merge_dialog_reasoning_lines", 0),
        ("agent_roles", ["tagger"]),
        ("merge_table_mouse_actions", "on"),
    ],
)
def test_normalize_invalid_value_falls_back_to_default(key, value):
    assert _normalize_loaded_config({key: value})[key] == _DEFAULTS[key]


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("merge_dialog_tags_temperature", 0, 0.0),
        ("merge_dialog_tags_temperature", 2, 2.0),
        ("llm_max_resolution_mpx", 0.01, 0.01),
        ("llm_max_resolution_mpx", 12, 12.0),
        ("llm_threads", 0, 0),
        ("llm_auto_warmup_items", 0, 0),
        ("font_point_size", 0, 0),
        ("merge_dialog_reasoning_lines", 1, 1),
    ],
)
def test_normalize_boundary_values_are_kept(key, value, expected):
    result = _normalize_loaded_config({key: value})[key]
    assert result == expected
    assert type(result) is type(expected)


def test_normalize_non_dict_gives_defaults():
    assert _normalize_loaded_config(["x"]) == _DEFAULTS
    assert _normalize_loaded_config(None) == _DEFAULTS


def test_normalize_agent_roles_keeps_only_string_pairs():
    roles = {"tags": "tagger", "desc": 5, "x": None, "caption": "writer"}
    assert _normalize_loaded_config({"agent_roles": roles})["agent_roles"] == {"tags": "tagger", "caption": "writer"}


def test_normalize_allowed_ratios_invalid_falls_back():
    assert _normalize_loaded_config({"allowed_ratios": 42})["allowed_ratios"] == _DEFAULTS["allowed_ratios"]


# ---------------------------------------------------------------------------
# Legacy keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "legacy_key, key, value",
    [
        ("ollama_server", "llm_endpoint", "http://old:11434"),
        ("ollama_model", "llm_model", "llava:13b"),
        ("ollama_max_resolution_mpx", "llm_max_resolution_mpx", 3.0),
        ("ollama_threads", "llm_threads", 6),
        ("ollama_auto_max_threads", "llm_auto_max_threads", 16),
        ("ollama_auto_warmup_items", "llm_auto_warmup_items", 2),
        ("ollama_auto_scale_up_every", "llm_auto_scale_up_every", 5),
    ],
)
def test_legacy_ollama_keys_migrate(isolated_config, legacy_key, key, value):
    _write(isolated_config, {legacy_key: value})
    cfg = config.load()
    assert cfg[key] == value
    assert legacy_key not in cfg

    config.save(cfg)
    on_disk = json.loads(isolated_config.read_text(encoding="utf-8"))
    assert on_disk[key] == value
    assert legacy_key not in on_disk


def test_new_key_wins_over_legacy_key():
    cfg = _normalize_loaded_config({"llm_model": "new", "ollama_model": "old", "llm_threads": 2, "ollama_threads": 7})
    assert cfg["llm_model"] == "new"
    assert cfg["llm_threads"] == 2


# ---------------------------------------------------------------------------
# merge_table_mouse_actions
# ---------------------------------------------------------------------------


def test_mouse_actions_missing_gives_defaults():
    assert _normalize_loaded_config({})["merge_table_mouse_actions"] == config.default_merge_table_mouse_actions()


def test_mouse_actions_partial_dict_fills_defaults_and_drops_unknown():
    result = _normalize_loaded_config(
        {"merge_table_mouse_actions": {"swipe_actions_enabled": True, "bogus": 1}}
    )["merge_table_mouse_actions"]
    expected = config.default_merge_table_mouse_actions()
    expected["swipe_actions_enabled"] = True
    assert result == expected


@pytest.mark.parametrize(
    "key, value",
    [
        ("double_click_action_enabled", "no"),
        ("swipe_actions_enabled", 1),
        ("horizontal_scroll_actions_enabled", None),
        ("horizontal_scroll_reverse_enabled", 0),
        ("horizontal_scroll_stop_idle_seconds", -0.5),
        ("horizontal_scroll_stop_idle_seconds", "0.3"),
        ("horizontal_scroll_stop_idle_seconds", True),
        ("horizontal_scroll_row_target_mode", 0),
        ("horizontal_scroll_row_target_mode", 4),
        ("horizontal_scroll_row_target_mode", 2.0),
        ("horizontal_scroll_row_target_mode", True),
    ],
)
def test_mouse_actions_invalid_values_fall_back(key, value):
    result = _normalize_loaded_config({"merge_table_mouse_actions": {key: value}})["merge_table_mouse_actions"]
    assert result[key] == config.default_merge_table_mouse_actions()[key]


@pytest.mark.parametrize("mode", sorted(config.MERGE_TABLE_HSCROLL_TARGET_ALLOWED_MODES))
def test_mouse_actions_every_row_target_mode_is_kept(mode):
    result = _normalize_loaded_config({"merge_table_mouse_actions": {"horizontal_scroll_row_target_mode": mode}})
    assert result["merge_table_mouse_actions"]["horizontal_scroll_row_target_mode"] == mode


def test_mouse_actions_idle_seconds_int_becomes_float():
    result = _normalize_loaded_config({"merge_table_mouse_actions": {"horizontal_scroll_stop_idle_seconds": 1}})
    value = result["merge_table_mouse_actions"]["horizontal_scroll_stop_idle_seconds"]
    assert value == 1.0 and isinstance(value, float)


def test_mouse_actions_legacy_top_level_keys_migrate():
    result = _normalize_loaded_config(
        {
            "merge_table_double_click_action_enabled": False,
            "merge_table_swipe_actions_enabled": True,
            "merge_table_horizontal_scroll_actions_enabled": True,
            "merge_table_horizontal_scroll_reverse_enabled": True,
            "merge_table_horizontal_scroll_stop_idle_seconds": 2.0,
            "merge_table_horizontal_scroll_row_target_mode": config.MERGE_TABLE_HSCROLL_TARGET_SELECTED_ROW,
        }
    )
    assert result["merge_table_mouse_actions"] == {
        "double_click_action_enabled": False,
        "swipe_actions_enabled": True,
        "horizontal_scroll_actions_enabled": True,
        "horizontal_scroll_reverse_enabled": True,
        "horizontal_scroll_stop_idle_seconds": 2.0,
        "horizontal_scroll_row_target_mode": config.MERGE_TABLE_HSCROLL_TARGET_SELECTED_ROW,
    }
    assert "merge_table_swipe_actions_enabled" not in result


def test_mouse_actions_nested_key_wins_over_legacy_key():
    result = _normalize_loaded_config(
        {
            "merge_table_swipe_actions_enabled": True,
            "merge_table_mouse_actions": {"swipe_actions_enabled": False},
        }
    )
    assert result["merge_table_mouse_actions"]["swipe_actions_enabled"] is False


def test_mouse_actions_invalid_nested_value_does_not_fall_back_to_legacy():
    """A present-but-invalid nested value means default, not the legacy value."""
    result = _normalize_loaded_config(
        {
            "merge_table_swipe_actions_enabled": True,
            "merge_table_mouse_actions": {"swipe_actions_enabled": "yes"},
        }
    )
    assert result["merge_table_mouse_actions"]["swipe_actions_enabled"] is False


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["main_window_geometry", "merge_dialog_geometry"])
def test_geometry_valid_is_kept_without_extra_keys(key):
    geometry = {"x": -1920, "y": 0, "width": 1, "height": 1, "maximized": True}
    assert _normalize_loaded_config({key: geometry})[key] == {"x": -1920, "y": 0, "width": 1, "height": 1}


@pytest.mark.parametrize("key", ["main_window_geometry", "merge_dialog_geometry"])
@pytest.mark.parametrize(
    "geometry",
    [
        None,
        [0, 0, 100, 100],
        {"x": 0, "y": 0, "width": 100},
        {"x": 0, "y": 0, "width": 100, "height": 0},
        {"x": 0, "y": 0, "width": -5, "height": 100},
        {"x": 0.5, "y": 0, "width": 100, "height": 100},
        {"x": "0", "y": 0, "width": 100, "height": 100},
        {"x": True, "y": 0, "width": 100, "height": 100},
        {"x": 0, "y": 0, "width": 100, "height": None},
    ],
    ids=["none", "list", "missing-height", "zero-height", "negative-width", "float", "string", "bool", "null-height"],
)
def test_geometry_invalid_becomes_empty(key, geometry):
    assert _normalize_loaded_config({key: geometry})[key] == {}
