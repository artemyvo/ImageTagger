from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# config.json lives in the project root (one level above this package directory)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

# Horizontal scroll row targeting modes for merge dialog actions.
MERGE_TABLE_HSCROLL_TARGET_POINTER_ROW = 1
MERGE_TABLE_HSCROLL_TARGET_SELECTED_ROW = 2
MERGE_TABLE_HSCROLL_TARGET_POINTER_ON_SELECTED = 3
MERGE_TABLE_HSCROLL_TARGET_ALLOWED_MODES = {
    MERGE_TABLE_HSCROLL_TARGET_POINTER_ROW,
    MERGE_TABLE_HSCROLL_TARGET_SELECTED_ROW,
    MERGE_TABLE_HSCROLL_TARGET_POINTER_ON_SELECTED,
}

_DEFAULT_MERGE_TABLE_MOUSE_ACTIONS: dict[str, Any] = {
    "double_click_action_enabled": True,
    "swipe_actions_enabled": False,
    "horizontal_scroll_actions_enabled": False,
    "horizontal_scroll_reverse_enabled": False,
    "horizontal_scroll_stop_idle_seconds": 0.45,
    "horizontal_scroll_row_target_mode": MERGE_TABLE_HSCROLL_TARGET_POINTER_ON_SELECTED,
}

_DEFAULTS: dict = {
    "last_open_directory": "",
    "main_window_geometry": {},
    "llm_endpoint": "",
    "llm_model": "",
    "llm_max_resolution_mpx": 5,
    "llm_threads": 1,
    "llm_auto_max_threads": 48,
    "llm_auto_warmup_items": 4,
    "llm_auto_scale_up_every": 3,
    "merge_dialog_geometry": {},
    "font_point_size": 0,
    "directory_loader_max_threads": 8,
    "confirm_on_delete": True,
    "last_selected_image": "",
    "debug_regenerate_prompt_console": False,
    "merge_table_mouse_actions": dict(_DEFAULT_MERGE_TABLE_MOUSE_ACTIONS),
    "agent_roles": {},
}


def _normalize_string(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default


def _normalize_int(value: Any, default: int, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _normalize_number(value: Any, default: float, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric_value = float(value)
    if minimum is not None and numeric_value < minimum:
        return default
    return numeric_value


def _normalize_bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _normalize_int_choice(value: Any, default: int, allowed: set[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if value in allowed else default


def _normalize_merge_table_mouse_actions(data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("merge_table_mouse_actions")
    if not isinstance(raw, dict):
        raw = {}

    def _value(key: str, legacy_key: str) -> Any:
        if key in raw:
            return raw.get(key)
        return data.get(legacy_key)

    return {
        "double_click_action_enabled": _normalize_bool(
            _value("double_click_action_enabled", "merge_table_double_click_action_enabled"),
            bool(_DEFAULT_MERGE_TABLE_MOUSE_ACTIONS["double_click_action_enabled"]),
        ),
        "swipe_actions_enabled": _normalize_bool(
            _value("swipe_actions_enabled", "merge_table_swipe_actions_enabled"),
            bool(_DEFAULT_MERGE_TABLE_MOUSE_ACTIONS["swipe_actions_enabled"]),
        ),
        "horizontal_scroll_actions_enabled": _normalize_bool(
            _value("horizontal_scroll_actions_enabled", "merge_table_horizontal_scroll_actions_enabled"),
            bool(_DEFAULT_MERGE_TABLE_MOUSE_ACTIONS["horizontal_scroll_actions_enabled"]),
        ),
        "horizontal_scroll_reverse_enabled": _normalize_bool(
            _value("horizontal_scroll_reverse_enabled", "merge_table_horizontal_scroll_reverse_enabled"),
            bool(_DEFAULT_MERGE_TABLE_MOUSE_ACTIONS["horizontal_scroll_reverse_enabled"]),
        ),
        "horizontal_scroll_stop_idle_seconds": _normalize_number(
            _value("horizontal_scroll_stop_idle_seconds", "merge_table_horizontal_scroll_stop_idle_seconds"),
            float(_DEFAULT_MERGE_TABLE_MOUSE_ACTIONS["horizontal_scroll_stop_idle_seconds"]),
            minimum=0.0,
        ),
        "horizontal_scroll_row_target_mode": _normalize_int_choice(
            _value("horizontal_scroll_row_target_mode", "merge_table_horizontal_scroll_row_target_mode"),
            int(_DEFAULT_MERGE_TABLE_MOUSE_ACTIONS["horizontal_scroll_row_target_mode"]),
            MERGE_TABLE_HSCROLL_TARGET_ALLOWED_MODES,
        ),
    }


def _normalize_geometry(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}

    x = value.get("x")
    y = value.get("y")
    width = value.get("width")
    height = value.get("height")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in (x, y, width, height)):
        return {}
    if width <= 0 or height <= 0:
        return {}

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def _normalize_loaded_config(data: Any) -> dict:
    if not isinstance(data, dict):
        return dict(_DEFAULTS)

    normalized = dict(_DEFAULTS)
    normalized["last_open_directory"] = _normalize_string(
        data.get("last_open_directory"),
        _DEFAULTS["last_open_directory"],
    )
    normalized["main_window_geometry"] = _normalize_geometry(data.get("main_window_geometry"))
    normalized["llm_endpoint"] = _normalize_string(
        data.get("llm_endpoint", data.get("ollama_server")),
        _DEFAULTS["llm_endpoint"],
    )
    normalized["llm_model"] = _normalize_string(
        data.get("llm_model", data.get("ollama_model")),
        _DEFAULTS["llm_model"],
    )
    normalized["llm_max_resolution_mpx"] = _normalize_number(
        data.get("llm_max_resolution_mpx", data.get("ollama_max_resolution_mpx")),
        float(_DEFAULTS["llm_max_resolution_mpx"]),
        minimum=0.01,
    )
    normalized["llm_threads"] = _normalize_int(
        data.get("llm_threads", data.get("ollama_threads")),
        _DEFAULTS["llm_threads"],
        minimum=0,
    )
    normalized["llm_auto_max_threads"] = _normalize_int(
        data.get("llm_auto_max_threads", data.get("ollama_auto_max_threads")),
        _DEFAULTS["llm_auto_max_threads"],
        minimum=1,
    )
    normalized["llm_auto_warmup_items"] = _normalize_int(
        data.get("llm_auto_warmup_items", data.get("ollama_auto_warmup_items")),
        _DEFAULTS["llm_auto_warmup_items"],
        minimum=0,
    )
    normalized["llm_auto_scale_up_every"] = _normalize_int(
        data.get("llm_auto_scale_up_every", data.get("ollama_auto_scale_up_every")),
        _DEFAULTS["llm_auto_scale_up_every"],
        minimum=1,
    )
    normalized["merge_dialog_geometry"] = _normalize_geometry(data.get("merge_dialog_geometry"))
    normalized["font_point_size"] = _normalize_int(
        data.get("font_point_size"),
        _DEFAULTS["font_point_size"],
        minimum=0,
    )
    normalized["directory_loader_max_threads"] = _normalize_int(
        data.get("directory_loader_max_threads"),
        _DEFAULTS["directory_loader_max_threads"],
        minimum=1,
    )
    normalized["confirm_on_delete"] = _normalize_bool(
        data.get("confirm_on_delete"),
        _DEFAULTS["confirm_on_delete"],
    )
    normalized["last_selected_image"] = _normalize_string(
        data.get("last_selected_image"),
        _DEFAULTS["last_selected_image"],
    )
    normalized["debug_regenerate_prompt_console"] = _normalize_bool(
        data.get("debug_regenerate_prompt_console"),
        _DEFAULTS["debug_regenerate_prompt_console"],
    )
    normalized["merge_table_mouse_actions"] = _normalize_merge_table_mouse_actions(data)

    raw_roles = data.get("agent_roles")
    if isinstance(raw_roles, dict):
        normalized["agent_roles"] = {
            k: v for k, v in raw_roles.items()
            if isinstance(k, str) and isinstance(v, str)
        }
    else:
        normalized["agent_roles"] = {}

    return normalized


def load() -> dict:
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            return _normalize_loaded_config(data)
        except (OSError, json.JSONDecodeError):
            pass
    return dict(_DEFAULTS)


def save(cfg: dict) -> None:
    try:
        normalized = _normalize_loaded_config(cfg)
        if not normalized.get("last_selected_image"):
            normalized.pop("last_selected_image", None)
        _CONFIG_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
