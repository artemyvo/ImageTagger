from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# config.json lives in the project root (one level above this package directory)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

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
    "last_selected_image": "",
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
    normalized["last_selected_image"] = _normalize_string(
        data.get("last_selected_image"),
        _DEFAULTS["last_selected_image"],
    )
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
