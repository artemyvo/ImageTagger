from __future__ import annotations

import json
from pathlib import Path

# config.json lives in the project root (one level above this package directory)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

_DEFAULTS: dict = {
    "last_open_directory": "",
    "main_window_geometry": {},
    "ollama_server": "",
    "ollama_model": "",
    "ollama_max_resolution_mpx": 5,
    "ollama_threads": 1,
    "ollama_auto_max_threads": 48,
    "ollama_auto_warmup_items": 4,
    "ollama_auto_scale_up_every": 3,
    "merge_dialog_geometry": {},
    "font_point_size": 0,
    "directory_loader_max_threads": 8,
}


def load() -> dict:
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {**_DEFAULTS, **data}
        except (OSError, json.JSONDecodeError):
            pass
    return dict(_DEFAULTS)


def save(cfg: dict) -> None:
    try:
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
