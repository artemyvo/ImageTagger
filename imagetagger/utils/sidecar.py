from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from imagetagger.utils.io_utils import atomic_write_text


def get_sidecar_json_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


# ---------------------------------------------------------------------------
# Module-level sidecar cache: { image_path -> (mtime, SidecarData) }
# Keyed on the *image* path; internally translates to the .json sidecar path.
# Thread-safe via _sidecar_cache_lock (FolderLoadWorker uses a thread pool).
# ---------------------------------------------------------------------------
_sidecar_cache: dict[Path, tuple[float, SidecarData]] = {}
_sidecar_cache_lock = threading.Lock()


@dataclass
class SidecarData:
    # Committed LLM output
    description: str = ""
    reasoning: str = ""

    # Pending fixup fields — None means the field is absent (no pending fixup)
    fixup_issues: str | None = None
    fixup_tags: list[str] | None = None
    fixup_description: str | None = None
    fixup_model: str | None = None
    fixup_date: str | None = None
    ai_find_matches: list[str] | None = None
    vision_tags: list[str] | None = None
    vision_caption: str | None = None

    # User-review stamp — ISO-8601 UTC datetime set when the user completes a merge
    validated: str | None = None
    # Who performed the validation: "user" when resolved via merge dialog, model name otherwise
    validated_by: str | None = None

    @property
    def has_pending_fixup(self) -> bool:
        return bool(
            self.fixup_issues
            or self.fixup_tags
            or self.fixup_description
            or self.ai_find_matches
            or self.vision_tags
            or self.vision_caption
        )


def read_sidecar_data(image_path: Path) -> SidecarData:
    path = get_sidecar_json_path(image_path)

    # Fast-path: return cached negative result without a stat().
    # Safe because write_sidecar_data always pops this entry on any write.
    with _sidecar_cache_lock:
        cached = _sidecar_cache.get(image_path)
    if cached is not None and cached[0] is None:
        return cached[1]

    try:
        mtime = path.stat().st_mtime
    except OSError:
        empty = SidecarData()
        with _sidecar_cache_lock:
            _sidecar_cache[image_path] = (None, empty)
        return empty

    if cached is not None and cached[0] == mtime:
        return cached[1]

    data = _parse_sidecar_file(path)
    with _sidecar_cache_lock:
        _sidecar_cache[image_path] = (mtime, data)
    return data


def _parse_sidecar_file(path: Path) -> SidecarData:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return SidecarData()

        def _str(key: str) -> str:
            return str(payload.get(key, "") or "")

        def _str_or_none(key: str) -> str | None:
            val = payload.get(key)
            if val is None:
                return None
            s = str(val).strip() if val else ""
            return s if s else None

        def _list_or_none(key: str) -> list[str] | None:
            val = payload.get(key)
            if val is None:
                return None
            if isinstance(val, list):
                result = [str(item) for item in val if item]
                return result if result else None
            return None

        return SidecarData(
            description=_str("description"),
            reasoning=_str("reasoning"),
            fixup_issues=_str_or_none("fixup_issues"),
            fixup_tags=_list_or_none("fixup_tags"),
            fixup_description=_str_or_none("fixup_description"),
            fixup_model=_str_or_none("fixup_model"),
            fixup_date=_str_or_none("fixup_date"),
            ai_find_matches=_list_or_none("ai_find_matches"),
            vision_tags=_list_or_none("vision_tags"),
            vision_caption=_str_or_none("vision_caption"),
            validated=_str_or_none("validated"),
            validated_by=_str_or_none("validated_by"),
        )
    except Exception:
        return SidecarData()


def write_sidecar_data(image_path: Path, data: SidecarData) -> None:
    path = get_sidecar_json_path(image_path)
    payload: dict[str, Any] = {
        "description": data.description,
        "reasoning": data.reasoning,
    }
    if data.fixup_issues is not None:
        payload["fixup_issues"] = data.fixup_issues
    if data.fixup_tags is not None:
        payload["fixup_tags"] = data.fixup_tags
    if data.fixup_description is not None:
        payload["fixup_description"] = data.fixup_description
    if data.fixup_model is not None:
        payload["fixup_model"] = data.fixup_model
    if data.fixup_date is not None:
        payload["fixup_date"] = data.fixup_date
    if data.ai_find_matches is not None:
        payload["ai_find_matches"] = data.ai_find_matches
    if data.vision_tags is not None:
        payload["vision_tags"] = data.vision_tags
    if data.vision_caption is not None:
        payload["vision_caption"] = data.vision_caption
    if data.validated is not None:
        payload["validated"] = data.validated
    if data.validated_by is not None:
        payload["validated_by"] = data.validated_by
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, content, encoding="utf-8")
    with _sidecar_cache_lock:
        _sidecar_cache.pop(image_path, None)
