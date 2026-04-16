from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from PyQt6.QtWidgets import QDialog, QMessageBox, QStyle, QWidget

from imagetagger.io_utils import atomic_write_text
from imagetagger.llm_provider import VisionLlmSession
from imagetagger.merge_dialog import FixupDialog, parse_fixup_data


_AI_FIND_SECTION_HEADER = "AI_FIND_MATCHES:"
_KNOWN_FIXUP_HEADERS = {
    "ISSUES:",
    "TAGS:",
    "DESCRIPTION:",
    _AI_FIND_SECTION_HEADER,
}


def fixup_path_for_image(image_path: Path) -> Path:
    return image_path.parent / f"{image_path.name}.fixup"


def legacy_fixup_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".fixup")


def existing_fixup_path_for_image(image_path: Path) -> Path | None:
    preferred = fixup_path_for_image(image_path)
    if preferred.exists():
        return preferred

    legacy = legacy_fixup_path_for_image(image_path)
    if legacy.exists():
        return legacy

    return None


def _dedupe_fixup_tags_content(content: str) -> str:
    """Normalize and deduplicate entries within each TAGS section."""
    lines = content.splitlines()
    if not lines:
        return ""

    output_lines: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("TAGS:"):
            raw_tags: list[str] = []
            inline_content = stripped[5:].strip()
            if inline_content:
                raw_tags.append(inline_content)

            index += 1
            while index < len(lines):
                next_line = lines[index]
                next_stripped = next_line.strip()
                next_upper = next_stripped.upper()
                if (
                    next_upper.startswith("ISSUES:")
                    or next_upper.startswith("TAGS:")
                    or next_upper.startswith("DESCRIPTION:")
                    or next_upper.startswith(_AI_FIND_SECTION_HEADER)
                ):
                    break
                if next_stripped:
                    raw_tags.append(next_stripped)
                index += 1

            unique_tags: list[str] = []
            seen_keys: set[str] = set()
            for value in raw_tags:
                normalized = _normalize_fixup_section_entry(value)
                if not normalized:
                    continue
                key = normalized.casefold()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_tags.append(normalized)

            output_lines.append("TAGS:")
            output_lines.extend(f"- {tag}" for tag in unique_tags)
            continue

        output_lines.append(line)
        index += 1

    return "\n".join(output_lines).strip() + "\n"


def write_fixup_for_image(image_path: Path, content: str) -> Path:
    fixup_path = fixup_path_for_image(image_path)
    cleaned_content = _dedupe_fixup_tags_content(content)
    atomic_write_text(fixup_path, cleaned_content, encoding="utf-8")
    return fixup_path


def _normalize_fixup_section_entry(value: str) -> str:
    text = value.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    return text


def _normalize_ai_find_match_query(
    query: str,
    normalize_annotation: Callable[[str], str] | None = None,
) -> str:
    cleaned_query = _normalize_fixup_section_entry(query)
    if normalize_annotation is not None:
        cleaned_query = normalize_annotation(cleaned_query)
    return cleaned_query.strip().lower()


def record_ai_find_match_for_image(
    image_path: Path,
    query: str,
    normalize_annotation: Callable[[str], str] | None = None,
) -> Path:
    normalized_query = _normalize_ai_find_match_query(query, normalize_annotation)
    if not normalized_query:
        raise OSError("AI Find query cannot be empty.")

    existing_path = existing_fixup_path_for_image(image_path)
    if existing_path is None:
        lines: list[str] = []
    else:
        try:
            lines = existing_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise OSError(f"Could not read fixup file: {exc}") from exc

    header_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip().upper() == _AI_FIND_SECTION_HEADER:
            header_index = index
            break

    section_start = -1
    section_end = -1
    section_entries: list[str] = []
    if header_index is not None:
        section_start = header_index + 1
        section_end = len(lines)
        for index in range(section_start, len(lines)):
            if lines[index].strip().upper() in _KNOWN_FIXUP_HEADERS:
                section_end = index
                break

        for line in lines[section_start:section_end]:
            entry = _normalize_fixup_section_entry(line)
            if entry:
                section_entries.append(entry)

    existing_keys = {entry.casefold() for entry in section_entries}
    if normalized_query.casefold() not in existing_keys:
        section_entries.append(normalized_query)

    new_section_lines = [_AI_FIND_SECTION_HEADER]
    new_section_lines.extend(f"- {entry}" for entry in section_entries)

    if header_index is None:
        output_lines = list(lines)
        while output_lines and not output_lines[-1].strip():
            output_lines.pop()
        if output_lines:
            output_lines.append("")
        output_lines.extend(new_section_lines)
    else:
        output_lines = lines[:header_index] + new_section_lines + lines[section_end:]

    fixup_path = fixup_path_for_image(image_path)
    atomic_write_text(fixup_path, "\n".join(output_lines).strip() + "\n", encoding="utf-8")
    return fixup_path


def clear_fixup_files_for_image(image_path: Path) -> None:
    seen_paths: set[Path] = set()
    for path in (fixup_path_for_image(image_path), legacy_fixup_path_for_image(image_path)):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def open_fixup_dialog_for_image(
    parent: QWidget,
    image_path: Path,
    current_annotations: list[str],
    title_text: str | None,
    parse_tags: Callable[[str], list[str]],
    sanitize_annotation: Callable[[str], str],
    apply_annotations: Callable[[list[str], str], None],
    show_status: Callable[[str], None],
    refresh_fixup_state: Callable[[Path], None],
    initial_geometry: dict[str, int] | None = None,
    save_geometry: Callable[[dict[str, int]], None] | None = None,
    can_navigate_prev: bool = False,
    can_navigate_next: bool = False,
    tag_suggestions: list[str] | None = None,
    provider_session: VisionLlmSession | None = None,
    regenerate_tags_enabled: bool = True,
    regenerate_description_enabled: bool = True,
    regenerate_timeout_seconds: int = 300,
    regenerate_retry_count: int = 3,
    regenerate_max_resolution_mpx: float = 5.0,
    save_regenerate_settings: Callable[[dict[str, int | float | bool]], None] | None = None,
) -> Literal["merged", "cancelled", "prev", "next", "missing", "error"]:
    fixup_path = existing_fixup_path_for_image(image_path)
    if fixup_path is None:
        refresh_fixup_state(image_path)
        return "missing"

    try:
        content = fixup_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        QMessageBox.warning(parent, "Fixup read failed", f"Could not read fixup file:\n{exc}")
        return "error"

    def clear_fixup() -> bool:
        try:
            fixup_path.unlink(missing_ok=True)
        except OSError as exc:
            QMessageBox.warning(parent, "Fixup cleanup failed", f"Could not remove fixup file:\n{exc}")
            show_status("Fixup resolution failed: could not remove .fixup")
            refresh_fixup_state(image_path)
            return False

        show_status("Fixup resolved and .fixup removed")
        refresh_fixup_state(image_path)
        return True

    def restore_fixup(original_content: str) -> bool:
        try:
            atomic_write_text(fixup_path, original_content, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(parent, "Fixup restore failed", f"Could not restore fixup file:\n{exc}")
            show_status("Undo failed: could not restore .fixup")
            refresh_fixup_state(image_path)
            return False

        show_status("Fixup restored")
        refresh_fixup_state(image_path)
        return True

    dialog = FixupDialog(
        current_annotations,
        parse_fixup_data(content, parse_tags, sanitize_annotation),
        image_path,
        title_text,
        apply_annotations,
        content,
        clear_fixup,
        restore_fixup,
        can_navigate_prev,
        can_navigate_next,
        tag_suggestions=tag_suggestions,
        normalize_annotation=sanitize_annotation,
        provider_session=provider_session,
        regenerate_tags_enabled=regenerate_tags_enabled,
        regenerate_description_enabled=regenerate_description_enabled,
        regenerate_timeout_seconds=regenerate_timeout_seconds,
        regenerate_retry_count=regenerate_retry_count,
        regenerate_max_resolution_mpx=regenerate_max_resolution_mpx,
        parent=parent,
    )

    if initial_geometry:
        x = initial_geometry.get("x")
        y = initial_geometry.get("y")
        width = initial_geometry.get("width")
        height = initial_geometry.get("height")
        if all(isinstance(value, int) for value in (x, y, width, height)) and width > 0 and height > 0:
            screen = dialog.screen()
            available = screen.availableGeometry() if screen is not None else None
            if available is None:
                dialog.setGeometry(x, y, width, height)
            else:
                clamped_width = min(width, available.width())
                clamped_height = min(height, available.height())
                max_x = available.left() + max(0, available.width() - clamped_width)
                max_y = available.top() + max(0, available.height() - clamped_height)
                clamped_x = min(max(x, available.left()), max_x)

                # Keep enough top inset so native title-bar controls remain reachable.
                title_bar_height = dialog.style().pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight, None, dialog)
                if title_bar_height <= 0:
                    title_bar_height = 32
                min_y = min(available.top() + title_bar_height, max_y)
                clamped_y = min(max(y, min_y), max_y)
                dialog.setGeometry(clamped_x, clamped_y, clamped_width, clamped_height)

    result = dialog.exec()

    if save_regenerate_settings is not None:
        timeout_raw = dialog.regenerate_timeout_input.text().strip()
        retry_raw = dialog.regenerate_retry_input.text().strip()
        max_resolution_raw = dialog.regenerate_max_resolution_input.text().strip()
        try:
            timeout_value = int(timeout_raw) if timeout_raw else max(1, int(regenerate_timeout_seconds))
        except ValueError:
            timeout_value = max(1, int(regenerate_timeout_seconds))
        try:
            retry_value = int(retry_raw) if retry_raw else max(0, int(regenerate_retry_count))
        except ValueError:
            retry_value = max(0, int(regenerate_retry_count))
        try:
            max_resolution_value = float(max_resolution_raw) if max_resolution_raw else float(regenerate_max_resolution_mpx)
            if max_resolution_value <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            try:
                max_resolution_value = float(regenerate_max_resolution_mpx)
                if max_resolution_value <= 0:
                    raise ValueError()
            except (TypeError, ValueError):
                max_resolution_value = 5.0

        save_regenerate_settings(
            {
                "tags_enabled": dialog.regenerate_tags_checkbox.isChecked(),
                "description_enabled": dialog.regenerate_description_checkbox.isChecked(),
                "timeout_seconds": max(1, timeout_value),
                "retry_count": max(0, retry_value),
                "max_resolution_mpx": max_resolution_value,
            }
        )

    if save_geometry is not None:
        geometry = dialog.geometry()
        save_geometry(
            {
                "x": int(geometry.x()),
                "y": int(geometry.y()),
                "width": int(geometry.width()),
                "height": int(geometry.height()),
            }
        )

    if result == FixupDialog.NAVIGATE_PREV_CODE:
        outcome: Literal["merged", "cancelled", "prev", "next", "missing", "error"] = "prev"
    elif result == FixupDialog.NAVIGATE_NEXT_CODE:
        outcome = "next"
    else:
        outcome = "cancelled"

    return outcome