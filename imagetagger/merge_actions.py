from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from PyQt6.QtWidgets import QDialog, QMessageBox, QWidget

from imagetagger.merge_dialog import FixupDialog, parse_fixup_data


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


def write_fixup_for_image(image_path: Path, content: str) -> Path:
    fixup_path = fixup_path_for_image(image_path)
    fixup_path.write_text(content.strip() + "\n", encoding="utf-8")
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
    ollama_server_url: str = "",
    ollama_model_name: str = "",
    regenerate_tags_enabled: bool = True,
    regenerate_description_enabled: bool = True,
    regenerate_timeout_seconds: int = 300,
    regenerate_retry_count: int = 3,
    save_regenerate_settings: Callable[[dict[str, int | bool]], None] | None = None,
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
            fixup_path.write_text(original_content, encoding="utf-8")
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
        ollama_server_url=ollama_server_url,
        ollama_model_name=ollama_model_name,
        regenerate_tags_enabled=regenerate_tags_enabled,
        regenerate_description_enabled=regenerate_description_enabled,
        regenerate_timeout_seconds=regenerate_timeout_seconds,
        regenerate_retry_count=regenerate_retry_count,
        parent=parent,
    )

    if initial_geometry:
        x = initial_geometry.get("x")
        y = initial_geometry.get("y")
        width = initial_geometry.get("width")
        height = initial_geometry.get("height")
        if all(isinstance(value, int) for value in (x, y, width, height)) and width > 0 and height > 0:
            dialog.setGeometry(x, y, width, height)

    result = dialog.exec()

    if save_regenerate_settings is not None:
        timeout_raw = dialog.regenerate_timeout_input.text().strip()
        retry_raw = dialog.regenerate_retry_input.text().strip()
        try:
            timeout_value = int(timeout_raw) if timeout_raw else max(1, int(regenerate_timeout_seconds))
        except ValueError:
            timeout_value = max(1, int(regenerate_timeout_seconds))
        try:
            retry_value = int(retry_raw) if retry_raw else max(0, int(regenerate_retry_count))
        except ValueError:
            retry_value = max(0, int(regenerate_retry_count))

        save_regenerate_settings(
            {
                "tags_enabled": dialog.regenerate_tags_checkbox.isChecked(),
                "description_enabled": dialog.regenerate_description_checkbox.isChecked(),
                "timeout_seconds": max(1, timeout_value),
                "retry_count": max(0, retry_value),
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