from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from PyQt6.QtWidgets import QDialog, QMessageBox, QStyle, QWidget

from imagetagger.utils.annotations import sanitize_description_text, sanitize_tag_text
from imagetagger.utils.fixup_parser import FixupData
from imagetagger.utils.sidecar import (
    SidecarData,
    get_sidecar_json_path,
    read_sidecar_data,
    write_sidecar_data,
    write_sidecar_data_async,
)
from imagetagger.providers.llm_provider import VisionLlmProvider, VisionLlmSession
from imagetagger.ui.merge_dialog import FixupDialog


def write_fixup_sidecar(
    image_path: Path,
    issues: str | None,
    tags: list[str] | None,
    description: str | None,
    model: str | None = None,
    date: str | None = None,
) -> None:
    data = read_sidecar_data(image_path)
    data.fixup_issues = issues or None
    data.fixup_tags = tags or None
    data.fixup_description = description or None
    data.fixup_model = model or None
    data.fixup_date = date or None
    write_sidecar_data(image_path, data)


def record_ai_find_match_for_image(
    image_path: Path,
    query: str,
    normalize_annotation: Callable[[str], str] | None = None,
) -> None:
    normalized = query.strip()
    if normalize_annotation is not None:
        normalized = normalize_annotation(normalized)
    normalized = normalized.strip().lower()
    if not normalized:
        raise OSError("AI Find query cannot be empty.")
    data = read_sidecar_data(image_path)
    matches = list(data.ai_find_matches or [])
    if normalized not in matches:
        matches.append(normalized)
        data.ai_find_matches = matches
        write_sidecar_data(image_path, data)


def record_refine_result_for_image(
    image_path: Path,
    tags: list[str],
    caption: str,
) -> None:
    normalized_tags = [sanitize_tag_text(t) for t in tags if sanitize_tag_text(t)]
    normalized_caption = sanitize_description_text(caption)
    data = read_sidecar_data(image_path)
    data.vision_tags = normalized_tags or None
    data.vision_caption = normalized_caption or None
    write_sidecar_data(image_path, data)


def clear_fixup_sidecar(image_path: Path) -> None:
    from datetime import datetime, timezone
    data = read_sidecar_data(image_path)
    data.fixup_issues = None
    data.fixup_tags = None
    data.fixup_description = None
    data.ai_find_matches = None
    data.vision_tags = None
    data.vision_caption = None
    data.validated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data.validated_by = "user"
    write_sidecar_data_async(image_path, data)


def clear_validation_fields_sidecar(image_path: Path, model: str | None = None, date: str | None = None) -> None:
    """Clear only the validation fixup fields, preserving ai_find_matches and vision data."""
    from datetime import datetime, timezone
    data = read_sidecar_data(image_path)
    data.fixup_issues = None
    data.fixup_tags = None
    data.fixup_description = None
    data.fixup_model = model or None
    data.fixup_date = date or None
    data.validated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data.validated_by = model or None
    write_sidecar_data(image_path, data)


def delete_sidecar_for_image(image_path: Path) -> None:
    path = get_sidecar_json_path(image_path)
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
    provider: VisionLlmProvider | None = None,
    regenerate_tags_enabled: bool = True,
    regenerate_description_enabled: bool = True,
    regenerate_timeout_seconds: int = 300,
    regenerate_retry_count: int = 3,
    regenerate_max_resolution_mpx: float = 5.0,
    regenerate_model_name: str = "",
    regenerate_model_endpoint: str = "",
    regenerate_user_hint: str = "",
    merge_table_double_click_action_enabled: bool = True,
    merge_table_swipe_actions_enabled: bool = False,
    merge_table_horizontal_scroll_actions_enabled: bool = False,
    merge_table_horizontal_scroll_reverse_enabled: bool = False,
    merge_table_horizontal_scroll_stop_idle_seconds: float = 0.45,
    merge_table_horizontal_scroll_row_target_mode: int = 3,
    delete_image: Callable[[], tuple[bool, bool]] | None = None,
    confirm_delete: bool = True,
    save_regenerate_settings: Callable[[dict[str, int | float | bool | str]], None] | None = None,
    reasoning_lines: int = 5,
) -> Literal["merged", "cancelled", "prev", "next", "missing", "error"]:
    try:
        sidecar = read_sidecar_data(image_path)
    except OSError as exc:
        QMessageBox.warning(parent, "Sidecar read failed", f"Could not read sidecar file:\n{exc}")
        return "error"

    original_sidecar = SidecarData(
        description=sidecar.description,
        reasoning=sidecar.reasoning,
        fixup_issues=sidecar.fixup_issues,
        fixup_tags=list(sidecar.fixup_tags) if sidecar.fixup_tags is not None else None,
        fixup_description=sidecar.fixup_description,
        ai_find_matches=list(sidecar.ai_find_matches) if sidecar.ai_find_matches is not None else None,
        vision_tags=list(sidecar.vision_tags) if sidecar.vision_tags is not None else None,
        vision_caption=sidecar.vision_caption,
    )

    if not sidecar.has_pending_fixup:
        # No pending fixup — open in clean mode: proposed mirrors current so the
        # table shows no differences.  The user can then use Regenerate to bring
        # in new proposed content.
        # Separate description from tags using the same heuristic as the comparison panel.
        clean_description = ""
        clean_tags = list(current_annotations)
        if clean_tags:
            first = clean_tags[0]
            if len(first.split()) >= 5 or len(first) >= 40:
                clean_description = first
                clean_tags = clean_tags[1:]
        fixup_data = FixupData(
            issues="",
            corrected_description=clean_description,
            corrected_description_raw=clean_description,
            corrected_tags=clean_tags,
            search_matches=[],
            vision_tags=[],
            vision_caption="",
            has_headers=False,
        )
    else:
        fixup_data = FixupData(
            issues=sidecar.fixup_issues or "",
            corrected_description=sanitize_description_text(sidecar.fixup_description or ""),
            corrected_description_raw=sidecar.fixup_description or "",
            corrected_tags=list(sidecar.fixup_tags or []),
            search_matches=list(sidecar.ai_find_matches or []),
            vision_tags=list(sidecar.vision_tags or []),
            vision_caption=sidecar.vision_caption or "",
            has_headers=True,
        )

    def clear_fixup() -> bool:
        try:
            clear_fixup_sidecar(image_path)
        except OSError as exc:
            QMessageBox.warning(parent, "Fixup cleanup failed", f"Could not clear sidecar fixup:\n{exc}")
            show_status("Fixup resolution failed")
            refresh_fixup_state(image_path)
            return False
        show_status("Fixup resolved")
        refresh_fixup_state(image_path)
        return True

    def restore_fixup() -> bool:
        try:
            write_sidecar_data(image_path, original_sidecar)
        except OSError as exc:
            QMessageBox.warning(parent, "Fixup restore failed", f"Could not restore sidecar:\n{exc}")
            show_status("Undo failed: could not restore sidecar")
            refresh_fixup_state(image_path)
            return False
        show_status("Fixup restored")
        refresh_fixup_state(image_path)
        return True

    dialog = FixupDialog(
        current_annotations,
        fixup_data,
        image_path,
        title_text,
        apply_annotations,
        clear_fixup,
        restore_fixup,
        can_navigate_prev,
        can_navigate_next,
        tag_suggestions=tag_suggestions,
        normalize_annotation=sanitize_annotation,
        normalize_tag=sanitize_tag_text,
        provider_session=provider_session,
        provider=provider,
        regenerate_tags_enabled=regenerate_tags_enabled,
        regenerate_description_enabled=regenerate_description_enabled,
        regenerate_timeout_seconds=regenerate_timeout_seconds,
        regenerate_retry_count=regenerate_retry_count,
        regenerate_max_resolution_mpx=regenerate_max_resolution_mpx,
        regenerate_model_name=regenerate_model_name,
        regenerate_model_endpoint=regenerate_model_endpoint,
        regenerate_user_hint=regenerate_user_hint,
        merge_table_double_click_action_enabled=merge_table_double_click_action_enabled,
        merge_table_swipe_actions_enabled=merge_table_swipe_actions_enabled,
        merge_table_horizontal_scroll_actions_enabled=merge_table_horizontal_scroll_actions_enabled,
        merge_table_horizontal_scroll_reverse_enabled=merge_table_horizontal_scroll_reverse_enabled,
        merge_table_horizontal_scroll_stop_idle_seconds=merge_table_horizontal_scroll_stop_idle_seconds,
        merge_table_horizontal_scroll_row_target_mode=merge_table_horizontal_scroll_row_target_mode,
        delete_image=delete_image,
        confirm_delete=confirm_delete,
        allow_left_delete=bool(sidecar.fixup_tags or sidecar.fixup_description),
        fixup_tag_keys={
            sanitize_tag_text(t).casefold()
            for t in sidecar.fixup_tags
            if sanitize_tag_text(t)
        } if sidecar.fixup_tags is not None else None,
        reasoning_lines=reasoning_lines,
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
        rp = dialog._regen_panel
        timeout_raw = rp.regenerate_timeout_input.text().strip()
        retry_raw = rp.regenerate_retry_input.text().strip()
        max_resolution_raw = rp.regenerate_max_resolution_input.text().strip()
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
                "tags_enabled": rp.regenerate_tags_checkbox.isChecked(),
                "description_enabled": rp.regenerate_description_checkbox.isChecked(),
                "timeout_seconds": max(1, timeout_value),
                "retry_count": max(0, retry_value),
                "max_resolution_mpx": max_resolution_value,
                "model_name": rp.current_model_name,
                "model_endpoint": rp.current_endpoint,
                "user_hint": rp.current_user_hint,
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

    # Release the dialog's C++ object immediately rather than waiting for
    # the parent window to be destroyed.  All data has been extracted above.
    dialog.deleteLater()

    return outcome

