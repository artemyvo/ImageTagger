from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


@dataclass(frozen=True)
class ExternalEditor:
    id: str
    display_name: str
    launch_target: str
    launch_kind: str = "executable"  # executable | mac_app


def discover_graphics_editors() -> list[ExternalEditor]:
    if sys.platform.startswith("win"):
        return _discover_windows_editors()
    if sys.platform == "darwin":
        return _discover_macos_editors()
    return _discover_linux_editors()


def launch_image_in_editor(editor: ExternalEditor, image_path: Path) -> None:
    image_str = str(image_path)
    if editor.launch_kind == "mac_app":
        subprocess.Popen(["open", "-a", editor.launch_target, image_str])
        return
    subprocess.Popen([editor.launch_target, image_str])


def launch_image_in_system_default(image_path: Path) -> None:
    image_str = str(image_path)
    if sys.platform.startswith("win"):
        os.startfile(image_str)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", image_str])
        return
    subprocess.Popen(["xdg-open", image_str])


def _discover_linux_editors() -> list[ExternalEditor]:
    candidates = [
        ("photoshop", "Adobe Photoshop", ["photoshop"]),
        ("gimp", "GIMP", ["gimp", "gimp-2.10", "gimp-3.0"]),
        ("krita", "Krita", ["krita"]),
        ("pinta", "Pinta", ["pinta"]),
        ("inkscape", "Inkscape", ["inkscape"]),
        ("kolourpaint", "KolourPaint", ["kolourpaint"]),
    ]
    return _discover_from_which(candidates)


def _discover_macos_editors() -> list[ExternalEditor]:
    applications = [
        ("photoshop", "Adobe Photoshop", ["Adobe Photoshop 2026.app", "Adobe Photoshop 2025.app", "Adobe Photoshop.app"]),
        ("gimp", "GIMP", ["GIMP.app"]),
        ("krita", "Krita", ["Krita.app"]),
        ("affinity_photo", "Affinity Photo", ["Affinity Photo 2.app", "Affinity Photo.app"]),
        ("pixelmator", "Pixelmator Pro", ["Pixelmator Pro.app", "Pixelmator.app"]),
    ]
    roots = [Path("/Applications"), Path.home() / "Applications"]

    found: list[ExternalEditor] = []
    seen: set[str] = set()

    for app_id, display_name, bundle_names in applications:
        for root in roots:
            for bundle_name in bundle_names:
                bundle = root / bundle_name
                if bundle.exists() and bundle.is_dir():
                    key = str(bundle).casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(
                        ExternalEditor(
                            id=app_id,
                            display_name=display_name,
                            launch_target=str(bundle),
                            launch_kind="mac_app",
                        )
                    )

    # Adobe Creative Cloud frequently installs Photoshop inside versioned folders,
    # for example /Applications/Adobe Photoshop 2024/Adobe Photoshop 2024.app.
    for bundle in _discover_macos_photoshop_bundles(roots):
        key = str(bundle).casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(
            ExternalEditor(
                id="photoshop",
                display_name=_macos_bundle_display_name(bundle, "Adobe Photoshop"),
                launch_target=str(bundle),
                launch_kind="mac_app",
            )
        )

    found.extend(
        _discover_from_which(
            [
                ("gimp", "GIMP", ["gimp", "gimp-2.10", "gimp-3.0"]),
                ("krita", "Krita", ["krita"]),
                ("inkscape", "Inkscape", ["inkscape"]),
            ],
            seen_targets=seen,
        )
    )

    return found


def _discover_macos_photoshop_bundles(roots: list[Path]) -> list[Path]:
    bundles: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for bundle in root.rglob("Adobe Photoshop*.app"):
                if bundle.is_dir():
                    bundles.append(bundle)
        except OSError:
            continue
    return bundles


def _macos_bundle_display_name(bundle_path: Path, fallback: str) -> str:
    name = bundle_path.name
    if name.lower().endswith(".app"):
        name = name[:-4]
    return name.strip() or fallback


def _discover_windows_editors() -> list[ExternalEditor]:
    candidates = [
        ("photoshop", "Adobe Photoshop", ["Photoshop.exe"]),
        ("gimp", "GIMP", ["gimp-2.10.exe", "gimp.exe"]),
        ("paintdotnet", "paint.net", ["paintdotnet.exe"]),
        ("krita", "Krita", ["krita.exe"]),
        ("affinity_photo", "Affinity Photo", ["AffinityPhoto2.exe", "AffinityPhoto.exe"]),
        ("paint", "Microsoft Paint", ["mspaint.exe"]),
    ]

    results = _discover_from_which(candidates)
    seen_targets = {editor.launch_target.casefold() for editor in results}

    for editor in _discover_windows_from_registry(candidates, seen_targets):
        seen_key = editor.launch_target.casefold()
        if seen_key in seen_targets:
            continue
        seen_targets.add(seen_key)
        results.append(editor)

    for editor in _discover_windows_photoshop_from_adobe_dirs(seen_targets):
        seen_key = editor.launch_target.casefold()
        if seen_key in seen_targets:
            continue
        seen_targets.add(seen_key)
        results.append(editor)

    return sorted(results, key=lambda editor: (editor.display_name.casefold(), editor.launch_target.casefold()))


def _discover_windows_photoshop_from_adobe_dirs(seen_targets: set[str]) -> list[ExternalEditor]:
    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Adobe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Adobe",
    ]

    found: list[ExternalEditor] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue

        try:
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                folder_name = child.name.casefold()
                if "photoshop" not in folder_name:
                    continue
                executable = child / "Photoshop.exe"
                if not executable.exists() or not executable.is_file():
                    continue
                key = str(executable).casefold()
                if key in seen_targets:
                    continue
                found.append(
                    ExternalEditor(
                        id="photoshop",
                        display_name=_photoshop_display_name_from_executable(executable),
                        launch_target=str(executable),
                    )
                )
        except OSError:
            continue

    return found


def _discover_windows_from_registry(
    candidates: list[tuple[str, str, list[str]]],
    seen_targets: set[str],
) -> list[ExternalEditor]:
    try:
        import winreg
    except Exception:
        return []

    uninstall_roots = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
    ]
    hives = [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]

    normalized_candidates: list[tuple[str, str, list[str]]]
    normalized_candidates = []
    for app_id, display_name, executable_names in candidates:
        normalized_candidates.append((app_id, display_name, [name.casefold() for name in executable_names]))

    known_targets = set(seen_targets)
    results: list[ExternalEditor] = []
    for hive in hives:
        for root in uninstall_roots:
            try:
                with winreg.OpenKey(hive, root) as uninstall_key:
                    index = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(uninstall_key, index)
                        except OSError:
                            break
                        index += 1
                        try:
                            with winreg.OpenKey(uninstall_key, subkey_name) as app_key:
                                display_name_raw = _read_winreg_string(winreg, app_key, "DisplayName")
                                install_location = _read_winreg_string(winreg, app_key, "InstallLocation")
                                display_icon = _read_winreg_string(winreg, app_key, "DisplayIcon")
                        except OSError:
                            continue

                        if not display_name_raw:
                            continue

                        lowered_name = display_name_raw.casefold()
                        for app_id, default_name, executable_names in normalized_candidates:
                            if app_id == "photoshop":
                                name_match = "photoshop" in lowered_name
                            elif app_id == "paintdotnet":
                                name_match = "paint.net" in lowered_name or "paintdotnet" in lowered_name
                            else:
                                name_match = app_id.replace("_", " ") in lowered_name
                            if not name_match:
                                continue

                            match = _best_editor_path_from_registry(install_location, display_icon, executable_names)
                            if match is None:
                                continue

                            seen_key = str(match).casefold()
                            if seen_key in known_targets:
                                continue
                            known_targets.add(seen_key)
                            results.append(
                                ExternalEditor(
                                    id=app_id,
                                    display_name=_display_name_for_registry_match(
                                        app_id,
                                        default_name,
                                        display_name_raw,
                                        match,
                                    ),
                                    launch_target=str(match),
                                )
                            )
                            break
            except OSError:
                continue

    return results


def _best_editor_path_from_registry(
    install_location: str | None,
    display_icon: str | None,
    executable_names: list[str],
) -> Path | None:
    possible_paths: list[Path] = []

    if install_location:
        install_path = Path(install_location)
        for executable_name in executable_names:
            possible_paths.append(install_path / executable_name)

    if display_icon:
        icon_path = display_icon.split(",", 1)[0].strip().strip('"')
        if icon_path:
            possible_paths.append(Path(icon_path))

    for path in possible_paths:
        try:
            if path.exists() and path.is_file():
                return path
        except OSError:
            continue

    return None


def _read_winreg_string(winreg_module, key, value_name: str) -> str | None:
    try:
        value, _ = winreg_module.QueryValueEx(key, value_name)
    except OSError:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return None


def _discover_from_which(
    candidates: Iterable[tuple[str, str, list[str]]],
    seen_targets: set[str] | None = None,
) -> list[ExternalEditor]:
    seen = seen_targets if seen_targets is not None else set()
    found: list[ExternalEditor] = []

    for app_id, display_name, executable_names in candidates:
        for executable_name in executable_names:
            path = shutil.which(executable_name)
            if not path:
                continue
            key = path.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(
                ExternalEditor(
                    id=app_id,
                    display_name=display_name,
                    launch_target=path,
                )
            )
            break

    return found


def _display_name_for_registry_match(
    app_id: str,
    default_name: str,
    registry_display_name: str,
    executable_path: Path,
) -> str:
    if app_id != "photoshop":
        return default_name

    normalized = registry_display_name.strip()
    if normalized:
        return normalized
    return _photoshop_display_name_from_executable(executable_path)


def _photoshop_display_name_from_executable(executable_path: Path) -> str:
    parent_name = executable_path.parent.name.strip()
    lowered = parent_name.casefold()
    if "photoshop" in lowered:
        return parent_name
    return "Adobe Photoshop"

