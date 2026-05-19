from __future__ import annotations

from typing import Callable
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from imagetagger.providers.llm_provider import LlmProviderCancelled, LlmProviderError
import os
import threading
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageCms, UnidentifiedImageError
from PyQt6.QtCore import QSize

from imagetagger.ui.models import ImageRecord
from imagetagger.utils.sidecar import read_sidecar_data


class _SimpleRunnable(QObject, QRunnable):
    """Minimal QRunnable that can emit a *finished* signal.

    ``QRunnable`` alone cannot emit signals; the ``QObject`` mixin enables it.
    """

    finished = pyqtSignal()

    def __init__(self, fn: Callable[[], None]) -> None:
        QObject.__init__(self)
        QRunnable.__init__(self)
        self._fn = fn

    def run(self) -> None:
        self._fn()
        self.finished.emit()


class RegenerateWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, task: Callable[[Callable[[str], None]], object]) -> None:
        super().__init__()
        self.task = task

    def run(self) -> None:
        try:
            result = self.task(self.progress.emit)
        except LlmProviderCancelled as exc:
            self.cancelled.emit(str(exc))
            return
        except LlmProviderError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"Unexpected LLM error: {exc}")
            return
        self.finished.emit(result)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
THUMB_SIZE = QSize(96, 96)
MIN_FONT_POINT_SIZE = 8
MAX_FONT_POINT_SIZE = 40


class TagPurgeWorker(QObject):
    """Write updated text files for a bulk tag removal in a background thread."""

    progress = pyqtSignal(int, int)   # (done, total)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, jobs: list[tuple[Path, str]]) -> None:
        """
        jobs: list of (text_path, new_text) pairs to write atomically.
        """
        super().__init__()
        self._jobs = jobs

    def run(self) -> None:
        total = len(self._jobs)
        try:
            for done, (path, text) in enumerate(self._jobs, 1):
                # Use a temp-file rename for atomicity but skip fsync —
                # full fsync durability is not needed for bulk tag edits
                # and makes bulk writes orders of magnitude slower.
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(path)
                self.progress.emit(done, total)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit()


class FolderLoadWorker(QObject):
    scan_progress = pyqtSignal(int, int, int)
    progress = pyqtSignal(int, int, int)
    item_loaded = pyqtSignal(object)  # emits a list[dict] batch per chunk
    finished = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    icc_warning = pyqtSignal(str)
    scan_ready = pyqtSignal(int)
    collision_detected = pyqtSignal(str, str)

    def __init__(self, folder: Path, max_thread_cap: int = 8) -> None:
        super().__init__()
        self.folder = folder
        self._max_thread_cap = max(1, int(max_thread_cap))
        self._cancelled = False
        self._allow_processing = threading.Event()

    def cancel(self) -> None:
        self._cancelled = True
        self._allow_processing.set()

    def allow_processing(self) -> None:
        self._allow_processing.set()

    @staticmethod
    def _has_invalid_icc_profile(image_path: Path) -> bool:
        try:
            with Image.open(image_path) as image:
                raw_profile = image.info.get("icc_profile")
                if not raw_profile:
                    return False
                if isinstance(raw_profile, str):
                    raw_profile = raw_profile.encode("utf-8", errors="ignore")
                if not isinstance(raw_profile, (bytes, bytearray)):
                    return False

                ImageCms.ImageCmsProfile(BytesIO(bytes(raw_profile)))
                return False
        except (OSError, ValueError, UnidentifiedImageError):
            return False
        except ImageCms.PyCMSError:
            return True

    def _scan_folder(self) -> tuple[list[Path], tuple[Path, Path] | None]:
        """
        Scan the folder once, returning sorted image paths and the first collision (if any).

        Collision definition: two image files that would map to the same `.txt` file
        (same stem after suffix replacement), but with different image extensions.
        """
        image_paths: list[Path] = []
        discovered_files = 0
        discovered_dirs = 0
        emit_every = 200

        try:
            for root, dirnames, filenames in os.walk(self.folder):
                if self._cancelled:
                    return [], None

                discovered_dirs += len(dirnames)
                discovered_files += len(filenames)
                root_path = Path(root)
                for filename in filenames:
                    image_path = root_path / filename
                    if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        image_paths.append(image_path)

                if (discovered_files + discovered_dirs) % emit_every == 0:
                    self.scan_progress.emit(discovered_files, discovered_dirs, len(image_paths))

            self.scan_progress.emit(discovered_files, discovered_dirs, len(image_paths))
            image_paths.sort(key=lambda p: p.relative_to(self.folder).as_posix().lower())
        except OSError as exc:
            raise exc

        seen_by_txt_path: dict[str, Path] = {}
        for image_path in image_paths:
            txt_key = str(image_path.with_suffix(".txt")).casefold()
            existing = seen_by_txt_path.get(txt_key)
            if existing is None:
                seen_by_txt_path[txt_key] = image_path
                continue

            if existing.suffix.lower() != image_path.suffix.lower():
                return image_paths, (existing, image_path)

        return image_paths, None

    @staticmethod
    def _thumbnail_rgba_bytes(image_path: Path) -> tuple[dict | None, bool]:
        """
        Build a small thumbnail using Pillow.

        Returns: (thumbnail_payload, icc_invalid)
        where thumbnail_payload is dict(width, height, bytes, bytes_per_line, has_alpha)
        suitable for reconstructing a QImage on the GUI thread.
        """
        try:
            img = Image.open(image_path)
        except (OSError, UnidentifiedImageError):
            return None, False

        with img:
            # Validate ICC profile using the raw ICC bytes (if present).
            raw_profile = img.info.get("icc_profile")
            icc_invalid = False
            if raw_profile:
                if isinstance(raw_profile, str):
                    raw_profile = raw_profile.encode("utf-8", errors="ignore")
                if isinstance(raw_profile, (bytes, bytearray)):
                    try:
                        ImageCms.ImageCmsProfile(BytesIO(bytes(raw_profile)))
                    except ImageCms.PyCMSError:
                        icc_invalid = True

            # Only transpose when EXIF orientation metadata is actually present,
            # skipping the overhead for images without it.
            if img.info.get("exif"):
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass

            # Determine whether we need an alpha channel before any pixel work.
            has_alpha = img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
                img.mode == "P" and "transparency" in img.info
            )

            # Downscale first in native color mode to minimise the number of pixels
            # converted: BOX is fast and accurate for large-to-small downscales.
            img.thumbnail((THUMB_SIZE.width(), THUMB_SIZE.height()), resample=Image.Resampling.BOX)

            # Convert only after downscaling — now just ~90 px² instead of full-res.
            target_mode = "RGBA" if has_alpha else "RGB"
            thumb = img.convert(target_mode)

            raw_bytes = thumb.tobytes()
            width, height = thumb.size
            channels = 4 if has_alpha else 3
            bytes_per_line = width * channels
            return (
                {
                    "width": width,
                    "height": height,
                    "bytes": raw_bytes,
                    "bytes_per_line": bytes_per_line,
                    "has_alpha": has_alpha,
                },
                icc_invalid,
            )

    def run(self) -> None:
        try:
            image_paths, collision = self._scan_folder()
        except OSError as exc:
            self.failed.emit(f"Failed to read folder: {exc}")
            return
        if self._cancelled:
            return

        if collision is not None:
            first, second = collision
            self.collision_detected.emit(str(first), str(second))
            return

        total = len(image_paths)
        self.scan_ready.emit(total)
        self.progress.emit(0, total, 0)

        # Wait until the GUI thread resets its state and is ready for results.
        while not self._allow_processing.is_set():
            if self._cancelled:
                return
            self._allow_processing.wait(timeout=0.05)

        if total == 0:
            self.finished.emit(0, str(self.folder))
            return

        processed = 0

        # Use a bounded worker pool to utilize multiple cores while avoiding huge memory spikes.
        max_workers = max(1, (os.cpu_count() or 1) - 1)
        # Cap to keep decoding and UI payloads bounded.
        max_workers = min(max_workers, self._max_thread_cap)
        chunk_size = 64

        def process_one(image_path: Path) -> dict:
            text_path = image_path.with_suffix(".txt")
            try:
                text = (
                    text_path.read_text(encoding="utf-8", errors="replace")
                    if text_path.exists()
                    else ""
                )
            except OSError:
                text = ""

            # Read sidecar data here in the thread pool so the main thread never
            # needs to call read_sidecar_data during the initial list build.
            try:
                sidecar = read_sidecar_data(image_path)
                active_badges: frozenset[str] = frozenset()
                _active: set[str] = set()
                if sidecar.fixup_issues or sidecar.fixup_tags or sidecar.fixup_description:
                    _active.add("⚖️")
                if sidecar.vision_tags or (sidecar.vision_caption or "").strip():
                    _active.add("✨")
                if sidecar.ai_find_matches:
                    _active.add("🔍")
                if sidecar.validated is not None:
                    _active.add("✅")
                active_badges = frozenset(_active)
                has_pending_fixup: bool = sidecar.has_pending_fixup
                validated: str | None = sidecar.validated
            except Exception:
                active_badges = frozenset()
                has_pending_fixup = False
                validated = None

            thumb_payload, icc_invalid = self._thumbnail_rgba_bytes(image_path)
            return {
                "image_path": str(image_path),
                "text_path": str(text_path),
                "text": text,
                "thumbnail": thumb_payload,
                "icc_invalid": icc_invalid,
                "active_badges": active_badges,
                "has_pending_fixup": has_pending_fixup,
                "validated": validated,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                for chunk_start in range(0, total, chunk_size):
                    if self._cancelled:
                        break

                    chunk = image_paths[chunk_start : chunk_start + chunk_size]
                    batch: list[dict] = []
                    for result in executor.map(process_one, chunk):
                        if self._cancelled:
                            break
                        processed += 1

                        if result.get("icc_invalid"):
                            self.icc_warning.emit(result["image_path"])

                        # Accumulate into batch; emit once per chunk to minimise
                        # cross-thread signal overhead (thumbnail bytes are ~96x96 RGBA).
                        batch.append(
                            {
                                "image_path": result["image_path"],
                                "text_path": result["text_path"],
                                "text": result["text"],
                                "thumbnail": result["thumbnail"],
                                "active_badges": result["active_badges"],
                                "has_pending_fixup": result["has_pending_fixup"],
                                "validated": result["validated"],
                            }
                        )

                    if not self._cancelled and batch:
                        self.item_loaded.emit(batch)
                        percent = int((processed / total) * 100) if total else 100
                        self.progress.emit(processed, total, percent)
            except Exception as exc:
                self.failed.emit(f"Failed while processing folder: {exc}")
                return

        self.finished.emit(processed, str(self.folder))


class LlmTaskWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    progress = pyqtSignal(str)
    item_ready = pyqtSignal(object)

    def __init__(self, task: Callable[[Callable[[str], None], Callable[[object], None]], object]) -> None:
        super().__init__()
        self.task = task

    def run(self) -> None:
        try:
            result = self.task(self.progress.emit, self.item_ready.emit)
        except LlmProviderCancelled as exc:
            self.cancelled.emit(str(exc))
            return
        except LlmProviderError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"Unexpected LLM error: {exc}")
            return
        self.finished.emit(result)


