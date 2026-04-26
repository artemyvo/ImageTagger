from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import threading

from imagetagger.utils.llm_queries import LlmQueryError


DEFAULT_MAX_IMAGE_PIXELS = 1_000_000

_max_image_pixels = DEFAULT_MAX_IMAGE_PIXELS
_resize_warning_lock = threading.Lock()
_resize_warning_pending = False

_MEDIA_TYPES_BY_FORMAT = {
    "BMP": "image/bmp",
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

_MEDIA_TYPES_BY_SUFFIX = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class PreparedImage:
    content: bytes
    media_type: str
    width: int | None = None
    height: int | None = None
    was_resized: bool = False


def configure_image_preparation(*, max_image_pixels: int | None = None) -> None:
    global _max_image_pixels

    if max_image_pixels is not None:
        _max_image_pixels = max(1, int(max_image_pixels))


def consume_image_preparation_warning() -> str | None:
    global _resize_warning_pending
    with _resize_warning_lock:
        if not _resize_warning_pending:
            try:
                import PIL  # noqa: F401
            except ImportError:
                _resize_warning_pending = True

        if not _resize_warning_pending:
            return None
        _resize_warning_pending = False
    return (
        "Pillow is not installed, so images are sent at original size.\n\n"
        "Install dependencies to enable resizing before upload."
    )


def _media_type_from_format(image_format: str | None, suffix: str) -> str:
    normalized_format = (image_format or "").upper()
    if normalized_format in _MEDIA_TYPES_BY_FORMAT:
        return _MEDIA_TYPES_BY_FORMAT[normalized_format]
    return _MEDIA_TYPES_BY_SUFFIX.get(suffix.casefold(), "application/octet-stream")


def prepare_image_for_query(
    image_path: Path,
    *,
    force_webp_to_png: bool = False,
) -> PreparedImage:
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise LlmQueryError(f"Could not read image file: {exc}") from exc

    return _prepare_image_bytes(
        image_bytes,
        suffix=image_path.suffix,
        force_webp_to_png=force_webp_to_png,
    )


def _prepare_image_bytes(
    image_bytes: bytes,
    *,
    suffix: str,
    force_webp_to_png: bool,
) -> PreparedImage:
    global _resize_warning_pending
    suffix_is_webp = suffix.casefold() == ".webp"
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError:
        if force_webp_to_png and suffix_is_webp:
            raise LlmQueryError(
                "Pillow is required to convert WEBP images to PNG for Ollama requests."
            )
        with _resize_warning_lock:
            _resize_warning_pending = True
        return PreparedImage(content=image_bytes, media_type=_media_type_from_format(None, suffix))

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            image_format = (image.format or "").upper() or None
            media_type = _media_type_from_format(image_format, suffix)
            is_webp = image_format == "WEBP" or suffix_is_webp
            should_transcode_webp = force_webp_to_png and is_webp

            if width <= 0 or height <= 0:
                if should_transcode_webp:
                    output = BytesIO()
                    image.save(output, format="PNG", optimize=True)
                    return PreparedImage(content=output.getvalue(), media_type="image/png")
                return PreparedImage(content=image_bytes, media_type=media_type)

            pixels = width * height
            if pixels <= _max_image_pixels and not should_transcode_webp:
                return PreparedImage(
                    content=image_bytes,
                    media_type=media_type,
                    width=width,
                    height=height,
                )

            scale = math.sqrt(_max_image_pixels / float(pixels))
            target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            resized = image.resize(target_size, Image.Resampling.LANCZOS)

            output = BytesIO()
            output_format = image_format or "PNG"
            if output_format in {"JPEG", "JPG"}:
                if resized.mode not in {"RGB", "L"}:
                    resized = resized.convert("RGB")
                resized.save(output, format="JPEG", quality=90, optimize=True)
                output_format = "JPEG"
            elif output_format == "PNG" or should_transcode_webp:
                resized.save(output, format="PNG", optimize=True)
                output_format = "PNG"
            else:
                resized.save(output, format="PNG", optimize=True)
                output_format = "PNG"

            return PreparedImage(
                content=output.getvalue(),
                media_type=_media_type_from_format(output_format, suffix),
                width=target_size[0],
                height=target_size[1],
                was_resized=True,
            )
    except (OSError, ValueError, UnidentifiedImageError):
        return PreparedImage(content=image_bytes, media_type=_media_type_from_format(None, suffix))