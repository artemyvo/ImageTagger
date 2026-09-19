"""In-place image cropping used by the merge dialog's "Fix ratio" action.

The crop box is expressed in the file's *stored* pixel grid — the same grid
Qt displays in the merge dialog (Qt does not apply EXIF orientation when
loading via ``QImage(path)``), so what the user frames is what gets kept.
Metadata (EXIF, ICC profile, DPI, PNG text chunks such as Stable Diffusion
generation parameters) is carried over untouched.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


class ImageCropError(Exception):
    """Raised when an image cannot be cropped and saved."""


def _webp_is_lossless(image_path: Path) -> bool:
    """Best-effort check of the WebP bitstream type (``VP8L`` = lossless)."""
    try:
        with image_path.open("rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WEBP":
                return False
            while True:
                chunk = handle.read(8)
                if len(chunk) < 8:
                    return False
                fourcc = chunk[:4]
                size = int.from_bytes(chunk[4:8], "little")
                if fourcc == b"VP8L":
                    return True
                if fourcc == b"VP8 ":
                    return False
                handle.seek(size + (size & 1), os.SEEK_CUR)
    except OSError:
        return False


def _png_is_16bit_truecolor(image_path: Path) -> bool:
    """True for 16-bit-per-channel RGB(A) PNGs, which Pillow decodes to 8 bits."""
    try:
        with image_path.open("rb") as handle:
            header = handle.read(26)
    except OSError:
        return False
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return False
    bit_depth, color_type = header[24], header[25]
    return bit_depth == 16 and color_type in (2, 6)


def _png_text_chunks(image):
    """Re-create the source PNG's text chunks (tEXt / zTXt / iTXt)."""
    from PIL import PngImagePlugin

    text = getattr(image, "text", None)
    if not text:
        return None
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in text.items():
        if isinstance(value, PngImagePlugin.iTXt):
            pnginfo.add_itxt(key, value, value.lang or "", value.tkey or "")
        else:
            pnginfo.add_text(key, value)
    return pnginfo


def _jpeg_save_variants(image) -> list[dict]:
    """JPEG encoder settings, best first.

    Re-using the source quantization tables and chroma subsampling keeps the
    re-encode at the original quality level instead of guessing a number.
    """
    base: dict = {"optimize": True}
    if image.info.get("progressive") or image.info.get("progression"):
        base["progressive"] = True

    variants: list[dict] = []
    qtables = getattr(image, "quantization", None)
    if qtables:
        keep = dict(base, qtables=qtables)
        try:
            from PIL import JpegImagePlugin

            sampling = JpegImagePlugin.get_sampling(image)
            if sampling in (0, 1, 2):
                keep["subsampling"] = sampling
        except Exception:
            pass
        variants.append(keep)
    variants.append(dict(base, quality=95, subsampling=0))
    return variants


def crop_image_file(
    image_path: Path,
    box: tuple[int, int, int, int],
    expected_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Crop ``image_path`` in place to ``box`` = (left, top, right, bottom).

    ``expected_size`` is the size the caller framed the crop on; a mismatch
    means the file changed on disk in the meantime and the crop is refused.
    The file is replaced atomically, so a failure never leaves a partial
    image behind.  Returns the new ``(width, height)``.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a hard requirement
        raise ImageCropError("Pillow is required to crop images.") from exc

    left, top, right, bottom = (int(value) for value in box)

    # Crop the real file, so a symlinked dataset entry keeps pointing at it
    # instead of being replaced by a detached copy.
    image_path = Path(os.path.realpath(image_path))

    try:
        # Decode, crop and collect encoder settings first, then close the source
        # before replacing it (Windows cannot replace a file that is still open).
        with Image.open(image_path) as image:
            image_format = (image.format or "").upper()
            if image_format == "MPO":
                # Multi-picture JPEG (phone/camera): keep the primary frame.
                image_format = "JPEG"
            elif getattr(image, "is_animated", False):
                raise ImageCropError("Animated images cannot be cropped.")

            if expected_size is not None and tuple(image.size) != tuple(expected_size):
                raise ImageCropError(
                    "The image changed on disk "
                    f"({image.size[0]}x{image.size[1]} now, "
                    f"{expected_size[0]}x{expected_size[1]} expected). "
                    "Open Fix ratio again."
                )

            width, height = image.size
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ImageCropError("Crop frame is outside the image.")

            cropped = image.crop((left, top, right, bottom))
            cropped.load()

            common: dict = {}
            icc_profile = image.info.get("icc_profile")
            if icc_profile:
                common["icc_profile"] = icc_profile
            exif = image.info.get("exif")
            if exif:
                common["exif"] = exif
            dpi = image.info.get("dpi")

            if image_format == "JPEG":
                if dpi:
                    common["dpi"] = dpi
                variants = [dict(common, **variant) for variant in _jpeg_save_variants(image)]
            elif image_format == "PNG":
                if _png_is_16bit_truecolor(image_path):
                    raise ImageCropError(
                        "16-bit RGB PNGs would be reduced to 8 bits per channel. "
                        "Crop this file in an external editor instead."
                    )
                if dpi:
                    common["dpi"] = dpi
                # Text chunks placed after the pixel data are only known once the
                # image is fully loaded, which crop() above has done.
                pnginfo = _png_text_chunks(image)
                if pnginfo is not None:
                    common["pnginfo"] = pnginfo
                variants = [dict(common, optimize=True)]
            elif image_format == "WEBP":
                if _webp_is_lossless(image_path):
                    variants = [dict(common, lossless=True, quality=90, method=4)]
                else:
                    variants = [dict(common, quality=95, method=4)]
            elif image_format in {"BMP", "GIF"}:
                variants = [{}]
            else:
                raise ImageCropError(f"Unsupported image format: {image_format or 'unknown'}")

        _save_atomically(cropped, image_path, image_format, variants)
        return cropped.size
    except ImageCropError:
        raise
    except Exception as exc:  # Pillow raises a wide range of decode/encode errors
        raise ImageCropError(str(exc) or exc.__class__.__name__) from exc


def _save_atomically(cropped, image_path: Path, image_format: str, variants: list[dict]) -> None:
    # Same-directory temp file + os.replace, mirroring io_utils.atomic_write_text.
    # The ".tmp" suffix keeps the folder loader from picking it up as an image;
    # the short fixed prefix stays clear of file-name length limits.
    try:
        original_mode: int | None = stat.S_IMODE(os.stat(image_path).st_mode)
    except OSError:
        original_mode = None

    fd, tmp_name = tempfile.mkstemp(
        prefix=".imagetagger-crop.",
        suffix=".tmp",
        dir=str(image_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            last_error: Exception | None = None
            for variant in variants:
                try:
                    handle.seek(0)
                    handle.truncate()
                    cropped.save(handle, format=image_format, **variant)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, image_path)

        # mkstemp creates the file as 0600; restore the image's own permissions.
        # Done after the replace so a read-only original never yields a read-only
        # temp file that cannot be cleaned up (Windows).
        if original_mode is not None:
            try:
                os.chmod(image_path, original_mode)
            except OSError:
                pass
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
