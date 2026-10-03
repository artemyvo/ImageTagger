"""In-place cropping for Fix/Autofix ratio: pixels, metadata and the atomic replace."""
from __future__ import annotations

import os
import stat
import sys
import zlib
from pathlib import Path

import pytest
from PIL import Image, ImageCms, PngImagePlugin

from imagetagger.utils import image_crop
from imagetagger.utils.image_crop import ImageCropError, crop_image_file

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Quadrant image: 20x10, each quadrant 10x5.
WIDTH, HEIGHT = 20, 10


def _quadrants(mode: str = "RGB") -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT))
    half_w, half_h = WIDTH // 2, HEIGHT // 2
    image.paste(RED, (0, 0, half_w, half_h))
    image.paste(GREEN, (half_w, 0, WIDTH, half_h))
    image.paste(BLUE, (0, half_h, half_w, HEIGHT))
    image.paste(YELLOW, (half_w, half_h, WIDTH, HEIGHT))
    return image.convert(mode) if mode != "RGB" else image


def _write(tmp_path: Path, name: str, image: Image.Image, **save_kwargs) -> Path:
    path = tmp_path / name
    image.save(path, **save_kwargs)
    return path


def _close(actual, expected, tolerance: int) -> bool:
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


def _leftovers(folder: Path) -> list:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith(".imagetagger-crop."))


# ---------------------------------------------------------------------------
# Pixels and formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "box, corner_colors",
    [
        # (top-left, top-right, bottom-left, bottom-right) of the result
        ((0, 0, 10, 5), (RED, RED, RED, RED)),
        ((10, 0, 20, 5), (GREEN, GREEN, GREEN, GREEN)),
        ((0, 5, 10, 10), (BLUE, BLUE, BLUE, BLUE)),
        ((10, 5, 20, 10), (YELLOW, YELLOW, YELLOW, YELLOW)),
        ((5, 0, 15, 10), (RED, GREEN, BLUE, YELLOW)),
        ((0, 0, 20, 10), (RED, GREEN, BLUE, YELLOW)),
        ((9, 4, 11, 6), (RED, GREEN, BLUE, YELLOW)),
    ],
)
def test_png_crop_keeps_the_framed_region(tmp_path, box, corner_colors):
    path = _write(tmp_path, "a.png", _quadrants())
    new_size = crop_image_file(path, box)

    left, top, right, bottom = box
    assert new_size == (right - left, bottom - top)
    with Image.open(path) as result:
        assert result.format == "PNG"
        assert result.size == new_size
        w, h = result.size
        rgb = result.convert("RGB")
        corners = (rgb.getpixel((0, 0)), rgb.getpixel((w - 1, 0)), rgb.getpixel((0, h - 1)), rgb.getpixel((w - 1, h - 1)))
    assert corners == corner_colors


@pytest.mark.parametrize(
    "name, save_kwargs, tolerance",
    [
        ("a.png", {}, 0),
        ("a.webp", {"lossless": True}, 0),
        ("a.webp", {"quality": 90}, 40),
        ("a.jpg", {"quality": 95}, 40),
        ("a.bmp", {}, 0),
    ],
)
def test_formats_are_kept_and_pixels_come_from_the_box(tmp_path, name, save_kwargs, tolerance):
    # Upscale the quadrants so lossy codecs have room; sample well inside each quadrant.
    image = _quadrants().resize((80, 40), Image.Resampling.NEAREST)
    path = _write(tmp_path, name, image, **save_kwargs)
    with Image.open(path) as original:
        original_format = original.format

    assert crop_image_file(path, (40, 0, 80, 40), expected_size=(80, 40)) == (40, 40)

    with Image.open(path) as result:
        assert result.format == original_format
        assert result.size == (40, 40)
        rgb = result.convert("RGB")
        assert _close(rgb.getpixel((10, 5)), GREEN, tolerance)
        assert _close(rgb.getpixel((30, 30)), YELLOW, tolerance)
    assert _leftovers(tmp_path) == []


def test_lossless_webp_stays_lossless(tmp_path):
    path = _write(tmp_path, "a.webp", _quadrants(), lossless=True)
    assert image_crop._webp_is_lossless(path)
    crop_image_file(path, (0, 0, 10, 10))
    assert image_crop._webp_is_lossless(path)


def test_lossy_webp_is_detected_as_lossy(tmp_path):
    path = _write(tmp_path, "a.webp", _quadrants(), quality=80)
    assert not image_crop._webp_is_lossless(path)


def test_rgba_png_keeps_alpha(tmp_path):
    image = _quadrants("RGBA")
    image.putpixel((15, 7), (1, 2, 3, 0))
    path = _write(tmp_path, "a.png", image)
    crop_image_file(path, (10, 5, 20, 10))
    with Image.open(path) as result:
        assert result.mode == "RGBA"
        assert result.getpixel((5, 2)) == (1, 2, 3, 0)
        assert result.getpixel((0, 0)) == YELLOW + (255,)


def test_gif_single_frame_is_cropped(tmp_path):
    path = _write(tmp_path, "a.gif", _quadrants().convert("P"))
    assert crop_image_file(path, (10, 0, 20, 5)) == (10, 5)
    with Image.open(path) as result:
        assert result.format == "GIF"
        assert result.convert("RGB").getpixel((0, 0)) == GREEN


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_jpeg_exif_orientation_is_carried_over_and_box_uses_stored_pixels(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    image = _quadrants().resize((80, 40), Image.Resampling.NEAREST)
    path = _write(tmp_path, "a.jpg", image, exif=exif.tobytes(), quality=95, dpi=(300, 300))

    # The box is in the stored 80x40 grid, not the displayed 40x80 one.
    assert crop_image_file(path, (0, 0, 40, 40), expected_size=(80, 40)) == (40, 40)

    with Image.open(path) as result:
        assert result.size == (40, 40)
        assert result.getexif().get(0x0112) == 6
        assert tuple(round(v) for v in result.info["dpi"]) == (300, 300)
        rgb = result.convert("RGB")
        assert _close(rgb.getpixel((10, 5)), RED, 40)
        assert _close(rgb.getpixel((10, 30)), BLUE, 40)


def test_png_text_chunks_icc_and_dpi_survive(tmp_path):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a cat, masterpiece\nSteps: 20")
    info.add_itxt("Comment", "naïve", lang="en", tkey="Kommentar")
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = _write(tmp_path, "a.png", _quadrants(), pnginfo=info, icc_profile=icc, dpi=(72, 72))

    crop_image_file(path, (0, 0, 10, 10))

    with Image.open(path) as result:
        result.load()
        assert result.text["parameters"] == "a cat, masterpiece\nSteps: 20"
        assert result.text["Comment"] == "naïve"
        assert result.info["icc_profile"] == icc
        assert tuple(round(v) for v in result.info["dpi"]) == (72, 72)


def test_jpeg_reencode_reuses_source_quantization(tmp_path):
    image = _quadrants().resize((64, 64), Image.Resampling.NEAREST)
    path = _write(tmp_path, "a.jpg", image, quality=60)
    with Image.open(path) as original:
        tables = original.quantization
    crop_image_file(path, (0, 0, 32, 32))
    with Image.open(path) as result:
        assert result.quantization == tables


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 21, 10),
        (0, 0, 20, 11),
        (-1, 0, 10, 10),
        (0, -1, 10, 10),
        (5, 0, 5, 10),  # empty width
        (0, 5, 10, 5),  # empty height
        (10, 0, 5, 10),  # inverted
    ],
)
def test_box_outside_the_image_is_refused(tmp_path, box):
    path = _write(tmp_path, "a.png", _quadrants())
    before = path.read_bytes()
    with pytest.raises(ImageCropError, match="outside the image"):
        crop_image_file(path, box)
    assert path.read_bytes() == before


def test_size_change_on_disk_is_refused(tmp_path):
    path = _write(tmp_path, "a.png", _quadrants())
    before = path.read_bytes()
    with pytest.raises(ImageCropError, match="changed on disk"):
        crop_image_file(path, (0, 0, 10, 10), expected_size=(40, 20))
    assert path.read_bytes() == before


def test_animated_image_is_refused(tmp_path):
    frames = [Image.new("RGB", (8, 8), RED), Image.new("RGB", (8, 8), BLUE)]
    path = tmp_path / "a.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    before = path.read_bytes()
    with pytest.raises(ImageCropError, match="Animated"):
        crop_image_file(path, (0, 0, 4, 4))
    assert path.read_bytes() == before


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")


def test_16bit_rgb_png_is_refused(tmp_path):
    # Pillow cannot write 16-bit RGB PNGs, so build one by hand: 4x4, color type 2.
    width = height = 4
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([16, 2, 0, 0, 0])
    rows = b"".join(b"\x00" + b"\xff\xff\x00\x00\x00\x00" * width for _ in range(height))
    data = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(rows)) + _png_chunk(b"IEND", b"")
    path = tmp_path / "a.png"
    path.write_bytes(data)
    with Image.open(path) as image:
        image.load()  # a valid file Pillow can decode
    assert image_crop._png_is_16bit_truecolor(path)

    with pytest.raises(ImageCropError, match="16-bit"):
        crop_image_file(path, (0, 0, 2, 2))
    assert path.read_bytes() == data


def test_16bit_grayscale_png_is_not_treated_as_truecolor(tmp_path):
    path = _write(tmp_path, "a.png", Image.new("I;16", (8, 8), 1000))
    assert not image_crop._png_is_16bit_truecolor(path)


def test_unsupported_format_is_refused(tmp_path):
    path = _write(tmp_path, "a.tiff", _quadrants())
    before = path.read_bytes()
    with pytest.raises(ImageCropError, match="Unsupported image format: TIFF"):
        crop_image_file(path, (0, 0, 4, 4))
    assert path.read_bytes() == before


@pytest.mark.parametrize("content", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_corrupt_file_raises_crop_error(tmp_path, content):
    path = tmp_path / "a.png"
    path.write_bytes(content)
    with pytest.raises(ImageCropError):
        crop_image_file(path, (0, 0, 1, 1))
    assert path.read_bytes() == content
    assert _leftovers(tmp_path) == []


def test_missing_file_raises_crop_error(tmp_path):
    with pytest.raises(ImageCropError):
        crop_image_file(tmp_path / "missing.png", (0, 0, 1, 1))


# ---------------------------------------------------------------------------
# Atomic replace
# ---------------------------------------------------------------------------


def test_success_leaves_no_temp_file_and_keeps_permissions(tmp_path):
    path = _write(tmp_path, "a.png", _quadrants())
    os.chmod(path, 0o640)
    crop_image_file(path, (0, 0, 10, 10))
    assert _leftovers(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png"]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640


def test_failed_replace_leaves_original_untouched_and_cleans_up(tmp_path, monkeypatch):
    path = _write(tmp_path, "a.png", _quadrants())
    before = path.read_bytes()

    def _fail(src, dst):
        raise PermissionError("disk says no")

    monkeypatch.setattr(image_crop.os, "replace", _fail)
    with pytest.raises(ImageCropError, match="disk says no"):
        crop_image_file(path, (0, 0, 10, 10))
    assert path.read_bytes() == before
    assert _leftovers(tmp_path) == []


def test_failed_encode_leaves_original_untouched_and_cleans_up(tmp_path, monkeypatch):
    path = _write(tmp_path, "a.png", _quadrants())
    before = path.read_bytes()

    def _fail(self, *args, **kwargs):
        raise OSError("encoder exploded")

    monkeypatch.setattr(Image.Image, "save", _fail)
    with pytest.raises(ImageCropError, match="encoder exploded"):
        crop_image_file(path, (0, 0, 10, 10))
    assert path.read_bytes() == before
    assert _leftovers(tmp_path) == []


def test_jpeg_falls_back_to_the_next_encoder_variant(tmp_path, monkeypatch):
    path = _write(tmp_path, "a.jpg", _quadrants().resize((64, 64)), quality=80)
    real_save = Image.Image.save
    seen = []

    def _save(self, fp, format=None, **params):
        seen.append(sorted(params))
        if "qtables" in params:
            raise ValueError("bad tables")
        return real_save(self, fp, format=format, **params)

    monkeypatch.setattr(Image.Image, "save", _save)
    assert crop_image_file(path, (0, 0, 32, 32)) == (32, 32)
    assert len(seen) == 2 and "qtables" in seen[0] and "quality" in seen[1]
    monkeypatch.undo()
    with Image.open(path) as result:
        assert result.format == "JPEG" and result.size == (32, 32)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_symlink_is_kept_and_its_target_is_cropped(tmp_path):
    target = _write(tmp_path, "real.png", _quadrants())
    link = tmp_path / "link.png"
    link.symlink_to(target)

    assert crop_image_file(link, (0, 0, 10, 10)) == (10, 10)
    assert link.is_symlink()
    with Image.open(target) as result:
        assert result.size == (10, 10)
