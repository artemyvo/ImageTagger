"""Preparing images for LLM upload: downscaling, formats, caching and the missing-Pillow warning."""
from __future__ import annotations

import sys
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from imagetagger.utils import image_prep
from imagetagger.utils.image_prep import (
    DEFAULT_MAX_IMAGE_PIXELS,
    configure_image_preparation,
    consume_image_preparation_warning,
    prepare_image_for_query,
)
from imagetagger.utils.llm_queries import LlmQueryError


@pytest.fixture(autouse=True)
def fresh_image_prep(monkeypatch):
    """Module-level settings, warning flag and cache are restored after each test."""
    monkeypatch.setattr(image_prep, "_max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS)
    monkeypatch.setattr(image_prep, "_resize_warning_pending", False)
    monkeypatch.setattr(image_prep, "_prepared_image_cache", OrderedDict())


@pytest.fixture
def no_pillow(monkeypatch):
    """Make ``import PIL`` / ``from PIL import ...`` fail like an install without Pillow."""
    monkeypatch.setitem(sys.modules, "PIL", None)


def _save(tmp_path: Path, name: str, size, mode: str = "RGB", color="red", **kwargs) -> Path:
    path = tmp_path / name
    Image.new(mode, size, color).save(path, **kwargs)
    return path


def _decode(prepared) -> Image.Image:
    image = Image.open(BytesIO(prepared.content))
    image.load()
    return image


# ---------------------------------------------------------------------------
# configure_image_preparation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (500, 500),
        (0, 1),
        (-10, 1),
        (1234.9, 1234),
        (None, DEFAULT_MAX_IMAGE_PIXELS),  # None leaves the setting alone
    ],
)
def test_configure_max_pixels(value, expected):
    configure_image_preparation(max_image_pixels=value)
    assert image_prep._max_image_pixels == expected


def test_configure_without_arguments_changes_nothing():
    configure_image_preparation(max_image_pixels=321)
    configure_image_preparation()
    assert image_prep._max_image_pixels == 321


# ---------------------------------------------------------------------------
# Below the limit: sent as is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, save_kwargs, media_type",
    [
        ("a.png", {}, "image/png"),
        ("a.jpg", {}, "image/jpeg"),
        ("a.JPEG", {"format": "JPEG"}, "image/jpeg"),
        ("a.webp", {}, "image/webp"),
        ("a.gif", {}, "image/gif"),
        ("a.bmp", {}, "image/bmp"),
    ],
)
def test_images_within_the_limit_are_sent_unchanged(tmp_path, name, save_kwargs, media_type):
    path = _save(tmp_path, name, (40, 20), **save_kwargs)
    prepared = prepare_image_for_query(path)
    assert prepared.content == path.read_bytes()
    assert prepared.media_type == media_type
    assert (prepared.width, prepared.height, prepared.was_resized) == (40, 20, False)


def test_exactly_at_the_limit_is_not_resized(tmp_path):
    configure_image_preparation(max_image_pixels=800)
    path = _save(tmp_path, "a.png", (40, 20))
    prepared = prepare_image_for_query(path)
    assert not prepared.was_resized and prepared.content == path.read_bytes()


def test_exif_orientation_is_reflected_in_the_reported_size(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, displayed portrait
    path = _save(tmp_path, "a.jpg", (40, 20), exif=exif.tobytes())
    prepared = prepare_image_for_query(path)
    assert prepared.content == path.read_bytes()
    assert (prepared.width, prepared.height) == (20, 40)


# ---------------------------------------------------------------------------
# Above the limit: downscaled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, max_pixels, expected",
    [
        ((200, 100), 5_000, (100, 50)),
        ((100, 200), 5_000, (50, 100)),
        ((300, 300), 900, (30, 30)),
        ((1000, 10), 100, (100, 1)),
        # Degenerate strips: the short side never drops below one pixel.
        ((1000, 1), 10, (100, 1)),
        ((1, 1000), 10, (1, 100)),
    ],
)
def test_large_images_are_downscaled_keeping_the_aspect(tmp_path, size, max_pixels, expected):
    configure_image_preparation(max_image_pixels=max_pixels)
    path = _save(tmp_path, "a.png", size)
    prepared = prepare_image_for_query(path)
    assert prepared.was_resized
    assert (prepared.width, prepared.height) == expected
    decoded = _decode(prepared)
    assert decoded.size == expected


@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.webp", "a.gif", "a.bmp"])
def test_downscaled_content_matches_its_media_type(tmp_path, name):
    configure_image_preparation(max_image_pixels=200)
    path = _save(tmp_path, name, (40, 20))
    prepared = prepare_image_for_query(path)
    assert prepared.was_resized
    decoded = _decode(prepared)
    assert Image.MIME[decoded.format] == prepared.media_type
    assert decoded.size == (prepared.width, prepared.height) == (20, 10)


def test_downscaled_png_keeps_alpha(tmp_path):
    configure_image_preparation(max_image_pixels=200)
    path = _save(tmp_path, "a.png", (40, 20), mode="RGBA", color=(0, 0, 255, 0))
    decoded = _decode(prepare_image_for_query(path))
    assert decoded.mode == "RGBA"
    assert decoded.getpixel((5, 5))[3] == 0


def test_downscale_applies_exif_orientation(tmp_path):
    configure_image_preparation(max_image_pixels=200)
    exif = Image.Exif()
    exif[0x0112] = 6
    path = _save(tmp_path, "a.jpg", (40, 20), exif=exif.tobytes())
    prepared = prepare_image_for_query(path)
    assert (prepared.width, prepared.height) == (10, 20)
    assert _decode(prepared).size == (10, 20)


def test_downscaled_jpeg_stays_jpeg(tmp_path):
    configure_image_preparation(max_image_pixels=200)
    path = _save(tmp_path, "a.jpg", (40, 20))
    prepared = prepare_image_for_query(path)
    assert prepared.media_type == "image/jpeg"
    assert _decode(prepared).format == "JPEG"


def test_cmyk_jpeg_above_the_limit_is_downscaled(tmp_path):
    configure_image_preparation(max_image_pixels=200)
    path = _save(tmp_path, "a.jpg", (40, 20), mode="CMYK", color=(0, 255, 255, 0))
    prepared = prepare_image_for_query(path)
    assert prepared.was_resized
    assert _decode(prepared).size == (20, 10)


def test_media_type_follows_the_actual_format_not_the_suffix(tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (4, 4)).save(path, format="PNG")
    assert prepare_image_for_query(path).media_type == "image/png"


# ---------------------------------------------------------------------------
# WEBP -> PNG for Ollama
# ---------------------------------------------------------------------------


def test_force_webp_to_png_transcodes_large_webp(tmp_path):
    configure_image_preparation(max_image_pixels=200)
    path = _save(tmp_path, "a.webp", (40, 20))
    prepared = prepare_image_for_query(path, force_webp_to_png=True)
    assert prepared.media_type == "image/png"
    assert _decode(prepared).format == "PNG"
    assert (prepared.width, prepared.height) == (20, 10)


def test_force_webp_to_png_keeps_small_webp_at_its_size(tmp_path):
    path = _save(tmp_path, "a.webp", (10, 10))
    prepared = prepare_image_for_query(path, force_webp_to_png=True)
    assert prepared.media_type == "image/png"
    assert _decode(prepared).size == (10, 10)
    assert (prepared.width, prepared.height) == (10, 10)


def test_force_webp_to_png_leaves_other_formats_alone(tmp_path):
    path = _save(tmp_path, "a.png", (10, 10))
    prepared = prepare_image_for_query(path, force_webp_to_png=True)
    assert prepared.content == path.read_bytes() and not prepared.was_resized


# ---------------------------------------------------------------------------
# Unreadable input
# ---------------------------------------------------------------------------


def test_missing_file_raises_llm_query_error(tmp_path):
    with pytest.raises(LlmQueryError, match="Could not read image file"):
        prepare_image_for_query(tmp_path / "missing.png")


@pytest.mark.parametrize(
    "name, media_type",
    [
        ("a.png", "image/png"),
        ("a.jpg", "image/jpeg"),
        ("a.tiff", "application/octet-stream"),
        ("a", "application/octet-stream"),
    ],
)
def test_corrupt_file_is_sent_as_is_with_a_suffix_media_type(tmp_path, name, media_type):
    path = tmp_path / name
    path.write_bytes(b"definitely not an image")
    prepared = prepare_image_for_query(path)
    assert prepared.content == b"definitely not an image"
    assert prepared.media_type == media_type
    assert (prepared.width, prepared.height, prepared.was_resized) == (None, None, False)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_repeated_requests_hit_the_cache(tmp_path, monkeypatch):
    configure_image_preparation(max_image_pixels=200)
    path = _save(tmp_path, "a.png", (40, 20))
    first = prepare_image_for_query(path)

    def _boom(*args, **kwargs):
        raise AssertionError("re-encoded instead of using the cache")

    monkeypatch.setattr(image_prep, "_prepare_image_bytes", _boom)
    assert prepare_image_for_query(path) is first


def test_changing_the_limit_or_the_file_bypasses_the_cache(tmp_path):
    configure_image_preparation(max_image_pixels=200)
    path = _save(tmp_path, "a.png", (40, 20))
    assert prepare_image_for_query(path).width == 20

    configure_image_preparation(max_image_pixels=50)
    assert prepare_image_for_query(path).width == 10

    # A different file size changes the key even within the same mtime tick.
    Image.new("RGB", (80, 20), "blue").save(path)
    configure_image_preparation(max_image_pixels=DEFAULT_MAX_IMAGE_PIXELS)
    assert prepare_image_for_query(path).width == 80


def test_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(image_prep, "_PREPARED_IMAGE_CACHE_MAX", 3)
    paths = [_save(tmp_path, f"{i}.png", (4, 4)) for i in range(5)]
    for path in paths:
        prepare_image_for_query(path)
    assert [key[0] for key in image_prep._prepared_image_cache] == paths[2:]


# ---------------------------------------------------------------------------
# Missing Pillow
# ---------------------------------------------------------------------------


def test_no_warning_when_pillow_is_installed(tmp_path):
    configure_image_preparation(max_image_pixels=10)
    prepare_image_for_query(_save(tmp_path, "a.png", (40, 20)))
    assert consume_image_preparation_warning() is None


def test_without_pillow_images_are_sent_as_is_and_a_warning_is_raised(tmp_path, no_pillow):
    path = tmp_path / "a.jpg"
    path.write_bytes(b"jpeg bytes")
    configure_image_preparation(max_image_pixels=1)

    prepared = prepare_image_for_query(path)
    assert prepared.content == b"jpeg bytes"
    assert prepared.media_type == "image/jpeg"
    assert not prepared.was_resized
    assert image_prep._resize_warning_pending

    warning = consume_image_preparation_warning()
    assert warning is not None and "Pillow is not installed" in warning
    assert not image_prep._resize_warning_pending


def test_warning_is_consumed_once_after_pillow_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(image_prep, "_resize_warning_pending", True)
    assert consume_image_preparation_warning() is not None
    assert consume_image_preparation_warning() is None


def test_consume_detects_missing_pillow_on_its_own(no_pillow):
    # Each consume while Pillow is missing reports it again.
    assert consume_image_preparation_warning() is not None
    assert consume_image_preparation_warning() is not None


def test_webp_to_png_without_pillow_is_an_error(tmp_path, no_pillow):
    path = tmp_path / "a.webp"
    path.write_bytes(b"RIFF....WEBP")
    with pytest.raises(LlmQueryError, match="Pillow is required"):
        prepare_image_for_query(path, force_webp_to_png=True)


def test_pillow_availability_check(no_pillow):
    assert image_prep.is_pillow_image_resize_available() is False


def test_pillow_availability_check_when_installed():
    assert image_prep.is_pillow_image_resize_available() is True
