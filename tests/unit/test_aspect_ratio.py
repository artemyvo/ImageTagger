"""Allowed-ratio parsing, the fits check and the crop candidates behind Fix/Autofix ratio."""
from __future__ import annotations

import pytest

from imagetagger.utils.aspect_ratio import (
    AUTOFIX_MIN_KEPT_FRACTION,
    DEFAULT_ALLOWED_RATIOS,
    AspectRatio,
    autofix_crop,
    centered_crop_box,
    closest_crop,
    crop_candidates,
    format_allowed_ratios,
    matching_ratio,
    normalize_allowed_ratios_setting,
    parse_allowed_ratios,
)


def _labels(ratios) -> list:
    return [ratio.label for ratio in ratios]


def _ratios(text: str) -> list:
    return parse_allowed_ratios(text)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1:1, 2:3, 16:9", ["1:1", "2:3", "16:9"]),
        ("16:9;4:3", ["16:9", "4:3"]),
        (" 2 : 3 ,, 3:4 ", ["2:3", "3:4"]),
        ("2x3, 3X4, 4×5, 9/16", ["2:3", "3:4", "4:5", "9:16"]),
        ("4:6", ["2:3"]),  # reduced
        ("1.5:1", ["3:2"]),  # decimal terms
        ("2:3, 3:2, 4:6, 6:4", ["2:3"]),  # orientation-agnostic duplicates keep the first spelling
        ("3:2, 2:3", ["3:2"]),
        (["1:1", "16:9", "9:16"], ["1:1", "16:9"]),
        (("2:3",), ["2:3"]),
        (["1:1", 5, None, "2:3"], ["1:1", "2:3"]),  # non-string list entries are skipped
    ],
)
def test_parse_allowed_ratios(value, expected):
    assert _labels(parse_allowed_ratios(value)) == expected


@pytest.mark.parametrize(
    "token",
    ["", "abc", "1", "1:2:3", "0:1", "1:0", "-1:2", "1:-2", "1:100000", "inf:1", "nan:1", "a:b", ":3"],
)
def test_invalid_tokens_are_skipped(token):
    assert _labels(parse_allowed_ratios(f"{token}, 5:4")) == ["5:4"]


@pytest.mark.parametrize("value", [None, 42, 1.5, {"a": "1:1"}])
def test_parse_non_string_non_list_gives_nothing(value):
    assert parse_allowed_ratios(value) == []


def test_largest_allowed_ratio_term_is_accepted():
    assert _labels(parse_allowed_ratios("1:10000, 1:10001")) == ["1:10000"]


def test_format_round_trips_through_parse():
    text = "1:1, 2:3, 3:4, 4:5, 16:9"
    assert format_allowed_ratios(parse_allowed_ratios(text)) == text


def test_ratio_key_and_flip():
    ratio = AspectRatio(2, 3)
    assert ratio.flipped() == AspectRatio(3, 2)
    assert ratio.key() == ratio.flipped().key() == (2, 3)
    assert not ratio.is_square and AspectRatio(1, 1).is_square


@pytest.mark.parametrize(
    "value, expected",
    [
        ("3:2, 16:9", "3:2, 16:9"),
        ("4:6 ; 2x3; 1:1", "2:3, 1:1"),
        (["9:16", "16:9"], "9:16"),
        # Empty means "ratio check off".
        ("", ""),
        ("   ", ""),
        ([], ""),
        ((), ""),
        # Missing, wrong type or nothing parseable falls back to the default.
        (None, DEFAULT_ALLOWED_RATIOS),
        (7, DEFAULT_ALLOWED_RATIOS),
        ("garbage", DEFAULT_ALLOWED_RATIOS),
        (["0:0", 3], DEFAULT_ALLOWED_RATIOS),
    ],
)
def test_normalize_allowed_ratios_setting(value, expected):
    assert normalize_allowed_ratios_setting(value) == expected


def test_normalize_uses_the_given_default():
    assert normalize_allowed_ratios_setting(None, default="1:1") == "1:1"


def test_default_setting_is_already_normalized():
    assert normalize_allowed_ratios_setting(DEFAULT_ALLOWED_RATIOS) == DEFAULT_ALLOWED_RATIOS


# ---------------------------------------------------------------------------
# Fits check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, allowed, expected",
    [
        ((512, 512), "1:1", "1:1"),
        ((1920, 1080), "16:9", "16:9"),
        ((1080, 1920), "16:9", "9:16"),  # oriented like the image
        ((1000, 1500), "3:2", "2:3"),
        ((1000, 1333), "3:4", "3:4"),  # rounded to whole pixels
        ((1000, 1334), "3:4", "3:4"),
        ((1000, 1335), "3:4", None),
        ((1000, 1001), "1:1", None),  # one full pixel off does not fit
        ((1920, 1081), "16:9", None),
        ((1921, 1080), "16:9", "16:9"),  # |1080 - 1921*9/16| < 1
        ((1922, 1080), "16:9", None),
        ((800, 600), "1:1, 2:3", None),
        ((800, 600), "1:1, 3:4", "4:3"),
        ((0, 100), "1:1", None),
        ((100, -1), "1:1", None),
        ((100, 100), "", None),
    ],
)
def test_matching_ratio(size, allowed, expected):
    ratio = matching_ratio(size[0], size[1], _ratios(allowed))
    assert (ratio.label if ratio else None) == expected


def test_matching_ratio_prefers_the_first_allowed_ratio():
    # 1000x1333 fits 3:4 (within rounding) and 1000:1333 exactly; the order decides.
    assert matching_ratio(1000, 1333, _ratios("3:4, 1000:1333")).label == "3:4"
    assert matching_ratio(1000, 1333, _ratios("1000:1333, 3:4")).label == "1000:1333"


# ---------------------------------------------------------------------------
# Crop candidates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, allowed, expected",
    [
        # (label, width, height) for the closest candidate
        ((530, 831), "9:16", ("9:16", 467, 831)),  # the docstring example: full height kept
        ((1000, 1100), "1:1", ("1:1", 1000, 1000)),  # full width kept
        ((1100, 1000), "1:1", ("1:1", 1000, 1000)),
        ((1000, 1400), "2:3, 3:4", ("3:4", 1000, 1333)),  # rounds, does not floor to a multiple
        ((2000, 1000), "16:9, 1:1", ("16:9", 1778, 1000)),
        ((1000, 2000), "16:9, 1:1", ("9:16", 1000, 1778)),
    ],
)
def test_closest_crop(size, allowed, expected):
    candidate = closest_crop(size[0], size[1], _ratios(allowed))
    assert (candidate.ratio.label, candidate.width, candidate.height) == expected
    assert candidate.kept_fraction == pytest.approx(candidate.pixels / (size[0] * size[1]))


@pytest.mark.parametrize("size", [(530, 831), (831, 530), (1000, 1001), (1234, 567), (7, 5), (3000, 2001)])
def test_every_candidate_fits_its_ratio_and_the_image(size):
    width, height = size
    candidates = crop_candidates(width, height, _ratios(DEFAULT_ALLOWED_RATIOS))
    assert candidates
    for candidate in candidates:
        assert 1 <= candidate.width <= width and 1 <= candidate.height <= height
        # A candidate spans one full side of the image.
        assert candidate.width == width or candidate.height == height
        assert matching_ratio(candidate.width, candidate.height, [candidate.ratio]) == candidate.ratio
        assert 0 < candidate.kept_fraction <= 1


def test_candidates_cover_both_orientations_sorted_by_kept_pixels():
    candidates = crop_candidates(1000, 800, _ratios("1:1, 2:3"))
    assert sorted(_labels(c.ratio for c in candidates)) == ["1:1", "2:3", "3:2"]
    pixels = [c.pixels for c in candidates]
    assert pixels == sorted(pixels, reverse=True)


def test_ties_prefer_the_images_own_orientation_and_square_counts_as_landscape():
    candidates = crop_candidates(100, 100, _ratios("2:3"))
    assert [(c.ratio.label, c.width, c.height) for c in candidates] == [("3:2", 100, 67), ("2:3", 67, 100)]


@pytest.mark.parametrize("size", [(0, 10), (10, 0), (-5, 10)])
def test_no_candidates_for_empty_sizes(size):
    assert crop_candidates(size[0], size[1], _ratios("1:1")) == []
    assert closest_crop(size[0], size[1], _ratios("1:1")) is None


def test_no_candidates_without_allowed_ratios():
    assert crop_candidates(100, 50, []) == []
    assert closest_crop(100, 50, []) is None


# ---------------------------------------------------------------------------
# Autofix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, allowed, expected",
    [
        ((1000, 1000), "1:1", None),  # already fits: nothing to fix
        ((1000, 1005), "1:1", (1000, 1000)),  # 0.5% strip goes
        ((1000, 1010), "1:1", (1000, 1000)),  # kept 1000/1010 = 0.990099 >= 0.99
        ((1000, 1011), "1:1", None),  # kept 0.98912 < 0.99: user frames it by hand
        ((1005, 1000), "1:1", (1000, 1000)),
        ((1925, 1080), "16:9", (1920, 1080)),
        ((1000, 1500), "1:1", None),
        ((1000, 1005), "", None),
    ],
)
def test_autofix_crop(size, allowed, expected):
    candidate = autofix_crop(size[0], size[1], _ratios(allowed))
    if expected is None:
        assert candidate is None
    else:
        assert (candidate.width, candidate.height) == expected
        assert candidate.kept_fraction >= AUTOFIX_MIN_KEPT_FRACTION


def test_autofix_takes_the_closest_candidate():
    width, height = 1010, 1000
    allowed = _ratios("1:1, 4:3")
    assert autofix_crop(width, height, allowed) == closest_crop(width, height, allowed)


# ---------------------------------------------------------------------------
# Centred crop box
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image, crop, expected",
    [
        ((100, 100), (100, 100), (0, 0, 100, 100)),
        ((100, 110), (100, 100), (0, 5, 100, 105)),
        ((100, 111), (100, 100), (0, 5, 100, 105)),  # odd strip: the extra pixel goes at the bottom
        ((111, 100), (100, 100), (5, 0, 105, 100)),
        ((530, 831), (467, 831), (31, 0, 498, 831)),
        # Out-of-range crop sizes are clamped into the image.
        ((50, 40), (80, 90), (0, 0, 50, 40)),
        ((50, 40), (0, -3), (24, 19, 25, 20)),
    ],
)
def test_centered_crop_box(image, crop, expected):
    assert centered_crop_box(image[0], image[1], crop[0], crop[1]) == expected


@pytest.mark.parametrize("size", [(530, 831), (1000, 1011), (1925, 1080), (3, 7)])
def test_centered_box_of_each_candidate_stays_inside_the_image(size):
    width, height = size
    for candidate in crop_candidates(width, height, _ratios(DEFAULT_ALLOWED_RATIOS)):
        left, top, right, bottom = centered_crop_box(width, height, candidate.width, candidate.height)
        assert all(isinstance(v, int) for v in (left, top, right, bottom))
        assert 0 <= left < right <= width and 0 <= top < bottom <= height
        assert (right - left, bottom - top) == (candidate.width, candidate.height)
        # Centred: the two discarded strips differ by at most one pixel.
        assert abs(left - (width - right)) <= 1 and abs(top - (height - bottom)) <= 1
