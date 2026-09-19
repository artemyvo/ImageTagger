"""Aspect-ratio helpers for dataset ratio unification.

Training pipelines batch images by aspect ratio, so a dataset whose images
share a small set of ratios batches cleanly.  This module is pure Python (no
Qt, no Pillow): it parses the ``allowed_ratios`` config value, checks whether
an image size fits one of the allowed ratios, and computes crop candidates.

A configured ratio is orientation-agnostic: ``2:3`` stands for both ``2:3``
(portrait) and ``3:2`` (landscape).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

DEFAULT_ALLOWED_RATIOS = "1:1, 2:3, 3:4, 4:5, 16:9"

# Guards against absurd config values such as "1:100000".
_MAX_RATIO_TERM = 10_000


@dataclass(frozen=True)
class AspectRatio:
    """A reduced ``w:h`` ratio, kept in the orientation it was written in."""

    w: int
    h: int

    @property
    def label(self) -> str:
        return f"{self.w}:{self.h}"

    @property
    def is_square(self) -> bool:
        return self.w == self.h

    def flipped(self) -> "AspectRatio":
        return AspectRatio(self.h, self.w)

    def key(self) -> tuple[int, int]:
        """Orientation-agnostic identity (``2:3`` and ``3:2`` share a key)."""
        return (min(self.w, self.h), max(self.w, self.h))


@dataclass(frozen=True)
class CropCandidate:
    """The largest crop of an image that has exactly ``ratio``."""

    ratio: AspectRatio  # oriented the same way as the crop (w:h)
    width: int
    height: int
    kept_fraction: float  # kept pixels / original pixels, 0..1

    @property
    def pixels(self) -> int:
        return self.width * self.height


def _parse_ratio_token(token: str) -> AspectRatio | None:
    text = token.strip().lower().replace("×", ":").replace("x", ":").replace("/", ":")
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        first = Fraction(parts[0].strip())
        second = Fraction(parts[1].strip())
    except (ValueError, ZeroDivisionError):
        return None
    if first <= 0 or second <= 0:
        return None
    value = first / second
    w, h = value.numerator, value.denominator
    if w > _MAX_RATIO_TERM or h > _MAX_RATIO_TERM:
        return None
    return AspectRatio(w, h)


def parse_allowed_ratios(value: Any) -> list[AspectRatio]:
    """Parse ``"1:1, 2:3, 16:9"`` (or a list of such tokens) into ratios.

    Invalid tokens are skipped.  ``2:3`` and ``3:2`` are the same allowed
    ratio, so only the first spelling is kept.
    """
    if isinstance(value, str):
        tokens: Iterable[Any] = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        tokens = value
    else:
        return []

    ratios: list[AspectRatio] = []
    seen: set[tuple[int, int]] = set()
    for token in tokens:
        if not isinstance(token, str):
            continue
        ratio = _parse_ratio_token(token)
        if ratio is None or ratio.key() in seen:
            continue
        seen.add(ratio.key())
        ratios.append(ratio)
    return ratios


def format_allowed_ratios(ratios: Iterable[AspectRatio]) -> str:
    return ", ".join(ratio.label for ratio in ratios)


def normalize_allowed_ratios_setting(value: Any, default: str = DEFAULT_ALLOWED_RATIOS) -> str:
    """Normalize the raw ``allowed_ratios`` config value to a canonical string.

    * missing / wrong type / nothing parseable -> ``default``
    * empty string or empty list              -> ``""`` (ratio check disabled)
    """
    if isinstance(value, str):
        if not value.strip():
            return ""
    elif isinstance(value, (list, tuple)):
        if not value:
            return ""
    else:
        return default

    ratios = parse_allowed_ratios(value)
    return format_allowed_ratios(ratios) if ratios else default


def _oriented(ratios: Iterable[AspectRatio]) -> list[AspectRatio]:
    """Expand orientation-agnostic ratios into both orientations."""
    oriented: list[AspectRatio] = []
    for ratio in ratios:
        oriented.append(ratio)
        if not ratio.is_square:
            oriented.append(ratio.flipped())
    return oriented


def _fits(width: int, height: int, ratio: AspectRatio) -> bool:
    # |width - height * w/h| < 1  or  |height - width * h/w| < 1, in integers.
    # The one-pixel slack accepts sizes that were rounded to whole pixels
    # (for example 1000x1333 for 3:4).
    return abs(width * ratio.h - height * ratio.w) < max(ratio.w, ratio.h)


def matching_ratio(width: int, height: int, allowed: Iterable[AspectRatio]) -> AspectRatio | None:
    """Return the allowed ratio (oriented like the image) that the size fits."""
    if width <= 0 or height <= 0:
        return None
    for ratio in _oriented(allowed):
        if _fits(width, height, ratio):
            return ratio
    return None


def crop_candidates(width: int, height: int, allowed: Iterable[AspectRatio]) -> list[CropCandidate]:
    """All possible ratio fixes for an image, closest (most pixels kept) first.

    Every candidate is the largest crop whose size is an exact multiple of the
    ratio, so the result has *exactly* that ratio (``999x1332`` rather than
    ``1000x1333`` for 3:4).  Both orientations of every allowed ratio are
    offered; ties prefer the image's own orientation.
    """
    if width <= 0 or height <= 0:
        return []

    total = width * height
    image_is_landscape = width >= height
    candidates: list[CropCandidate] = []
    for ratio in _oriented(allowed):
        scale = min(width // ratio.w, height // ratio.h)
        if scale <= 0:
            continue
        crop_w, crop_h = ratio.w * scale, ratio.h * scale
        candidates.append(
            CropCandidate(
                ratio=ratio,
                width=crop_w,
                height=crop_h,
                kept_fraction=(crop_w * crop_h) / total,
            )
        )

    def _sort_key(candidate: CropCandidate) -> tuple[int, int]:
        same_orientation = (candidate.ratio.w >= candidate.ratio.h) == image_is_landscape
        return (-candidate.pixels, 0 if same_orientation else 1)

    candidates.sort(key=_sort_key)
    return candidates


def closest_crop(width: int, height: int, allowed: Iterable[AspectRatio]) -> CropCandidate | None:
    """The ratio fix that discards the fewest pixels, or ``None``."""
    candidates = crop_candidates(width, height, allowed)
    return candidates[0] if candidates else None
