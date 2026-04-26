from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette


def blend_colors(base: QColor, overlay: QColor, alpha: float) -> QColor:
    """Return an opaque blend of two colors using the provided alpha factor."""
    clamped_alpha = max(0.0, min(1.0, alpha))
    inv = 1.0 - clamped_alpha
    return QColor(
        round(base.red() * inv + overlay.red() * clamped_alpha),
        round(base.green() * inv + overlay.green() * clamped_alpha),
        round(base.blue() * inv + overlay.blue() * clamped_alpha),
    )


def danger_accent_color(palette: QPalette) -> QColor:
    """Return a theme-aware danger accent used for warning/delete affordances."""
    base = palette.color(QPalette.ColorRole.Base)
    highlight = palette.color(QPalette.ColorRole.Highlight)
    text = palette.color(QPalette.ColorRole.Text)
    is_dark_theme = base.lightness() < 128

    # Build an accent from current theme colors so warning/delete visuals adapt naturally.
    accent = blend_colors(highlight, text, 0.12 if is_dark_theme else 0.06)
    return accent.lighter(125) if is_dark_theme else accent.darker(135)


def danger_text_on_accent_color(palette: QPalette) -> QColor:
    """Return text color intended to sit on top of danger accents."""
    return palette.color(QPalette.ColorRole.HighlightedText)