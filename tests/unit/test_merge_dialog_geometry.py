"""The merge dialog is recreated per image; its saved geometry must not drift."""
from __future__ import annotations

from PyQt6.QtCore import QRect

from imagetagger.ui.merge_actions import _geometry_to_save


def _windows_platform(requested: QRect) -> QRect:
    """What Windows reported back: slightly taller, with the top higher."""
    return requested.adjusted(0, -8, 0, 0)


def test_first_open_saves_what_the_dialog_ended_with():
    final = QRect(10, 20, 800, 600)
    assert _geometry_to_save(None, [], final) == final


def test_untouched_dialog_keeps_the_saved_geometry():
    requested = QRect(100, 100, 800, 600)
    settled = _windows_platform(requested)
    assert _geometry_to_save(requested, [settled], QRect(settled)) is None


def test_dialog_closed_before_settling_keeps_the_saved_geometry():
    assert _geometry_to_save(QRect(100, 100, 800, 600), [], QRect(90, 90, 810, 610)) is None


def test_user_move_and_resize_apply_on_top_of_the_request():
    requested = QRect(100, 100, 800, 600)
    settled = _windows_platform(requested)
    final = QRect(settled.x() + 30, settled.y() + 10, settled.width() - 50, settled.height() + 20)
    assert _geometry_to_save(requested, [settled], final) == QRect(130, 110, 750, 620)


def test_reopening_many_times_does_not_creep_upward():
    saved = QRect(100, 100, 800, 600)
    for _ in range(25):
        settled = _windows_platform(saved)
        to_save = _geometry_to_save(saved, [settled], QRect(settled))
        if to_save is not None:
            saved = to_save
    assert saved == QRect(100, 100, 800, 600)
