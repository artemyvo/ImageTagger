"""Settings dialog (File > Settings; on macOS also in the application menu).

Shows settings from the main window's config dict and writes them back on
OK.  For now that is the merge dialog's mouse and trackpad actions
(``merge_table_mouse_actions``), which the Fixup dialog reads each time it
opens an image.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLayout,
    QVBoxLayout,
    QWidget,
)

from imagetagger import config as _config


class SettingsDialog(QDialog):
    """Edits settings held in a config dict; ``apply_to`` writes them back."""

    # Horizontal-scroll row target modes, safest first.
    _ROW_TARGET_CHOICES = (
        (_config.MERGE_TABLE_HSCROLL_TARGET_POINTER_ON_SELECTED, "Selected row, while the pointer is on it"),
        (_config.MERGE_TABLE_HSCROLL_TARGET_POINTER_ROW, "Row under the pointer"),
        (_config.MERGE_TABLE_HSCROLL_TARGET_SELECTED_ROW, "Selected row, wherever the pointer is"),
    )

    def __init__(self, cfg: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")

        # ── Merge dialog: mouse and trackpad ──────────────────────────────
        mouse_group = QGroupBox("Merge dialog: mouse and trackpad", self)

        self.double_click_checkbox = QCheckBox("Double-click a row to edit it or run its action", mouse_group)
        self.double_click_checkbox.setToolTip(
            "Double-clicking an editable Current cell edits it; elsewhere in the row it runs "
            "the row's action, like Enter"
        )

        self.swipe_checkbox = QCheckBox("Swipe (drag) a row sideways to apply or remove it", mouse_group)
        self.swipe_checkbox.setToolTip(
            "Drag a row to the left to apply its proposed value, to the right to remove its current value"
        )

        self.hscroll_checkbox = QCheckBox("Scroll sideways over a row to apply or remove it", mouse_group)
        self.hscroll_checkbox.setToolTip(
            "Two-finger sideways scroll on a trackpad, or a mouse's tilt or thumb wheel. "
            "One direction applies the proposed value, the other removes the current value"
        )

        # Horizontal scroll options: indented under their checkbox and
        # disabled together with their labels while it is off.
        self._hscroll_options = QWidget(mouse_group)
        self.hscroll_reverse_checkbox = QCheckBox("Reverse direction", self._hscroll_options)
        self.hscroll_reverse_checkbox.setToolTip(
            "Swap the direction that applies the proposed value and the one that removes the current value"
        )
        self.hscroll_pause_spinbox = QDoubleSpinBox(self._hscroll_options)
        self.hscroll_pause_spinbox.setRange(0.0, 5.0)
        self.hscroll_pause_spinbox.setDecimals(2)
        self.hscroll_pause_spinbox.setSingleStep(0.05)
        self.hscroll_pause_spinbox.setSuffix(" s")
        self.hscroll_pause_spinbox.setToolTip(
            "After an action, scrolling must stop for this long before the next one; "
            "0 lets one long scroll run several actions"
        )
        self.hscroll_target_combo = QComboBox(self._hscroll_options)
        for mode, label in self._ROW_TARGET_CHOICES:
            self.hscroll_target_combo.addItem(label, mode)
        self.hscroll_target_combo.setToolTip("Which row a sideways scroll acts on")

        options_form = QFormLayout(self._hscroll_options)
        options_form.setContentsMargins(24, 0, 0, 0)
        options_form.addRow(self.hscroll_reverse_checkbox)
        options_form.addRow("Pause between actions:", self.hscroll_pause_spinbox)
        options_form.addRow("Acts on:", self.hscroll_target_combo)
        self.hscroll_checkbox.toggled.connect(self._hscroll_options.setEnabled)

        mouse_layout = QVBoxLayout(mouse_group)
        mouse_layout.addWidget(self.double_click_checkbox)
        mouse_layout.addWidget(self.swipe_checkbox)
        mouse_layout.addWidget(self.hscroll_checkbox)
        mouse_layout.addWidget(self._hscroll_options)

        # ── Buttons ───────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        restore_button = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        restore_button.setToolTip("Put back the default settings (saved only on OK)")
        restore_button.clicked.connect(self._restore_defaults)

        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        layout.addWidget(mouse_group)
        layout.addWidget(buttons)

        mouse_actions = cfg.get("merge_table_mouse_actions")
        self._show_mouse_actions({
            **_config.default_merge_table_mouse_actions(),
            **(mouse_actions if isinstance(mouse_actions, dict) else {}),
        })

    def _show_mouse_actions(self, values: dict) -> None:
        self.double_click_checkbox.setChecked(bool(values["double_click_action_enabled"]))
        self.swipe_checkbox.setChecked(bool(values["swipe_actions_enabled"]))
        self.hscroll_checkbox.setChecked(bool(values["horizontal_scroll_actions_enabled"]))
        self.hscroll_reverse_checkbox.setChecked(bool(values["horizontal_scroll_reverse_enabled"]))
        self.hscroll_pause_spinbox.setValue(float(values["horizontal_scroll_stop_idle_seconds"]))
        index = self.hscroll_target_combo.findData(values["horizontal_scroll_row_target_mode"])
        self.hscroll_target_combo.setCurrentIndex(max(0, index))
        self._hscroll_options.setEnabled(self.hscroll_checkbox.isChecked())

    def _restore_defaults(self) -> None:
        self._show_mouse_actions(_config.default_merge_table_mouse_actions())

    def apply_to(self, cfg: dict) -> None:
        """Write the settings shown into ``cfg``; the caller saves it."""
        cfg["merge_table_mouse_actions"] = {
            "double_click_action_enabled": self.double_click_checkbox.isChecked(),
            "swipe_actions_enabled": self.swipe_checkbox.isChecked(),
            "horizontal_scroll_actions_enabled": self.hscroll_checkbox.isChecked(),
            "horizontal_scroll_reverse_enabled": self.hscroll_reverse_checkbox.isChecked(),
            "horizontal_scroll_stop_idle_seconds": round(self.hscroll_pause_spinbox.value(), 2),
            "horizontal_scroll_row_target_mode": int(self.hscroll_target_combo.currentData()),
        }
