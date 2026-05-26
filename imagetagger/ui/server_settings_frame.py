from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


def create_server_settings_frame(
    *,
    parent: QWidget,
    endpoint_input: QLineEdit,
    fetch_button: QPushButton,
    model_combo: QComboBox,
    use_button: QPushButton,
    include_tags_checkbox: QCheckBox | None = None,
    include_description_checkbox: QCheckBox | None = None,
    include_vision_checkbox: QCheckBox | None = None,
    include_refine_checkbox: QCheckBox | None = None,
    timeout_input: QLineEdit | None = None,
    retry_input: QLineEdit | None = None,
    max_resolution_input: QLineEdit | None = None,
    threads_input: QLineEdit | None = None,
) -> QFrame:
    frame = QFrame(parent)
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setFrameShadow(QFrame.Shadow.Sunken)

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(6)

    server_row = QHBoxLayout()
    server_row.setContentsMargins(0, 0, 0, 0)
    server_row.setSpacing(4)
    server_row.addWidget(endpoint_input, stretch=1)
    server_row.addWidget(fetch_button)

    model_row = QHBoxLayout()
    model_row.setContentsMargins(0, 0, 0, 0)
    model_row.setSpacing(4)
    model_row.addWidget(model_combo, stretch=1)
    model_row.addWidget(use_button)

    layout.addLayout(server_row)
    layout.addLayout(model_row)

    checkboxes = [cb for cb in (include_tags_checkbox, include_description_checkbox, include_vision_checkbox, include_refine_checkbox) if cb is not None]
    if checkboxes:
        checkboxes_row = QHBoxLayout()
        checkboxes_row.setContentsMargins(0, 0, 0, 0)
        for cb in checkboxes:
            checkboxes_row.addWidget(cb)
        checkboxes_row.addStretch(1)
        layout.addLayout(checkboxes_row)

    param_entries = [("Timeout", timeout_input), ("Retries", retry_input), ("Downscale", max_resolution_input), ("Threads", threads_input)]
    param_entries = [(label, w) for label, w in param_entries if w is not None]
    if param_entries:
        params_row = QHBoxLayout()
        params_row.setContentsMargins(0, 0, 0, 0)
        for i, (label_text, widget) in enumerate(param_entries):
            if i > 0:
                params_row.addSpacing(8)
            params_row.addWidget(QLabel(label_text, parent))
            params_row.addWidget(widget)
        params_row.addStretch(1)
        layout.addLayout(params_row)

    return frame