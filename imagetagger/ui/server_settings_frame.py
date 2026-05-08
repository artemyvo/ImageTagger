from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


def create_server_settings_frame(
    *,
    parent: QWidget,
    endpoint_input: QLineEdit,
    fetch_button: QPushButton,
    model_combo: QComboBox,
    use_button: QPushButton,
    include_tags_checkbox: QCheckBox,
    include_description_checkbox: QCheckBox,
    include_vision_checkbox: QCheckBox | None = None,
    include_refine_checkbox: QCheckBox | None = None,
    timeout_input: QLineEdit,
    retry_input: QLineEdit,
    max_resolution_input: QLineEdit,
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

    options_row = QHBoxLayout()
    options_row.setContentsMargins(0, 0, 0, 0)
    options_row.addWidget(include_tags_checkbox)
    options_row.addWidget(include_description_checkbox)
    if include_vision_checkbox is not None:
        options_row.addWidget(include_vision_checkbox)
    if include_refine_checkbox is not None:
        options_row.addWidget(include_refine_checkbox)
    options_row.addSpacing(12)
    options_row.addWidget(QLabel("Timeout", parent))
    options_row.addWidget(timeout_input)
    options_row.addSpacing(8)
    options_row.addWidget(QLabel("Retries", parent))
    options_row.addWidget(retry_input)
    options_row.addSpacing(8)
    options_row.addWidget(QLabel("Downscale", parent))
    options_row.addWidget(max_resolution_input)
    if threads_input is not None:
        options_row.addSpacing(8)
        options_row.addWidget(QLabel("Threads", parent))
        options_row.addWidget(threads_input)
    options_row.addStretch(1)

    layout.addLayout(server_row)
    layout.addLayout(model_row)
    layout.addLayout(options_row)
    return frame