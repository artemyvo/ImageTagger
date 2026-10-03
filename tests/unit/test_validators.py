"""LLM setting inputs: Qt validator ranges and InputValidator parsing."""
from __future__ import annotations

import math
from typing import List

import pytest
from PyQt6.QtCore import QLocale
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QValidator
from PyQt6.QtWidgets import QWidget

from imagetagger.providers.llm_provider import LlmProviderError
from imagetagger.utils import validators
from imagetagger.utils.input_validators import InputValidator

State = QValidator.State


class Errors:
    """Recording on_error callback."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)


# ---------------------------------------------------------------------------
# Qt validator factories
# ---------------------------------------------------------------------------


@pytest.fixture
def parent(qtbot) -> QWidget:
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize(
    "factory, bottom, top",
    [
        (validators.create_timeout_validator, 1, 86400),
        (validators.create_retry_validator, 0, 10),
        (validators.create_threads_validator, 0, 128),
    ],
)
def test_int_validator_ranges(parent, factory, bottom, top):
    validator = factory(parent)
    assert isinstance(validator, QIntValidator)
    assert (validator.bottom(), validator.top()) == (bottom, top)
    assert validator.parent() is parent


@pytest.mark.parametrize(
    "factory, bottom, top",
    [
        (validators.create_max_resolution_validator, 0.01, 1000.0),
        (validators.create_temperature_validator, 0.0, 2.0),
    ],
)
def test_double_validator_ranges(parent, factory, bottom, top):
    validator = factory(parent)
    assert isinstance(validator, QDoubleValidator)
    assert (validator.bottom(), validator.top(), validator.decimals()) == (bottom, top, 3)
    assert validator.notation() == QDoubleValidator.Notation.StandardNotation
    assert validator.parent() is parent


@pytest.mark.parametrize(
    "factory, text, state",
    [
        (validators.create_timeout_validator, "1", State.Acceptable),
        (validators.create_timeout_validator, "86400", State.Acceptable),
        (validators.create_timeout_validator, "0", State.Intermediate),
        (validators.create_timeout_validator, "-1", State.Invalid),
        (validators.create_timeout_validator, "2.5", State.Invalid),
        (validators.create_retry_validator, "0", State.Acceptable),
        (validators.create_retry_validator, "10", State.Acceptable),
        (validators.create_retry_validator, "128", State.Invalid),
        (validators.create_threads_validator, "128", State.Acceptable),
        (validators.create_threads_validator, "abc", State.Invalid),
        (validators.create_max_resolution_validator, "0.01", State.Acceptable),
        (validators.create_max_resolution_validator, "1000", State.Acceptable),
        (validators.create_max_resolution_validator, "1.2345", State.Invalid),
        (validators.create_max_resolution_validator, "1e3", State.Invalid),
        (validators.create_max_resolution_validator, "nan", State.Invalid),
        (validators.create_max_resolution_validator, "-1", State.Invalid),
        (validators.create_temperature_validator, "0", State.Acceptable),
        (validators.create_temperature_validator, "2", State.Acceptable),
        (validators.create_temperature_validator, "0.001", State.Acceptable),
        (validators.create_temperature_validator, "10", State.Invalid),
        (validators.create_temperature_validator, "-1", State.Invalid),
    ],
)
def test_validator_states(parent, factory, text, state):
    validator = factory(parent)
    validator.setLocale(QLocale.c())  # decimal point must not depend on the machine's locale
    assert validator.validate(text, 0)[0] == state


# ---------------------------------------------------------------------------
# parse_timeout_seconds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [("30", 30.0), ("  1 ", 1.0), ("+5", 5.0), ("86400", 86400.0), ("\t600\n", 600.0)],
)
def test_parse_timeout_seconds(text, expected):
    errors = Errors()
    value = InputValidator.parse_timeout_seconds(text, errors)
    assert value == expected
    assert isinstance(value, float)
    assert errors.messages == []


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "Enter timeout in seconds."),
        ("   ", "Enter timeout in seconds."),
        ("1.5", "Timeout must be a whole number of seconds."),
        ("abc", "Timeout must be a whole number of seconds."),
        ("10s", "Timeout must be a whole number of seconds."),
        ("0", "Timeout must be at least 1 second."),
        ("-5", "Timeout must be at least 1 second."),
    ],
)
def test_parse_timeout_seconds_errors(text, message):
    errors = Errors()
    with pytest.raises(LlmProviderError) as info:
        InputValidator.parse_timeout_seconds(text, errors)
    assert str(info.value) == message
    assert errors.messages == [message]


def test_parse_timeout_seconds_without_callback_still_raises():
    with pytest.raises(LlmProviderError, match="at least 1 second"):
        InputValidator.parse_timeout_seconds("0")


# ---------------------------------------------------------------------------
# parse_retry_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3", 3),
        (" 10 ", 10),
        ("0", 0),
        ("-2", 0),
        ("", 0),
        ("abc", 0),
        ("1.5", 0),
    ],
)
def test_parse_retry_count(text, expected):
    assert InputValidator.parse_retry_count(text) == expected


# ---------------------------------------------------------------------------
# parse_max_resolution_mpx
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [("5", 5.0), (" 0.5 ", 0.5), ("1000", 1000.0), ("0.01", 0.01), ("2.25", 2.25)],
)
def test_parse_max_resolution_mpx(text, expected):
    errors = Errors()
    assert InputValidator.parse_max_resolution_mpx(text, errors) == expected
    assert errors.messages == []


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "Enter query downscale in megapixels."),
        ("  ", "Enter query downscale in megapixels."),
        ("abc", "Query downscale must be a number."),
        ("5 mpx", "Query downscale must be a number."),
        ("0", "Query downscale must be greater than 0."),
        ("-1.5", "Query downscale must be greater than 0."),
    ],
)
def test_parse_max_resolution_mpx_errors(text, message):
    errors = Errors()
    with pytest.raises(LlmProviderError) as info:
        InputValidator.parse_max_resolution_mpx(text, errors)
    assert str(info.value) == message
    assert errors.messages == [message]


def test_parse_max_resolution_mpx_rejects_nan():
    errors = Errors()
    with pytest.raises(LlmProviderError):
        InputValidator.parse_max_resolution_mpx("nan", errors)
    assert len(errors.messages) == 1


# ---------------------------------------------------------------------------
# parse_temperature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [("", None), ("   ", None), ("0", 0.0), ("2", 2.0), (" 0.7 ", 0.7), ("1.25", 1.25)],
)
def test_parse_temperature(text, expected):
    errors = Errors()
    assert InputValidator.parse_temperature(text, errors) == expected
    assert errors.messages == []


@pytest.mark.parametrize(
    "text, message",
    [
        ("warm", "Temperature must be a number between 0 and 2."),
        ("-0.1", "Temperature must be between 0 and 2."),
        ("2.001", "Temperature must be between 0 and 2."),
        ("inf", "Temperature must be a number between 0 and 2."),
    ],
)
def test_parse_temperature_errors(text, message):
    errors = Errors()
    with pytest.raises(LlmProviderError) as info:
        InputValidator.parse_temperature(text, errors)
    assert str(info.value) == message
    assert errors.messages == [message]


def test_parse_temperature_rejects_nan():
    with pytest.raises(LlmProviderError):
        InputValidator.parse_temperature("nan", Errors())


# ---------------------------------------------------------------------------
# format_megapixels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (5, "5.0"),
        (5.0, "5.0"),
        (0.5, "0.5"),
        (2.25, "2.25"),
        (1.2345, "1.234"),  # three decimals at most (1.2345 is stored just below .5)
        (0.01, "0.01"),
        (10, "10.0"),
        (1000, "1000.0"),
        (0.0001, "0.0"),
    ],
)
def test_format_megapixels(value, expected):
    assert InputValidator.format_megapixels(value) == expected


@pytest.mark.parametrize("value", [0.01, 0.5, 2.25, 5, 12.345, 1000])
def test_format_megapixels_round_trips_through_parse(value):
    text = InputValidator.format_megapixels(value)
    assert math.isclose(InputValidator.parse_max_resolution_mpx(text), value)
