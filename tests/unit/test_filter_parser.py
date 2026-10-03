"""Image list filter expressions: tokenizing, parsing and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

import pytest

from imagetagger.utils.filter_parser import (
    FilterSyntaxError,
    _FilterRuntime,
    _parse_filter_expression,
    _tokenize_filter_expression,
)


@dataclass(frozen=True)
class Record:
    """Stand-in for ImageRecord: just what the fake runtime looks at."""

    flags: FrozenSet[str] = frozenset()
    tags: FrozenSet[str] = frozenset()
    text: str = ""
    mpx: Optional[float] = None


@dataclass
class Calls:
    tags: List[str] = field(default_factory=list)
    freetext: List[str] = field(default_factory=list)
    named: List[str] = field(default_factory=list)


def _runtime(calls: Optional[Calls] = None) -> _FilterRuntime:
    calls = calls if calls is not None else Calls()

    def _named(name: str):
        def _check(record: Record) -> bool:
            calls.named.append(name)
            return name in record.flags

        return _check

    def _tag(record: Record, tag: str) -> bool:
        calls.tags.append(tag)
        return tag in record.tags

    def _freetext(record: Record, text: str) -> bool:
        calls.freetext.append(text)
        return text.casefold() in record.text.casefold()

    return _FilterRuntime(
        named_filters={name: _named(name) for name in ("fixup", "untagged", "vision", "validated")},
        tag_filter=_tag,
        freetext_filter=_freetext,
        get_resolution_mpx=lambda record: record.mpx,
    )


def _matches(expression: str, record: Record) -> bool:
    parsed = _parse_filter_expression(expression)
    assert parsed is not None
    return parsed.evaluate(record, _runtime())


def _kinds(expression: str) -> List[Tuple[str, str]]:
    return [(token.kind, token.value) for token in _tokenize_filter_expression(expression)]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("fixup", [("NAME", "fixup")]),
        ("a&b|c", [("NAME", "a"), ("&", "&"), ("NAME", "b"), ("|", "|"), ("NAME", "c")]),
        ("!a ~b", [("NOT", "!"), ("NAME", "a"), ("NOT", "~"), ("NAME", "b")]),
        ("(a)", [("(", "("), ("NAME", "a"), (")", ")")]),
        ("resolution<=1.5", [("NAME", "resolution"), ("COMP", "<="), ("NUMBER", "1.5")]),
        ("resolution >= .5", [("NAME", "resolution"), ("COMP", ">="), ("NUMBER", ".5")]),
        ("< > <= >=", [("COMP", "<"), ("COMP", ">"), ("COMP", "<="), ("COMP", ">=")]),
        ('"red car"', [("STRING", "red car")]),
        ("'red car'", [("FREETEXT", "red car")]),
        ('"say \\"hi\\""', [("STRING", 'say "hi"')]),
        ("'it\\'s'", [("FREETEXT", "it's")]),
        ('"a\\\\b"', [("STRING", "a\\b")]),
        ('""', [("STRING", "")]),
        ("\"it's\"", [("STRING", "it's")]),
        ("'say \"hi\"'", [("FREETEXT", 'say "hi"')]),
        ('"a, b & (c)"', [("STRING", "a, b & (c)")]),
        ("résumé", [("NAME", "résumé")]),
        ("\t fixup \n", [("NAME", "fixup")]),
        ("", []),
        ("   ", []),
    ],
)
def test_tokenizer(expression, expected):
    assert _kinds(expression) == expected


def test_token_positions_are_zero_based_offsets():
    tokens = _tokenize_filter_expression('  !"x" & y')
    assert [(token.kind, token.position) for token in tokens] == [
        ("NOT", 2),
        ("STRING", 3),
        ("&", 7),
        ("NAME", 9),
    ]


@pytest.mark.parametrize(
    "expression, message",
    [
        ('"abc', "Missing closing quote for tag at position 1."),
        ("x 'abc", "Missing closing quote for freetext at position 3."),
        ('"abc\\', "Unfinished escape sequence at position 1."),
        ("'abc\\", "Unfinished escape sequence at position 1."),
        ("resolution > 1.2.3", "Invalid number '1.2.3' at position 14."),
    ],
)
def test_tokenizer_errors(expression, message):
    with pytest.raises(FilterSyntaxError) as info:
        _tokenize_filter_expression(expression)
    assert str(info.value) == message


def test_filter_syntax_error_is_a_value_error():
    assert issubclass(FilterSyntaxError, ValueError)


# ---------------------------------------------------------------------------
# Parser: structure and precedence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expression", ["", "   ", "\t\n"])
def test_blank_expression_parses_to_none(expression):
    assert _parse_filter_expression(expression) is None


@pytest.mark.parametrize(
    "expression, equivalent",
    [
        # Precedence, highest to lowest: NOT, AND, OR (docs/usage.md).
        ("a | b & c", "a | (b & c)"),
        ("a & b | c", "(a & b) | c"),
        ("!a & b", "(!a) & b"),
        ("!a | b", "(!a) | b"),
        ("~a & !b | c", "((~a) & (!b)) | c"),
        # Left associative.
        ("a & b & c", "(a & b) & c"),
        ("a | b | c", "(a | b) | c"),
        # Redundant grouping and whitespace do not change the tree.
        ("((a))", "a"),
        ("a&b", "a & b"),
        ("!!a", "!(!a)"),
        ("!~a", "!(!a)"),
    ],
)
def test_precedence_and_grouping(expression, equivalent):
    assert _parse_filter_expression(expression) == _parse_filter_expression(equivalent)


def test_grouping_overrides_precedence():
    assert _parse_filter_expression("(a | b) & c") != _parse_filter_expression("a | b & c")
    assert _parse_filter_expression("!(a & b)") != _parse_filter_expression("!a & b")


@pytest.mark.parametrize(
    "expression, message",
    [
        ("a b", "Unexpected token 'b' at position 3."),
        ("a)", "Unexpected token ')' at position 2."),
        (")", "Unexpected token ')' at position 1."),
        ("()", "Unexpected token ')' at position 2."),
        ("(a", "Missing ')' for group near position 1."),
        ("((a) & b", "Missing ')' for group near position 1."),
        ("(a b)", "Missing ')' for group near position 4."),
        ("a &", "Unexpected end of filter expression."),
        ("a |", "Unexpected end of filter expression."),
        ("!", "Unexpected end of filter expression."),
        ("& a", "Unexpected token '&' at position 1."),
        ("a || b", "Unexpected token '|' at position 4."),
        ("a & & b", "Unexpected token '&' at position 5."),
        ("5", "Unexpected token '5' at position 1."),
        ("< 5", "Unexpected token '<' at position 1."),
        ('fixup"x"', "Unexpected token 'x' at position 6."),
        ("resolution", "Expected comparison operator after 'resolution' at position 11."),
        ("resolution 5", "Expected comparison operator after 'resolution' at position 11."),
        ("resolution <", "Expected number after '<' at position 13."),
        ("resolution <= x", "Expected number after '<=' at position 14."),
        ("resolution < -1", "Expected number after '<' at position 13."),
    ],
)
def test_malformed_expressions(expression, message):
    with pytest.raises(FilterSyntaxError) as info:
        _parse_filter_expression(expression)
    assert str(info.value) == message


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

FIXUP = Record(flags=frozenset({"fixup"}))
VALIDATED = Record(flags=frozenset({"validated"}))
BOTH = Record(flags=frozenset({"fixup", "validated"}))
NEITHER = Record()


@pytest.mark.parametrize(
    "expression, record, expected",
    [
        ("fixup", FIXUP, True),
        ("fixup", NEITHER, False),
        ("FixUp", FIXUP, True),
        ("!fixup", FIXUP, False),
        ("~fixup", NEITHER, True),
        ("!!fixup", FIXUP, True),
        ("fixup & validated", BOTH, True),
        ("fixup & validated", FIXUP, False),
        ("fixup | validated", VALIDATED, True),
        ("fixup | validated", NEITHER, False),
        ("validated & !fixup", VALIDATED, True),
        ("validated & !fixup", BOTH, False),
        ("!(fixup | validated)", NEITHER, True),
        ("!(fixup | validated)", VALIDATED, False),
        # a | b & c  ==  a | (b & c): true through a alone.
        ("fixup | validated & vision", FIXUP, True),
        ("(fixup | validated) & vision", FIXUP, False),
        # Unknown names are not errors; they match nothing.
        ("landscape", BOTH, False),
        ("!landscape", NEITHER, True),
    ],
)
def test_named_filters_and_operators(expression, record, expected):
    assert _matches(expression, record) is expected


@pytest.mark.parametrize(
    "expression, mpx, expected",
    [
        ("resolution < 1", 0.5, True),
        ("resolution < 1", 1.0, False),
        ("resolution <= 1", 1.0, True),
        ("resolution > 5", 5.0, False),
        ("resolution > 5", 5.01, True),
        ("resolution >= 5", 5.0, True),
        ("resolution >= 5", 4.99, False),
        ("RESOLUTION>=.5", 0.5, True),
        ("resolution < 1.0", 0.999, True),
        # Unknown resolution never matches a comparison, whatever the operator.
        ("resolution < 1000", None, False),
        ("resolution >= 0", None, False),
        ("!(resolution >= 0)", None, True),
    ],
)
def test_resolution_comparisons(expression, mpx, expected):
    assert _matches(expression, Record(mpx=mpx)) is expected


def test_tag_and_freetext_values_reach_the_runtime_verbatim():
    """Normalizing (case, punctuation) is the runtime's job, not the parser's."""
    calls = Calls()
    parsed = _parse_filter_expression('"Red Car, Big" | \'Sunset (Late)\' | "Straße"')
    parsed.evaluate(NEITHER, _runtime(calls))
    assert calls.tags == ["Red Car, Big", "Straße"]
    assert calls.freetext == ["Sunset (Late)"]


@pytest.mark.parametrize(
    "expression, expected",
    [
        ('"portrait"', True),
        ('"landscape"', False),
        ('!"landscape" | \'sunset\'', True),
        ("'SUNSET'", True),
        ("'night'", False),
        ('"portrait" & \'sunset\'', True),
        ("(resolution > 5) & 'sunset'", True),
        ("(resolution > 50) & 'sunset'", False),
        ("~validated & (\"animal\" | 'sunset')", True),
    ],
)
def test_tags_freetext_and_resolution_combined(expression, expected):
    record = Record(tags=frozenset({"portrait"}), text="portrait, a sunset over the sea", mpx=12.0)
    assert _matches(expression, record) is expected


@pytest.mark.parametrize(
    "expression, record, evaluated",
    [
        ("fixup & validated", NEITHER, ["fixup"]),
        ("fixup & validated", FIXUP, ["fixup", "validated"]),
        ("fixup | validated", FIXUP, ["fixup"]),
        ("fixup | validated", NEITHER, ["fixup", "validated"]),
    ],
)
def test_and_or_short_circuit(expression, record, evaluated):
    calls = Calls()
    _parse_filter_expression(expression).evaluate(record, _runtime(calls))
    assert calls.named == evaluated
