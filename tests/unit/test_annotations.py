"""Annotation text normalization: tags, descriptions and .txt tag parsing."""
from __future__ import annotations

import pytest

from imagetagger.utils.annotations import (
    normalize_description_text,
    parse_tags_text,
    remove_commas_from_description,
    sanitize_annotation_text,
    sanitize_description_text,
    sanitize_tag_text,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a photo", "a photo"),
        ("  a   photo  ", "a photo"),
        ("a\tphoto\nof\r\na cat", "a photo of a cat"),
        ("a photo", "a photo"),  # no-break space counts as whitespace
        ("", ""),
        ("   \n\t", ""),
        ("Keeps, Punctuation. And Case!", "Keeps, Punctuation. And Case!"),
    ],
)
def test_normalize_description_text(text, expected):
    assert normalize_description_text(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a, b", "a b"),
        ("a,b", "a b"),
        (",a,", "a"),
        ("  no commas  ", "no commas"),
        ("", ""),
        (",,,", ""),
    ],
)
def test_remove_commas_from_description(text, expected):
    assert remove_commas_from_description(text) == expected


@pytest.mark.parametrize("text", ["a , b", "a,,,b", "x, , ,y", ",,,"])
def test_remove_commas_leaves_no_comma(text):
    assert "," not in remove_commas_from_description(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A dog, running on the beach.", "A dog running on the beach."),
        ("A dog (brown) [blurry]", "A dog brown blurry"),
        ("  lots   of \n space ,, and , commas ", "lots of space and commas"),
        ("Keeps periods. And Case! And 'quotes'?", "Keeps periods. And Case! And 'quotes'?"),
        ("Café, naïve — résumé", "Café naïve — résumé"),
        ("()[],", ""),
        ("", ""),
    ],
)
def test_sanitize_description_text(text, expected):
    assert sanitize_description_text(text) == expected


@pytest.mark.parametrize(
    "text",
    ["a, b, c", "one,two", "(x, y)", "trailing,", ",leading", "a ,, b"],
)
def test_sanitized_description_is_safe_in_comma_separated_txt(text):
    """A description is stored as one field of the comma-separated .txt."""
    sanitized = sanitize_description_text(text)
    assert "," not in sanitized
    assert parse_tags_text(f"tag1, {sanitized}, tag2")[1] == sanitize_tag_text(sanitized)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Red Car", "red car"),
        ("  red   car  ", "red car"),
        ("red car.", "red car"),
        ("1.5 mpx", "1 5 mpx"),
        ("[red] (car)", "red car"),
        ("red, car", "red car"),
        ("ÉCOLE", "école"),
        ("Straße", "straße"),
        ("don't", "don't"),
        ("red-car", "red-car"),
        ("...", ""),
        ("", ""),
    ],
)
def test_sanitize_tag_text(text, expected):
    assert sanitize_tag_text(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Red Car.", "Red Car"),
        ("a, b. c", "a b c"),
        ("[x] (y)", "x y"),
        ("  keep   Case  ", "keep Case"),
        (".,[]()", ""),
    ],
)
def test_sanitize_annotation_text_strips_punctuation_but_keeps_case(text, expected):
    assert sanitize_annotation_text(text) == expected


@pytest.mark.parametrize(
    "text",
    ["Red Car", "red car.", "  [red]  car ", "a (b), c"],
)
def test_sanitizers_are_idempotent(text):
    for sanitize in (sanitize_tag_text, sanitize_annotation_text, sanitize_description_text):
        once = sanitize(text)
        assert sanitize(once) == once


@pytest.mark.parametrize(
    "text, expected",
    [
        ("dog, cat, bird", ["dog", "cat", "bird"]),
        ("dog,cat", ["dog", "cat"]),
        ("  Dog ,  CAT  ", ["dog", "cat"]),
        ("dog\ncat", ["dog", "cat"]),
        ("dog\r\ncat\r\n", ["dog", "cat"]),
        ("dog, , ,cat,", ["dog", "cat"]),
        ("dog, ..., (), cat", ["dog", "cat"]),
        ("big dog. [blurry], (sunset)", ["big dog blurry", "sunset"]),
        # Duplicates are kept; de-duplication is left to the callers.
        ("dog, Dog, dog", ["dog", "dog", "dog"]),
        ("Straße, ÉCOLE", ["straße", "école"]),
        ("", []),
        ("  \n , \r\n ", []),
    ],
)
def test_parse_tags_text(text, expected):
    assert parse_tags_text(text) == expected
