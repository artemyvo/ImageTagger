"""Parsing of the validation (fixup) answer: ISSUES / TAGS / DESCRIPTION sections."""
from __future__ import annotations

import pytest

from imagetagger.utils.annotations import parse_tags_text, sanitize_annotation_text
from imagetagger.utils.fixup_parser import (
    has_fixup_section_headers,
    parse_fixup_data,
    strip_tag_list_prefix,
)


def _parse(content: str):
    """Parse with the same tag parser and sanitizer MainWindow passes in."""
    return parse_fixup_data(content, parse_tags_text, sanitize_annotation_text)


def test_full_answer_splits_into_sections():
    parsed = _parse(
        "ISSUES:\n"
        "The tags miss the dog.\n"
        "TAGS:\n"
        "red car\n"
        "dog\n"
        "DESCRIPTION:\n"
        "Car parked on a street next to a dog.\n"
    )
    assert parsed.has_headers
    assert parsed.issues == "The tags miss the dog."
    assert parsed.corrected_tags == ["red car", "dog"]
    assert parsed.corrected_description_raw == "Car parked on a street next to a dog."
    assert parsed.corrected_description == "Car parked on a street next to a dog."


def test_sections_in_any_order():
    parsed = _parse("DESCRIPTION:\nCat on a sofa.\nTAGS:\ncat\nsofa\nISSUES:\nWrong animal.")
    assert parsed.issues == "Wrong animal."
    assert parsed.corrected_tags == ["cat", "sofa"]
    assert parsed.corrected_description_raw == "Cat on a sofa."


@pytest.mark.parametrize(
    "header",
    [
        "TAGS:",
        "tags:",
        "Tags:",
        "TAGS :",
        "### Tags:",
        "> TAGS:",
        "- TAGS:",
        "  TAGS:",
    ],
)
def test_header_variants_are_recognised(header):
    parsed = _parse(f"ISSUES:\nMissing tag.\n{header}\ncat\n")
    assert parsed.has_headers
    assert parsed.issues == "Missing tag."
    assert parsed.corrected_tags == ["cat"]


def test_bold_header_with_colon_outside_is_recognised():
    parsed = _parse("**Issues**:\nMissing tag.\n**Tags**:\ncat\n")
    assert parsed.issues == "Missing tag."
    assert parsed.corrected_tags == ["cat"]


@pytest.mark.parametrize(
    "line",
    [
        "Description incorrectly names the animal.",
        "Tags are missing the dog",
        "TAGSET: x",
        "TAGS_EXTRA: x",
    ],
)
def test_keyword_without_colon_is_content_not_a_header(line):
    parsed = _parse(f"ISSUES:\n{line}\nTAGS:\ncat")
    assert parsed.issues == line
    assert parsed.corrected_tags == ["cat"]


def test_content_on_the_header_line_is_kept():
    parsed = _parse("ISSUES: Wrong color.\nTAGS: red car, blue sky\nDESCRIPTION: Red car under a blue sky.")
    assert parsed.issues == "Wrong color."
    assert parsed.corrected_tags == ["red car", "blue sky"]
    assert parsed.corrected_description_raw == "Red car under a blue sky."


@pytest.mark.parametrize(
    "tags_block, expected",
    [
        ("cat\ndog", ["cat", "dog"]),
        ("cat, dog", ["cat", "dog"]),
        ("cat,\ndog,\n", ["cat", "dog"]),
        ("- cat\n- dog", ["cat", "dog"]),
        ("- - cat", ["cat"]),
        ("Red Car.", ["red car"]),
        ("cat\n\n\ndog", ["cat", "dog"]),
    ],
)
def test_tags_are_split_and_cleaned(tags_block, expected):
    parsed = _parse(f"ISSUES:\nx\nTAGS:\n{tags_block}\nDESCRIPTION:\nd")
    assert parsed.corrected_tags == expected


def test_description_commas_are_removed_but_raw_keeps_them():
    """MainWindow compares raw and clean to warn that the model broke the no-commas rule."""
    parsed = _parse("ISSUES:\nx\nDESCRIPTION:\nCat sits, calm and still (mostly).")
    assert parsed.corrected_description_raw == "Cat sits, calm and still (mostly)."
    assert "," not in parsed.corrected_description
    assert "(" not in parsed.corrected_description
    assert parsed.corrected_description == "Cat sits calm and still mostly ."


def test_multiline_description_is_joined_by_newlines_in_raw():
    parsed = _parse("DESCRIPTION:\nFirst sentence.\n\nSecond sentence.")
    assert parsed.corrected_description_raw == "First sentence.\nSecond sentence."
    assert parsed.corrected_description == "First sentence. Second sentence."


def test_empty_sections_give_empty_fields():
    parsed = _parse("ISSUES:\nWrong tag.\nTAGS:\nDESCRIPTION:\n")
    assert parsed.issues == "Wrong tag."
    assert parsed.corrected_tags == []
    assert parsed.corrected_description_raw == ""
    assert parsed.corrected_description == ""


def test_text_before_any_header_counts_as_issues():
    parsed = _parse("The dog is not a cat.\nTAGS:\ndog")
    assert parsed.issues == "The dog is not a cat."
    assert parsed.corrected_tags == ["dog"]


def test_answer_without_headers_becomes_the_issues_text():
    parsed = _parse("  The image shows a dog, not a cat.  ")
    assert not parsed.has_headers
    assert parsed.issues == "The image shows a dog, not a cat."
    assert parsed.corrected_tags == []
    assert parsed.corrected_description == ""


def test_ok_answer_parses_as_issues_text():
    """Callers check for a bare OK before parsing; the parser itself has no special case."""
    parsed = _parse("OK")
    assert not parsed.has_headers
    assert parsed.issues == "OK"


@pytest.mark.parametrize(
    "content, field, expected",
    [
        ("ISSUES:\nx\n**TAGS:**\ncat", "corrected_tags", ["cat"]),
        ("ISSUES:\nx\n**TAGS:** cat, dog", "corrected_tags", ["cat", "dog"]),
        ("ISSUES:\nx\n**DESCRIPTION:** Cat sits.", "corrected_description_raw", "Cat sits."),
        ("**ISSUES:** Wrong animal.", "issues", "Wrong animal."),
    ],
)
def test_bold_header_markup_is_not_content(content, field, expected):
    assert getattr(_parse(content), field) == expected


def test_headers_with_nothing_under_them_parse_as_empty():
    parsed = _parse("ISSUES:\nTAGS:\nDESCRIPTION:\n")
    assert parsed.has_headers
    assert parsed.issues == ""
    assert parsed.corrected_tags == []
    assert parsed.corrected_description_raw == ""


def test_search_matches_are_normalised_and_deduplicated():
    parsed = _parse("AI_FIND_MATCHES:\n- Red Car.\nred car\n\n(dog)\nDog")
    assert parsed.search_matches == ["red car", "dog"]


def test_vision_sections():
    parsed = _parse(
        "ISSUES:\nx\n"
        "VISIONTAGS:\n- Oil Painting\nportrait, dark background\n"
        "VISIONDESC:\nOil painting of a man.\nDark background behind him."
    )
    assert parsed.vision_tags == ["oil painting", "portrait", "dark background"]
    assert parsed.vision_caption == "Oil painting of a man. Dark background behind him."
    assert parsed.corrected_tags == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ISSUES:\nx", True),
        ("Some preamble\n  **Description:** y", True),
        ("visiontags: a", True),
        ("AI_FIND_MATCHES :", True),
        ("OK", False),
        ("", False),
        ("Description of the image without a colon", False),
        ("The TAGS: are wrong", False),
    ],
)
def test_has_fixup_section_headers(text, expected):
    assert has_fixup_section_headers(text) is expected


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("cat", "cat"),
        ("  - cat  ", "cat"),
        ("- - - cat", "cat"),
        ("-cat", "-cat"),
        ("-", "-"),
        ("cat - dog", "cat - dog"),
    ],
)
def test_strip_tag_list_prefix(tag, expected):
    assert strip_tag_list_prefix(tag) == expected
