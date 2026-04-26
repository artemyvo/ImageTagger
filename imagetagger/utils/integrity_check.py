"""
Integrity checking utilities for detecting corrupted annotations in .txt files.

Two historical bugs caused annotation files to be corrupted:
  1. Commas were allowed inside descriptions, which split them into multiple fake tags
     because the .txt format is comma-separated.
  2. Descriptions were stored at a random position instead of the beginning of the file.

Both corruptions share a common marker: a period immediately followed by a comma (".,")
in the raw file text, which is the boundary where the description ended and was followed
by real tags. The description start is identified by an uppercase letter at the beginning
of an item (tags are always lowercase; descriptions start with a capital letter).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class IntegrityIssue:
    """Represents a detected integrity issue in an annotation file."""
    image_path: str
    issue_type: str       # "description_with_commas" | "description_not_at_start" | "both"
    description: str      # The reconstructed clean description (original commas replaced by spaces)
    tags: list[str]       # The correct tags, in order (pre-description + post-description)
    raw_content: str      # Original file content for reference


def detect_corrupted_annotation(file_content: str) -> IntegrityIssue | None:
    """
    Detect and extract corrected data from a corrupted annotation file.

    A file is considered corrupted when a period-comma marker (".,") exists in the raw
    text, indicating that a description (ending with a period) was embedded inside the
    comma-separated tag stream rather than cleanly separated from it.

    Detection rules:
    - Description end:   identified by ".,": a period directly followed by a comma.
    - Description start: the first comma-separated item whose first character is
                         uppercase (descriptions start with a capital letter; tags are
                         always lowercase).
    - Pre-description tags: any lowercase items appearing before the uppercase start are
                         misplaced tags (corruption pattern 2: wrong position).
    - Description fragments: items from the uppercase start to ".,"; if more than one
                         item (due to commas inside the description), they are joined with
                         spaces to reconstruct the original sentence (corruption pattern 1).

    A clean file — description at position 0 with no internal commas — is never flagged
    (even though it also contains ".,").

    Returns an IntegrityIssue with the corrected description and tags, or None if no
    corruption is detected.
    """
    text = file_content.strip()
    if not text:
        return None

    # Locate the first ".,": marks the end of the (possibly fragmented) description.
    period_comma_match = re.search(r'\.,', text)
    if period_comma_match is None:
        return None

    period_pos = period_comma_match.start()
    after_period_comma = text[period_comma_match.end():].strip()

    # Split the portion up to and including the period into comma-separated items.
    before_section = text[:period_pos + 1]  # includes the trailing '.'
    before_items = [item.strip() for item in before_section.split(",") if item.strip()]
    if not before_items:
        return None

    # Find the description start: the first item whose first character is uppercase.
    # All items before it are lowercase tags that ended up in front of the description.
    desc_start_idx: int | None = None
    for i, item in enumerate(before_items):
        if item and item[0].isupper():
            desc_start_idx = i
            break

    if desc_start_idx is None:
        # No uppercase-starting item found before ".,"; not a recognisable corruption.
        return None

    pre_desc_tags = before_items[:desc_start_idx]
    description_items = before_items[desc_start_idx:]

    desc_has_commas = len(description_items) > 1   # description was split by commas
    desc_not_at_start = len(pre_desc_tags) > 0     # lowercase tags precede the description

    if not desc_has_commas and not desc_not_at_start:
        # Clean file: description is at position 0 with no internal commas.
        return None

    # Reconstruct the description by space-joining the fragments.
    # The last fragment retains the trailing '.' from the original text.
    description = " ".join(description_items)

    # Collect all tags: those that appeared before the description + those after it.
    post_tags = [t.strip() for t in after_period_comma.split(",") if t.strip()] if after_period_comma else []
    all_tags = pre_desc_tags + post_tags

    if desc_has_commas and desc_not_at_start:
        issue_type = "both"
    elif desc_has_commas:
        issue_type = "description_with_commas"
    else:
        issue_type = "description_not_at_start"

    return IntegrityIssue(
        image_path="",  # Caller fills this in if needed
        issue_type=issue_type,
        description=description,
        tags=all_tags,
        raw_content=text,
    )


def check_all_patterns(file_content: str) -> list[IntegrityIssue]:
    """
    Check a file for all known corruption patterns.

    Returns a list with at most one IntegrityIssue per file.
    """
    issue = detect_corrupted_annotation(file_content)
    return [issue] if issue else []
