from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from imagetagger.utils.annotations import sanitize_description_text


def strip_tag_list_prefix(tag: str) -> str:
    """Remove one or more leading markdown list markers from a tag value."""
    cleaned = tag.strip()
    while cleaned.startswith("- "):
        cleaned = cleaned[2:].lstrip()
    return cleaned


def _normalize_fixup_section_entry(value: str) -> str:
    """Normalize entry by stripping whitespace and leading dash."""
    text = value.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    return text


def _normalize_search_match_entry(value: str, sanitize_annotation: Callable[[str], str]) -> str:
    normalized = sanitize_annotation(_normalize_fixup_section_entry(value)).strip()
    return normalized.lower()


# Headers like "ISSUES:", "### Tags :", "**Description**:" or "**TAGS:**".
# The ':' is required so content lines such as "Description incorrectly..."
# are not taken for a DESCRIPTION header; bold/italic markup may close
# before it.  "markup" is the opening '*'/'_' run, stripped again after the
# colon.
_HEADER_PATTERN = re.compile(
    r"^[#\s>\-]*(?P<markup>[*_]*)\s*"
    r"(?P<keyword>ISSUES|TAGS|DESCRIPTION|AI_FIND_MATCHES|VISIONTAGS|VISIONDESC)\b[*_\s]*:",
    re.IGNORECASE,
)


@dataclass
class FixupData:
    issues: str
    corrected_description: str
    corrected_description_raw: str
    corrected_tags: list[str]
    search_matches: list[str] = field(default_factory=list)
    vision_tags: list[str] = field(default_factory=list)
    vision_caption: str = ""
    has_headers: bool = False


def parse_fixup_data(
    content: str,
    parse_tags: Callable[[str], list[str]],
    sanitize_annotation: Callable[[str], str],
) -> FixupData:
    sections: dict[str, list[str]] = {"issues": [], "tags": [], "description": [], "ai_find": [], "visiontags": [], "visiondesc": []}
    current_section = "issues"
    has_headers = False

    header_pattern = _HEADER_PATTERN
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = header_pattern.match(line)
        if match:
            has_headers = True
            header_keyword = match.group("keyword").upper()
            if header_keyword == "ISSUES":
                current_section = "issues"
            elif header_keyword == "TAGS":
                current_section = "tags"
            elif header_keyword == "DESCRIPTION":
                current_section = "description"
            elif header_keyword == "AI_FIND_MATCHES":
                current_section = "ai_find"
            elif header_keyword == "VISIONTAGS":
                current_section = "visiontags"
            elif header_keyword == "VISIONDESC":
                current_section = "visiondesc"

            # Content on the same line after the header, without the closing
            # markup of a bold/italic header ("**TAGS:** cat" -> "cat").
            inline_content = line[match.end():].strip()
            opening_markup = match.group("markup")
            if opening_markup and inline_content.startswith(opening_markup):
                inline_content = inline_content[len(opening_markup):].strip()
            if inline_content:
                sections[current_section].append(inline_content)
            continue

        sections[current_section].append(raw_line.rstrip())

    issues = "\n".join(line for line in sections["issues"] if line.strip()).strip()
    corrected_description_raw = "\n".join(line for line in sections["description"] if line.strip()).strip()
    corrected_description = sanitize_description_text(corrected_description_raw)
    tags_text = "\n".join(line.strip() for line in sections["tags"] if line.strip())
    corrected_tags = [
        cleaned
        for tag in parse_tags(tags_text)
        if (cleaned := strip_tag_list_prefix(tag))
    ]

    search_matches = []
    seen_search_matches: set[str] = set()
    for line in sections["ai_find"]:
        normalized_match = _normalize_search_match_entry(line, sanitize_annotation)
        if not normalized_match:
            continue
        if normalized_match in seen_search_matches:
            continue
        seen_search_matches.add(normalized_match)
        search_matches.append(normalized_match)

    vision_tags_text = "\n".join(line.strip() for line in sections["visiontags"] if line.strip())
    vision_tags = [
        cleaned
        for tag in parse_tags(vision_tags_text)
        if (cleaned := strip_tag_list_prefix(tag))
    ]
    vision_caption = " ".join(line.strip() for line in sections["visiondesc"] if line.strip())

    # Without any header the whole answer is the issue text.  With headers
    # and nothing under them, the answer is clean: leave everything empty.
    if not has_headers and not issues and not corrected_description and not corrected_tags:
        issues = content.strip()

    return FixupData(
        issues=issues,
        corrected_description=corrected_description,
        corrected_description_raw=corrected_description_raw,
        corrected_tags=corrected_tags,
        search_matches=search_matches,
        vision_tags=vision_tags,
        vision_caption=vision_caption,
        has_headers=has_headers,
    )


def has_fixup_section_headers(text: str) -> bool:
    """Return True if the text contains at least one recognised fixup section header."""
    return any(_HEADER_PATTERN.match(line.strip()) for line in text.splitlines())
