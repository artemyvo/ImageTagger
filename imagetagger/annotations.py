from __future__ import annotations


def normalize_description_text(text: str) -> str:
    return " ".join(text.split())


def sanitize_annotation_text(text: str) -> str:
    sanitized = text
    for char in [",", ".", "[", "]", "(", ")"]:
        sanitized = sanitized.replace(char, " ")
    return normalize_description_text(sanitized)


def parse_tags_text(text: str) -> list[str]:
    raw = text.replace("\r", "").replace("\n", ",")
    tags: list[str] = []
    for part in raw.split(","):
        cleaned = sanitize_annotation_text(part)
        if cleaned:
            tags.append(cleaned)
    return tags
