from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


class LlmQueryError(Exception):
    pass


@dataclass(frozen=True)
class PreparedVisionQuery:
    kind: str
    prompt: str
    metadata: dict[str, str] = field(default_factory=dict)


_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_USER_HINT_PLACEHOLDER = "{user_hint}"


def _load_prompt(filename: str, default: str) -> str:
    try:
        return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return default


_DEFAULT_TAGS_PROMPT = (
    "Analyze the image and return a list of short descriptive tags.\n\n"
    "Rules:\n"
    "- Do not include introductory text or greetings.\n"
    "- Return between 10 and 20 tags.\n"
    "- One tag per line.\n"
    "- One or two words maximum per tag.\n"
    "- Lowercase only.\n"
    "- No numbering or sentences.\n"
    "- Cover a range of aspects: subject, action or pose, setting, mood, lighting, colors, and style where applicable.\n"
    "- Do not repeat the same concept with different words.\n\n"
    "Example (if subject is a person outdoors):\n"
    "outdoor setting\n"
    "natural light\n"
    "seated pose\n"
    "warm tones\n"
    "relaxed mood\n"
    "wooden surface\n"
    "casual style\n"
    "green foliage\n"
    "shallow depth\n"
    "soft shadows\n\n"
    "Optional user hint for this revalidation:\n"
    f"{_USER_HINT_PLACEHOLDER}\n"
    "If provided, treat the hint as a correction constraint for this output."
)

_DEFAULT_DESCRIPTION_PROMPT = (
    "Analyze the image and return ONLY a single paragraph of 2-3 descriptive sentences.\n\n"
    "Rules:\n"
    "- Identify the main subject (e.g., woman, cat, car, pan) and start the very first sentence with that noun.\n"
    "- Use the bare noun without an article (no 'A', 'An', or 'The') to start the first sentence.\n"
    "- Use the specific noun instead of pronouns like \"it\", \"she\", or \"he\" to start first sentence.\n"
    "- Write declarative statements; do not use hedging phrases like \"appears to\" or \"seems to\".\n"
    "- Avoid comma-separated lists.\n"
    "- Do NOT include any introductory text, greetings, or meta-talk.\n\n"
    "Example (if subject is a cat):\n"
    "Cat with orange tabby fur sits on a blue velvet sofa in a sunlit room. Cat gazes out a large window while twitching its tail in a playful manner. Soft dust motes dance in the light to create a peaceful atmosphere.\n\n"
    "Optional user hint for this revalidation:\n"
    f"{_USER_HINT_PLACEHOLDER}\n"
    "If provided, treat the hint as a correction constraint for this output."
)

_DEFAULT_VALIDATION_PROMPT = (
    "You are validating image annotations. The annotation list may include short tags and one long description.\n"
    "The original storage format may use commas only as separators between annotations. Those separator commas are not mistakes.\n"
    "Missing commas are also not a problem.\n"
    "Current annotations, one per line:\n{tags}\n"
    "Analyze the image and verify whether the annotations are accurate and complete.\n"
    "If everything is correct, reply with exactly: OK\n"
    "If there are problems, reply using exactly this plain-text format:\n"
    "ISSUES:\n"
    "<brief explanation of what is wrong>\n"
    "TAGS:\n"
    "<corrected tags, one per line, no commas>\n"
    "DESCRIPTION:\n"
    "<corrected description without commas, or leave blank if no description is needed>\n"
    "Do not return JSON or any extra headings."
)

_DEFAULT_SEARCH_PROMPT = (
    "You are checking whether an image contains a target concept.\n"
    "Target concept: \"{query}\"\n"
    "Respond with exactly one token: YES or NO.\n"
    "Do not add punctuation, explanations, or extra words."
)

_PROMPT_DEFAULTS: dict[str, str] = {
    "tagging": _DEFAULT_TAGS_PROMPT,
    "description": _DEFAULT_DESCRIPTION_PROMPT,
    "validation": _DEFAULT_VALIDATION_PROMPT,
    "search": _DEFAULT_SEARCH_PROMPT,
}

_PROMPT_FILENAMES: dict[str, str] = {
    "tagging": "tags_prompt.txt",
    "description": "description_prompt.txt",
    "validation": "validation_prompt.txt",
    "search": "search_prompt.txt",
}

_PROMPT_OVERRIDES: dict[str, str] = {}


def _assert_prompt_kind(kind: str) -> None:
    if kind not in _PROMPT_DEFAULTS:
        raise LlmQueryError(f"Unknown prompt kind: {kind}")


def get_default_prompt(kind: str) -> str:
    _assert_prompt_kind(kind)
    return _PROMPT_DEFAULTS[kind]


def load_prompt_for_kind(kind: str) -> str:
    _assert_prompt_kind(kind)
    return _load_prompt(_PROMPT_FILENAMES[kind], _PROMPT_DEFAULTS[kind])


def prompt_source_for_kind(kind: str) -> str:
    _assert_prompt_kind(kind)
    if kind in _PROMPT_OVERRIDES:
        return "memory"

    prompt_path = _PROMPTS_DIR / _PROMPT_FILENAMES[kind]
    try:
        prompt_path.read_text(encoding="utf-8")
    except OSError:
        return "default"
    return "file"


def set_prompt_override(kind: str, prompt: str) -> None:
    _assert_prompt_kind(kind)
    _PROMPT_OVERRIDES[kind] = prompt.strip()


def clear_prompt_override(kind: str) -> None:
    _assert_prompt_kind(kind)
    _PROMPT_OVERRIDES.pop(kind, None)


def save_prompt_for_kind(kind: str, prompt: str) -> str:
    _assert_prompt_kind(kind)
    text = prompt.strip()
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        (_PROMPTS_DIR / _PROMPT_FILENAMES[kind]).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise LlmQueryError(f"Could not save prompt file: {exc}") from exc
    return text


def reset_prompt_to_default(kind: str) -> str:
    _assert_prompt_kind(kind)
    default_text = _PROMPT_DEFAULTS[kind]
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        (_PROMPTS_DIR / _PROMPT_FILENAMES[kind]).write_text(default_text, encoding="utf-8")
    except OSError as exc:
        raise LlmQueryError(f"Could not reset prompt file: {exc}") from exc
    clear_prompt_override(kind)
    return default_text


def _active_prompt(kind: str) -> str:
    _assert_prompt_kind(kind)
    override = _PROMPT_OVERRIDES.get(kind)
    if override is not None:
        return override
    return load_prompt_for_kind(kind)


def active_prompt_for_kind(kind: str) -> str:
    _assert_prompt_kind(kind)
    return _active_prompt(kind)


def render_prompt_with_user_hint(prompt: str, user_hint: str | None = None) -> str:
    normalized_hint = user_hint.strip() if isinstance(user_hint, str) else ""
    replacement = normalized_hint if normalized_hint else "none"
    return prompt.replace(_USER_HINT_PLACEHOLDER, replacement)


def format_annotations_for_validation(annotations: str) -> str:
    lines = [part.strip() for part in annotations.replace("\n", ",").split(",") if part.strip()]
    return "\n".join(f"- {line}" for line in lines)


def prepare_tagging_query() -> PreparedVisionQuery:
    return PreparedVisionQuery(
        kind="tagging",
        prompt=render_prompt_with_user_hint(_active_prompt("tagging")),
        metadata={"task": "tags"},
    )


def prepare_description_query() -> PreparedVisionQuery:
    return PreparedVisionQuery(
        kind="description",
        prompt=render_prompt_with_user_hint(_active_prompt("description")),
        metadata={"task": "description"},
    )


def prepare_validation_query(annotations: str) -> PreparedVisionQuery:
    return PreparedVisionQuery(
        kind="validation",
        prompt=_active_prompt("validation").replace("{tags}", format_annotations_for_validation(annotations)),
        metadata={"task": "validation"},
    )


def prepare_search_query(query: str) -> PreparedVisionQuery:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise LlmQueryError("Enter text to search for.")

    return PreparedVisionQuery(
        kind="search",
        prompt=_active_prompt("search").replace("{query}", cleaned_query),
        metadata={"task": "search", "query": cleaned_query},
    )


def parse_yes_no_response(response: str, *, context: str = "AI Find") -> bool:
    from imagetagger.llm_provider import LlmProviderError

    match = re.search(r"\b(YES|NO)\b", response, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        if val == "YES":
            return True
        if val == "NO":
            return False
    raise LlmProviderError(f"Model returned ambiguous {context} response: {response.strip()}")
