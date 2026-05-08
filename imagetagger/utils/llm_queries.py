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
_EXISTING_TAGS_PLACEHOLDER = "{existing_tags}"
_AGENT_ROLE_PLACEHOLDER = "{agent_role}"


_prompt_file_cache: dict[str, str] = {}


def _load_prompt(filename: str, default: str) -> str:
    cached = _prompt_file_cache.get(filename)
    if cached is not None:
        return cached
    try:
        text = (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return default
    _prompt_file_cache[filename] = text
    return text


_DEFAULT_TAGS_PROMPT = (
    f"{_AGENT_ROLE_PLACEHOLDER}\n"
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
    "Seed tags already confirmed for this image (optional):\n"
    f"{_EXISTING_TAGS_PLACEHOLDER}\n"
    "If provided, include these in your output and generate additional complementary tags.\n\n"
    "Optional user hint for this revalidation:\n"
    f"{_USER_HINT_PLACEHOLDER}\n"
    "If provided, treat the hint as a correction constraint for this output."
)

_DEFAULT_DESCRIPTION_PROMPT = (
    f"{_AGENT_ROLE_PLACEHOLDER}\n"
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
    "Confirmed tags for this image (optional):\n"
    f"{_EXISTING_TAGS_PLACEHOLDER}\n"
    "If provided, use these as known facts about the subject when writing the description.\n\n"
    "Optional user hint for this revalidation:\n"
    f"{_USER_HINT_PLACEHOLDER}\n"
    "If provided, treat the hint as a correction constraint for this output."
)

_DEFAULT_VISION_PROMPT = (
    "You are a Lead Visual Auditor. Your goal is to generate high-density, synthetically complete training captions grounded in physical and technical logic.\n\n"
    "INPUT:\n"
    "Tags: {tags}\n"
    f"User Hint: {_USER_HINT_PLACEHOLDER}\n\n"
    "TASK:\n"
    "1. THOUGHT (Technical Scene Audit):\n"
    "- [Medium Detection]: Identify the medium (Photograph, 2D Animation, Painting, CGI).\n"
    "- [Identity Synthesis]: Identify the subject directly using {tags} as truth. Verify via silhouette markers. NEVER mention \"tags\" or \"input\".\n"
    "- [Structural & Kinetic Logic]: For Photo/CGI: analyze weight distribution, fabric tension, and forces. For Art/2D: analyze line weight, color fills, and spatial flattening.\n"
    "- [Material & Light Physics]: For Photo/CGI: trace light interaction (SSS, specularity, diffusion). For Art/2D: describe surface treatment (cel shading, brushwork).\n"
    "- [Staging]: Map depth layers (FG, MG, BG).\n\n"
    "2. DESCRIPTION (High-Density Prompt):\n"
    "- [Medium Lead]: If NOT a standard photograph, start by identifying the medium (e.g., \"A 2D cel-shaded animation of...\", \"An oil painting of...\").\n"
    "- [Syntactic Completeness]: Use full, complex sentences. NO telegraphic lists.\n"
    "- [Style-Appropriate Density]:\n"
    "    * For Photography: Use rich, sensory-visual language to describe textures, light behavior, and physical tension (e.g., \"stretched taut\", \"bathes in warm illumination\", \"glossy reflections\").\n"
    "    * For 2D/Stylized: Use technical-graphic language (e.g., \"flat color fills\", \"bold black outlines\", \"clean vector-like edges\").\n"
    "- [Bans]: NO \"This image shows\". NO meta-commentary. NO internal jargon (vector, SSS, hydraulic, mass).\n\n"
    "OUTPUT FORMAT:\n"
    "THOUGHT:\n"
    "(Expert deduction—internal use only.)\n\n"
    "DESCRIPTION:\n"
    "(Fluid, high-density generative prompt.)\n"
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

_DEFAULT_REFINE_PROMPT = (
    "You are a high-precision data distillation unit. Your sole purpose is to convert narrative vision analysis into structured metadata. \n\n"
    "INPUT:\n"
    "{vision_data}\n\n"
    "TASK:\n"
    "1. PRIMARY IDENTIFICATION: Extract the core subject and its specific type or name. \n"
    "2. MEDIUM & STYLE: Explicitly identify the format (e.g., photo, 2D vector, oil painting, screencap, digital illustration).\n"
    "3. ATTRIBUTES: List all verifiable physical properties, colors, and textures mentioned in the source.\n"
    "4. ENVIRONMENT: Describe the setting, spatial orientation, and lighting conditions.\n"
    "5. CONSTRAINTS: \n"
    "   - Use only objective, physical descriptors.\n"
    "   - Strictly prohibit subjective or emotional terms (e.g., epic, shocking, cute, impressive).\n"
    "   - Do not speculate on intent.\n\n"
    "OUTPUT STRUCTURE:\n"
    "TAGS: [comma-separated list of keywords]\n"
    "CAPTION: [One concise sentence following the structure: \"A [Medium] of [Subject] [Action/State] in [Environment].\"]"
)

_PROMPT_DEFAULTS: dict[str, str] = {
    "tagging": _DEFAULT_TAGS_PROMPT,
    "description": _DEFAULT_DESCRIPTION_PROMPT,
    "vision": _DEFAULT_VISION_PROMPT,
    "validation": _DEFAULT_VALIDATION_PROMPT,
    "search": _DEFAULT_SEARCH_PROMPT,
    "refine": _DEFAULT_REFINE_PROMPT,
}

_PROMPT_FILENAMES: dict[str, str] = {
    "tagging": "tags_prompt.txt",
    "description": "description_prompt.txt",
    "vision": "vision_prompt.txt",
    "validation": "validation_prompt.txt",
    "search": "search_prompt.txt",
    "refine": "refine_prompt.txt",
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
    _prompt_file_cache.pop(_PROMPT_FILENAMES[kind], None)
    return text


def reset_prompt_to_default(kind: str) -> str:
    _assert_prompt_kind(kind)
    default_text = _PROMPT_DEFAULTS[kind]
    try:
        _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        (_PROMPTS_DIR / _PROMPT_FILENAMES[kind]).write_text(default_text, encoding="utf-8")
    except OSError as exc:
        raise LlmQueryError(f"Could not reset prompt file: {exc}") from exc
    _prompt_file_cache.pop(_PROMPT_FILENAMES[kind], None)
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


def render_prompt_with_agent_role(prompt: str, agent_role: str | None = None) -> str:
    normalized_role = agent_role.strip() if isinstance(agent_role, str) else ""
    if _AGENT_ROLE_PLACEHOLDER in prompt:
        if normalized_role:
            return prompt.replace(_AGENT_ROLE_PLACEHOLDER, normalized_role)
        # Remove the placeholder and the trailing newline that follows it
        return prompt.replace(_AGENT_ROLE_PLACEHOLDER + "\n", "").replace(_AGENT_ROLE_PLACEHOLDER, "")

    if not normalized_role:
        return prompt

    return f"{normalized_role}\n{prompt}"


def render_prompt_with_existing_tags(prompt: str, existing_tags: list[str] | None = None) -> str:
    has_tags = bool(existing_tags)
    replacement = "\n".join(existing_tags) if has_tags else "none"
    if _EXISTING_TAGS_PLACEHOLDER in prompt:
        return prompt.replace(_EXISTING_TAGS_PLACEHOLDER, replacement)

    if not has_tags:
        return prompt

    return (
        f"{prompt.rstrip()}\n\n"
        "Confirmed tags for this image (treat as known facts):\n"
        + "\n".join(existing_tags) + "\n"
    )


def render_prompt_with_user_hint(prompt: str, user_hint: str | None = None) -> str:
    normalized_hint = user_hint.strip() if isinstance(user_hint, str) else ""
    replacement = normalized_hint if normalized_hint else "none"
    if _USER_HINT_PLACEHOLDER in prompt:
        return prompt.replace(_USER_HINT_PLACEHOLDER, replacement)

    if not normalized_hint:
        return prompt

    return (
        f"{prompt.rstrip()}\n\n"
        "User correction hint (must be followed if consistent with visible content):\n"
        f"{normalized_hint}\n"
    )


def format_annotations_for_validation(annotations: str) -> str:
    lines = [part.strip() for part in annotations.replace("\n", ",").split(",") if part.strip()]
    return "\n".join(f"- {line}" for line in lines)


def prepare_tagging_query(*, existing_tags: list[str] | None = None, agent_role: str | None = None) -> PreparedVisionQuery:
    prompt = render_prompt_with_agent_role(_active_prompt("tagging"), agent_role)
    prompt = render_prompt_with_existing_tags(prompt, existing_tags)
    return PreparedVisionQuery(
        kind="tagging",
        prompt=render_prompt_with_user_hint(prompt),
        metadata={"task": "tags"},
    )


def prepare_description_query(*, existing_tags: list[str] | None = None, agent_role: str | None = None) -> PreparedVisionQuery:
    prompt = render_prompt_with_agent_role(_active_prompt("description"), agent_role)
    prompt = render_prompt_with_existing_tags(prompt, existing_tags)
    return PreparedVisionQuery(
        kind="description",
        prompt=render_prompt_with_user_hint(prompt),
        metadata={"task": "description"},
    )


def prepare_vision_query(*, tags_text: str, user_hint: str | None = None) -> PreparedVisionQuery:
    prompt = _active_prompt("vision").replace("{tags}", tags_text.strip())
    return PreparedVisionQuery(
        kind="vision",
        prompt=render_prompt_with_user_hint(prompt, user_hint),
        metadata={"task": "vision"},
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


def prepare_refine_query(*, description: str, reasoning: str) -> PreparedVisionQuery:
    parts: list[str] = []
    if description:
        parts.append(f'description: "{description}"')
    if reasoning:
        parts.append(f'reasoning: "{reasoning}"')
    vision_text = "\n".join(parts) if parts else "(no vision data available for this image)"
    prompt = _active_prompt("refine").replace("{vision_data}", vision_text)
    return PreparedVisionQuery(
        kind="refine",
        prompt=prompt,
        metadata={"task": "refine"},
    )


def parse_refine_response(response: str) -> tuple[list[str], str]:
    """
    Parse the refine prompt response into (tags, caption).

    Expected format:
    TAGS: keyword1, keyword2, ...
    CAPTION: A photo of ...
    """
    text = normalize_model_text_block(response)
    if not text:
        return ([], "")

    tags: list[str] = []
    caption = ""

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TAGS:"):
            raw = stripped[5:].strip().strip("[]")
            tags = [t.strip() for t in raw.split(",") if t.strip()]
        elif upper.startswith("CAPTION:"):
            caption = stripped[8:].strip().strip("[]\"'")

    return (tags, caption)


def parse_yes_no_response(response: str, *, context: str = "AI Find") -> bool:
    from imagetagger.providers.llm_provider import LlmProviderError

    match = re.search(r"\b(YES|NO)\b", response, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        if val == "YES":
            return True
        if val == "NO":
            return False
    raise LlmProviderError(f"Model returned ambiguous {context} response: {response.strip()}")


def normalize_model_text_block(text: str) -> str:
    # Some models return literal escape sequences (e.g. "\\n") in plain text.
    normalized = str(text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\\n", "\n").replace("\\t", "\t")
    return normalized.strip()


def parse_vision_response(response: str) -> tuple[str, str]:
    """
    Parse the vision prompt response into (reasoning, description).

    Expected format:
    THOUGHT:
    ...
    DESCRIPTION:
    ...
    """
    text = normalize_model_text_block(response)
    if not text:
        return ("", "")

    upper = text.upper()
    thought_idx = upper.find("THOUGHT:")
    desc_idx = upper.find("DESCRIPTION:")

    if desc_idx < 0:
        # Allow models that omit headers; treat entire response as description.
        return ("", text)

    thought = ""
    if thought_idx >= 0 and thought_idx < desc_idx:
        thought = text[thought_idx + len("THOUGHT:") : desc_idx].strip()
    description = text[desc_idx + len("DESCRIPTION:") :].strip()
    return (thought, description)
