"""Prompt rendering, prompt file/override handling and model answer parsing."""
from __future__ import annotations

import os

import pytest

from imagetagger.providers.llm_provider import LlmProviderError
from imagetagger.utils import llm_queries as q

KINDS = ["tagging", "description", "vision", "validation", "search", "refine"]


@pytest.fixture(autouse=True)
def prompts_dir(tmp_path, monkeypatch):
    """Point prompt files at an empty temp dir and give each test fresh override/cache state.

    The real prompts/ directory holds the user's custom prompts: never read or write it here.
    """
    directory = tmp_path / "prompts"
    monkeypatch.setattr(q, "_PROMPTS_DIR", directory)
    monkeypatch.setattr(q, "_PROMPT_OVERRIDES", {})
    monkeypatch.setattr(q, "_prompt_file_cache", {})
    return directory


def _write_prompt(directory, kind: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / q._PROMPT_FILENAMES[kind]).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response, expected",
    [
        ("YES", True),
        ("NO", False),
        ("yes", True),
        ("no", False),
        ("Yes.", True),
        ("No!", False),
        ("  YES\n", True),
        ("**YES**", True),
        ('"no"', False),
        ("Answer: YES", True),
        ("The answer is no, it is not there.", False),
        ("Nothing like that. NO", False),
        ("Nope. Not here. Yes", True),
    ],
)
def test_parse_yes_no_response(response, expected):
    assert q.parse_yes_no_response(response) is expected


@pytest.mark.parametrize("response", ["", "   ", "maybe", "Nope", "Yesterday", "not sure", "YESNO"])
def test_parse_yes_no_response_rejects_ambiguous_answers(response):
    with pytest.raises(LlmProviderError, match="ambiguous AI Find response"):
        q.parse_yes_no_response(response)


def test_parse_yes_no_response_error_names_the_context():
    with pytest.raises(LlmProviderError, match="ambiguous Search response: maybe"):
        q.parse_yes_no_response("  maybe  ", context="Search")


def test_parse_yes_no_response_ignores_inline_thinking():
    assert q.parse_yes_no_response("<think>No cat at first glance... wait, there is one.</think>\nYES") is True


@pytest.mark.parametrize(
    "text, expected",
    [
        (None, ""),
        ("", ""),
        ("  a  ", "a"),
        ("a\r\nb\rc", "a\nb\nc"),
        ("a\\nb\\tc", "a\nb\tc"),
        ("\\n a \\n", "a"),
    ],
)
def test_normalize_model_text_block(text, expected):
    assert q.normalize_model_text_block(text) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("THOUGHT:\nit is a cat\nDESCRIPTION:\nCat on a sofa.", ("it is a cat", "Cat on a sofa.")),
        ("thought: x\ndescription: y", ("x", "y")),
        ("THOUGHT:\\nx\\nDESCRIPTION:\\ny", ("x", "y")),
        ("DESCRIPTION:\nCat on a sofa.", ("", "Cat on a sofa.")),
        ("Preamble\nDESCRIPTION: y", ("", "y")),
        # THOUGHT after DESCRIPTION is not reasoning; it stays in the description.
        ("DESCRIPTION: y\nTHOUGHT: x", ("", "y\nTHOUGHT: x")),
        ("Cat on a sofa.", ("", "Cat on a sofa.")),
        ("THOUGHT: only reasoning", ("", "THOUGHT: only reasoning")),
        ("", ("", "")),
        ("DESCRIPTION:", ("", "")),
    ],
)
def test_parse_vision_response(response, expected):
    assert q.parse_vision_response(response) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("TAGS: cat, sofa\nCAPTION: A photo of a cat.", (["cat", "sofa"], "A photo of a cat.")),
        ("TAGS: [cat, sofa]\nCAPTION: [A photo of a cat.]", (["cat", "sofa"], "A photo of a cat.")),
        ('tags: cat,, sofa ,\ncaption: "A photo of a cat."', (["cat", "sofa"], "A photo of a cat.")),
        ("CAPTION: 'A photo.'\nTAGS: a", (["a"], "A photo.")),
        ("  TAGS: a\\nCAPTION: b", (["a"], "b")),
        ("Here you go:\nTAGS: a\nnoise\nCAPTION: b\ntrailing", (["a"], "b")),
        ("TAGS: a\nTAGS: b", (["b"], "")),
        ("TAGS:\nCAPTION:", ([], "")),
        ("just some text", ([], "")),
        ("", ([], "")),
    ],
)
def test_parse_refine_response(response, expected):
    assert q.parse_refine_response(response) == expected


# ---------------------------------------------------------------------------
# Placeholder rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt, role, expected",
    [
        ("{agent_role}\nDo it.", "You are an expert.", "You are an expert.\nDo it."),
        ("{agent_role}\nDo it.", "  You are an expert.  ", "You are an expert.\nDo it."),
        ("{agent_role}\nDo it.", None, "Do it."),
        ("{agent_role}\nDo it.", "   ", "Do it."),
        ("Start {agent_role} end", None, "Start  end"),
        ("Do it.", "Expert.", "Expert.\nDo it."),
        ("Do it.", None, "Do it."),
        ("Do it.", "", "Do it."),
    ],
)
def test_render_prompt_with_agent_role(prompt, role, expected):
    assert q.render_prompt_with_agent_role(prompt, role) == expected


@pytest.mark.parametrize(
    "prompt, tags, expected",
    [
        ("Seed:\n{existing_tags}\nGo", ["cat", "sofa"], "Seed:\ncat\nsofa\nGo"),
        ("Seed:\n{existing_tags}\nGo", [], "Seed:\nnone\nGo"),
        ("Seed:\n{existing_tags}\nGo", None, "Seed:\nnone\nGo"),
        ("Go\n\n", None, "Go\n\n"),
        ("Go\n\n", ["cat", "sofa"], "Go\n\nConfirmed tags for this image (treat as known facts):\ncat\nsofa\n"),
    ],
)
def test_render_prompt_with_existing_tags(prompt, tags, expected):
    assert q.render_prompt_with_existing_tags(prompt, tags) == expected


@pytest.mark.parametrize(
    "prompt, hint, expected",
    [
        ("Hint: {user_hint}", "  it is a dog ", "Hint: it is a dog"),
        ("Hint: {user_hint}", None, "Hint: none"),
        ("Hint: {user_hint}", "  ", "Hint: none"),
        ("Go ", None, "Go "),
        ("Go ", "dog", "Go\n\nUser correction hint (must be followed if consistent with visible content):\ndog\n"),
    ],
)
def test_render_prompt_with_user_hint(prompt, hint, expected):
    assert q.render_prompt_with_user_hint(prompt, hint) == expected


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("a {unknown} b", "a  b"),
        ("{agent_role}{tags}{_x1}", ""),
        ("keep {1abc} {with space} {} {a-b}", "keep {1abc} {with space} {} {a-b}"),
        ("no placeholders", "no placeholders"),
    ],
)
def test_strip_unresolved_placeholders(prompt, expected):
    assert q._strip_unresolved_placeholders(prompt) == expected


@pytest.mark.parametrize(
    "annotations, expected",
    [
        ("cat, sofa", "- cat\n- sofa"),
        ("cat\nsofa", "- cat\n- sofa"),
        ("cat,, \n ,sofa,", "- cat\n- sofa"),
        ("Cat sits on a sofa", "- Cat sits on a sofa"),
        ("", ""),
    ],
)
def test_format_annotations_for_validation(annotations, expected):
    assert q.format_annotations_for_validation(annotations) == expected


# ---------------------------------------------------------------------------
# Prepared queries (built-in prompts: the prompts dir is empty)
# ---------------------------------------------------------------------------


def test_prepare_tagging_query_fills_role_and_seed_tags():
    query = q.prepare_tagging_query(existing_tags=["cat", "sofa"], agent_role="You are a tagger.")
    assert query.kind == "tagging"
    assert query.metadata == {"task": "tags"}
    assert query.prompt.startswith("You are a tagger.\nAnalyze the image")
    assert "\ncat\nsofa\n" in query.prompt
    assert "Optional user hint for this revalidation:\nnone\n" in query.prompt
    assert "{" not in query.prompt


def test_prepare_tagging_query_without_role_or_tags():
    query = q.prepare_tagging_query()
    assert query.prompt.startswith("Analyze the image")
    assert "(optional):\nnone\n" in query.prompt
    assert "{" not in query.prompt


def test_prepare_description_query():
    query = q.prepare_description_query(existing_tags=["cat"], agent_role=None)
    assert query.kind == "description"
    assert query.metadata == {"task": "description"}
    assert query.prompt.startswith("Analyze the image and return ONLY")
    assert "Confirmed tags for this image (optional):\ncat\n" in query.prompt
    assert "{" not in query.prompt


def test_prepare_vision_query_fills_tags_and_hint():
    query = q.prepare_vision_query(tags_text="  cat, sofa \n", user_hint=" orange cat ")
    assert query.kind == "vision"
    assert query.metadata == {"task": "vision"}
    assert "Tags: cat, sofa\n" in query.prompt
    assert "User Hint: orange cat\n" in query.prompt
    assert "using cat, sofa as truth" in query.prompt
    assert "{" not in query.prompt


def test_prepare_vision_query_without_hint():
    assert "User Hint: none\n" in q.prepare_vision_query(tags_text="", user_hint=None).prompt


def test_prepare_validation_query_lists_annotations():
    query = q.prepare_validation_query("cat, sofa\nCat sits on a sofa")
    assert query.kind == "validation"
    assert query.metadata == {"task": "validation"}
    assert "one per line:\n- cat\n- sofa\n- Cat sits on a sofa\nAnalyze" in query.prompt
    assert "{" not in query.prompt


def test_prepare_search_query():
    query = q.prepare_search_query("  red car ")
    assert query.kind == "search"
    assert query.metadata == {"task": "search", "query": "red car"}
    assert 'Target concept: "red car"' in query.prompt


@pytest.mark.parametrize("text", ["", "   \n"])
def test_prepare_search_query_rejects_empty_text(text):
    with pytest.raises(q.LlmQueryError, match="Enter text"):
        q.prepare_search_query(text)


@pytest.mark.parametrize(
    "description, reasoning, expected_input",
    [
        ("Cat.", "Fur.", 'INPUT:\ndescription: "Cat."\nreasoning: "Fur."\n\n'),
        ("Cat.", "", 'INPUT:\ndescription: "Cat."\n\n'),
        ("", "Fur.", 'INPUT:\nreasoning: "Fur."\n\n'),
        ("", "", "INPUT:\n(no vision data available for this image)\n\n"),
    ],
)
def test_prepare_refine_query(description, reasoning, expected_input):
    query = q.prepare_refine_query(description=description, reasoning=reasoning)
    assert query.kind == "refine"
    assert query.metadata == {"task": "refine"}
    assert expected_input in query.prompt
    assert "{" not in query.prompt


def test_custom_prompt_placeholders_are_filled_or_stripped(prompts_dir):
    """A custom validation prompt may use tokens the validation path does not fill."""
    _write_prompt(prompts_dir, "validation", "{agent_role}\nRole-free.\nTags:\n{tags}\nSeed: {existing_tags}\nHint: {user_hint}\n{mystery}")
    prompt = q.prepare_validation_query("cat").prompt
    assert prompt == "Role-free.\nTags:\n- cat\nSeed: none\nHint: none\n"


# ---------------------------------------------------------------------------
# Prompt files and in-memory overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_defaults_are_used_when_no_file_exists(kind):
    assert q.prompt_source_for_kind(kind) == "default"
    assert q.load_prompt_for_kind(kind) == q.get_default_prompt(kind)
    assert q.active_prompt_for_kind(kind) == q.get_default_prompt(kind)


@pytest.mark.parametrize(
    "call",
    [
        lambda: q.get_default_prompt("bogus"),
        lambda: q.load_prompt_for_kind("bogus"),
        lambda: q.prompt_source_for_kind("bogus"),
        lambda: q.set_prompt_override("bogus", "x"),
        lambda: q.clear_prompt_override("bogus"),
        lambda: q.save_prompt_for_kind("bogus", "x"),
        lambda: q.reset_prompt_to_default("bogus"),
        lambda: q.active_prompt_for_kind("bogus"),
    ],
)
def test_unknown_prompt_kind_is_rejected(call, prompts_dir):
    with pytest.raises(q.LlmQueryError, match="Unknown prompt kind: bogus"):
        call()
    assert not prompts_dir.exists()


def test_prompt_file_is_used_and_stripped(prompts_dir):
    _write_prompt(prompts_dir, "search", "\n  Is there {query}?  \n")
    assert q.prompt_source_for_kind("search") == "file"
    assert q.load_prompt_for_kind("search") == "Is there {query}?"
    assert q.prepare_search_query("a dog").prompt == "Is there a dog?"


def test_edited_prompt_file_is_reloaded(prompts_dir):
    _write_prompt(prompts_dir, "search", "first {query}")
    assert q.load_prompt_for_kind("search") == "first {query}"
    path = prompts_dir / q._PROMPT_FILENAMES["search"]
    path.write_text("second {query}", encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    assert q.load_prompt_for_kind("search") == "second {query}"


def test_override_wins_over_file_and_clears_back(prompts_dir):
    _write_prompt(prompts_dir, "search", "file {query}")
    q.set_prompt_override("search", "  memory {query}  ")
    assert q.prompt_source_for_kind("search") == "memory"
    assert q.active_prompt_for_kind("search") == "memory {query}"
    assert q.load_prompt_for_kind("search") == "file {query}"
    assert q.prepare_search_query("x").prompt == "memory x"

    q.clear_prompt_override("search")
    assert q.prompt_source_for_kind("search") == "file"
    assert q.active_prompt_for_kind("search") == "file {query}"
    q.clear_prompt_override("search")  # clearing twice is harmless


def test_override_only_affects_its_kind():
    q.set_prompt_override("search", "memory")
    assert q.prompt_source_for_kind("refine") == "default"
    assert q.active_prompt_for_kind("refine") == q.get_default_prompt("refine")


def test_save_writes_stripped_text_and_invalidates_cache(prompts_dir):
    _write_prompt(prompts_dir, "search", "old {query}")
    assert q.load_prompt_for_kind("search") == "old {query}"

    assert q.save_prompt_for_kind("search", "  new {query}\n") == "new {query}"
    assert (prompts_dir / "search_prompt.txt").read_text(encoding="utf-8") == "new {query}"
    # Same-second rewrite: the cache must not serve the old text.
    assert q.load_prompt_for_kind("search") == "new {query}"
    assert q.prompt_source_for_kind("search") == "file"


def test_save_creates_the_prompts_dir(prompts_dir):
    assert not prompts_dir.exists()
    q.save_prompt_for_kind("refine", "custom")
    assert sorted(p.name for p in prompts_dir.iterdir()) == ["refine_prompt.txt"]


def test_reset_writes_default_and_drops_override(prompts_dir):
    _write_prompt(prompts_dir, "description", "custom")
    q.set_prompt_override("description", "memory")
    default = q.reset_prompt_to_default("description")

    assert default == q.get_default_prompt("description")
    assert (prompts_dir / "description_prompt.txt").read_text(encoding="utf-8") == default
    assert q.prompt_source_for_kind("description") == "file"
    assert q.active_prompt_for_kind("description") == default


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: q.save_prompt_for_kind("search", "x"), "Could not save prompt file"),
        (lambda: q.reset_prompt_to_default("search"), "Could not reset prompt file"),
    ],
)
def test_unwritable_prompts_dir_raises_query_error(tmp_path, monkeypatch, call, message):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(q, "_PROMPTS_DIR", blocker / "prompts")
    with pytest.raises(q.LlmQueryError, match=message):
        call()


def test_unreadable_prompt_path_falls_back_to_default(prompts_dir):
    # A directory where the file should be: stat works, reading fails.
    (prompts_dir / "search_prompt.txt").mkdir(parents=True)
    assert q.load_prompt_for_kind("search") == q.get_default_prompt("search")
    assert q.prompt_source_for_kind("search") == "default"


@pytest.mark.parametrize(
    "response, expected",
    [
        ("<think>No cat at first glance... wait, there is one.</think>\nYES", True),
        ("<THINK>yes?</THINK> no", False),
        ("Looks like yes. </think>\n\nNO", False),  # template emitted only the closing tag
        ("<think>a</think> <think>no</think> YES", True),
    ],
)
def test_parse_yes_no_response_answer_comes_after_the_trace(response, expected):
    assert q.parse_yes_no_response(response) is expected


def test_parse_yes_no_response_unclosed_trace_is_not_an_answer():
    with pytest.raises(LlmProviderError):
        q.parse_yes_no_response("<think>yes, it might be a cat, but")
