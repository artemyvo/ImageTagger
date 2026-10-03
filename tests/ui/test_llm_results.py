"""How single LLM answers and prompt-editor actions land in the app."""
from __future__ import annotations

import pytest

from imagetagger.ui import llm_controller as llm_controller_module
from imagetagger.utils.llm_queries import LlmQueryError
from imagetagger.utils.sidecar import read_sidecar_data


@pytest.mark.parametrize(
    "answer, outcome",
    [
        ("OK", "clean"),
        ("ISSUES:\nTAGS:\nDESCRIPTION:\n", "clean"),  # headers with nothing to fix
        ("**ISSUES:** Wrong animal.\n**TAGS:** dog", "issues"),
    ],
)
def test_validation_answer_outcome(main_window, answer, outcome):
    record = main_window.records[0]
    result, _ = main_window._apply_validation_result_to_record(0, answer)
    assert result == outcome
    sidecar = read_sidecar_data(record.image_path)
    if outcome == "clean":
        assert sidecar.validated is not None and not sidecar.has_pending_fixup
    else:
        assert (sidecar.fixup_issues, sidecar.fixup_tags) == ("Wrong animal.", ["dog"])


@pytest.mark.parametrize(
    "action, patched, title",
    [
        ("_save_prompt_to_file", "save_prompt_for_kind", "Save prompt failed"),
        ("_reset_prompt_to_default", "reset_prompt_to_default", "Reset prompt failed"),
        ("_apply_prompt_override", "set_prompt_override", "Apply prompt failed"),
    ],
)
def test_prompt_editor_failure_shows_a_dialog(main_window, monkeypatch, message_boxes, action, patched, title):
    def fail(*args, **kwargs):
        raise LlmQueryError("Could not save prompt file: read-only file system")

    monkeypatch.setattr(llm_controller_module, patched, fail)
    getattr(main_window.llm_controller, action)("tagging")
    assert message_boxes.titles("critical") == [title]
