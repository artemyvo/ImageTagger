"""Sidecar helpers in merge_actions: fixups, AI Find matches, refine results, validation stamps."""
from __future__ import annotations

import json
import re
import threading

import pytest

from imagetagger.ui import merge_actions
from imagetagger.utils import io_utils
from imagetagger.utils import sidecar as sidecar_module
from imagetagger.utils.sidecar import (
    SidecarData,
    get_sidecar_json_path,
    read_sidecar_data,
    write_sidecar_data,
)

_UTC_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@pytest.fixture(autouse=True)
def clear_sidecar_caches():
    sidecar_module._sidecar_cache.clear()
    sidecar_module._pending_sidecar.clear()
    yield
    sidecar_module._sidecar_cache.clear()
    sidecar_module._pending_sidecar.clear()


@pytest.fixture
def image(tmp_path):
    return tmp_path / "a.png"


def _disk(image) -> dict:
    return json.loads(get_sidecar_json_path(image).read_text(encoding="utf-8"))


def _fresh(image) -> SidecarData:
    """What is on disk, bypassing every cache."""
    return sidecar_module._parse_sidecar_file(get_sidecar_json_path(image))


def _reviewed() -> SidecarData:
    return SidecarData(
        description="desc",
        reasoning="why",
        fixup_issues="issue",
        fixup_tags=["t1"],
        fixup_description="better",
        fixup_model="m",
        fixup_date="2026-01-01",
        ai_find_matches=["red car"],
        vision_tags=["car"],
        vision_caption="a car",
    )


# ---------------------------------------------------------------------------
# write_fixup_sidecar
# ---------------------------------------------------------------------------


def test_write_fixup_sets_fields_and_keeps_the_rest(image):
    write_sidecar_data(image, SidecarData(description="desc", reasoning="why", ai_find_matches=["q"],
                                          validated="2026-01-01T00:00:00Z", validated_by="user"))
    merge_actions.write_fixup_sidecar(image, "issue", ["t1", "t2"], "better", model="m", date="d")

    data = _fresh(image)
    assert (data.fixup_issues, data.fixup_tags, data.fixup_description) == ("issue", ["t1", "t2"], "better")
    assert (data.fixup_model, data.fixup_date) == ("m", "d")
    assert (data.description, data.reasoning, data.ai_find_matches) == ("desc", "why", ["q"])
    assert (data.validated, data.validated_by) == ("2026-01-01T00:00:00Z", "user")
    assert data.has_pending_fixup


def test_write_fixup_empty_values_are_stored_as_absent(image):
    write_sidecar_data(image, _reviewed())
    merge_actions.write_fixup_sidecar(image, "", [], "", model="", date="")
    on_disk = _disk(image)
    for key in ("fixup_issues", "fixup_tags", "fixup_description", "fixup_model", "fixup_date"):
        assert key not in on_disk
    assert on_disk["ai_find_matches"] == ["red car"]


def test_write_fixup_creates_the_sidecar(image):
    merge_actions.write_fixup_sidecar(image, "issue", None, None)
    assert _disk(image) == {"description": "", "reasoning": "", "fixup_issues": "issue"}


def test_failed_fixup_write_does_not_change_what_reads_return(image, monkeypatch):
    write_sidecar_data(image, SidecarData(description="desc"))
    assert read_sidecar_data(image).fixup_issues is None  # populates the mtime cache

    def failing_write(path, content, encoding="utf-8"):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(io_utils, "atomic_write_text", failing_write)
    with pytest.raises(OSError):
        merge_actions.write_fixup_sidecar(image, "issue", ["t"], None)

    assert _fresh(image).fixup_issues is None
    assert read_sidecar_data(image).fixup_issues is None


# ---------------------------------------------------------------------------
# record_ai_find_match_for_image
# ---------------------------------------------------------------------------


def test_ai_find_match_is_stripped_and_lowercased(image):
    merge_actions.record_ai_find_match_for_image(image, "  Red Car  ")
    assert _fresh(image).ai_find_matches == ["red car"]


def test_ai_find_matches_accumulate_and_dedupe(image):
    for query in ["red car", "Blue Sky", " RED CAR ", "blue sky"]:
        merge_actions.record_ai_find_match_for_image(image, query)
    assert _fresh(image).ai_find_matches == ["red car", "blue sky"]


def test_ai_find_duplicate_does_not_rewrite(image, monkeypatch):
    merge_actions.record_ai_find_match_for_image(image, "red car")
    calls = []
    monkeypatch.setattr(merge_actions, "write_sidecar_data", lambda *args: calls.append(args))
    merge_actions.record_ai_find_match_for_image(image, "Red Car")
    assert calls == []


def test_ai_find_uses_the_normalizer_before_lowercasing(image):
    seen = []

    def normalize(text):
        seen.append(text)
        return "  " + text.replace(",", " ").replace("  ", " ") + "  "

    merge_actions.record_ai_find_match_for_image(image, " Red,Car ", normalize_annotation=normalize)
    assert seen == ["Red,Car"]
    assert _fresh(image).ai_find_matches == ["red car"]


def test_ai_find_keeps_other_fields(image):
    write_sidecar_data(image, SidecarData(description="desc", fixup_tags=["t"], vision_caption="c"))
    merge_actions.record_ai_find_match_for_image(image, "q")
    data = _fresh(image)
    assert (data.description, data.fixup_tags, data.vision_caption) == ("desc", ["t"], "c")


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_ai_find_empty_query_raises_and_writes_nothing(image, query):
    with pytest.raises(OSError, match="cannot be empty"):
        merge_actions.record_ai_find_match_for_image(image, query)
    assert not get_sidecar_json_path(image).exists()


def test_ai_find_query_emptied_by_normalizer_raises(image):
    with pytest.raises(OSError):
        merge_actions.record_ai_find_match_for_image(image, "...", normalize_annotation=lambda t: "  ")


# ---------------------------------------------------------------------------
# record_refine_result_for_image
# ---------------------------------------------------------------------------


def test_refine_result_is_sanitized(image):
    merge_actions.record_refine_result_for_image(image, ["Red Car", "  ", "(blue)", ""], "A red car, parked.")
    data = _fresh(image)
    assert data.vision_tags == ["red car", "blue"]
    assert data.vision_caption and "," not in data.vision_caption
    assert data.has_pending_fixup


def test_refine_result_empty_clears_vision_fields(image):
    write_sidecar_data(image, SidecarData(description="desc", vision_tags=["old"], vision_caption="old"))
    merge_actions.record_refine_result_for_image(image, ["", " "], "  ")
    on_disk = _disk(image)
    assert "vision_tags" not in on_disk
    assert "vision_caption" not in on_disk
    assert on_disk["description"] == "desc"


# ---------------------------------------------------------------------------
# clear_fixup_sidecar
# ---------------------------------------------------------------------------


def test_clear_fixup_drops_review_fields_and_stamps_user(image, drain_writes):
    write_sidecar_data(image, _reviewed())
    merge_actions.clear_fixup_sidecar(image)

    # Visible immediately, before the async write lands.
    assert not read_sidecar_data(image).has_pending_fixup
    drain_writes()

    on_disk = _disk(image)
    for key in ("fixup_issues", "fixup_tags", "fixup_description", "ai_find_matches", "vision_tags", "vision_caption"):
        assert key not in on_disk
    assert on_disk["description"] == "desc"
    assert on_disk["reasoning"] == "why"
    assert on_disk["fixup_model"] == "m"  # who produced the fixup is kept
    assert on_disk["fixup_date"] == "2026-01-01"
    assert on_disk["validated_by"] == "user"
    assert _UTC_STAMP.match(on_disk["validated"])


def test_clear_fixup_while_queue_is_stalled(image, drain_writes):
    write_sidecar_data(image, _reviewed())
    gate = threading.Event()
    io_utils._enqueue_write_job(lambda: gate.wait(30))
    try:
        merge_actions.clear_fixup_sidecar(image)
        assert _disk(image)["fixup_tags"] == ["t1"]  # not landed yet
        assert read_sidecar_data(image).validated_by == "user"
    finally:
        gate.set()
    drain_writes()
    assert _disk(image)["validated_by"] == "user"
    assert "fixup_tags" not in _disk(image)


# ---------------------------------------------------------------------------
# clear_validation_fields_sidecar
# ---------------------------------------------------------------------------


def test_clear_validation_fields_keeps_ai_find_and_vision_data(image):
    write_sidecar_data(image, _reviewed())
    merge_actions.clear_validation_fields_sidecar(image, model="model-y", date="2026-02-02")
    data = _fresh(image)
    assert (data.fixup_issues, data.fixup_tags, data.fixup_description) == (None, None, None)
    assert (data.fixup_model, data.fixup_date) == ("model-y", "2026-02-02")
    assert data.ai_find_matches == ["red car"]
    assert (data.vision_tags, data.vision_caption) == (["car"], "a car")
    assert data.validated_by == "model-y"
    assert _UTC_STAMP.match(data.validated)
    assert data.has_pending_fixup  # AI Find / vision review still pending


def test_clear_validation_fields_without_model(image):
    write_sidecar_data(image, SidecarData(fixup_issues="x", fixup_model="old", fixup_date="old"))
    merge_actions.clear_validation_fields_sidecar(image)
    on_disk = _disk(image)
    for key in ("fixup_issues", "fixup_model", "fixup_date", "validated_by"):
        assert key not in on_disk
    assert _UTC_STAMP.match(on_disk["validated"])
    assert not _fresh(image).has_pending_fixup


# ---------------------------------------------------------------------------
# clear_validation_stamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validated, validated_by",
    [("2026-01-01T00:00:00Z", "user"), ("2026-01-01T00:00:00Z", None), (None, "model")],
)
def test_clear_validation_stamp_drops_it(image, validated, validated_by):
    write_sidecar_data(image, SidecarData(description="d", fixup_tags=["t"],
                                          validated=validated, validated_by=validated_by))
    assert merge_actions.clear_validation_stamp(image) is True
    on_disk = _disk(image)
    assert "validated" not in on_disk and "validated_by" not in on_disk
    assert on_disk["fixup_tags"] == ["t"]


def test_clear_validation_stamp_without_stamp_returns_false_and_does_not_write(image, monkeypatch):
    write_sidecar_data(image, SidecarData(description="d"))
    calls = []
    monkeypatch.setattr(merge_actions, "write_sidecar_data", lambda *args: calls.append(args))
    assert merge_actions.clear_validation_stamp(image) is False
    assert calls == []


def test_clear_validation_stamp_without_sidecar_creates_nothing(image):
    assert merge_actions.clear_validation_stamp(image) is False
    assert not get_sidecar_json_path(image).exists()


# ---------------------------------------------------------------------------
# delete_sidecar_for_image
# ---------------------------------------------------------------------------


def test_delete_sidecar_for_image(image):
    write_sidecar_data(image, _reviewed())
    read_sidecar_data(image)
    merge_actions.delete_sidecar_for_image(image)
    assert not get_sidecar_json_path(image).exists()
    assert read_sidecar_data(image) == SidecarData()


def test_delete_sidecar_for_image_swallows_os_errors(image, monkeypatch):
    def failing_delete(path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(merge_actions, "delete_sidecar_data", failing_delete)
    merge_actions.delete_sidecar_for_image(image)  # does not raise
