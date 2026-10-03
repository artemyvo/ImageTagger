"""Sidecar JSON: serialization, parsing, and the read / pending-write caches."""
from __future__ import annotations

import json
import os

import pytest

from imagetagger.utils import sidecar as sidecar_module
from imagetagger.utils.sidecar import (
    SidecarData,
    _parse_sidecar_file,
    delete_sidecar_data,
    get_sidecar_json_path,
    read_sidecar_data,
    write_sidecar_data,
    write_sidecar_data_async,
)

OPTIONAL_KEYS = [
    "fixup_issues",
    "fixup_tags",
    "fixup_description",
    "fixup_model",
    "fixup_date",
    "ai_find_matches",
    "vision_tags",
    "vision_caption",
    "validated",
    "validated_by",
]


@pytest.fixture(autouse=True)
def clear_sidecar_caches():
    """The read and pending-write caches are module globals keyed by path."""
    sidecar_module._sidecar_cache.clear()
    sidecar_module._pending_sidecar.clear()
    yield
    sidecar_module._sidecar_cache.clear()
    sidecar_module._pending_sidecar.clear()


def _full() -> SidecarData:
    return SidecarData(
        description="a café",
        reasoning="because",
        fixup_issues="missing tag",
        fixup_tags=["tag one", "tag two"],
        fixup_description="better description",
        fixup_model="model-x",
        fixup_date="2026-01-02T03:04:05+00:00",
        ai_find_matches=["red car"],
        vision_tags=["car", "red"],
        vision_caption="a red car",
        validated="2026-01-02T03:04:05Z",
        validated_by="user",
    )


def _disk(image) -> dict:
    return json.loads(get_sidecar_json_path(image).read_text(encoding="utf-8"))


def _write_raw(image, payload) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    get_sidecar_json_path(image).write_text(text, encoding="utf-8")


def _bump_mtime(path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


# ---------------------------------------------------------------------------
# Serialization round trip
# ---------------------------------------------------------------------------


def test_sidecar_path_replaces_image_suffix(tmp_path):
    assert get_sidecar_json_path(tmp_path / "a.png") == tmp_path / "a.json"
    assert get_sidecar_json_path(tmp_path / "a.b.jpeg") == tmp_path / "a.b.json"


def test_every_field_round_trips(tmp_path):
    image = tmp_path / "a.png"
    write_sidecar_data(image, _full())
    sidecar_module._sidecar_cache.clear()
    assert read_sidecar_data(image) == _full()
    assert _disk(image)["description"] == "a café"  # ensure_ascii=False keeps it readable


def test_default_data_writes_only_description_and_reasoning(tmp_path):
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData())
    assert _disk(image) == {"description": "", "reasoning": ""}
    assert read_sidecar_data(image) == SidecarData()


def test_file_ends_with_newline_and_is_indented(tmp_path):
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData(description="d"))
    text = get_sidecar_json_path(image).read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert '\n  "description": "d"' in text


@pytest.mark.parametrize("key", OPTIONAL_KEYS)
def test_none_fields_are_omitted(tmp_path, key):
    image = tmp_path / "a.png"
    data = _full()
    setattr(data, key, None)
    write_sidecar_data(image, data)
    on_disk = _disk(image)
    assert key not in on_disk
    assert set(on_disk) == {"description", "reasoning", *OPTIONAL_KEYS} - {key}


@pytest.mark.parametrize(
    "key, empty",
    [
        ("fixup_issues", ""),
        ("fixup_tags", []),
        ("fixup_description", ""),
        ("ai_find_matches", []),
        ("vision_tags", []),
        ("vision_caption", ""),
        ("validated", ""),
    ],
)
def test_empty_values_are_written_but_read_back_as_none(tmp_path, key, empty):
    image = tmp_path / "a.png"
    data = SidecarData()
    setattr(data, key, empty)
    write_sidecar_data(image, data)
    assert _disk(image)[key] == empty
    sidecar_module._sidecar_cache.clear()
    assert getattr(read_sidecar_data(image), key) is None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "{not json", "[1, 2, 3]", '"a string"', "42", "null", "\x00\x01"],
    ids=["empty", "malformed", "list", "string", "number", "null", "binary"],
)
def test_unparseable_or_non_dict_file_reads_as_empty(tmp_path, text):
    path = tmp_path / "a.json"
    path.write_text(text, encoding="utf-8")
    assert _parse_sidecar_file(path) == SidecarData()


def test_invalid_utf8_reads_as_empty(tmp_path):
    path = tmp_path / "a.json"
    path.write_bytes(b'{"description": "\xff\xfe"}')
    assert _parse_sidecar_file(path) == SidecarData()


def test_missing_file_parses_as_empty(tmp_path):
    assert _parse_sidecar_file(tmp_path / "missing.json") == SidecarData()


def test_wrong_types_are_coerced_or_dropped(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "description": 123,
                "reasoning": None,
                "fixup_issues": 7,
                "fixup_tags": "a, b",  # not a list
                "fixup_description": "   ",  # whitespace only
                "fixup_model": 0,  # falsy
                "fixup_date": {"x": 1},
                "ai_find_matches": ["q", "", None, 0, 5],
                "vision_tags": [],
                "vision_caption": False,
                "validated": "  2026-01-01T00:00:00Z  ",
                "unknown_key": "ignored",
            }
        ),
        encoding="utf-8",
    )
    data = _parse_sidecar_file(path)
    assert data.description == "123"
    assert data.reasoning == ""
    assert data.fixup_issues == "7"
    assert data.fixup_tags is None
    assert data.fixup_description is None
    assert data.fixup_model is None
    assert data.fixup_date == "{'x': 1}"
    assert data.ai_find_matches == ["q", "5"]
    assert data.vision_tags is None
    assert data.vision_caption is None
    assert data.validated == "2026-01-01T00:00:00Z"
    assert data.validated_by is None


def test_description_is_not_stripped(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"description": "  spaced  "}), encoding="utf-8")
    assert _parse_sidecar_file(path).description == "  spaced  "


# ---------------------------------------------------------------------------
# has_pending_fixup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("fixup_issues", "x"),
        ("fixup_tags", ["x"]),
        ("fixup_description", "x"),
        ("ai_find_matches", ["x"]),
        ("vision_tags", ["x"]),
        ("vision_caption", "x"),
    ],
)
def test_has_pending_fixup_for_each_review_field(field, value):
    assert SidecarData(**{field: value}).has_pending_fixup


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"description": "d", "reasoning": "r"},
        {"fixup_model": "m", "fixup_date": "2026-01-01"},
        {"validated": "2026-01-01T00:00:00Z", "validated_by": "user"},
        {"fixup_issues": "", "fixup_tags": [], "fixup_description": "", "ai_find_matches": [],
         "vision_tags": [], "vision_caption": ""},
    ],
    ids=["default", "committed-text", "model-and-date", "validated", "empty-values"],
)
def test_no_pending_fixup(kwargs):
    assert not SidecarData(**kwargs).has_pending_fixup


# ---------------------------------------------------------------------------
# Read cache
# ---------------------------------------------------------------------------


def test_missing_sidecar_reads_empty_and_is_negatively_cached(tmp_path):
    image = tmp_path / "a.png"
    assert read_sidecar_data(image) == SidecarData()
    assert sidecar_module._sidecar_cache[image][0] is None


def test_write_through_this_module_invalidates_the_negative_cache(tmp_path):
    image = tmp_path / "a.png"
    read_sidecar_data(image)
    write_sidecar_data(image, SidecarData(description="ours"))
    assert read_sidecar_data(image).description == "ours"


def test_forgetting_missing_sidecars_finds_externally_created_ones(tmp_path):
    """Folder load/refresh calls this; a sidecar made by another program then shows up."""
    image = tmp_path / "a.png"
    other = tmp_path / "b.png"
    read_sidecar_data(image)
    write_sidecar_data(other, SidecarData(description="kept"))
    read_sidecar_data(other)
    _write_raw(image, {"description": "external"})

    sidecar_module.forget_missing_sidecars()

    assert read_sidecar_data(image).description == "external"
    assert sidecar_module._sidecar_cache[other][0] is not None


def test_unchanged_file_is_served_from_cache(tmp_path, monkeypatch):
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData(description="d"))
    first = read_sidecar_data(image)

    def fail(path):
        raise AssertionError("re-parsed an unchanged file")

    monkeypatch.setattr(sidecar_module, "_parse_sidecar_file", fail)
    assert read_sidecar_data(image) == first


def test_changing_a_read_result_does_not_change_the_cache(tmp_path):
    """Callers change what they read and then write it; a failed write must not leave the change behind."""
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData(description="d", fixup_tags=["x"]))
    read = read_sidecar_data(image)
    read.description = "changed"
    read.fixup_tags.append("y")
    assert read_sidecar_data(image) == SidecarData(description="d", fixup_tags=["x"])


def test_file_changed_on_disk_is_reread(tmp_path):
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData(description="before"))
    assert read_sidecar_data(image).description == "before"

    path = get_sidecar_json_path(image)
    _write_raw(image, {"description": "after"})
    _bump_mtime(path)
    assert read_sidecar_data(image).description == "after"


def test_sidecar_deleted_externally_reads_empty(tmp_path):
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData(description="d"))
    read_sidecar_data(image)
    get_sidecar_json_path(image).unlink()
    assert read_sidecar_data(image) == SidecarData()


def test_sync_write_invalidates_cache(tmp_path):
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData(description="one"))
    read_sidecar_data(image)
    write_sidecar_data(image, SidecarData(description="two"))
    assert image not in sidecar_module._sidecar_cache
    assert read_sidecar_data(image).description == "two"


# ---------------------------------------------------------------------------
# Async writes
# ---------------------------------------------------------------------------


def test_async_write_is_visible_before_it_lands(tmp_path, stalled_write_queue, drain_writes):
    image = tmp_path / "a.png"
    data = SidecarData(description="pending", fixup_tags=["t"])
    write_sidecar_data_async(image, data)

    assert not get_sidecar_json_path(image).exists()
    assert read_sidecar_data(image) == data

    # Changing the object after handing it over changes neither the pending read nor the write.
    data.fixup_tags.append("late")
    assert read_sidecar_data(image).fixup_tags == ["t"]

    stalled_write_queue.set()
    drain_writes()
    assert image not in sidecar_module._pending_sidecar
    assert _disk(image) == {"description": "pending", "reasoning": "", "fixup_tags": ["t"]}
    assert read_sidecar_data(image) == SidecarData(description="pending", fixup_tags=["t"])


def test_async_write_replaces_older_disk_content_in_reads(tmp_path, stalled_write_queue, drain_writes):
    image = tmp_path / "a.png"
    write_sidecar_data_async(image, SidecarData(description="old"))
    newer = SidecarData(description="new")
    write_sidecar_data_async(image, newer)
    assert read_sidecar_data(image).description == "new"

    stalled_write_queue.set()
    drain_writes()
    assert _disk(image)["description"] == "new"
    assert read_sidecar_data(image).description == "new"


def test_async_write_after_negative_cache(tmp_path, drain_writes):
    image = tmp_path / "a.png"
    assert read_sidecar_data(image) == SidecarData()
    write_sidecar_data_async(image, SidecarData(description="d"))
    assert read_sidecar_data(image).description == "d"
    drain_writes()
    sidecar_module._sidecar_cache.clear()
    assert read_sidecar_data(image).description == "d"


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_removes_file_and_cache(tmp_path):
    image = tmp_path / "a.png"
    write_sidecar_data(image, SidecarData(description="d"))
    read_sidecar_data(image)
    delete_sidecar_data(image)
    assert not get_sidecar_json_path(image).exists()
    assert image not in sidecar_module._sidecar_cache
    assert read_sidecar_data(image) == SidecarData()


def test_delete_waits_for_a_queued_async_write(tmp_path, stalled_write_queue):
    """The queued write must not recreate the file after the delete."""
    import threading

    image = tmp_path / "a.png"
    write_sidecar_data_async(image, SidecarData(description="queued"))
    deleter = threading.Thread(target=delete_sidecar_data, args=(image,))
    deleter.start()
    deleter.join(0.2)
    assert deleter.is_alive(), "delete must wait behind the queued write"
    stalled_write_queue.set()
    deleter.join(10)

    assert not get_sidecar_json_path(image).exists()
    assert image not in sidecar_module._pending_sidecar
    assert read_sidecar_data(image) == SidecarData()


def test_delete_missing_sidecar_is_a_no_op(tmp_path):
    image = tmp_path / "a.png"
    delete_sidecar_data(image)
    assert read_sidecar_data(image) == SidecarData()
