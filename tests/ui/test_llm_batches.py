"""LLM batch results must reach the image they were requested for."""
from __future__ import annotations

import pytest

from imagetagger.utils.sidecar import SidecarData, read_sidecar_data


def _prepare_llm(window, session, monkeypatch) -> None:
    window.llm_model_name = "fake-model"
    window.llm_threads_input.setText("1")  # one at a time: a, then b, then c
    window.llm_retry_input.setText("0")
    monkeypatch.setattr(window.llm_controller, "_active_provider_session", lambda: session)


def _run_batch_deleting_first_image_midway(qtbot, window, session, start_batch) -> None:
    """Start a batch over a, b, c; delete a while its request is in flight."""
    folder = window.records[0].image_path.parent
    window.list_widget.selectAll()
    start_batch()
    assert session.started.wait(10), "the request for a never started"
    deleted, _ = window._delete_image_and_related_files(folder / "a.png", confirm=False)
    assert deleted
    session.release()
    qtbot.waitUntil(lambda: window.llm_controller.llm_thread is None, timeout=10_000)


def test_generate_result_for_deleted_image_is_not_applied_to_another(
    qtbot, main_window, fake_session, monkeypatch, drain_writes
):
    session = fake_session(lambda path, prompt: f"gen_{path.stem}")
    session.hold.add("a")
    _prepare_llm(main_window, session, monkeypatch)
    main_window.generate_tags_checkbox.setChecked(True)
    main_window.generate_description_checkbox.setChecked(False)
    main_window.generate_vision_checkbox.setChecked(False)
    main_window.generate_refine_checkbox.setChecked(False)

    _run_batch_deleting_first_image_midway(qtbot, main_window, session, main_window.llm_controller.generate_with_llm)
    drain_writes()

    folder = main_window.records[0].image_path.parent
    assert [path.stem for path in session.calls] == ["a", "b", "c"]
    assert (folder / "b.txt").read_text(encoding="utf-8") == "tag_b, gen_b"
    assert (folder / "c.txt").read_text(encoding="utf-8") == "tag_c, gen_c"
    assert [record.text for record in main_window.records] == ["tag_b, gen_b", "tag_c, gen_c"]


def test_ai_find_result_for_deleted_image_is_not_applied_to_another(
    qtbot, main_window, fake_session, monkeypatch
):
    session = fake_session(lambda path, prompt: "YES" if path.stem == "a" else "NO")
    session.hold.add("a")
    _prepare_llm(main_window, session, monkeypatch)
    main_window.ai_find_input.setText("red square")

    _run_batch_deleting_first_image_midway(qtbot, main_window, session, main_window.llm_controller.ai_find_with_llm)

    for record in main_window.records:
        assert read_sidecar_data(record.image_path).ai_find_matches is None, record.image_path.name


def test_validate_result_for_deleted_image_is_not_applied_to_another(
    qtbot, main_window, fake_session, monkeypatch
):
    # a is clean; b and c get an answer without fixup headers, which is an error.
    session = fake_session(lambda path, prompt: "OK" if path.stem == "a" else "no headers here")
    session.hold.add("a")
    _prepare_llm(main_window, session, monkeypatch)

    _run_batch_deleting_first_image_midway(
        qtbot, main_window, session, main_window.llm_controller.validate_tags_with_llm
    )

    for record in main_window.records:
        assert read_sidecar_data(record.image_path).validated is None, record.image_path.name
    assert main_window.llm_controller.validate_pending_paths == set()


@pytest.mark.parametrize(
    ("generated", "expect_stamp_kept"),
    [
        (["new_tag"], False),  # annotations changed: the old review no longer applies
        (["tag_b"], True),  # nothing new: the stamp still describes the text
    ],
)
def test_generate_clears_validation_stamp_only_when_annotations_change(
    make_main_window, image_folder, generated, expect_stamp_kept
):
    stamp = SidecarData(validated="2026-09-01T00:00:00Z", validated_by="user")
    window = make_main_window(image_folder({"a": "tag_a", "b": "tag_b"}, sidecars={"b": stamp}))
    index = window._record_index_for_image_path(window.records[1].image_path)
    record = window.records[index]
    assert record.sidecar_validated == stamp.validated

    window._apply_generated_items_to_record(index, "", generated)

    sidecar = read_sidecar_data(record.image_path)
    if expect_stamp_kept:
        assert (sidecar.validated, sidecar.validated_by) == (stamp.validated, "user")
        assert record.sidecar_validated == stamp.validated
    else:
        assert (sidecar.validated, sidecar.validated_by) == (None, None)
        assert record.sidecar_validated is None


def test_validate_result_for_annotations_edited_meanwhile_is_dropped(
    qtbot, main_window, fake_session, monkeypatch
):
    """Path identity is not enough: the answer describes the text that was sent."""
    session = fake_session(lambda path, prompt: "OK")
    session.hold.add("a")
    _prepare_llm(main_window, session, monkeypatch)
    main_window.list_widget.selectAll()
    main_window.llm_controller.validate_tags_with_llm()
    assert session.started.wait(10), "the request for a never started"

    main_window.list_widget.setCurrentRow(0)
    main_window._set_current_tags(["tag_a", "added_meanwhile"])
    session.release()
    qtbot.waitUntil(lambda: main_window.llm_controller.llm_thread is None, timeout=10_000)

    stamps = {record.image_path.stem: read_sidecar_data(record.image_path).validated for record in main_window.records}
    assert stamps["a"] is None
    assert stamps["b"] is not None and stamps["c"] is not None
    assert main_window.llm_controller.validate_pending_paths == set()


@pytest.mark.parametrize("save", ["auto", "explicit"])
def test_manual_edit_clears_validation_stamp(make_main_window, image_folder, save):
    stamp = SidecarData(validated="2026-09-01T00:00:00Z", validated_by="user")
    window = make_main_window(image_folder({"a": "tag_a"}, sidecars={"a": stamp}))
    record = window.records[0]
    window.list_widget.setCurrentRow(0)
    assert record.sidecar_validated == stamp.validated

    if save == "auto":
        window._set_current_tags(["tag_a", "manual"])
    else:
        window._populate_tag_list(["tag_a", "manual"])
        window.save_current_text()

    assert record.text == "tag_a, manual"
    sidecar = read_sidecar_data(record.image_path)
    assert (sidecar.validated, sidecar.validated_by) == (None, None)
    assert record.sidecar_validated is None


def test_vision_result_for_deleted_image_does_not_recreate_its_sidecar(
    qtbot, main_window, fake_session, monkeypatch, drain_writes
):
    session = fake_session(lambda path, prompt: f"THOUGHT:\nlooked at {path.stem}\nDESCRIPTION:\nvision_{path.stem}")
    session.hold.add("a")
    _prepare_llm(main_window, session, monkeypatch)
    main_window.generate_tags_checkbox.setChecked(False)
    main_window.generate_description_checkbox.setChecked(False)
    main_window.generate_vision_checkbox.setChecked(True)
    main_window.generate_refine_checkbox.setChecked(False)
    folder = main_window.records[0].image_path.parent

    _run_batch_deleting_first_image_midway(qtbot, main_window, session, main_window.llm_controller.generate_with_llm)
    drain_writes()

    assert not (folder / "a.json").exists()
    for stem in ("b", "c"):
        sidecar = read_sidecar_data(folder / f"{stem}.png")
        assert (sidecar.description, sidecar.reasoning) == (f"vision_{stem}", f"looked at {stem}")
