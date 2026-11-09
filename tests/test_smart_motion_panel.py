"""Native panel contracts that do not require importing Cascadeur or Qt."""

import builtins
import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import smart_motion


def test_inline_qml_object_bindings_are_not_terminated_as_javascript_statements():
    # A validator is a declarative object binding. Qt 6.5 rejects the extra
    # semicolon even though the surrounding property statements accept it.
    assert not re.search(r"validator:\s*\w+Validator\s*\{[^\n]*\}\s*;", smart_motion.QML_SOURCE)


def test_panel_module_can_load_without_optional_native_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"PySide6", "csc", "numpy"}:
            raise AssertionError("Native dependencies were imported by MCP discovery")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    loaded = importlib.reload(smart_motion)
    assert loaded.panel_state() == {"visible": False, "title": "Smart Motion", "scene_changes": False}


@pytest.mark.parametrize("text", ["0", "2,1", "0,0,5", "-1,12", "0.5,12", "", "0,nan"])
def test_anchor_input_rejects_ambiguous_or_invalid_frames(text):
    with pytest.raises(ValueError):
        smart_motion.parse_keyframes(text)


def test_anchor_input_preserves_absolute_frame_indices_and_accepts_pasted_lists():
    assert smart_motion.parse_keyframes(" 30, 60; 90\n149 ") == [30, 60, 90, 149]


@pytest.mark.parametrize("fraction", [None, True, float("nan"), float("inf"), -0.5, 1.5, "0.8"])
def test_progress_never_invents_a_percentage(fraction):
    result = smart_motion.progress_display({"state": "running", "stage": "rendering", "progress": {
        "fraction": fraction, "message": "waiting for frame"}})
    assert result["fraction"] is None
    assert result["stage"] == "rendering"
    assert result["message"] == "waiting for frame"


def test_progress_uses_measured_fraction_when_available():
    assert smart_motion.progress_display({"progress": {"fraction": 0.3}})["fraction"] == 0.3


def test_long_preview_retains_every_frame_and_has_time_for_a_slow_viewport():
    result = smart_motion._preview_request({"frames": 540}, 541)
    assert result == {"first": 0, "last": 539, "timeout": 190}
    assert result["timeout"] > 540 / 6
    assert smart_motion._preview_request({"frames": 2}, 3)["timeout"] == 30


def test_preview_without_run_metadata_uses_all_available_frames_and_checks_actual_storage():
    assert smart_motion._preview_request({}, 600) == {"first": 0, "last": 599, "timeout": 210}
    assert smart_motion._preview_request({"frames": 600}, 450)["last"] == 449
    with pytest.raises(ValueError):
        smart_motion._preview_request({"frames": 540}, 1)


def test_history_labels_show_readable_state_and_retained_timestamp_instead_of_internal_ids(monkeypatch):
    monkeypatch.setattr(smart_motion.time, "strftime", lambda pattern, value: "09.11 20:43")
    result = smart_motion._job_label({"prompt": "Тяжёлая походка", "state": "completed", "created_at": 123456,
                                      "job_id": "private-internal-id"})
    assert result == "Тяжёлая походка · Готово · 09.11 20:43"
    assert smart_motion._job_label({"action": "runtime_start", "state": "failed"}) == "Загрузка Kimodo · Ошибка"
    assert smart_motion._job_label({"prompt": "walk", "state": "cancel_requested"}) == "walk · Ожидание отмены"


def test_completed_job_replaces_acceptance_notice_once_without_overwriting_new_user_errors():
    job = {"state": "completed", "action": "run"}
    assert smart_motion._notice_after_update("running", job, task_notice=True) == ("Анимация готова и сохранена.", False)
    assert smart_motion._notice_after_update("completed", job, task_notice=True) is None
    assert smart_motion._notice_after_update("running", job, task_notice=False) is None
    assert smart_motion._notice_after_update("queued", {"state": "running"}, task_notice=True) is None


def test_terminal_notices_describe_actual_result_including_already_finished_submissions():
    assert smart_motion._completion_notice({"state": "completed", "action": "select"}) == ("Анимация открыта.", False)
    assert smart_motion._completion_notice({"state": "completed", "action": "runtime_start"}) == ("Kimodo запущен.", False)
    assert smart_motion._completion_notice({"state": "completed", "run": {"mode": "generated_only"}}) == ("Файлы анимации готовы и сохранены.", False)
    text, error = smart_motion._notice_after_update("running", {"state": "failed", "error": "Native save failed"}, task_notice=True)
    assert error is True and "Native save failed" in text
    text, error = smart_motion._notice_after_update("cancelling", {"state": "cancelled"}, task_notice=True)
    assert text.startswith("Задача отменена.") and error is False
    assert smart_motion._completion_notice({"state": "interrupted"})[1] is True


def test_raw_mode_settings_omit_blank_native_paths_without_erasing_existing_configuration():
    value = {"checkout": "C:/prepared", "calibration": "", "template_motion": "  ", "template_scene": "",
             "defaults": {"generate_only": True}, "review": {"provider": "gemini"}}
    assert smart_motion._settings_payload(value) == {
        "checkout": "C:/prepared", "defaults": {"generate_only": True}}
    assert smart_motion._settings_payload({"checkout": ""}) == {}
    assert value["calibration"] == "" and value["template_motion"] == "  "


@pytest.mark.parametrize("state, expected", [("running", True), ("failed", True), ("interrupted", True),
                                            ("cancelled", True), ("completed", False), ("queued", False),
                                            ("unknown", False)])
def test_resume_button_matches_pipeline_record_states_and_excludes_raw_runs(state, expected):
    run = {"format": "kimodo-cascadeur-run-v1", "mode": "text", "state": state}
    assert smart_motion._can_resume({"run": run}) is expected
    assert smart_motion._can_resume({"run": dict(run, mode="generated_only")}) is False
    assert smart_motion._can_resume({"run": dict(run, format="unrecognised-run")}) is False
    assert smart_motion._can_resume({}) is False


def test_readable_stage_labels_preserve_raw_job_progress_and_hide_boilerplate():
    job = {"state": "completed", "stage": "completed", "progress": {"message": "Stage finished"}}
    assert smart_motion._stage_label(job) == "Готово"
    assert smart_motion.progress_display(job)["message"] == "Stage finished"
    assert smart_motion._stage_caption("rendering_candidate-03") == "Съёмка анимации 03"
    assert smart_motion._stage_caption("starting_runtime") == "Загрузка Kimodo"
    assert smart_motion._stage_caption("new_backend_stage") == "new_backend_stage"


def test_shutdown_queues_each_qml_deletion_only_after_the_previous_object_is_destroyed():
    queued, completed = [], []

    class DeferredObject:
        def __init__(self, name):
            self.name, self.listeners = name, []
            self.destroyed = SimpleNamespace(connect=self.listeners.append)

        def deleteLater(self):
            queued.append(self.name)

        def destroyed_event(self):
            for callback in self.listeners:
                callback()

    window, component, engine = [DeferredObject(name) for name in ("window", "component", "engine")]
    cleanup = smart_motion._DeferredQmlDisposal(
        (window, component, engine), valid=lambda item: True, finished=lambda: completed.append("done"))
    cleanup.start()
    cleanup.start()
    assert queued == ["window"] and not completed
    window.destroyed_event()
    window.destroyed_event()
    assert queued == ["window", "component"] and not completed
    component.destroyed_event()
    assert queued == ["window", "component", "engine"] and not completed
    engine.destroyed_event()
    cleanup.start()
    assert cleanup.done and completed == ["done"]


def test_deferred_shutdown_does_not_access_previously_destroyed_qt_wrappers():
    completed = []
    cleanup = smart_motion._DeferredQmlDisposal(
        (None, object(), object()), valid=lambda item: False, finished=lambda: completed.append(True))
    cleanup.start()
    cleanup.start()
    assert completed == [True]


@pytest.mark.parametrize("has_window", [False, True])
def test_failed_qml_constructor_retains_parent_until_children_and_parent_are_naturally_destroyed(monkeypatch, has_window):
    queued = []

    class DeferredObject:
        def __init__(self, name):
            self.name, self.listeners = name, []
            self.destroyed = SimpleNamespace(connect=self.listeners.append)

        def deleteLater(self):
            queued.append(self.name)

        def destroyed_event(self):
            for callback in self.listeners:
                callback()

    monkeypatch.setattr(smart_motion, "_failed_panels", [])
    panel = DeferredObject("parent")
    window = DeferredObject("window") if has_window else None
    component, engine = DeferredObject("component"), DeferredObject("engine")
    panel.window, panel.component, panel.engine = window, component, engine
    smart_motion._dispose_failed_panel(panel, valid=lambda item: True,
                                       queue_delete=lambda item: queued.append("queued-parent-delete"))
    assert smart_motion._failed_panels == [panel]
    if window:
        assert queued == ["window"]
        window.destroyed_event()
    assert queued[-1] == "component" and "queued-parent-delete" not in queued
    component.destroyed_event()
    assert queued[-1] == "engine" and smart_motion._failed_panels == [panel]
    engine.destroyed_event()
    assert queued[-1] == "queued-parent-delete" and smart_motion._failed_panels == [panel]
    assert panel.window is panel.component is panel.engine is None
    panel.destroyed_event()
    assert smart_motion._failed_panels == []


def test_results_do_not_read_visual_model_ratings_and_keep_saved_outputs():
    candidates = [{"id": "candidate-01", "scene": "one.casc"}, {"id": "candidate-02"}]
    result = smart_motion.candidate_rows({"candidates": candidates, "review": {"review": {"candidates": [
        {"id": "candidate-01", "confidence": 0.6, "observations": ["Left foot slides"]}]}}, "selection": {
        "selected_id": "candidate-01", "ranking": [{"id": "candidate-01", "score": 78}]}})
    assert result[0]["selected"] is True
    assert result[0]["scene"] == "one.casc"
    assert "visual_review" not in result[0] and "ranking" not in result[0]
    assert result[1]["selected"] is False
    assert candidates == [{"id": "candidate-01", "scene": "one.casc"}, {"id": "candidate-02"}]


def test_panel_uses_the_full_selected_interval_and_forces_one_result_without_visual_calls():
    value = {"prompt": "walk", "character_id": "native-rig", "first_frame": 90, "last_frame": 449,
             "samples": 4, "sample_index": 3, "no_review": False, "no_render": False,
             "provider": "gemini", "model": "anything", "use_keyposes": True, "keyframes": "150"}
    result = smart_motion._panel_request(value, "snapshot.casc")
    assert result["frames"] == 360 and result["first_frame"] == 90
    assert result["samples"] == 1 and result["sample_index"] == 0
    assert result["no_review"] is result["no_render"] is True
    assert result["no_activate"] is False
    assert result["keyframes"] == [90, 150, 449]
    assert result["target_scene"] == result["source_scene"] == "snapshot.casc"
    assert "provider" not in result and "model" not in result
    assert value["samples"] == 4 and value["keyframes"] == "150"


def test_text_only_generation_targets_the_current_character_without_adding_pose_constraints():
    result = smart_motion._panel_request({"first_frame": 0, "last_frame": 4999,
                                         "prompt": "walk", "character_id": "uefn-id", "use_keyposes": False}, "copy.casc")
    assert result["frames"] == 5000 and result["target_scene"] == "copy.casc"
    assert "source_scene" not in result and "keyframes" not in result


def test_blank_seed_selects_a_new_uint32_and_explicit_seed_stays_reproducible(monkeypatch):
    monkeypatch.setattr(smart_motion.uuid, "uuid4", lambda: SimpleNamespace(int=0x123456789))
    options = {"first_frame": 0, "last_frame": 200, "character_id": "rig", "prompt": "walk"}
    assert smart_motion._panel_request(options, "copy.casc")["seed"] == 0x23456789
    assert smart_motion._panel_request(dict(options, seed=123), "copy.casc")["seed"] == 123
    with pytest.raises(ValueError, match="Seed"):
        smart_motion._panel_request(dict(options, seed=9999999999), "copy.casc")


@pytest.mark.parametrize("first,last", [(0, 0), (5, 4), (-1, 30), (True, 30), (0, "30")])
def test_panel_rejects_invalid_ranges_before_saving_source(first, last):
    with pytest.raises(ValueError):
        smart_motion._panel_request({"first_frame": first, "last_frame": last, "character_id": "rig"}, "copy.casc")


def test_panel_rejects_out_of_range_pose_and_missing_target_before_saving():
    with pytest.raises(ValueError, match="внутри"):
        smart_motion._panel_request({"first_frame": 20, "last_frame": 199, "character_id": "rig",
                                      "use_keyposes": True, "keyframes": "10, 50"}, "copy.casc")
    with pytest.raises(ValueError, match="персонажа"):
        smart_motion._panel_request({"first_frame": 0, "last_frame": 200, "prompt": "walk"}, "copy.casc")


def test_raw_generation_requires_no_scene_or_character():
    result = smart_motion._panel_request({"first_frame": 50, "last_frame": 2050, "generate_only": True,
                                         "prompt": "walk", "character_id": "stale-rig", "rig_mapping": {"pelvis": "joint"}})
    assert result["frames"] == 2001
    assert "target_scene" not in result and "character_id" not in result and "rig_mapping" not in result


def test_pose_generation_accepts_no_text_and_preserves_an_explicit_description():
    options = {"first_frame": 30, "last_frame": 299, "character_id": "rig", "use_keyposes": True,
               "keyframes": "90, 210", "prompt": "  "}
    result = smart_motion._panel_request(options, "snapshot.casc")
    assert result["prompt"] == "A person moves naturally between the specified key poses."
    assert result["keyframes"] == [30, 90, 210, 299]
    assert smart_motion._panel_request(dict(options, prompt="  with a heavy backpack  "), "snapshot.casc")["prompt"] == "with a heavy backpack"


@pytest.mark.parametrize("raw, anchors", [(False, False), (True, False), (True, True)])
def test_text_and_raw_modes_reject_blank_descriptions_before_saving(raw, anchors):
    with pytest.raises(ValueError, match="Опишите"):
        smart_motion._panel_request({"first_frame": 0, "last_frame": 299, "prompt": " ",
                                      "character_id": "rig", "generate_only": raw,
                                      "use_keyposes": anchors}, "snapshot.casc")


def test_only_complete_rig_preferences_persist_and_are_keyed_by_native_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    rig = {"fingerprint": "a" * 64, "character_id": "native-rig", "ready": False,
           "roles": [{"id": "pelvis"}], "mapping": {"pelvis": "native-joint"}}
    assert smart_motion._save_rig_profile(rig) is False
    assert not list(tmp_path.iterdir())
    rig["ready"] = True
    assert smart_motion._save_rig_profile(rig) is True
    assert smart_motion._load_rig_profile(rig) == {"pelvis": "native-joint"}
    assert smart_motion._load_rig_profile(dict(rig, fingerprint="b" * 64)) == {}
    assert len(list(tmp_path.iterdir())) == 1
    value = json.loads((tmp_path / "smart-motion-rigs.json").read_text(encoding="utf-8"))
    assert value["profiles"]["a" * 64]["character_id"] == "native-rig"


@pytest.mark.parametrize("content", ['{"profiles":[]}', '{"format":"wrong","profiles":{}}', "x", "x" * 65537])
def test_invalid_or_oversized_rig_preferences_do_not_block_scene_discovery(tmp_path, monkeypatch, content):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    (tmp_path / "smart-motion-rigs.json").write_text(content, encoding="utf-8")
    assert smart_motion._load_rig_profile({"fingerprint": "a" * 64, "roles": [{"id": "pelvis"}]}) == {}


def test_native_context_uses_selected_rig_membership_and_selected_timeline_interval(tmp_path, monkeypatch):
    from mcp_for_cascadeur.host import characters, common, humanoid

    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    rigs = {"first": SimpleNamespace(owned={"first-point"}, tracks={"first-track"}),
            "second": SimpleNamespace(owned={"second-point"}, tracks={"second-track"})}
    native_selection = SimpleNamespace(item_ids=lambda: ["second-track"],
                                      frames_interval=lambda: SimpleNamespace(first=30, last=389))
    tracks = {"first-track": [0, 10], "second-track": [0, 40, 250, 500]}
    scene = SimpleNamespace(selector=lambda: SimpleNamespace(selected=lambda: SimpleNamespace(ids=["second-point"])),
                            behaviour_viewer=lambda: SimpleNamespace(get_behaviours=lambda name: []),
                            model_viewer=lambda: SimpleNamespace(),
                            get_layers_selector=lambda: SimpleNamespace(selection=lambda: native_selection),
                            layers_viewer=lambda: SimpleNamespace(layer=lambda key: SimpleNamespace(key_frame_indices=lambda: tracks[key])),
                            get_current_frame=lambda: 40)
    view = SimpleNamespace(animation_boundary=lambda: SimpleNamespace(first_frame=0, last_frame=99),
                           name=lambda: "Custom character", get_path_name=lambda: "")
    monkeypatch.setattr(common, "current_context", lambda: (None, None, view, scene))
    monkeypatch.setattr(characters, "characters_list", lambda: {"characters": [{"id": key, "name": key} for key in rigs]})
    monkeypatch.setattr(characters, "resolve_character", lambda scene, identifier: rigs[identifier])
    seen = []
    monkeypatch.setattr(humanoid, "discover_mapping", lambda scene, identifier: seen.append(identifier) or {"ready": True})
    result = smart_motion._motion_context()
    assert result["character_id"] == "second" and seen == ["second"]
    assert (result["first"], result["last"], result["range_source"]) == (30, 389, "selection")
    assert result["keys"] == [40, 250]
    scene.selector = lambda: SimpleNamespace(selected=lambda: SimpleNamespace(ids=[]))
    result = smart_motion._motion_context()
    assert result["character_id"] == "" and result["keys"] == []
    assert result.get("rig", {}) == {}


def test_character_label_uses_only_meshes_owned_by_that_rig_and_preserves_native_name():
    meshes = {"mesh-a": "Manny", "mesh-b": "Other character"}
    viewer = SimpleNamespace(get_behaviours=lambda name: list(meshes), get_behaviour_owner=lambda item: item)
    scene = SimpleNamespace(behaviour_viewer=lambda: viewer,
                            model_viewer=lambda: SimpleNamespace(get_object_name=lambda item: meshes[item]))
    rig = SimpleNamespace(owned={"joint", "mesh-a"})
    assert smart_motion._character_label(scene, rig, "UE5 Manny.casc", "Rig info") == "Manny"
    assert smart_motion._character_label(scene, SimpleNamespace(owned={"joint"}), "UE5 Manny.casc", "Rig info") == "UE5 Manny · Rig info"


@pytest.mark.parametrize("endpoint", ["https://example.com", "http://user:pass@127.0.0.1:9560",
                                      "http://127.0.0.1:9560/path", "http://127.0.0.1:9560?key=x"])
def test_panel_does_not_forward_local_service_authentication_to_other_endpoints(tmp_path, monkeypatch, endpoint):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    (tmp_path / "smart-motion-service.json").write_text(json.dumps({
        "endpoint": endpoint, "token": "private-token-value-not-for-reports"}), encoding="utf-8")
    with pytest.raises(ValueError):
        smart_motion._connection()


def test_bootstrap_launches_only_the_registered_external_interpreter_without_a_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    monkeypatch.setenv("PYTHONHOME", "embedded-interpreter-home")
    monkeypatch.setenv("PYTHONPATH", "embedded-module-search-path")
    interpreter = tmp_path / "interpreter.exe"
    interpreter.write_bytes(b"fixture")
    (tmp_path / "smart-motion-panel.json").write_text(json.dumps({
        "external_python": str(interpreter)}), encoding="utf-8")
    commands = []
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(smart_motion, "_launcher", None)
    monkeypatch.setattr(smart_motion.subprocess, "Popen", lambda argv, **options: commands.append((argv, options)) or process)
    smart_motion._start_service()
    smart_motion._start_service()
    assert len(commands) == 1
    argv, options = commands[0]
    assert argv == [str(interpreter), "-m", "mcp_for_cascadeur.motion_service"]
    assert not options.get("shell")
    assert options["env"]["MCP_FOR_CASCADEUR_HOME"] == str(tmp_path.resolve())
    assert "PYTHONHOME" not in options["env"]
    assert "PYTHONPATH" not in options["env"]


def test_explicit_save_copy_uses_a_new_workspace_destination_and_existing_native_save(tmp_path, monkeypatch):
    from mcp_for_cascadeur.host import common

    original = tmp_path / "original.casc"
    original.write_bytes(b"original scene")
    view = SimpleNamespace(name=lambda: "Character Walk.casc", get_path_name=lambda: str(original))
    scene = SimpleNamespace(get_current_frame=lambda: 30, layers_viewer=lambda: SimpleNamespace(frames_count=lambda: 121))
    monkeypatch.setattr(common, "current_context", lambda: (None, None, view, scene))
    monkeypatch.setattr(common, "get_settings", lambda: {"workspace": tmp_path})
    writes = []

    def save(operation, arguments):
        assert operation == "scene_save"
        path = arguments["path"]
        writes.append(path)
        Path(path).write_bytes(b"saved current pose edits")
        return {"path": path, "verification": "fresh_file"}

    monkeypatch.setattr(smart_motion, "_native_call", save)
    first = smart_motion._scene_info(save_copy=True)
    second = smart_motion._scene_info(save_copy=True)
    assert first["path"] != second["path"]
    assert first["frame"] == 30 and first["frames"] == 121
    assert all(Path(path).is_relative_to(tmp_path / "kimodo/sources") for path in writes)
    assert original.read_bytes() == b"original scene"
