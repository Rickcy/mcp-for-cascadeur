import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_for_cascadeur.host import characters, curves, playback, recovery, workflows
from mcp_for_cascadeur.host import native_operations as operations


def section(interpolation="BEZIER", key=True):
    return {
        "interpolation": interpolation,
        "key": {} if key else None,
        "interval_common": {"ik_fk": "IK", "fixation": "None"},
        "ai_description": "",
    }


class Layer:
    def __init__(self, objects):
        self.obj_ids = set(objects)
        self.sections = {0: section(), 6: section(), 7: section("NONE")}

    def is_key(self, frame):
        return frame in self.sections


@pytest.fixture
def native(monkeypatch, tmp_path):
    context = SimpleNamespace(count=8, positions={}, queued=[], selection=set(), log="")
    context.layers = {"body": Layer(["point"]), "finger": Layer(["finger"])}
    context.rig = SimpleNamespace(
        id="character", controls={"point": "point"}, owned={"point", "finger"}, fingers={}
    )
    scene = SimpleNamespace(
        layers_viewer=lambda: SimpleNamespace(
            all_layer_ids=lambda: list(context.layers), layer=lambda identifier: context.layers[identifier]
        ),
        model_viewer=lambda: SimpleNamespace(
            get_objects=lambda name=None: (
                ["point", "finger"] if name is None else (["plane"] if name == "plane" else [])
            )
        ),
        selector=lambda: SimpleNamespace(select=lambda values: setattr(context, "selection", values)),
    )
    app = SimpleNamespace(get_action_manager=lambda: SimpleNamespace(call_action=Mock()))
    view = SimpleNamespace(name=lambda: "fixture", get_path_name=lambda: str(tmp_path / "fixture.casc"))
    csc = SimpleNamespace(
        SystemVariables=SimpleNamespace(git_version=lambda: "2026.2.3", git_count=lambda: "44356")
    )
    context.scene, context.app, context.view, context.csc = scene, app, view, csc
    monkeypatch.setattr(operations.common, "current_context", lambda: (csc, app, view, scene))
    monkeypatch.setattr(characters, "resolve_character", lambda *_args, **_kwargs: context.rig)
    monkeypatch.setattr(curves, "character_tracks", lambda *_args: context.layers)
    monkeypatch.setattr(playback, "require_idle", lambda: None)
    monkeypatch.setattr(operations.animation, "_size", lambda _scene: context.count)
    monkeypatch.setattr(operations.animation, "_tracks", lambda _scene, tracks: tracks)
    monkeypatch.setattr(
        operations.animation,
        "_read",
        lambda _scene, obj, frame, _space: {
            "position": context.positions.get((obj, frame), [0.0, float(frame), 0.0])
        },
    )
    monkeypatch.setattr(operations.animation, "timeline_set_frame", Mock())
    monkeypatch.setattr(operations.state, "section_state", copy.deepcopy)
    monkeypatch.setattr(
        operations, "_ui_snapshot", lambda: {"mocap": [], "video": [], "apply": [], "download": []}
    )
    monkeypatch.setattr(operations, "_log_cursor", lambda: {"path": "log", "offset": 123})
    monkeypatch.setattr(
        operations,
        "_read_log",
        lambda cursor: (context.log, {**cursor, "offset": cursor["offset"] + len(context.log)}),
    )
    monkeypatch.setattr(operations, "_model_ready", lambda: True)
    monkeypatch.setattr(operations, "_interval_mode_state", lambda: {"mode": 0, "selection_mode": 1})
    monkeypatch.setattr(
        operations,
        "_qt",
        lambda: (
            None,
            None,
            None,
            None,
            SimpleNamespace(singleShot=lambda _delay, callback: context.queued.append(callback)),
            None,
            None,
        ),
    )
    monkeypatch.setattr(workflows, "tracks_select_frames", lambda *_args: {"verification": {"matches": True}})
    operations._callbacks.clear()
    return context


def plan(native, operation="motion_generation", **arguments):
    return operations.prepare(operation, {"first": 0, "last": 6, **arguments})


class _ToolbarButton:
    def __init__(self, *, visible=True, enabled=True):
        self.visible, self.enabled, self.invalid = visible, enabled, False

    def property(self, name):
        assert not self.invalid
        return operations.ROUTES["autophysics"] if name == "actionId" else True

    def isVisible(self):
        assert not self.invalid
        return self.visible

    def isEnabled(self):
        assert not self.invalid
        return self.enabled

    def childItems(self):
        assert not self.invalid
        return []


def _frontend_qt(native, monkeypatch, *, items=None, on_release=lambda: None, accepted=True):
    buttons = [_ToolbarButton()] if items is None else items
    root = SimpleNamespace(childItems=lambda: buttons, property=lambda _name: None)
    monkeypatch.setattr(operations, "_roots", lambda kind: [root] if kind == "AutoPhysics" else [])
    invoked = []

    def invoke(item, signal, connection):
        invoked.append((item, signal, connection))
        if accepted:
            on_release()
        return accepted

    timer = operations._qt()[4]
    queued = object()
    monkeypatch.setattr(
        operations,
        "_qt",
        lambda: (
            None,
            None,
            SimpleNamespace(invokeMethod=invoke),
            SimpleNamespace(QueuedConnection=queued),
            timer,
            None,
            None,
        ),
    )
    return buttons, invoked, queued


def test_prepare_is_json_only_and_does_not_dispatch(native):
    result = plan(native)
    json.dumps(result, allow_nan=False)
    assert result["tracks"] == ["body", "finger"]
    assert result["point_tracks"] == ["body"]
    assert result["owned_objects"] == ["finger", "point"]
    assert result["expected_count"] == 8
    assert not native.queued
    assert not native.selection
    assert result["baseline"]["tracks"]["finger"]["sections"]["0"]["interpolation"] == "BEZIER"


def test_unknown_build_rejected_before_snapshot_or_ui(native, monkeypatch):
    native.csc.SystemVariables.git_count = lambda: "44357"
    snapshot = Mock()
    monkeypatch.setattr(operations, "_snapshot", snapshot)
    with pytest.raises(operations.NativeOperationError, match="44356") as caught:
        plan(native)
    assert caught.value.code == "NATIVE_BUILD_UNVERIFIED"
    snapshot.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        {"first": True},
        {"last": 7},
        {"position_tolerance": float("nan")},
        {"position_tolerance": 0},
        {"tracks": ["foreign"]},
        {"nonsense": 1},
    ],
)
def test_invalid_plan_rejected_without_writes(native, arguments):
    with pytest.raises((operations.NativeOperationError, ValueError)):
        plan(native, **arguments)
    assert not native.queued and not native.selection


def test_mocap_can_extend_a_single_frame_scene(native, monkeypatch, tmp_path):
    native.count = 1
    native.layers["body"].sections = {0: section()}
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"test")
    monkeypatch.setattr(operations.common, "workspace_path", lambda *_args, **_kwargs: video)
    result = plan(native, "mocap", video_path=str(video))
    assert result["phase"] == "allocate"
    assert result["expected_count"] == 8
    calls = []

    def allocate(tracks, frames, **kwargs):
        calls.append((tracks, frames, kwargs))
        native.count = 8

    monkeypatch.setattr(operations.animation, "tracks_set_keys", allocate)
    patch = operations.step({"job_id": "a", "plan": result})
    assert patch["phase"] == "allocation_settings"
    assert calls == [(["body", "finger"], [0, 6], {"extend_timeline": True})]
    padding = Mock(return_value={"verified": True})
    monkeypatch.setattr(operations, "_hold_extended_settings", padding)
    following = operations.step({"job_id": "a", "plan": result, "driver": patch["driver"]})
    assert following["phase"] == "scope"
    padding.assert_called_once_with(native.scene, result)


def test_native_frame_budget_is_not_the_curve_editor_121_limit(native):
    native.count = 152
    native.layers["body"].sections[150] = section()
    assert plan(native, last=150)["expected_count"] == 152


def test_motion_generation_requires_existing_endpoint_keys(native):
    del native.layers["body"].sections[6]
    with pytest.raises(operations.NativeOperationError) as caught:
        plan(native)
    assert caught.value.code == "NATIVE_ENDPOINT_KEYS_REQUIRED"


def test_unbaking_completion_waits_for_late_auto_unlock():
    assert not operations._completed("unbake", "Auto keys: done")
    assert operations._completed("unbake", "Auto keys: done\nAuto posing.Auto unlock: done")


def test_autophysics_completion_requires_calculation_and_application():
    assert not operations._completed("autophysics", "Command: Snap to autophysics")
    assert not operations._completed("autophysics", "Autophysics core finished")
    assert operations._completed("autophysics", "Autophysics core finished\nCommand: Snap to autophysics")


def test_unbaking_requires_baked_input(native):
    with pytest.raises(operations.NativeOperationError) as caught:
        plan(native, "unbake")
    assert caught.value.code == "NATIVE_BAKED_INPUT_REQUIRED"
    native.layers["body"].sections[0] = section("FIXED")
    assert plan(native, "unbake")["operation"] == "unbake"


def test_dispatch_queues_and_reacquires_context_without_synchronous_native_action(native):
    result = plan(native)
    called = Mock()
    native.app.get_action_manager = lambda: SimpleNamespace(call_action=called)
    record = {"job_id": "owned", "plan": result, "driver": {"phase": "dispatch"}}
    patch = operations.step(record)
    assert patch["phase"] == "wait"
    called.assert_not_called()
    assert len(native.queued) == 1
    native.queued.pop()()
    called.assert_called_once_with("View.Inbetweening_RunMotionGeneration")
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step(record)
    assert caught.value.code == "NATIVE_DISPATCH_DUPLICATE"


def test_queue_fails_closed_if_scene_changed_before_callback(native):
    result = plan(native)
    called = Mock()
    native.app.get_action_manager = lambda: SimpleNamespace(call_action=called)
    patch = operations.step({"job_id": "owned", "plan": result, "driver": {"phase": "dispatch"}})
    native.view.name = lambda: "another scene"
    native.queued.pop()()
    observation = operations._callbacks[patch["driver"]["callback_id"]]
    assert observation["status"] == "failed"
    assert observation["code"] == "NATIVE_JOB_CONTEXT_LOST"
    called.assert_not_called()


def test_lost_callback_observation_is_never_replayed(native):
    result = plan(native)
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step(
            {"job_id": "owned", "plan": result, "driver": {"phase": "wait", "callback_id": "lost"}}
        )
    assert caught.value.code == "NATIVE_DISPATCH_OUTCOME_UNKNOWN"
    assert not native.queued


def test_native_callback_still_running_cannot_complete_from_old_log(native):
    result = plan(native)
    patch = operations.step({"job_id": "owned", "plan": result, "driver": {"phase": "dispatch"}})
    native.log = "Root motion: done"
    observed = operations.step({"job_id": "owned", "plan": result, "driver": patch["driver"]})
    assert observed["status"] == "running"
    assert observed["progress"]["native_callback"] == "queued"


def test_nested_native_picker_can_be_serviced_while_choose_callback_is_started(native, monkeypatch):
    result = plan(native)
    operations._callbacks["nested"] = {"status": "started"}
    picker = Mock(return_value={"phase": "video_pick_wait", "status": "running"})
    monkeypatch.setattr(operations, "_pick_video", picker)
    record = {"job_id": "owned", "plan": result, "driver": {"phase": "video_pick", "callback_id": "nested"}}
    patch = operations.step(record)
    assert patch["phase"] == "video_pick_wait"
    picker.assert_called_once()
    blocked = operations.step({**record, "driver": {"phase": "video_pick_wait", "callback_id": "nested"}})
    assert blocked["progress"]["native_callback"] == "started"


class GuardRotation:
    def to_quaternion(self):
        return SimpleNamespace(w=lambda: -1, x=lambda: 0, y=lambda: 0, z=lambda: 0)

    def to_euler_angles(self):
        return [[0.0], [0.0], [6.283185307179586]]


def _finger_guard_fixture(native):
    native.layers["finger"].sections.pop(7)
    native.rig.fingers = {"left": {"index1": SimpleNamespace(track_id="finger", rotation_data="rotation")}}
    native.scene.data_viewer = lambda: SimpleNamespace(
        get_data=lambda _ref: SimpleNamespace(mode=SimpleNamespace(name="Animation")),
        get_data_value=lambda _ref, _frame: GuardRotation(),
    )


def test_guard_plan_preserves_source_metadata_and_full_turn(native):
    _finger_guard_fixture(native)
    native.layers["body"].sections[0] = section("FIXED")
    native.layers["finger"].sections[6]["ai_description"] = "original description"
    result = plan(native, "unbake")
    assert result["guard_anchors"] == {"finger": 7}
    assert result["guard_anchor_sections"]["finger"]["interpolation"] == "STEP"
    assert result["guard_anchor_sections"]["finger"]["ai_description"] == ""
    assert result["guard_anchor_channels"]["rotation"]["value"]["euler_radians"][2] == pytest.approx(
        6.283185307179586
    )
    assert native.layers["finger"].sections[6]["ai_description"] == "original description"


@pytest.mark.parametrize("poison_protected_value", [False, True])
def test_guard_anchor_checks_every_value_without_terminal_solver(native, monkeypatch, poison_protected_value):
    _finger_guard_fixture(native)
    native.layers["body"].sections[0] = section("FIXED")
    result = plan(native, "unbake")
    value = operations.state.encode(GuardRotation())
    original = {
        "count": 8,
        "tracks": {"finger": {"keys": [0, 6], "sections": {"0": section(), "6": section()}}},
        "channels": {
            "rotation": {
                "kind": "data",
                "mode": "Animation",
                "values": [copy.deepcopy(value) for _ in range(8)],
            },
            "setting": {"kind": "setting", "mode": "Static", "value": True},
        },
    }
    observed = [original]
    monkeypatch.setattr(recovery, "latest_committed_checkpoint", lambda _scene: observed[0])
    monkeypatch.setattr(operations.state, "validate_section_metadata", lambda _value: None)
    monkeypatch.setattr(operations.state, "decode", copy.deepcopy)

    def apply_metadata(target, descriptor):
        target.clear()
        target.update(copy.deepcopy(descriptor))

    monkeypatch.setattr(operations.state, "apply_section_metadata", apply_metadata)
    events = []

    def modify(_scene, _label, callback):
        observed[0] = copy.deepcopy(original)
        current = observed[0]

        def insert(track, frame, is_key):
            assert is_key is True
            events.append(("insert", track, frame))
            current["tracks"][track]["keys"].append(frame)
            current["tracks"][track]["sections"][str(frame)] = section("STEP")

        editor = SimpleNamespace(
            set_fixed_interpolation_or_key_if_need=insert,
            normalize_sections=lambda _scene: None,
            change_section=lambda frame, track, fn: fn(current["tracks"][track]["sections"][str(frame)]),
        )
        model = SimpleNamespace(
            layers_editor=lambda: editor,
            fit_animation_size_by_layers=lambda: None,
            data_editor=lambda: SimpleNamespace(
                set_data_value=lambda _ref, frame, saved: current["channels"]["rotation"][
                    "values"
                ].__setitem__(frame, saved)
            ),
        )
        updater = SimpleNamespace(
            generate_update=lambda: events.append(("generate",)),
            run_update=Mock(side_effect=AssertionError("terminal solver is forbidden")),
            get_interpolator=Mock(side_effect=AssertionError("terminal interpolation is forbidden")),
        )
        callback(model, updater)
        if poison_protected_value:
            current["channels"]["setting"]["value"] = False

    monkeypatch.setattr(operations.common, "modify", modify)
    if poison_protected_value:
        with pytest.raises(operations.NativeOperationError) as caught:
            operations._anchor_guards(native.scene, native.rig, result)
        assert caught.value.safe_to_rollback is True
    else:
        report = operations._anchor_guards(native.scene, native.rig, result)
        assert report["verified"] is True
        assert observed[0]["channels"] == original["channels"]
        assert observed[0]["tracks"]["finger"]["sections"]["7"] == result["guard_anchor_sections"]["finger"]
    assert events == [("insert", "finger", 7), ("generate",)]


def test_settings_padding_holds_only_new_protected_samples():
    checkpoint = {
        "count": 2,
        "channels": {
            "setting:owned": {
                "kind": "setting",
                "mode": "Animation",
                "id": "owned",
                "owner": "character",
                "values": [True, False],
            },
            "setting:foreign": {
                "kind": "setting",
                "mode": "Animation",
                "id": "foreign",
                "owner": "prop",
                "values": [2, 7],
            },
            "setting:static": {"kind": "setting", "mode": "Static", "value": True},
            "data:derived": {"kind": "data", "mode": "Animation", "values": [0, 1]},
        },
    }
    planned = operations._settings_padding(
        checkpoint,
        {
            "expected_count": 6,
            "owned_objects": ["character"],
            "first": 3,
            "last": 4,
        },
    )
    assert [(item["id"], item["frame"], item["value"]) for item in planned] == [
        ("owned", 2, False),
        ("owned", 5, False),
        ("foreign", 2, 7),
        ("foreign", 3, 7),
        ("foreign", 4, 7),
        ("foreign", 5, 7),
    ]


@pytest.mark.parametrize("poison", [False, True])
def test_extension_settings_write_only_planned_padding_and_check_other_state(native, monkeypatch, poison):
    source = {
        "count": 1,
        "channels": {
            "setting:owned": {
                "kind": "setting",
                "mode": "Animation",
                "id": "owned",
                "owner": "point",
                "values": [True],
            },
            "setting:foreign": {
                "kind": "setting",
                "mode": "Animation",
                "id": "foreign",
                "owner": "prop",
                "values": [9],
            },
            "data:derived": {"kind": "data", "mode": "Static", "value": 100},
        },
    }
    current = [copy.deepcopy(source)]
    current[0]["count"] = 8
    current[0]["channels"]["setting:owned"]["values"] += [False] * 7
    current[0]["channels"]["setting:foreign"]["values"] += [0] * 7
    native.scene.model_viewer = lambda: SimpleNamespace(get_objects=lambda: ["point", "prop"])
    native.scene.data_viewer = lambda: SimpleNamespace(
        get_all_settings_id=lambda obj: ["owned" if obj == "point" else "foreign"],
        get_setting=lambda _ref: SimpleNamespace(mode=SimpleNamespace(name="Animation")),
        get_setting_value=lambda ref, frame: current[0]["channels"]["setting:" + ref]["values"][frame],
    )
    monkeypatch.setattr(recovery, "active_checkpoint", lambda _scene: source)
    monkeypatch.setattr(recovery, "latest_committed_checkpoint", lambda _scene: current[0])
    capture = Mock(side_effect=lambda _scene: copy.deepcopy(current[0]))
    monkeypatch.setattr(operations.state, "capture", capture)
    writes = []

    def modify(_scene, _label, callback):
        current[0] = copy.deepcopy(current[0])

        def set_value(ref, frame, value):
            writes.append((ref, frame, value))
            current[0]["channels"]["setting:" + ref]["values"][frame] = value

        callback(
            SimpleNamespace(data_editor=lambda: SimpleNamespace(set_setting_value=set_value)),
            SimpleNamespace(run_update=Mock(side_effect=AssertionError("guard solver forbidden"))),
        )
        if poison:
            current[0]["channels"]["data:derived"]["value"] = 101

    monkeypatch.setattr(operations.common, "modify", modify)
    scope = {"expected_count": 8, "owned_objects": ["point"], "first": 0, "last": 6}
    if poison:
        with pytest.raises(operations.NativeOperationError) as caught:
            operations._hold_extended_settings(native.scene, scope, after_native=True)
        assert caught.value.code == "NATIVE_SETTING_PADDING_CHANGED_VALUES"
        assert caught.value.safe_to_rollback is True
    else:
        proof = operations._hold_extended_settings(native.scene, scope, after_native=True)
        assert proof["verified"] is True and proof["changed_samples"] == 8
        assert current[0]["channels"]["setting:owned"]["values"] == [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ]
        assert current[0]["channels"]["setting:foreign"]["values"] == [9] * 8
    assert ("owned", 7, True) in writes
    assert not any(ref == "owned" and frame < 7 for ref, frame, _value in writes)
    capture.assert_called_once_with(native.scene)


def test_apply_waits_for_exact_owned_confirmation(native, monkeypatch):
    result = plan(native, "autophysics")
    click = Mock()
    monkeypatch.setattr(operations, "_queued_click", click)
    record = {"job_id": "owned", "plan": result, "driver": {"phase": "physics_confirm"}}
    waiting = operations.step(record)
    assert waiting["status"] == "awaiting_input"
    click.assert_not_called()
    monkeypatch.setattr(operations, "_ui_snapshot", lambda: {"apply": [{"title": "Apply AutoPhysics"}]})
    patch = operations.step(record)
    assert patch["phase"] == "wait"
    click.assert_called_once_with("Apply AutoPhysics", "All Frames")


def test_completion_needs_log_marker_and_scoped_output(native):
    result = plan(native)
    after = copy.deepcopy(result["baseline"])
    report = operations._verify(result, after, "Root motion: done")
    assert not report["matches"]
    assert "the scoped native result is unchanged" in report["errors"]
    after["positions"]["point"][3][0] += 5
    after["tracks"]["body"]["sections"]["0"] = section("FIXED")
    assert operations._verify(result, after, "Root motion: done")["matches"]
    assert not operations._verify(result, after, "dispatch returned")["matches"]


def test_untouched_tracks_and_frames_remain_protected(native):
    result = plan(native, tracks=["body"])
    after = copy.deepcopy(result["baseline"])
    after["positions"]["point"][3][0] += 5
    after["tracks"]["body"]["sections"]["0"] = section("FIXED")
    after["tracks"]["finger"]["sections"]["0"]["ai_description"] = "wrong"
    after["positions"]["point"][7][0] += 1
    report = operations._verify(result, after, "Root motion: done")
    assert not report["matches"]
    assert any("unselected track" in item for item in report["errors"])
    assert any("protected Point" in item for item in report["errors"])


def test_new_reference_track_requires_exact_imported_plane_ownership(native):
    result = plan(native)
    after = copy.deepcopy(result["baseline"])
    after["positions"]["point"][3][0] += 5
    after["tracks"]["body"]["sections"]["0"] = section("FIXED")
    result["operation"] = "mocap"
    after["tracks"]["body"]["sections"].update({str(frame): section() for frame in range(7)})
    after["tracks"]["reference"] = {"objects": ["plane"], "sections": {"0": section()}}
    marker = "All frames have been processed"
    assert operations._verify(result, after, marker, reference_object="plane")["matches"]
    assert not operations._verify(result, after, marker)["matches"]
    after["tracks"]["reference"]["objects"].append("unrelated")
    assert not operations._verify(result, after, marker, reference_object="plane")["matches"]


def test_new_terminal_metadata_is_allowed_only_when_timeline_growth_was_planned(native):
    result = plan(native)
    result["baseline"]["count"] = 1
    result["baseline"]["positions"]["point"] = [[0, 0, 0]]
    for track in result["baseline"]["tracks"].values():
        track["sections"] = {"0": section()}
    after = copy.deepcopy(result["baseline"])
    after["count"] = 8
    after["positions"]["point"] = [[0, 0, 0]] * 8
    after["tracks"]["body"]["sections"] = {"0": section("FIXED"), "6": section(), "7": section("NONE")}
    assert operations._verify(result, after, "Root motion: done")["matches"]
    result["expected_count"] = 1
    assert not operations._verify(result, after, "Root motion: done")["matches"]


def test_mocap_readback_requires_every_requested_frame_to_be_a_key(native):
    result = plan(native, "mocap", reference_object="plane")
    after = copy.deepcopy(result["baseline"])
    after["positions"]["point"][3][0] += 5
    marker = "All frames have been processed"
    assert not operations._verify(result, after, marker)["matches"]
    after["tracks"]["body"]["sections"].update({str(frame): section() for frame in range(7)})
    assert operations._verify(result, after, marker)["matches"]


def test_queued_cancel_does_not_mean_worker_stopped(native):
    result = plan(native)
    record = {"job_id": "owned", "plan": result, "driver": {"phase": "dispatch"}}
    running = operations.step(record)
    cancelled = operations.cancel({**record, "driver": running["driver"]})
    assert cancelled["status"] == "awaiting_input"
    assert cancelled["result"]["safe_to_rollback"] is False


def test_cancel_before_first_stage_has_no_native_effects(native):
    result = plan(native)
    cancelled = operations.cancel({"job_id": "owned", "plan": result})
    assert cancelled["status"] == "cancelled"
    assert cancelled["safe_to_rollback"] is True
    assert not native.queued


def test_mocap_cancel_waits_after_dialog_closed(native):
    result = plan(native, "mocap", reference_object="plane")
    patch = operations.step(
        {
            "job_id": "owned",
            "plan": result,
            "driver": {"phase": "cancelling", "log_cursor": {"path": "log", "offset": 123}},
        }
    )
    assert patch["status"] == "awaiting_input"
    assert patch["result"]["safe_to_rollback"] is False


def test_mocap_cancel_can_finish_only_after_native_completion_and_closed_dialog(native):
    result = plan(native, "mocap", reference_object="plane")
    native.log = "All frames have been processed"
    patch = operations.cancel(
        {
            "job_id": "owned",
            "plan": result,
            "driver": {"phase": "cancelling", "log_cursor": {"path": "log", "offset": 123}},
        }
    )
    assert patch["status"] == "cancelled"
    assert patch["safe_to_rollback"] is True


def test_cancel_never_confirms_model_download(native, monkeypatch):
    result = plan(native, "mocap", reference_object="plane", allow_model_download=True)
    click = Mock()
    monkeypatch.setattr(operations, "_queued_click", click)
    monkeypatch.setattr(operations, "_ui_snapshot", lambda: {"download": [{"title": "Download Mocap"}]})
    patch = operations.step(
        {
            "job_id": "owned",
            "plan": result,
            "cancel_requested": True,
            "driver": {"phase": "cancelling", "log_cursor": {"path": "log", "offset": 123}},
        }
    )
    assert patch["status"] == "awaiting_input"
    assert patch["safe_to_rollback"] is False
    click.assert_not_called()


def test_physics_state_uses_component_context_before_window_context(native, monkeypatch):
    component = object()
    monkeypatch.setattr(operations, "_roots", lambda kind: [component] if kind == "AutoPhysics" else [])
    monkeypatch.setattr(operations, "_windows", lambda: [])
    query = Mock(return_value="false")
    monkeypatch.setattr(operations, "_eval", query)
    assert operations._physics_enabled() is False
    assert query.call_args.args[0] is component


@pytest.mark.parametrize("initial_mode", [0, 1])
def test_physics_scope_sets_bounds_before_toggle_and_reads_back_exact_mode(native, monkeypatch, initial_mode):
    values = {"mode": initial_mode, "selection_mode": 1}
    component = object()
    events = []
    expected = {"first": 0, "last": 6}

    def set_bounds(first, last, visible):
        assert (first, last, visible) == (0, 6, True)
        events.append("bounds")
        return {"playback": expected, "visible": expected}

    def toggle(item, expression):
        assert item is component and "toggleEditMode" in expression
        events.append("toggle")
        values["mode"] = 1

    monkeypatch.setattr(operations.animation, "timeline_set_range", set_bounds)
    monkeypatch.setattr(
        operations.animation, "timeline_get", lambda: {"playback": expected, "visible": expected}
    )
    monkeypatch.setattr(operations, "_interval_mode_context", lambda: (component, dict(values)))
    monkeypatch.setattr(operations, "_interval_mode_state", lambda: dict(values))
    monkeypatch.setattr(operations, "_eval", toggle)
    report = operations._prepare_physics_scope({"first": 0, "last": 6})
    assert report["verified"] is True
    assert events == (["bounds", "toggle"] if initial_mode == 0 else ["bounds"])


def test_physics_wrong_bounds_fail_before_mode_change_or_dispatch(native, monkeypatch):
    result = plan(native, "autophysics")
    monkeypatch.setattr(
        operations.animation,
        "timeline_set_range",
        lambda *_args, **_kwargs: {
            "playback": {"first": 0, "last": 7},
            "visible": {"first": 0, "last": 7},
        },
    )
    toggle = Mock()
    monkeypatch.setattr(operations, "_eval", toggle)
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step({"job_id": "owned", "plan": result})
    assert caught.value.code == "NATIVE_PHYSICS_BOUNDS_MISMATCH"
    assert caught.value.execution_state == "not_started"
    toggle.assert_not_called()
    assert not native.queued


def test_physics_wrong_mode_readback_fails_before_dispatch(native, monkeypatch):
    result = plan(native, "autophysics")
    expected = {"first": 0, "last": 6}
    monkeypatch.setattr(
        operations.animation,
        "timeline_set_range",
        lambda *_args, **_kwargs: {"playback": expected, "visible": expected},
    )
    monkeypatch.setattr(
        operations.animation, "timeline_get", lambda: {"playback": expected, "visible": expected}
    )
    monkeypatch.setattr(
        operations, "_interval_mode_context", lambda: (object(), {"mode": 0, "selection_mode": 1})
    )
    monkeypatch.setattr(operations, "_eval", Mock())
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step({"job_id": "owned", "plan": result})
    assert caught.value.code == "NATIVE_PHYSICS_MODE_MISMATCH"
    assert caught.value.execution_state == "not_started"
    assert not native.queued


def test_physics_scope_is_rechecked_immediately_before_native_action(native, monkeypatch):
    result = plan(native, "autophysics")
    monkeypatch.setattr(
        operations.animation,
        "timeline_get",
        lambda: {
            "playback": {"first": 0, "last": 7},
            "visible": {"first": 0, "last": 7},
        },
    )
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step({"job_id": "owned", "plan": result, "driver": {"phase": "dispatch"}})
    assert caught.value.code == "NATIVE_PHYSICS_BOUNDS_MISMATCH"
    assert not native.queued


def test_enabled_autophysics_is_stopped_before_a_fresh_calculation(native, monkeypatch):
    result = plan(native, "autophysics")
    monkeypatch.setattr(operations, "_assert_physics_scope", lambda _plan: {"verified": True})
    enabled = [True]
    monkeypatch.setattr(operations, "_physics_enabled", lambda: enabled[0])
    actions = []

    def switch(action):
        actions.append(action)
        enabled[0] = not enabled[0]

    native.app.get_tools_manager = Mock(side_effect=AssertionError("editor.turn_off is unverified"))
    native.app.get_action_manager = lambda: SimpleNamespace(call_action=switch)
    _frontend_qt(native, monkeypatch, on_release=lambda: switch(operations.ROUTES["autophysics"]))
    record = {"job_id": "owned", "plan": result, "driver": {"phase": "dispatch"}}
    queued_off = operations.step(record)
    assert queued_off["phase"] == "physics_disabled" and actions == []
    native.queued.pop(0)()
    assert enabled[0] is False
    assert actions == [operations.ROUTES["autophysics"]]
    queued_on = operations.step(
        {**record, "driver": {**queued_off["driver"], "log_text": "Autophysics core finished old"}}
    )
    assert queued_on["phase"] == "physics_solve"
    assert queued_on["driver"]["log_text"] == ""
    assert actions == [operations.ROUTES["autophysics"]]
    native.queued.pop(0)()
    assert enabled[0] is True
    assert actions == [operations.ROUTES["autophysics"]] * 2
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step({**record, "driver": queued_off["driver"]})
    assert caught.value.code == "NATIVE_PHYSICS_STOP_UNVERIFIED"
    assert actions == [operations.ROUTES["autophysics"]] * 2


def test_cancel_legacy_failed_physics_stop_queues_unique_off_and_never_recalculates(native, monkeypatch):
    result = plan(native, "autophysics")
    enabled, actions = [True], []
    monkeypatch.setattr(operations, "_physics_enabled", lambda: enabled[0])

    def switch(action):
        actions.append(action)
        enabled[0] = False

    native.app.get_action_manager = lambda: SimpleNamespace(call_action=switch)
    _frontend_qt(native, monkeypatch, on_release=lambda: switch(operations.ROUTES["autophysics"]))
    operations._callbacks["legacy_turn_off"] = {"status": "returned"}
    record = {
        "job_id": "owned",
        "plan": result,
        "cancel_requested": True,
        "driver": {"phase": "physics_disabled", "callback_id": "legacy_turn_off"},
    }
    first = operations.cancel(record)
    assert first["phase"] == "physics_cancel_disabled"
    assert first["driver"]["callback_id"] == "owned:physics_cancel_disabled"
    assert first["safe_to_rollback"] is False
    waiting = operations.cancel({**record, "driver": first["driver"]})
    assert waiting["status"] != "cancelled"
    assert len(native.queued) == 1
    native.queued.pop(0)()
    final = operations.cancel({**record, "driver": first["driver"]})
    assert final["status"] == "cancelled" and final["safe_to_rollback"] is True
    assert actions == [operations.ROUTES["autophysics"]]
    assert "owned:physics_solve" not in operations._callbacks


def test_failed_physics_cancel_off_does_not_toggle_again(native, monkeypatch):
    result = plan(native, "autophysics")
    monkeypatch.setattr(operations, "_physics_enabled", lambda: True)
    operations._callbacks["cancel_off"] = {"status": "returned"}
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.cancel(
            {
                "job_id": "owned",
                "plan": result,
                "cancel_requested": True,
                "driver": {"phase": "physics_cancel_disabled", "callback_id": "cancel_off"},
            }
        )
    assert caught.value.code == "NATIVE_PHYSICS_STOP_UNVERIFIED"
    assert not native.queued


@pytest.mark.parametrize("enabled", [True, None])
def test_failed_autophysics_off_postcondition_never_toggles_on(native, monkeypatch, enabled):
    result = plan(native, "autophysics")
    monkeypatch.setattr(operations, "_assert_physics_scope", lambda _plan: {"verified": True})
    monkeypatch.setattr(operations, "_physics_enabled", lambda: enabled)
    operations._callbacks["off"] = {"status": "returned"}
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step(
            {"job_id": "owned", "plan": result, "driver": {"phase": "physics_disabled", "callback_id": "off"}}
        )
    assert caught.value.code == "NATIVE_PHYSICS_STOP_UNVERIFIED"
    assert caught.value.safe_to_rollback is False
    assert not native.queued


def test_autophysics_false_branch_starts_one_fresh_calculation(native, monkeypatch):
    result = plan(native, "autophysics")
    monkeypatch.setattr(operations, "_assert_physics_scope", lambda _plan: {"verified": True})
    monkeypatch.setattr(operations, "_physics_enabled", lambda: False)
    patch = operations.step({"job_id": "owned", "plan": result, "driver": {"phase": "dispatch"}})
    assert patch["phase"] == "physics_solve"
    assert patch["effects"] == [
        {
            "kind": "native_frontend_action_queued",
            "action": operations.ROUTES["autophysics"],
            "signal": "released",
        }
    ]
    assert len(native.queued) == 1


def test_idempotent_autophysics_requires_both_markers_and_protects_guard(native):
    result = plan(native, "autophysics")
    after = copy.deepcopy(result["baseline"])
    both = "Autophysics core finished\nCommand: Snap to autophysics"
    accepted = operations._verify(result, after, both)
    assert accepted["matches"] is True
    assert accepted["unchanged"] is True
    assert not operations._verify(result, after, "Command: Snap to autophysics")["matches"]
    after["positions"]["point"][7][0] += 10
    assert not operations._verify(result, after, both)["matches"]


def test_physics_frontend_releases_unique_effectively_visible_button_once(native, monkeypatch):
    hidden = _ToolbarButton(visible=False)
    disabled = _ToolbarButton(enabled=False)
    visible = _ToolbarButton()
    buttons, invoked, queued = _frontend_qt(
        native,
        monkeypatch,
        items=[hidden, visible, disabled],
        on_release=lambda: setattr(visible, "invalid", True),
    )
    result = plan(native, "autophysics")
    record = {"job_id": "owned", "plan": result}
    patch = operations._physics_switch(record, {}, "physics_disabled")
    assert invoked == []
    assert patch["effects"][0]["kind"] == "native_frontend_action_queued"
    with pytest.raises(operations.NativeOperationError) as caught:
        operations._physics_switch(record, {}, "physics_disabled")
    assert caught.value.code == "NATIVE_DISPATCH_DUPLICATE"
    native.queued.pop(0)()
    assert invoked == [(visible, "released", queued)]
    assert operations._callbacks[patch["driver"]["callback_id"]]["status"] == "returned"
    assert not native.queued


@pytest.mark.parametrize("items", [[], [_ToolbarButton(), _ToolbarButton()], [_ToolbarButton(visible=False)]])
def test_physics_frontend_ambiguous_or_hidden_buttons_never_trigger(native, monkeypatch, items):
    _buttons, invoked, _queued = _frontend_qt(native, monkeypatch, items=items)
    result = plan(native, "autophysics")
    patch = operations._physics_switch({"job_id": "owned", "plan": result}, {}, "physics_disabled")
    native.queued.pop(0)()
    with pytest.raises(operations.NativeOperationError) as caught:
        operations._callback(patch["driver"])
    assert caught.value.code == "NATIVE_PHYSICS_BUTTON_AMBIGUOUS"
    assert invoked == []


@pytest.mark.parametrize("value", ["null", '{"mode":true,"selection_mode":1}', "unavailable"])
def test_interval_mode_unreadable_context_is_not_started(native, monkeypatch, value):
    item = SimpleNamespace(property=lambda _name: True)
    monkeypatch.setattr(operations, "_roots", lambda _kind: [item])
    monkeypatch.setattr(operations, "_eval", lambda *_args: value)
    with pytest.raises(operations.NativeOperationError) as caught:
        operations._interval_mode_context()
    assert caught.value.code == "NATIVE_INTERVAL_MODE_UNAVAILABLE"
    assert caught.value.execution_state == "not_started"


def test_window_discovery_uses_static_gui_api_with_core_application_wrapper(monkeypatch):
    visible = SimpleNamespace(isVisible=lambda: True)
    hidden = SimpleNamespace(isVisible=lambda: False)
    calls = []

    def register_quick(name):
        calls.append(("import", name))
        visible.contentItem = lambda: "quick content"

    def windows():
        calls.append(("enumerate",))
        assert callable(getattr(visible, "contentItem", None))
        return [hidden, visible]

    application = SimpleNamespace(
        instance=Mock(return_value=SimpleNamespace()), allWindows=Mock(side_effect=windows)
    )
    gui = SimpleNamespace(QGuiApplication=application)
    monkeypatch.setitem(sys.modules, "PySide6", SimpleNamespace(QtGui=gui))
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", gui)
    monkeypatch.setattr(operations.importlib, "import_module", register_quick)
    assert operations._windows() == [visible]
    assert calls == [("import", "PySide6.QtQuick"), ("enumerate",)]
    application.allWindows.assert_called_once_with()
    application.instance.assert_not_called()


def test_bounded_visual_walk_reaches_shallow_toolbar_before_large_outliner_branches():
    class Item:
        def __init__(self, name, children=()):
            self.name, self.items = name, list(children)

        def childItems(self):
            return self.items

    first, last = Item("first end"), Item("last end")
    for _ in range(10020):
        first, last = Item("first branch", [first]), Item("last branch", [last])
    toolbar = Item("AutoPhysics toolbar")
    root = Item("root", [first, toolbar, last])
    visited = list(operations._walk(root))
    assert len(visited) == 10000
    assert visited[:4] == [root, first, toolbar, last]
    assert toolbar in visited


def test_file_picker_uses_static_widgets_api_with_core_application_wrapper(native, monkeypatch, tmp_path):
    class FileDialog:
        def isVisible(self):
            return True

        def selectFile(self, path):
            self.path = path

        def selectedFiles(self):
            return [self.path]

    dialog = FileDialog()
    application = SimpleNamespace(
        instance=Mock(return_value=SimpleNamespace()), topLevelWidgets=Mock(return_value=[dialog])
    )
    meta, qt = (
        SimpleNamespace(invokeMethod=Mock(return_value=True)),
        SimpleNamespace(QueuedConnection=object()),
    )
    monkeypatch.setattr(operations, "_qt", lambda: (application, FileDialog, meta, qt, None, None, None))
    monkeypatch.setattr(operations, "_roots", lambda _kind: [])
    path = str(tmp_path / "reference.mp4")
    patch = operations._pick_video({"plan": {"arguments": {"video_path": path}}}, {})
    assert patch["phase"] == "video_pick_wait"
    application.topLevelWidgets.assert_called_once_with()
    application.instance.assert_not_called()
    meta.invokeMethod.assert_called_once_with(dialog, "accept", qt.QueuedConnection)


def test_failed_output_after_native_completion_is_safe_to_restore(native):
    result = plan(native)
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step(
            {
                "job_id": "owned",
                "plan": result,
                "driver": {"phase": "verify", "log_text": "Root motion: done"},
            }
        )
    assert caught.value.code == "NATIVE_RESULT_VERIFICATION_FAILED"
    assert caught.value.safe_to_rollback is True
    assert caught.value.execution_state == "verification_failed"


def test_truncated_log_is_unknown_not_success(tmp_path):
    path = tmp_path / "native.log"
    path.write_bytes(b"short")
    with pytest.raises(operations.NativeOperationError) as caught:
        operations._read_log({"path": str(path), "offset": 100})
    assert caught.value.execution_state == "unknown"
    assert caught.value.code == "NATIVE_LOG_REPLACED"


class _WinFunction:
    def __init__(self, call):
        self.call = call

    def __call__(self, *arguments):
        return self.call(*arguments)


def _fake_native_picker(monkeypatch, *, sibling=False, sibling_visible=True, sibling_pid=77):
    import ctypes

    def node(class_name, text="", *, parent=0, owner=0, control_id=0, pid=77, visible=True):
        return {
            "class": class_name,
            "text": text,
            "parent": parent,
            "owner": owner,
            "id": control_id,
            "pid": pid,
            "visible": visible,
        }

    nodes = {
        1: node("Qt653QWindowIcon", "Cascadeur"),
        2: node("#32770", "Load video", owner=1),
        3: node("ComboBoxEx32", parent=2, control_id=1148),
        4: node("ComboBox", parent=3, control_id=1148),
        5: node("Edit", parent=4, control_id=1001),
        6: node("Button", "&Open", parent=2, control_id=1),
        90: node("Qt653QWindowIcon", "Cascadeur", pid=88),
        91: node("#32770", "Load video", owner=90, pid=88),
    }
    if sibling:
        nodes[7] = node("ComboBox", parent=2, control_id=1148)
        nodes[8] = node("Edit", parent=7, pid=sibling_pid, visible=sibling_visible)
    calls = {"sent": [], "posted": []}

    def enumerate_windows(callback, parameter):
        for handle, value in nodes.items():
            if not value["parent"]:
                callback(handle, parameter)
        return True

    def enumerate_children(parent, callback, parameter):
        pending = [parent]
        while pending:
            current = pending.pop()
            for handle, value in nodes.items():
                if value["parent"] == current:
                    callback(handle, parameter)
                    pending.append(handle)
        return True

    def get_pid(handle, pointer):
        pointer._obj.value = nodes[handle]["pid"]
        return 1

    def copy_text(key, handle, buffer, size):
        buffer.value = nodes[handle][key][: size - 1]
        return len(buffer.value)

    def send(handle, message, wparam, lparam, flags, timeout, result):
        calls["sent"].append((handle, message, flags, timeout))
        nodes[handle]["text"] = ctypes.wstring_at(lparam)
        result._obj.value = 1
        return 1

    def post(*arguments):
        calls["posted"].append(arguments)
        return 1

    functions = {
        "EnumWindows": enumerate_windows,
        "EnumChildWindows": enumerate_children,
        "GetWindowThreadProcessId": get_pid,
        "GetClassNameW": lambda *args: copy_text("class", *args),
        "GetWindowTextW": lambda *args: copy_text("text", *args),
        "GetWindow": lambda handle, _kind: nodes[handle]["owner"],
        "GetDlgItem": lambda parent, identifier: next(
            (h for h, n in nodes.items() if n["parent"] == parent and n["id"] == identifier), 0
        ),
        "GetDlgCtrlID": lambda handle: nodes[handle]["id"],
        "IsWindowVisible": lambda handle: nodes[handle]["visible"],
        "SendMessageTimeoutW": send,
        "PostMessageW": post,
    }
    user = SimpleNamespace(**{name: _WinFunction(call) for name, call in functions.items()})
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: user, raising=False)
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", lambda *_args: lambda callback: callback, raising=False)
    monkeypatch.setattr(operations, "os", SimpleNamespace(name="nt", getpid=lambda: 77))
    return calls


def test_win32_nested_filename_aliases_resolve_to_one_edit(monkeypatch, tmp_path):
    calls = _fake_native_picker(monkeypatch)
    result = operations._win32_pick_video(str(tmp_path / "reference.mp4"))
    assert result == {"queued": True}
    assert calls["sent"] == [(5, 0x000C, 0x0002, 1000)]
    assert calls["posted"] == [(2, 0x0111, 1, 6)]


def test_win32_distinct_filename_edits_fail_without_messages(monkeypatch, tmp_path):
    calls = _fake_native_picker(monkeypatch, sibling=True)
    result = operations._win32_pick_video(str(tmp_path / "reference.mp4"))
    assert result["queued"] is False
    assert calls == {"sent": [], "posted": []}


@pytest.mark.parametrize("options", [{"sibling_visible": False}, {"sibling_pid": 88}])
def test_win32_ignores_hidden_or_foreign_filename_edits(monkeypatch, tmp_path, options):
    calls = _fake_native_picker(monkeypatch, sibling=True, **options)
    result = operations._win32_pick_video(str(tmp_path / "reference.mp4"))
    assert result["queued"] is True
    assert calls["sent"][0][0] == 5


def test_mocap_processing_prompt_is_separate_from_download(monkeypatch):
    window = SimpleNamespace(title=lambda: "Mocap")
    monkeypatch.setattr(operations, "_roots", lambda _kind: [])
    monkeypatch.setattr(operations, "_windows", lambda: [window])
    monkeypatch.setattr(operations, "_buttons", lambda _window: [("Yes", object()), ("No", object())])
    monkeypatch.setattr(
        operations,
        "_window_texts",
        lambda _window: [
            "Video-mocap for the selected interval of 31 frames will take approximately 51s. Proceed?"
        ],
    )
    ui = operations._ui_snapshot()
    assert ui["mocap_confirmations"] == [{"title": "Mocap", "frames": 31, "buttons": ["Yes", "No"]}]
    assert ui["download"] == []


def _processing_prompt(frames=7, *, progress=False):
    return {
        "mocap": [{"state": 0}] if progress else [],
        "video": [],
        "apply": [],
        "download": [],
        "mocap_confirmations": [{"title": "Mocap", "frames": frames, "buttons": ["Yes", "No"]}],
    }


def test_authorized_mocap_start_is_confirmed_once_without_download_permission(native, monkeypatch):
    result = plan(native, "mocap", reference_object="plane")
    monkeypatch.setattr(operations, "_ui_snapshot", lambda: _processing_prompt())
    click = Mock()
    monkeypatch.setattr(operations, "_queued_click", click)
    record = {
        "job_id": "owned",
        "plan": result,
        "driver": {"phase": "wait", "log_cursor": {"path": "log", "offset": 123}},
    }
    first = operations.step(record)
    assert first["driver"]["mocap_processing_confirmed"] is True
    second = operations.step({**record, "driver": first["driver"]})
    assert second["safe_to_rollback"] is False
    click.assert_called_once_with("Mocap", "Yes")


def test_mocap_processing_prompt_must_match_the_requested_frame_count(native, monkeypatch):
    result = plan(native, "mocap", reference_object="plane")
    monkeypatch.setattr(operations, "_ui_snapshot", lambda: _processing_prompt(frames=6))
    click = Mock()
    monkeypatch.setattr(operations, "_queued_click", click)
    with pytest.raises(operations.NativeOperationError) as caught:
        operations.step(
            {
                "job_id": "owned",
                "plan": result,
                "driver": {"phase": "wait", "log_cursor": {"path": "log", "offset": 123}},
            }
        )
    assert caught.value.code == "NATIVE_MOCAP_CONFIRMATION_MISMATCH"
    click.assert_not_called()


def test_mocap_cancel_declines_processing_once_and_waits_for_all_windows_to_close(native, monkeypatch):
    result = plan(native, "mocap", reference_object="plane")
    ui = [_processing_prompt()]
    monkeypatch.setattr(operations, "_ui_snapshot", lambda: ui[0])
    click = Mock()
    monkeypatch.setattr(operations, "_queued_click", click)
    record = {
        "job_id": "owned",
        "plan": result,
        "cancel_requested": True,
        "driver": {"phase": "cancelling", "log_cursor": {"path": "log", "offset": 123}},
    }
    first = operations.cancel(record)
    assert first["safe_to_rollback"] is False
    operations.cancel({**record, "driver": first["driver"]})
    click.assert_called_once_with("Mocap", "No")
    ui[0] = {**_processing_prompt(progress=True), "mocap_confirmations": []}
    waiting = operations.cancel({**record, "driver": first["driver"]})
    assert waiting["status"] != "cancelled"
    ui[0]["mocap"] = []
    final = operations.cancel({**record, "driver": waiting["driver"]})
    assert final["status"] == "cancelled"
    assert final["safe_to_rollback"] is True


@pytest.mark.parametrize(
    "phase", ["video_choose", "video_pick", "video_pick_wait", "video_configure", "video_start", "video_wait"]
)
def test_cancel_keeps_owned_importer_progressing_without_inference(native, monkeypatch, phase):
    result = plan(native, "mocap", reference_object="plane")
    operations._callbacks["preprocess"] = {"status": "started"}
    progress = {
        "phase": phase,
        "status": "running",
        "driver": {"phase": phase, "callback_id": "preprocess"},
        "effects": [],
    }
    advance = Mock(return_value=progress)
    monkeypatch.setattr(operations, "step", advance)
    patch = operations.cancel(
        {
            "job_id": "owned",
            "plan": result,
            "cancel_requested": True,
            "driver": {"phase": phase, "callback_id": "preprocess"},
            "effects": [{"kind": "native_action_queued", "action": "View.Bind video"}],
        }
    )
    advance.assert_called_once()
    assert patch["status"] == "running"
    assert "Mocap inference will not start" in patch["message"]
    assert not native.queued


def test_cancel_after_import_callback_returned_stops_before_mocap_dispatch(native):
    result = plan(native, "mocap", reference_object="plane")
    operations._callbacks["importer"] = {"status": "returned"}
    patch = operations.cancel(
        {
            "job_id": "owned",
            "plan": result,
            "cancel_requested": True,
            "driver": {"phase": "dispatch", "callback_id": "importer"},
            "effects": [{"kind": "native_action_queued", "action": "View.Bind video"}],
        }
    )
    assert patch["status"] == "cancelled"
    assert patch["safe_to_rollback"] is True
    assert not native.queued


def test_cancel_finishes_real_import_readback_then_stops_without_inference(native):
    result = plan(native, "mocap", reference_object="plane")
    operations._callbacks["importer"] = {"status": "returned"}
    native.scene.model_viewer = lambda: SimpleNamespace(get_objects=lambda: ["point", "finger", "plane"])
    patch = operations.cancel(
        {
            "job_id": "owned",
            "plan": result,
            "cancel_requested": True,
            "driver": {"phase": "video_wait", "callback_id": "importer"},
            "effects": [{"kind": "native_action_queued", "action": "View.Bind video"}],
        }
    )
    assert patch["status"] == "cancelled"
    assert patch["safe_to_rollback"] is True
    assert patch["driver"]["reference_object"] == "plane"
    assert not native.queued


def test_mocap_dispatch_effect_blocks_false_preprocess_cancellation(native):
    result = plan(native, "mocap", reference_object="plane")
    operations._callbacks["inference"] = {"status": "started"}
    patch = operations.cancel(
        {
            "job_id": "owned",
            "plan": result,
            "cancel_requested": True,
            "driver": {"phase": "dispatch", "callback_id": "inference"},
            "effects": [{"kind": "native_action_queued", "action": operations.ROUTES["mocap"]}],
        }
    )
    assert patch["status"] != "cancelled"
    assert patch["safe_to_rollback"] is False
