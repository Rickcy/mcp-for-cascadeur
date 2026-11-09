"""Isolated Point verification scales by batch samples and protects authored data."""

import importlib.util
import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("isolated_point_test", ROOT / "integrations/kimodo/point_retarget.py")
points = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(points)


def guard_fixture():
    frames = list(range(520, 540))
    reads, values = [], {"finger": 0.5, "outside": 3.0, "static": 1.0}
    channels = {"character": ["derived", "finger", "static"], "outside-object": ["outside"]}

    def value(identifier, frame=None):
        if frame is not None:
            assert frame in frames, "The batch guard sampled animation outside its bounded edit range"
            assert identifier != "derived", "The solver's derived channel must not become a full-clip checkpoint"
            reads.append((identifier, frame))
        return values[identifier]

    data = SimpleNamespace(
        get_animation_size=lambda: 100_000,
        get_all_data_id=lambda obj: channels[obj], get_all_settings_id=lambda _obj: [],
        get_data=lambda key: SimpleNamespace(name=key, mode=SimpleNamespace(name="Static" if key == "static" else "Animation")),
        get_setting=lambda _key: None, get_data_value=value, get_setting_value=value,
    )
    model = SimpleNamespace(get_objects=lambda: list(channels), get_object_name=lambda obj: obj,
                            get_object_type_name=lambda _obj: "Point")
    layer = SimpleNamespace(header=SimpleNamespace(name="body", parent="root"), obj_ids=["character"],
                            is_locked=False, is_visible=True, key_frame_indices=lambda: frames, sections={})
    scene = SimpleNamespace(data_viewer=lambda: data, model_viewer=lambda: model,
                            layers_viewer=lambda: SimpleNamespace(all_layer_ids=lambda: ["track"], layer=lambda _id: layer))
    return scene, SimpleNamespace(owned={"character"}), frames, reads, values


def test_guard_reads_protected_samples_only_for_the_current_twenty_frames():
    scene, rig, frames, reads, _ = guard_fixture()
    result = points._batch_guard(scene, rig, frames, {"finger"})
    assert result["protected_samples"] == 40
    assert len(reads) == 40
    assert {frame for _, frame in reads} == set(frames)


def test_guard_detects_authored_finger_changes_and_static_changes_separately():
    scene, rig, frames, _, values = guard_fixture()
    before = points._batch_guard(scene, rig, frames, {"finger"})
    values["finger"] += 0.1
    after = points._batch_guard(scene, rig, frames, {"finger"})
    assert before["protected_sha256"] != after["protected_sha256"]
    assert before["metadata_sha256"] == after["metadata_sha256"]
    values["static"] += 1
    assert after["metadata_sha256"] != points._batch_guard(scene, rig, frames, {"finger"})["metadata_sha256"]


@pytest.mark.parametrize("delta,accepted", [(1e-7, True), (0.02, False)])
def test_protected_animated_values_use_existing_native_numeric_tolerances(delta, accepted):
    scene, rig, frames, _, values = guard_fixture()
    before = points._batch_guard(scene, rig, frames, {"finger"}, capture_protected=True)
    values["finger"] += delta
    after = points._batch_guard(scene, rig, frames, {"finger"}, capture_protected=True)
    report = points._compare_batch_guards(before, after, frames)
    assert (not report["mismatches"]) is accepted
    assert report["changed_numeric_samples"] == len(frames)
    if not accepted:
        assert report["mismatches"][0]["channel"] == "data:finger"
        assert report["mismatches"][0]["frame"] == frames[0]


def test_static_metadata_remains_exact_when_animated_numeric_tolerances_are_used():
    scene, rig, frames, _, values = guard_fixture()
    before = points._batch_guard(scene, rig, frames, {"finger"}, capture_protected=True)
    values["static"] += 1e-7
    after = points._batch_guard(scene, rig, frames, {"finger"}, capture_protected=True)
    with pytest.raises(RuntimeError, match="curve/static metadata"):
        points._compare_batch_guards(before, after, frames)


def test_animated_settings_are_exact_even_when_data_samples_have_numeric_tolerances():
    scene, rig, frames, _, values = guard_fixture()
    before = points._batch_guard(scene, rig, frames, {"finger"}, capture_protected=True)
    values["finger"] += 1e-7
    after = points._batch_guard(scene, rig, frames, {"finger"}, capture_protected=True)
    before["protected_values"]["data:finger"]["kind"] = "setting"
    after["protected_values"]["data:finger"]["kind"] = "setting"
    assert points._compare_batch_guards(before, after, frames)["mismatches"]


def test_native_conversion_accepts_long_world_travel_and_rejects_overflow_before_writing():
    converted = points._float32_targets([(520, {"point": [20000, 100, 0]})])
    assert converted[0][1]["point"].tolist() == [20000, 100, 0]
    with pytest.raises(ValueError, match="finite native float32"):
        points._float32_targets([(520, {"point": [1e100, 100, 0]})])


@pytest.mark.parametrize("orientation_changed", [False, True])
def test_protected_rotation_checks_orientation_and_reports_euler_branch_changes(orientation_changed):
    scene, rig, frames, _, _ = guard_fixture()
    before = points._batch_guard(scene, rig, frames, {"finger"}, capture_protected=True)
    after = points._batch_guard(scene, rig, frames, {"finger"}, capture_protected=True)
    old = {"__native__": "rotation", "quaternion_wxyz": [1, 0, 0, 0], "euler_radians": [0, 0, 0]}
    new = {"__native__": "rotation", "quaternion_wxyz": [1, 0, 0, 0], "euler_radians": [math.pi] * 3}
    if orientation_changed:
        new["quaternion_wxyz"] = [0.5, math.sqrt(0.75), 0, 0]
    before["protected_values"]["data:finger"]["values"] = [old] * len(frames)
    after["protected_values"]["data:finger"]["values"] = [new] * len(frames)
    report = points._compare_batch_guards(before, after, frames)
    assert bool(report["mismatches"]) is orientation_changed
    assert report["euler_representation_changes"] == (0 if orientation_changed else len(frames))


@pytest.mark.parametrize("miswired", [False, True])
def test_auxiliary_joint_exemption_requires_both_actual_native_graph_connections(tmp_path, miswired):
    description = {"Name": "custom_aux", "Path": []}
    document = {"objects": [
        {"id": "main", "name": "custom_body", "type_name": "Joint", "behaviours": [{"id": "main-ref", "type": "Joint"}]},
        {"id": "aux", "name": "custom_aux", "type_name": "Joint", "behaviours": []},
        {"id": "box", "name": "controller", "type_name": "BasicObject", "behaviours": [
            {"type": "RigAdditionalInfo", "id": "box-ref", "joint": "main-ref"}]},
        {"id": "rig", "name": "Character", "type_name": "Tool", "behaviours": [{
            "type": "RigInfo", "id": "rig-ref", "related_joints": ["main", "aux"], "rig_objects": ["box"],
            "proto_json": json.dumps({"Units": [{"MainJoint": {"Name": "custom_body", "Path": []}, "AdditionalJoint": description}]})}]},
    ], "update": {"objects": [{"object": "box", "group": "from"}, {"object": "aux", "group": "to"}],
        "groups": [{"id": "from", "outputs": [{"id": "pos", "name": "Position Additional Joint"},
                                                {"id": "rot", "name": "Rotation Additional Joint"}]},
                   {"id": "to", "inputs": [{"id": "pos", "name": "Position"}, {"id": "rot", "name": "Rotation"}]}],
        "connections": [{"source": {"group_id": "from", "attribute_id": channel},
                         "destination": {"group_id": "bad-target" if miswired and channel == "rot" else "to", "attribute_id": channel}}
                        for channel in ("pos", "rot")]}}
    path = tmp_path / "before.casc"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("scene/scene.json", json.dumps(document))
    plan = {"character_id": "rig", "target_mapping": {"hand_l": "main"}}
    if miswired:
        with pytest.raises(ValueError, match="unique body-controller Rotation"):
            points._verified_auxiliary_bindings(path, plan)
    else:
        result = points._verified_auxiliary_bindings(path, plan)
        assert len(result) == 1 and result[0]["joint"] == "aux" and result[0]["box"] == "box"


@pytest.mark.parametrize("drift", [0, 0.5])
def test_graph_bound_auxiliary_joint_must_follow_the_actual_moving_controller(monkeypatch, drift):
    rig = SimpleNamespace(owned={"box", "aux"})
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def pose(_scene, obj, frame):
        return ([frame * 10 + (1 + drift if obj == "aux" else 0), 0, 0], identity)

    monkeypatch.setattr(points, "_pose", pose)
    bindings = [{"joint": "aux", "box": "box"}]
    reference = {"aux": {"position": [1, 0, 0], "rotation": identity}}
    if drift:
        with pytest.raises(RuntimeError, match="stopped following"):
            points._verify_auxiliary("scene", rig, bindings, reference, range(20))
    else:
        result = points._verify_auxiliary("scene", rig, bindings, reference, range(20))
        assert result["frames_checked"] == 20 and result["max_position_error_cm"] == 0


def track_fixture(*, unrelated_child=False, unrelated_track_member=False):
    class NativeId(str):
        def is_null(self):
            return not self

    objects = {key: NativeId(key) for key in ("joint", "mesh", "proxy", "rig")}
    parents = {"mesh": "proxy"}
    types = {"joint": "Joint", "mesh": "Mesh Object", "proxy": "Transform", "rig": "Tool"}
    membership = [objects[key] for key in ("joint", "mesh", "proxy")]
    if unrelated_child or unrelated_track_member:
        objects["foreign"] = NativeId("foreign")
        types["foreign"] = "Joint"
        if unrelated_child:
            parents["foreign"] = "proxy"
        if unrelated_track_member:
            membership.append(objects["foreign"])

    def behaviour(obj, name):
        exists = name == "Basic" or (name == "MeshObject" and obj == "mesh")
        return SimpleNamespace(owner=obj, is_null=lambda: not exists)

    own_info = SimpleNamespace(owner="rig")
    viewer = SimpleNamespace(get_behaviour_by_name=behaviour,
        get_behaviour_object=lambda ref, _field: NativeId(parents.get(ref.owner, "")),
        get_behaviour_objects_range=lambda _ref, _field: {objects["joint"]},
        get_behaviours=lambda _name: [own_info], get_behaviour_owner=lambda ref: ref.owner)
    model = SimpleNamespace(get_objects=lambda: list(objects.values()), get_object_type_name=lambda obj: types[obj])
    layer = SimpleNamespace(obj_ids=membership, is_locked=False, sections={})
    layers = SimpleNamespace(all_layer_ids=lambda: ["torso"], layer=lambda _identifier: layer)
    scene = SimpleNamespace(behaviour_viewer=lambda: viewer, model_viewer=lambda: model, layers_viewer=lambda: layers)
    rig = SimpleNamespace(id="rig", owned={objects[key] for key in ("joint", "mesh", "rig")},
                          joints={objects["joint"]})
    return scene, rig


def test_selected_skin_container_is_allowed_without_expanding_writable_ownership():
    scene, rig = track_fixture()
    before = set(rig.owned)
    assert set(points._isolated_tracks(scene, rig)) == {"torso"}
    assert rig.owned == before and "proxy" not in rig.owned


@pytest.mark.parametrize("field", ["unrelated_child", "unrelated_track_member"])
def test_skin_whitelist_does_not_admit_other_characters_or_unrelated_track_objects(field):
    scene, rig = track_fixture(**{field: True})
    with pytest.raises(ValueError, match="unrelated"):
        points._isolated_tracks(scene, rig)


@pytest.mark.parametrize("behaviour_name", ["Twist", "Untwist"])
def test_only_native_linked_twist_channels_are_solver_derived(monkeypatch, behaviour_name):
    from mcp_for_cascadeur.host import characters

    rig = SimpleNamespace(joints={"pelvis", "twist", "finger"}, owned={"pelvis", "twist", "finger", "twist-box"},
                          native_links=[{"box_id": "twist-box", "joint": "twist"}])
    viewer = SimpleNamespace(get_behaviour_by_name=lambda _box, name:
                             SimpleNamespace(is_null=lambda: name != behaviour_name))
    scene = SimpleNamespace(behaviour_viewer=lambda: viewer)
    monkeypatch.setattr(characters, "_parent", lambda _scene, _obj: None)
    monkeypatch.setattr(characters, "_transform_data", lambda _scene, obj, channel: obj + ":" + channel)
    protected = points._protected_local_channels(scene, rig, {"pelvis": "pelvis"})
    assert protected == {"finger:" + channel for channel in ("local_position", "local_rotation", "local_scale")}


def test_uncertain_native_batch_is_quarantined_without_advancing_or_replaying(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"point_bindings": {"pelvis_MainPoint": "native-point"}}))
    receipt = tmp_path / "session.json"
    record = {"schema": "kimodo-point-session-v1", "directory": str(tmp_path),
              "phase": "points_ready:20", "next_frame": 20, "events": [],
              "plan_path": str(plan), "plan_sha256": points._sha(plan)}
    points._write(receipt, record)
    calls = []

    def native_write(current):
        calls.append(1)
        current["events"][-1]["before_point_targets"] = [{"frame": 20, "points": {"native-point": [0, 1, 0]}}]
        raise RuntimeError("Native verification failed after committing a batch")

    try:
        points._stage(receipt, "points_ready:20", "points_ready:40", native_write)
    except RuntimeError:
        pass
    persisted = json.loads(receipt.read_text())
    assert len(calls) == 1 and persisted["next_frame"] == 20
    assert persisted["phase"] == "needs_review:points_ready:40"
    assert persisted["events"][-1]["before_point_targets"][0]["frame"] == 20


@pytest.mark.parametrize("last_point_error", [0, 4])
def test_final_save_checks_last_frame_and_keeps_source_immutable(tmp_path, monkeypatch, last_point_error):
    from mcp_for_cascadeur.host import curves

    before = tmp_path / "before.casc"
    before.write_bytes(b"immutable user source copy")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"frames": [{"frame": frame, "points": {"pelvis_MainPoint": [0, 1, frame]}}
                                          for frame in range(3)]}))
    receipt = tmp_path / "session.json"
    points._write(receipt, {"schema": "kimodo-point-session-v1", "directory": str(tmp_path),
        "phase": "points_applied", "writer": "isolated-point-batch-v1", "num_frames": 3,
        "next_frame": 3, "events": [], "point_bindings": {"pelvis_MainPoint": "point"},
        "plan_path": str(plan), "plan_sha256": points._sha(plan), "before_sha256": points._sha(before)})
    saved = []

    def save(path):
        saved.append(path)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("scene/scene_settings.json", json.dumps({"Data": {"FPS": 30, "UnitLength": "cm"}}))

    scene = SimpleNamespace(set_current_frame=lambda _frame: None)
    view = SimpleNamespace(animation_boundary=lambda: SimpleNamespace(), domain_scene=lambda: scene, save=save)
    monkeypatch.setattr(points, "_view", lambda _record: view)
    observed = [[0, 1, frame] for frame in range(3)]
    observed[-1][2] += last_point_error
    monkeypatch.setattr(points, "snapshot", lambda *_args: {"positions": {"pelvis_MainPoint": observed}})
    monkeypatch.setattr(points, "resolve_rig", lambda *_args: "rig")
    geometry_frames = []

    def geometry(_scene, _rig, *, frames):
        geometry_frames.extend(frames)
        return {"max_connection_anchor_error": 0, "connection_count": 20}

    monkeypatch.setattr(curves, "connection_geometry", geometry)
    if last_point_error:
        with pytest.raises(RuntimeError, match="Final clip Point deviation"):
            points.save(receipt)
        assert not saved
        assert json.loads(receipt.read_text())["phase"] == "needs_review:points_saved"
    else:
        points.save(receipt)
        persisted = json.loads(receipt.read_text())
        assert len(saved) == 1 and geometry_frames == [0, 1, 2]
        assert persisted["clip_verification"]["all_point_targets_verified"]
        assert persisted["clip_verification"]["frames_checked"] == 3
    assert before.read_bytes() == b"immutable user source copy"
