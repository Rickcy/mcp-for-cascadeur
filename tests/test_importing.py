import copy
import json
import math
from enum import Enum
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import importing


class NativeId(str):
    def is_null(self):
        return not self


class Mode(Enum):
    Static = 0
    Animation = 1


class Rotation:
    def __init__(self, angle=0.0):
        self.angle = angle

    def to_quaternion(self):
        return self

    def to_euler_angles(self):
        return [[0.0], [0.0], [self.angle]]

    def w(self):
        return math.cos(self.angle / 2)

    def x(self):
        return 0.0

    y = x

    def z(self):
        return math.sin(self.angle / 2)


class NativeSkeleton:
    """Synthetic authored skeleton and Point membership, without vendor assets."""

    def __init__(self, *, rig=True, points=2, count=3):
        self.count, self.frame = count, count - 1
        self.names, self.parents, self.kinds, self.values, self.modes = {}, {}, {}, {}, {}
        self.reads = []
        self.joints = {self.add("pelvis", "Joint"), self.add("knee", "Joint", "pelvis")}
        self.owner = self.add("Rig", "RigInfo") if rig else None
        self.points = (
            {self.add("Point" + str(index), "Point", "Rig") for index in range(points)} if rig else set()
        )
        for item in self.joints:
            for name in importing._JOINT_CHANNELS:
                self.add_channel(item, name)
        for item in self.points:
            self.add_channel(item, "global_position")
        self.info = NativeId("Rig:RigInfo")
        self.model = SimpleNamespace(
            get_objects=lambda: list(self.names),
            get_object_name=self.names.__getitem__,
            get_object_type_name=self.kinds.__getitem__,
        )
        self.behaviour = SimpleNamespace(
            get_behaviour_by_name=self.behaviour_id,
            get_behaviour_object=lambda reference, _: self.parents[NativeId(reference.split(":")[0])],
            get_behaviour_data=lambda reference, name: NativeId(reference.split(":")[0] + ":" + name),
            get_behaviours=lambda name: [self.info] if rig and name == "RigInfo" else [],
            get_behaviour_owner=lambda _: self.owner,
            get_behaviour_objects_range=lambda *_: list(self.joints),
        )
        self.data = SimpleNamespace(
            get_animation_size=lambda: self.count,
            get_data=lambda reference: SimpleNamespace(mode=self.modes[reference]),
            has_data=lambda reference: reference in self.values,
            get_data_value=self.read,
        )
        self.scene = SimpleNamespace(
            model_viewer=lambda: self.model,
            behaviour_viewer=lambda: self.behaviour,
            data_viewer=lambda: self.data,
            get_current_frame=lambda: self.frame,
        )
        self.rig = (
            None
            if not rig
            else SimpleNamespace(
                id=str(self.owner),
                owner=self.owner,
                name="Rig",
                joints=self.joints,
                controls={str(item): item for item in self.points},
                owned=self.joints | self.points | {self.owner},
                native_links=[
                    {
                        "box_id": str(self.owner),
                        "joint": "pelvis",
                        "main_point": "Point0",
                        "direction_point": None,
                    }
                ],
            )
        )

    def add(self, name, kind, parent=None):
        identifier = NativeId(name)
        self.names[identifier], self.kinds[identifier] = name, kind
        self.parents[identifier] = NativeId(parent or "")
        return identifier

    def add_channel(self, item, name):
        reference = NativeId(str(item) + ":" + name)
        self.modes[reference] = Mode.Static if name == "local_scale" else Mode.Animation
        if name.endswith("rotation"):
            self.values[reference] = [Rotation(frame / 10) for frame in range(self.count)]
        elif name == "local_scale":
            self.values[reference] = [1.0, 1.0, 1.0]
        else:
            self.values[reference] = [
                [
                    float(frame) if name == "global_position" or item == "pelvis" else 0.0,
                    10.0 if item == "knee" else 0.0,
                    0.0,
                ]
                for frame in range(self.count)
            ]

    def behaviour_id(self, item, name):
        return NativeId(str(item) + ":" + name) if item in self.names else NativeId("")

    def read(self, reference, *frame):
        self.reads.append((reference, frame))
        if self.modes[reference] == Mode.Static:
            assert frame == (), "Static native getter must use its no-frame overload"
            return self.values[reference]
        assert len(frame) == 1
        return self.values[reference][frame[0]]


@pytest.fixture
def native(monkeypatch):
    skeleton = NativeSkeleton()
    monkeypatch.setattr(importing.characters, "resolve_character", lambda *_args, **_kwargs: skeleton.rig)
    return skeleton


def manifest(native, **kwargs):
    return importing.skeleton_manifest(native.scene, frames=[0, 2], **kwargs)


def test_manifest_is_json_only_with_exact_parent_membership_and_observed_bone_geometry(native):
    observed = manifest(native)
    assert observed["measurements"]["frame"] == 2
    assert not observed["measurements"]["is_rest_pose"]
    assert observed["measurements"]["bone_lengths"]['["pelvis","knee"]'] == 10.0
    joint = observed["joints"]['["pelvis","knee"]']
    assert joint["parent_joint"] == '["pelvis"]'
    assert joint["owned_rigs"] == ['["Rig"]']
    assert observed["character"]["native_links"][0]["main_point"] == '["Rig","Point0"]'
    assert observed["samples"]["2"]["joints"]['["pelvis"]']["global_position"] == [2.0, 0.0, 0.0]
    assert observed["sha256"] == importing.state.digest(
        {key: value for key, value in observed.items() if key != "sha256"}
    )
    json.loads(json.dumps(observed, allow_nan=False))
    assert all(frame == () for reference, frame in native.reads if str(reference).endswith(":local_scale"))


def test_raw_joint_only_source_does_not_require_a_control_rig():
    skeleton = NativeSkeleton(rig=False)
    observed = importing.skeleton_manifest(skeleton.scene, frames=[0, 2])
    assert observed["character"] is None and not observed["points"]
    assert len(observed["joints"]) == 2


def test_incomplete_raw_rig_can_be_inspected_without_claiming_a_finished_character(native, monkeypatch):
    def unsupported(*args, **kwargs):
        raise AssertionError("The raw source must not resolve a finished Point rig")

    monkeypatch.setattr(importing.characters, "resolve_character", unsupported)
    observed = importing.skeleton_manifest(native.scene, capture_character=False)
    assert observed["character"] is None
    assert observed["joints"]['["pelvis"]']["owned_rigs"] == ['["Rig"]']
    with pytest.raises(ValueError, match="requires"):
        importing.skeleton_manifest(native.scene, capture_character=False, require_cascy=True)


@pytest.mark.parametrize("frames", [[True], [3], [-1], [2, 0], [0, 0]])
def test_invalid_sample_frames_are_rejected_before_native_reads(native, frames):
    with pytest.raises(ValueError):
        importing.skeleton_manifest(native.scene, frames=frames)
    assert not native.reads


def test_duplicate_named_paths_and_parent_cycles_fail_closed(native):
    duplicate = native.add("other", "Point", "Rig")
    native.names[duplicate] = "Point0"
    with pytest.raises(ValueError, match="ambiguous"):
        manifest(native)
    native.names[duplicate] = "other"
    native.parents[NativeId("pelvis")] = NativeId("knee")
    with pytest.raises(ValueError, match="cycle"):
        manifest(native)


def test_native_nonfinite_transform_fails_before_a_manifest_can_be_published(native):
    native.values[NativeId("knee:global_position")][2][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        manifest(native)


def test_cascy_requirement_does_not_accept_a_partial_point_rig(native):
    with pytest.raises(ValueError, match="43"):
        manifest(native, require_cascy=True)


def test_exact_import_accepts_same_skeleton_and_ignores_animated_root_location(native):
    target = manifest(native)
    source = copy.deepcopy(target)
    source["character"], source["points"] = None, {}
    source["samples"]["0"]["joints"]['["pelvis"]']["global_position"][0] = 100.0
    source["joints"]['["pelvis"]']["id"] = "different-file-id"
    plan = importing.preflight_exact_import(target, source, expected_points=2)
    assert plan["ready"] and plan["route"] == "exact_skeleton_animation_import"
    assert not plan["rest_pose_verified"]


def test_identity_scene_wrapper_does_not_change_joint_only_import_topology(native):
    target = manifest(native)
    group = native.add("FBX_Group", "Transform")
    reference = NativeId("FBX_Group:global_matrix")
    native.modes[reference] = Mode.Static
    native.values[reference] = [[float(row == column) for column in range(4)] for row in range(4)]
    native.parents[NativeId("pelvis")] = group
    source = manifest(native)
    assert source["joints"]['["pelvis"]']["scene_path"] == ["FBX_Group", "pelvis"]
    assert importing.preflight_exact_import(target, source, expected_points=2)["ready"]
    assert not importing.compare_topology(target, source)["equivalent"]
    assert importing.compare_topology(target, source, require_character=False)["equivalent"]
    native.values[reference][0][0] = 100.0
    scaled_source = manifest(native)
    plan = importing.preflight_exact_import(target, scaled_source, expected_points=2)
    assert not plan["ready"]
    assert any("coordinate basis" in item["reason"] for item in plan["blockers"])


@pytest.mark.parametrize("change", ["parent", "bone_length", "scale", "foreign_joint", "point_count"])
def test_exact_import_rejects_mapping_proportion_and_character_scope_ambiguity(native, change):
    target, source = manifest(native), manifest(native)
    key = '["pelvis","knee"]'
    if change == "parent":
        source["joints"][key]["parent"] = '["other"]'
    elif change == "bone_length":
        source["measurements"]["bone_lengths"][key] = 20.0
    elif change == "scale":
        source["measurements"]["local_scales"][key] = [1.0, 1.1, 1.0]
    elif change == "foreign_joint":
        target["character"]["joint_paths"].remove(key)
    else:
        target["points"].pop(next(iter(target["points"])))
    assert not importing.preflight_exact_import(target, source, expected_points=2)["ready"]


def test_reopen_topology_can_require_native_identity_and_exact_point_links(native):
    before, after = manifest(native), manifest(native)
    after["joints"]['["pelvis"]']["id"] = "other-uuid"
    assert importing.compare_topology(before, after)["equivalent"]
    assert not importing.compare_topology(before, after, require_same_ids=True)["equivalent"]
    after = copy.deepcopy(before)
    after["character"]["native_links"][0]["main_point"] = '["Rig","Point1"]'
    assert not importing.compare_topology(before, after)["equivalent"]
    after = copy.deepcopy(before)
    after["points"]['["Rig","Point0"]']["position_data"]["mode"] = "Static"
    assert not importing.compare_topology(before, after)["equivalent"]


def test_equal_pose_samples_survive_quaternion_sign_and_report_point_drift(native):
    before, after = manifest(native), manifest(native)
    rotation = after["samples"]["0"]["joints"]['["pelvis"]']["local_rotation"]
    rotation["quaternion_wxyz"] = [-item for item in rotation["quaternion_wxyz"]]
    assert importing.compare_samples(before, after)["equivalent"]
    after["samples"]["2"]["points"]['["Rig","Point0"]'][0] += 0.1
    report = importing.compare_samples(before, after)
    assert not report["equivalent"] and report["max_errors"]["position"] == pytest.approx(0.1)
    assert importing.compare_samples(before, after, include_points=False)["equivalent"]


def test_euler_winding_is_not_discarded_when_quaternion_orientation_is_equal(native):
    before, after = manifest(native), manifest(native)
    after["samples"]["0"]["joints"]['["pelvis"]']["local_rotation"]["euler_radians"][2] += 2 * math.pi
    report = importing.compare_samples(before, after)
    assert not report["equivalent"] and report["max_errors"]["euler_radians"] == pytest.approx(2 * math.pi)
    assert importing.compare_samples(before, after, preserve_euler=False)["equivalent"]


def test_frame_count_empty_samples_and_mismatched_frame_evidence_cannot_pass(native):
    before = manifest(native)
    after = copy.deepcopy(before)
    after["stored_frames"] += 1
    assert not importing.compare_samples(before, after)["equivalent"]
    assert importing.compare_samples(before, after, require_same_count=False)["equivalent"]
    after = copy.deepcopy(before)
    after["sample_frames"] = [0]
    assert not importing.compare_samples(before, after)["equivalent"]
    empty = importing.skeleton_manifest(native.scene)
    assert not importing.compare_samples(empty, empty)["equivalent"]


def test_sample_rotation_and_scale_mismatch_are_separate_metrics(native):
    before, after = manifest(native), manifest(native)
    joint = after["samples"]["0"]["joints"]['["pelvis"]']
    joint["local_rotation"] = importing._rotation(Rotation(0.05))
    joint["local_scale"][0] = 1.1
    report = importing.compare_samples(before, after)
    assert report["max_errors"]["rotation_radians"] == pytest.approx(0.05)
    assert report["max_errors"]["scale"] == pytest.approx(0.1)
    assert not report["equivalent"]


def test_large_sampling_and_misleading_tolerances_fail_closed(native):
    with pytest.raises(ValueError, match="at most"):
        importing.skeleton_manifest(native.scene, frames=list(range(65)))
    observed = manifest(native)
    with pytest.raises(ValueError, match="finite"):
        importing.compare_samples(observed, observed, position_tolerance=float("nan"))
    with pytest.raises(ValueError, match="expected_points"):
        importing.preflight_exact_import(observed, observed, expected_points=True)
