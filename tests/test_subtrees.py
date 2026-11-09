import copy
import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_for_cascadeur.host import animation, common, curves, state, subtrees


class NativeId(str):
    def is_null(self):
        return not self


class NativeRotation(list):
    """A native-style rotation whose Euler winding survives quaternion equality."""

    def __init__(self, angle=0.0):
        self.angle = angle
        super().__init__([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])

    def to_quaternion(self):
        return self

    def to_euler_angles(self):
        return [self.angle, 0.0, 0.0]

    def w(self):
        return self[0]

    def x(self):
        return self[1]

    def y(self):
        return self[2]

    def z(self):
        return self[3]


class NativeFk:
    """Synthetic independent FK hierarchy with actual per-frame derived channels."""

    def __init__(self, count=4):
        self.count = count
        self.objects = list(map(NativeId, ("root", "child", "other")))
        self.parents = {
            self.objects[0]: NativeId(""),
            self.objects[1]: self.objects[0],
            self.objects[2]: NativeId(""),
        }
        self.kinds = {item: "Joint" for item in self.objects}
        self.names = {item: str(item) for item in self.objects}
        self.behaviors = {
            item: {
                name: NativeId(f"{item}:{name}")
                for name in ("Basic", "Transform", "Joint", "Node3d", "FbxExport", "JointExtraAttributes")
            }
            for item in self.objects
        }
        self.rig_joints = set()
        self.modes = {}
        self.values = {}
        for index, item in enumerate(self.objects):
            for field in ("local_position", "global_position", "local_rotation", "global_rotation"):
                reference = NativeId(f"{item}/{field}")
                self.modes[reference] = SimpleNamespace(name="Animation")
                self.values[reference] = [
                    [float(frame * (index + 1)), index * 10.0, 0.0]
                    if field.endswith("position")
                    else NativeRotation()
                    for frame in range(count)
                ]
            reference = NativeId(f"{item}/local_scale")
            self.values[reference] = [[1.0, 1.0, 1.0]] * count
            self.modes[reference] = SimpleNamespace(name="Static")
        self.tracks = {}
        for item in self.objects:
            self.tracks[NativeId("track:" + item)] = SimpleNamespace(
                header=SimpleNamespace(name=str(item), parent=NativeId("folder")),
                obj_ids={item},
                is_locked=False,
                key_frame_indices=lambda: [0, self.count - 1],
                is_key=lambda frame: frame in (0, self.count - 1),
                sections={0: self.section(), count - 1: self.section()},
                cycles=False,
            )
        self.model = SimpleNamespace(
            get_objects=lambda: self.objects,
            get_object_name=self.names.__getitem__,
            get_object_type_name=self.kinds.__getitem__,
            data_editor=lambda: self.editor,
        )
        self.viewer = SimpleNamespace(
            get_behaviour_by_name=lambda item, name: self.behaviors[item].get(name, NativeId("")),
            get_behaviour_object=lambda behavior, name: self.parents[NativeId(behavior.split(":")[0])],
            get_behaviours=lambda value: (
                ([NativeId("rig")] if self.rig_joints else [])
                if value == "RigInfo"
                else list(self.behaviors[value].values())
            ),
            get_behaviour_objects_range=lambda *_: self.rig_joints,
            get_behaviour_name=lambda behavior: behavior.split(":")[1],
            get_behaviour_data=lambda behavior, name: NativeId(behavior.split(":")[0] + "/" + name),
        )
        self.layers = SimpleNamespace(
            layer_id_by_obj_id_or_null=lambda item: NativeId("track:" + item), layer=self.tracks.__getitem__
        )
        self.data = SimpleNamespace(
            get_animation_size=lambda: self.count,
            get_data_value=self.get_data_value,
            has_data=lambda reference: reference in self.values,
            get_data=lambda reference: SimpleNamespace(mode=self.modes[reference]),
        )
        self.scene = SimpleNamespace(
            model_viewer=lambda: self.model,
            behaviour_viewer=lambda: self.viewer,
            data_viewer=lambda: self.data,
            layers_viewer=lambda: self.layers,
            get_current_frame=lambda: 0,
        )
        self.view = SimpleNamespace(
            name=lambda: "Independent FK", get_path_name=lambda: "C:/fixtures/fk.casc"
        )
        self.editor = SimpleNamespace(set_data_value=self.set_value)
        self.updater = SimpleNamespace(
            run_update=lambda changed, frame: self.derive(frame),
            get_interpolator=lambda: SimpleNamespace(reload=lambda: None, interpolate=self.interpolate),
        )
        for frame in range(count):
            self.derive(frame)

    @staticmethod
    def section():
        common_flags = SimpleNamespace(
            ik_fk=SimpleNamespace(name="FK"), fixation=SimpleNamespace(name="FREE")
        )
        return SimpleNamespace(
            interval=SimpleNamespace(
                interpolation=SimpleNamespace(name="LINEAR"),
                common=common_flags,
                description_for_ai_interpolation="",
            ),
            key=SimpleNamespace(
                common=common_flags,
                label=None,
                tangents=SimpleNamespace(name="AUTO"),
                left_weight_velocity=1.0,
                right_weight_velocity=1.0,
            ),
        )

    def set_value(self, reference, frame, value):
        self.values[reference][frame] = copy.deepcopy(value)

    def get_data_value(self, reference, *frames):
        if self.modes[reference].name == "Static":
            assert not frames, "Static native scale must be read without a frame"
            return self.values[reference][0]
        assert len(frames) == 1
        return self.values[reference][frames[0]]

    def derive(self, frame):
        for item in self.objects:
            local = self.values[NativeId(item + "/local_position")][frame]
            parent = self.parents[item]
            offset = self.values[NativeId(parent + "/global_position")][frame] if parent else [0, 0, 0]
            self.values[NativeId(item + "/global_position")][frame] = [a + b for a, b in zip(local, offset)]
            self.values[NativeId(item + "/global_rotation")][frame] = list(
                self.values[NativeId(item + "/local_rotation")][frame]
            )

    def interpolate(self):
        end = self.count - 1
        for item in self.objects:
            values = self.values[NativeId(item + "/local_position")]
            first, last = list(values[0]), list(values[end])
            for frame in range(1, end):
                values[frame] = [a + (b - a) * frame / end for a, b in zip(first, last)]
            rotations = self.values[NativeId(item + "/local_rotation")]
            if all(isinstance(value, NativeRotation) for value in rotations):
                first_angle, last_angle = rotations[0].angle, rotations[end].angle
                for frame in range(1, end):
                    rotations[frame] = NativeRotation(first_angle + (last_angle - first_angle) * frame / end)
        for frame in range(self.count):
            self.derive(frame)

    def read(self, scene, item, frame, space):
        return {
            "object": str(item),
            "name": self.names[item],
            "position": list(self.values[NativeId(item + "/" + space + "_position")][frame]),
            "rotation_wxyz": list(self.values[NativeId(item + "/" + space + "_rotation")][frame]),
        }

    def capture(self, scene):
        channels = {}
        for reference, values in self.values.items():
            mode = self.modes[reference].name
            channels[str(reference)] = {
                "owner": reference.split("/")[0],
                "id": str(reference),
                "name": str(reference),
                "kind": "data",
                "mode": mode,
                **({"values": copy.deepcopy(values)} if mode == "Animation" else {"value": list(values[0])}),
            }
        return {
            "format_version": 1,
            "count": self.count,
            "objects": sorted(map(str, self.objects)),
            "channels": channels,
            "tracks": {
                str(key): {
                    "objects": sorted(map(str, value.obj_ids)),
                    "locked": value.is_locked,
                    "sections": {
                        str(frame): state.section_state(section) for frame, section in value.sections.items()
                    },
                }
                for key, value in self.tracks.items()
            },
            "auxiliary": {},
            "external": {},
        }


@pytest.fixture
def native(monkeypatch, tmp_path, request):
    fixture = NativeFk(getattr(request, "param", 4))
    subtrees._snapshots.clear()
    common.configure(tmp_path)
    fake_csc = SimpleNamespace(
        math=SimpleNamespace(Rotation=SimpleNamespace(from_euler=lambda x, y, z: NativeRotation(x))),
        layers=SimpleNamespace(
            CyclesViewer=lambda track: SimpleNamespace(any_cycles_exist_in_frames=lambda *_: track.cycles)
        ),
    )
    monkeypatch.setitem(sys.modules, "csc", fake_csc)
    monkeypatch.setattr(
        common, "current_context", lambda require_scene=True: (fake_csc, None, fixture.view, fixture.scene)
    )
    monkeypatch.setattr(
        common,
        "resolve_object",
        lambda scene, value: next(
            item for item in fixture.objects if value in (str(item), fixture.names[item])
        ),
    )
    monkeypatch.setattr(animation, "_read", fixture.read)
    monkeypatch.setattr(subtrees, "_native_value", lambda csc, template, field, value: list(value))
    monkeypatch.setattr(
        common, "modify", lambda scene, label, callback: callback(fixture.model, fixture.updater)
    )
    monkeypatch.setattr(state, "capture", fixture.capture)
    monkeypatch.setattr(curves, "checkpoint_before", fixture.capture)
    return fixture


def test_discovers_real_descendants_and_reads_without_playhead_mutation(native):
    result = subtrees.subtree_pose_read("root", frame=2)
    assert list(result["joints"]) == ["root", "child"]
    assert result["joints"]["child"]["parent_id"] == "root"
    assert result["joints"]["child"]["global"]["position"] == [6, 10, 0]


@pytest.mark.parametrize(
    "defect,match", [("cycle", "cycle"), ("nonjoint", "native Joint"), ("duplicate", "unique")]
)
def test_subtree_hierarchy_rejection(native, defect, match):
    if defect == "cycle":
        native.parents[NativeId("root")] = NativeId("child")
    elif defect == "nonjoint":
        native.kinds[NativeId("child")] = "Point"
    else:
        native.names[NativeId("other")] = "child"
    with pytest.raises(ValueError, match=match):
        subtrees.subtree_snapshot("root")


@pytest.mark.parametrize(
    "defect,match",
    [
        ("driven", "RigInfo-driven"),
        ("locked", "unlocked"),
        ("shared", "exactly"),
        ("scale", "unit local scale"),
        ("ik", "FK intervals"),
        ("cycle", "Cyclic"),
        ("static", "authored animated"),
        ("constraint", "constraint behaviors"),
    ],
)
def test_authored_independent_fk_guard_before_any_write(native, monkeypatch, defect, match):
    if defect == "driven":
        native.rig_joints.add(NativeId("root"))
    elif defect == "locked":
        native.tracks[NativeId("track:root")].is_locked = True
    elif defect == "shared":
        native.tracks[NativeId("track:root")].obj_ids.add(NativeId("other"))
    elif defect == "scale":
        native.values[NativeId("root/local_scale")][0] = [1, 2, 1]
    elif defect == "ik":
        native.tracks[NativeId("track:root")].sections[0].interval.common.ik_fk.name = "IK"
    elif defect == "cycle":
        native.tracks[NativeId("track:root")].cycles = True
    elif defect == "static":
        native.modes[NativeId("root/local_rotation")].name = "Static"
    else:
        native.behaviors[NativeId("root")]["Constraint"] = NativeId("root:Constraint")
    mutation = Mock()
    monkeypatch.setattr(common, "modify", mutation)
    with pytest.raises(ValueError, match=match):
        subtrees.subtree_pose_apply("root", [{"object": "root", "position": [10, 20, 30]}])
    mutation.assert_not_called()
    assert not subtrees._snapshots


def test_full_clip_restore_preserves_edits_made_after_snapshot_outside_subtree(native):
    original = copy.deepcopy(native.values)
    snapshot = subtrees.subtree_snapshot("root")
    edited = subtrees.subtree_pose_apply("root", [{"object": "root", "position": [10, 20, 30]}])
    assert edited["verification"]["matches"]
    assert (
        native.values[NativeId("child/global_position")][1] != original[NativeId("child/global_position")][1]
    )
    subtrees.subtree_pose_apply("other", [{"object": "other", "position": [80, 90, 100]}])
    protected = {
        key: copy.deepcopy(value) for key, value in native.values.items() if key.startswith("other/")
    }
    result = subtrees.subtree_restore(snapshot_id=snapshot["snapshot_id"])
    assert result["frames_verified"] == 4
    assert result["spaces_verified"] == ["local", "global"]
    for key, values in native.values.items():
        assert values == (protected[key] if key.startswith("other/") else original[key])
    assert result["recovery_snapshot_id"] in subtrees._snapshots


@pytest.mark.parametrize("native", [6], indirect=True)
def test_v2_whole_clip_restore_retains_full_turns_and_future_linear_interpolation(native):
    reference = NativeId("root/local_rotation")
    expected = [2 * math.pi * frame / 5 for frame in range(6)]
    native.values[reference] = [NativeRotation(angle) for angle in expected]
    native.interpolate()
    snapshot = subtrees.subtree_snapshot("root")
    stored = subtrees._snapshots[snapshot["snapshot_id"]]
    assert stored["version"] == 2
    assert [
        frame["root"]["local_rotation_state"]["euler_radians"][0] for frame in stored["frames"]
    ] == expected
    native.values[reference] = [NativeRotation(0.3 * frame / 5) for frame in range(6)]
    native.interpolate()
    subtrees.subtree_pose_apply("other", [{"object": "other", "position": [80, 90, 100]}])
    unrelated = copy.deepcopy(native.values[NativeId("other/local_position")])
    result = subtrees.subtree_restore(snapshot_id=snapshot["snapshot_id"])
    assert [value.angle for value in native.values[reference]] == expected
    assert result["frames_verified"] == 6
    assert result["local_euler_fidelity"] is True
    assert result["max_local_euler_error_radians"] == 0
    assert native.values[NativeId("other/local_position")] == unrelated
    # Rebuilding LINEAR inbetweens later must still retain the complete turn.
    native.interpolate()
    assert [value.angle for value in native.values[reference]] == expected


def test_legacy_v1_whole_clip_restore_is_readable_and_reports_absent_euler_fidelity(native):
    identifier = subtrees.subtree_snapshot("root")["snapshot_id"]
    record = subtrees._snapshots[identifier]
    record["version"] = 1
    for frame in record["frames"]:
        for joint in frame.values():
            del joint["local_rotation_state"]
    record["sha256"] = subtrees._digest({key: value for key, value in record.items() if key != "sha256"})
    result = subtrees.subtree_restore(snapshot_id=identifier)
    assert result["format_version"] == 1
    assert result["local_euler_fidelity"] is False
    assert "cannot be recovered" in result["compatibility_note"]


def test_v2_missing_native_euler_sample_is_rejected_before_mutation(native, monkeypatch):
    identifier = subtrees.subtree_snapshot("root")["snapshot_id"]
    record = subtrees._snapshots[identifier]
    del record["frames"][1]["child"]["local_rotation_state"]["euler_radians"]
    record["sha256"] = subtrees._digest({key: value for key, value in record.items() if key != "sha256"})
    mutation = Mock()
    monkeypatch.setattr(common, "modify", mutation)
    with pytest.raises(ValueError, match="Euler rotation state"):
        subtrees.subtree_restore(snapshot_id=identifier)
    mutation.assert_not_called()


def test_durable_snapshot_rejects_checksum_and_rechecks_identity(native, tmp_path, monkeypatch):
    target = tmp_path / "pose.json"
    saved = subtrees.subtree_snapshot("root", path=str(target))
    subtrees._snapshots.clear()
    assert subtrees.subtree_restore(path=str(target))["restored"]
    native.names[NativeId("child")] = "renamed_child"
    mutation = Mock()
    monkeypatch.setattr(common, "modify", mutation)
    with pytest.raises(ValueError, match="topology"):
        subtrees.subtree_restore(path=str(target))
    mutation.assert_not_called()
    target.write_text(target.read_text().replace(saved["sha256"], "0" * 64))
    with pytest.raises(ValueError, match="checksum"):
        subtrees.subtree_restore(path=str(target))


def test_snapshot_scope_cannot_be_truncated_even_with_new_checksum(native):
    identifier = subtrees.subtree_snapshot("root")["snapshot_id"]
    record = subtrees._snapshots[identifier]
    del record["frames"][1]["child"]
    record["sha256"] = subtrees._digest({key: value for key, value in record.items() if key != "sha256"})
    with pytest.raises(ValueError, match="every subtree Joint"):
        subtrees.subtree_restore(snapshot_id=identifier)


def test_restore_rejects_other_scene_before_mutation(native, monkeypatch):
    identifier = subtrees.subtree_snapshot("root")["snapshot_id"]
    native.view.get_path_name = lambda: "C:/other.casc"
    mutation = Mock()
    monkeypatch.setattr(common, "modify", mutation)
    with pytest.raises(ValueError, match="different native scene"):
        subtrees.subtree_restore(snapshot_id=identifier)
    mutation.assert_not_called()


def test_terminal_pose_write_is_rejected_while_terminal_read_remains_available(native, monkeypatch):
    assert subtrees.subtree_pose_read("root", frame=3)["frame"] == 3
    mutation = Mock()
    monkeypatch.setattr(common, "modify", mutation)
    with pytest.raises(ValueError, match="no allocated successor"):
        subtrees.subtree_pose_apply("root", [{"object": "root", "position": [1, 2, 3]}], frame=3)
    mutation.assert_not_called()


def test_float32_quaternion_norm_does_not_create_a_false_rotation_difference():
    import math

    reference = [math.cos(0.15), 0.0, 0.0, math.sin(0.15)]
    encoded = {
        "__native__": "rotation",
        "euler_radians": [0.0, 0.0, 0.3],
        "quaternion_wxyz": [value * 0.99999994 for value in reference],
    }
    subtrees._validate_rotation_state(encoded, reference)
    different = [math.cos(0.1501), 0.0, 0.0, math.sin(0.1501)]
    with pytest.raises(ValueError, match="corresponding local Joint pose"):
        subtrees._validate_rotation_state(encoded, different)


def test_restore_detects_native_side_effect_on_unrelated_state(native, monkeypatch):
    identifier = subtrees.subtree_snapshot("root")["snapshot_id"]
    original = common.modify

    def side_effect(scene, label, callback):
        original(scene, label, callback)
        native.values[NativeId("other/local_position")][2] = [999, 0, 0]

    monkeypatch.setattr(common, "modify", side_effect)
    with pytest.raises(RuntimeError, match="protected native channel"):
        subtrees.subtree_restore(snapshot_id=identifier)
