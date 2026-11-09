from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_for_cascadeur.host import characters, common


class NativeId(str):
    def is_null(self):
        return not self


class Mode(Enum):
    Static = 0
    Animation = 1


class Quaternion:
    def to_quaternion(self):
        return self

    def to_euler_angles(self):
        return [0.0, 0.0, 0.0]

    def w(self):
        return 1.0

    def x(self):
        return 0.0

    y = x
    z = x


class NativeCharacter:
    """Synthetic native relationships; contains no vendor character resources."""

    def __init__(self):
        self.names, self.kinds, self.parents, self.owners = {}, {}, {}, {}
        self.behaviors, self.links, self.channels, self.modes = {}, {}, {}, {}
        self.point_names = set(name for slots in characters.BODY_GROUPS.values() for name in slots.values())
        self.owner = self.add("Rig", "RigInfo")
        self.information = self.behaviors[self.owner]["RigInfo"]
        self.joints = set()
        for name, parent in characters.PARENT_NAMES.items():
            self.joints.add(self.add(name, "Joint", parent))
        for side in ("l", "r"):
            for finger in characters.FINGERS:
                parent = "hand_" + side
                for index in range(1, 5):
                    name = "f_" + finger + str(index) + "_" + side
                    self.joints.add(self.add(name, "Joint", parent))
                    parent = name
        self.points = {self.add(name, "Point", "Rig") for name in sorted(self.point_names)}
        self.boxes = set()
        for joint_name in characters.PARENT_NAMES:
            box = self.add("box_" + joint_name, "Box", "Rig")
            self.boxes.add(box)
            suffix = "_" + joint_name[-1] if joint_name.endswith(("_l", "_r")) else ""
            stem = joint_name[:-2] if suffix else joint_name
            link = {"joint": self.behaviors[NativeId(joint_name)]["Joint"]}
            for point_kind, property_name in (
                ("MainPoint", "main_point"),
                ("DirectionPoint", "direction_point"),
                ("AdditionalPoint", "additional_point"),
            ):
                point_name = stem + "_" + point_kind + suffix
                link[property_name] = (
                    self.behaviors[NativeId(point_name)]["Point"]
                    if point_name in self.point_names
                    else NativeId("")
                )
            self.links[self.behaviors[box]["RigAdditionalInfo"]] = link
        self.layer = SimpleNamespace(
            obj_ids=set(self.joints) | self.points | self.boxes, is_locked=False, sections={}
        )
        self.viewer = SimpleNamespace(
            get_behaviours=self.get_behaviours,
            get_behaviour_owner=self.owners.__getitem__,
            get_behaviour_objects_range=self.object_range,
            get_behaviour_by_name=lambda obj, name: self.behaviors[obj].get(name, NativeId("")),
            get_behaviour_object=lambda behavior, _: self.parents[self.owners[behavior]],
            get_behaviour_reference=lambda behavior, name: self.links[behavior].get(name, NativeId("")),
            get_behaviour_data=self.data_id,
        )
        self.model = SimpleNamespace(
            get_object_name=self.names.__getitem__, get_object_type_name=self.kinds.__getitem__
        )
        self.data = SimpleNamespace(
            has_data=lambda _: True,
            get_data=lambda key: SimpleNamespace(mode=self.modes.get(key, Mode.Animation)),
            get_data_value=self.data_value,
            get_animation_size=lambda: 24,
        )
        self.layers = SimpleNamespace(
            layer_id_by_obj_id_or_null=lambda _: NativeId("track"), layer=lambda _: self.layer
        )
        self.scene = SimpleNamespace(
            behaviour_viewer=lambda: self.viewer,
            model_viewer=lambda: self.model,
            data_viewer=lambda: self.data,
            layers_viewer=lambda: self.layers,
            get_current_frame=lambda: 0,
        )

    def add(self, name, kind, parent=None):
        identifier = NativeId(name)
        self.names[identifier], self.kinds[identifier] = name, kind
        self.parents[identifier] = NativeId(parent or "")
        types = {"Basic", "Transform", kind}
        if kind == "Box":
            types.add("RigAdditionalInfo")
        self.behaviors[identifier] = {}
        for behavior in types:
            key = NativeId("behavior:" + name + ":" + behavior)
            self.behaviors[identifier][behavior] = key
            self.owners[key] = identifier
        return identifier

    def get_behaviours(self, value):
        if value in self.behaviors:
            return list(self.behaviors[value].values())
        if value == "RigInfo":
            return [self.information]
        return []

    def object_range(self, _, name):
        return {
            "related_joints": list(self.joints),
            "rig_objects": list(self.points | self.boxes),
            "related_objects": [],
        }[name]

    def data_id(self, behavior, name):
        key = NativeId("data:" + str(self.owners[behavior]) + ":" + name)
        self.channels[key] = name
        return key

    def data_value(self, key, frame=0):
        return Quaternion() if "rotation" in key else [float(frame), 0.0, 0.0]


@pytest.fixture
def native(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "csc",
        SimpleNamespace(
            model=SimpleNamespace(DataMode=Mode),
            math=SimpleNamespace(
                Rotation=SimpleNamespace(
                    from_euler=lambda *args: Quaternion(), from_quaternion=lambda *args: Quaternion()
                )
            ),
        ),
    )
    return NativeCharacter()


def test_native_membership_builds_complete_semantics_without_name_heuristics(native):
    rig = characters.resolve_character(native.scene, "Rig", require_semantics=True)
    assert len(rig.controls) == 43
    assert len(rig.roles) == 11
    assert rig.roles["right_hand"]["center"] == "hand_MainPoint_r"
    assert rig.bindings["hand_MainPoint_r"].position_data == "data:hand_MainPoint_r:global_position"
    assert rig.role_joints["right_hand"] == "hand_r"


def test_semantics_rejects_correct_names_with_wrong_native_binding(native):
    link = native.behaviors[NativeId("box_hand_r")]["RigAdditionalInfo"]
    native.links[link]["main_point"] = native.behaviors[NativeId("hand_MainPoint_l")]["Point"]
    with pytest.raises(ValueError, match="binding differs"):
        characters.resolve_character(native.scene, "Rig", require_semantics=True)


def test_semantics_rejects_parent_topology_changes(native):
    native.parents[NativeId("hand_r")] = NativeId("pelvis")
    with pytest.raises(ValueError, match="hierarchy differs"):
        characters.resolve_character(native.scene, "Rig", require_semantics=True)


def test_semantics_rejects_extra_native_point_membership(native):
    native.points.add(native.add("unknown_point", "Point", "Rig"))
    with pytest.raises(ValueError, match="complete native Point"):
        characters.resolve_character(native.scene, "Rig", require_semantics=True)


def test_semantics_rejects_ambiguous_display_names(native):
    extra = native.add("duplicate", "Point", "Rig")
    native.names[extra] = "hand_MainPoint_r"
    native.points.add(extra)
    with pytest.raises(ValueError, match="expected one native Point"):
        characters.resolve_character(native.scene, "Rig", require_semantics=True)


def test_hands_validate_all_native_finger_chains_and_local_channels(native):
    rig = characters.resolve_character(native.scene, require_hands=True)
    assert set(rig.fingers) == {"left", "right"}
    assert len(rig.fingers["right"]) == 15
    assert rig.fingers["left"]["index2"].object_id == "f_index2_l"
    assert rig.fingers["left"]["index2"].rotation_data == "data:f_index2_l:local_rotation"


def test_hands_reject_wrong_terminal_chain_even_when_terminal_is_not_written(native):
    native.parents[NativeId("f_index4_l")] = NativeId("hand_l")
    with pytest.raises(ValueError, match="Finger parent"):
        characters.resolve_character(native.scene, require_hands=True)


def test_hands_reject_static_local_rotation(native):
    native.modes[NativeId("data:f_index1_r:local_rotation")] = Mode.Static
    with pytest.raises(ValueError, match="not animated"):
        characters.resolve_character(native.scene, require_hands=True)


def test_partial_semantic_patch_merges_each_frames_actual_values(native):
    rig = characters.resolve_character(native.scene, require_semantics=True)
    entries = characters._semantic_entries(
        native.scene,
        rig,
        [
            {"frame": 2, "pose": {"right_hand": {"center": [9, 0, 0]}}},
            {"frame": 5, "pose": {"right_hand": {"center": [10, 0, 0]}}},
        ],
    )
    assert entries[0]["pose"]["pelvis_MainPoint"]["position"] == [2, 0, 0]
    assert entries[1]["pose"]["pelvis_MainPoint"]["position"] == [5, 0, 0]
    assert entries[0]["pose"]["hand_MainPoint_r"]["position"] == [9, 0, 0]


def test_sampled_semantics_refuses_partial_role_targets(native):
    rig = characters.resolve_character(native.scene, require_semantics=True)
    with pytest.raises(ValueError, match="every role"):
        characters._semantic_entries(
            native.scene, rig, [{"frame": 0, "pose": {"head": {"center": [0, 0, 0]}}}], sampled_motion=True
        )


def test_sampled_plan_rejects_rigid_cluster_distortion(native):
    rig = characters.resolve_character(native.scene, require_semantics=True)
    targets = {key: [0, 0, 0] for key in rig.controls}
    targets["hand_DirectionPoint_r"] = [100, 0, 0]
    with pytest.raises(ValueError, match="rigid control cluster"):
        characters._validate_rigid_targets(native.scene, rig, 0, targets)


def test_point_plan_requires_complete_id_set_before_any_write(native):
    rig = characters.resolve_character(native.scene)
    with pytest.raises(ValueError, match="exactly every"):
        characters._point_plan(
            native.scene, rig, [{"frame": 0, "pose": {"hand_MainPoint_r": {"position": [0, 0, 0]}}}], False
        )


def test_hand_nonunit_quaternion_rejected_before_native_modification(native, monkeypatch):
    view = SimpleNamespace(get_path_name=lambda: "saved.casc")
    monkeypatch.setattr(common, "current_context", lambda: (None, None, view, native.scene))
    modify = Mock()
    monkeypatch.setattr(common, "modify", modify)
    with pytest.raises(ValueError, match="unit quaternions"):
        characters.hand_pose_apply_sequence([{"frame": 0, "hands": {"right": {"index1": [2, 0, 0, 0]}}}])
    modify.assert_not_called()


def test_durable_character_pose_restores_points_without_static_rotations(native, monkeypatch, tmp_path):
    import json
    import sys

    view = SimpleNamespace(get_path_name=lambda: str(tmp_path / "scene.casc"))
    monkeypatch.setattr(common, "current_context", lambda: (sys.modules["csc"], None, view, native.scene))
    common.configure(tmp_path)
    destination = tmp_path / "pose.json"
    characters.character_pose_snapshot(str(destination))
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert len(record["points"]) == 43
    assert all(set(patch) == {"position"} for patch in record["points"].values())
    apply = Mock(return_value={"execution_state": "completed"})
    monkeypatch.setattr(characters, "_apply_points", apply)
    result = characters.character_pose_restore(str(destination))
    assert result["joint_pose_verified"] is True
    plan = apply.call_args.args[2]
    assert all(set(patch) == {"position"} for patch in plan[0]["pose"].values())
    assert result["durable_format_version"] == 2
    assert result["finger_euler_fidelity"] is True
    record["version"] = 1
    del record["finger_rotations"]
    record["sha256"] = characters._pose_digest(
        {key: value for key, value in record.items() if key != "sha256"}
    )
    destination.write_text(json.dumps(record))
    legacy = characters.character_pose_restore(str(destination))
    assert legacy["durable_format_version"] == 1
    assert legacy["finger_euler_fidelity"] is False
    assert "cannot be recovered" in legacy["compatibility_note"]


def test_durable_pose_checksum_failure_prevents_any_native_edit(native, monkeypatch, tmp_path):
    import json

    view = SimpleNamespace(get_path_name=lambda: str(tmp_path / "scene.casc"))
    monkeypatch.setattr(common, "current_context", lambda: (None, None, view, native.scene))
    common.configure(tmp_path)
    destination = tmp_path / "pose.json"
    characters.character_pose_snapshot(str(destination))
    record = json.loads(destination.read_text(encoding="utf-8"))
    record["points"]["hand_MainPoint_r"]["position"] = [99, 0, 0]
    destination.write_text(json.dumps(record), encoding="utf-8")
    apply = Mock()
    monkeypatch.setattr(characters, "_apply_points", apply)
    with pytest.raises(ValueError, match="checksum differs"):
        characters.character_pose_restore(str(destination))
    apply.assert_not_called()


@pytest.mark.parametrize("defect", ["wrong_finger", "missing_euler", "wrong_quaternion"])
def test_durable_v2_rejects_incompatible_native_finger_records_before_any_edit(
    native, monkeypatch, tmp_path, defect
):
    import json
    import sys

    view = SimpleNamespace(get_path_name=lambda: str(tmp_path / "scene.casc"))
    monkeypatch.setattr(common, "current_context", lambda: (sys.modules["csc"], None, view, native.scene))
    common.configure(tmp_path)
    path = tmp_path / "invalid-fingers.json"
    characters.character_pose_snapshot(str(path))
    record = json.loads(path.read_text())
    identifier = next(iter(record["finger_rotations"]))
    if defect == "wrong_finger":
        other = next(key for key in record["joints"] if key not in record["finger_rotations"])
        record["finger_rotations"][other] = record["finger_rotations"].pop(identifier)
    elif defect == "missing_euler":
        del record["finger_rotations"][identifier]["euler_radians"]
    else:
        record["finger_rotations"][identifier]["quaternion_wxyz"] = [0.0, 1.0, 0.0, 0.0]
    record["sha256"] = characters._pose_digest(
        {key: value for key, value in record.items() if key != "sha256"}
    )
    path.write_text(json.dumps(record))
    apply = Mock()
    monkeypatch.setattr(characters, "_apply_points", apply)
    with pytest.raises(ValueError, match="finger|Euler"):
        characters.character_pose_restore(str(path))
    apply.assert_not_called()


def test_durable_pose_topology_change_prevents_any_native_edit(native, monkeypatch, tmp_path):
    view = SimpleNamespace(get_path_name=lambda: str(tmp_path / "scene.casc"))
    monkeypatch.setattr(common, "current_context", lambda: (None, None, view, native.scene))
    common.configure(tmp_path)
    destination = tmp_path / "pose.json"
    characters.character_pose_snapshot(str(destination))
    native.parents[NativeId("hand_r")] = NativeId("chest")
    apply = Mock()
    monkeypatch.setattr(characters, "_apply_points", apply)
    with pytest.raises(ValueError, match="another saved scene or native rig topology"):
        characters.character_pose_restore(str(destination))
    apply.assert_not_called()


def test_terminal_native_frame_remains_readable_but_not_writable(native):
    assert characters._frame(native.scene, 23) == 23
    with pytest.raises(ValueError, match="allocated successor"):
        characters._editable_frame(native.scene, 23)
    assert characters._editable_frame(native.scene, 22) == 22


def test_durable_restore_restores_changed_authored_fingers_after_body_interpolation(
    native, monkeypatch, tmp_path
):
    import math
    import sys

    class StoredQuaternion:
        def __init__(self, *components):
            self.components = components

        def w(self):
            return self.components[0]

        def x(self):
            return self.components[1]

        def y(self):
            return self.components[2]

        def z(self):
            return self.components[3]

        def to_quaternion(self):
            return self

        def to_euler_angles(self):
            return [2 * math.atan2(self.components[1], self.components[0]), 0.0, 0.0]

    overrides = {}
    original_read = native.data.get_data_value

    def read(reference, frame=0):
        # The synthetic FK finger output follows its authored local rotation.
        local = str(reference).replace(":global_rotation", ":local_rotation")
        return overrides.get(local, original_read(reference, frame))

    native.data.get_data_value = read
    csc = SimpleNamespace(
        model=SimpleNamespace(DataMode=Mode),
        math=SimpleNamespace(
            Rotation=SimpleNamespace(
                from_quaternion=StoredQuaternion,
                from_euler=lambda x, y, z: StoredQuaternion(math.cos(x / 2), math.sin(x / 2), 0, 0),
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "csc", csc)
    view = SimpleNamespace(get_path_name=lambda: str(tmp_path / "scene.casc"))
    monkeypatch.setattr(common, "current_context", lambda: (csc, None, view, native.scene))
    common.configure(tmp_path)
    path = tmp_path / "complete-pose.json"
    characters.character_pose_snapshot(str(path))
    overrides["data:f_index1_r:local_rotation"] = StoredQuaternion(math.cos(0.2), math.sin(0.2), 0, 0)
    assert characters.hand_pose_read(side="right")["hands"]["right"]["index1"][1] > 0.1
    calls = []

    def write(reference, frame, value):
        calls.append((str(reference), frame))
        overrides[str(reference)] = value

    interpolator = SimpleNamespace(reload=Mock(), interpolate=Mock())
    updater = SimpleNamespace(run_update=Mock(), get_interpolator=lambda: interpolator)
    model = SimpleNamespace(data_editor=lambda: SimpleNamespace(set_data_value=write))

    def apply_points(scene, rig, entries, **options):
        assert options["preserve_existing_curves"] is True
        assert options["defer_checkpoint_acceptance"] is True
        assert len(options["edited_finger_channels"]) == 30
        # The production Point transaction invokes this hook after its first
        # native interpolation and before any final native postcondition.
        options["after_interpolation"](model, updater)
        return {"execution_state": "completed"}

    monkeypatch.setattr(characters, "_apply_points", apply_points)
    result = characters.character_pose_restore(str(path))
    assert result["authored_finger_channels_restored"] == 30
    assert result["joint_pose_verified"] is True
    assert characters.hand_pose_read(side="right")["hands"]["right"]["index1"] == [1, 0, 0, 0]
    assert len(calls) == 30
    assert all("f_" in reference and reference.endswith(":local_rotation") for reference, _ in calls)
    interpolator.reload.assert_called_once()
    interpolator.interpolate.assert_called_once()

    def solver_perturbs_finger(scene, rig, entries, **options):
        value = apply_points(scene, rig, entries, **options)
        overrides["data:f_index1_r:local_rotation"] = StoredQuaternion(
            math.cos(0.0002), math.sin(0.0002), 0, 0
        )
        return value

    monkeypatch.setattr(characters, "_apply_points", solver_perturbs_finger)
    with pytest.raises(RuntimeError, match="authored finger-local rotation differs"):
        characters.character_pose_restore(str(path))


def test_durable_joint_verification_rejects_small_quaternion_component_error_and_accepts_antipodes():
    import copy
    import math

    expected = {
        "joint": {
            "id": "joint",
            "name": "Joint",
            "local": {"position": [0.0, 0.0, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "global": {"position": [0.0, 0.0, 0.0], "rotation_wxyz": [1.0, 0.0, 0.0, 0.0]},
        }
    }
    actual = copy.deepcopy(expected)
    actual["joint"]["local"]["rotation_wxyz"] = [math.sqrt(1 - 0.005**2), 0.005, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="native Joint pose differs"):
        characters._verify_joint_poses(expected, actual)
    actual["joint"]["local"]["rotation_wxyz"] = [-1.0, 0.0, 0.0, 0.0]
    assert characters._verify_joint_poses(expected, actual)["max_rotation_error_radians"] == 0
    actual["joint"]["global"]["position"] = [0.011, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="native Joint pose differs"):
        characters._verify_joint_poses(expected, actual)
    actual = copy.deepcopy(expected)
    actual["joint"]["name"] = "Renamed"
    with pytest.raises(RuntimeError, match="identity differs"):
        characters._verify_joint_poses(expected, actual)


def test_durable_v2_retains_a_full_finger_spin_that_quaternion_equivalence_cannot_describe(
    native, monkeypatch, tmp_path
):
    import json
    import math
    import sys

    class EulerRotation(Quaternion):
        def __init__(self, x, y=0.0, z=0.0):
            self.angles = [x, y, z]

        def w(self):
            return math.cos(self.angles[0] / 2)

        def x(self):
            return math.sin(self.angles[0] / 2)

        def y(self):
            return 0.0

        z = y

        def to_euler_angles(self):
            return list(self.angles)

    channel = "data:f_index1_r:local_rotation"
    winding = 2 * math.pi + 0.3
    overrides = {channel: EulerRotation(winding)}
    original = native.data.get_data_value
    native.data.get_data_value = lambda reference, frame=0: overrides.get(
        str(reference).replace(":global_rotation", ":local_rotation"), original(reference, frame)
    )
    csc = SimpleNamespace(
        model=SimpleNamespace(DataMode=Mode),
        math=SimpleNamespace(Rotation=SimpleNamespace(from_euler=EulerRotation)),
    )
    monkeypatch.setitem(sys.modules, "csc", csc)
    view = SimpleNamespace(get_path_name=lambda: str(tmp_path / "scene.casc"))
    monkeypatch.setattr(common, "current_context", lambda: (csc, None, view, native.scene))
    common.configure(tmp_path)
    path = tmp_path / "full-spin.json"
    characters.character_pose_snapshot(str(path))
    record = json.loads(path.read_text())
    assert record["finger_rotations"]["f_index1_r"]["euler_radians"][0] == winding
    overrides[channel] = EulerRotation(0.3)
    model = SimpleNamespace(
        data_editor=lambda: SimpleNamespace(
            set_data_value=lambda reference, frame, value: overrides.__setitem__(str(reference), value)
        )
    )
    updater = SimpleNamespace(
        run_update=lambda *_: None,
        get_interpolator=lambda: SimpleNamespace(reload=lambda: None, interpolate=lambda: None),
    )

    def apply_points(scene, rig, entries, **options):
        options["after_interpolation"](model, updater)
        return {}

    monkeypatch.setattr(characters, "_apply_points", apply_points)
    result = characters.character_pose_restore(str(path))
    assert overrides[channel].to_euler_angles()[0] == winding
    assert result["finger_max_euler_error_radians"] == 0
    assert result["finger_euler_fidelity"] is True


def preserved_pose_state():
    import copy

    metadata = {
        "interpolation": "BEZIER",
        "interval_common": {"ik_fk": "IK", "fixation": "FREE"},
        "ai_description": "",
        "key": {"label": "retained", "left_weight_velocity": 0.23, "right_weight_velocity": 0.71},
    }
    channels = {}
    for name, owner in (("point-position", "point"), ("finger-rotation", "finger"), ("outside", "outside")):
        channels["data:" + name] = {
            "owner": owner,
            "kind": "data",
            "mode": "Animation",
            "values": [[float(frame), 0.0, 0.0] for frame in range(6)],
        }
    tracks = {
        name: {
            "keys": [0, 5],
            "objects": [owner],
            "sections": {"0": copy.deepcopy(metadata), "5": copy.deepcopy(metadata)},
        }
        for name, owner in (
            ("body-track", "point"),
            ("finger-track", "finger"),
            ("external-track", "outside"),
        )
    }
    return {"channels": channels, "tracks": tracks}


def test_character_postconditions_reuse_the_latest_complete_native_commit_observation(native, monkeypatch):
    from mcp_for_cascadeur.host import recovery, state

    captured = {"channels": {}, "tracks": {}}
    monkeypatch.setattr(recovery, "latest_committed_checkpoint", lambda scene: captured)
    capture = Mock(side_effect=AssertionError("A post-commit checkpoint already exists"))
    monkeypatch.setattr(state, "capture", capture)
    result = characters._state_after_character(native.scene, SimpleNamespace(fingers={}), captured, set())
    assert result is captured
    capture.assert_not_called()


def test_durable_curve_protection_allows_only_new_target_keys_and_changed_target_pose():
    import copy

    before = preserved_pose_state()
    after = copy.deepcopy(before)
    for name in ("body-track", "finger-track"):
        after["tracks"][name]["keys"] = [0, 3, 5]
        after["tracks"][name]["sections"]["3"] = copy.deepcopy(after["tracks"][name]["sections"]["0"])
    after["channels"]["data:point-position"]["values"][3] = [99.0, 0.0, 0.0]
    after["channels"]["data:finger-rotation"]["values"][3] = [88.0, 0.0, 0.0]
    rig = SimpleNamespace(
        owned={"point", "finger"},
        bindings={"point": characters.ControlBinding("point", "point-position")},
        fingers={"right": {"index1": characters.FingerBinding("finger", "finger-rotation", "finger-track")}},
    )
    result = characters._verify_preserved_pose_curves(
        None, rig, before, after, [3], ["body-track", "finger-track"]
    )
    assert result["existing_curve_metadata"] == "exact"
    assert result["untouched_authored_keys_verified"] is True


@pytest.mark.parametrize(
    "defect,match",
    [
        ("weight", "existing curve metadata"),
        ("label", "existing curve metadata"),
        ("point_key", "untouched authored key"),
        ("finger_key", "untouched authored key"),
        ("external", "protected native data"),
        ("extra_key", "unrelated native keys"),
    ],
)
def test_durable_curve_protection_rejects_metadata_or_untouched_key_changes(defect, match):
    import copy

    from mcp_for_cascadeur.host.curves import CharacterVerificationError

    before = preserved_pose_state()
    after = copy.deepcopy(before)
    if defect in ("weight", "label"):
        field = "left_weight_velocity" if defect == "weight" else "label"
        after["tracks"]["body-track"]["sections"]["0"]["key"][field] = 0.99 if defect == "weight" else "lost"
    elif defect in ("point_key", "finger_key", "external"):
        key = {"point_key": "point-position", "finger_key": "finger-rotation", "external": "outside"}[defect]
        after["channels"]["data:" + key]["values"][5] = [999.0, 0.0, 0.0]
    else:
        after["tracks"]["body-track"]["keys"] = [0, 2, 5]
    rig = SimpleNamespace(
        owned={"point", "finger"},
        bindings={"point": characters.ControlBinding("point", "point-position")},
        fingers={"right": {"index1": characters.FingerBinding("finger", "finger-rotation", "finger-track")}},
    )
    with pytest.raises(CharacterVerificationError, match=match):
        characters._verify_preserved_pose_curves(
            None, rig, before, after, [0], ["body-track", "finger-track"]
        )
