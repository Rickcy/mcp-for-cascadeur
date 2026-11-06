import copy
import math
import sys
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import animation


class NativeId(str):
    def is_null(self):
        return self == ""


class NativeRotation:
    def __init__(self, *values):
        self.values = values

    def to_quaternion(self):
        return self

    def w(self):
        return self.values[0]

    def x(self):
        return self.values[1]

    def y(self):
        return self.values[2]

    def z(self):
        return self.values[3]


class DataViewer:
    def __init__(self):
        self.size = 8
        self.values = {("position", frame): [0.0, 1.0, 2.0] for frame in range(8)}
        self.values.update({("rotation", frame): NativeRotation(1, 0, 0, 0) for frame in range(8)})
        self.modes = {"position": "animated", "rotation": "animated"}
        self.settings = {("enabled", frame): frame >= 4 for frame in range(8)}

    def get_animation_size(self):
        return self.size

    def get_data(self, identifier):
        return SimpleNamespace(mode=self.modes[identifier])

    def get_data_value(self, identifier, frame):
        return self.values[identifier, frame]

    def get_all_data_id(self, identifier):
        return [NativeId(name) for name in self.modes]

    def get_all_settings_id(self, identifier):
        return [NativeId("enabled")]

    def get_setting(self, identifier):
        return SimpleNamespace(mode="animated")

    def get_setting_value(self, identifier, frame):
        return self.settings[identifier, frame]


class DataEditor:
    def __init__(self, viewer):
        self.viewer = viewer

    def set_data_value(self, identifier, frame, value):
        self.viewer.values[identifier, frame] = value

    def set_setting_value(self, identifier, frame, value):
        self.viewer.settings[identifier, frame] = value


class BehaviourViewer:
    parent = NativeId("")

    def get_behaviour_by_name(self, obj, name):
        return NativeId(name.lower()) if name in ("Transform", "Basic") else NativeId("")

    def get_behaviour_data(self, behavior, property_name):
        return NativeId("position" if property_name.endswith("position") else "rotation")

    def get_behaviours(self, identifier):
        return [NativeId("basic"), NativeId("transform")]

    def get_behaviour_name(self, identifier):
        return identifier.title()

    def get_behaviour_object(self, behavior, property_name):
        return self.parent


class ModelViewer:
    def __init__(self):
        self.identifiers = [NativeId("object-uuid")]

    def get_objects(self, name=None):
        return self.identifiers if name is None or name == "Hand" else []

    def get_object_name(self, identifier):
        return "Hand"

    def get_object_type_name(self, identifier):
        return "Point"


class NativeSection:
    def __init__(self):
        self.key = SimpleNamespace(
            common=SimpleNamespace(ik_fk="FK", fixation="Free"),
            tangents="Continuous",
            label=None,
            left_weight_velocity=1.25,
            right_weight_velocity=2.5,
        )
        self.interval = SimpleNamespace(
            common=SimpleNamespace(ik_fk="FK", fixation="Free"),
            interpolation=SimpleNamespace(name="LINEAR"),
            description_for_ai_interpolation="",
        )


class Track:
    is_locked = False

    def __init__(self):
        self.sections = {frame: NativeSection() for frame in (0, 4, 7)}
        self.obj_ids = {NativeId("object-uuid")}

    def is_key(self, frame):
        return frame in self.sections

    def key_frame_indices(self):
        return sorted(self.sections)

    def section(self, frame):
        return self.sections[frame]


class Layers:
    def __init__(self):
        self.track = Track()

    def layer_id_by_obj_id_or_null(self, identifier):
        return NativeId("track-uuid")

    def all_layer_ids(self):
        return [NativeId("track-uuid")]

    def layer(self, identifier):
        return self.track

    def header(self, identifier):
        return SimpleNamespace(name="Hand track")


class Scene:
    def __init__(self):
        self.data = DataViewer()
        self.model = ModelViewer()
        self.behaviors = BehaviourViewer()
        self.layers = Layers()
        self.modify_calls = 0
        self.updates = []
        self.events = []
        self.requested_size = self.data.size

        def raw_endpoint(section, frame, identifier):
            self.events.append(("endpoint", frame))
            self.layers.track.sections[frame] = section
            self.requested_size = max(self.requested_size, frame + 1)

        def insert_key(identifier, frame, create_key):
            assert frame < self.data.size, "native key helper sampled unallocated data"
            self.events.append(("key", frame))
            self.layers.track.sections.setdefault(frame, NativeSection())

        def fit():
            self.events.append(("fit", self.requested_size))
            for reference in self.data.modes:
                for frame in range(self.data.size, self.requested_size):
                    self.data.values[reference, frame] = copy.deepcopy(
                        self.data.values[reference, self.data.size - 1]
                    )
            self.data.size = self.requested_size

        def set_sections(sections, identifier):
            self.layers.track.sections = dict(sections)

        layer_editor = SimpleNamespace(
            normalize_sections=lambda scene: None,
            set_fixed_interpolation_or_key_if_need=insert_key,
            set_section=raw_endpoint,
            set_sections=set_sections,
        )
        self.editor = SimpleNamespace(
            data_editor=lambda: DataEditor(self.data),
            layers_editor=lambda: layer_editor,
            fit_animation_size_by_layers=fit,
        )
        self.updater = SimpleNamespace(
            run_update=lambda ids, frame: self.updates.append((ids, frame)),
            generate_update=lambda: self.events.append(("generate", self.data.size)),
        )

    def data_viewer(self):
        return self.data

    def model_viewer(self):
        return self.model

    def behaviour_viewer(self):
        return self.behaviors

    def layers_viewer(self):
        return self.layers

    def get_current_frame(self):
        return 0

    def modify_update(self, label, callback):
        self.modify_calls += 1
        callback(self.editor, None, self.updater)
        return True


@pytest.fixture
def native(monkeypatch):
    scene = Scene()
    csc = SimpleNamespace(
        model=SimpleNamespace(
            DataMode=SimpleNamespace(Animation="animated", Static="static"),
            SettingMode=SimpleNamespace(Animation="animated", Static="static"),
        ),
        math=SimpleNamespace(
            Rotation=SimpleNamespace(from_quaternion=lambda *values: NativeRotation(*values)),
            Quaternion=lambda *values: NativeRotation(*values),
        ),
        layers=SimpleNamespace(
            layer=SimpleNamespace(Section=NativeSection),
            CyclesViewer=lambda track: SimpleNamespace(get_cycles_in_frames=lambda first, last: []),
        ),
    )
    view = SimpleNamespace(name=lambda: "Test", get_path_name=lambda: "C:/work/test.casc")
    monkeypatch.setattr(animation, "current_context", lambda: (csc, None, view, scene))
    monkeypatch.setitem(
        sys.modules, "numpy", SimpleNamespace(asarray=lambda value, dtype: list(value), float32=float)
    )
    animation._snapshots.clear()
    yield scene, csc, view
    animation._snapshots.clear()


def test_batch_validation_precedes_every_native_mutation(native):
    scene, _, _ = native
    before = copy.deepcopy(scene.data.values)
    with pytest.raises(ValueError, match="nonfinite"):
        animation.pose_apply_sequence(
            [
                {"frame": 0, "poses": [{"object": "Hand", "position": [4, 5, 6]}]},
                {"frame": 4, "poses": [{"object": "Hand", "position": [float("nan"), 2, 3]}]},
            ]
        )
    assert scene.modify_calls == 0
    assert scene.data.values["position", 0] == before["position", 0]


def test_native_aliases_cannot_write_same_object_twice(native):
    scene, _, _ = native
    with pytest.raises(ValueError, match="same object"):
        animation.pose_apply(
            [
                {"object": "Hand", "position": [1, 2, 3]},
                {"object": "object-uuid", "position": [3, 4, 5]},
            ]
        )
    assert scene.modify_calls == 0


@pytest.mark.parametrize("blocked", ["locked", "static", "missing_key"])
def test_native_write_preconditions_do_not_modify_scene(native, blocked):
    scene, _, _ = native
    frame = 0
    if blocked == "locked":
        scene.layers.track.is_locked = True
    elif blocked == "static":
        scene.data.modes["position"] = "static"
    else:
        frame = 2
    with pytest.raises(ValueError):
        animation.pose_apply([{"object": "Hand", "position": [3, 4, 5]}], frame=frame, create_keys=False)
    assert scene.modify_calls == 0


def test_pose_write_updates_native_scene_once_and_reads_actual_targets(native):
    scene, _, _ = native
    result = animation.pose_apply(
        [{"object": "Hand", "position": [3, 4, 5], "rotation_wxyz": [2, 0, 0, 0]}], create_keys=False
    )
    assert scene.modify_calls == 1
    assert scene.updates == [({"position", "rotation"}, 0)]
    assert result["verification"]["matches"] is True
    assert result["observed"][0]["poses"][0]["position"] == [3, 4, 5]
    assert result["observed"][0]["poses"][0]["rotation_wxyz"] == [1, 0, 0, 0]


def test_snapshot_requires_same_topology_before_restore(native):
    scene, _, _ = native
    result = animation.pose_snapshot(["Hand"])
    scene.model.identifiers.append(NativeId("second-object"))
    with pytest.raises(ValueError, match="topology"):
        animation.pose_restore(snapshot_id=result["snapshot_id"])
    assert scene.modify_calls == 0


def test_snapshot_requires_same_saved_scene_path(native):
    scene, _, view = native
    result = animation.pose_snapshot(["Hand"])
    view.get_path_name = lambda: "C:/work/other.casc"
    with pytest.raises(ValueError, match="different saved scene"):
        animation.pose_restore(snapshot_id=result["snapshot_id"])
    assert scene.modify_calls == 0


def test_existing_key_offset_accumulates_from_actual_values(native):
    scene, _, _ = native
    animation.pose_offset(["Hand"], [0], [1, 0, 0])
    animation.pose_offset(["Hand"], [0], [1, 0, 0])
    assert scene.data.values["position", 0] == [2, 1, 2]


def test_property_batch_checks_all_value_types_before_writing(native):
    scene, _, _ = native
    scene.data.values["position", 0] = 2.5
    changes = [
        {"object": "Hand", "behavior": "Transform", "property": "global_position", "value": 3.5},
        {
            "object": "Hand",
            "behavior": "Transform",
            "property": "global_rotation",
            "value": "invalid rotation",
        },
    ]
    with pytest.raises(ValueError, match="Rotation values"):
        animation.object_properties_set(changes)
    assert scene.modify_calls == 0
    assert scene.data.values["position", 0] == 2.5


def test_extension_allocates_and_generates_before_native_key_helpers(native):
    scene, _, _ = native
    result = animation.tracks_set_keys(["track-uuid"], [0, 12, 24], extend_timeline=True)
    assert result["stored_frames"] == 26
    assert scene.events[:3] == [("endpoint", 25), ("fit", 26), ("generate", 26)]
    assert scene.events[3:6] == [("key", 0), ("key", 12), ("key", 24)]


def test_pose_extension_allocates_before_keys_and_data_writes(native):
    scene, _, _ = native
    result = animation.pose_apply_sequence(
        [
            {"frame": 0, "poses": [{"object": "Hand", "position": [0, 0, 0]}]},
            {"frame": 24, "poses": [{"object": "Hand", "position": [4, 0, 0]}]},
        ],
        extend_timeline=True,
    )
    assert scene.events[:3] == [("endpoint", 25), ("fit", 26), ("generate", 26)]
    assert result["verification"]["matches"] is True


def test_retime_preserves_settings_key_metadata_and_authored_values(native):
    scene, _, _ = native
    scene.data.values["position", 4] = [4, 5, 6]
    result = animation.tracks_retime(["track-uuid"], [0, 4, 7], [0, 3, 7])
    assert scene.layers.track.key_frame_indices() == [0, 3, 7]
    assert scene.layers.track.section(3).key.left_weight_velocity == 1.25
    assert scene.data.values["position", 3] == [4, 5, 6]
    assert scene.data.settings["enabled", 3] is True
    assert result["key_mapping_verified"] and result["channel_values_verified"]


def test_retime_rejects_parent_dependencies_before_modifying(native):
    scene, _, _ = native
    scene.behaviors.parent = NativeId("parent-object")
    with pytest.raises(ValueError, match="parented"):
        animation.tracks_retime(["track-uuid"], [0, 4, 7], [0, 3, 7])
    assert scene.modify_calls == 0


def test_integer_properties_reject_cpp_overflow(native):
    _, csc, _ = native
    with pytest.raises(ValueError, match="32-bit"):
        animation._coerce_property_value(csc, 0, 2**40)


def test_solver_failure_reports_exact_stage_and_unknown_native_outcome(native):
    scene, _, _ = native

    def fail_solver(changed, frame):
        raise RuntimeError("native map entry unavailable")

    def swallowed_callback_error(label, callback):
        scene.modify_calls += 1
        try:
            callback(scene.editor, None, scene.updater)
        except RuntimeError:
            return False
        return True

    scene.updater.run_update = fail_solver
    scene.modify_update = swallowed_callback_error
    with pytest.raises(RuntimeError, match="solve changed controllers at frame 0") as failure:
        animation.pose_apply([{"object": "Hand", "position": [3, 4, 5]}], create_keys=False)
    assert failure.value.execution_state == "unknown"
    assert scene.modify_calls == 1
    assert scene.data.values["position", 0] == [3, 4, 5]


def test_position_tolerance_changes_verification_without_changing_native_targets(native):
    scene, _, _ = native

    def native_precision(changed, frame):
        scene.data.values["position", frame][0] += 0.0015

    scene.updater.run_update = native_precision
    keyframes = [{"frame": 0, "poses": [{"object": "Hand", "position": [3, 4, 5]}]}]
    strict = animation.pose_apply_sequence(keyframes)
    accepted = animation.pose_apply_sequence(keyframes, position_tolerance=0.002)
    assert strict["verification"]["matches"] is False
    assert strict["verification"]["position_tolerance"] == 0.001
    assert accepted["verification"]["matches"] is True
    assert accepted["verification"]["position_tolerance"] == 0.002
    assert math.isclose(accepted["verification"]["max_position_error"], 0.0015, rel_tol=1e-6)
    assert strict["observed"] == accepted["observed"]


def test_rotation_tolerance_is_an_angle_in_radians(native):
    scene, _, _ = native

    def native_precision(changed, frame):
        scene.data.values["rotation", frame] = NativeRotation(
            math.cos(0.0015 / 2), 0, math.sin(0.0015 / 2), 0
        )

    scene.updater.run_update = native_precision
    pose = [{"object": "Hand", "rotation_wxyz": [1, 0, 0, 0]}]
    strict = animation.pose_apply(pose)
    accepted = animation.pose_apply(pose, rotation_tolerance=0.002)
    assert strict["verification"]["matches"] is False
    assert accepted["verification"]["matches"] is True
    assert accepted["verification"]["rotation_tolerance_radians"] == 0.002
    assert math.isclose(accepted["verification"]["max_rotation_error_radians"], 0.0015, rel_tol=1e-6)
    assert strict["observed"] == accepted["observed"]


@pytest.mark.parametrize("field", ["position_tolerance", "rotation_tolerance"])
@pytest.mark.parametrize("value", [0, -0.1, 1.001, float("nan"), float("inf"), True])
def test_tolerance_validation_precedes_native_writes(native, field, value):
    scene, _, _ = native
    with pytest.raises(ValueError, match=field):
        animation.pose_apply([{"object": "Hand", "position": [3, 4, 5]}], **{field: value})
    assert scene.modify_calls == 0


def test_restore_forwards_explicit_verification_tolerances(native):
    scene, _, _ = native
    snapshot = animation.pose_snapshot(["Hand"])
    scene.data.values["position", 0] = [100, 0, 0]

    def native_precision(changed, frame):
        scene.data.values["position", frame][0] += 0.0015

    scene.updater.run_update = native_precision
    result = animation.pose_restore(
        snapshot_id=snapshot["snapshot_id"], position_tolerance=0.002, rotation_tolerance=0.005
    )
    assert result["verification"]["matches"] is True
    assert result["verification"]["position_tolerance"] == 0.002
    assert result["verification"]["rotation_tolerance_radians"] == 0.005
    assert result["observed"][0]["poses"][0]["position"] == [0.0015, 1, 2]
