import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_for_cascadeur.host import common, recovery, workflows
from mcp_for_cascadeur.host import scene as scene_tools


class Identifier(str):
    def is_null(self):
        return not self


class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.values = (w, x, y, z)

    def w(self):
        return self.values[0]

    def x(self):
        return self.values[1]

    def y(self):
        return self.values[2]

    def z(self):
        return self.values[3]


class Rotation(Quaternion):
    def to_quaternion(self):
        return self

    def __mul__(self, other):
        a, b, c, d = self.values
        w, x, y, z = other.values
        return Rotation(
            a * w - b * x - c * y - d * z,
            a * x + b * w + c * z - d * y,
            a * y - b * z + c * w + d * x,
            a * z + b * y - c * x + d * w,
        )

    @classmethod
    def from_quaternion(cls, *values):
        return cls(*values)

    @classmethod
    def from_euler(cls, x, y, z):
        return (
            cls(math.cos(x / 2), math.sin(x / 2), 0, 0)
            * cls(math.cos(y / 2), 0, math.sin(y / 2), 0)
            * cls(math.cos(z / 2), 0, 0, math.sin(z / 2))
        )

    def to_euler_angles(self):
        w, x, y, z = self.values
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return [[roll], [pitch], [yaw]]


def enum(name):
    return SimpleNamespace(name=name)


class Section:
    def __init__(self):
        self.interval = SimpleNamespace(interpolation=enum("BEZIER"))


class Track:
    def __init__(self, identifier):
        self.header = SimpleNamespace(name=identifier + " track")
        self.is_locked = False
        self.sections = {frame: Section() for frame in (0, 4, 7)}


class NativeScene:
    def __init__(self):
        self.size = 8
        self.requested_size = self.size
        self.modifications = 0
        self.updates = []
        self.events = []
        self.values = {
            (f"{obj}:{space}_{channel}", frame): (
                [float(frame), 0.0, 0.0] if channel == "position" else Rotation()
            )
            for obj in ("cube", "prop")
            for space in ("local", "global")
            for channel in ("position", "rotation")
            for frame in range(self.size)
        }
        self.tracks = {Identifier(obj + "-track"): Track(obj) for obj in ("cube", "prop")}
        self.track_selection = []
        self.selection_expansion = []
        self.selection_bounds = (0, 0)
        self.selection_calls = 0
        self.scrub = Mock()

    def data_viewer(self):
        return self

    def get_animation_size(self):
        return self.size

    def get_data(self, reference):
        return SimpleNamespace(mode="Animation")

    def get_data_value(self, reference, frame):
        return self.values[reference, frame]

    def model_viewer(self):
        def objects(name=None):
            if name is None:
                return [Identifier("cube"), Identifier("prop")]
            return (
                [Identifier("cube")]
                if name in ("cube", "Cube")
                else [Identifier("prop")]
                if name == "prop"
                else []
            )

        return SimpleNamespace(
            get_objects=objects,
            get_object_name=lambda identifier: str(identifier),
            behaviour_viewer=self.behaviour_viewer,
            data_viewer=self.data_viewer,
        )

    def behaviour_viewer(self):
        return SimpleNamespace(
            get_behaviour_by_name=lambda obj, behavior: Identifier(obj),
            get_behaviour_data=lambda obj, name: Identifier(obj + ":" + name),
        )

    def layers_viewer(self):
        return SimpleNamespace(
            all_layer_ids=lambda: list(self.tracks),
            layer=lambda identifier: self.tracks[identifier],
            header=lambda identifier: self.tracks[identifier].header,
            layer_id_by_obj_id_or_null=lambda obj: Identifier(obj + "-track"),
            folders_map=lambda: {Identifier("root"): SimpleNamespace()},
            root_id=lambda: Identifier("root"),
            all_parent_ids=lambda identifier: [Identifier("root")],
        )

    def get_current_frame(self):
        return 0

    def set_current_frame(self, frame):
        self.scrub(frame)

    def get_layers_selector(self):
        def select(tracks, first, last):
            self.selection_calls += 1
            self.track_selection, self.selection_bounds = (
                list(tracks) + self.selection_expansion,
                (first, last),
            )

        def selected():
            return SimpleNamespace(
                item_ids=lambda: list(self.track_selection),
                frames_interval=lambda: SimpleNamespace(
                    first=self.selection_bounds[0], last=self.selection_bounds[1]
                ),
            )

        return SimpleNamespace(set_full_selection_by_parts=select, selection=selected)

    def modify_update(self, label, callback):
        self.modifications += 1

        def endpoint(section, frame, track):
            self.events.append(("allocate", frame))
            self.tracks[track].sections[frame] = section
            self.requested_size = max(self.requested_size, frame + 1)

        def fit():
            for reference in {key for key, _ in self.values}:
                for frame in range(self.size, self.requested_size):
                    self.values[reference, frame] = self.values[reference, self.size - 1]
            self.size = self.requested_size

        def key(track, frame, active):
            assert frame < self.size - 1
            self.tracks[track].sections.setdefault(frame, Section())

        layer_editor = SimpleNamespace(
            set_section=endpoint,
            set_fixed_interpolation_or_key_if_need=key,
            change_section=lambda frame, track, callback: callback(self.tracks[track].sections[frame]),
            normalize_sections=lambda scene: None,
        )

        def write(reference, frame, value):
            self.values[reference, frame] = value

        model = SimpleNamespace(
            layers_editor=lambda: layer_editor,
            data_editor=lambda: SimpleNamespace(set_data_value=write),
            fit_animation_size_by_layers=fit,
        )
        updater = SimpleNamespace(
            generate_update=lambda: None,
            run_update=lambda references, frame: self.updates.append((references, frame)),
        )
        callback(model, None, updater)
        return True


@pytest.fixture
def native(monkeypatch):
    scene = NativeScene()
    vendor = SimpleNamespace(
        model=SimpleNamespace(DataMode=SimpleNamespace(Animation="Animation")),
        math=SimpleNamespace(Rotation=Rotation, Quaternion=Quaternion),
        layers=SimpleNamespace(
            layer=SimpleNamespace(
                Section=Section,
                Interpolation=SimpleNamespace(
                    **{name: enum(name) for name in workflows.animation.INTERPOLATIONS}
                ),
            )
        ),
    )
    core = SimpleNamespace(mirror_frame=Mock(), mirror_interval=Mock())
    view = SimpleNamespace()
    app = SimpleNamespace(
        get_tools_manager=lambda: SimpleNamespace(
            get_tool=lambda name: SimpleNamespace(
                editor=lambda selected_view: SimpleNamespace(core=lambda: core)
            )
        )
    )
    monkeypatch.setattr(common, "current_context", lambda: (vendor, app, view, scene))
    monkeypatch.setitem(
        sys.modules, "numpy", SimpleNamespace(asarray=lambda value, dtype: list(value), float32=float)
    )
    return scene, vendor, app, view, core


def test_native_frame_selection_reads_actual_selected_bounds(native):
    scene, _, _, _, _ = native
    result = workflows.tracks_select_frames(["cube-track"], 1, 5)
    assert scene.track_selection == ["cube-track"]
    assert scene.selection_bounds == (1, 5)
    assert result["verification"]["matches"] is True


def test_native_selection_accepts_canonical_ancestor_groups_but_reports_only_selected_leaf_tracks(native):
    scene, _, _, _, _ = native
    scene.selection_expansion = [Identifier("root")]
    result = workflows.tracks_select_frames(["cube-track"], 1, 5)
    assert result["tracks"] == ["cube-track"]
    assert result["all_selection_items"] == ["cube-track", "root"]
    assert result["expanded_groups"] == ["root"]
    assert result["unknown_selection_items"] == []
    assert result["verification"]["matches"] is True


def test_native_selection_rejects_an_unrequested_leaf_even_with_a_valid_parent_group(native):
    scene, _, _, _, _ = native
    scene.selection_expansion = [Identifier("root"), Identifier("prop-track")]
    result = workflows.tracks_select_frames(["cube-track"], 1, 5)
    assert set(result["tracks"]) == {"cube-track", "prop-track"}
    assert result["verification"]["matches"] is False


def test_native_selection_does_not_silently_ignore_unknown_selected_items(native):
    scene, _, _, _, _ = native
    scene.selection_expansion = [Identifier("unrecognized-group")]
    result = workflows.tracks_select_frames(["cube-track"], 1, 5)
    assert result["unknown_selection_items"] == ["unrecognized-group"]
    assert result["verification"]["matches"] is False


def test_native_selection_rejects_a_known_folder_that_is_not_an_ancestor(native):
    scene, _, _, _, _ = native
    original_viewer = scene.layers_viewer

    def layers():
        viewer = original_viewer()
        viewer.folders_map = lambda: {
            Identifier("root"): SimpleNamespace(),
            Identifier("unrelated"): SimpleNamespace(),
        }
        return viewer

    scene.layers_viewer = layers
    scene.selection_expansion = [Identifier("unrelated")]
    result = workflows.tracks_select_frames(["cube-track"], 1, 5)
    assert result["unknown_selection_items"] == []
    assert result["verification"]["unrelated_groups"] == ["unrelated"]
    assert result["verification"]["matches"] is False


def test_selection_prevalidates_all_tracks_before_native_change(native):
    scene, _, _, _, _ = native
    scene.tracks["prop-track"].is_locked = True
    with pytest.raises(ValueError, match="locked"):
        workflows.tracks_select_frames(["cube-track", "prop-track"], 1, 5)
    assert scene.selection_calls == 0


def test_motion_measure_samples_actual_curved_native_positions_without_scrubbing(native):
    scene, _, _, _, _ = native
    for frame in range(8):
        scene.values["cube:global_position", frame] = [float(frame * frame), 0.0, 0.0]
    result = workflows.motion_measure(["Cube"], 0, 6, step=2)
    assert result["frames"] == [0, 2, 4, 6]
    assert result["objects"][0]["positions"] == [[0, 0, 0], [4, 0, 0], [16, 0, 0], [36, 0, 0]]
    assert result["objects"][0]["path_length"] == 36
    assert result["objects"][0]["max_step_distance"] == 20
    scene.scrub.assert_not_called()


@pytest.mark.parametrize("interval", [False, True])
def test_native_mirror_invokes_installed_core_and_records_commit(native, monkeypatch, interval):
    scene, _, _, _, core = native
    order = []
    monkeypatch.setattr(recovery, "note_native_attempt", lambda actual: order.append(("attempt", actual)))
    monkeypatch.setattr(recovery, "note_native_commit", lambda actual: order.append(("commit", actual)))
    result = workflows.native_mirror(["Cube"], interval=interval)
    selected = core.mirror_interval if interval else core.mirror_frame
    selected.assert_called_once_with({"cube"})
    assert order == [("attempt", scene), ("commit", scene)]
    assert result["interval"] is interval


def test_native_mirror_failure_does_not_record_a_successful_commit(native, monkeypatch):
    _, _, _, _, core = native
    core.mirror_frame.side_effect = RuntimeError("native symmetry unavailable")
    committed = Mock()
    monkeypatch.setattr(recovery, "note_native_commit", committed)
    with pytest.raises(RuntimeError, match="symmetry"):
        workflows.native_mirror(["cube"])
    committed.assert_not_called()


def test_rotation_delta_postmultiplies_the_reference_frame_instead_of_target_frame(native):
    scene, _, _, _, _ = native
    scene.values["cube:local_rotation", 0] = Rotation(math.sqrt(0.5), 0, 0, math.sqrt(0.5))
    scene.values["cube:local_rotation", 2] = Rotation()
    result = workflows.transforms_animate(
        [
            {
                "frame": 2,
                "transforms": [
                    {"object": "cube", "rotation_delta_degrees": [90, 0, 0], "reference_frame": 0}
                ],
            }
        ]
    )
    observed = result["observed"][0]["rotation_wxyz"]
    assert all(math.isclose(component, 0.5, abs_tol=1e-12) for component in observed)
    assert result["verification"]["matches"] is True


def test_delta_accepts_quaternion_channels_and_writes_their_native_type(native):
    scene, _, _, _, _ = native
    for frame in range(8):
        scene.values["cube:local_rotation", frame] = Quaternion()
    workflows.transforms_animate(
        [{"frame": 2, "transforms": [{"object": "cube", "rotation_delta_degrees": [90, 0, 0]}]}]
    )
    assert type(scene.values["cube:local_rotation", 2]) is Quaternion
    assert math.isclose(scene.values["cube:local_rotation", 2].x(), math.sqrt(0.5))


def test_euler_read_uses_static_overload_and_retains_native_euler_winding(native):
    scene, _, _, _, _ = native
    rotation = Rotation.from_euler(0.25, 0.0, 0.0)
    rotation.to_euler_angles = lambda: [[2 * math.pi + 0.25], [0.0], [0.0]]
    original_getter = scene.get_data_value

    def get_value(reference, *frames):
        if reference == "cube:local_rotation":
            assert frames == (), "Native Static reads have no frame overload"
            return rotation
        return original_getter(reference, *frames)

    scene.get_data = lambda reference: SimpleNamespace(
        mode="Static" if reference == "cube:local_rotation" else "Animation"
    )
    scene.get_data_value = get_value
    result = workflows.pose_read_euler(["cube"], frame=3, space="local")
    assert result["poses"][0]["rotation_euler_radians"] == [2 * math.pi + 0.25, 0.0, 0.0]
    assert scene.modifications == 0


def test_transform_batch_rejects_static_rotation_without_using_an_invalid_native_read_overload(native):
    scene, _, _, _, _ = native
    original_getter = scene.get_data_value

    def get_value(reference, *frames):
        if reference == "cube:local_rotation":
            assert frames == ()
            return Rotation()
        return original_getter(reference, *frames)

    scene.get_data = lambda reference: SimpleNamespace(
        mode="Static" if reference == "cube:local_rotation" else "Animation"
    )
    scene.get_data_value = get_value
    with pytest.raises(ValueError, match="animated channels"):
        workflows.transforms_animate(
            [
                {"frame": 1, "transforms": [{"object": "prop", "position": [1.0, 2.0, 3.0]}]},
                {"frame": 2, "transforms": [{"object": "cube", "rotation_delta_degrees": [90.0, 0.0, 0.0]}]},
            ]
        )
    assert scene.modifications == 0


def test_batch_prevalidation_rejects_late_invalid_data_without_any_native_mutation(native):
    scene, _, _, _, _ = native
    with pytest.raises(ValueError, match="finite"):
        workflows.transforms_animate(
            [
                {"frame": 1, "transforms": [{"object": "cube", "position": [1, 2, 3]}]},
                {"frame": 2, "transforms": [{"object": "prop", "position": [float("nan"), 0, 0]}]},
            ]
        )
    assert scene.modifications == 0


def test_aliases_cannot_author_competing_channels_for_the_same_object_frame(native):
    scene, _, _, _, _ = native
    with pytest.raises(ValueError, match="only once"):
        workflows.transforms_animate(
            [
                {
                    "frame": 2,
                    "transforms": [
                        {"object": "cube", "position": [1, 2, 3]},
                        {"object": "Cube", "rotation_wxyz": [1, 0, 0, 0]},
                    ],
                }
            ]
        )
    assert scene.modifications == 0


def test_existing_terminal_frame_is_rejected_before_solver_mutation(native):
    scene, _, _, _, _ = native
    with pytest.raises(ValueError, match="terminal stored frame"):
        workflows.transforms_animate(
            [{"frame": 7, "transforms": [{"object": "cube", "position": [7, 0, 0]}]}]
        )
    assert scene.modifications == 0


def test_mixed_space_authoring_supports_500_keys_with_a_native_successor_guard(native):
    scene, _, _, _, _ = native
    keys = [
        {
            "frame": frame,
            "transforms": [{"object": "cube", "space": "global", "position": [float(frame), 2, 3]}],
        }
        for frame in range(500)
    ]
    result = workflows.transforms_animate(keys, extend_timeline=True, interpolation="LINEAR")
    assert result["frames"] == list(range(500))
    assert result["verification"]["matches"] is True
    assert scene.size == 501
    assert scene.values["cube:global_position", 499] == [499, 2, 3]
    assert scene.tracks["cube-track"].sections[200].interval.interpolation.name == "LINEAR"
    assert len(scene.updates) == 500


def test_euler_read_uses_native_rotation_and_flattens_column_vector(native):
    scene, _, _, _, _ = native
    scene.values["cube:local_rotation", 0] = Quaternion(math.sqrt(0.5), 0, 0, math.sqrt(0.5))
    result = workflows.pose_read_euler(["cube"])
    angles = result["poses"][0]["rotation_euler_radians"]
    assert math.isclose(angles[2], math.pi / 2)
    assert angles[:2] == [0, 0]
    assert scene.modifications == 0


def test_camera_framing_passes_world_points_to_native_zoom_without_treating_polar_position_as_xyz(
    native, monkeypatch
):
    scene, _, _, view, _ = native
    scene.values["cube:global_position", 0] = [10000.0, 500.0, -200.0]
    scene.values["prop:global_position", 0] = [10020.0, 520.0, -220.0]
    zoom = Mock()
    camera = SimpleNamespace(position=[0.5, 0.3, 250.0], target=[10000.0, 0.0, 0.0], type="PERSPECTIVE")
    viewport = SimpleNamespace(
        camera=lambda: SimpleNamespace(zoom_to_points=zoom),
        camera_struct=lambda: camera,
        mode_visualizers=lambda: "Controller",
    )
    view.active_viewport = lambda: SimpleNamespace(domain_viewport=lambda: viewport)
    set_camera = Mock(side_effect=AssertionError("Framing must use the native world-space API"))
    monkeypatch.setattr(scene_tools, "camera_set", set_camera)
    result = scene_tools.camera_frame(["cube", "prop"])
    zoom.assert_called_once_with([[10000.0, 500.0, -200.0], [10020.0, 520.0, -220.0]])
    assert camera.position == [0.5, 0.3, 250.0]
    assert result["distance_scale"] == 2.5
    set_camera.assert_not_called()
