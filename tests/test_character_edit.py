import copy
import math
import sys
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import character_edit, curves, state


def enum(name):
    return SimpleNamespace(name=name)


class Identifier(str):
    def is_null(self):
        return not self


class Rotation:
    def __init__(self, *values):
        self.values = values or (1, 0, 0, 0)

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


class Section:
    def __init__(self):
        self.interval = SimpleNamespace(
            interpolation=enum("BEZIER"),
            common=SimpleNamespace(ik_fk=enum("IK"), fixation=enum("Free")),
            description_for_ai_interpolation="original interpolation description",
        )
        self.key = SimpleNamespace(
            common=SimpleNamespace(ik_fk=enum("IK"), fixation=enum("Free")),
            tangents=enum("Continuous"),
            label=7,
            left_weight_velocity=0.2,
            right_weight_velocity=0.8,
        )


class Track:
    def __init__(self, name, objects, *, fixed=False):
        self.header = SimpleNamespace(name=name, parent=Identifier("root"))
        self.obj_ids = {Identifier(obj) for obj in objects}
        self.sections = {frame: Section() for frame in (0, 2, 4)}
        self.is_locked = False
        self.is_visible = True
        if fixed:
            for section in self.sections.values():
                section.interval.interpolation = enum("FIXED")

    def is_key(self, frame):
        return frame in self.sections

    def key_frame_indices(self):
        return sorted(self.sections)

    def section(self, frame):
        return self.sections[frame]


class Data:
    def __init__(self):
        self.values = {}
        self.owners = {
            "point-pos": "point",
            "finger-rot": "finger",
            "joint-global-pos": "joint",
            "joint-local-pos": "joint",
            "joint-global-rot": "joint",
            "joint-local-rot": "joint",
            "external": "prop",
        }
        for frame in range(5):
            for channel in ("point-pos", "joint-global-pos", "joint-local-pos"):
                self.values[channel, frame] = [float(frame * 10), 1.0, 2.0]
            for channel in ("finger-rot", "joint-global-rot", "joint-local-rot"):
                self.values[channel, frame] = Rotation()
            self.values["external", frame] = [float(frame), 0.0, 0.0]
        self.settings = {frame: frame >= 2 for frame in range(5)}
        self.writes = []

    def get_animation_size(self):
        return 5

    def get_data(self, reference):
        return SimpleNamespace(mode=enum("Animation"))

    def get_data_value(self, reference, frame=None):
        return self.values[reference, frame]

    def get_setting_value(self, reference, frame):
        return self.settings[frame]

    def set_data_value(self, reference, frame, value):
        self.writes.append(("data", str(reference), frame))
        self.values[reference, frame] = copy.deepcopy(value)

    def set_setting_value(self, reference, frame, value):
        self.writes.append(("setting", str(reference), frame))
        self.settings[frame] = value


class Scene:
    def __init__(self):
        self.data = Data()
        self.tracks = {
            Identifier("point-track"): Track("Point", ["point"]),
            Identifier("finger-track"): Track("Finger", ["finger", "joint"]),
            Identifier("external-track"): Track("Baked prop", ["prop"], fixed=True),
        }
        self.modifications = 0
        self.interpolation_failure = None
        self.corrupt_external = False
        self.layer_viewer = SimpleNamespace(
            all_layer_ids=lambda: list(self.tracks),
            layer=lambda identifier: self.tracks[identifier],
            layer_id_by_obj_id_or_null=lambda obj: Identifier(
                "point-track" if obj == "point" else "finger-track"
            ),
        )
        self.model_viewer_instance = SimpleNamespace(
            get_objects=lambda name=None: (
                [Identifier(name)]
                if name is not None
                else [Identifier(obj) for obj in ("point", "finger", "joint", "prop")]
            ),
            get_object_name=lambda identifier: str(identifier),
            get_object_type_name=lambda identifier: "Point" if identifier == "point" else "Joint",
        )
        self.behaviours = SimpleNamespace(
            get_behaviour_by_name=lambda obj, kind: Identifier(obj + ":" + kind),
            get_behaviour_data=lambda behavior, channel: Identifier(
                behavior.split(":")[0]
                + "-"
                + channel.replace("position", "pos").replace("rotation", "rot").replace("_", "-")
            ),
        )

    def data_viewer(self):
        return self.data

    def layers_viewer(self):
        return self.layer_viewer

    def model_viewer(self):
        return self.model_viewer_instance

    def behaviour_viewer(self):
        return self.behaviours

    def get_current_frame(self):
        return 0

    def modify_update(self, label, callback):
        self.modifications += 1

        def unset_section(frame, identifier):
            removed = self.tracks[identifier].sections.pop(frame)
            removed.key.label = "invalidated native wrapper"

        def insert_key(identifier, frame, active):
            self.tracks[identifier].sections.setdefault(frame, Section())

        def change_section(frame, identifier, edit):
            edit(self.tracks[identifier].sections[frame])

        editor = SimpleNamespace(
            data_editor=lambda: self.data,
            layers_editor=lambda: SimpleNamespace(
                unset_section=unset_section,
                set_fixed_interpolation_or_key_if_need=insert_key,
                change_section=change_section,
                normalize_sections=lambda scene: None,
            ),
        )

        def update(references, frame):
            if frame == 4:
                raise RuntimeError("The terminal native guard must not be solved")
            if "point-pos" in references:
                self.data.values["joint-global-pos", frame] = list(self.data.values["point-pos", frame])
                self.data.values["joint-local-pos", frame] = list(self.data.values["point-pos", frame])

        def interpolate():
            if self.interpolation_failure:
                raise RuntimeError(self.interpolation_failure)
            if self.corrupt_external:
                self.data.values["external", 1][0] += 1.0

        interpolator = SimpleNamespace(reload=lambda: None, interpolate=interpolate)
        updater = SimpleNamespace(
            run_update=update, get_interpolator=lambda: interpolator, generate_update=lambda: None
        )
        try:
            callback(editor, None, updater)
        except RuntimeError:
            return False
        return True


def checkpoint(scene):
    channels = {
        "data:" + reference: {
            "id": reference,
            "owner": owner,
            "name": reference,
            "kind": "data",
            "mode": "Animation",
            "values": [state.encode(scene.data.values[reference, frame]) for frame in range(5)],
        }
        for reference, owner in scene.data.owners.items()
    }
    channels["setting:enabled"] = {
        "id": "enabled",
        "owner": "finger",
        "name": "enabled",
        "kind": "setting",
        "mode": "Animation",
        "values": [scene.data.settings[frame] for frame in range(5)],
    }
    return {
        "format_version": 1,
        "count": 5,
        "objects": {obj: {} for obj in ("point", "finger", "joint", "prop")},
        "channels": channels,
        "auxiliary": {},
        "external": {},
        "tracks": {
            str(identifier): {
                "name": track.header.name,
                "parent": str(track.header.parent),
                "objects": sorted(map(str, track.obj_ids)),
                "locked": track.is_locked,
                "visible": track.is_visible,
                "keys": list(track.key_frame_indices()),
                "sections": {
                    str(frame): state.section_state(section) for frame, section in track.sections.items()
                },
            }
            for identifier, track in scene.tracks.items()
        },
    }


@pytest.fixture
def native(monkeypatch):
    scene = Scene()
    rig = SimpleNamespace(
        id="rig",
        controls={"point": Identifier("point")},
        bindings={
            "point": SimpleNamespace(position_data=Identifier("point-pos"), role="left_hand", slot="center")
        },
        owned={Identifier(obj) for obj in ("point", "finger", "joint")},
        joints={Identifier("joint")},
        tracks={Identifier("point-track"), Identifier("finger-track")},
        roles={},
        fingers={
            "left": {
                "index1": SimpleNamespace(
                    object_id=Identifier("finger"),
                    rotation_data=Identifier("finger-rot"),
                    track_id=Identifier("finger-track"),
                )
            }
        },
    )
    vendor = SimpleNamespace(
        model=SimpleNamespace(SettingId=Identifier),
        math=SimpleNamespace(
            Rotation=SimpleNamespace(from_quaternion=lambda *values: Rotation(*values)), Quaternion=Rotation
        ),
        additive_layers=SimpleNamespace(
            get_manager=lambda scene: SimpleNamespace(get_stacked_layers_count=lambda: 0)
        ),
        layers=SimpleNamespace(
            layer=SimpleNamespace(
                Section=Section,
                Interpolation=SimpleNamespace(
                    LINEAR=enum("LINEAR"),
                    BEZIER=enum("BEZIER"),
                    LOW_AMPLITUDE_BEZIER=enum("LOW_AMPLITUDE_BEZIER"),
                    CLAMPED_BEZIER=enum("CLAMPED_BEZIER"),
                    FIXED=enum("FIXED"),
                ),
                IkFk=SimpleNamespace(IK=enum("IK"), FK=enum("FK")),
                Fixation=SimpleNamespace(Free=enum("Free"), Fulcrum=enum("Fulcrum")),
                Tangents=SimpleNamespace(Continuous=enum("Continuous"), UserDefined=enum("UserDefined")),
            ),
            CyclesViewer=lambda track: SimpleNamespace(any_cycles_exist_in_frames=lambda a, b: False),
        ),
    )
    for name in ("Interpolation", "IkFk", "Fixation", "Tangents"):
        values = getattr(vendor.layers.layer, name)
        values.__members__ = vars(values).copy()
    monkeypatch.setitem(sys.modules, "csc", vendor)
    monkeypatch.setitem(
        sys.modules, "numpy", SimpleNamespace(asarray=lambda value, dtype: list(value), float32=float)
    )
    monkeypatch.setattr(character_edit, "current_context", lambda: (vendor, None, None, scene))
    monkeypatch.setattr(character_edit, "_resolve", lambda *args, **kwargs: rig)
    monkeypatch.setattr(curves, "checkpoint_before", checkpoint)
    monkeypatch.setattr(curves, "_capture", checkpoint)
    monkeypatch.setattr(
        curves, "verify_geometry", lambda *args, **kwargs: {"max_connection_anchor_error": 0.001}
    )
    return scene, rig


@pytest.mark.parametrize("interpolation", ["BEZIER", "CLAMPED_BEZIER"])
def test_character_retime_stages_authored_sources_and_preserves_key_metadata(native, interpolation):
    scene, _ = native
    for identifier in ("point-track", "finger-track"):
        for section in scene.tracks[identifier].sections.values():
            section.interval.interpolation = enum(interpolation)
    old_metadata = checkpoint(scene)["tracks"]["point-track"]["sections"]["2"]
    result = character_edit.character_retime(
        "rig", [{"source": 0, "target": 0}, {"source": 2, "target": 1}, {"source": 4, "target": 4}]
    )
    assert scene.data.values["point-pos", 1] == [20.0, 1.0, 2.0]
    assert checkpoint(scene)["tracks"]["point-track"]["sections"]["1"] == old_metadata
    assert scene.data.settings[1] is True
    assert result["curve_metadata_verified"] is True
    assert result["joint_key_verification"]["max_position_error"] == 0
    assert result["unchanged_terminal_key_skipped"] == 4
    written = {reference for _, reference, _ in scene.data.writes}
    assert written == {"point-pos", "finger-rot", "enabled"}
    assert not any(frame == 4 for _, _, frame in scene.data.writes)


@pytest.mark.parametrize("interpolation", ["LINEAR", "CLAMPED_BEZIER"])
def test_retime_explicit_easing_changes_only_requested_metadata(native, interpolation):
    scene, _ = native
    character_edit.character_retime(
        "rig",
        [
            {"source": 0, "target": 0},
            {
                "source": 2,
                "target": 1,
                "interpolation": interpolation,
                "left_weight": 0.3,
                "right_weight": 0.7,
            },
            {"source": 4, "target": 4},
        ],
    )
    moved = scene.tracks["point-track"].sections[1]
    assert moved.interval.interpolation.name == interpolation
    assert moved.key.left_weight_velocity == 0.3
    assert moved.key.right_weight_velocity == 0.7
    assert moved.key.label == 7
    assert moved.interval.description_for_ai_interpolation == "original interpolation description"


def test_retime_rejects_a_native_downgrade_of_clamped_interpolation(native):
    scene, _ = native
    for section in scene.tracks["point-track"].sections.values():
        section.interval.interpolation = enum("CLAMPED_BEZIER")
    original = scene.modify_update

    def native_downgrade(label, callback):
        accepted = original(label, callback)
        scene.tracks["point-track"].sections[1].interval.interpolation = enum("BEZIER")
        return accepted

    scene.modify_update = native_downgrade
    with pytest.raises(curves.CharacterVerificationError, match="metadata"):
        character_edit.character_retime(
            "rig", [{"source": 0, "target": 0}, {"source": 2, "target": 1}, {"source": 4, "target": 4}]
        )
    assert scene.modifications == 1


def test_retime_validates_requested_clamped_support_before_removing_native_keys(native):
    scene, _ = native
    del sys.modules["csc"].layers.layer.Interpolation.__members__["CLAMPED_BEZIER"]
    before = checkpoint(scene)
    with pytest.raises(state.StateError, match="exact curve metadata"):
        character_edit.character_retime(
            "rig",
            [
                {"source": 0, "target": 0},
                {"source": 2, "target": 1, "interpolation": "CLAMPED_BEZIER"},
                {"source": 4, "target": 4},
            ],
        )
    assert scene.modifications == 0
    assert scene.data.writes == []
    assert checkpoint(scene) == before


def test_retime_schema_advertises_native_clamped_without_baked_or_ai_modes():
    tool = next(tool for tool in character_edit.TOOLS if tool.name == "character_retime")
    modes = tool.properties["keys"]["items"]["properties"]["interpolation"]["enum"]
    assert "CLAMPED_BEZIER" in modes
    assert not {"FIXED", "AI", "NONE"}.intersection(modes)


@pytest.mark.parametrize("collapse_branch", [False, True])
def test_retime_checks_finger_euler_branch_even_when_quaternion_orientation_is_unchanged(
    native, collapse_branch
):
    scene, _ = native

    class EulerRotation(Rotation):
        def __init__(self, angle):
            super().__init__(math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)
            self.angle = angle

        def to_euler_angles(self):
            return [[self.angle], [0.0], [0.0]]

    original_angle = 2 * math.pi + 0.25
    for frame in range(5):
        scene.data.values["finger-rot", frame] = EulerRotation(original_angle)
    sys.modules["csc"].math.Rotation.from_euler = lambda x, y, z: EulerRotation(x)
    original_modify = scene.modify_update

    def native_commit(label, callback):
        accepted = original_modify(label, callback)
        if collapse_branch:
            scene.data.values["finger-rot", 1] = EulerRotation(0.25)
        return accepted

    scene.modify_update = native_commit
    mapping = [{"source": 0, "target": 0}, {"source": 2, "target": 1}, {"source": 4, "target": 4}]
    assert character_edit.animation._native_values_match(EulerRotation(original_angle), EulerRotation(0.25))
    if collapse_branch:
        with pytest.raises(curves.CharacterVerificationError, match="Euler branch") as failure:
            character_edit.character_retime("rig", mapping)
        assert failure.value.execution_state == "verification_failed"
    else:
        result = character_edit.character_retime("rig", mapping)
        assert result["finger_local_channels_verified"] == 1
        assert scene.data.values["finger-rot", 1].angle == original_angle


def test_retime_does_not_retain_or_construct_detached_native_sections(native):
    scene, _ = native
    vendor = sys.modules["csc"]

    def detached_section():
        raise AssertionError("Detached Section construction is not a supported retime path")

    vendor.layers.layer.Section = detached_section
    for track in scene.tracks.values():
        track.section = lambda frame: (_ for _ in ()).throw(
            AssertionError("Native Section references must not be retained")
        )
    character_edit.character_retime(
        "rig", [{"source": 0, "target": 0}, {"source": 2, "target": 1}, {"source": 4, "target": 4}]
    )
    assert scene.tracks["point-track"].sections[1].key.label == 7
    assert scene.tracks["point-track"].key_frame_indices() == [0, 1, 4]


def test_retime_preflight_rejects_unsynchronized_tracks_before_mutation(native):
    scene, _ = native
    del scene.tracks["finger-track"].sections[2]
    with pytest.raises(curves.UnsupportedCurveState, match="synchronized"):
        character_edit.character_retime(
            "rig", [{"source": 0, "target": 0}, {"source": 2, "target": 1}, {"source": 4, "target": 4}]
        )
    assert scene.modifications == 0
    assert scene.data.writes == []


def test_curve_preserving_terminal_edit_is_rejected_before_native_mutation(native):
    scene, _ = native
    with pytest.raises(curves.UnsupportedCurveState, match="terminal stored frame"):
        character_edit.character_pose_preserving_curves(
            "rig", [{"frame": 4, "poses": [{"object": "point", "position": [40.5, 1, 2]}]}]
        )
    assert scene.modifications == 0
    assert scene.data.writes == []


@pytest.mark.parametrize("interpolation", ["BEZIER", "CLAMPED_BEZIER"])
def test_curve_preserving_pose_keeps_easing_and_baked_external_samples(native, interpolation):
    scene, _ = native
    for section in scene.tracks["point-track"].sections.values():
        section.interval.interpolation = enum(interpolation)
    before = checkpoint(scene)
    result = character_edit.character_pose_preserving_curves(
        "rig", [{"frame": 2, "poses": [{"object": "point", "position": [20.5, 1, 2]}]}]
    )
    after = checkpoint(scene)
    assert after["tracks"] == before["tracks"]
    assert after["channels"]["data:external"] == before["channels"]["data:external"]
    assert after["channels"]["data:finger-rot"] == before["channels"]["data:finger-rot"]
    assert result["curve_preservation"]["verified"] is True
    impact = result["edit_impact"]["frames"][0]["direct"][0]
    assert (impact["role"], impact["slot"], impact["target_error"]) == ("left_hand", "center", 0)


def test_protected_change_fails_before_success_receipt(native):
    scene, _ = native
    scene.corrupt_external = True
    with pytest.raises(curves.CharacterVerificationError, match="Protected") as failure:
        character_edit.character_pose_preserving_curves(
            "rig", [{"frame": 2, "poses": [{"object": "point", "position": [20.5, 1, 2]}]}]
        )
    assert failure.value.execution_state == "verification_failed"


def test_injected_native_interpolator_failure_never_claims_success(native):
    scene, _ = native
    scene.interpolation_failure = "injected native interpolation failure"
    with pytest.raises(RuntimeError, match="injected native interpolation failure") as failure:
        character_edit.character_retime(
            "rig", [{"source": 0, "target": 0}, {"source": 2, "target": 1}, {"source": 4, "target": 4}]
        )
    assert failure.value.execution_state == "unknown"
    assert scene.modifications == 1


def test_contact_profile_pins_only_stationary_support_slots():
    slots = ("center", "heel", "toe", "toe_direction", "toe_orientation")
    roles = {side + "_foot": {slot: side + "_" + slot for slot in slots} for side in ("left", "right")}
    first = {identifier: [0.0, 0.0, 0.0] for group in roles.values() for identifier in group.values()}
    last = copy.deepcopy(first)
    last["right_heel"] = [0.0, 2.0, 0.0]
    rig = SimpleNamespace(roles=roles)
    anchors = character_edit._contact_targets(rig, {0: first, 4: last})
    assert len(anchors) == 8
    assert "right_heel" not in anchors
    last["left_heel"] = [0.0, 0.1, 0.0]
    with pytest.raises(ValueError, match="stationary"):
        character_edit._contact_targets(rig, {0: first, 4: last})


def test_contact_baking_expects_native_step_for_adjacent_keys_and_keeps_other_metadata_exact(native):
    scene, _ = native
    before = checkpoint(scene)
    saved = copy.deepcopy(before)
    mapping = [{"source": 0, "target": 0}, {"source": 2, "target": 1}, {"source": 4, "target": 4}]
    expected = character_edit._retime_expected_tracks(
        before, [Identifier("point-track"), Identifier("finger-track")], mapping, {Identifier("point-track")}
    )
    foot = expected["point-track"]
    assert foot["keys"] == [0, 1, 4]
    assert foot["sections"]["0"]["interpolation"] == "STEP"
    assert foot["sections"]["1"]["interpolation"] == "FIXED"
    assert foot["sections"]["4"] == before["tracks"]["point-track"]["sections"]["4"]
    for change in mapping:
        assert (
            foot["sections"][str(change["target"])]["key"]
            == before["tracks"]["point-track"]["sections"][str(change["source"])]["key"]
        )
        assert (
            expected["finger-track"]["sections"][str(change["target"])]
            == before["tracks"]["finger-track"]["sections"][str(change["source"])]
        )
    assert expected["external-track"] == before["tracks"]["external-track"]
    assert before == saved


def test_semantic_offsets_keep_validated_role_labels_without_rebinding(native, monkeypatch):
    scene, rig = native
    rig.roles = {"left_hand": {"center": "point"}}
    resolved = []

    def resolve(scene, identifier, **options):
        resolved.append(options)
        return rig

    monkeypatch.setattr(character_edit, "_resolve", resolve)
    result = character_edit.character_offset_preserving_curves(
        "rig", [2], offsets={"left_hand": [0.25, 0, 0]}
    )
    assert len(resolved) == 1
    assert resolved[0]["semantics"] is True
    assert result["edit_impact"]["frames"][0]["direct"][0]["role"] == "left_hand"
    assert scene.data.values["point-pos", 2] == [20.25, 1, 2]


def test_motion_inspection_uses_native_masses_and_center_targets(native, monkeypatch):
    scene, rig = native
    rig.owned.update({"body-a", "body-b", "center"})
    owners = {"rigid-a": "body-a", "rigid-b": "body-b", "center-behavior": "center"}
    scene.behaviours.get_behaviours = lambda kind: (
        ["rigid-a", "rigid-b"] if kind == "RigidBody" else ["center-behavior"]
    )
    scene.behaviours.get_behaviour_owner = lambda behavior: owners[behavior]
    scene.behaviours.get_behaviour_data = lambda behavior, field: Identifier(behavior + ":" + field)
    for frame in range(5):
        scene.data.values["rigid-a:mass", frame] = 2.0
        scene.data.values["rigid-b:mass", frame] = 1.0
        scene.data.values["center-behavior:pos_global_target", frame] = [5.0, 0.0, 0.0]
    positions = {"body-a": [3.0, 0.0, 0.0], "body-b": [9.0, 0.0, 0.0], "center": [5.0, 0.0, 0.0]}
    monkeypatch.setattr(
        character_edit.animation, "_read", lambda scene, owner, frame, space: {"position": positions[owner]}
    )
    monkeypatch.setattr(
        curves,
        "connection_geometry",
        lambda scene, rig: {
            "connection_count": 1,
            "max_connection_anchor_error": 0.004,
            "frames": [{"frame": frame, "max_connection_anchor_error": frame * 0.001} for frame in range(5)],
        },
    )
    result = character_edit.character_motion_inspect("rig")
    assert result["rigid_body_count"] == 2
    assert result["frames"][0]["total_mass"] == 3.0
    assert result["frames"][0]["mass_weighted_center_of_mass"] == [5.0, 0.0, 0.0]
    assert result["frames"][0]["native_centers_of_mass"][0]["target"] == [5.0, 0.0, 0.0]
    assert result["tracks"]["point-track"]["sections"]["2"]["key"]["left_weight_velocity"] == 0.2
    assert scene.modifications == 0


def test_edit_impact_reports_solver_coupled_native_movement():
    rig = SimpleNamespace(
        bindings={
            "hand": SimpleNamespace(role="left_hand", slot="center"),
            "elbow": SimpleNamespace(role="left_elbow", slot="bend"),
        }
    )
    report = character_edit._impact(
        rig,
        {2: {"hand": [0, 0, 0], "elbow": [1, 0, 0]}},
        {2: {"hand": [0.5, 0, 0]}},
        {2: {"hand": [0.5, 0, 0], "elbow": [1.1, 0, 0]}},
    )
    assert report["frames"][0]["direct"][0]["role"] == "left_hand"
    assert report["frames"][0]["solver_coupled"][0]["slot"] == "bend"
