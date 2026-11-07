import copy
import math
import sys
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import state


class Identifier(str):
    def to_string(self):
        return str(self)

    def is_null(self):
        return not self


def checkpoint(value=1.0, setting=False):
    return {
        "format_version": 1,
        "count": 2,
        "objects": {"object": {"name": "Control", "type": "Point", "behaviours": {}}},
        "channels": {
            "data:position": {
                "id": "position",
                "owner": "object",
                "name": "position",
                "kind": "data",
                "mode": "Animation",
                "description": None,
                "values": [value, value],
            },
            "setting:mode": {
                "id": "mode",
                "owner": "object",
                "name": "mode",
                "kind": "setting",
                "mode": "Static",
                "description": None,
                "setting_type": 0,
                "value": setting,
            },
        },
        "tracks": {},
        "auxiliary": {},
        "external": {},
    }


@pytest.fixture
def native_model(monkeypatch):
    obj, did, sid, behaviour = map(Identifier, ("object", "position", "mode", "behaviour"))
    model = checkpoint()
    props = {
        "parent": ("ObjectType", Identifier("")),
        "position": ("DataType", did),
        "mode": ("SettingType", sid),
        "description": ("StringType", "actual text, not a hash"),
    }
    behaviours = SimpleNamespace(
        get_behaviours=lambda _: [behaviour],
        get_behaviour_property_names=lambda _: list(props),
        get_property_type=lambda _, name: SimpleNamespace(name=props[name][0]),
        get_behaviour_object=lambda _, name: props[name][1],
        get_behaviour_data=lambda _, name: props[name][1],
        get_behaviour_setting=lambda _, name: props[name][1],
        get_behaviour_string=lambda _, name: props[name][1],
        get_behaviour_name=lambda _: "Transform",
        is_hidden=lambda _: False,
    )
    no_clusters = SimpleNamespace(clusters=lambda: [], clusters_bindings=lambda: [])
    viewer = SimpleNamespace(
        get_animation_size=lambda: model["count"],
        get_all_data_id=lambda _: [did],
        get_all_settings_id=lambda _: [sid],
        get_data=lambda _: SimpleNamespace(mode=SimpleNamespace(name="Animation"), name="position"),
        get_setting=lambda _: SimpleNamespace(mode=SimpleNamespace(name="Static"), name="mode", type=0),
        get_data_value=lambda _, frame: model["channels"]["data:position"]["values"][frame],
        get_setting_value=lambda _: model["channels"]["setting:mode"]["value"],
        get_description_id_by_linked_id=lambda _: Identifier(""),
        cluster_viewer=lambda: no_clusters,
        get_description_names=lambda: [],
    )
    layers = SimpleNamespace(
        layers_map=lambda: {},
        folders_map=lambda: {},
        root_id=lambda: Identifier("root"),
        all_layer_ids=lambda: [],
    )
    model_viewer = SimpleNamespace(
        get_objects=lambda: [obj], get_object_name=lambda _: "Control", get_object_type_name=lambda _: "Point"
    )
    scene = SimpleNamespace(
        data_viewer=lambda: viewer,
        model_viewer=lambda: model_viewer,
        behaviour_viewer=lambda: behaviours,
        layers_viewer=lambda: layers,
        assets_manager=lambda: SimpleNamespace(all_ids=lambda: []),
    )
    csc = SimpleNamespace(
        additive_layers=SimpleNamespace(get_manager=lambda _: None),
        layers=SimpleNamespace(),
        model=SimpleNamespace(BehaviourId=Identifier),
    )
    monkeypatch.setitem(sys.modules, "csc", csc)
    return scene, model, props


def test_capture_covers_every_static_setting_animated_frame_and_property(native_model):
    scene, _, _ = native_model
    before = state.capture(scene)
    assert before["count"] == 2
    assert before["channels"]["data:position"]["values"] == [1.0, 1.0]
    assert before["channels"]["setting:mode"]["value"] is False
    properties = before["objects"]["object"]["behaviours"]["behaviour"]["properties"]
    assert properties["description"]["value"] == "actual text, not a hash"
    assert properties["parent"]["value"] == {"__native__": "id", "value": ""}
    assert state.digest(before) == state.digest(state.capture(scene))


def test_tolerance_applies_only_to_animated_data_not_settings_or_metadata():
    before, after = checkpoint(), checkpoint(1.005)
    assert state.compare(before, after)["equivalent"]
    after["channels"]["setting:mode"]["value"] = True
    assert not state.compare(before, after)["equivalent"]
    after = copy.deepcopy(before)
    after["objects"]["object"]["name"] = "Other"
    assert not state.compare(before, after)["equivalent"]
    after = checkpoint(1.02)
    assert not state.compare(before, after)["equivalent"]


def test_static_float_constraints_are_exact():
    first = {"kind": "data", "mode": "Static", "value": 1.0}
    second = {"kind": "data", "mode": "Static", "value": 1.00000001}
    assert not state.channel_equal(first, second)


def test_rotations_are_sign_canonical_but_have_stricter_tolerance():
    class Rotation:
        def __init__(self, sign):
            self.sign = sign

        def to_quaternion(self):
            return SimpleNamespace(w=lambda: self.sign, x=lambda: 0.0, y=lambda: 0.0, z=lambda: 0.0)

    assert state.encode(Rotation(-1.0)) == state.encode(Rotation(1.0))
    left = state.encode(Rotation(1.0))
    right = copy.deepcopy(left)
    right["quaternion_wxyz"][1] = 0.001
    assert not state.equivalent_values(left, right)


class EulerRotation:
    def __init__(self, x, y=0.0, z=0.0):
        self.angles = [x, y, z]

    def to_euler_angles(self):
        return [[value] for value in self.angles]

    def to_quaternion(self):
        x, y, z = (angle / 2 for angle in self.angles)
        cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
        sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
        return SimpleNamespace(
            w=lambda: cx * cy * cz - sx * sy * sz,
            x=lambda: sx * cy * cz + cx * sy * sz,
            y=lambda: cx * sy * cz - sx * cy * sz,
            z=lambda: cx * cy * sz + sx * sy * cz,
        )


def test_rotation_checkpoint_roundtrips_native_euler_winding_without_quaternion_conversion(monkeypatch):
    calls = []

    def from_euler(*angles):
        calls.append(angles)
        return EulerRotation(*angles)

    monkeypatch.setitem(
        sys.modules,
        "csc",
        SimpleNamespace(
            math=SimpleNamespace(
                Rotation=SimpleNamespace(
                    from_euler=from_euler,
                    from_quaternion=lambda *args: pytest.fail(
                        "Euler checkpoints must preserve their native representation"
                    ),
                )
            )
        ),
    )
    original = EulerRotation(2 * math.pi + 0.25, -3 * math.pi / 2, 5 * math.pi)
    encoded = state.encode(original)
    assert encoded["euler_radians"] == original.angles
    assert state.decode(encoded).angles == original.angles
    assert calls == [tuple(original.angles)]


def test_rotation_comparison_preserves_euler_branch_even_for_equivalent_quaternions():
    principal = state.encode(EulerRotation(0.25))
    wound = state.encode(EulerRotation(2 * math.pi + 0.25))
    assert all(
        math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(principal["quaternion_wxyz"], wound["quaternion_wxyz"])
    )
    assert not state.equivalent_values(principal, wound)
    assert not state.equivalent_values(principal, wound, exact=True)


def test_equivalent_tait_bryan_branches_remain_distinct_native_rotation_values():
    original = state.encode(EulerRotation(0.3, 0.5, -0.4))
    other_branch = state.encode(EulerRotation(0.3 + math.pi, math.pi - 0.5, -0.4 + math.pi))
    assert all(
        math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(original["quaternion_wxyz"], other_branch["quaternion_wxyz"])
    )
    assert not state.equivalent_values(original, other_branch)


def test_euler_turn_count_does_not_gain_relative_tolerance_at_large_angles():
    first = {
        "__native__": "rotation",
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "euler_radians": [1e8, 0.0, 0.0],
    }
    other_turn = copy.deepcopy(first)
    other_turn["euler_radians"][0] += 2 * math.pi
    assert not state.equivalent_values(first, other_turn)


@pytest.fixture
def derived_angle_case():
    before = checkpoint()
    channel = before["channels"]["data:position"]
    channel["name"] = "Angle Rotation"
    rotation = state.encode(EulerRotation(0.25))
    channel["values"] = [copy.deepcopy(rotation), copy.deepcopy(rotation)]
    after = copy.deepcopy(before)
    after["channels"]["data:position"]["values"][0]["euler_radians"][0] += 2 * math.pi
    return before, after


@pytest.mark.parametrize(
    "frame,euler_before,quaternion_before,euler_after,quaternion_after",
    [
        (
            1,
            [3.1272382736206055, -3.1415927410125732, -3.1415927410125732],
            [0.999974250793457, -0.007177128456532955, 4.339653969509527e-08, -4.402398445790823e-08],
            [3.127237319946289, -3.141592502593994, 3.1415927410125732],
            [0.999974250793457, -0.007177605293691158, -7.518221423197247e-08, 4.425215749392919e-08],
        ),
        (
            10,
            [2.6741714477539062, 3.141592502593994, -3.1415927410125732],
            [0.9728137850761414, -0.23158884048461914, 6.33223322665799e-08, -6.000750829571189e-08],
            [2.674170970916748, -3.141592502593994, 3.141592502593994],
            [0.9728137254714966, -0.23158906400203705, -9.092988051406792e-08, -5.59609034667119e-08],
        ),
        (
            12,
            [3.14143705368042, 3.1415927410125732, 3.1401376724243164],
            [0.9999997615814209, -7.779990119161084e-05, -1.0031010333477752e-07, -0.000727490521967411],
            [3.141436815261841, -3.141592502593994, 3.140137195587158],
            [0.9999997615814209, -7.791908137733117e-05, -1.3220190453466785e-07, -0.0007277289405465126],
        ),
    ],
)
def test_recovery_comparison_accepts_saved_native_unbound_point_angle_outputs_at_any_frame(
    derived_angle_case, frame, euler_before, quaternion_before, euler_after, quaternion_after
):
    # Native save/reopen rows from Unbaking toe/neck Point solver outputs.
    before, after = derived_angle_case
    old = {"__native__": "rotation", "euler_radians": euler_before, "quaternion_wxyz": quaternion_before}
    new = {"__native__": "rotation", "euler_radians": euler_after, "quaternion_wxyz": quaternion_after}
    for snapshot in (before, after):
        snapshot["count"] = 14
        snapshot["channels"]["data:position"]["values"] = [copy.deepcopy(old) for _ in range(14)]
    after["channels"]["data:position"]["values"][frame] = new
    original = copy.deepcopy((before, after))

    assert not state.equivalent_values(old, new)
    assert not state.channel_equal(before["channels"]["data:position"], after["channels"]["data:position"])
    assert state.compare(before, after)["equivalent"]
    assert state.compare(after, before)["equivalent"]
    assert (before, after) == original


@pytest.mark.parametrize(
    "name,kind",
    [("Local Rotation", "DataType"), ("Angle Rotation", "DataType"), ("Angle Rotation", "DataRangeType")],
)
def test_recovery_comparison_keeps_bound_authored_rotation_winding_strict(derived_angle_case, name, kind):
    before, after = derived_angle_case
    reference = {"__native__": "id", "value": "position"}
    for snapshot in (before, after):
        snapshot["channels"]["data:position"]["name"] = name
        snapshot["objects"]["object"]["behaviours"]["authored"] = {
            "name": "Transform",
            "properties": {
                "local_rotation": {"kind": kind, "value": [reference] if "Range" in kind else reference}
            },
        }
    proof = state.compare(before, after)
    assert not proof["equivalent"]
    assert proof["differences"] == [{"path": "channels/data:position", "reason": "channel state differs"}]


@pytest.mark.parametrize(
    "case",
    ["quaternion", "one_radian", "other_name", "owner_type", "metadata", "static", "setting", "extra_field"],
)
def test_recovery_periodic_equivalence_cannot_hide_authored_or_structural_changes(derived_angle_case, case):
    before, after = derived_angle_case
    if case == "quaternion":
        after["channels"]["data:position"]["values"][0]["quaternion_wxyz"][0] += 0.001
    elif case == "one_radian":
        after["channels"]["data:position"]["values"][0]["euler_radians"][0] += 1.0
    elif case == "metadata":
        after["channels"]["data:position"]["description"] = {"label": "new"}
    elif case == "extra_field":
        before["channels"]["data:position"]["values"][0]["extra"] = 1.0
        after["channels"]["data:position"]["values"][0]["extra"] = 1.0 + 1e-10
    else:
        for snapshot in (before, after):
            channel = snapshot["channels"]["data:position"]
            if case == "other_name":
                channel["name"] = "Local Rotation"
            elif case == "owner_type":
                snapshot["objects"]["object"]["type"] = "Joint"
            elif case == "static":
                channel["mode"] = "Static"
                channel["value"] = channel.pop("values")[0]
            elif case == "setting":
                channel["kind"] = "setting"
                channel["setting_type"] = 0
                snapshot["channels"]["setting:position"] = snapshot["channels"].pop("data:position")
    assert not state.compare(before, after)["equivalent"]


def test_removing_a_binding_cannot_make_authored_rotation_equivalent_to_derived_output(derived_angle_case):
    before, after = derived_angle_case
    before["objects"]["object"]["behaviours"]["authored"] = {
        "name": "Transform",
        "properties": {
            "local_rotation": {"kind": "DataType", "value": {"__native__": "id", "value": "position"}}
        },
    }
    proof = state.compare(before, after)
    assert not proof["equivalent"]
    assert {item["path"] for item in proof["differences"]} == {"channels/data:position", "objects"}


def test_old_quaternion_only_checkpoint_compares_with_new_euler_records_without_losing_other_metadata():
    modern = checkpoint()
    rotation = state.encode(EulerRotation(2 * math.pi + 0.25))
    modern["channels"]["data:position"]["values"] = [rotation, copy.deepcopy(rotation)]
    modern["channels"]["data:static_rotation"] = {
        "owner": "object",
        "id": "static_rotation",
        "name": "rotation",
        "kind": "data",
        "mode": "Static",
        "value": copy.deepcopy(rotation),
    }
    modern["objects"]["object"]["behaviours"]["constant"] = {
        "name": "Constant",
        "hidden": False,
        "properties": {"rotation": {"kind": "RotationType", "value": copy.deepcopy(rotation)}},
    }
    legacy = copy.deepcopy(modern)
    for value in legacy["channels"]["data:position"]["values"]:
        value.pop("euler_radians")
    legacy["channels"]["data:static_rotation"]["value"].pop("euler_radians")
    legacy["objects"]["object"]["behaviours"]["constant"]["properties"]["rotation"]["value"].pop(
        "euler_radians"
    )
    assert state.compare(legacy, modern)["equivalent"]
    assert state.compare(modern, legacy)["equivalent"]
    assert state.equivalent_values(
        state._structural_identity(legacy), state._structural_identity(modern), exact=True
    )
    modern["objects"]["object"]["name"] = "Different"
    assert not state.compare(legacy, modern)["equivalent"]


def test_legacy_rotation_decode_remains_compatible_and_invalid_euler_is_rejected(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "csc",
        SimpleNamespace(
            math=SimpleNamespace(
                Rotation=SimpleNamespace(
                    from_quaternion=lambda *values: ("legacy", values),
                    from_euler=lambda *values: pytest.fail("Invalid Euler data must not reach native code"),
                )
            )
        ),
    )
    assert state.decode({"__native__": "rotation", "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}) == (
        "legacy",
        (1.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(state.StateError, match="three finite radians"):
        state.decode(
            {
                "__native__": "rotation",
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "euler_radians": [True, 0.0, 0.0],
            }
        )


def test_section_metadata_keeps_ai_label_tangents_and_easing_weights():
    common = SimpleNamespace(ik_fk=SimpleNamespace(name="IK"), fixation=SimpleNamespace(name="Fulcrum"))
    section = SimpleNamespace(
        interval=SimpleNamespace(
            interpolation=SimpleNamespace(name="BEZIER"),
            common=common,
            description_for_ai_interpolation="walk naturally",
        ),
        key=SimpleNamespace(
            common=common,
            label=Identifier("label"),
            tangents=SimpleNamespace(name="UserDefined"),
            left_weight_velocity=0.75,
            right_weight_velocity=1.25,
        ),
    )
    encoded = state.section_state(section)
    assert encoded["ai_description"] == "walk naturally"
    assert encoded["key"]["label"] == {"__native__": "id", "value": "label"}
    assert encoded["key"]["tangents"] == "UserDefined"
    assert encoded["key"]["left_weight_velocity"] == 0.75
    assert encoded["key"]["right_weight_velocity"] == 1.25


def test_capture_limits_refuse_before_any_mutation(native_model):
    scene, model, _ = native_model
    model["count"] = 513
    with pytest.raises(state.StateError, match="at most"):
        state.capture(scene)


def test_unknown_native_values_and_nonfinite_data_are_rejected(native_model):
    scene, model, _ = native_model
    with pytest.raises(state.StateError, match="cannot encode"):
        state.encode(object())
    model["channels"]["data:position"]["values"][1] = math.nan
    with pytest.raises(state.StateError, match="nonfinite"):
        state.capture(scene)


def test_validate_requires_all_frames_and_json_model():
    bad = checkpoint()
    bad["channels"]["data:position"]["values"].pop()
    with pytest.raises(state.StateError, match="missing stored frames"):
        state.validate(bad)


def test_manual_restore_refuses_changed_topology_before_native_edit(monkeypatch):
    before, actual = checkpoint(), checkpoint()
    actual["objects"]["unexpected"] = actual["objects"]["object"]
    monkeypatch.setattr(state, "capture", lambda _: actual)
    scene = SimpleNamespace(modify_update=lambda *args: pytest.fail("Must not mutate unsupported topology"))
    with pytest.raises(state.StateError, match="Topology"):
        state.restore(scene, before)


def test_manual_restore_writes_all_frames_and_static_settings_then_compares_complete_model(native_model):
    scene, model, _ = native_model
    before = state.capture(scene)
    model["channels"]["data:position"]["values"] = [8.0, 9.0]
    model["channels"]["setting:mode"]["value"] = True

    def set_data(identifier, frame, value):
        model["channels"]["data:position"]["values"][frame] = value

    def set_setting(identifier, value):
        model["channels"]["setting:mode"]["value"] = value

    data_editor = SimpleNamespace(set_data_value=set_data, set_setting_value=set_setting)
    editor = SimpleNamespace(data_editor=lambda: data_editor, layers_editor=lambda: SimpleNamespace())
    interpolator = SimpleNamespace(reload=lambda: None, interpolate=lambda: None)
    updater = SimpleNamespace(
        run_update=lambda *args: None, get_interpolator=lambda: interpolator, generate_update=lambda: None
    )

    def modify(label, callback):
        callback(editor, None, updater)
        return True

    scene.modify_update = modify
    result = state.restore(scene, before)
    assert result["restored"]
    assert result["verification"]["equivalent"]
    assert model["channels"]["data:position"]["values"] == [1.0, 1.0]
    assert model["channels"]["setting:mode"]["value"] is False


@pytest.mark.parametrize("interpolation", ["BEZIER", "CLAMPED_BEZIER"])
def test_manual_track_restore_uses_native_key_helpers_and_scalar_metadata_not_detached_sections(
    monkeypatch, interpolation
):
    def metadata(weight):
        return {
            "interpolation": interpolation,
            "interval_common": {"ik_fk": "IK", "fixation": "Free"},
            "ai_description": "authored interval",
            "key": {
                "common": {"ik_fk": "IK", "fixation": "Free"},
                "label": None,
                "tangents": "Continuous",
                "left_weight_velocity": weight,
                "right_weight_velocity": 2.0,
            },
        }

    wanted = {
        "format_version": 1,
        "count": 4,
        "objects": {},
        "channels": {},
        "auxiliary": {},
        "external": {},
        "tracks": {
            "track": {
                "name": "Body",
                "parent": "root",
                "objects": [],
                "locked": False,
                "visible": True,
                "keys": [0, 2, 3],
                "cycles": [],
                "sections": {"0": metadata(0.5), "2": metadata(0.75), "3": metadata(1.0)},
            }
        },
    }
    actual = copy.deepcopy(wanted)
    actual["tracks"]["track"]["sections"].pop("2")
    actual["tracks"]["track"]["sections"]["1"] = metadata(9.0)
    actual["tracks"]["track"]["keys"] = [0, 1, 3]
    actual["tracks"]["track"]["sections"]["0"] = metadata(8.0)
    actual["tracks"]["track"]["sections"]["0"]["interpolation"] = (
        "BEZIER" if interpolation == "CLAMPED_BEZIER" else "CLAMPED_BEZIER"
    )
    calls = []

    def enum(name):
        return SimpleNamespace(name=name)

    native = SimpleNamespace(
        **{
            name: SimpleNamespace(__members__={item: enum(item) for item in values})
            for name, values in {
                "Interpolation": ["BEZIER", "CLAMPED_BEZIER"],
                "IkFk": ["IK"],
                "Fixation": ["Free"],
                "Tangents": ["Continuous"],
            }.items()
        }
    )
    native.Section = lambda: pytest.fail("Detached native sections are unsafe")
    monkeypatch.setitem(sys.modules, "csc", SimpleNamespace(layers=SimpleNamespace(layer=native)))

    def section(value):
        def common(value):
            return SimpleNamespace(ik_fk=enum(value["ik_fk"]), fixation=enum(value["fixation"]))

        key = value["key"]
        return SimpleNamespace(
            interval=SimpleNamespace(
                interpolation=enum(value["interpolation"]),
                common=common(value["interval_common"]),
                description_for_ai_interpolation=value["ai_description"],
            ),
            key=SimpleNamespace(
                common=common(key["common"]),
                label=key["label"],
                tangents=enum(key["tangents"]),
                left_weight_velocity=key["left_weight_velocity"],
                right_weight_velocity=key["right_weight_velocity"],
            ),
        )

    class Layers:
        def set_sections(self, *args):
            pytest.fail("Bulk set_sections must not enter native recovery")

        def unset_section(self, frame, identifier):
            calls.append(("unset", frame))
            actual["tracks"][identifier]["sections"].pop(str(frame), None)

        def set_fixed_interpolation_or_key_if_need(self, identifier, frame, is_key):
            assert is_key
            calls.append(("native_key", frame))
            actual["tracks"][identifier]["sections"][str(frame)] = metadata(1.0)

        def normalize_sections(self, scene):
            calls.append(("normalize",))
            actual["tracks"]["track"]["keys"] = sorted(map(int, actual["tracks"]["track"]["sections"]))

        def change_section(self, frame, identifier, callback):
            value = section(actual["tracks"][identifier]["sections"][str(frame)])
            callback(value)
            actual["tracks"][identifier]["sections"][str(frame)] = state.section_state(value)

        def set_locked_for_layer(self, value, identifier):
            actual["tracks"][identifier]["locked"] = value

        def set_visible_for_layer(self, value, identifier):
            actual["tracks"][identifier]["visible"] = value

        def set_name(self, value, identifier):
            actual["tracks"][identifier]["name"] = value

    editor = SimpleNamespace(layers_editor=lambda: Layers(), data_editor=lambda: SimpleNamespace())
    interpolator = SimpleNamespace(reload=lambda: None, interpolate=lambda: None)
    updater = SimpleNamespace(generate_update=lambda: None, get_interpolator=lambda: interpolator)
    scene = SimpleNamespace(
        model_viewer=lambda: SimpleNamespace(get_objects=lambda: []),
        data_viewer=lambda: SimpleNamespace(get_all_data_id=lambda _: [], get_all_settings_id=lambda _: []),
        layers_viewer=lambda: SimpleNamespace(all_layer_ids=lambda: ["track"]),
    )

    def modify(label, callback):
        callback(editor, None, updater)
        return True

    scene.modify_update = modify
    monkeypatch.setattr(state, "capture", lambda _: copy.deepcopy(actual))
    assert state.restore(scene, wanted)["restored"]
    assert actual == wanted
    assert ("native_key", 2) in calls and ("unset", 1) in calls


def test_saved_curve_type_unavailable_in_native_build_never_reaches_track_mutation(monkeypatch):
    metadata = {
        "interpolation": "CLAMPED_BEZIER",
        "interval_common": {"ik_fk": "IK", "fixation": "Free"},
        "ai_description": "legacy saved interval",
        "key": None,
    }
    wanted = {
        "format_version": 1,
        "count": 2,
        "objects": {},
        "channels": {},
        "auxiliary": {},
        "external": {},
        "tracks": {
            "track": {
                "objects": [],
                "keys": [],
                "cycles": [],
                "sections": {"0": metadata},
            }
        },
    }
    actual = copy.deepcopy(wanted)
    actual["tracks"]["track"]["sections"]["0"]["interpolation"] = "BEZIER"
    native = SimpleNamespace(Interpolation=SimpleNamespace(__members__={"BEZIER": object()}))
    monkeypatch.setitem(sys.modules, "csc", SimpleNamespace(layers=SimpleNamespace(layer=native)))
    monkeypatch.setattr(state, "capture", lambda _: copy.deepcopy(actual))
    scene = SimpleNamespace(modify_update=lambda *args: pytest.fail("Unavailable curve must not be edited"))
    with pytest.raises(state.StateError, match="exact curve metadata") as caught:
        state.restore(scene, wanted)
    assert caught.value.code == "UNSUPPORTED_SECTION_RESTORE"
    assert caught.value.execution_state == "not_started"


@pytest.fixture
def authored_character(monkeypatch):
    from mcp_for_cascadeur.host import characters, curves

    wanted = {
        "format_version": 1,
        "count": 3,
        "tracks": {},
        "auxiliary": {},
        "external": {},
        "objects": {
            name: {"name": name, "type": "Point", "behaviours": {}}
            for name in ("rig", "point", "joint", "finger")
        },
        "channels": {},
    }
    wanted["objects"]["rig"]["behaviours"]["info"] = {
        "name": "RigInfo",
        "hidden": False,
        "properties": {},
    }
    for identifier, owner, value in (
        ("p", "point", 1.0),
        ("jg", "joint", 2.0),
        ("jl", "joint", 3.0),
        ("f", "finger", 4.0),
    ):
        wanted["channels"]["data:" + identifier] = {
            "id": identifier,
            "owner": owner,
            "name": identifier,
            "kind": "data",
            "mode": "Animation",
            "values": [value] * 3,
        }
    wanted["channels"]["setting:s"] = {
        "id": "s",
        "owner": "point",
        "name": "s",
        "kind": "setting",
        "mode": "Animation",
        "values": [False] * 3,
    }
    actual = copy.deepcopy(wanted)
    actual["channels"]["data:p"]["values"][0] = 5.0
    actual["channels"]["data:f"]["values"][0] = 6.0
    for identifier in ("jg", "jl"):
        actual["channels"]["data:" + identifier]["values"][0] = 999.0
    actual["channels"]["setting:s"]["values"][0] = True
    rig = SimpleNamespace(
        owned=set(wanted["objects"]),
        controls={"point": "point"},
        bindings={"point": SimpleNamespace(position_data="p")},
        fingers={"left": {"index1": SimpleNamespace(rotation_data="f")}},
    )
    monkeypatch.setattr(characters, "resolve_character", lambda *args, **kwargs: rig)
    monkeypatch.setattr(curves, "verify_supported", lambda _: {"stored_frames": 3})
    monkeypatch.setitem(sys.modules, "csc", SimpleNamespace(model=SimpleNamespace(BehaviourId=Identifier)))
    monkeypatch.setattr(state, "capture", lambda _: copy.deepcopy(actual))
    writes, solves = [], []

    def set_data(identifier, frame, value):
        assert identifier in {"p", "f"}, "Derived channels must never be native inputs"
        writes.append((identifier, frame))
        actual["channels"]["data:" + identifier]["values"][frame] = value

    def solve(identifiers, frame):
        assert frame < 2, "The terminal guard has no following solver interval"
        solves.append((identifiers, frame))
        if "p" in identifiers:
            value = actual["channels"]["data:p"]["values"][frame]
            actual["channels"]["data:jg"]["values"][frame] = value * 2
            actual["channels"]["data:jl"]["values"][frame] = value * 3

    data_editor = SimpleNamespace(
        set_data_value=set_data,
        set_setting_value=lambda identifier, frame, value: actual["channels"]["setting:" + identifier][
            "values"
        ].__setitem__(frame, value),
    )
    updater = SimpleNamespace(
        generate_update=lambda: None,
        run_update=solve,
        get_interpolator=lambda: SimpleNamespace(reload=lambda: None, interpolate=lambda: None),
    )
    scene = SimpleNamespace(
        model_viewer=lambda: SimpleNamespace(get_objects=lambda: list(wanted["objects"])),
        data_viewer=lambda: SimpleNamespace(
            get_all_data_id=lambda owner: [
                record["id"]
                for record in wanted["channels"].values()
                if record["owner"] == owner and record["kind"] == "data"
            ],
            get_all_settings_id=lambda owner: [
                record["id"]
                for record in wanted["channels"].values()
                if record["owner"] == owner and record["kind"] == "setting"
            ],
        ),
        layers_viewer=lambda: SimpleNamespace(all_layer_ids=lambda: []),
    )

    def modify(label, callback):
        callback(SimpleNamespace(data_editor=lambda: data_editor, layers_editor=lambda: None), None, updater)
        return True

    scene.modify_update = modify
    return scene, wanted, actual, writes, solves


def test_character_restore_authors_points_finger_locals_settings_and_regenerates_joint_outputs(
    authored_character,
):
    scene, wanted, actual, writes, solves = authored_character
    result = state.restore(scene, wanted)
    assert result["verification"]["equivalent"]
    assert actual == wanted
    assert set(writes) == {("p", 0), ("f", 0)}
    assert solves == [({"p"}, 0), ({"f"}, 0)]


def test_format_one_character_checkpoint_recovers_with_clamped_curves_and_exact_metadata(authored_character):
    scene, wanted, actual, writes, solves = authored_character
    metadata = {
        "interpolation": "CLAMPED_BEZIER",
        "interval_common": {"ik_fk": "IK", "fixation": "Fulcrum"},
        "ai_description": "original clamp interval",
        "key": {
            "common": {"ik_fk": "IK", "fixation": "Free"},
            "label": 7,
            "tangents": "Continuous",
            "left_weight_velocity": 0.31,
            "right_weight_velocity": 0.67,
        },
    }
    wanted["tracks"]["body"] = {
        "objects": ["point"],
        "keys": [0],
        "sections": {"0": metadata},
        "cycles": [],
    }
    actual["tracks"] = copy.deepcopy(wanted["tracks"])
    result = state.restore(scene, wanted)
    assert result["verification"]["equivalent"]
    assert actual == wanted
    assert actual["format_version"] == 1
    assert actual["tracks"]["body"]["sections"]["0"] == metadata
    assert set(writes) == {("p", 0), ("f", 0)}
    assert solves == [({"p"}, 0), ({"f"}, 0)]


def test_manual_restore_registers_one_fresh_native_history_state_and_accepts_only_after_verification(
    authored_character, monkeypatch
):
    from mcp_for_cascadeur.host import recovery

    scene, wanted, _, _, _ = authored_character
    calls = []
    monkeypatch.setattr(recovery, "note_native_attempt", lambda target: calls.append(("attempt", target)))
    monkeypatch.setattr(
        recovery, "note_native_commit", lambda target, *, checkpoint: calls.append(("commit", checkpoint))
    )
    monkeypatch.setattr(
        recovery, "accept_verified_checkpoint", lambda target, snapshot: calls.append(("verified", snapshot))
    )
    assert state.restore(scene, wanted)["verification"]["equivalent"]
    assert [call[0] for call in calls] == ["attempt", "commit", "verified"]
    assert calls[0][1] is scene
    assert calls[1][1] is calls[2][1]
    assert calls[1][1] == wanted
    assert calls[1][1] is not wanted


def test_restore_mismatch_records_native_history_but_never_accepts_stale_or_unverified_state(
    authored_character, monkeypatch
):
    from mcp_for_cascadeur.host import recovery

    scene, wanted, actual, _, _ = authored_character
    original = scene.modify_update

    def native_drift(*args):
        accepted = original(*args)
        actual["channels"]["data:jg"]["values"][1] += 0.25
        return accepted

    scene.modify_update = native_drift
    commits = []
    monkeypatch.setattr(
        recovery, "note_native_commit", lambda target, *, checkpoint: commits.append(checkpoint)
    )
    monkeypatch.setattr(
        recovery,
        "accept_verified_checkpoint",
        lambda *args: pytest.fail("Unverified state must remain fenced"),
    )
    with pytest.raises(state.StateError, match="complete recorded checkpoint") as caught:
        state.restore(scene, wanted)
    assert caught.value.execution_state == "recovery_required"
    assert len(commits) == 1 and commits[0] == actual
    assert not state.compare(wanted, commits[0])["equivalent"]


def test_character_restore_reasserts_unchanged_inputs_when_only_derived_state_is_damaged(authored_character):
    scene, wanted, actual, writes, solves = authored_character
    actual["channels"]["data:p"] = copy.deepcopy(wanted["channels"]["data:p"])
    actual["channels"]["data:f"] = copy.deepcopy(wanted["channels"]["data:f"])
    actual["channels"]["setting:s"] = copy.deepcopy(wanted["channels"]["setting:s"])
    assert state.restore(scene, wanted)["verification"]["equivalent"]
    assert writes == []
    assert solves[0] == ({"p"}, 0)


def test_character_restore_rejects_changed_terminal_inputs_before_mutation(authored_character):
    scene, wanted, actual, writes, solves = authored_character
    actual["channels"]["data:p"]["values"][2] = 42.0
    scene.modify_update = lambda *args: pytest.fail("Terminal rejection must precede native mutation")
    with pytest.raises(state.StateError, match="following solver interval"):
        state.restore(scene, wanted)
    assert writes == solves == []


def test_unknown_constrained_animation_restore_fails_closed_before_mutation(monkeypatch):
    wanted, actual = checkpoint(), checkpoint(9.0)
    wanted["objects"]["object"]["behaviours"]["constraint"] = {
        "name": "UnknownConstraint",
        "hidden": False,
        "properties": {},
    }
    actual["objects"] = copy.deepcopy(wanted["objects"])
    monkeypatch.setattr(state, "capture", lambda _: actual)
    scene = SimpleNamespace(modify_update=lambda *args: pytest.fail("Unknown inputs must not be written"))
    with pytest.raises(state.StateError, match="Unknown constrained"):
        state.restore(scene, wanted)


@pytest.mark.parametrize("unsupported", ["custom_tangents", "point_fk"])
def test_character_restore_rejects_unverified_checkpoint_curves_before_mutation(
    authored_character, unsupported
):
    scene, wanted, actual, writes, solves = authored_character
    section = {
        "interpolation": "BEZIER",
        "interval_common": {"ik_fk": "FK" if unsupported == "point_fk" else "IK", "fixation": "Free"},
        "ai_description": "",
        "key": {
            "common": {"ik_fk": "IK", "fixation": "Free"},
            "label": None,
            "tangents": "UserDefined" if unsupported == "custom_tangents" else "Continuous",
            "left_weight_velocity": 1.0,
            "right_weight_velocity": 1.0,
        },
    }
    wanted["tracks"]["body"] = {
        "name": "Body",
        "parent": "root",
        "objects": ["point"],
        "locked": False,
        "visible": True,
        "cycles": [],
        "keys": [0],
        "sections": {"0": section},
    }
    actual["tracks"] = copy.deepcopy(wanted["tracks"])
    scene.modify_update = lambda *args: pytest.fail(
        "Unsupported checkpoint curves must not reach native editing"
    )
    with pytest.raises(state.StateError):
        state.restore(scene, wanted)
    assert writes == solves == []
