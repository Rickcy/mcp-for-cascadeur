import copy

import pytest

from mcp_for_cascadeur.host import state
from mcp_for_cascadeur.host.job_verification import protected_report


def _channel(owner, identifier, *, mode="Animation", kind="data", value=0.0, count=5):
    record = {
        "owner": owner,
        "id": identifier,
        "name": identifier,
        "kind": kind,
        "mode": mode,
        "description": None,
    }
    if kind == "setting":
        record["setting_type"] = 0
    record["values" if mode == "Animation" else "value"] = (
        [copy.deepcopy(value) for _ in range(count)] if mode == "Animation" else value
    )
    return record


def _section():
    return {
        "interpolation": "CLAMPED_BEZIER",
        "interval_common": {"ik_fk": "IK", "fixation": "Free"},
        "ai_description": "original",
        "key": {
            "common": {"ik_fk": "IK", "fixation": "Free"},
            "label": 3,
            "tangents": "Continuous",
            "left_weight_velocity": 0.31,
            "right_weight_velocity": 0.67,
        },
    }


def _track(owners):
    return {
        "objects": owners,
        "name": "Original track",
        "parent": "root",
        "locked": False,
        "visible": True,
        "cycles": [],
        "keys": [0, 2, 4],
        "sections": {str(frame): _section() for frame in (0, 2, 4)},
    }


@pytest.fixture
def native_case():
    before = {
        "format_version": 1,
        "count": 5,
        "objects": {
            identifier: {"name": identifier, "type": "Point", "behaviours": {}}
            for identifier in ("point", "finger", "prop")
        },
        "channels": {
            "data:point": _channel("point", "point", value=[0.0, 1.0, 2.0]),
            "data:finger": _channel("finger", "finger", value=0.0),
            "data:prop": _channel("prop", "prop", value=[3.0, 4.0, 5.0]),
            "data:mass": _channel("point", "mass", mode="Static", value=75.0),
            "setting:ik": _channel("point", "ik", kind="setting", value=False),
            "setting:visible": _channel("prop", "visible", kind="setting", mode="Static", value=True),
        },
        "tracks": {
            "body": _track(["point"]),
            "hand": _track(["finger"]),
            "prop": _track(["prop"]),
        },
        "auxiliary": {"folders": {"root": {"children": ["body", "hand", "prop"]}}},
        "external": {"assets": {"mesh": {"type": "Mesh", "payload_sha256": "original"}}},
    }
    plan = {"operation": "unbake", "tracks": ["body"], "first": 1, "last": 3, "expected_count": 5}
    return before, copy.deepcopy(before), plan


def test_owned_animated_targets_and_settings_can_change_only_inside_the_declared_interval(native_case):
    before, after, plan = native_case
    after["channels"]["data:point"]["values"][2] = [20.0, 21.0, 22.0]
    after["channels"]["setting:ik"]["values"][1] = True
    after["tracks"]["body"]["sections"]["2"]["interpolation"] = "BEZIER"
    proof = protected_report(before, after, plan, {})
    assert proof["matches"] is True
    assert proof["error_count"] == 0
    assert proof["counts"]["static_channels"] == 2
    assert proof["counts"]["editable_samples"] == 6


@pytest.mark.parametrize("identifier", ["data:mass", "setting:visible"])
def test_even_small_static_or_setting_changes_are_rejected(native_case, identifier):
    before, after, plan = native_case
    old = after["channels"][identifier]["value"]
    after["channels"][identifier]["value"] = old + 1e-10 if type(old) is float else False
    proof = protected_report(before, after, plan, {})
    assert proof["matches"] is False
    assert proof["errors"][0]["path"] == f"channels/{identifier}/value"


@pytest.mark.parametrize("channel,frame", [("data:finger", 2), ("data:prop", 2), ("data:point", 0)])
def test_foreign_fingers_props_and_outside_interval_targets_remain_protected(native_case, channel, frame):
    before, after, plan = native_case
    after["channels"][channel]["values"][frame] = 99.0
    proof = protected_report(before, after, plan, {})
    assert proof["matches"] is False
    assert any(error["path"] == f"channels/{channel}/values/{frame}" for error in proof["errors"])


def test_explicit_character_ownership_can_author_fingers_without_unlocking_other_objects(native_case):
    before, after, plan = native_case
    plan["owned_objects"] = ["point", "finger"]
    after["channels"]["data:finger"]["values"][2] = 1.0
    assert protected_report(before, after, plan, {})["matches"]
    after["channels"]["data:prop"]["values"][2] = [99.0, 0.0, 0.0]
    assert not protected_report(before, after, plan, {})["matches"]


@pytest.mark.parametrize(
    "field,value", [("owner", "point"), ("name", "renamed"), ("description", {"id": "new"})]
)
def test_channel_metadata_cannot_be_reassigned_to_evade_scope(native_case, field, value):
    before, after, plan = native_case
    after["channels"]["data:finger"][field] = value
    proof = protected_report(before, after, plan, {})
    assert not proof["matches"]
    assert any(error["path"] == "channels/data:finger" for error in proof["errors"])


@pytest.mark.parametrize("frame", [0, 4])
def test_protected_animated_settings_are_exact_even_below_data_tolerance(native_case, frame):
    before, after, plan = native_case
    before["channels"]["setting:ik"]["values"] = [1.0] * 5
    after["channels"]["setting:ik"]["values"] = [1.0] * 5
    after["channels"]["setting:ik"]["values"][frame] += 1e-10
    assert not protected_report(before, after, plan, {})["matches"]


def test_protected_data_uses_the_shared_native_readback_tolerance(native_case):
    before, after, plan = native_case
    after["channels"]["data:prop"]["values"][2][0] += 0.005
    assert protected_report(before, after, plan, {})["matches"]
    after["channels"]["data:prop"]["values"][2][0] += 0.02
    assert not protected_report(before, after, plan, {})["matches"]


@pytest.fixture
def periodic_guard_case(native_case):
    before, after, plan = native_case
    # Native Unbaking readback from the Cascy left-hand terminal guard.
    old = {
        "__native__": "rotation",
        "euler_radians": [2.9315617084503174, -3.141589641571045, 3.141592264175415],
        "quaternion_wxyz": [
            0.994490921497345,
            -0.1048225536942482,
            -1.5181224171101348e-06,
            -3.577076768124243e-08,
        ],
    }
    new = {
        "__native__": "rotation",
        "euler_radians": [2.9315593242645264, 3.1415913105010986, -3.141592502593994],
        "quaternion_wxyz": [
            0.9944908022880554,
            -0.10482373833656311,
            6.757586561434437e-07,
            4.688182286827214e-09,
        ],
    }
    channel = _channel("point", "derived-angle", value=old)
    channel["name"] = "Angle Rotation"
    before["channels"]["data:derived-angle"] = channel
    after["channels"]["data:derived-angle"] = copy.deepcopy(channel)
    after["channels"]["data:derived-angle"]["values"][-1] = new
    return before, after, plan


def test_unbound_unbake_terminal_angle_output_accepts_native_periodic_euler_readback(periodic_guard_case):
    before, after, plan = periodic_guard_case
    old = before["channels"]["data:derived-angle"]["values"][-1]
    new = after["channels"]["data:derived-angle"]["values"][-1]
    original = copy.deepcopy((before, after, plan))

    assert not state.equivalent_values(old, new)
    proof = protected_report(before, after, plan, {})

    assert proof["matches"]
    assert proof["counts"]["periodic_euler_samples"] == 1
    assert (before, after, plan) == original
    assert not state.equivalent_values(old, new)


@pytest.mark.parametrize("component", ["quaternion_wxyz", "euler_radians"])
def test_periodic_guard_still_rejects_changed_orientation_or_nonperiodic_euler(
    periodic_guard_case, component
):
    before, after, plan = periodic_guard_case
    after["channels"]["data:derived-angle"]["values"][-1][component][0] += 0.01
    proof = protected_report(before, after, plan, {})
    assert not proof["matches"]
    assert proof["counts"]["periodic_euler_samples"] == 0


@pytest.mark.parametrize("kind", ["DataType", "DataRangeType", "SettingRangeType"])
def test_bound_authored_angle_channels_do_not_gain_periodic_guard_tolerance(periodic_guard_case, kind):
    before, after, plan = periodic_guard_case
    reference = {"__native__": "id", "value": "derived-angle"}
    prop = {"kind": kind, "value": [reference] if "Range" in kind else reference}
    behaviour = {"name": "Transform", "properties": {"local_rotation": prop}}
    for checkpoint in (before, after):
        checkpoint["objects"]["point"]["behaviours"]["authored"] = copy.deepcopy(behaviour)
    proof = protected_report(before, after, plan, {})
    assert not proof["matches"]
    assert proof["counts"]["periodic_euler_samples"] == 0


@pytest.mark.parametrize(
    "case", ["operation", "other_name", "foreign", "other_type", "setting", "static", "first_frame", "growth"]
)
def test_periodic_tolerance_is_limited_to_owned_unbake_point_data_at_the_original_terminal_guard(
    periodic_guard_case, case
):
    before, after, plan = periodic_guard_case
    if case == "operation":
        plan["operation"] = "motion_generation"
    elif case == "first_frame":
        channel = after["channels"]["data:derived-angle"]
        channel["values"][0] = channel["values"][-1]
        channel["values"][-1] = copy.deepcopy(before["channels"]["data:derived-angle"]["values"][-1])
    elif case == "growth":
        _extend(after, 6)
        plan["expected_count"] = 6
    else:
        for checkpoint in (before, after):
            channel = checkpoint["channels"]["data:derived-angle"]
            if case == "other_name":
                channel["name"] = "Local Rotation"
            elif case == "foreign":
                channel["owner"] = "prop"
            elif case == "other_type":
                checkpoint["objects"]["point"]["type"] = "Joint"
            elif case == "setting":
                channel["kind"] = "setting"
                channel["setting_type"] = 0
                checkpoint["channels"]["setting:derived-angle"] = checkpoint["channels"].pop(
                    "data:derived-angle"
                )
            else:
                channel["mode"] = "Static"
                channel["value"] = channel.pop("values")[-1]
    proof = protected_report(before, after, plan, {})
    assert not proof["matches"]
    assert proof["counts"]["periodic_euler_samples"] == 0


def test_periodic_guard_requires_all_other_rotation_fields_to_match_exactly(periodic_guard_case):
    before, after, plan = periodic_guard_case
    before["channels"]["data:derived-angle"]["values"][-1]["extra"] = 1.0
    after["channels"]["data:derived-angle"]["values"][-1]["extra"] = 1.0 + 1e-10
    proof = protected_report(before, after, plan, {})
    assert not proof["matches"]
    assert proof["counts"]["periodic_euler_samples"] == 0


@pytest.mark.parametrize("kind", ["object", "track", "track_key", "outside_curve", "auxiliary", "asset"])
def test_existing_native_structure_and_outside_curve_metadata_cannot_change(native_case, kind):
    before, after, plan = native_case
    if kind == "object":
        after["objects"]["point"]["name"] = "renamed"
    elif kind == "track":
        after["tracks"]["body"]["visible"] = False
    elif kind == "track_key":
        after["tracks"]["prop"]["sections"]["2"]["interpolation"] = "STEP"
    elif kind == "outside_curve":
        after["tracks"]["body"]["sections"]["0"]["key"]["label"] = 7
    elif kind == "auxiliary":
        after["auxiliary"]["folders"]["root"]["children"].reverse()
    else:
        after["external"]["assets"]["mesh"]["payload_sha256"] = "modified"
    assert not protected_report(before, after, plan, {})["matches"]


def _extend(snapshot, count):
    snapshot["count"] = count
    for channel in snapshot["channels"].values():
        if channel["mode"] == "Animation":
            channel["values"].extend(copy.deepcopy(channel["values"][-1]) for _ in range(count - 5))


def test_declared_timeline_growth_holds_foreign_and_terminal_samples_at_the_old_last_value(native_case):
    before, after, plan = native_case
    before["channels"]["data:prop"]["values"][-1] = [9.0, 8.0, 7.0]
    after["channels"]["data:prop"]["values"][-1] = [9.0, 8.0, 7.0]
    plan.update(first=0, last=6, expected_count=8)
    _extend(after, 8)
    after["tracks"]["body"]["keys"].append(7)
    after["tracks"]["body"]["sections"]["7"] = _section()
    after["channels"]["data:point"]["values"][6] = [42.0, 1.0, 2.0]
    assert protected_report(before, after, plan, {})["matches"]
    after["channels"]["data:prop"]["values"][6][0] += 0.5
    assert not protected_report(before, after, plan, {})["matches"]


def test_extended_foreign_settings_keep_the_last_value_exactly(native_case):
    before, after, plan = native_case
    before["channels"]["setting:prop"] = _channel("prop", "prop", kind="setting", value=1.0)
    after["channels"]["setting:prop"] = copy.deepcopy(before["channels"]["setting:prop"])
    plan.update(first=0, last=6, expected_count=8)
    _extend(after, 8)
    assert protected_report(before, after, plan, {})["matches"]
    after["channels"]["setting:prop"]["values"][6] += 1e-10
    proof = protected_report(before, after, plan, {})
    assert not proof["matches"]
    assert any(error["path"] == "channels/setting:prop/values/6" for error in proof["errors"])


def test_undeclared_growth_and_owned_changes_on_the_terminal_guard_are_rejected(native_case):
    before, after, plan = native_case
    _extend(after, 8)
    assert not protected_report(before, after, plan, {})["matches"]
    plan.update(first=0, last=6, expected_count=8)
    after["channels"]["data:point"]["values"][7] = [42.0, 1.0, 2.0]
    assert not protected_report(before, after, plan, {})["matches"]


@pytest.mark.parametrize("missing", ["channel", "object", "track"])
def test_removing_native_identities_is_rejected(native_case, missing):
    before, after, plan = native_case
    del after[{"channel": "channels", "object": "objects", "track": "tracks"}[missing]][
        {"channel": "data:finger", "object": "prop", "track": "prop"}[missing]
    ]
    assert not protected_report(before, after, plan, {})["matches"]


def test_invalid_or_nonfinite_checkpoint_cannot_receive_a_success_report(native_case):
    before, after, plan = native_case
    after["channels"]["data:point"]["values"][2] = float("nan")
    assert not protected_report(before, after, plan, {})["matches"]


def test_report_does_not_modify_either_checkpoint_or_job_plan(native_case):
    before, after, plan = native_case
    original = copy.deepcopy((before, after, plan))
    assert protected_report(before, after, plan, {})["matches"]
    assert (before, after, plan) == original


def test_large_scope_violation_keeps_an_exact_count_and_bounded_diagnostics(native_case):
    before, _after, plan = native_case
    _extend(before, 512)
    after = copy.deepcopy(before)
    plan.update(first=0, last=510, expected_count=512)
    after["channels"]["data:prop"]["values"] = [[999.0, 0.0, 0.0]] * 512
    proof = protected_report(before, after, plan, {})
    assert proof["error_count"] == 512
    assert len(proof["errors"]) == 100


def _guard_anchor_case(native_case, *, nonkey_section=False):
    before, after, plan = native_case
    plan.update(operation="unbake", tracks=["body", "hand"])
    for checkpoint in (before, after):
        checkpoint["tracks"]["hand"]["keys"].remove(4)
        if nonkey_section:
            checkpoint["tracks"]["hand"]["sections"]["4"]["key"] = None
        else:
            del checkpoint["tracks"]["hand"]["sections"]["4"]
    section = _section()
    section["interpolation"] = "STEP"
    section["ai_description"] = ""
    plan["guard_anchors"] = {"hand": 4}
    plan["guard_anchor_sections"] = {"hand": copy.deepcopy(section)}
    after["tracks"]["hand"]["keys"].append(4)
    after["tracks"]["hand"]["sections"]["4"] = section
    return before, after, plan


@pytest.mark.parametrize("nonkey_section", [False, True])
def test_unbake_may_add_only_the_predeclared_missing_terminal_key(native_case, nonkey_section):
    before, after, plan = _guard_anchor_case(native_case, nonkey_section=nonkey_section)
    proof = protected_report(before, after, plan, {})
    assert proof["matches"]
    assert proof["counts"]["guard_anchors"] == 1


@pytest.mark.parametrize("channel,value", [("data:finger", 99.0), ("setting:ik", True)])
def test_guard_anchoring_never_allows_terminal_channel_drift(native_case, channel, value):
    before, after, plan = _guard_anchor_case(native_case)
    after["channels"][channel]["values"][4] = value
    proof = protected_report(before, after, plan, {})
    assert not proof["matches"]
    assert any(error["path"] == f"channels/{channel}/values/4" for error in proof["errors"])


@pytest.mark.parametrize("field", ["interpolation", "label", "easing"])
def test_unbake_anchor_metadata_cannot_drift_from_its_prepared_descriptor(native_case, field):
    before, after, plan = _guard_anchor_case(native_case)
    actual = after["tracks"]["hand"]["sections"]["4"]
    if field == "interpolation":
        actual["interpolation"] = "BEZIER"
    elif field == "label":
        actual["key"]["label"] = 7
    else:
        actual["key"]["left_weight_velocity"] += 1e-10
    assert not protected_report(before, after, plan, {})["matches"]


@pytest.mark.parametrize("field", ["label", "ik_fk", "easing"])
def test_guard_plan_cannot_authorize_a_different_key_recipe_even_when_native_output_matches(
    native_case, field
):
    before, after, plan = _guard_anchor_case(native_case)
    section = after["tracks"]["hand"]["sections"]["4"]
    if field == "label":
        section["key"]["label"] = 99
    elif field == "ik_fk":
        section["key"]["common"]["ik_fk"] = "FK"
    else:
        section["key"]["left_weight_velocity"] = 0.5
    plan["guard_anchor_sections"]["hand"] = copy.deepcopy(section)
    assert not protected_report(before, after, plan, {})["matches"]


@pytest.mark.parametrize(
    "violation",
    ["operation", "unselected", "interior", "existing_key", "missing_descriptor", "missing_key", "growth"],
)
def test_guard_anchor_exception_cannot_expand_native_job_scope(native_case, violation):
    before, after, plan = _guard_anchor_case(native_case)
    if violation == "operation":
        plan["operation"] = "motion_generation"
    elif violation == "unselected":
        plan["tracks"] = ["body"]
    elif violation == "interior":
        plan["guard_anchors"]["hand"] = 3
    elif violation == "existing_key":
        before["tracks"]["hand"]["keys"].append(4)
        before["tracks"]["hand"]["sections"]["4"] = _section()
    elif violation == "missing_descriptor":
        plan["guard_anchor_sections"] = {}
    elif violation == "missing_key":
        after["tracks"]["hand"]["keys"].remove(4)
    else:
        _extend(after, 6)
        plan["expected_count"] = 6
    assert not protected_report(before, after, plan, {})["matches"]


def _reference_plane(after):
    def ref(identifier):
        return {"__native__": "id", "value": identifier}

    identifier = "video-plane"
    names = (
        "Basic",
        "Node3d",
        "Transform",
        "CustomTransformData",
        "MeshObject",
        "TextureContainer",
        "Material",
        "FbxExport",
    )
    obj = {
        "name": "MeshObject(0)",
        "type": "Mesh Object",
        "behaviours": {name: {"name": name, "hidden": False, "properties": {}} for name in names},
    }
    texture = obj["behaviours"]["TextureContainer"]["properties"]
    texture["texture_paths"] = {"kind": "DataRangeType", "value": [ref("texture-0")]}
    texture["start_frame"] = {"kind": "DataType", "value": ref("video-start")}
    mesh = obj["behaviours"]["MeshObject"]["properties"]
    mesh["mesh"] = {"kind": "AssetType", "value": ref("video-mesh")}
    mesh["mesh_dependency"] = {"kind": "AssetType", "value": ref("video-dependency")}
    mesh["linked_objects"] = {"kind": "ObjectRangeType", "value": [ref(identifier)]}
    mesh["textures"] = {"kind": "BehaviourType", "value": ref("TextureContainer")}
    obj["behaviours"]["Basic"]["properties"]["parent"] = {
        "kind": "ObjectType",
        "value": ref("00000000-0000-0000-0000-000000000000"),
    }
    after["objects"][identifier] = obj
    after["channels"]["data:texture-0"] = _channel(
        identifier, "texture-0", mode="Static", value="reference/frame0.jpg"
    )
    after["channels"]["data:video-start"] = _channel(identifier, "video-start", mode="Static", value=0)
    after["tracks"]["video-track"] = _track([identifier])
    after["auxiliary"]["folders"]["root"]["children"].append("video-track")
    after["external"]["assets"].update(
        {
            "video-mesh": {"type": "Mesh", "payload_sha256": "a" * 64},
            "video-dependency": {"type": "MeshDependency", "payload_sha256": "b" * 64},
        }
    )
    return identifier


def test_mocap_may_add_only_its_explicit_native_video_plane_channels_mesh_and_folder_entry(native_case):
    before, after, plan = native_case
    plan["operation"] = "mocap"
    reference = _reference_plane(after)
    proof = protected_report(before, after, plan, {"reference_object": reference})
    assert proof["matches"] is True
    assert proof["counts"]["reference_objects"] == 1
    assert proof["counts"]["reference_channels"] == 2
    assert proof["counts"]["reference_assets"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_operation",
        "wrong_id",
        "second_object",
        "rig_behaviour",
        "wrong_type",
        "foreign_channel",
        "foreign_behaviour",
    ],
)
def test_reference_exception_does_not_authorize_an_arbitrary_object_or_foreign_bindings(
    native_case, mutation
):
    before, after, plan = native_case
    plan["operation"] = "mocap"
    reference = _reference_plane(after)
    result = {"reference_object": reference}
    if mutation == "wrong_operation":
        plan["operation"] = "motion_generation"
    elif mutation == "wrong_id":
        result["reference_object"] = "prop"
    elif mutation == "second_object":
        after["objects"]["extra"] = copy.deepcopy(after["objects"][reference])
    elif mutation == "rig_behaviour":
        after["objects"][reference]["behaviours"]["RigInfo"] = {
            "name": "RigInfo",
            "hidden": False,
            "properties": {},
        }
    elif mutation == "wrong_type":
        after["objects"][reference]["type"] = "Joint"
    elif mutation == "foreign_channel":
        after["objects"][reference]["behaviours"]["TextureContainer"]["properties"]["start_frame"]["value"][
            "value"
        ] = "point"
    else:
        after["objects"][reference]["behaviours"]["MeshObject"]["properties"]["textures"]["value"][
            "value"
        ] = "foreign-behaviour"
    assert not protected_report(before, after, plan, result)["matches"]


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign_asset",
        "existing_asset",
        "foreign_folder",
        "new_channel",
        "reordered_siblings",
        "missing_parent",
    ],
)
def test_reference_exception_does_not_hide_other_scene_changes(native_case, mutation):
    before, after, plan = native_case
    plan["operation"] = "mocap"
    reference = _reference_plane(after)
    if mutation == "foreign_asset":
        after["external"]["assets"]["other"] = {"type": "Mesh", "payload_sha256": "c" * 64}
    elif mutation == "existing_asset":
        after["external"]["assets"]["mesh"]["payload_sha256"] = "d" * 64
    elif mutation == "foreign_folder":
        after["auxiliary"]["folders"]["other"] = {"children": [], "name": "unrelated"}
    elif mutation == "new_channel":
        after["channels"]["data:injected"] = _channel("prop", "injected")
    elif mutation == "reordered_siblings":
        after["auxiliary"]["folders"]["root"]["children"].reverse()
    else:
        after["tracks"]["video-track"]["parent"] = "other"
    assert not protected_report(before, after, plan, {"reference_object": reference})["matches"]


def test_existing_reference_object_has_no_new_object_or_asset_exceptions(native_case):
    before, after, plan = native_case
    _reference_plane(before)
    after = copy.deepcopy(before)
    plan["operation"] = "mocap"
    assert protected_report(before, after, plan, {"reference_object": "video-plane"})["matches"]
    after["channels"]["data:video-start"]["value"] = 1
    assert not protected_report(before, after, plan, {"reference_object": "video-plane"})["matches"]
