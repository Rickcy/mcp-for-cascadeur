import copy
import math
import sys
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import curves


def enum(name):
    return SimpleNamespace(name=name)


def section(interpolation="BEZIER", tangents="Continuous", ik_fk="IK"):
    def common():
        return SimpleNamespace(ik_fk=enum(ik_fk), fixation=enum("Free"))

    return SimpleNamespace(
        interval=SimpleNamespace(
            interpolation=enum(interpolation),
            common=common(),
            description_for_ai_interpolation="authored description",
        ),
        key=SimpleNamespace(
            tangents=enum(tangents),
            common=common(),
            label=None,
            left_weight_velocity=0.2,
            right_weight_velocity=0.8,
        ),
    )


class Identifier(str):
    def is_null(self):
        return not self


class Track:
    def __init__(self, objects, interpolation="BEZIER"):
        self.obj_ids = set(objects)
        self.sections = {frame: section(interpolation) for frame in (0, 2, 4)}
        self.is_locked = False

    def key_frame_indices(self):
        return sorted(self.sections)

    def is_key(self, frame):
        return frame in self.sections


def channel(owner, kind="data", mode="Animation"):
    result = {"owner": owner, "id": owner, "name": owner, "kind": kind, "mode": mode}
    if mode == "Animation":
        result["values"] = [[float(frame), 0.0, 0.0] for frame in range(5)] if kind == "data" else [False] * 5
    else:
        result["value"] = 1.0
    return result


def guard_case(monkeypatch):
    tracks = {
        Identifier("a-track"): Track(["a"]),
        Identifier("b-track"): Track(["b"]),
        Identifier("baked-track"): Track(["baked"], "FIXED"),
    }
    layer_viewer = SimpleNamespace(
        all_layer_ids=lambda: list(tracks),
        layer=lambda identifier: tracks[identifier],
        layer_id_by_obj_id_or_null=lambda obj: Identifier(obj + "-track"),
    )
    data = SimpleNamespace(
        get_animation_size=lambda: 5, get_data=lambda reference: SimpleNamespace(mode=enum("Animation"))
    )
    scene = SimpleNamespace(layers_viewer=lambda: layer_viewer, data_viewer=lambda: data)
    rig = SimpleNamespace(
        owned={"a", "b", "baked", "finger"},
        controls={"a": "a", "b": "b"},
        bindings={"a": SimpleNamespace(position_data="a-pos"), "b": SimpleNamespace(position_data="b-pos")},
        fingers={"left": {"index1": SimpleNamespace(rotation_data="finger-rotation")}},
    )
    vendor = SimpleNamespace(
        additive_layers=SimpleNamespace(
            get_manager=lambda scene: SimpleNamespace(get_stacked_layers_count=lambda: 0)
        ),
        layers=SimpleNamespace(
            CyclesViewer=lambda layer: SimpleNamespace(any_cycles_exist_in_frames=lambda a, b: False)
        ),
    )
    monkeypatch.setitem(sys.modules, "csc", vendor)
    snapshot = {
        "format_version": 1,
        "count": 5,
        "objects": {"a": {}},
        "auxiliary": {},
        "external": {},
        "tracks": {
            str(identifier): {"objects": sorted(track.obj_ids), "sections": {"0": {"weight": 0.2}}}
            for identifier, track in tracks.items()
        },
        "channels": {
            "data:a-pos": channel("a"),
            "data:b-pos": channel("b"),
            "data:baked-pos": channel("baked"),
            "data:external": channel("outside"),
            "data:derived": channel("a"),
            "data:static": channel("a", mode="Static"),
            "setting:flag": channel("a", kind="setting"),
            "data:finger-rotation": channel("finger"),
        },
    }
    after = copy.deepcopy(snapshot)
    monkeypatch.setattr(curves, "checkpoint_before", lambda scene: snapshot)
    monkeypatch.setattr(curves, "_capture", lambda scene: after)
    monkeypatch.setattr(curves, "verify_geometry", lambda scene, rig: {"max_connection_anchor_error": 0.001})
    return scene, rig, tracks, snapshot, after, vendor


def test_curve_guard_preserves_metadata_and_unedited_authored_keys(monkeypatch):
    scene, rig, _, _, after, _ = guard_case(monkeypatch)
    guard = curves.CurveProtection.prepare(scene, rig, [2], {"a"})
    after["channels"]["data:a-pos"]["values"][2][0] += 0.25
    after["channels"]["data:b-pos"]["values"][2][0] += 0.1
    after["channels"]["data:derived"]["values"][1][0] += 0.1
    report = guard.verify(scene)
    assert report["verified"] is True
    assert report["fixed_tracks_protected"] == 1
    assert report["track_and_curve_metadata"] == "exact"
    assert report["trajectory_shape_preserved"] is False


@pytest.mark.parametrize("identifier", ["data:baked-pos", "data:external", "data:finger-rotation"])
def test_unedited_baked_external_and_finger_samples_are_protected(monkeypatch, identifier):
    scene, rig, _, _, after, _ = guard_case(monkeypatch)
    guard = curves.CurveProtection.prepare(scene, rig, [2], {"a"})
    after["channels"][identifier]["values"][3][0] += 0.1
    with pytest.raises(curves.CharacterVerificationError, match="Protected"):
        guard.verify(scene)


@pytest.mark.parametrize("identifier", ["data:a-pos", "data:b-pos"])
def test_unedited_point_key_values_cannot_move(monkeypatch, identifier):
    scene, rig, _, _, after, _ = guard_case(monkeypatch)
    guard = curves.CurveProtection.prepare(scene, rig, [2], {"a"})
    after["channels"][identifier]["values"][4][0] += 0.1
    with pytest.raises(curves.CharacterVerificationError, match="Untouched authored"):
        guard.verify(scene)


def test_static_and_setting_state_is_compared_exactly(monkeypatch):
    scene, rig, _, _, after, _ = guard_case(monkeypatch)
    guard = curves.CurveProtection.prepare(scene, rig, [2], {"a"})
    after["channels"]["data:static"]["value"] += 1e-10
    with pytest.raises(curves.CharacterVerificationError, match="Protected"):
        guard.verify(scene)


def test_curve_easing_metadata_is_exact_not_numeric_tolerant(monkeypatch):
    scene, rig, _, _, after, _ = guard_case(monkeypatch)
    guard = curves.CurveProtection.prepare(scene, rig, [2], {"a"})
    after["tracks"]["a-track"]["sections"]["0"]["weight"] += 1e-10
    with pytest.raises(curves.CharacterVerificationError, match="metadata"):
        guard.verify(scene)


@pytest.mark.parametrize(
    "blocker", ["unknown_interpolation", "tangents", "cycles", "additive", "missing_key", "baked_edit"]
)
def test_curve_readiness_rejects_unverified_state_without_checkpoint_capture(monkeypatch, blocker):
    scene, rig, tracks, _, _, vendor = guard_case(monkeypatch)
    if blocker == "unknown_interpolation":
        tracks["b-track"].sections[0].interval.interpolation = enum("FUTURE_CURVE_TYPE")
    elif blocker == "tangents":
        tracks["b-track"].sections[0].key.tangents = enum("UserDefined")
    elif blocker == "cycles":
        vendor.layers.CyclesViewer = lambda layer: SimpleNamespace(
            any_cycles_exist_in_frames=lambda a, b: True
        )
    elif blocker == "additive":
        vendor.additive_layers.get_manager = lambda scene: SimpleNamespace(get_stacked_layers_count=lambda: 1)
    elif blocker == "missing_key":
        del tracks["a-track"].sections[2]
    elif blocker == "baked_edit":
        tracks["a-track"].sections[0].interval.interpolation = enum("FIXED")
    captured = []
    monkeypatch.setattr(curves, "checkpoint_before", lambda scene: captured.append(True))
    with pytest.raises(curves.UnsupportedCurveState):
        curves.curve_preflight(scene, rig, [2], {"a"})
    assert captured == []


@pytest.mark.parametrize("downgrade", [False, True])
def test_clamped_curve_guard_keeps_the_native_mode_and_all_authored_metadata(monkeypatch, downgrade):
    from mcp_for_cascadeur.host import state

    scene, rig, tracks, before, after, _ = guard_case(monkeypatch)
    for identifier in ("a-track", "b-track"):
        for frame, saved in tracks[identifier].sections.items():
            saved.interval.interpolation = enum("CLAMPED_BEZIER")
            saved.interval.common.fixation = enum("Fulcrum")
            saved.key.label = frame + 10
            saved.key.left_weight_velocity = 0.31
            saved.key.right_weight_velocity = 0.67
        before["tracks"][identifier]["sections"] = {
            str(frame): state.section_state(saved) for frame, saved in tracks[identifier].sections.items()
        }
    after["tracks"] = copy.deepcopy(before["tracks"])
    guard = curves.CurveProtection.prepare(scene, rig, [2], {"a"})
    after["channels"]["data:a-pos"]["values"][2][0] += 0.25
    if downgrade:
        after["tracks"]["a-track"]["sections"]["2"]["interpolation"] = "BEZIER"
        with pytest.raises(curves.CharacterVerificationError, match="metadata"):
            guard.verify(scene)
    else:
        result = guard.verify(scene)
        assert result["track_and_curve_metadata"] == "exact"
        assert result["trajectory_shape_preserved"] is False


def test_quaternion_anchor_rotation_matches_quarter_turn():
    result = curves._rotate([math.sqrt(0.5), 0, 0, math.sqrt(0.5)], [1, 0, 0])
    assert result == pytest.approx([0, 1, 0])


def test_native_anchor_geometry_checks_rotated_offsets_at_every_frame():
    class Quaternion:
        def __init__(self, w, x, y, z):
            self.components = (w, x, y, z)

        def to_quaternion(self):
            return self

        def w(self):
            return self.components[0]

        def x(self):
            return self.components[1]

        def y(self):
            return self.components[2]

        def z(self):
            return self.components[3]

    references = {("connection", "rigid_body_first"): "rb-a", ("connection", "rigid_body_second"): "rb-b"}
    owners = {"connection": "link", "rb-a": "body-a", "rb-b": "body-b"}
    viewer = SimpleNamespace(
        get_behaviours=lambda name: ["connection"],
        get_behaviour_owner=lambda ref: owners[ref],
        get_behaviour_reference=lambda owner, name: Identifier(references[owner, name]),
        get_behaviour_by_name=lambda obj, name: obj,
        get_behaviour_data=lambda owner, name: Identifier(owner + ":" + name),
    )

    def value(reference, frame=None):
        if reference == "connection:pos_local_first":
            return [1.0, 0.0, 0.0]
        if reference == "connection:pos_local_second":
            return [0.0, 0.0, 0.0]
        if reference.endswith("global_rotation"):
            return (
                Quaternion(math.sqrt(0.5), 0, 0, math.sqrt(0.5))
                if reference.startswith("body-a")
                else Quaternion(1, 0, 0, 0)
            )
        if reference.startswith("body-a"):
            return [0.0, 0.0, 0.0]
        return [0.0, 1.05 if frame == 1 else 1.0, 0.0]

    data = SimpleNamespace(
        get_animation_size=lambda: 3,
        get_data_value=value,
        get_data=lambda reference: SimpleNamespace(
            mode=enum("Static" if "pos_local" in reference else "Animation")
        ),
    )
    scene = SimpleNamespace(behaviour_viewer=lambda: viewer, data_viewer=lambda: data)
    rig = SimpleNamespace(owned={"link"})
    result = curves.connection_geometry(scene, rig)
    assert result["frames_checked"] == 3
    assert result["connection_count"] == 1
    assert result["max_connection_anchor_error"] == pytest.approx(0.05)
    with pytest.raises(curves.CharacterVerificationError, match="anchors separated"):
        curves.verify_geometry(scene, rig, maximum=0.04)
