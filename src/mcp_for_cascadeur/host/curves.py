"""Read-only curve preflight and postconditions for checked character edits."""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import animation


class UnsupportedCurveState(ValueError):
    execution_state = "not_started"


class CharacterVerificationError(RuntimeError):
    execution_state = "verification_failed"


def _name(value):
    return value.name if hasattr(value, "name") else str(value)


def _layers(scene):
    viewer = scene.layers_viewer()
    return {identifier: viewer.layer(identifier) for identifier in viewer.all_layer_ids()}


def verify_supported(scene):
    """Read every track because a native recovery can affect the whole scene."""
    import csc

    count = animation._size(scene)
    if count < 1:
        raise UnsupportedCurveState("Checked character edits require stored animation frames")
    manager = csc.additive_layers.get_manager(scene)
    stacked = manager.get_stacked_layers_count()
    if type(stacked) is not int or stacked < 0:
        raise UnsupportedCurveState("The native additive layer count could not be verified")
    if stacked:
        raise UnsupportedCurveState("Checked recovery does not support additive layer stacks")
    fixed = []
    for identifier, layer in _layers(scene).items():
        if csc.layers.CyclesViewer(layer).any_cycles_exist_in_frames(0, count - 1):
            raise UnsupportedCurveState("Checked recovery does not support animation cycles")
        for frame, section in layer.sections.items():
            interpolation = _name(section.interval.interpolation)
            if interpolation not in {
                *animation.CLASSIC_INTERPOLATIONS,
                "FIXED",
                "NONE",
                "AI",
            }:
                raise UnsupportedCurveState(
                    f"Unsupported native interpolation on track {identifier}: {interpolation}"
                )
            if interpolation == "FIXED":
                fixed.append(str(identifier))
            if section.key is not None:
                if _name(section.key.tangents) != "Continuous":
                    raise UnsupportedCurveState("Checked recovery does not support custom spatial tangents")
                if any(
                    not math.isfinite(value)
                    for value in (section.key.left_weight_velocity, section.key.right_weight_velocity)
                ):
                    raise UnsupportedCurveState(f"Track {identifier} has nonfinite easing at frame {frame}")
    return {
        "stored_frames": count,
        "fixed_tracks": sorted(set(fixed)),
        "additive_stacks": 0,
        "cycles": False,
        "custom_tangents": False,
    }


def character_tracks(scene, character):
    owned = set(character.owned)
    tracks = {
        identifier: layer
        for identifier, layer in _layers(scene).items()
        if set(layer.obj_ids).intersection(owned)
    }
    if not tracks:
        raise UnsupportedCurveState("The character has no native animation tracks")
    for identifier, layer in tracks.items():
        if layer.is_locked:
            raise UnsupportedCurveState(f"Character track {identifier} is locked")
        if set(layer.obj_ids) - owned:
            raise UnsupportedCurveState(f"Character track {identifier} contains unrelated objects")
        if any(_name(section.interval.interpolation) == "AI" for section in layer.sections.values()):
            raise UnsupportedCurveState("AI character interpolation is outside the verified edit contract")
    return tracks


def point_channels(character):
    result = {}
    for identifier in character.controls:
        binding = character.bindings[identifier]
        result[identifier] = binding.position_data
    if not result or len(result) > 128:
        raise UnsupportedCurveState("The character needs 1-128 inspected native Point controls")
    return result


def finger_channels(character):
    result = {}
    for hand in character.fingers.values():
        for binding in hand.values():
            result[str(binding.rotation_data)] = binding.rotation_data
    return result


def curve_preflight(scene, character, frames, edited_ids):
    result = verify_supported(scene)
    if type(frames) is not list or not 1 <= len(frames) <= 8:
        raise UnsupportedCurveState("Curve-preserving edits accept 1-8 existing frames")
    if frames != sorted(set(frames)):
        raise UnsupportedCurveState("Frames must be distinct and increasing")
    for frame in frames:
        animation._existing_frame(scene, frame)
    if frames[-1] == result["stored_frames"] - 1:
        raise UnsupportedCurveState(
            "The terminal stored frame has no following solver interval; allocate a successor frame before editing it"
        )
    selected = set(edited_ids)
    if not selected or selected - set(character.controls):
        raise UnsupportedCurveState("Curve-preserving targets must be Point controls from this character")
    channels = point_channels(character)
    tracks, found = character_tracks(scene, character), set()
    fixed = set()
    for identifier, layer in tracks.items():
        controls = {str(obj) for obj in layer.obj_ids}.intersection(character.controls)
        selected_here = controls.intersection(selected)
        if selected_here:
            for section in layer.sections.values():
                if _name(section.interval.interpolation) in {"FIXED", "NONE"}:
                    raise UnsupportedCurveState(
                        "Edited Point tracks must have classic, non-baked interpolation"
                    )
                if _name(section.interval.common.ik_fk) != "IK" or (
                    section.key is not None and _name(section.key.common.ik_fk) != "IK"
                ):
                    raise UnsupportedCurveState("Edited Point tracks must use native IK mode")
            if any(not layer.is_key(frame) for frame in frames):
                raise UnsupportedCurveState("Every edited frame must already be a key on every edited track")
        if any(_name(section.interval.interpolation) == "FIXED" for section in layer.sections.values()):
            fixed.add(str(identifier))
        found.update(selected_here)
    if found != selected:
        raise UnsupportedCurveState("A selected Point has no verified exclusive character track")
    data = scene.data_viewer()
    for identifier in selected:
        if _name(data.get_data(channels[identifier]).mode) != "Animation":
            raise UnsupportedCurveState("Point target channels must be animated native data")
    return {
        **result,
        "character_tracks": len(tracks),
        "edited_controls": len(selected),
        "fixed_character_tracks": sorted(fixed),
        "frames": list(frames),
        "advisory_only": True,
        "history_allocated": False,
    }


def _capture(scene):
    from . import recovery, state

    committed = recovery.latest_committed_checkpoint(scene)
    return state.capture(scene) if committed is None else committed


def accept_verified_checkpoint(scene, checkpoint):
    from . import recovery

    if recovery.latest_committed_checkpoint(scene) is checkpoint:
        recovery.accept_verified_checkpoint(scene, checkpoint)


def checkpoint_before(scene):
    from . import recovery

    snapshot = recovery.active_checkpoint(scene)
    return _capture(scene) if snapshot is None else snapshot


def _protected_compare(before, after, keys):
    from . import state

    for key in keys:
        if not state.channel_equal(before["channels"][key], after["channels"][key]):
            raise CharacterVerificationError(f"Protected authored/baked channel changed: {key}")


def verify_structure(before, after, *, preserve_tracks):
    fields = ("format_version", "count", "objects", "auxiliary", "external")
    for field in fields:
        if before.get(field) != after.get(field):
            raise CharacterVerificationError(f"Character edit changed scene {field}")
    if before["channels"].keys() != after["channels"].keys():
        raise CharacterVerificationError("Character edit added or removed native data channels")
    for key, old in before["channels"].items():
        new = after["channels"][key]
        if {field: value for field, value in old.items() if field not in ("value", "values")} != {
            field: value for field, value in new.items() if field not in ("value", "values")
        }:
            raise CharacterVerificationError(f"Character edit changed native channel identity: {key}")
    if before["tracks"].keys() != after["tracks"].keys():
        raise CharacterVerificationError("Character edit changed the track set")
    for key, old in before["tracks"].items():
        new = after["tracks"][key]
        if preserve_tracks:
            if old != new:
                raise CharacterVerificationError(f"Curve metadata changed on track {key}")
        elif {field: value for field, value in old.items() if field not in ("keys", "sections")} != {
            field: value for field, value in new.items() if field not in ("keys", "sections")
        }:
            raise CharacterVerificationError(f"Track identity changed on track {key}")


def _as_vector(value):
    encoded = animation.json_value(value)
    if len(encoded) == 3 and all(isinstance(item, list) and len(item) == 1 for item in encoded):
        encoded = [item[0] for item in encoded]
    return animation.finite_vector(encoded, 3)


def _rotate(quaternion, vector):
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx]


def connection_geometry(scene, character, frames=None):
    viewer, data = scene.behaviour_viewer(), scene.data_viewer()
    indices = list(range(animation._size(scene))) if frames is None else list(frames)
    connections = []
    for behaviour in viewer.get_behaviours("ConnectionPointTwoBody"):
        if viewer.get_behaviour_owner(behaviour) not in character.owned:
            continue
        anchors = []
        for side in ("first", "second"):
            rigid_body = viewer.get_behaviour_reference(behaviour, "rigid_body_" + side)
            if rigid_body.is_null():
                raise UnsupportedCurveState("A connection has no native rigid-body reference")
            body = viewer.get_behaviour_owner(rigid_body)
            transform = viewer.get_behaviour_by_name(body, "Transform")
            position = viewer.get_behaviour_data(transform, "global_position")
            rotation = viewer.get_behaviour_data(transform, "global_rotation")
            offset = viewer.get_behaviour_data(behaviour, "pos_local_" + side)
            if any(reference.is_null() for reference in (position, rotation, offset)):
                raise UnsupportedCurveState("A connection is missing its native anchor data")
            anchors.append((position, rotation, offset))
        connections.append(anchors)
    if not connections:
        raise UnsupportedCurveState("Character connection geometry could not be verified")
    samples = []
    for frame in indices:
        worst = 0.0
        for anchors in connections:
            points = []
            for position, rotation, offset in anchors:
                translation = _as_vector(data.get_data_value(position, frame))
                orientation = animation._rotation(data.get_data_value(rotation, frame))
                native_offset = (
                    data.get_data_value(offset, frame)
                    if _name(data.get_data(offset).mode) == "Animation"
                    else data.get_data_value(offset)
                )
                local = _rotate(orientation, _as_vector(native_offset))
                points.append([origin + shift for origin, shift in zip(translation, local)])
            worst = max(worst, math.dist(*points))
        samples.append({"frame": frame, "max_connection_anchor_error": worst})
    maximum = max((sample["max_connection_anchor_error"] for sample in samples), default=0)
    return {
        "connection_count": len(connections),
        "frames_checked": len(indices),
        "max_connection_anchor_error": maximum,
        "frames": samples,
    }


def verify_geometry(scene, character, *, maximum=0.1):
    result = connection_geometry(scene, character)
    if (
        not math.isfinite(result["max_connection_anchor_error"])
        or result["max_connection_anchor_error"] > maximum
    ):
        raise CharacterVerificationError(
            "Connected rigid-body anchors separated across the stored animation: "
            + str(result["max_connection_anchor_error"])
        )
    return result


@dataclass
class CurveProtection:
    before: dict
    character: object
    frames: list[int]
    edited_ids: set[str]
    protected_channels: set[str]
    untouched_keys: dict[str, list[int]]
    fixed_tracks: list[str]

    @classmethod
    def prepare(cls, scene, character, frames, edited_ids):
        readiness = curve_preflight(scene, character, frames, edited_ids)
        before = checkpoint_before(scene)
        owned = {str(obj) for obj in character.owned}
        fixed_objects = {
            obj for identifier in readiness["fixed_tracks"] for obj in before["tracks"][identifier]["objects"]
        }
        protected = {
            key
            for key, channel in before["channels"].items()
            if channel["mode"] == "Static"
            or channel["kind"] == "setting"
            or channel["owner"] not in owned
            or channel["owner"] in fixed_objects
        }
        protected.update("data:" + key for key in finger_channels(character))
        protected.intersection_update(before["channels"])
        point_keys = {}
        native_layers = scene.layers_viewer()
        for identifier, reference in point_channels(character).items():
            track_id = native_layers.layer_id_by_obj_id_or_null(character.controls[identifier])
            if track_id.is_null():
                raise UnsupportedCurveState("A Point control is missing its native track")
            layer = native_layers.layer(track_id)
            preserved_frames = list(layer.key_frame_indices())
            preserved_frames = [frame for frame in preserved_frames if frame not in frames]
            point_keys["data:" + str(reference)] = preserved_frames
        return cls(
            before, character, list(frames), set(edited_ids), protected, point_keys, readiness["fixed_tracks"]
        )

    def verify(self, scene):
        from . import state

        verify_supported(scene)
        after = _capture(scene)
        verify_structure(self.before, after, preserve_tracks=True)
        _protected_compare(self.before, after, self.protected_channels)
        for key, frames in self.untouched_keys.items():
            old, new = self.before["channels"][key], after["channels"][key]
            for frame in frames:
                if not state.equivalent_values(old["values"][frame], new["values"][frame]):
                    raise CharacterVerificationError(
                        f"Untouched authored Point key changed: {key}, frame {frame}"
                    )
        metrics = verify_geometry(scene, self.character)
        accept_verified_checkpoint(scene, after)
        return {
            "verified": True,
            "track_and_curve_metadata": "exact",
            "settings_and_static_data": "exact",
            "fixed_tracks_protected": len(self.fixed_tracks),
            "protected_channels": len(self.protected_channels),
            "frames_checked": self.before["count"],
            "max_connection_anchor_error": metrics["max_connection_anchor_error"],
            "trajectory_shape_preserved": False,
            "scope": "Existing keys and metadata; selected and solver-coupled trajectories may change.",
        }


TOOLS = []
