"""Bounded independent FK Joint subtrees with scoped whole-clip pose recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

from ..contracts import Tool
from . import animation, common, state
from .data_channels import read_data_value
from .motion import checked_pose, finite_vector, normalized_quaternion

MAX_JOINTS = 32
MAX_FRAMES = 121
MAX_SNAPSHOTS = 32
_snapshots: dict[str, dict] = {}


@dataclass
class JointSubtree:
    root: Any
    objects: list
    parents: dict


def _parent(scene, object_id):
    viewer = scene.behaviour_viewer()
    basic = viewer.get_behaviour_by_name(object_id, "Basic")
    if basic.is_null():
        raise ValueError("A subtree object has no native Basic parent relationship.")
    reference = viewer.get_behaviour_object(basic, "parent")
    return None if reference.is_null() else reference


def resolve_subtree(scene, root):
    """Discover descendants from actual native parent IDs, with bounded cycle checks."""
    model, viewer = scene.model_viewer(), scene.behaviour_viewer()
    objects = list(model.get_objects())
    if len(objects) > 10000 or len(objects) != len(set(objects)):
        raise ValueError("Native hierarchy scan exceeds 10000 unique objects.")
    identifier = common.resolve_object(scene, root)
    parents = {item: _parent(scene, item) for item in objects}
    children = {}
    for item, parent in parents.items():
        children.setdefault(parent, []).append(item)
    ordered, visited, stack = [], set(), [identifier]
    names = [model.get_object_name(item) for item in objects]
    while stack:
        item = stack.pop()
        if item in visited:
            raise ValueError("Native Joint subtree contains a hierarchy cycle.")
        visited.add(item)
        if len(visited) > MAX_JOINTS:
            raise ValueError("Joint subtree exceeds 32 objects.")
        if (
            model.get_object_type_name(item) != "Joint"
            or viewer.get_behaviour_by_name(item, "Joint").is_null()
        ):
            raise ValueError("Every descendant of the selected root must be a native Joint.")
        if names.count(model.get_object_name(item)) != 1:
            raise ValueError("Subtree Joint names must be unique in the native scene.")
        ordered.append(item)
        stack.extend(sorted(children.get(item, []), key=str, reverse=True))
    return JointSubtree(identifier, ordered, {item: parents[item] for item in ordered})


def _track(scene, object_id):
    identifier = scene.layers_viewer().layer_id_by_obj_id_or_null(object_id)
    if identifier.is_null():
        raise ValueError("Every Joint needs a native animation track.")
    return identifier, scene.layers_viewer().layer(identifier)


def _pose(scene, subtree, frame):
    rows = {}
    model = scene.model_viewer()
    for item in subtree.objects:
        identifier, track = _track(scene, item)
        parent = subtree.parents[item]
        rows[str(item)] = {
            "name": model.get_object_name(item),
            "parent_id": str(parent) if parent is not None else None,
            "track_id": str(identifier),
            "is_key": bool(track.is_key(frame)),
            "local": animation._read(scene, item, frame, "local"),
            "global": animation._read(scene, item, frame, "global"),
        }
    return rows


def subtree_pose_read(root, frame=None):
    _, _, view, scene = common.current_context()
    subtree = resolve_subtree(scene, root)
    position = animation._existing_frame(scene, frame)
    return {
        "root": str(subtree.root),
        "scene": view.name(),
        "frame": position,
        "joint_count": len(subtree.objects),
        "joints": _pose(scene, subtree, position),
    }


def _signature(scene, subtree):
    """Validate a standalone, unit-scale, isolated two-key linear FK clip."""
    count = animation._size(scene)
    if not 2 <= count <= MAX_FRAMES:
        raise ValueError("Checked FK subtrees require 2-121 stored frames.")
    if subtree.parents[subtree.root] is not None:
        raise ValueError("Select the topmost Joint root; external parent dependencies are unsupported.")
    viewer, data, model = scene.behaviour_viewer(), scene.data_viewer(), scene.model_viewer()
    owned = set(subtree.objects)
    for information in viewer.get_behaviours("RigInfo"):
        if owned.intersection(viewer.get_behaviour_objects_range(information, "related_joints")):
            raise ValueError(
                "RigInfo-driven body Joints require character Point tools, not standalone FK editing."
            )
    objects, tracks = {}, {}
    for item in subtree.objects:
        behaviors = {viewer.get_behaviour_name(value) for value in viewer.get_behaviours(item)}
        if behaviors - {"Basic", "Transform", "Joint", "Node3d", "FbxExport", "JointExtraAttributes"}:
            raise ValueError(
                "The Joint has additional rig/constraint behaviors outside the independent FK contract."
            )
        identifier, track = _track(scene, item)
        if track.is_locked or set(track.obj_ids) != {item}:
            raise ValueError("FK Joint tracks must be unlocked and contain exactly their own Joint.")
        if list(track.key_frame_indices()) != [0, count - 1] or sorted(track.sections) != [0, count - 1]:
            raise ValueError("FK subtree tracks need exactly two endpoint keys and sections.")
        for frame, section in track.sections.items():
            allowed = {"LINEAR", "STEP"} if frame == count - 1 else {"LINEAR"}
            if (
                section.key is None
                or section.interval.interpolation.name not in allowed
                or section.interval.common.ik_fk.name != "FK"
                or section.key.common.ik_fk.name != "FK"
            ):
                raise ValueError(
                    "FK subtree tracks require LINEAR/FK intervals; the terminal STEP flag is retained."
                )
        if __import__("csc").layers.CyclesViewer(track).any_cycles_exist_in_frames(0, count - 1):
            raise ValueError("Cyclic Joint subtrees require a separate animation workflow.")
        transform = viewer.get_behaviour_by_name(item, "Transform")
        scale = viewer.get_behaviour_data(transform, "local_scale")
        if scale.is_null():
            raise ValueError("A Joint has no inspectable native local scale.")
        for frame in range(count):
            encoded = common.json_value(read_data_value(data, scale, frame))
            if len(encoded) == 3 and all(isinstance(value, list) and len(value) == 1 for value in encoded):
                encoded = [value[0] for value in encoded]
            if any(abs(value - 1) > 1e-5 for value in finite_vector(encoded, 3)):
                raise ValueError("FK subtree restoration requires unit local scale at every stored frame.")
        channels = {}
        for name in ("local_position", "local_rotation", "global_position", "global_rotation"):
            reference = animation._channel(scene, item, name)
            if not data.has_data(reference):
                raise ValueError("Joint transform data is unavailable.")
            if name.startswith("local_") and data.get_data(reference).mode.name != "Animation":
                raise ValueError("FK local transform channels must be authored animated data.")
            channels[name] = str(reference)
        objects[str(item)] = {
            "name": model.get_object_name(item),
            "type": "Joint",
            "parent_id": str(subtree.parents[item]) if subtree.parents[item] is not None else None,
            "behaviors": sorted(behaviors),
            "channels": channels,
            "scale_id": str(scale),
        }
        tracks[str(identifier)] = {
            "objects": sorted(map(str, track.obj_ids)),
            "name": track.header.name,
            "parent": str(track.header.parent),
            "locked": bool(track.is_locked),
            "keys": list(track.key_frame_indices()),
            "sections": {
                str(frame): state.section_state(section) for frame, section in track.sections.items()
            },
        }
    return {"root": str(subtree.root), "count": count, "objects": objects, "tracks": tracks}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _capture(view, scene, subtree):
    signature = _signature(scene, subtree)
    frames = []
    for frame in range(signature["count"]):
        pose = _pose(scene, subtree, frame)
        for item in subtree.objects:
            reference = animation._channel(scene, item, "local_rotation")
            encoded = state.encode(scene.data_viewer().get_data_value(reference, frame))
            _validate_rotation_state(encoded, pose[str(item)]["local"]["rotation_wxyz"])
            pose[str(item)]["local_rotation_state"] = encoded
        frames.append(pose)
    return {
        "format": "mcp-for-cascadeur.fk-subtree",
        "version": 2,
        "scene_path": os.path.normcase(view.get_path_name()).replace("\\", "/"),
        "scene_name": view.name(),
        "root": str(subtree.root),
        "signature": signature,
        "frames": frames,
    }


def _validate_rotation_state(value, quaternion):
    import math

    if (
        type(value) is not dict
        or set(value) != {"__native__", "quaternion_wxyz", "euler_radians"}
        or value["__native__"] != "rotation"
    ):
        raise ValueError("FK v2 snapshot requires native local Euler rotation state for every Joint/frame.")
    finite_vector(value["euler_radians"], 3)
    actual = finite_vector(value["quaternion_wxyz"], 4)
    if abs(sum(component * component for component in actual) - 1) > 1e-5:
        raise ValueError("FK native rotation state requires a unit quaternion.")
    actual = normalized_quaternion(actual)
    reference = normalized_quaternion(quaternion)
    dot = abs(sum(first * second for first, second in zip(actual, reference)))
    if 2 * math.acos(min(1.0, dot)) > 1e-4:
        raise ValueError("FK native Euler rotation differs from the corresponding local Joint pose.")


def _store(payload, *, reserve_restore_slot=True):
    limit = MAX_SNAPSHOTS - (1 if reserve_restore_slot else 0)
    if len(_snapshots) >= limit:
        raise ValueError("FK subtree snapshot capacity reached; release a snapshot before another edit.")
    identifier = uuid.uuid4().hex
    _snapshots[identifier] = {**copy.deepcopy(payload), "sha256": _digest(payload)}
    return identifier


def subtree_snapshot(root, path=None):
    _, _, view, scene = common.current_context()
    subtree = resolve_subtree(scene, root)
    payload = _capture(view, scene, subtree)
    destination = common.workspace_path(path, suffixes=(".json",)) if path is not None else None
    identifier = _store(payload)
    if destination is not None:
        try:
            with destination.open("x", encoding="utf-8") as stream:
                json.dump(_snapshots[identifier], stream, ensure_ascii=False, allow_nan=False, indent=2)
        except Exception:
            del _snapshots[identifier]
            raise
    return {
        "snapshot_id": identifier,
        "root": str(subtree.root),
        "joint_count": len(subtree.objects),
        "frame_count": len(payload["frames"]),
        "format_version": 2,
        "local_euler_fidelity": True,
        "sha256": _snapshots[identifier]["sha256"],
        "path": path,
        "scope": "Every stored local/global Joint pose in this FK subtree; unrelated scene edits are preserved.",
    }


def _load(snapshot_id, path):
    if (snapshot_id is None) == (path is None):
        raise ValueError("Provide exactly one subtree snapshot_id or workspace JSON path.")
    if path is not None:
        source = common.workspace_path(path, must_exist=True, suffixes=(".json",))
        if source.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("FK subtree snapshot exceeds eight MiB.")
        record = json.loads(source.read_text(encoding="utf-8"))
    else:
        if snapshot_id not in _snapshots:
            raise ValueError("Subtree snapshot is not present in this host session.")
        record = copy.deepcopy(_snapshots[snapshot_id])
    fields = {"format", "version", "scene_path", "scene_name", "root", "signature", "frames", "sha256"}
    if (
        type(record) is not dict
        or set(record) != fields
        or record["format"] != "mcp-for-cascadeur.fk-subtree"
        or type(record["version"]) is not int
        or record["version"] not in (1, 2)
    ):
        raise ValueError("Unsupported scoped FK snapshot format.")
    payload = {key: value for key, value in record.items() if key != "sha256"}
    if _digest(payload) != record["sha256"]:
        raise ValueError("FK snapshot checksum differs; no native edit started.")
    if (
        type(record["signature"]) is not dict
        or type(record["frames"]) is not list
        or not 2 <= len(record["frames"]) <= MAX_FRAMES
    ):
        raise ValueError("Invalid full-clip FK snapshot dimensions.")
    expected = record["signature"].get("objects")
    if type(expected) is not dict or not 1 <= len(expected) <= MAX_JOINTS:
        raise ValueError("Invalid FK subtree membership.")
    if (
        record["signature"].get("count") != len(record["frames"])
        or record["signature"].get("root") != record["root"]
        or record["root"] not in expected
        or any(
            type(key) is not str or type(value) is not dict or "parent_id" not in value
            for key, value in expected.items()
        )
    ):
        raise ValueError("FK snapshot frame count or root identity differs.")
    for frame in record["frames"]:
        if type(frame) is not dict or frame.keys() != expected.keys():
            raise ValueError("FK snapshot must contain every subtree Joint at every stored frame.")
        for identifier, joint in frame.items():
            if type(joint) is not dict or joint.get("parent_id") != expected[identifier]["parent_id"]:
                raise ValueError("FK snapshot Joint identity differs.")
            for space in ("local", "global"):
                pose = joint.get(space)
                if (
                    type(pose) is not dict
                    or pose.get("object") != identifier
                    or not {"position", "rotation_wxyz"}.issubset(pose)
                ):
                    raise ValueError("FK snapshot contains an incomplete Joint transform.")
                checked_pose({key: pose[key] for key in ("object", "position", "rotation_wxyz")})
                if abs(sum(value * value for value in pose["rotation_wxyz"]) - 1) > 1e-5:
                    raise ValueError("FK snapshot rotations must be native unit quaternions.")
            if record["version"] == 2:
                _validate_rotation_state(joint.get("local_rotation_state"), joint["local"]["rotation_wxyz"])
    return record


def _native_value(csc, template, field, value):
    if field == "position":
        import numpy

        return numpy.asarray(value, dtype=numpy.float32).reshape(template.shape)
    return (
        csc.math.Rotation.from_quaternion(*value)
        if hasattr(template, "to_quaternion")
        else csc.math.Quaternion(*value)
    )


def _write(csc, scene, subtree, entries, space, *, rotation_states=None):
    data = scene.data_viewer()
    plan = []
    for frame, poses in entries:
        values = []
        for identifier, pose in poses.items():
            object_id = next(item for item in subtree.objects if str(item) == identifier)
            for field, suffix in (("position", "position"), ("rotation_wxyz", "rotation")):
                if field in pose:
                    reference = animation._channel(scene, object_id, space + "_" + suffix)
                    template = data.get_data_value(reference, frame)
                    if field == "rotation_wxyz" and rotation_states is not None:
                        encoded = rotation_states[frame, identifier]
                        native = state.decode(encoded)
                        staged = state.encode(native)
                        _validate_rotation_state(staged, pose[field])
                        if any(
                            abs(a - b) > 1e-4
                            for a, b in zip(encoded["euler_radians"], staged["euler_radians"])
                        ):
                            raise ValueError(
                                "Native FK Euler construction changed stored winding; no edit started."
                            )
                    else:
                        native = _native_value(csc, template, field, pose[field])
                    values.append((reference, native))
        plan.append((frame, values))

    def apply(model, updater):
        editor = model.data_editor()
        for frame, values in plan:
            for reference, value in values:
                editor.set_data_value(reference, frame, value)
            updater.run_update({reference for reference, _ in values}, frame)
        if len(plan) == 1:
            interpolator = updater.get_interpolator()
            interpolator.reload()
            interpolator.interpolate()

    common.modify(scene, "Edit checked independent FK subtree", apply)


def _verify_unrelated(before, scene, subtree):
    from . import curves

    after = curves._capture(scene)
    curves.verify_structure(before, after, preserve_tracks=True)
    owned = {str(item) for item in subtree.objects}
    protected = [
        key
        for key, channel in before["channels"].items()
        if channel["owner"] not in owned or channel["kind"] == "setting" or channel["mode"] == "Static"
    ]
    for key in protected:
        if not state.channel_equal(before["channels"][key], after["channels"][key]):
            raise RuntimeError("Scoped FK edit changed protected native channel: " + key)
    curves.accept_verified_checkpoint(scene, after)
    return len(protected)


def _verify_pose(expected, actual):
    worst_position, worst_angle = 0.0, 0.0
    import math

    for identifier, pose in expected.items():
        observed = actual[identifier]
        if "position" in pose:
            worst_position = max(worst_position, math.dist(pose["position"], observed["position"]))
        if "rotation_wxyz" in pose:
            dot = abs(
                sum(first * second for first, second in zip(pose["rotation_wxyz"], observed["rotation_wxyz"]))
            )
            worst_angle = max(worst_angle, 2 * math.acos(min(1.0, dot)))
    if worst_position > 0.01 or worst_angle > 0.001:
        raise RuntimeError("Native FK transform readback differs from the scoped pose snapshot.")
    return {"max_position_error": worst_position, "max_rotation_error_radians": worst_angle}


def subtree_pose_apply(root, poses, frame=None, space="local"):
    from . import curves

    csc, _, view, scene = common.current_context()
    subtree = resolve_subtree(scene, root)
    _signature(scene, subtree)
    position, space = animation._existing_frame(scene, frame), animation._space(space)
    if position == animation._size(scene) - 1:
        raise ValueError(
            "The terminal stored frame has no allocated successor; edit an earlier existing FK key."
        )
    if any(not _track(scene, item)[1].is_key(position) for item in subtree.objects):
        raise ValueError("Every subtree track must already have a key at the edited frame.")
    if type(poses) is not list or not 1 <= len(poses) <= MAX_JOINTS:
        raise ValueError("Provide 1-32 Joint transform patches.")
    patches = {}
    for value in poses:
        pose = checked_pose(value)
        identifier = str(common.resolve_object(scene, pose.pop("object")))
        if identifier not in {str(item) for item in subtree.objects} or identifier in patches:
            raise ValueError("Each patch must identify a distinct Joint inside the selected subtree.")
        patches[identifier] = pose
    snapshot = _store(_capture(view, scene, subtree))
    before = curves.checkpoint_before(scene)
    _write(csc, scene, subtree, [(position, patches)], space)
    observed = _pose(scene, subtree, position)
    verification = _verify_pose(patches, {key: value[space] for key, value in observed.items()})
    protected = _verify_unrelated(before, scene, subtree)
    return {
        "root": str(subtree.root),
        "snapshot_id": snapshot,
        "frame": position,
        "space": space,
        "execution_state": "completed",
        "transaction_count": 1,
        "verification": {"matches": True, **verification},
        "protected_channels_verified": protected,
        "joints": observed,
    }


def subtree_restore(snapshot_id=None, path=None):
    from . import curves

    csc, _, view, scene = common.current_context()
    record = _load(snapshot_id, path)
    if (
        record["scene_path"] != os.path.normcase(view.get_path_name()).replace("\\", "/")
        or record["scene_name"] != view.name()
    ):
        raise ValueError("Scoped FK snapshot belongs to a different native scene.")
    subtree = resolve_subtree(scene, record["root"])
    if _signature(scene, subtree) != record["signature"]:
        raise ValueError(
            "Scoped FK topology, scale, frame count, channel identity or track metadata changed."
        )
    current_scope = _store(_capture(view, scene, subtree), reserve_restore_slot=False)
    before = curves.checkpoint_before(scene)
    entries = [
        (frame, {identifier: joint["local"] for identifier, joint in pose.items()})
        for frame, pose in enumerate(record["frames"])
    ]
    rotation_states = (
        {
            (frame, identifier): joint["local_rotation_state"]
            for frame, pose in enumerate(record["frames"])
            for identifier, joint in pose.items()
        }
        if record["version"] == 2
        else None
    )
    _write(csc, scene, subtree, entries, "local", rotation_states=rotation_states)
    worst_euler = None if record["version"] == 1 else 0.0
    for frame, expected in enumerate(record["frames"]):
        actual = _pose(scene, subtree, frame)
        for space in ("local", "global"):
            _verify_pose(
                {key: value[space] for key, value in expected.items()},
                {key: value[space] for key, value in actual.items()},
            )
        if rotation_states is not None:
            for item in subtree.objects:
                reference = animation._channel(scene, item, "local_rotation")
                observed = state.encode(scene.data_viewer().get_data_value(reference, frame))
                _validate_rotation_state(observed, actual[str(item)]["local"]["rotation_wxyz"])
                error = max(
                    abs(a - b)
                    for a, b in zip(
                        rotation_states[frame, str(item)]["euler_radians"], observed["euler_radians"]
                    )
                )
                worst_euler = max(worst_euler, error)
                if error > 1e-4:
                    raise RuntimeError("Restored native FK Euler winding differs from the scoped snapshot.")
    protected = _verify_unrelated(before, scene, subtree)
    return {
        "root": str(subtree.root),
        "restored": True,
        "execution_state": "completed",
        "snapshot_id": snapshot_id,
        "recovery_snapshot_id": current_scope,
        "frames_verified": len(record["frames"]),
        "spaces_verified": ["local", "global"],
        "protected_channels_verified": protected,
        "unrelated_scene_edits_preserved": True,
        "format_version": record["version"],
        "local_euler_fidelity": record["version"] == 2,
        "max_local_euler_error_radians": worst_euler,
        "compatibility_note": (
            "Native local Euler winding restored at every stored frame."
            if record["version"] == 2
            else "Legacy v1 stores quaternions only; original local Euler winding cannot be recovered from that snapshot."
        ),
    }


def subtree_release_snapshot(snapshot_id):
    if snapshot_id not in _snapshots:
        raise ValueError("Unknown FK subtree snapshot ID.")
    del _snapshots[snapshot_id]
    return {"released": snapshot_id, "remaining": len(_snapshots)}


_IDENTIFIER = {"type": "string", "minLength": 1, "maxLength": 512}
TOOLS = [
    Tool(
        "subtree_pose_read",
        "Discover all real Joint descendants of an exact root and read parent/local/global/key state without changing the playhead.",
        subtree_pose_read,
        {"root": _IDENTIFIER, "frame": animation.FRAME},
        ("root",),
    ),
    Tool(
        "subtree_snapshot",
        "Capture every stored local/global pose of a guarded independent FK subtree; optional integrity-checked workspace JSON. Other scene edits are outside its restore scope.",
        subtree_snapshot,
        {"root": _IDENTIFIER, "path": {"type": "string", "minLength": 1}},
        ("root",),
        False,
    ),
    Tool(
        "subtree_pose_apply",
        "Edit up to 32 independent linear-FK Joints at existing endpoint keys, with scoped full-clip snapshot and protected unrelated-state verification.",
        subtree_pose_apply,
        {"root": _IDENTIFIER, "poses": animation.POSES, "frame": animation.FRAME, "space": animation.SPACE},
        ("root", "poses"),
        False,
    ),
    Tool(
        "subtree_restore",
        "Restore every stored local/global pose of a checked FK subtree; reject changed topology/keys and preserve current unrelated scene edits.",
        subtree_restore,
        {"snapshot_id": _IDENTIFIER, "path": {"type": "string", "minLength": 1}},
        (),
        False,
    ),
    Tool(
        "subtree_release_snapshot",
        "Release one host-memory FK subtree pose snapshot without changing native scene state.",
        subtree_release_snapshot,
        {"snapshot_id": _IDENTIFIER},
        ("snapshot_id",),
        False,
    ),
]
