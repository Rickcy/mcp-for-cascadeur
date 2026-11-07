"""Read-only skeletal import preflight and save/reopen observations.

Measurements describe an explicitly identified pose, never an inferred rest
pose. These helpers neither import files nor authorize native mutations.
"""

from __future__ import annotations

import json
import math

from . import characters, state
from .data_channels import read_data_value

MAX_SAMPLE_FRAMES = 64
MAX_SAMPLED_OBJECTS = 65536
_JOINT_CHANNELS = (
    "global_position",
    "global_rotation",
    "local_position",
    "local_rotation",
    "local_scale",
)


def _key(path):
    return json.dumps(path, ensure_ascii=False, separators=(",", ":"))


def _vector(value):
    result = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(result, (list, tuple)) and len(result) == 3:
        result = [item[0] if isinstance(item, (list, tuple)) and len(item) == 1 else item for item in result]
        if all(type(item) in (int, float) and math.isfinite(item) for item in result):
            return [float(item) for item in result]
    raise ValueError("A native Transform vector must contain three finite values")


def _rotation(value):
    encoded = state.encode(value)
    if not isinstance(encoded, dict) or encoded.get("__native__") not in {"rotation", "quaternion"}:
        raise ValueError("A native Transform rotation must be a Rotation or Quaternion")
    quaternion = encoded["quaternion_wxyz"]
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("A native Transform rotation cannot be zero or nonfinite")
    encoded["quaternion_wxyz"] = [item / norm for item in quaternion]
    return encoded


def _matrix(value):
    result = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(result, (list, tuple)) or len(result) != 4:
        raise ValueError("A native coordinate-parent matrix must have four rows")
    if any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in result):
        raise ValueError("A native coordinate-parent matrix must be 4 by 4")
    if not all(type(item) in (int, float) and math.isfinite(item) for row in result for item in row):
        raise ValueError("A native coordinate-parent matrix must be finite")
    return [[float(item) for item in row] for row in result]


def _frame(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("Manifest frames must be existing stored integer frames")
    return value


def _paths(scene, objects):
    model, behaviour = scene.model_viewer(), scene.behaviour_viewer()
    names = {item: model.get_object_name(item) for item in objects}
    parents = {}
    for item in objects:
        basic = behaviour.get_behaviour_by_name(item, "Basic")
        if basic.is_null():
            raise ValueError("An observed native object has no Basic parent relationship")
        parent = behaviour.get_behaviour_object(basic, "parent")
        if not parent.is_null() and parent not in names:
            raise ValueError("An observed native object refers to a missing parent")
        parents[item] = None if parent.is_null() else parent
    paths = {}
    visiting = set()

    def resolve(item):
        if item in paths:
            return paths[item]
        if item in visiting:
            raise ValueError("Native object hierarchy contains a parent cycle")
        visiting.add(item)
        parent = parents[item]
        paths[item] = ([] if parent is None else resolve(parent)) + [names[item]]
        visiting.remove(item)
        return paths[item]

    for item in objects:
        resolve(item)
    keys = {item: _key(path) for item, path in paths.items()}
    if len(set(keys.values())) != len(keys):
        raise ValueError("Native hierarchy has ambiguous duplicate paths; use a unique skeleton")
    return names, parents, paths, keys


def skeleton_manifest(
    scene,
    character_id=None,
    *,
    frames=(),
    measurement_frame=None,
    require_cascy=False,
    capture_character=True,
):
    """Observe native identity, parent paths, rig membership and sampled poses.

    ``measurement_frame`` defaults to the current stored frame. Bone lengths
    measured there are geometric observations, not proof of a neutral pose.
    A raw imported Joint-only scene is allowed unless ``require_cascy`` is set.
    Use ``capture_character=False`` for a raw FBX containing rig prototypes
    rather than a finished Point rig; native Joint ownership is still recorded.
    No native wrapper is retained in the returned JSON.
    """
    if type(capture_character) is not bool or type(require_cascy) is not bool:
        raise ValueError("capture_character and require_cascy must be booleans")
    if not capture_character and (require_cascy or character_id is not None):
        raise ValueError("An explicit target character requires capture_character=True")
    model, behaviour, data = scene.model_viewer(), scene.behaviour_viewer(), scene.data_viewer()
    count = int(data.get_animation_size())
    objects = list(model.get_objects())
    if not 1 <= count <= state.MAX_FRAMES or not 1 <= len(objects) <= state.MAX_OBJECTS:
        raise ValueError("Manifest exceeds the checkpoint frame/object limits")
    if type(frames) not in (list, tuple) or len(frames) > MAX_SAMPLE_FRAMES:
        raise ValueError(f"Provide at most {MAX_SAMPLE_FRAMES} sample frames")
    requested = [_frame(frame, count) for frame in frames]
    if requested != sorted(set(requested)):
        raise ValueError("Sample frames must be distinct and increasing")
    measured = _frame(scene.get_current_frame() if measurement_frame is None else measurement_frame, count)
    names, parents, paths, keys = _paths(scene, objects)
    kinds = {item: model.get_object_type_name(item) for item in objects}
    joints = {item for item in objects if kinds[item] == "Joint"}
    if not 1 <= len(joints) <= 256:
        raise ValueError("Manifest requires 1-256 native Joints")
    joint_paths, joint_parents = {}, {}
    for item in joints:
        ancestor = parents[item]
        while ancestor is not None and ancestor not in joints:
            ancestor = parents[ancestor]
        joint_parents[item] = ancestor

    def joint_path(item):
        if item not in joint_paths:
            ancestor = joint_parents[item]
            joint_paths[item] = ([] if ancestor is None else joint_path(ancestor)) + [names[item]]
        return joint_paths[item]

    joint_keys = {item: _key(joint_path(item)) for item in joints}
    if len(set(joint_keys.values())) != len(joint_keys):
        raise ValueError("Native skeleton has ambiguous duplicate Joint-only paths")
    rig_infos = list(behaviour.get_behaviours("RigInfo"))
    if len(rig_infos) > 16:
        raise ValueError("Manifest supports at most 16 native RigInfo records")
    membership = {item: [] for item in joints}
    for info in rig_infos:
        owner = behaviour.get_behaviour_owner(info)
        if owner not in keys:
            raise ValueError("RigInfo refers to an unknown owner")
        for item in behaviour.get_behaviour_objects_range(info, "related_joints"):
            if item not in joints:
                raise ValueError("RigInfo related_joints includes a missing or non-Joint object")
            membership[item].append(keys[owner])
    rig = None
    if capture_character and (rig_infos or character_id is not None or require_cascy):
        rig = characters.resolve_character(scene, character_id, require_semantics=require_cascy)
    controls = {} if rig is None else rig.controls
    if require_cascy and len(controls) != 43:
        raise ValueError("The verified Cascy topology requires exactly 43 native Point controls")
    sampled_frames = sorted(set(requested) | {measured})
    if len(sampled_frames) * (len(joints) + len(controls)) > MAX_SAMPLED_OBJECTS:
        raise ValueError("Manifest exceeds the sampled-object budget")

    def channel(item, name):
        transform = behaviour.get_behaviour_by_name(item, "Transform")
        if transform.is_null():
            raise ValueError("Observed Joint/Point lacks a Transform behaviour")
        reference = behaviour.get_behaviour_data(transform, name)
        if reference.is_null() or not data.has_data(reference):
            raise ValueError("Observed Joint/Point lacks native " + name)
        mode = data.get_data(reference).mode.name
        if mode not in {"Static", "Animation"}:
            raise ValueError("Observed Transform has an unsupported native data mode")
        return reference, {"id": str(reference), "mode": mode}

    references, joint_records, coordinate_parents = {}, {}, {}
    for item in sorted(joints, key=lambda value: joint_keys[value]):
        channels = {}
        references[item] = {}
        for name in _JOINT_CHANNELS:
            reference, channels[name] = channel(item, name)
            references[item][name] = reference
        ancestor = joint_parents[item]
        parent_key = None if ancestor is None else joint_keys[ancestor]
        joint_records[joint_keys[item]] = {
            "id": str(item),
            "name": names[item],
            "path": joint_paths[item],
            "parent": parent_key,
            "parent_joint": parent_key,
            "scene_path": paths[item],
            "scene_parent": None if parents[item] is None else keys[parents[item]],
            "owned_rigs": sorted(membership[item]),
            "channels": channels,
        }
        if ancestor is None:
            coordinate_parent = parents[item]
            while coordinate_parent is not None:
                transform = behaviour.get_behaviour_by_name(coordinate_parent, "Transform")
                if not transform.is_null():
                    break
                coordinate_parent = parents[coordinate_parent]
            if coordinate_parent is None:
                matrix = [[float(row == column) for column in range(4)] for row in range(4)]
            else:
                reference, _ = channel(coordinate_parent, "global_matrix")
                matrix = _matrix(read_data_value(data, reference, measured))
            coordinate_parents[joint_keys[item]] = {
                "scene_path": None if coordinate_parent is None else paths[coordinate_parent],
                "global_matrix": matrix,
            }
    point_records, point_references = {}, {}
    for item in controls.values():
        reference, descriptor = channel(item, "global_position")
        point_references[item] = reference
        point_records[keys[item]] = {
            "id": str(item),
            "name": names[item],
            "path": paths[item],
            "parent": None if parents[item] is None else keys[parents[item]],
            "position_data": descriptor,
        }
    native_ids = {str(item): item for item in objects}
    links = []
    if rig is not None:
        for link in rig.native_links:
            try:
                links.append(
                    {
                        name: None
                        if value is None
                        else joint_keys.get(native_ids[value], keys[native_ids[value]])
                        for name, value in link.items()
                    }
                )
            except KeyError as error:
                raise ValueError("RigAdditionalInfo refers to an unknown native object") from error
    character = None
    if rig is not None:
        character = {
            "id": rig.id,
            "owner": keys[rig.owner],
            "name": rig.name,
            "joint_paths": sorted(joint_keys[item] for item in rig.joints),
            "owned_objects": {
                keys[item]: {
                    "id": str(item),
                    "kind": kinds[item],
                    "parent": None if parents[item] is None else keys[parents[item]],
                }
                for item in rig.owned
            },
            "point_paths": sorted(point_records),
            "native_links": sorted(links, key=state.canonical_bytes),
            "cascy_semantics_verified": require_cascy,
        }
    observations = {}
    for frame in sampled_frames:
        poses = {}
        for item, channels in references.items():
            pose = {}
            for name, reference in channels.items():
                value = read_data_value(data, reference, frame)
                pose[name] = _rotation(value) if name.endswith("rotation") else _vector(value)
            poses[joint_keys[item]] = pose
        for key, pose in poses.items():
            parent = joint_records[key]["parent_joint"]
            pose["bone_length"] = (
                None
                if parent is None
                else math.dist(pose["global_position"], poses[parent]["global_position"])
            )
        observations[str(frame)] = {
            "joints": poses,
            "points": {
                keys[item]: _vector(read_data_value(data, reference, frame))
                for item, reference in point_references.items()
            },
        }
    result = {
        "format_version": 1,
        "stored_frames": count,
        "joints": joint_records,
        "points": point_records,
        "character": character,
        "measurements": {
            "frame": measured,
            "is_rest_pose": False,
            "note": "Observed pose only; no frame, including frame 0, is assumed neutral",
            "bone_lengths": {
                key: pose["bone_length"] for key, pose in observations[str(measured)]["joints"].items()
            },
            "local_scales": {
                key: pose["local_scale"] for key, pose in observations[str(measured)]["joints"].items()
            },
            "coordinate_parents": coordinate_parents,
        },
        "sample_frames": requested,
        "samples": {str(frame): observations[str(frame)] for frame in requested},
    }
    # This also rejects accidentally retained wrappers/nonfinite values.
    result["sha256"] = state.digest(result)
    return result


def _difference(differences, path, reason, expected=None, actual=None):
    differences.append({"path": path, "reason": reason, "expected": expected, "actual": actual})


def compare_topology(
    expected, actual, *, require_same_ids=False, require_character=True, compare_scene_groups=None
):
    """Compare exact paths/parent structure; UUID equality is separately chosen.

    Raw FBX inspection uses ``require_character=False``. Save/reopen acceptance
    uses True and can require the original native IDs as well.
    """
    differences = []
    compare_scene_groups = require_character if compare_scene_groups is None else compare_scene_groups
    for key in sorted(set(expected["joints"]) | set(actual["joints"])):
        left, right = expected["joints"].get(key), actual["joints"].get(key)
        if left is None or right is None:
            _difference(differences, f"joints/{key}", "Joint path missing or added", left, right)
            continue
        fields = ("name", "path", "parent", "parent_joint")
        if compare_scene_groups:
            fields += ("scene_path", "scene_parent")
        for field in fields:
            if left[field] != right[field]:
                _difference(
                    differences, f"joints/{key}/{field}", "Joint hierarchy differs", left[field], right[field]
                )
        if require_same_ids and left["id"] != right["id"]:
            _difference(differences, f"joints/{key}/id", "Native identity changed", left["id"], right["id"])
        if require_character:
            for channel in _JOINT_CHANNELS:
                first, second = left["channels"][channel], right["channels"][channel]
                if first["mode"] != second["mode"] or (require_same_ids and first["id"] != second["id"]):
                    _difference(
                        differences,
                        f"joints/{key}/channels/{channel}",
                        "Native Joint channel mode or identity differs",
                        first,
                        second,
                    )
    if require_character:
        for key in sorted(set(expected["points"]) | set(actual["points"])):
            first, second = expected["points"].get(key), actual["points"].get(key)
            if first is None or second is None:
                _difference(differences, f"points/{key}", "Point control missing or added", first, second)
                continue
            for field in ("name", "path", "parent"):
                if first[field] != second[field]:
                    _difference(
                        differences,
                        f"points/{key}/{field}",
                        "Point topology differs",
                        first[field],
                        second[field],
                    )
            first_channel, second_channel = first["position_data"], second["position_data"]
            if first_channel["mode"] != second_channel["mode"] or (
                require_same_ids and first_channel["id"] != second_channel["id"]
            ):
                _difference(
                    differences,
                    f"points/{key}/position_data",
                    "Native Point binding mode or identity differs",
                    first_channel,
                    second_channel,
                )
        left, right = expected.get("character"), actual.get("character")
        if left is None or right is None:
            _difference(
                differences, "character", "Both observations must identify an owned native rig", left, right
            )
        else:
            for field in ("owner", "joint_paths", "point_paths", "native_links"):
                if left[field] != right[field]:
                    _difference(
                        differences,
                        f"character/{field}",
                        "Owned rig topology differs",
                        left[field],
                        right[field],
                    )
            for key in sorted(set(left["owned_objects"]) | set(right["owned_objects"])):
                first, second = left["owned_objects"].get(key), right["owned_objects"].get(key)
                if (
                    first is None
                    or second is None
                    or any(first[name] != second[name] for name in ("kind", "parent"))
                ):
                    _difference(
                        differences, f"character/owned_objects/{key}", "Owned object differs", first, second
                    )
                elif require_same_ids and first["id"] != second["id"]:
                    _difference(
                        differences,
                        f"character/owned_objects/{key}/id",
                        "Native identity changed",
                        first["id"],
                        second["id"],
                    )
            for key in sorted(set(expected["joints"]) & set(actual["joints"])):
                if expected["joints"][key]["owned_rigs"] != actual["joints"][key]["owned_rigs"]:
                    _difference(differences, f"joints/{key}/owned_rigs", "Native rig membership changed")
            if require_same_ids and left["id"] != right["id"]:
                _difference(
                    differences, "character/id", "RigInfo owner identity changed", left["id"], right["id"]
                )
    return {"equivalent": not differences, "differences": differences}


def _tolerance(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " must be a finite number from 0 to 1")
    return float(value)


def compare_samples(
    expected,
    actual,
    *,
    position_tolerance=state.POSITION_TOLERANCE,
    rotation_tolerance=state.ROTATION_TOLERANCE,
    scale_tolerance=1e-6,
    include_points=True,
    preserve_euler=True,
    require_same_count=True,
):
    """Compare the same explicit frames, without retiming or coordinate guessing.

    Euler branches are checked for save/reopen fidelity by default. A separate
    raw FBX orientation comparison may explicitly disable that branch check.
    This is sampled-pose evidence, not a full-scene protection comparison.
    """
    position_tolerance = _tolerance(position_tolerance, "position_tolerance")
    rotation_tolerance = _tolerance(rotation_tolerance, "rotation_tolerance")
    scale_tolerance = _tolerance(scale_tolerance, "scale_tolerance")
    differences = []
    maxima = {"position": 0.0, "rotation_radians": 0.0, "euler_radians": 0.0, "scale": 0.0}
    if require_same_count and expected["stored_frames"] != actual["stored_frames"]:
        _difference(
            differences,
            "stored_frames",
            "Stored frame count differs",
            expected["stored_frames"],
            actual["stored_frames"],
        )
    if expected["sample_frames"] != actual["sample_frames"] or not expected["sample_frames"]:
        _difference(
            differences,
            "sample_frames",
            "The same nonempty explicit frame list is required",
            expected["sample_frames"],
            actual["sample_frames"],
        )
    for frame in sorted(set(expected["samples"]) | set(actual["samples"]), key=int):
        left, right = expected["samples"].get(frame), actual["samples"].get(frame)
        if left is None or right is None:
            _difference(differences, f"samples/{frame}", "Sample frame missing")
            continue
        groups = ("joints", "points") if include_points else ("joints",)
        for group in groups:
            for key in sorted(set(left[group]) | set(right[group])):
                first, second = left[group].get(key), right[group].get(key)
                path = f"samples/{frame}/{group}/{key}"
                if first is None or second is None:
                    _difference(differences, path, "Sample object missing")
                    continue
                if group == "points":
                    error = math.dist(first, second)
                    maxima["position"] = max(maxima["position"], error)
                    if error > position_tolerance:
                        _difference(differences, path, "Point position differs", first, second)
                    continue
                for name in _JOINT_CHANNELS:
                    a, b = first[name], second[name]
                    if name.endswith("rotation"):
                        qa, qb = a["quaternion_wxyz"], b["quaternion_wxyz"]
                        dot = sum(x * y for x, y in zip(qa, qb, strict=True))
                        angle = 2 * math.acos(min(1.0, abs(dot)))
                        maxima["rotation_radians"] = max(maxima["rotation_radians"], angle)
                        if angle > rotation_tolerance:
                            _difference(differences, path + "/" + name, "Joint orientation differs", a, b)
                        if preserve_euler:
                            if ("euler_radians" in a) != ("euler_radians" in b):
                                _difference(
                                    differences, path + "/" + name, "Euler fidelity evidence missing", a, b
                                )
                            elif "euler_radians" in a:
                                error = max(
                                    abs(x - y)
                                    for x, y in zip(a["euler_radians"], b["euler_radians"], strict=True)
                                )
                                maxima["euler_radians"] = max(maxima["euler_radians"], error)
                                if error > rotation_tolerance:
                                    _difference(
                                        differences, path + "/" + name, "Joint Euler branch differs", a, b
                                    )
                    else:
                        error = math.dist(a, b)
                        metric, limit = (
                            ("scale", scale_tolerance)
                            if name == "local_scale"
                            else ("position", position_tolerance)
                        )
                        maxima[metric] = max(maxima[metric], error)
                        if error > limit:
                            _difference(differences, path + "/" + name, "Joint vector differs", a, b)
    return {
        "equivalent": not differences,
        "differences": differences,
        "max_errors": maxima,
        "frames_checked": len(set(expected["samples"]) & set(actual["samples"])),
        "scope": "Explicit sampled Joint transforms and optional Point positions; not a full checkpoint",
    }


def preflight_exact_import(
    target, source, *, expected_points=43, bone_length_tolerance=0.01, scale_tolerance=1e-6
):
    """Fail closed before direct FBX animation import to the existing rig.

    Different hierarchies require an explicit retargeting route. Observed bone
    lengths/scales are checked, while animated root positions are not mistaken
    for rig geometry. Neither pose observation is declared a bind/rest pose.
    """
    bone_length_tolerance = _tolerance(bone_length_tolerance, "bone_length_tolerance")
    scale_tolerance = _tolerance(scale_tolerance, "scale_tolerance")
    if expected_points is not None and (type(expected_points) is not int or not 1 <= expected_points <= 128):
        raise ValueError("expected_points must be 1-128 or None")
    topology = compare_topology(target, source, require_character=False)
    blockers = list(topology["differences"])
    character = target.get("character")
    if character is None:
        _difference(blockers, "character", "Target must have an identified native RigInfo and Point controls")
    elif set(character["joint_paths"]) != set(target["joints"]):
        _difference(
            blockers,
            "character/joint_paths",
            "Exact import requires all scene Joints to belong to the target character",
        )
    if expected_points is not None and len(target["points"]) != expected_points:
        _difference(
            blockers, "points", "Unexpected Point-control count", expected_points, len(target["points"])
        )
    for key in sorted(set(target["joints"]) & set(source["joints"])):
        first, second = (
            target["measurements"]["bone_lengths"][key],
            source["measurements"]["bone_lengths"][key],
        )
        if (first is None) != (second is None) or (
            first is not None and abs(first - second) > bone_length_tolerance
        ):
            _difference(
                blockers, f"measurements/bone_lengths/{key}", "Observed bone length differs", first, second
            )
        first, second = (
            target["measurements"]["local_scales"][key],
            source["measurements"]["local_scales"][key],
        )
        if math.dist(first, second) > scale_tolerance:
            _difference(
                blockers, f"measurements/local_scales/{key}", "Observed local scale differs", first, second
            )
    target_coordinates = target["measurements"]["coordinate_parents"]
    source_coordinates = source["measurements"]["coordinate_parents"]
    for key in sorted(set(target_coordinates) & set(source_coordinates)):
        first, second = target_coordinates[key]["global_matrix"], source_coordinates[key]["global_matrix"]
        if any(
            abs(first[row][column] - second[row][column])
            > (bone_length_tolerance if column == 3 and row < 3 else scale_tolerance)
            for row in range(4)
            for column in range(4)
        ):
            _difference(
                blockers,
                f"measurements/coordinate_parents/{key}",
                "Observed root-parent coordinate basis, offset or scale differs",
                first,
                second,
            )
    return {
        "ready": not blockers,
        "route": "exact_skeleton_animation_import" if not blockers else "blocked",
        "blockers": blockers,
        "target_character": None if character is None else character["id"],
        "target_joint_count": len(target["joints"]),
        "source_joint_count": len(source["joints"]),
        "point_count": len(target["points"]),
        "rest_pose_verified": False,
        "verification": "Preflight observations only; native import, Point editing and save/reopen still require acceptance",
    }
