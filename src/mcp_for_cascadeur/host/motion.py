"""Deterministic pose interpolation that can be previewed before native editing."""

from __future__ import annotations

import bisect
import math


def finite_vector(value, size, *, name="vector"):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    if any(
        type(component) not in (int, float) or not math.isfinite(component) or abs(component) > 1e9
        for component in value
    ):
        raise ValueError(f"{name} contains an invalid or nonfinite number")
    return [float(component) for component in value]


def normalized_quaternion(value):
    components = finite_vector(value, 4, name="rotation_wxyz")
    length = math.sqrt(sum(component * component for component in components))
    if length < 1e-12:
        raise ValueError("rotation_wxyz must have nonzero length")
    return [component / length for component in components]


def quaternion_slerp(first, last, amount):
    """Interpolate rotations along the shortest arc, including antipodal input."""
    start, end = normalized_quaternion(first), normalized_quaternion(last)
    if type(amount) not in (int, float) or not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError("interpolation amount must be in [0, 1]")
    dot = sum(left * right for left, right in zip(start, end))
    if dot < 0:
        end, dot = [-component for component in end], -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalized_quaternion([left + amount * (right - left) for left, right in zip(start, end)])
    angle = math.acos(dot)
    denominator = math.sin(angle)
    weights = (math.sin((1 - amount) * angle) / denominator, math.sin(amount * angle) / denominator)
    return [weights[0] * left + weights[1] * right for left, right in zip(start, end)]


def checked_frame(value):
    if type(value) is not int or not 0 <= value <= 100000:
        raise ValueError("frame must be an integer in [0, 100000]")
    return value


def checked_pose(value):
    if type(value) is not dict or set(value) - {"object", "position", "rotation_wxyz"}:
        raise ValueError("pose entries accept object, position and rotation_wxyz")
    identifier = value.get("object")
    if type(identifier) is not str or not identifier.strip() or len(identifier) > 512:
        raise ValueError("pose object must be an exact ID or nonempty name")
    if not {"position", "rotation_wxyz"}.intersection(value):
        raise ValueError("pose entries need position and/or rotation_wxyz")
    result = {"object": identifier}
    if "position" in value:
        result["position"] = finite_vector(value["position"], 3, name="position")
    if "rotation_wxyz" in value:
        result["rotation_wxyz"] = normalized_quaternion(value["rotation_wxyz"])
    return result


def checked_sequence(keyframes, *, max_frames=128, max_objects=128):
    if type(keyframes) is not list or not 1 <= len(keyframes) <= max_frames:
        raise ValueError(f"provide 1-{max_frames} keyframes")
    result, previous = [], -1
    for entry in keyframes:
        if type(entry) is not dict or set(entry) != {"frame", "poses"}:
            raise ValueError("each keyframe requires exactly frame and poses")
        frame = checked_frame(entry["frame"])
        if frame <= previous:
            raise ValueError("keyframe numbers must be strictly increasing")
        previous = frame
        poses = entry["poses"]
        if type(poses) is not list or not 1 <= len(poses) <= max_objects:
            raise ValueError(f"each keyframe needs 1-{max_objects} pose entries")
        converted = [checked_pose(pose) for pose in poses]
        identifiers = [pose["object"] for pose in converted]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("an object may appear only once per keyframe")
        result.append({"frame": frame, "poses": converted})
    return result


def _eased(amount, mode):
    if mode == "linear":
        return amount
    if mode == "smoothstep":
        return amount * amount * (3 - 2 * amount)
    if mode == "smootherstep":
        return amount**3 * (amount * (amount * 6 - 15) + 10)
    if mode == "hold":
        return 0.0 if amount < 1 else 1.0
    raise ValueError("easing must be linear, smoothstep, smootherstep or hold")


def sample_motion(keyframes, step=1, easing="smoothstep"):
    sequence = checked_sequence(keyframes)
    if type(step) is not int or not 1 <= step <= 10000:
        raise ValueError("step must be an integer in [1, 10000]")
    _eased(0.5, easing)
    timelines = {}
    for entry in sequence:
        for pose in entry["poses"]:
            timelines.setdefault(pose["object"], []).append((entry["frame"], pose))
    for entries in timelines.values():
        channel_sets = [set(pose) - {"object"} for _, pose in entries]
        if any(channels != channel_sets[0] for channels in channel_sets):
            raise ValueError("each object's channels must be consistent across its keyframes")
    first, last = sequence[0]["frame"], sequence[-1]["frame"]
    frames = sorted(set(range(first, last + 1, step)) | {entry["frame"] for entry in sequence})
    if len(frames) > 2048 or len(frames) * len(timelines) > 32768:
        raise ValueError("motion plan exceeds 2048 frames or 32768 object samples; increase step")
    sampled = []
    for frame in frames:
        poses = []
        for identifier, entries in timelines.items():
            times = [time for time, _ in entries]
            if frame < times[0] or frame > times[-1]:
                continue
            upper = bisect.bisect_right(times, frame)
            left_index = max(0, upper - 1)
            right_index = min(len(entries) - 1, upper)
            start_frame, start = entries[left_index]
            end_frame, end = entries[right_index]
            amount = (
                0
                if start_frame == end_frame
                else _eased((frame - start_frame) / (end_frame - start_frame), easing)
            )
            pose = {"object": identifier}
            if "position" in start:
                pose["position"] = [a + amount * (b - a) for a, b in zip(start["position"], end["position"])]
            if "rotation_wxyz" in start:
                pose["rotation_wxyz"] = quaternion_slerp(start["rotation_wxyz"], end["rotation_wxyz"], amount)
            poses.append(pose)
        if poses:
            sampled.append({"frame": frame, "poses": poses})
    return {
        "keyframes": sampled,
        "easing": easing,
        "step": step,
        "frame_count": len(sampled),
        "object_count": len(timelines),
        "native_scene_changed": False,
        "note": "Position interpolation and shortest-arc quaternion SLERP; this does not solve contact or physics.",
    }


def mirror_motion(keyframes, axis="x", object_map=None):
    """Reflect pose data in a world/local coordinate plane with explicit name swaps."""
    sequence = checked_sequence(keyframes)
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y or z")
    mapping = {} if object_map is None else object_map
    if type(mapping) is not dict or any(
        type(a) is not str or type(b) is not str or not b for a, b in mapping.items()
    ):
        raise ValueError("object_map must map exact source identifiers to target identifiers")
    coordinate = ("x", "y", "z").index(axis)
    for entry in sequence:
        for pose in entry["poses"]:
            pose["object"] = mapping.get(pose["object"], pose["object"])
            if "position" in pose:
                pose["position"][coordinate] *= -1
            if "rotation_wxyz" in pose:
                # Under a reflection R -> S R S, the quaternion vector part is axial.
                for index in range(3):
                    if index != coordinate:
                        pose["rotation_wxyz"][index + 1] *= -1
        identifiers = [pose["object"] for pose in entry["poses"]]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("object_map maps multiple poses to the same target")
    return {"keyframes": sequence, "axis": axis, "native_scene_changed": False}
