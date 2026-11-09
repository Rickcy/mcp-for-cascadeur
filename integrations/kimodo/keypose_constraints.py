"""Capture native Cascy poses and express them as coherent SOMA constraints.

Capture only reads Cascadeur's stored data on its application thread. Conversion
and return planning use ordinary Python maths; NumPy is needed only when writing
the constraint NPZ. The neutral Cascy calibration must come from an unchanged
template, not from the first pose of an animated clip.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def _points():
    path = Path(__file__).with_name("point_retarget.py")
    spec = importlib.util.spec_from_file_location("kimodo_keypose_point_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_calibration(calibration):
    if calibration.get("format") != "cascy-point-calibration-v1":
        raise ValueError("A neutral Cascy Point calibration is required")
    if calibration.get("units") != "centimeters" or calibration.get("up_axis") != "Y":
        raise ValueError("Native calibration must use Y-up centimetres")
    if not calibration.get("fingerprint"):
        raise ValueError("Native calibration has no topology fingerprint")


def capture(view, frames, *, calibration, expected_scene_path=None):
    """Read real Joint poses and Point targets without moving the playhead.

    ``frames`` uses Cascadeur's zero-based stored-frame numbering. Scene file,
    rig topology, named Joint ownership and parent links are checked before any
    pose is exported. Capturing an unsaved edit is allowed; the pose digest, not
    the disk checksum alone, identifies the captured input.
    """
    from mcp_for_cascadeur.host import characters

    helper = _points()
    _check_calibration(calibration)
    path = Path(view.get_path_name()).resolve() if view.get_path_name() else None
    if path is None or not path.is_file() or path.suffix.lower() != ".casc":
        raise ValueError("Save the source Cascadeur scene before capturing key poses")
    if expected_scene_path is not None and path != Path(expected_scene_path).resolve():
        raise ValueError("Active source scene differs from the requested keypose scene")
    saved_timebase = helper._bridge().scene_settings(path)
    scene = view.domain_scene()
    rig = characters.resolve_character(scene, require_semantics=True, require_hands=True)
    if rig.fingerprint != calibration["fingerprint"]:
        raise ValueError("Source topology differs from the neutral Cascy calibration")
    model, data = scene.model_viewer(), scene.data_viewer()
    size = int(data.get_animation_size())
    _frames(frames, size)
    by_name = {}
    for obj in rig.joints:
        name = model.get_object_name(obj)
        if name in by_name:
            raise ValueError("Ambiguous native Joint name: " + name)
        by_name[name] = obj
    required = set(helper.MAPPING) | {"f_middle1_l", "f_middle1_r"}
    if required - by_name.keys():
        raise ValueError("Native source is missing Cascy body or hand Joints")
    for name in helper.MAPPING:
        parent = characters._parent(scene, by_name[name])
        parent_name = model.get_object_name(parent) if parent is not None else None
        if parent_name != calibration["joints"][name]["parent"]:
            raise ValueError("Native parent changed at " + name)
    controls = {model.get_object_name(rig.controls[key]): binding for key, binding in rig.bindings.items()}
    if set(controls) != set(calibration["points"]):
        raise ValueError("Native Point membership differs from the neutral calibration")
    original_frame = int(scene.get_current_frame(False))
    rows = []
    for frame in frames:
        joints = {}
        for name in sorted(required):
            obj = by_name[name]
            position = data.get_data_value(characters._transform_data(scene, obj, "global_position"), frame)
            rotation = data.get_data_value(characters._transform_data(scene, obj, "global_rotation"), frame)
            joints[name] = {
                "id": str(obj),
                "position": helper._vector(position),
                "rotation": helper._rotation(rotation.to_rotation_matrix().tolist()),
            }
        rows.append({
            "frame": frame,
            "joints": joints,
            "points": {
                name: helper._vector(data.get_data_value(binding.position_data, frame))
                for name, binding in controls.items()
            },
        })
    if int(scene.get_current_frame(False)) != original_frame:
        raise RuntimeError("The source playhead changed during read-only capture")
    result = {
        "format": "cascy-keypose-snapshot-v1",
        "scene_path": str(path),
        "scene_sha256": _file_sha(path),
        "target_fingerprint": rig.fingerprint,
        "calibration_sha256": _digest(calibration),
        "units": "centimeters",
        "up_axis": "Y",
        "fps": 30,
        "saved_timebase": saved_timebase,
        "stored_frames": size,
        "captured_playhead": original_frame,
        "poses": rows,
    }
    result["capture_sha256"] = _digest(result)
    return result


def _frames(frames, size):
    if (
        not isinstance(frames, list)
        or not 2 <= len(frames) <= 120
        or any(type(frame) is not int or not 0 <= frame < size for frame in frames)
        or frames != sorted(set(frames))
    ):
        raise ValueError("Choose 2-120 sorted, unique, existing zero-based frames")


def _validate_snapshot(snapshot, calibration):
    _check_calibration(calibration)
    if snapshot.get("format") != "cascy-keypose-snapshot-v1":
        raise ValueError("Unsupported native keypose snapshot")
    if snapshot.get("target_fingerprint") != calibration["fingerprint"]:
        raise ValueError("Snapshot topology differs from the neutral calibration")
    if snapshot.get("calibration_sha256") != _digest(calibration):
        raise ValueError("Neutral calibration changed after pose capture")
    if snapshot.get("units") != "centimeters" or snapshot.get("up_axis") != "Y" or snapshot.get("fps") != 30:
        raise ValueError("Snapshot must use Y-up centimetres at 30 FPS")
    expected = snapshot.get("capture_sha256")
    without_digest = {key: value for key, value in snapshot.items() if key != "capture_sha256"}
    if expected != _digest(without_digest):
        raise ValueError("Keypose capture checksum does not match its contents")
    rows = snapshot["poses"]
    _frames([row["frame"] for row in rows], snapshot["stored_frames"])
    helper = _points()
    for row in rows:
        if set(helper.MAPPING) - row["joints"].keys():
            raise ValueError("Captured pose is missing mapped native Joints")
        if set(row["points"]) != set(calibration["points"]):
            raise ValueError("Captured pose Point membership changed")
        for name, value in row["joints"].items():
            helper._vector(value["position"])
            helper._rotation(value["rotation"])
            expected_id = calibration["joints"].get(name, {}).get("id")
            if expected_id is not None and value.get("id") != expected_id:
                raise ValueError("Captured native Joint identity changed at " + name)
        for point in row["points"].values():
            helper._vector(point)


def _description(calibration, template):
    """Reproduce the forward retarget's anatomical frames and length scale."""
    helper = _points()
    if template.get("format") != "kimodo-retarget-motion-v1":
        raise ValueError("A SOMA retarget template is required")
    if template.get("units") != "centimeters" or template.get("up_axis") != "Y":
        raise ValueError("SOMA template must use Y-up centimetres")
    names = template["joint_names"]
    if len(names) != len(set(names)):
        raise ValueError("SOMA template Joint names are not unique")
    index = {name: i for i, name in enumerate(names)}
    rest = [helper._vector(value) for value in template["rest_positions"]]
    rotations = [helper._rotation(value) for value in template["rest_global_rotations"]]
    parents = template["parents"]
    if len(parents) != len(names) or len(rest) != len(names) or len(rotations) != len(names):
        raise ValueError("SOMA neutral skeleton arrays differ in size")
    root = template["root_index"]
    if type(root) is not int or not 0 <= root < len(names) or names[root] != "Hips":
        raise ValueError("SOMA skeleton root must identify Hips")
    for i, parent in enumerate(parents):
        if type(parent) is not int or (i == root and parent not in {-1, root}) or (
            i != root and not 0 <= parent < i
        ):
            raise ValueError("SOMA parents must form a root-first acyclic hierarchy")
    anchors = {name: helper._vector(calibration["points"][helper._main_point(name)]["position"])
               for name in helper.MAPPING}
    target = calibration["joints"]
    descriptions = {}
    for name, source_name in helper.MAPPING.items():
        if target[name]["parent"] != helper.PARENTS[name]:
            raise ValueError("Unexpected native hierarchy at " + name)
        if name.startswith("hand_"):
            side = name[-1]
            source_child = ("Left" if side == "l" else "Right") + "HandMiddle1"
            target_axis = helper._sub(target["f_middle1_" + side]["position"], anchors[name])
        elif name.startswith("toe_"):
            side = name[-1]
            source_child = ("Left" if side == "l" else "Right") + "ToeEnd"
            target_axis = helper._sub(anchors[name], anchors[helper.PARENTS[name]])
        elif name == "head":
            source_child = "HeadEnd"
            target_axis = helper._sub(anchors[name], anchors["neck"])
        else:
            source_child = helper.MAPPING[helper.CHILDREN[name]]
            target_axis = helper._sub(anchors[helper.CHILDREN[name]], anchors[name])
        if source_name not in index or source_child not in index:
            raise ValueError("SOMA template is missing anatomical Joint " + source_name + "/" + source_child)
        source_axis = helper._sub(rest[index[source_child]], rest[index[source_name]])
        primary = helper._unit(source_axis)
        seed = [0, 0, 1] if abs(primary[2]) < 0.9 else [0, 1, 0]
        descriptions[name] = {
            "source_index": index[source_name],
            "source_basis": helper._basis(source_axis, seed),
            "target_basis": helper._basis(target_axis, seed),
        }
    target_leg = sum(math.dist(anchors["thigh_" + side], anchors["calf_" + side])
                     + math.dist(anchors["calf_" + side], anchors["foot_" + side]) for side in ("l", "r"))
    source_leg = sum(math.dist(rest[index[side + "Leg"]], rest[index[side + "Shin"]])
                     + math.dist(rest[index[side + "Shin"]], rest[index[side + "Foot"]])
                     for side in ("Left", "Right"))
    scale = target_leg / source_leg
    if not 0.25 < scale < 4:
        raise ValueError("Source/target leg-length ratio is implausible")
    basis = helper._rotation(template.get("basis_from_kimodo", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    return helper, index, rest, rotations, parents, root, anchors, descriptions, scale, basis


def convert(snapshot, calibration, soma_template, *, start_frame=None, num_frames=None):
    """Convert captured native body poses to coherent SOMA FK in canonical metres.

    Mapped body joints retain their anatomical orientation and twist. Extra SOMA
    joints inherit their parent's motion with neutral relative rotations. Finger
    articulation is intentionally outside this body Point-retarget contract.
    """
    _validate_snapshot(snapshot, calibration)
    helper, index, rest, rest_rotations, parents, root, anchors, descriptions, scale, basis = _description(
        calibration, soma_template
    )
    source_frames = [row["frame"] for row in snapshot["poses"]]
    start_frame = source_frames[0] if start_frame is None else start_frame
    num_frames = source_frames[-1] - start_frame + 1 if num_frames is None else num_frames
    if type(start_frame) is not int or type(num_frames) is not int or not 2 <= num_frames <= 120:
        raise ValueError("Keypose return currently supports 2-120 frames")
    if source_frames[0] < start_frame or source_frames[-1] >= start_frame + num_frames:
        raise ValueError("Captured keyframes fall outside the requested generation interval")
    if source_frames[0] != start_frame:
        raise ValueError("The first captured pose must anchor frame zero of the generation interval")
    inverse_basis = helper._transpose(basis)
    global_positions, global_rotations = [], []
    for row in snapshot["poses"]:
        source_rotations = {}
        for name, description in descriptions.items():
            current = helper._rotation(row["joints"][name]["rotation"])
            neutral_native = helper._rotation(calibration["joints"][name]["rotation"])
            native_delta = helper._mm(current, helper._transpose(neutral_native))
            animated_basis = helper._mm(native_delta, description["target_basis"])
            source_delta = helper._mm(animated_basis, helper._transpose(description["source_basis"]))
            i = description["source_index"]
            source_rotations[i] = helper._mm(source_delta, rest_rotations[i])
        rotations, positions = [], []
        native_root = helper._vector(row["joints"]["pelvis"]["position"])
        source_root = helper._add(rest[root], helper._scale(helper._sub(native_root, anchors["pelvis"]), 1 / scale))
        for i, parent in enumerate(parents):
            if i == root:
                rotation = source_rotations[i]
                position = source_root
            else:
                rest_relative = helper._mm(helper._transpose(rest_rotations[parent]), rest_rotations[i])
                rotation = source_rotations.get(i, helper._mm(rotations[parent], rest_relative))
                rest_offset = helper._mv(helper._transpose(rest_rotations[parent]), helper._sub(rest[i], rest[parent]))
                position = helper._add(positions[parent], helper._mv(rotations[parent], rest_offset))
            rotations.append(helper._rotation(rotation))
            positions.append(position)
        global_positions.append([helper._scale(helper._mv(inverse_basis, value), 0.01) for value in positions])
        global_rotations.append([helper._mm(helper._mm(inverse_basis, value), basis) for value in rotations])
    # SOMA's neutral skeleton is pelvis-relative: Hips is at Y=0, its feet are
    # about a metre lower. Kimodo generation/correction instead uses floor Y=0.
    # A mere inverse bone retarget therefore produces underground constraints.
    # Calibrate that constant vertical origin, and separately canonicalize the
    # initial planar root/heading required by the model. Keep the exact inverse
    # for returning to the original Cascadeur world; never re-ground each pose.
    foot_indices = [index[name] for name in (
        "LeftFoot", "LeftToeBase", "LeftToeEnd", "RightFoot", "RightToeBase", "RightToeEnd"
    ) if name in index]
    floor_offset = -min(helper._mv(inverse_basis, rest[i])[1] * 0.01 for i in foot_indices)
    first_root = global_positions[0][root]
    root_origin = [first_root[0], 0.0, first_root[2]]
    hip_vector = helper._sub(global_positions[0][index["RightLeg"]], global_positions[0][index["LeftLeg"]])
    if math.hypot(hip_vector[0], hip_vector[2]) < 1e-6:
        raise ValueError("First-pose hip direction cannot define a planar heading")
    heading = math.atan2(hip_vector[2], -hip_vector[0])
    cosine, sine = math.cos(-heading), math.sin(-heading)
    canonical_rotation = [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    global_positions = [
        [helper._add(helper._mv(canonical_rotation, helper._sub(value, root_origin)), [0, floor_offset, 0])
         for value in row] for row in global_positions
    ]
    global_rotations = [[helper._mm(canonical_rotation, value) for value in row] for row in global_rotations]
    canonicalization = {
        "format": "kimodo-canonical-world-v1",
        "root_origin_m": root_origin,
        "first_heading_radians": heading,
        "world_to_canonical_rotation": canonical_rotation,
        "canonical_floor_offset_m": floor_offset,
        "floor_reference": "Lowest neutral SOMA Foot/ToeBase/ToeEnd joint defines canonical floor Y=0",
    }
    return {
        "format": "kimodo-keypose-constraints-v1",
        "capture_sha256": snapshot["capture_sha256"],
        "target_fingerprint": snapshot["target_fingerprint"],
        "template_sha256": _digest(soma_template),
        "source_scene": snapshot["scene_path"],
        "source_frames": source_frames,
        "start_frame": start_frame,
        "num_frames": num_frames,
        "fps": 30,
        "units": "meters",
        "up_axis": "Y",
        "joint_names": soma_template["joint_names"],
        "frame_indices": [frame - start_frame for frame in source_frames],
        "global_positions": global_positions,
        "global_rot_mats": global_rotations,
        "height_scale": scale,
        "basis_from_kimodo": basis,
        "canonicalization": canonicalization,
        "scope": "Full body constraints; unmapped intermediate and finger joints use neutral relative rotations",
    }


def write_npz(payload, output):
    import numpy as np

    output = Path(output).resolve()
    if output.exists() or output.suffix.lower() != ".npz":
        raise ValueError("Choose a new .npz constraint output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents numpy.savez from silently overwriting a file.
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            frame_indices=np.asarray(payload["frame_indices"], dtype=np.int64),
            global_positions=np.asarray(payload["global_positions"], dtype=np.float32),
            global_rot_mats=np.asarray(payload["global_rot_mats"], dtype=np.float32),
        )
    return {"path": str(output), "sha256": _file_sha(output), "keyframes": len(payload["frame_indices"])}


def compare_plan(snapshot, plan, *, start_frame=None):
    """Measure all captured Point targets against a planned return; no snapping."""
    helper = _points()
    if snapshot.get("target_fingerprint") != plan.get("target_fingerprint"):
        raise ValueError("Return plan belongs to a different native topology")
    start = snapshot["poses"][0]["frame"] if start_frame is None else start_frame
    by_frame = {row["frame"]: row for row in plan["frames"]}
    rows = []
    for pose in snapshot["poses"]:
        frame = pose["frame"] - start
        if frame not in by_frame or set(by_frame[frame]["points"]) != set(pose["points"]):
            raise ValueError("Return plan is missing captured frames or Point controls")
        errors = {name: math.dist(helper._vector(position), helper._vector(by_frame[frame]["points"][name]))
                  for name, position in pose["points"].items()}
        worst = max(errors, key=errors.get)
        rows.append({"source_frame": pose["frame"], "frame": frame,
                     "max_point_error_cm": errors[worst], "worst_point": worst,
                     "pelvis_error_cm": errors["pelvis_MainPoint"]})
    return {"frames": rows, "max_point_error_cm": max(row["max_point_error_cm"] for row in rows),
            "max_pelvis_error_cm": max(row["pelvis_error_cm"] for row in rows)}


def build_return_plan(calibration, generated_motion, snapshot, *, start_frame=None):
    """Return generated root placement to native world space without re-grounding."""
    _validate_snapshot(snapshot, calibration)
    helper = _points()
    reference = convert(snapshot, calibration, generated_motion, start_frame=start_frame,
                        num_frames=generated_motion["num_frames"])
    world_motion = restore_world_motion(generated_motion, reference["canonicalization"])
    plan = helper.build_plan(calibration, world_motion, ground_to_floor=False)
    _align_world_root(plan, world_motion, helper)
    plan["canonicalization"] = reference["canonicalization"]
    plan["keypose_capture_sha256"] = snapshot["capture_sha256"]
    plan["keypose_deviation"] = compare_plan(snapshot, plan, start_frame=start_frame)
    return plan


def _align_world_root(plan, generated_motion, helper):
    root = generated_motion["joint_names"].index("Hips")
    offset = helper._scale(helper._sub(generated_motion["positions"][0][root],
                                      generated_motion["rest_positions"][root]), plan["height_scale"])
    for frame in plan["frames"]:
        for name, position in frame["points"].items():
            frame["points"][name] = helper._add(position, offset)
    plan["world_root_offset_cm"] = offset


def restore_world_motion(generated_motion, canonicalization):
    """Undo the fixed model-space affine transform on a generated retarget JSON."""
    helper = _points()
    if canonicalization.get("format") != "kimodo-canonical-world-v1":
        raise ValueError("An explicit model/world canonicalization transform is required")
    basis = helper._rotation(generated_motion.get("basis_from_kimodo", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    inverse_canonical = helper._transpose(helper._rotation(canonicalization["world_to_canonical_rotation"]))
    origin = helper._vector(canonicalization["root_origin_m"])
    floor_offset = canonicalization["canonical_floor_offset_m"]
    if isinstance(floor_offset, bool) or not isinstance(floor_offset, (int, float)) or not math.isfinite(floor_offset):
        raise ValueError("Canonical floor origin must be finite")
    world_rotation = helper._mm(helper._mm(basis, inverse_canonical), helper._transpose(basis))
    world_offset = helper._scale(helper._mv(basis, helper._sub(origin, [0, floor_offset, 0])), 100)
    result = dict(generated_motion)
    result["positions"] = [[helper._add(helper._mv(world_rotation, helper._vector(value)), world_offset)
                            for value in row] for row in generated_motion["positions"]]
    result["rotations"] = [[helper._mm(world_rotation, helper._rotation(value)) for value in row]
                           for row in generated_motion["rotations"]]
    return result


def conversion_roundtrip_report(snapshot, calibration, soma_template, *, position_tolerance_cm=0.1):
    """Verify reverse/forward retarget maths at every keypose without a model run."""
    if (isinstance(position_tolerance_cm, bool) or not isinstance(position_tolerance_cm, (int, float))
            or not math.isfinite(position_tolerance_cm) or position_tolerance_cm < 0):
        raise ValueError("Conversion tolerance must be a finite nonnegative number")
    payload = convert(snapshot, calibration, soma_template)
    helper = _points()
    basis = payload["basis_from_kimodo"]
    motion = dict(soma_template)
    motion["num_frames"] = len(payload["frame_indices"])
    motion["positions"] = [
        [helper._scale(helper._mv(basis, value), 100) for value in row]
        for row in payload["global_positions"]
    ]
    motion["rotations"] = [
        [helper._mm(helper._mm(basis, value), helper._transpose(basis)) for value in row]
        for row in payload["global_rot_mats"]
    ]
    motion = restore_world_motion(motion, payload["canonicalization"])
    plan = helper.build_plan(calibration, motion, ground_to_floor=False)
    _align_world_root(plan, motion, helper)
    for row, frame in zip(plan["frames"], payload["frame_indices"]):
        row["frame"] = frame
    report = compare_plan(snapshot, plan)
    return {**report, "position_tolerance_cm": position_tolerance_cm,
            "accepted": report["max_point_error_cm"] <= position_tolerance_cm,
            "scope": "Reverse/forward retarget geometry only; no model generation or native write"}


def compare_captures(expected, actual, *, start_frame=None, position_tolerance_cm=1.0,
                     rotation_tolerance_degrees=5.0):
    """Check original native poses against independently read-back return poses.

    The report exposes both Point and solved Joint drift. Tolerances are explicit;
    passing is never inferred from the existence of a generated scene or NPZ.
    """
    helper = _points()
    for value in (expected, actual):
        if value.get("format") != "cascy-keypose-snapshot-v1":
            raise ValueError("Native keypose snapshots are required for comparison")
        if value.get("capture_sha256") != _digest({k: v for k, v in value.items() if k != "capture_sha256"}):
            raise ValueError("Keypose capture checksum does not match its contents")
    if expected["target_fingerprint"] != actual["target_fingerprint"]:
        raise ValueError("Native comparison topology differs")
    for tolerance in (position_tolerance_cm, rotation_tolerance_degrees):
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Comparison tolerances must be finite nonnegative numbers")
    start = expected["poses"][0]["frame"] if start_frame is None else start_frame
    actual_frames = {row["frame"]: row for row in actual["poses"]}
    rows = []
    excluded_joints = set()
    for pose in expected["poses"]:
        frame = pose["frame"] - start
        if frame not in actual_frames:
            raise ValueError("Native return capture is missing an expected keyframe")
        observed = actual_frames[frame]
        if set(pose["points"]) != set(observed["points"]):
            raise ValueError("Native return capture changed body membership")
        body_joints = set(helper.MAPPING)
        if body_joints - pose["joints"].keys() or body_joints - observed["joints"].keys():
            raise ValueError("Native return capture is missing a mapped body Joint")
        excluded_joints.update((pose["joints"].keys() | observed["joints"].keys()) - body_joints)
        point_errors = {name: math.dist(helper._vector(value), helper._vector(observed["points"][name]))
                        for name, value in pose["points"].items()}
        joint_errors, rotation_errors = {}, {}
        for name in sorted(body_joints):
            joint = pose["joints"][name]
            returned = observed["joints"][name]
            if joint.get("id") != returned.get("id"):
                raise ValueError("Native return capture changed Joint identity at " + name)
            joint_errors[name] = math.dist(helper._vector(joint["position"]), helper._vector(returned["position"]))
            delta = helper._mm(helper._rotation(joint["rotation"]),
                               helper._transpose(helper._rotation(returned["rotation"])))
            cosine = max(-1.0, min(1.0, (sum(delta[i][i] for i in range(3)) - 1) / 2))
            rotation_errors[name] = math.degrees(math.acos(cosine))
        worst_point, worst_joint = max(point_errors, key=point_errors.get), max(joint_errors, key=joint_errors.get)
        worst_rotation = max(rotation_errors, key=rotation_errors.get)
        rows.append({"source_frame": pose["frame"], "frame": frame,
                     "max_point_error_cm": point_errors[worst_point], "worst_point": worst_point,
                     "max_joint_error_cm": joint_errors[worst_joint], "worst_joint": worst_joint,
                     "max_joint_rotation_error_degrees": rotation_errors[worst_rotation],
                     "worst_rotated_joint": worst_rotation})
    maximum_point = max(row["max_point_error_cm"] for row in rows)
    maximum_joint = max(row["max_joint_error_cm"] for row in rows)
    maximum_rotation = max(row["max_joint_rotation_error_degrees"] for row in rows)
    return {"frames": rows, "max_point_error_cm": maximum_point, "max_joint_error_cm": maximum_joint,
            "max_joint_rotation_error_degrees": maximum_rotation,
            "position_tolerance_cm": position_tolerance_cm,
            "rotation_tolerance_degrees": rotation_tolerance_degrees,
            "excluded_joints": sorted(excluded_joints),
            "scope": "All native body Point targets and mapped body Joints; finger articulation is excluded",
            "accepted": maximum_point <= position_tolerance_cm and maximum_joint <= position_tolerance_cm
                        and maximum_rotation <= rotation_tolerance_degrees}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True, help="New NPZ constraints path")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--num-frames", type=int)
    args = parser.parse_args()
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    payload = convert(read(args.snapshot), read(args.calibration), read(args.template),
                      start_frame=args.start_frame, num_frames=args.num_frames)
    result = write_npz(payload, args.output)
    print(json.dumps({**result, "num_frames": payload["num_frames"], "frame_indices": payload["frame_indices"]}))


if __name__ == "__main__":
    main()
