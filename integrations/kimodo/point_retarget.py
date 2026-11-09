"""Transfer generated SOMA motion to mapped native humanoid Points with independent maths.

No native Retargeting action, model patch or license-dependent command is used.
Scene reads/writes belong on Cascadeur's application thread. Planning is pure
Python and can run outside the application from the calibration JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import time
import uuid
import zipfile
from pathlib import Path

MAPPING = {"pelvis": "Hips", "stomach": "Spine1", "chest": "Chest", "neck": "Neck1", "head": "Head"}
PARENTS = {"pelvis": None, "stomach": "pelvis", "chest": "stomach", "neck": "chest", "head": "neck"}
CHILDREN = {"pelvis": "stomach", "stomach": "chest", "chest": "neck", "neck": "head"}
for _side, _source in (("l", "Left"), ("r", "Right")):
    for _target, _name in (
        ("clavicle", "Shoulder"),
        ("arm", "Arm"),
        ("forearm", "ForeArm"),
        ("hand", "Hand"),
        ("thigh", "Leg"),
        ("calf", "Shin"),
        ("foot", "Foot"),
        ("toe", "ToeBase"),
    ):
        MAPPING[_target + "_" + _side] = _source + _name
    for _chain in (
        ("chest", "clavicle_" + _side, "arm_" + _side, "forearm_" + _side, "hand_" + _side),
        ("pelvis", "thigh_" + _side, "calf_" + _side, "foot_" + _side, "toe_" + _side),
    ):
        for _parent, _child in zip(_chain, _chain[1:]):
            PARENTS[_child] = _parent
            CHILDREN[_parent] = _child if _parent not in {"chest", "pelvis"} else CHILDREN[_parent]


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _scale(a, amount):
    return [x * amount for x in a]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _unit(a):
    length = math.sqrt(_dot(a, a))
    if not math.isfinite(length) or length < 1e-7:
        raise ValueError("Degenerate anatomical bone axis")
    return _scale(a, 1 / length)


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _transpose(a):
    return [list(row) for row in zip(*a)]


def _mv(a, vector):
    return [_dot(row, vector) for row in a]


def _mm(a, b):
    return [[_dot(row, column) for column in zip(*b)] for row in a]


def _basis(primary, secondary):
    first = _unit(primary)
    second = _sub(secondary, _scale(first, _dot(first, secondary)))
    if _dot(second, second) < 1e-8:
        alternative = min(([1, 0, 0], [0, 1, 0], [0, 0, 1]), key=lambda axis: abs(_dot(first, axis)))
        second = _sub(alternative, _scale(first, _dot(first, alternative)))
    second = _unit(second)
    third = _unit(_cross(first, second))
    second = _cross(third, first)
    return _transpose([first, second, third])


def _rotation(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Expected a 3x3 rotation matrix")
    result = [_vector(row) for row in value]
    product = _mm(result, _transpose(result))
    error = max(abs(product[i][j] - (1 if i == j else 0)) for i in range(3) for j in range(3))
    determinant = _dot(result[0], _cross(result[1], result[2]))
    if error > 0.002 or abs(determinant - 1) > 0.002:
        raise ValueError("Motion contains a non-orthonormal or reflected rotation matrix")
    return result


def _main_point(joint):
    return joint.split("_")[0] + "_MainPoint" + ("_" + joint[-1] if joint.endswith(("_l", "_r")) else "")


def _hinge_point(rest_points, points, center_name, first_name, second_name, additional_name):
    """Keep the native shared hinge anchor on both adjacent rigid-body triangles."""
    center = points[center_name]
    a, b = _sub(points[first_name], center), _sub(points[second_name], center)
    rest_center = rest_points[center_name]["position"]
    rest_u = _sub(rest_points[additional_name]["position"], rest_center)
    rest_a = _sub(rest_points[first_name]["position"], rest_center)
    rest_b = _sub(rest_points[second_name]["position"], rest_center)
    radius2 = _dot(rest_u, rest_u)
    ua, ub = _dot(rest_u, rest_a), _dot(rest_u, rest_b)
    aa, bb, ab = _dot(a, a), _dot(b, b), _dot(a, b)
    determinant = aa * bb - ab * ab
    nominal = _sub(points[additional_name], center)
    if determinant > aa * bb * 1e-8:
        projection = _add(
            _scale(a, (ua * bb - ub * ab) / determinant), _scale(b, (ub * aa - ua * ab) / determinant)
        )
        normal = _unit(_cross(a, b))
    else:
        projection = _scale(a, ua / aa)
        perpendicular = _sub(nominal, _scale(a, _dot(nominal, a) / aa))
        normal = (
            _unit(perpendicular)
            if _dot(perpendicular, perpendicular) > 1e-8
            else [row[1] for row in _basis(a, [0, 0, 1])]
        )
    remaining = radius2 - _dot(projection, projection)
    if remaining < -0.01:
        raise ValueError("Generated bend cannot satisfy native shared hinge geometry: " + additional_name)
    if _dot(normal, nominal) < 0:
        normal = _scale(normal, -1)
    points[additional_name] = _add(center, _add(projection, _scale(normal, math.sqrt(max(remaining, 0)))))


def validate_native_clusters(calibration, frames):
    by_id = {point["id"]: name for name, point in calibration["points"].items()}
    groups = []
    for link in calibration.get("native_links", []):
        group = list(
            dict.fromkeys(
                by_id[link[field]]
                for field in ("main_point", "direction_point", "additional_point")
                if link.get(field) in by_id
            )
        )
        if len(group) >= 2:
            groups.append((link["box_id"], group))
    if not groups:
        groups = [
            (joint, [name for name, point in calibration["points"].items() if point["joint"] == joint])
            for joint in MAPPING
        ]
    maximum = 0.0
    pairs = 0
    for box, group in groups:
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                pairs += 1
                rest = math.dist(
                    calibration["points"][first]["position"], calibration["points"][second]["position"]
                )
                error = max(
                    abs(math.dist(frame["points"][first], frame["points"][second]) - rest) for frame in frames
                )
                maximum = max(maximum, error)
                if error > max(0.02, rest * 0.001):
                    raise ValueError(
                        f"Native rigid cluster {box} changes length by {error:g}: {first}, {second}"
                    )
    return {
        "native_clusters": len(groups),
        "native_pairs": pairs,
        "frames_checked": len(frames),
        "max_error": maximum,
    }


def build_plan(calibration, motion, *, ground_to_floor=True):
    """Preserve source bone directions/twist, target lengths and rigid Point offsets."""
    if (
        calibration.get("format") != "cascy-point-calibration-v1"
        or motion.get("format") != "kimodo-retarget-motion-v1"
    ):
        raise ValueError("Unsupported calibration or source-motion format")
    if any(
        data.get("units") != "centimeters" or data.get("up_axis") != "Y" for data in (calibration, motion)
    ):
        raise ValueError("Source and target must use Y-up centimetres")
    count = motion["num_frames"]
    if type(count) is not int or count < 2 or motion.get("fps") != 30:
        raise ValueError("Point retargeting requires at least two frames at 30 FPS")
    source_index = {name: index for index, name in enumerate(motion["joint_names"])}
    missing = set(MAPPING.values()) - source_index.keys()
    if missing:
        raise ValueError("Source motion is missing mapped joints: " + ", ".join(sorted(missing)))
    target = calibration["joints"]
    anchors = {name: calibration["points"][_main_point(name)]["position"] for name in MAPPING}
    if set(MAPPING) - target.keys():
        raise ValueError("Target calibration is missing mapped native body joints")
    rest = [_vector(row) for row in motion["rest_positions"]]
    rest_rotations = [_rotation(row) for row in motion["rest_global_rotations"]]
    positions = [[_vector(row) for row in frame] for frame in motion["positions"]]
    rotations = [[_rotation(row) for row in frame] for frame in motion["rotations"]]
    if len(positions) != count or len(rotations) != count:
        raise ValueError("Source frame counts differ from the motion manifest")
    names_count = len(source_index)
    if any(len(frame) != names_count for frame in positions + rotations):
        raise ValueError("Source motion has inconsistent joint counts")
    lengths = {}
    for name, parent in PARENTS.items():
        if target[name]["parent"] != parent:
            raise ValueError("Unexpected mapped native hierarchy at " + name)
        if parent is not None:
            lengths[name] = math.dist(anchors[name], anchors[parent])
    target_legs = sum(lengths["calf_" + side] + lengths["foot_" + side] for side in ("l", "r"))
    source_legs = sum(
        math.dist(rest[source_index[side + "Leg"]], rest[source_index[side + "Shin"]])
        + math.dist(rest[source_index[side + "Shin"]], rest[source_index[side + "Foot"]])
        for side in ("Left", "Right")
    )
    height_scale = target_legs / source_legs
    if not 0.25 < height_scale < 4:
        raise ValueError("Source/target leg-length ratio is implausible")
    descriptions = {}
    for name, source_name in MAPPING.items():
        index = source_index[source_name]
        child = CHILDREN.get(name)
        if name.startswith("hand_"):
            side = name[-1]
            source_child = ("Left" if side == "l" else "Right") + "HandMiddle1"
            target_axis = _sub(target["f_middle1_" + side]["position"], anchors[name])
        elif name.startswith("toe_"):
            side = name[-1]
            source_child = ("Left" if side == "l" else "Right") + "ToeEnd"
            target_axis = _sub(anchors[name], anchors[PARENTS[name]])
        elif name == "head":
            source_child = "HeadEnd"
            target_axis = _sub(anchors[name], anchors["neck"])
        else:
            source_child = MAPPING[child]
            target_axis = _sub(anchors[child], anchors[name])
        source_child_index = source_index[source_child]
        source_axis = _sub(rest[source_child_index], rest[index])
        primary = _unit(source_axis)
        seed = [0, 0, 1] if abs(primary[2]) < 0.9 else [0, 1, 0]
        source_basis = _basis(source_axis, seed)
        target_basis = _basis(target_axis, seed)
        descriptions[name] = {
            "index": index,
            "child_index": source_child_index,
            "source_secondary": [row[1] for row in source_basis],
            "target_basis": target_basis,
        }
    frames = []
    source_root = source_index["Hips"]
    for frame in range(count):
        transforms = {}
        for name, description in descriptions.items():
            index = description["index"]
            delta = _mm(rotations[frame][index], _transpose(rest_rotations[index]))
            secondary = _mv(delta, description["source_secondary"])
            primary = _sub(positions[frame][description["child_index"]], positions[frame][index])
            animated_basis = _basis(primary, secondary)
            transforms[name] = _mm(animated_basis, _transpose(description["target_basis"]))
        desired = {}
        for name, parent in PARENTS.items():
            if parent is None:
                desired[name] = _add(
                    anchors[name],
                    _scale(_sub(positions[frame][source_root], positions[0][source_root]), height_scale),
                )
            else:
                desired[name] = _add(
                    desired[parent], _mv(transforms[parent], _sub(anchors[name], anchors[parent]))
                )
        points = {}
        for name, point in calibration["points"].items():
            joint = point["joint"]
            offset = _sub(point["position"], anchors[joint])
            points[name] = _add(desired[joint], _mv(transforms[joint], offset))
        for side in ("l", "r"):
            for center, parent, child in (
                ("forearm", "arm", "hand"),
                ("calf", "thigh", "foot"),
                ("toe", "foot", None),
            ):
                center_name = center + "_MainPoint_" + side
                first_name = parent + "_MainPoint_" + side
                second_name = child + "_MainPoint_" + side if child else "toe_DirectionPoint_" + side
                additional_name = center + "_AdditionalPoint_" + side
                if second_name in points:
                    _hinge_point(
                        calibration["points"], points, center_name, first_name, second_name, additional_name
                    )
        frames.append({"frame": frame, "points": points})
    ground_names = [
        name for name in calibration["points"] if "foot_Self0Point" in name or "toe_MainPoint" in name
    ]
    lowest = min(row["points"][name][1] for row in frames for name in ground_names)
    ground_shift = -lowest if ground_to_floor else 0.0
    for frame in frames:
        for point in frame["points"].values():
            point[1] += ground_shift
    rigid_validation = validate_native_clusters(calibration, frames)
    root_name = "pelvis_MainPoint"
    displacement = _sub(frames[-1]["points"][root_name], frames[0]["points"][root_name])
    return {
        "format": "cascy-point-motion-v1",
        "fps": 30,
        "num_frames": count,
        "units": "centimeters",
        "up_axis": "Y",
        "target_fingerprint": calibration["fingerprint"],
        "character_id": calibration.get("character_id"),
        "target_mapping": calibration.get("mapping"),
        "source_npz_sha256": motion.get("source_npz_sha256"),
        "mapping": MAPPING,
        "height_scale": height_scale,
        "ground_shift": ground_shift,
        "root_displacement": displacement,
        "max_rigid_cluster_error": rigid_validation["max_error"],
        "native_cluster_validation": rigid_validation,
        "point_bindings": {name: point["id"] for name, point in calibration["points"].items()},
        "frames": frames,
    }


def _bridge():
    path = Path(__file__).with_name("cascadeur_bridge.py")
    spec = importlib.util.spec_from_file_location("kimodo_bridge_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read(receipt_path):
    path = Path(receipt_path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "kimodo-point-session-v1" or record.get("directory") != str(path.parent):
        raise ValueError("Not a valid isolated Point-retarget receipt")
    if "point_bindings" not in record:
        plan_path = Path(record["plan_path"])
        if _sha(plan_path) != record["plan_sha256"]:
            raise RuntimeError("The retained Point plan changed after session preparation")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        record["point_bindings"] = plan["point_bindings"]
        record["character_id"] = plan.get("character_id")
    return path, record


def _write(receipt_path, record):
    """Share the atomic receipt writer with the native CLI transport."""
    _bridge()._write(receipt_path, record)


def prepare(workspace_root, plan_path, template_path):
    """Create a fresh isolated Point-rig session; never reuse the failed Pro-paste scene."""
    helper = _bridge()
    template = helper._input_file(template_path, ".casc")
    helper.scene_settings(template)
    plan_path = helper._input_file(plan_path, ".json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format") != "cascy-point-motion-v1" or plan.get("max_rigid_cluster_error", 1) > 0.01:
        raise ValueError("A checked rigid Point plan is required")
    workspace = Path(workspace_root).resolve()
    directory = workspace / ("points-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    record = {
        "schema": "kimodo-point-session-v1",
        "directory": str(directory),
        "phase": "points_prepared",
        "plan_path": str(plan_path),
        "plan_sha256": _sha(plan_path),
        "before_sha256": _sha(template),
        "num_frames": plan["num_frames"],
        "next_frame": 0,
        "events": [],
        "target_fingerprint": plan["target_fingerprint"],
        "active_target": "target",
        "character_id": plan.get("character_id"),
        "point_bindings": plan["point_bindings"],
        "writer": "isolated-point-batch-v1",
    }
    shutil.copyfile(template, directory / "before.casc")
    shutil.copyfile(template, directory / "working.casc")
    if plan.get("character_id") and plan.get("target_mapping"):
        record["auxiliary_bindings"] = _verified_auxiliary_bindings(directory / "before.casc", plan)
    receipt = directory / "session.json"
    helper._write(receipt, record)
    return {
        "receipt_path": str(receipt),
        "working_path": str(directory / "working.casc"),
        "phase": record["phase"],
    }


def _verified_auxiliary_bindings(path, plan):
    """Read saved native rig declarations and prove their actual graph wiring."""
    with zipfile.ZipFile(path) as archive:
        try:
            content = archive.read("scene/scene.json")
        except NotImplementedError:
            executable = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/tar.exe"
            content = subprocess.run([str(executable), "-xOf", str(path), "scene/scene.json"],
                capture_output=True, check=True, timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    document = json.loads(content)
    objects = {obj["id"]: obj for obj in document["objects"]}
    owner = objects[plan["character_id"]]
    information = next(item for item in owner["behaviours"] if item["type"] == "RigInfo")
    joints = set(information["related_joints"])
    proto = json.loads(information["proto_json"])
    parents = {key: next((item["parent"] for item in obj["behaviours"] if item["type"] == "Basic"), None)
               for key, obj in objects.items()}

    def resolve(description):
        candidates = []
        for key in joints:
            if objects[key]["name"] != description["Name"]:
                continue
            path_names, current, seen = [], parents.get(key), set()
            while current in objects and current not in seen:
                seen.add(current)
                if objects[current]["name"] and objects[current]["type_name"] == "Joint":
                    path_names.append(objects[current]["name"])
                current = parents.get(current)
            if list(reversed(path_names)) == description.get("Path", []):
                candidates.append(key)
        if len(candidates) != 1:
            raise ValueError("A native auxiliary Joint path is missing or ambiguous")
        return candidates[0]

    behaviour_owners = {item["id"]: key for key, obj in objects.items() for item in obj["behaviours"]}
    boxes = {}
    for key in information["rig_objects"]:
        for item in objects[key]["behaviours"]:
            if item["type"] == "RigAdditionalInfo":
                joint = behaviour_owners.get(item["joint"])
                if joint:
                    boxes[joint] = key
    update = document["update"]
    object_groups = {item["object"]: item["group"] for item in update["objects"]}
    groups = {item["id"]: item for item in update["groups"]}
    bindings = []
    for unit in proto.get("Units", []):
        if not unit.get("AdditionalJoint"):
            continue
        driver, additional = resolve(unit["MainJoint"]), resolve(unit["AdditionalJoint"])
        if driver not in plan["target_mapping"].values():
            continue
        box = boxes[driver]
        source_id, target_id = object_groups[box], object_groups[additional]
        proven = {}
        for channel in ("Position", "Rotation"):
            outputs = {item["id"] for item in groups[source_id]["outputs"] if item["name"] in {
                channel + " Additional Joint", "Additional Joint " + channel}}
            inputs = {item["id"] for item in groups[target_id]["inputs"] if item["name"] == channel}
            matches = [connection for connection in update["connections"]
                       if connection["source"].get("group_id") == source_id
                       and connection["source"].get("attribute_id") in outputs
                       and connection["destination"].get("group_id") == target_id
                       and connection["destination"].get("attribute_id") in inputs]
            if len(matches) != 1:
                raise ValueError("Native auxiliary Joint has no unique body-controller " + channel + " binding")
            proven[channel.lower()] = matches[0]
        bindings.append({"joint": additional, "driver_joint": driver, "box": box, "graph": proven})
    return bindings


def _pose(scene, obj, frame):
    from mcp_for_cascadeur.host import characters

    data = scene.data_viewer()
    return (_vector(data.get_data_value(characters._transform_data(scene, obj, "global_position"), frame)),
            _rotation(data.get_data_value(characters._transform_data(scene, obj, "global_rotation"), frame)
                      .to_rotation_matrix().tolist()))


def _matrix_angle(first, second):
    delta = _mm(_transpose(first), second)
    sine = math.sqrt(sum((delta[i][j] - delta[j][i]) ** 2 for i, j in ((2, 1), (0, 2), (1, 0)))) / 2
    cosine = (sum(delta[i][i] for i in range(3)) - 1) / 2
    return math.atan2(sine, cosine)


def _auxiliary_reference(scene, rig, bindings, frame):
    objects = {str(obj): obj for obj in rig.owned}
    result = {}
    for binding in bindings:
        origin, basis = _pose(scene, objects[binding["box"]], frame)
        position, rotation = _pose(scene, objects[binding["joint"]], frame)
        result[binding["joint"]] = {"position": _mv(_transpose(basis), _sub(position, origin)),
                                    "rotation": _mm(_transpose(basis), rotation)}
    return result


def _verify_auxiliary(scene, rig, bindings, reference, frames):
    if not bindings:
        return {"joints": 0, "frames_checked": len(frames), "max_position_error_cm": 0.0,
                "max_rotation_error_radians": 0.0, "saved_native_graph_verified": True}
    objects = {str(obj): obj for obj in rig.owned}
    maximum_position, maximum_rotation, count = 0.0, 0.0, 0
    for frame in frames:
        count += 1
        for binding in bindings:
            origin, basis = _pose(scene, objects[binding["box"]], frame)
            actual_position, actual_rotation = _pose(scene, objects[binding["joint"]], frame)
            relative = reference[binding["joint"]]
            expected = _add(origin, _mv(basis, relative["position"]))
            maximum_position = max(maximum_position, math.dist(expected, actual_position))
            maximum_rotation = max(maximum_rotation, _matrix_angle(_mm(basis, relative["rotation"]), actual_rotation))
    if maximum_position > 0.01 or maximum_rotation > 0.0001:
        raise RuntimeError("A native graph-bound auxiliary Joint stopped following its body controller")
    return {"joints": len(bindings), "frames_checked": count,
            "max_position_error_cm": maximum_position, "max_rotation_error_radians": maximum_rotation,
            "saved_native_graph_verified": True}


def _stage(receipt_path, expected, completed, operation):
    helper = _bridge()
    receipt, record = _read(receipt_path)
    if record["phase"] != expected:
        raise RuntimeError(f"Expected {expected}, found {record['phase']}; do not replay uncertain writes")
    record["phase"] = "running:" + completed
    record["events"].append({"stage": completed, "started_ns": time.time_ns()})
    helper._write(receipt, record)
    try:
        result = operation(record)
    except Exception as error:
        record["phase"] = "needs_review:" + completed
        record["error"] = f"{type(error).__name__}: {error}"
        helper._write(receipt, record)
        raise
    record["phase"] = completed
    record["events"][-1]["finished_ns"] = time.time_ns()
    helper._write(receipt, record)
    return {"receipt_path": str(receipt), "phase": completed, **(result or {})}


def _view(record):
    return _bridge()._find_view(record, record.get("active_target", "target"))


def attach(receipt_path):
    """Attach only the new owned working tab, allocating a terminal guard frame."""

    def operation(record):
        from mcp_for_cascadeur.host import characters, curves

        helper = _bridge()
        csc, app, manager = helper._application()
        working = Path(record["directory"]) / "working.casc"
        matches = [
            view
            for view in manager.scenes()
            if view.get_path_name() and Path(view.get_path_name()).resolve() == working
        ]
        if not matches:
            if not app.get_data_source_manager().load_scene(str(working)):
                raise RuntimeError("Cascadeur declined the new isolated character working scene")
        view = _view(record)
        manager.set_current_scene(view)
        scene = view.domain_scene()
        rig = characters.resolve_character(scene, record.get("character_id"))
        if rig.fingerprint != record["target_fingerprint"]:
            raise RuntimeError("The source character template differs from the calibrated native topology")
        curves.verify_supported(scene)
        tracks = list(_isolated_tracks(scene, rig)) if record.get("writer") == "isolated-point-batch-v1" else list(rig.tracks)
        count = record["num_frames"]

        def allocate(model, _update, _updater, _session):
            editor = model.layers_editor()
            editor.set_section(csc.layers.layer.Section(), count, tracks[0])
            model.fit_animation_size_by_layers()
            for track in tracks:
                for frame in range(count + 1):
                    editor.set_fixed_interpolation_or_key_if_need(track, frame, True)
            editor.normalize_sections(scene)

        helper._modify(scene, "Prepare isolated generated Point motion", allocate)
        boundary = view.animation_boundary()
        boundary.first_frame, boundary.first_visible_frame = 0, 0
        boundary.last_frame, boundary.last_visible_frame = count, count
        return {
            "native_points": len(rig.controls),
            "stored_frames": int(scene.data_viewer().get_animation_size()),
        }

    return _stage(receipt_path, "points_prepared", "points_ready:0", operation)


def _isolated_tracks(scene, rig):
    """Permit only verified skin containers beside the selected rig on its tracks.

    Cascadeur's UE templates place a mesh's plain Transform parent on Torso
    without listing that container in RigInfo. It remains a protected outsider;
    this whitelist does not add it to writable rig membership.
    """
    from mcp_for_cascadeur.host import curves

    viewer, model, layers = scene.behaviour_viewer(), scene.model_viewer(), scene.layers_viewer()
    objects = list(model.get_objects())
    by_id = {str(obj): obj for obj in objects}
    parents = {}
    for obj in objects:
        basic = viewer.get_behaviour_by_name(obj, "Basic")
        parent = None if basic.is_null() else viewer.get_behaviour_object(basic, "parent")
        parents[str(obj)] = None if parent is None or parent.is_null() else str(parent)
    selected = {str(obj) for obj in rig.owned}
    allowed, skins = set(), set()
    for obj in rig.owned:
        mesh = viewer.get_behaviour_by_name(obj, "MeshObject")
        if not mesh.is_null():
            skin = set(viewer.get_behaviour_objects_range(mesh, "linked_objects"))
            if skin and skin <= rig.joints:
                skins.add(str(obj))
    for identifier in skins:
        parent, seen = parents.get(identifier), set()
        while parent and parent not in selected and parent not in seen:
            seen.add(parent)
            if model.get_object_type_name(by_id[parent]) != "Transform":
                raise ValueError("A selected skin has a non-Transform external container")
            allowed.add(parent)
            parent = parents.get(parent)
    foreign = set()
    for information in viewer.get_behaviours("RigInfo"):
        owner = viewer.get_behaviour_owner(information)
        if str(owner) == rig.id:
            continue
        foreign.add(str(owner))
        for field in ("related_joints", "rig_objects", "related_objects"):
            foreign.update(str(obj) for obj in viewer.get_behaviour_objects_range(information, field))
    if allowed & foreign:
        raise ValueError("A skin container belongs to another native character")
    for identifier in by_id:
        current, seen = parents.get(identifier), set()
        while current and current not in seen:
            seen.add(current)
            if current in allowed and identifier not in selected | allowed:
                raise ValueError("A selected skin container also owns unrelated scene objects")
            current = parents.get(current)
    result = {}
    for identifier in layers.all_layer_ids():
        layer = layers.layer(identifier)
        membership = {str(obj) for obj in layer.obj_ids}
        if not membership & selected:
            continue
        if layer.is_locked or membership - (selected | allowed):
            raise ValueError("Selected character track is locked or contains unrelated objects")
        if any(curves._name(section.interval.interpolation) == "AI" for section in layer.sections.values()):
            raise ValueError("AI interpolation is outside the isolated Point write contract")
        result[identifier] = layer
    if not result:
        raise ValueError("The selected character has no native animation tracks")
    return result


def _protected_local_channels(scene, rig, mapping, *, auxiliary_bindings=()):
    """Keep authored fingers/attachments, allowing native spine and twist solvers."""
    from mcp_for_cascadeur.host import characters

    by_id = {str(obj): obj for obj in rig.joints}
    derived = set()
    derived.update(binding["joint"] for binding in auxiliary_bindings)
    for identifier in mapping.values():
        current, seen = by_id.get(identifier), set()
        while current is not None and str(current) not in seen:
            seen.add(str(current))
            derived.add(str(current))
            current = characters._parent(scene, current)
    viewer = scene.behaviour_viewer()
    objects = {str(obj): obj for obj in rig.owned}
    for link in rig.native_links:
        box = objects.get(link["box_id"])
        if box is not None and any(not viewer.get_behaviour_by_name(box, name).is_null()
                                   for name in ("Twist", "Untwist")):
            derived.add(link["joint"])
    result = set()
    for identifier, obj in by_id.items():
        if identifier not in derived:
            for channel in ("local_position", "local_rotation", "local_scale"):
                result.add(str(characters._transform_data(scene, obj, channel)))
    return result


def _batch_guard(scene, rig, frames, protected_locals, *, capture_protected=False):
    """Hash metadata/static state and protected samples without a whole-clip copy."""
    from mcp_for_cascadeur.host import state

    data, model = scene.data_viewer(), scene.model_viewer()
    metadata, protected = hashlib.sha256(), hashlib.sha256()
    protected_values = {}
    owned = {str(obj) for obj in rig.owned}
    counters = {"channels": 0, "protected_samples": 0, "sections": 0}

    def feed(hasher, value):
        payload = state.canonical_bytes(state.encode(value))
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)

    feed(metadata, int(data.get_animation_size()))
    for obj in sorted(model.get_objects(), key=str):
        oid = str(obj)
        feed(metadata, [oid, model.get_object_name(obj), model.get_object_type_name(obj)])
        for kind, getter_ids, getter_info, getter_value in (
            ("data", data.get_all_data_id, data.get_data, data.get_data_value),
            ("setting", data.get_all_settings_id, data.get_setting, data.get_setting_value),
        ):
            for identifier in sorted(getter_ids(obj), key=str):
                info = getter_info(identifier)
                mode = info.mode.name
                if mode not in {"Animation", "Static"}:
                    raise ValueError("Unsupported native channel mode in isolated Point session")
                key = str(identifier)
                feed(metadata, [kind, key, info.name, mode])
                counters["channels"] += 1
                if mode == "Static":
                    feed(metadata, getter_value(identifier))
                elif kind == "setting" or oid not in owned or key in protected_locals:
                    feed(protected, [kind, key])
                    values = []
                    for frame in frames:
                        value = getter_value(identifier, frame)
                        feed(protected, value)
                        if capture_protected:
                            values.append(state.encode(value))
                        counters["protected_samples"] += 1
                    if capture_protected:
                        protected_values[kind + ":" + key] = {
                            "owner": oid, "name": info.name, "kind": kind, "values": values}
    layers = scene.layers_viewer()
    for identifier in sorted(layers.all_layer_ids(), key=str):
        layer = layers.layer(identifier)
        feed(metadata, [str(identifier), layer.header.name, str(layer.header.parent),
                        sorted(map(str, layer.obj_ids)), bool(layer.is_locked), bool(layer.is_visible)])
        for frame in layer.key_frame_indices():
            feed(metadata, ["key", int(frame)])
        for frame, section in sorted(layer.sections.items()):
            feed(metadata, [int(frame), state.section_state(section)])
            counters["sections"] += 1
    result = {"metadata_sha256": metadata.hexdigest(), "protected_sha256": protected.hexdigest(), **counters}
    if capture_protected:
        result["protected_values"] = protected_values
    return result


def _component_delta(first, second):
    if type(first) in (int, float) and type(second) in (int, float):
        return abs(first - second)
    if isinstance(first, dict) and isinstance(second, dict):
        return max((_component_delta(value, second[key]) for key, value in first.items() if key in second), default=0.0)
    if isinstance(first, list) and isinstance(second, list):
        return max((_component_delta(a, b) for a, b in zip(first, second)), default=0.0)
    return 0.0


def _quaternion_angle(first, second):
    a, b = first["quaternion_wxyz"], second["quaternion_wxyz"]
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Protected rotations need complete native quaternions")
    aa, bb = math.sqrt(sum(value * value for value in a)), math.sqrt(sum(value * value for value in b))
    if not math.isfinite(aa) or not math.isfinite(bb) or min(aa, bb) < 1e-8:
        raise ValueError("Protected native rotation is degenerate")
    cosine = min(1.0, abs(sum(x * y for x, y in zip(a, b)) / aa / bb))
    return 2 * math.acos(cosine)


def _compare_batch_guards(before, after, frames):
    """Use existing native numeric tolerances only for animated data samples."""
    from mcp_for_cascadeur.host import state

    exact_fields = ("metadata_sha256", "channels", "protected_samples", "sections")
    if any(before[field] != after[field] for field in exact_fields):
        raise RuntimeError("Isolated Point batch changed existing curve/static metadata or channel membership")
    old, new = before["protected_values"], after["protected_values"]
    if old.keys() != new.keys():
        raise RuntimeError("Isolated Point batch changed protected channel membership")
    report = {"changed_numeric_samples": 0, "max_numeric_component_delta": 0.0,
              "max_orientation_error_radians": 0.0, "euler_representation_changes": 0, "mismatches": []}
    for identifier, channel in old.items():
        current = new[identifier]
        if {key: value for key, value in channel.items() if key != "values"} != {
                key: value for key, value in current.items() if key != "values"}:
            raise RuntimeError("Isolated Point batch changed protected channel identity: " + identifier)
        if len(channel["values"]) != len(frames) or len(current["values"]) != len(frames):
            raise RuntimeError("Protected native sample count differs from the edited range")
        for frame, a, b in zip(frames, channel["values"], current["values"]):
            delta = _component_delta(a, b)
            report["max_numeric_component_delta"] = max(report["max_numeric_component_delta"], delta)
            report["changed_numeric_samples"] += a != b
            equivalent = state.equivalent_values(a, b, exact=channel["kind"] == "setting")
            if channel["kind"] != "setting" and isinstance(a, dict) and isinstance(b, dict) and (
                    a.get("__native__") == b.get("__native__") == "rotation"):
                angle = _quaternion_angle(a, b)
                report["max_orientation_error_radians"] = max(report["max_orientation_error_radians"], angle)
                equivalent = angle <= state.ROTATION_TOLERANCE
                if equivalent and _component_delta(a.get("euler_radians"), b.get("euler_radians")) > state.ROTATION_TOLERANCE:
                    report["euler_representation_changes"] += 1
            if not equivalent and len(report["mismatches"]) < 16:
                report["mismatches"].append({"channel": identifier, "owner": channel["owner"],
                    "name": channel["name"], "frame": frame, "max_numeric_component_delta": delta,
                    "before": a, "after": b})
    return report


def _float32_targets(prepared):
    """Validate the whole fixed-size batch before any native mutation."""
    import numpy

    result = []
    with numpy.errstate(over="ignore", invalid="ignore"):
        for frame, targets in prepared:
            converted = {key: numpy.asarray(position, dtype=numpy.float32) for key, position in targets.items()}
            if any(value.shape != (3,) or not numpy.isfinite(value).all() for value in converted.values()):
                raise ValueError("Generated Point coordinates cannot be represented as finite native float32 values")
            result.append((frame, converted))
    return result


def _isolated_batch(scene, rig, entries, plan, record, receipt_path):
    """Write one owned batch; failure quarantines its scene instead of replaying it.

    The source file is the durable recovery boundary. This path deliberately
    avoids the general MCP whole-model checkpoint, whose cost grows with the
    entire clip rather than this fixed twenty-frame operation.
    """
    from mcp_for_cascadeur.host import characters, common, curves, recovery

    if recovery.active_transaction() is not None:
        raise RuntimeError("An isolated Point batch cannot run inside another recovery transaction")
    before_file = Path(record["directory"]) / "before.casc"
    if _sha(before_file) != record["before_sha256"]:
        raise RuntimeError("The immutable source copy changed before the native batch")
    prepared = characters._point_plan(scene, rig, entries, True, coordinate_limit=None)
    converted = _float32_targets(prepared)
    frames = [frame for frame, _ in prepared]
    tracks = _isolated_tracks(scene, rig)
    data = scene.data_viewer()
    for frame in frames:
        characters._editable_frame(scene, frame)
    for layer in tracks.values():
        if any(not layer.is_key(frame) for frame in frames):
            raise ValueError("The isolated writer requires preallocated keys at every edited frame")
        for section in layer.sections.values():
            if section.interval.common.ik_fk.name != "IK" or (
                    section.key is not None and section.key.common.ik_fk.name != "IK"):
                raise ValueError("The isolated writer requires native IK character tracks")
    if any(data.get_data(binding.position_data).mode.name != "Animation" for binding in rig.bindings.values()):
        raise ValueError("Native Point target channels must be animated")
    mapping = plan.get("target_mapping")
    if not mapping:
        from mcp_for_cascadeur.host import humanoid

        mapping = humanoid.discover_mapping(scene, rig.id)["mapping"]
    auxiliaries = record.get("auxiliary_bindings", [])
    if auxiliaries and "auxiliary_reference" not in record:
        record["auxiliary_reference"] = _auxiliary_reference(scene, rig, auxiliaries, frames[0])
    if auxiliaries:
        _verify_auxiliary(scene, rig, auxiliaries, record["auxiliary_reference"], frames)
    protected_locals = _protected_local_channels(scene, rig, mapping, auxiliary_bindings=auxiliaries)
    before = _batch_guard(scene, rig, frames, protected_locals, capture_protected=True)
    before_points = [{"frame": frame, "points": {
        key: _vector(data.get_data_value(rig.bindings[key].position_data, frame)) for key in rig.controls}}
        for frame in frames]
    event = record["events"][-1]
    event["before_point_targets"] = before_points
    event["batch_guard_before"] = before
    _write(receipt_path, record)

    def write(model, updater):
        editor = model.data_editor()
        for frame, targets in converted:
            changed = set()
            for key, position in targets.items():
                channel = rig.bindings[key].position_data
                template = data.get_data_value(channel, frame)
                editor.set_data_value(channel, frame, position.reshape(template.shape))
                changed.add(channel)
            updater.run_update(changed, frame)

    common.modify(scene, "Write isolated generated Point batch", write)
    after = _batch_guard(scene, rig, frames, protected_locals, capture_protected=True)
    event["batch_guard_after"] = {key: value for key, value in after.items() if key != "protected_values"}
    protection = _compare_batch_guards(before, after, frames)
    event["protected_comparison"] = protection
    if protection["mismatches"]:
        first = protection["mismatches"][0]
        raise RuntimeError("Isolated Point batch changed protected animated channel " + first["channel"]
                           + " (" + first["name"] + ") at frame " + str(first["frame"]))
    auxiliary_report = _verify_auxiliary(scene, rig, auxiliaries, record.get("auxiliary_reference", {}), frames)
    current_rig = characters.resolve_character(scene, record.get("character_id"))
    if current_rig.fingerprint != record["target_fingerprint"] or _sha(before_file) != record["before_sha256"]:
        raise RuntimeError("Native ownership or immutable source changed during the batch")
    worst = 0.0
    for frame, targets in prepared:
        for key, position in targets.items():
            actual = _vector(data.get_data_value(rig.bindings[key].position_data, frame))
            worst = max(worst, math.dist(actual, position))
    if worst > 3 or not math.isfinite(worst):
        raise RuntimeError("Native solver target deviation exceeds three centimetres")
    geometry = curves.connection_geometry(scene, rig, frames=frames)
    if not math.isfinite(geometry["max_connection_anchor_error"]) or geometry["max_connection_anchor_error"] > 0.1:
        raise RuntimeError("Native connection anchors separated in the generated batch")
    event.pop("before_point_targets", None)
    before.pop("protected_values", None)
    return {"max_solver_adjustment": worst, "geometry": geometry,
            "verification": {"method": "isolated-point-batch-v1", "frames": frames,
                "curve_and_static_metadata": "exact-stream-digest", "protected_local_channels": len(protected_locals),
                "protected_samples": before["protected_samples"], "metadata_sha256": before["metadata_sha256"],
                "protected_sha256": before["protected_sha256"], "immutable_source_verified": True,
                "protected_comparison": protection,
                "auxiliary_joint_verification": auxiliary_report,
                "protected_numeric_policy": "Animated data uses native tolerances; protected rotation uses SO(3), with Euler representation changes reported; settings/static/metadata exact",
                "failure_policy": "Retain owned working scene and receipt for inspection; never replay uncertain writes"}}


def apply(receipt_path):
    """Write at most twenty generated full-body poses, then verify native geometry."""
    _, current = _read(receipt_path)
    first = current["next_frame"]
    last = min(first + 20, current["num_frames"])
    expected = f"points_ready:{first}"
    completed = f"points_ready:{last}" if last < current["num_frames"] else "points_applied"

    def operation(record):
        from mcp_for_cascadeur.host import characters

        if _sha(record["plan_path"]) != record["plan_sha256"]:
            raise RuntimeError("The generated Point plan changed after the session was prepared")
        plan = json.loads(Path(record["plan_path"]).read_text(encoding="utf-8"))
        helper = _bridge()
        _, _, manager = helper._application()
        view = _view(record)
        manager.set_current_scene(view)
        scene = view.domain_scene()
        rig = characters.resolve_character(scene, record.get("character_id"))
        if rig.fingerprint != record["target_fingerprint"]:
            raise RuntimeError("Target Point topology changed before the next generated chunk")
        names = plan["point_bindings"]
        if set(names.values()) != set(rig.controls):
            raise RuntimeError("Target Point identities differ from the generated plan")
        entries = [
            {
                "frame": row["frame"],
                "pose": {
                    names[name]: {"position": _vector(position)} for name, position in row["points"].items()
                },
            }
            for row in plan["frames"][first:last]
        ]
        if record.get("writer") == "isolated-point-batch-v1":
            result = _isolated_batch(scene, rig, entries, plan, record, receipt_path)
            record["events"][-1]["verification"] = result["verification"]
        else:
            result = characters._apply_points(
                scene, rig, entries, sampled_motion=True, preserve_existing_curves=True
            )
        record["next_frame"] = last
        record["events"][-1]["first_frame"] = first
        record["events"][-1]["last_frame"] = last - 1
        record["events"][-1]["max_solver_adjustment"] = result["max_solver_adjustment"]
        record["events"][-1]["max_connection_anchor_error"] = result["geometry"][
            "max_connection_anchor_error"
        ]
        return {
            "first_frame": first,
            "last_frame": last - 1,
            "max_solver_adjustment": result["max_solver_adjustment"],
            "geometry": result["geometry"],
        }

    return _stage(receipt_path, expected, completed, operation)


def save(receipt_path):
    def operation(record):
        helper = _bridge()
        view = _view(record)
        record["expected"] = snapshot(view, record["num_frames"], record)
        if record.get("writer") == "isolated-point-batch-v1":
            from mcp_for_cascadeur.host import curves

            if _sha(Path(record["directory"]) / "before.casc") != record["before_sha256"]:
                raise RuntimeError("The immutable source copy changed before final clip verification")
            if _sha(record["plan_path"]) != record["plan_sha256"]:
                raise RuntimeError("The generated Point plan changed before final clip verification")
            plan = json.loads(Path(record["plan_path"]).read_text(encoding="utf-8"))
            maximum = 0.0
            for row in plan["frames"]:
                for name, target in row["points"].items():
                    maximum = max(maximum, math.dist(target, record["expected"]["positions"][name][row["frame"]]))
            if not math.isfinite(maximum) or maximum > 3:
                raise RuntimeError("Final clip Point deviation exceeds three centimetres")
            scene = view.domain_scene()
            rig = resolve_rig(scene, record)
            geometry = curves.connection_geometry(scene, rig, frames=range(record["num_frames"]))
            if not math.isfinite(geometry["max_connection_anchor_error"]) or geometry["max_connection_anchor_error"] > 0.1:
                raise RuntimeError("Native connection anchors separated during final clip verification")
            auxiliary_report = _verify_auxiliary(scene, rig, record.get("auxiliary_bindings", []),
                record.get("auxiliary_reference", {}), range(record["num_frames"]))
            record["clip_verification"] = {
                "method": "isolated-point-batch-v1", "frames_checked": record["num_frames"],
                "max_solver_adjustment_cm": maximum,
                "max_connection_anchor_error_cm": geometry["max_connection_anchor_error"],
                "native_connections": geometry["connection_count"], "immutable_source_verified": True,
                "all_point_targets_verified": True,
                "auxiliary_joint_verification": auxiliary_report,
                "scope": "Every generated frame and Point, native connection geometry; batch guards retain curve/static and authored-channel evidence",
            }
        boundary = view.animation_boundary()
        boundary.first_frame, boundary.first_visible_frame = 0, 0
        boundary.last_frame, boundary.last_visible_frame = record["num_frames"] - 1, record["num_frames"] - 1
        output = Path(record["directory"]) / "result.casc"
        if output.exists():
            raise RuntimeError("Point result already exists; inspect before any overwrite")
        view.domain_scene().set_current_frame(0)
        view.save(str(output))
        record["active_target"] = "result"
        record["result_sha256"] = _sha(output)
        record["result_settings"] = helper.scene_settings(output)
        return {
            "path": str(output),
            "sha256": record["result_sha256"],
            "motion": helper._summary(record["expected"]),
        }

    return _stage(receipt_path, "points_applied", "points_saved", operation)


def reopen_prepare(receipt_path):
    """Keep the result alive while activating the existing recovery copy as anchor."""

    def operation(record):
        helper = _bridge()
        _, _, manager = helper._application()
        before = helper._owned_path(record, "before")
        if _sha(before) != record["before_sha256"]:
            raise RuntimeError("The retained recovery copy changed before reopen preparation")
        matches = [
            view
            for view in manager.scenes()
            if view.get_path_name() and Path(view.get_path_name()).resolve() == before
        ]
        anchor = helper._find_view(record, "before") if matches else helper._open(record, "before")
        manager.set_current_scene(anchor)
        return {
            "anchor_path": anchor.get_path_name(),
            "next_step": "point_reopen in a separate native request",
        }

    return _stage(receipt_path, "points_saved", "points_reopen_ready", operation)


def reopen(receipt_path, *, callback_scene=None):
    def operation(record):
        helper = _bridge()
        _, app, _ = helper._application()
        helper._require_callback_anchor(helper._find_view(record, "before"), callback_scene)
        path = Path(record["directory"]) / "result.casc"
        if _sha(path) != record["result_sha256"]:
            raise RuntimeError("Saved Point result changed before reopen")
        view = _view(record)
        helper._close_inactive_view(view, callback_scene)
        if not app.get_data_source_manager().load_scene(str(path)):
            raise RuntimeError("Cascadeur declined to reopen the generated Point result")
        reopened = _view(record)
        observed = snapshot(reopened, record["num_frames"], record)
        result = helper.compare_snapshots(record["expected"], observed)
        record["reopen_verification"] = result
        return result

    return _stage(receipt_path, "points_reopen_ready", "points_verified", operation)


def inspect(receipt_path):
    helper = _bridge()
    receipt, record = _read(receipt_path)
    result = {key: value for key, value in record.items() if key != "expected"}
    result["receipt_path"] = str(receipt)
    result["before_unchanged"] = _sha(Path(record["directory"]) / "before.casc") == record["before_sha256"]
    try:
        result["motion"] = helper._summary(snapshot(_view(record), record["num_frames"], record))
    except Exception as error:
        result["inspection_error"] = f"{type(error).__name__}: {error}"
    return result


def cleanup(receipt_path, *, callback_scene=None):
    def operation(record):
        helper = _bridge()
        helper._require_callback_anchor(_view(record), callback_scene)
        helper._close_inactive_view(helper._find_view(record, "before"), callback_scene)
        path = Path(record["directory"]) / "working.casc"
        if path.is_symlink() or not path.resolve().is_relative_to(Path(record["directory"]).resolve()):
            raise RuntimeError("Working copy was replaced or escaped the Point session")
        if path.exists():
            path.unlink()
        record.pop("expected", None)
        return {"retained": ["before.casc", "result.casc", "session.json"]}

    return _stage(receipt_path, "points_verified", "completed", operation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser(
        "plan", help="Map canonical generated motion onto calibrated native Points"
    )
    plan_parser.add_argument("--calibration", required=True)
    plan_parser.add_argument("--motion", required=True)
    plan_parser.add_argument("--output", required=True)
    prepare_parser = commands.add_parser("prepare", help="Create a new isolated character session")
    prepare_parser.add_argument("--workspace", required=True)
    prepare_parser.add_argument("--plan", required=True)
    prepare_parser.add_argument("--template", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
            motion = json.loads(Path(args.motion).read_text(encoding="utf-8"))
            plan = build_plan(calibration, motion)
            output = Path(args.output).resolve()
            if output.exists():
                if json.loads(output.read_text(encoding="utf-8")) != plan:
                    raise ValueError(
                        "Plan output already contains different data; choose a new exact output path"
                    )
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                _bridge()._write(output, plan)
            result = {
                "plan_path": str(output),
                "sha256": _sha(output),
                "root_displacement": plan["root_displacement"],
                "native_cluster_validation": plan["native_cluster_validation"],
            }
        else:
            result = prepare(args.workspace, args.plan, args.template)
        print(json.dumps({"ok": True, "result": result}, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error), "native_write_attempted": False}, indent=2))
        return 1


def _vector(value):
    rows = value.tolist() if hasattr(value, "tolist") else list(value)
    result = [float(row[0] if isinstance(row, (tuple, list)) else row) for row in rows]
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError("Expected three finite native coordinates")
    return result


def resolve_rig(scene, calibration):
    """Re-resolve durable native ownership instead of trusting cached objects."""
    from mcp_for_cascadeur.host import characters

    rig = characters.resolve_character(scene, identifier=calibration.get("character_id"))
    fingerprint = calibration.get("fingerprint") or calibration.get("target_fingerprint")
    if fingerprint and rig.fingerprint != fingerprint:
        raise ValueError("The target character topology changed after calibration")
    return rig


def snapshot(view, count, record):
    """Read the exact selected character using durable Point IDs and role aliases."""
    scene = view.domain_scene()
    rig = resolve_rig(scene, record)
    stored = int(scene.data_viewer().get_animation_size())
    if stored < count:
        raise RuntimeError(f"Expected {count} stored motion frames, found {stored}")
    bindings = record["point_bindings"]
    if set(bindings.values()) != set(rig.controls):
        raise RuntimeError("Saved character Point membership changed")
    data = scene.data_viewer()
    positions = {name: [_vector(data.get_data_value(rig.bindings[key].position_data, frame))
                        for frame in range(count)] for name, key in bindings.items()}
    pelvis = positions["pelvis_MainPoint"]
    delta = _sub(pelvis[-1], pelvis[0])
    return {"fingerprint": rig.fingerprint, "native_points": len(positions),
            "roles": sorted(MAPPING), "positions": positions, "stored_frames": stored,
            "root_delta": delta, "root_distance": math.sqrt(_dot(delta, delta))}


def calibrate(view=None, *, character=None, frame=0, mapping=None):
    """Read the chosen rig's anatomical reference and real Point bindings.

    This reads existing transforms and never edits names, the scene or playhead.
    Native joints outside the mapped body remain owned by Cascadeur's solver.
    """
    from mcp_for_cascadeur.host import characters, common, humanoid

    if view is None:
        _, _, view, _ = common.current_context()
    scene = view.domain_scene()
    rig = characters.resolve_character(scene, character)
    profile = humanoid.discover_mapping(scene, rig.id, mapping)
    if not profile["ready"]:
        raise ValueError("Complete the skeleton mapping before generation: " + ", ".join(profile["missing"]))
    if type(frame) is not int or not 0 <= frame < scene.data_viewer().get_animation_size():
        raise ValueError("Choose an existing reference frame")
    aliases = humanoid.point_aliases(scene, rig, profile["mapping"])
    model, data = scene.model_viewer(), scene.data_viewer()
    by_id = {str(obj): obj for obj in rig.joints}
    joints = {}
    for role, identifier in profile["mapping"].items():
        obj = by_id[identifier]
        parent = characters._parent(scene, obj)
        joints[role] = {
            "id": identifier, "name": model.get_object_name(obj), "parent": PARENTS[role],
            "native_parent": str(parent) if parent is not None else None,
            "position": _vector(data.get_data_value(characters._transform_data(scene, obj, "global_position"), frame)),
            "rotation": data.get_data_value(characters._transform_data(scene, obj, "global_rotation"), frame)
                            .to_rotation_matrix().tolist(),
        }
    points = {}
    for alias, identifier in aliases.items():
        stem = alias.split("_", 1)[0]
        side = "_" + alias[-1] if alias.endswith(("_l", "_r")) else ""
        points[alias] = {"id": identifier, "name": model.get_object_name(rig.controls[identifier]),
                         "joint": stem + side,
                         "position": _vector(data.get_data_value(rig.bindings[identifier].position_data, frame))}
    # The palm's native direction controller supplies its anatomical axis even
    # when the skeleton has no fingers or uses a different finger hierarchy.
    for side in ("l", "r"):
        joints["f_middle1_" + side] = {"parent": "hand_" + side,
            "position": points["hand_DirectionPoint_" + side]["position"], "rotation": joints["hand_" + side]["rotation"]}
    return {"format": "cascy-point-calibration-v1", "profile": "native-humanoid-v1",
            "scene_path": view.get_path_name(), "character_id": rig.id, "character_name": rig.name,
            "mapping": profile["mapping"], "fingerprint": rig.fingerprint, "frame": frame,
            "reference": "Existing native anatomical reference; source file and playhead unchanged",
            "units": "centimeters", "up_axis": "Y", "joints": joints, "points": points,
            "native_links": rig.native_links, "roles": profile["mapping"]}


if __name__ == "__main__":
    raise SystemExit(main())
