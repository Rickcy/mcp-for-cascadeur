"""Transfer generated SOMA motion to native Cascy Points with independent maths.

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
import shutil
import time
import uuid
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
    if type(count) is not int or not 2 <= count <= 120 or motion.get("fps") != 30:
        raise ValueError("Point retargeting currently accepts 2-120 frames at 30 FPS")
    source_index = {name: index for index, name in enumerate(motion["joint_names"])}
    missing = set(MAPPING.values()) - source_index.keys()
    if missing:
        raise ValueError("Source motion is missing mapped joints: " + ", ".join(sorted(missing)))
    target = calibration["joints"]
    anchors = {name: calibration["points"][_main_point(name)]["position"] for name in MAPPING}
    if set(MAPPING) - target.keys():
        raise ValueError("Target calibration is missing native Cascy body joints")
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
            raise ValueError("Unexpected native Cascy hierarchy at " + name)
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
    return path, record


def _write(receipt_path, record):
    """Share the atomic receipt writer with the native CLI transport."""
    _bridge()._write(receipt_path, record)


def prepare(workspace_root, plan_path, template_path):
    """Create a fresh Cascy-only session; never reuse the failed Pro-paste scene."""
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
    }
    shutil.copyfile(template, directory / "before.casc")
    shutil.copyfile(template, directory / "working.casc")
    receipt = directory / "session.json"
    helper._write(receipt, record)
    return {
        "receipt_path": str(receipt),
        "working_path": str(directory / "working.casc"),
        "phase": record["phase"],
    }


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
        from mcp_for_cascadeur.host import characters

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
                raise RuntimeError("Cascadeur declined the new isolated Cascy working scene")
        view = _view(record)
        manager.set_current_scene(view)
        scene = view.domain_scene()
        rig = characters.resolve_character(scene, require_semantics=True)
        if rig.fingerprint != record["target_fingerprint"]:
            raise RuntimeError("The fresh Cascy template differs from the calibrated native topology")
        tracks = list(rig.tracks)
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
        rig = characters.resolve_character(scene, require_semantics=True, require_hands=True)
        if rig.fingerprint != record["target_fingerprint"]:
            raise RuntimeError("Target Point topology changed before the next generated chunk")
        names = {scene.model_viewer().get_object_name(obj): key for key, obj in rig.controls.items()}
        entries = [
            {
                "frame": row["frame"],
                "pose": {
                    names[name]: {"position": _vector(position)} for name, position in row["points"].items()
                },
            }
            for row in plan["frames"][first:last]
        ]
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
        record["expected"] = helper._snapshot(view, record["num_frames"])
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
        observed = helper._snapshot(reopened, record["num_frames"])
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
        result["motion"] = helper._summary(helper._snapshot(_view(record), record["num_frames"]))
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
    prepare_parser = commands.add_parser("prepare", help="Create a new isolated Cascy session")
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


def calibrate(view=None):
    """Read complete rest Joint/Point bindings; do not change scene or playhead."""
    from mcp_for_cascadeur.host import characters, common

    if view is None:
        _, _, view, _ = common.current_context()
    scene = view.domain_scene()
    rig = characters.resolve_character(scene, require_semantics=True)
    model, data = scene.model_viewer(), scene.data_viewer()
    joints = {}
    for obj in sorted(rig.joints, key=lambda item: model.get_object_name(item)):
        name = model.get_object_name(obj)
        parent = characters._parent(scene, obj)
        parent_name = model.get_object_name(parent) if parent is not None else None
        position = data.get_data_value(characters._transform_data(scene, obj, "global_position"), 0)
        rotation = data.get_data_value(characters._transform_data(scene, obj, "global_rotation"), 0)
        joints[name] = {
            "id": str(obj),
            "parent": parent_name,
            "position": _vector(position),
            "rotation": rotation.to_rotation_matrix().tolist(),
        }
    points = {}
    for key, binding in rig.bindings.items():
        name = model.get_object_name(rig.controls[key])
        stem = name.split("_", 1)[0]
        side = "_" + name[-1] if name.endswith(("_l", "_r")) else ""
        joint = stem + side
        if joint not in joints:
            raise ValueError(f"Point {name} has no unambiguous native Joint binding")
        points[name] = {
            "id": key,
            "joint": joint,
            "role": binding.role,
            "slot": binding.slot,
            "position": _vector(data.get_data_value(binding.position_data, 0)),
        }
    return {
        "format": "cascy-point-calibration-v1",
        "scene_path": view.get_path_name(),
        "fingerprint": rig.fingerprint,
        "frame": 0,
        "units": "centimeters",
        "up_axis": "Y",
        "joints": joints,
        "points": points,
        "native_links": rig.native_links,
        "roles": rig.roles,
    }


if __name__ == "__main__":
    raise SystemExit(main())
