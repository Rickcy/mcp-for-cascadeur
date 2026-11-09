"""Capture key poses, transfer retained candidates and review native motion.

Generation remains in the GPU service. This host-side entry point deliberately
uses separate native requests for timeline changes and image capture: Cascadeur
must finish evaluating a pose before its viewport is rendered.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location("kimodo_workflow_" + name, HERE / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    from mcp_for_cascadeur.atomic import replace_file

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    replace_file(temporary, path)


def _native_error_detail(response):
    error = response.get("error")
    if isinstance(error, dict):
        error = error.get("message") or error.get("type")
    parts = [error] if isinstance(error, str) and error else []
    parts.extend(item["text"] for item in response.get("messages", [])[-3:]
                 if isinstance(item, dict) and str(item.get("level", "")).lower() == "error"
                 and isinstance(item.get("text"), str))
    text = "\n".join(parts)
    text = re.sub(r"(?:AIza[\w-]{20,}|AQ\.[\w-]{20,}|sk-[\w-]{20,}|gh[pousr]_[\w-]{20,}|hf_[\w-]{20,})",
                  "[redacted]", text)
    text = re.sub(r"(?i)(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|secret)[\"']?\s*[:=]\s*"
                  r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", "credential=[redacted]", text)
    text = re.sub(r"(?i)(?:authorization\s*:\s*)?(?:bearer|basic)\s+[A-Za-z0-9_+/=.-]+", "[redacted authorization]", text)
    text = re.sub(r"(https?://)[^\s/@:]+:[^\s/@]+@", r"\1[redacted]@", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
    return ("..." + text[-1600:]) if len(text) > 1600 else text


def native(client, operation, payload):
    code = (
        "import importlib.util,json\n"
        f"s=importlib.util.spec_from_file_location('kimodo_workflow_native',{str(Path(__file__).resolve())!r})\n"
        "m=importlib.util.module_from_spec(s)\ns.loader.exec_module(m)\n"
        f"result=m.native_dispatch({operation!r},{payload!r})\n"
    )
    expression = (
        "__import__('json').dumps("
        f"(lambda n:(exec({code!r},n,n),n['result'])[1])({{}}),allow_nan=False)"
    )
    result = client._request("/run", {"code": expression})
    if result.get("ok") is not True:
        detail = _native_error_detail(result)
        raise RuntimeError("Native request outcome is unknown; inspect before retrying"
                           + (". Native error: " + detail if detail else ""))
    value = json.loads(ast.literal_eval(result["value"]))
    if any(str(item.get("level", "")).lower() == "error" for item in result.get("messages", [])):
        detail = _native_error_detail(result)
        raise RuntimeError("Cascadeur reported a native error; inspect before continuing"
                           + (": " + detail if detail else ""))
    return value


def native_dispatch(operation, payload):
    """Fixed operations on the application thread; never execute caller code."""
    if operation in {"render_start", "render_status", "render_cancel"}:
        renderer = module("native_render")
        if operation == "render_start":
            return renderer.start(payload)
        return getattr(renderer, operation.removeprefix("render_"))(payload["id"], payload["receipt"])
    from mcp_for_cascadeur.host import characters, common, jobs, playback, recovery
    from mcp_for_cascadeur.host import scene as scene_api

    _, _, view, scene = common.current_context()
    if not common.get_settings().get("allow_python"):
        raise RuntimeError("Trusted Python integration is disabled in the installed adapter")
    if recovery.status().get("writes_blocked") or recovery.active_transaction() is not None:
        raise RuntimeError("Native recovery or transaction blocks the workflow")
    jobs.require_idle("kimodo_workflow")
    playback.require_idle()
    expected = payload.get("scene")
    if expected and Path(view.get_path_name()).resolve() != Path(expected).resolve():
        raise ValueError("The active scene differs from the requested workflow scene")
    if operation == "capture":
        return module("keypose_constraints").capture(
            view, payload["frames"], calibration=read(payload["calibration"]), expected_scene_path=expected
        )
    if operation == "calibrate":
        return module("point_retarget").calibrate(view, character=payload["character_id"],
                                                   frame=payload.get("frame", 0), mapping=payload.get("mapping"))
    if operation == "frame":
        frame = payload["frame"]
        if type(frame) is not int or not 0 <= frame < scene.data_viewer().get_animation_size():
            raise ValueError("An existing frame is required")
        scene.set_current_frame(frame)
        return {"frame": int(scene.get_current_frame(False))}
    if operation == "camera":
        return scene_api.camera_set(position=payload["position"], target=payload["target"])
    if operation == "camera_read":
        return {"camera": scene_api.camera_get(), "frame": int(scene.get_current_frame(False))}
    if operation == "image":
        if int(scene.get_current_frame(False)) != payload["frame"]:
            raise ValueError("The playhead changed before rendering")
        return scene_api.viewport_capture(payload["path"], width=960, height=540, samples=1)
    if operation == "mode":
        return scene_api.viewport_mode_set(payload["mode"])
    if operation == "measure":
        calibration = read(payload["calibration"]) if payload.get("calibration") else None
        rig = (module("point_retarget").resolve_rig(scene, calibration) if calibration else
               characters.resolve_character(scene, require_semantics=True))
        model, data = scene.model_viewer(), scene.data_viewer()
        if calibration:
            by_id = {str(obj): obj for obj in rig.joints}
            names = {name: by_id[item["id"]] for name, item in calibration["joints"].items() if "id" in item}
        else:
            names = {model.get_object_name(obj): obj for obj in rig.joints}
        count = payload["count"]
        first = payload.get("first_frame", 0)
        if (type(count) is not int or count < 1 or type(first) is not int or first < 0
                or first + count > data.get_animation_size()):
            raise ValueError("Unsupported native sample count")
        wanted = ("pelvis", "foot_l", "toe_l", "foot_r", "toe_r")
        rows = []
        for frame in range(first, first + count):
            rows.append({
                name: module("point_retarget")._vector(data.get_data_value(
                    characters._transform_data(scene, names[name], "global_position"), frame
                )) for name in wanted
            })
        return {"fps": 30, "units": "centimeters", "frames": rows}
    raise ValueError("Unknown workflow operation")


def measure(client, scene, count, *, calibration=None):
    """Read long clips in short read-only requests without monopolizing Qt."""
    if type(count) is not int or count < 2:
        raise ValueError("At least two native frames are required")
    rows = []
    for first in range(0, count, 120):
        payload = {"scene": str(scene), "first_frame": first, "count": min(120, count - first)}
        if calibration:
            payload["calibration"] = str(calibration)
        chunk = native(client, "measure", payload)
        if chunk.get("fps") != 30 or chunk.get("units") != "centimeters" or len(chunk.get("frames", [])) != payload["count"]:
            raise RuntimeError("Native measurement differs from the requested frame range")
        rows.extend(chunk["frames"])
    return {"fps": 30, "units": "centimeters", "frames": rows}


def constraint_preflight(payload, *, floor_tolerance_m=0.05, origin_tolerance_m=0.0001,
                         heading_tolerance_degrees=5.0):
    """Reject misplaced neural input before spending time on model inference.

    SOMA neutral joints are pelvis-relative, whereas generation constraints use
    canonical world space with a floor at Y=0. A reverse/forward algebra check
    alone cannot detect a shared translation error between those spaces.
    """
    names, rows = payload["joint_names"], payload["global_positions"]
    if payload.get("units") != "meters" or payload.get("up_axis") != "Y":
        raise ValueError("Kimodo constraints must use Y-up metres")
    for tolerance in (floor_tolerance_m, origin_tolerance_m, heading_tolerance_degrees):
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Constraint preflight tolerances must be finite nonnegative numbers")
    if not rows or not payload.get("frame_indices") or payload["frame_indices"][0] != 0:
        raise ValueError("The first captured anchor must identify canonical frame zero")
    if len(names) != len(set(names)) or any(len(row) != len(names) for row in rows):
        raise ValueError("Constraint Joint membership differs between poses")
    if any(len(point) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                                 for x in point) for row in rows for point in row):
        raise ValueError("Constraint positions must contain finite xyz vectors")
    required = {"Hips", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"}
    if not required.issubset(names):
        raise ValueError("SOMA root, hips and foot Joints are required for preflight")
    root = rows[0][names.index("Hips")]
    root_origin_error = math.hypot(root[0], root[2])
    right, left = rows[0][names.index("RightLeg")], rows[0][names.index("LeftLeg")]
    hip_x, hip_z = right[0] - left[0], right[2] - left[2]
    if math.hypot(hip_x, hip_z) < 1e-6:
        raise ValueError("The first pose has no measurable horizontal hip heading")
    heading = math.degrees(math.atan2(hip_z, -hip_x))
    feet = [i for i, name in enumerate(names)
            if name in {side + suffix for side in ("Left", "Right") for suffix in ("Foot", "ToeBase", "ToeEnd")}]
    minimum = min(row[index][1] for row in rows for index in feet)
    reasons = []
    if minimum < -floor_tolerance_m:
        reasons.append("SOMA foot anchors are below the canonical floor")
    if root_origin_error > origin_tolerance_m:
        reasons.append("The first root XZ is outside the canonical origin")
    if abs(heading) > heading_tolerance_degrees:
        reasons.append("The first pose heading conflicts with the backend's +Z initial heading")
    return {
        "accepted": not reasons, "reasons": reasons,
        "minimum_foot_anchor_y_m": minimum, "floor_tolerance_m": floor_tolerance_m,
        "first_root_origin_error_m": root_origin_error, "origin_tolerance_m": origin_tolerance_m,
        "first_heading_degrees": heading, "heading_tolerance_degrees": heading_tolerance_degrees,
        "scope": "Canonical neural-input placement; not mesh penetration or contact-quality verification",
    }


def capture_constraints(client, scene, frames, calibration, template, output):
    output = Path(output).resolve()
    if any(path.exists() for path in (output, output.with_suffix(".capture.json"), output.with_suffix(".json"))):
        raise FileExistsError("Choose a new constraint output")
    snapshot = native(client, "capture", {"scene": str(scene), "frames": frames, "calibration": str(calibration)})
    keys = module("keypose_constraints")
    neutral, skeleton = read(calibration), read(template)
    roundtrip = keys.conversion_roundtrip_report(snapshot, neutral, skeleton)
    if roundtrip.get("accepted") is not True:
        raise ValueError("Native/SOMA conversion does not preserve the captured anchor poses")
    payload = keys.convert(snapshot, neutral, skeleton)
    preflight = constraint_preflight(payload)
    if not preflight["accepted"]:
        raise ValueError("Invalid Kimodo input placement: " + "; ".join(preflight["reasons"]))
    result = keys.write_npz(payload, output)
    write(output.with_suffix(".capture.json"), snapshot)
    write(output.with_suffix(".json"), {
        **{key: value for key, value in payload.items() if key not in {"global_positions", "global_rot_mats"}},
        "conversion_roundtrip": roundtrip, "neural_input_preflight": preflight,
    })
    return {**result, "capture": str(output.with_suffix(".capture.json")), "num_frames": payload["num_frames"],
            "conversion_roundtrip": roundtrip, "neural_input_preflight": preflight}


def target_settings(scene):
    """Inspect the saved target timebase before starting a model job."""
    return module("cascadeur_bridge").scene_settings(scene)


def transfer(client, motion, calibration, template_scene, output, snapshot=None, *, progress_callback=None):
    """Keep native recovery stages separate; retain their receipt on any failure."""
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "point-plan.json"
    if plan_path.exists():
        raise FileExistsError("A transfer plan already exists; inspect its session instead of replaying")
    points, importer = module("point_retarget"), module("native_import")
    captured = read(snapshot) if snapshot else None
    plan = (module("keypose_constraints").build_return_plan(read(calibration), read(motion), captured)
            if captured else points.build_plan(read(calibration), read(motion)))
    write(plan_path, plan)
    session = points.prepare(output, plan_path, template_scene)
    receipt = session["receipt_path"]
    write(output / "transfer.json", session)
    reported = (-1, None)
    while True:
        state = importer.status(client, receipt)
        if progress_callback is not None:
            observed = read(receipt)
            phase = observed.get("phase")
            total, count = observed.get("num_frames"), observed.get("next_frame")
            if (phase != state["phase"] or type(total) is not int or total != plan.get("num_frames")
                    or type(count) is not int or not 0 <= count <= total or count < reported[0] or total < 2):
                raise RuntimeError("Native transfer progress differs from its verified session")
            cursor = re.fullmatch(r"points_ready:(\d+)", phase or "")
            if (phase == "points_prepared" and count != 0
                    or phase in {"points_applied", "points_saved", "points_reopen_ready", "points_verified", "completed"}
                       and count != total):
                raise RuntimeError("Native transfer completion differs from its durable cursor")
            verified = phase in {"points_prepared", "points_applied", "points_saved", "points_reopen_ready",
                                 "points_verified", "completed"} or cursor is not None and int(cursor[1]) == count
            if verified and (count, phase) != reported:
                progress_callback({"stage": "native_transfer", "completed": count, "total": total, "phase": phase})
                reported = (count, phase)
        if state["phase"] == "completed":
            break
        print(json.dumps({"stage": "native_transfer", "phase": state["phase"], "receipt": receipt}), flush=True)
        importer.step(client, receipt)
    record = read(receipt)
    result = {
        "scene": str(Path(record["directory"]) / "result.casc"), "receipt": receipt,
        "validation": {"geometry": True,
                       "persistence": record["reopen_verification"].get("point_coordinates_verified") is True,
                       "constraints_required": bool(snapshot)},
        "native_cluster_validation": plan["native_cluster_validation"],
    }
    if snapshot:
        observed = native(client, "capture", {
            "scene": result["scene"], "frames": [p["frame"] - captured["poses"][0]["frame"]
                                                for p in captured["poses"]],
            "calibration": str(calibration),
        })
        comparison = module("keypose_constraints").compare_captures(captured, observed)
        result["keypose_deviation"] = comparison
        result["validation"]["constraints"] = comparison["accepted"]
    write(output / "transfer.json", result)
    return result


def native_metrics(measurement, contacts):
    """Source contact labels are a diagnostic mask, not native contact truth."""
    rows, fps = measurement["frames"], measurement["fps"]
    if len(rows) != len(contacts):
        raise ValueError("Contact labels and native samples differ in length")
    speeds = []
    names = ("foot_l", "toe_l", "foot_r", "toe_r")
    for frame in range(len(rows) - 1):
        for slot, name in enumerate(names):
            if contacts[frame][slot] >= 0.5 and contacts[frame + 1][slot] >= 0.5:
                delta = [rows[frame + 1][name][axis] - rows[frame][name][axis] for axis in (0, 2)]
                speeds.append(math.hypot(*delta) * fps / 100)
    return {
        "native_contact_speed_m_s": sum(speeds) / len(speeds) if speeds else None,
        "native_contact_coverage_ratio": len(speeds) / (4 * (len(rows) - 1)),
        "contact_mask_scope": "Kimodo predicted contacts applied to native Joint trajectories; not ground truth",
    }


def wait_for_image(path, *, timeout=10, poll_interval=0.1):
    """The native renderer may acknowledge before its asynchronous PNG exists."""
    path = Path(path)
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        if path.is_file() and not path.is_symlink():
            stamp = path.stat()
            signature = (stamp.st_size, stamp.st_mtime_ns)
            if stamp.st_size >= 100:
                with path.open("rb") as stream:
                    header = stream.read(8)
                    stream.seek(-12, 2)
                    ending = stream.read(12)
                if (header == b"\x89PNG\r\n\x1a\n" and ending == b"\x00\x00\x00\x00IEND\xaeB`\x82"
                        and signature == previous):
                    return
                previous = signature
        time.sleep(poll_interval)
    raise RuntimeError("Native rendering did not finish a stable, complete PNG before the timeout")


def _encode_rendered_images(images, frame_count, video, ffmpeg):
    # Hold each sampled image for its actual source-frame interval. Appending the
    # final source frame therefore retains exactly T/30 seconds, including frame119.
    chunks, size = [], 0
    for index, item in enumerate(images):
        stop = images[index + 1]["frame"] if index + 1 < len(images) else frame_count
        repeats = stop - item["frame"]
        data = Path(item["path"]).read_bytes()
        size += len(data) * repeats
        if size > 256 * 1024 * 1024:
            raise RuntimeError("Rendered PNG stream exceeds its bounded host-memory budget")
        chunks.extend([data] * repeats)
    command = [ffmpeg, "-nostdin", "-v", "error", "-n", "-framerate", "30", "-f", "image2pipe",
               "-vcodec", "png", "-i", "pipe:0", "-frames:v", str(frame_count), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "20", str(video)]
    subprocess.run(command, input=b"".join(chunks), check=True, timeout=90, capture_output=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError("Video encoding did not produce a complete review file")


def _render_record(task, output, scene, frame_count, frame_step):
    """Read only the fresh task's bounded, identity-checked native journal."""
    expected_path = output / "native-render.json"
    original = Path(task["receipt"])
    if original.is_symlink() or original.resolve() != expected_path or not original.is_file():
        raise RuntimeError("Native render receipt is missing or outside its owned output")
    with original.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise RuntimeError("Native render receipt exceeds 4 MiB")
    record = json.loads(raw)
    frames = list(range(0, frame_count, frame_step))
    if frames[-1] != frame_count - 1:
        frames.append(frame_count - 1)
    expected_images = [
        {"view": label, "frame": frame, "path": str(output / ("frames-" + label) / f"{index:04d}.png")}
        for label in ("front", "side", "three-quarter") for index, frame in enumerate(frames)
    ]
    total = len(expected_images)
    if (not isinstance(record, dict) or record.get("id") != task["id"]
            or type(record.get("pid")) is not int or record["pid"] < 1 or record["pid"] != task.get("pid")
            or record.get("scene") != str(Path(scene).resolve()) or record.get("scene") != task.get("scene")
            or record.get("receipt") != str(expected_path)
            or type(record.get("completed")) is not int or not 0 <= record["completed"] <= total
            or record.get("total") != total or task.get("total") != total
            or record.get("frame_count") != frame_count or record.get("frame_step") != frame_step
            or record.get("fps") != 30 or record.get("frame_indices") != frames
            or record.get("images") != expected_images
            or record.get("state") not in {"running", "completed", "failed", "cancelled"}):
        raise RuntimeError("Native render receipt identity, frame coverage or owned paths changed")
    if record["state"] == "completed" and record["completed"] != total:
        raise RuntimeError("Native render receipt declares completion without every requested image")
    return record


def render(client, scene, measurement, output, *, frame_step=2, progress_callback=None):
    """Render on a bounded native Qt timer; encode exact-duration videos on CPU."""
    import imageio_ffmpeg

    if type(frame_step) is not int or not 1 <= frame_step <= 30:
        raise ValueError("frame_step must be an integer from 1 to 30")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    if measurement.get("fps") != 30 or measurement.get("units") != "centimeters":
        raise ValueError("Native render samples must use 30 FPS and centimetres")
    roots = [row["pelvis"] for row in measurement["frames"]]
    if len(roots) < 2 or any(
        len(point) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                               for x in point) for point in roots
    ):
        raise ValueError("Native render samples require at least two finite pelvis positions")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Render output is not empty; inspect its prior native receipt")
    task = None
    terminal = False
    videos = []
    try:
        task = native(client, "render_start", {"scene": str(scene), "roots": roots,
                      "frame_step": frame_step, "output": str(output), "timeout": 900})
        identity = {"id": task["id"], "receipt": task["receipt"]}
        deadline = time.monotonic() + 930
        reported = -1
        last_completed = 0
        while time.monotonic() < deadline:
            observed = _render_record(task, output, scene, len(roots), frame_step)
            if observed["completed"] < last_completed:
                raise RuntimeError("Native render receipt progress moved backwards")
            last_completed = observed["completed"]
            percent = int(observed["completed"] * 100 / observed["total"])
            if progress_callback:
                progress_callback({"stage": "native_render", "completed": observed["completed"],
                                   "total": observed["total"], "state": observed["state"]})
            if percent // 10 != reported:
                print(json.dumps({"stage": "native_render", "progress": percent, "state": observed["state"]}), flush=True)
                reported = percent // 10
            if observed["state"] != "running":
                # A busy render need not produce scene-idle events. Confirm its
                # live native owner once, after the timer has restored the scene.
                confirmed = native(client, "render_status", identity)
                if any(confirmed.get(key) != observed.get(key) for key in ("id", "pid", "scene", "receipt", "total", "completed")):
                    raise RuntimeError("Native render completion differs from its durable receipt")
                task = confirmed
                terminal = task["state"] in {"completed", "failed", "cancelled"}
                break
            time.sleep(1)
        if task["state"] != "completed" or task.get("restoration", {}).get("restored") is not True:
            raise RuntimeError("Native PNG rendering did not complete with verified state restoration: "
                               + str(task.get("error")))
        receipt = _render_record(task, output, scene, len(roots), frame_step)
        if receipt["state"] != "completed" or receipt.get("restoration", {}).get("restored") is not True:
            raise RuntimeError("Native render journal lacks confirmed completion and restoration")
        for index, label in enumerate(("front", "side", "three-quarter")):
            if progress_callback:
                progress_callback({"stage": "encoding_review", "completed": index,
                                   "total": 3, "state": "running", "view": label})
            images = [item for item in receipt["images"] if item["view"] == label]
            video = output / (label + ".mp4")
            _encode_rendered_images(images, len(roots), video, imageio_ffmpeg.get_ffmpeg_exe())
            videos.append({"path": str(video), "view": label, "fps": 30, "render_sample_step": frame_step,
                           "camera": "tracks pelvis horizontally; sampled poses held to their source durations"})
        return videos
    finally:
        if task and not terminal:
            try:
                stopped = native(client, "render_cancel", {"id": task["id"], "receipt": task["receipt"]})
                terminal = stopped["state"] in {"completed", "failed", "cancelled"}
            except Exception:
                terminal = False
        # Never erase a PNG while an unknown native owner may still be writing it.
        receipt = None
        if terminal and task and Path(task["receipt"]).is_file():
            try:
                receipt = _render_record(task, output, scene, len(roots), frame_step)
            except (OSError, ValueError, KeyError, RuntimeError):
                # A changed journal cannot grant ownership of files for cleanup.
                pass
        if receipt is not None:
            directories = set()
            for item in receipt.get("images", []):
                path = Path(item["path"])
                if path.parent.parent.resolve() == output and path.parent.name.startswith("frames-"):
                    directories.add(path.parent)
                    if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(output):
                        path.unlink()
            for directory in directories:
                if directory.is_dir() and directory.resolve().is_relative_to(output) and not any(directory.iterdir()):
                    directory.rmdir()
            for label in ("front", "side", "three-quarter"):
                video = output / (label + ".mp4")
                if (video.is_file() and not video.is_symlink() and video.resolve().parent == output
                        and not any(item["path"] == str(video) for item in videos)):
                    video.unlink()



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--scene", required=True)
    capture.add_argument("--frames", required=True, help="Comma-separated existing zero-based frames")
    capture.add_argument("--calibration", required=True)
    capture.add_argument("--template-motion", required=True)
    capture.add_argument("--output", required=True)
    retarget = commands.add_parser("transfer")
    for name in ("motion", "calibration", "template-scene", "output"):
        retarget.add_argument("--" + name, required=True)
    retarget.add_argument("--snapshot")
    review = commands.add_parser("review")
    review.add_argument("--candidates", required=True)
    review.add_argument("--prompt", required=True)
    review.add_argument("--model")
    review.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "review":
        if Path(args.output).exists():
            parser.error("Review output exists; choose a new report path")
        reviewer = module("motion_review")
        candidates = read(args.candidates)
        report = reviewer.review_candidates(candidates, args.prompt, model=args.model)
        result = {"review": report, "selection": reviewer.rank_candidates(candidates, report)}
        write(args.output, result)
    else:
        importer = module("native_import")
        client, _ = importer._runner_client(None, 35)
        if args.command == "capture":
            result = capture_constraints(client, args.scene, [int(f) for f in args.frames.split(",")],
                                         args.calibration, args.template_motion, args.output)
        else:
            result = transfer(client, args.motion, args.calibration, args.template_scene, args.output, args.snapshot)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
