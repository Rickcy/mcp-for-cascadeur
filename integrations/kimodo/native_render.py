"""Bounded PNG rendering on Cascadeur's Qt thread, without native video export."""

from __future__ import annotations

import builtins
import json
import math
import os
import time
import uuid
from pathlib import Path

SLOT = "_mcp_kimodo_render_task"
VIEWS = {"front": [0, 30, 550], "side": [550, 30, 0], "three-quarter": [400, 60, 400]}


def _write(task):
    record = {key: value for key, value in task.items() if not key.startswith("_")}
    target = Path(task["receipt"])
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def _public(task):
    return {key: task[key] for key in ("id", "pid", "scene", "state", "phase", "completed", "total",
                                      "receipt", "error", "restoration")}


def _complete_png(path, previous):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        return False, None
    info = path.stat()
    signature = (info.st_size, info.st_mtime_ns)
    if info.st_size < 100:
        return False, signature
    with path.open("rb") as stream:
        header = stream.read(8)
        stream.seek(-12, 2)
        ending = stream.read(12)
    complete = header == b"\x89PNG\r\n\x1a\n" and ending == b"\x00\x00\x00\x00IEND\xaeB`\x82"
    return complete and signature == previous, signature


class _Native:
    def __init__(self):
        from PySide6.QtCore import QCoreApplication, QThread, QTimer

        from mcp_for_cascadeur.host import characters, common, jobs, playback, recovery
        from mcp_for_cascadeur.host import scene as scene_api

        self.application = QCoreApplication.instance()
        if self.application is None or QThread.currentThread() != self.application.thread():
            raise RuntimeError("Rendering must start on Cascadeur's application thread")
        self.QTimer = QTimer
        self.characters, self.common, self.jobs = characters, common, jobs
        self.playback, self.recovery, self.scene_api = playback, recovery, scene_api

    def inspect(self):
        if not self.common.get_settings().get("allow_python"):
            raise RuntimeError("Trusted Python integration is disabled")
        if self.recovery.status().get("writes_blocked") or self.recovery.active_transaction() is not None:
            raise RuntimeError("Native recovery or transaction became active")
        self.jobs.require_idle("kimodo_render")
        self.playback.require_idle()
        _, _, view, scene = self.common.current_context()
        rig = self.characters.resolve_character(scene, require_semantics=True)
        path = Path(view.get_path_name()).resolve(strict=True)
        stamp = path.stat()
        return {"scene": str(path), "fingerprint": rig.fingerprint, "frame": int(scene.get_current_frame(False)),
                "frame_count": scene.data_viewer().get_animation_size(), "file_stamp": [stamp.st_size, stamp.st_mtime_ns],
                "camera": self.scene_api.camera_get()}

    def validate_output(self, output):
        output = Path(output).resolve(strict=True)
        workspace = self.common.get_settings()["workspace"].resolve()
        if not output.is_relative_to(workspace) or not output.is_dir():
            raise ValueError("Render output must be an existing directory in the configured workspace")
        return output

    def prepare(self, frame, position, target):
        _, _, _, scene = self.common.current_context()
        scene.set_current_frame(frame)
        self.scene_api.viewport_mode_set("View")
        self.scene_api.camera_set(position=position, target=target)

    def capture(self, path):
        self.scene_api.viewport_capture(str(path), width=960, height=540, samples=1)

    def restore(self, task):
        import numpy as np

        csc, app, _, _ = self.common.current_context(require_scene=False)
        views = [view for view in app.get_scene_manager().scenes()
                 if view.get_path_name() and str(Path(view.get_path_name()).resolve()) == task["scene"]]
        if len(views) != 1:
            raise RuntimeError("The original render scene is missing or ambiguous; no other scene was changed")
        view = views[0]
        scene = view.domain_scene()
        if self.characters.resolve_character(scene, require_semantics=True).fingerprint != task["fingerprint"]:
            raise RuntimeError("The original render topology changed; restoration was not applied")
        viewport = view.active_viewport().domain_viewport()
        original = task["original"]["camera"]
        errors = []
        try:
            camera = viewport.camera_struct()
            camera.position = np.asarray(original["position"], dtype=np.float32)
            camera.target = np.asarray(original["target"], dtype=np.float32)
            viewport.set_camera_struct(camera)
        except Exception as error:
            errors.append("camera: " + str(error))
        try:
            viewport.set_mode_visualizers(csc.view.ViewportMode.__members__[original["mode"].split(".")[-1]])
        except Exception as error:
            errors.append("mode: " + str(error))
        try:
            scene.set_current_frame(task["original"]["frame"])
        except Exception as error:
            errors.append("frame: " + str(error))
        return {"restored": not errors, "frame": task["original"]["frame"], "errors": errors}

    def timer(self, callback):
        timer = self.QTimer(self.application)
        timer.setInterval(50)
        timer.timeout.connect(callback)
        return timer

    def connect_shutdown(self, callback):
        self.application.aboutToQuit.connect(callback)

    def disconnect_shutdown(self, callback):
        self.application.aboutToQuit.disconnect(callback)


def _finish(task, state, error=None):
    environment = task.get("_environment")
    timer = task.get("_timer")
    if timer is not None:
        timer.stop()
        try:
            timer.timeout.disconnect(task["_callback"])
        except (RuntimeError, TypeError):
            pass
        timer.deleteLater()
    if environment is not None:
        try:
            environment.disconnect_shutdown(task["_shutdown"])
        except (RuntimeError, TypeError):
            pass
        try:
            task["restoration"] = environment.restore(task)
            if task["restoration"].get("restored") is not True:
                state = "failed"
                if error is None:
                    error = "Native render state restoration did not complete"
        except Exception as restore_error:
            task["restoration"] = {"restored": False, "error": str(restore_error)}
            state = "failed"
    task.update(state=state, phase="terminal", error=str(error) if error else None, finished_at=time.time())
    task["_timer"] = task["_callback"] = task["_shutdown"] = task["_environment"] = None
    _write(task)


def _tick(task, *, now=None):
    if task["state"] != "running":
        return
    now = time.monotonic() if now is None else now
    try:
        if now - task["_started"] > task["timeout"]:
            raise RuntimeError("Native PNG rendering exceeded its bounded timeout")
        current = task["_environment"].inspect()
        if os.getpid() != task["pid"] or any(current[key] != task[key] for key in ("scene", "fingerprint", "file_stamp")):
            raise RuntimeError("The render scene, saved file, process or topology changed")
        if current["frame"] != task["_expected_frame"]:
            raise RuntimeError("The playhead moved outside the render task")
        if task["completed"] == task["total"]:
            _finish(task, "completed")
            return
        item = task["images"][task["completed"]]
        if task["phase"] == "prepare":
            center = task["roots"][item["frame"]]
            target = [center[0], 105, center[2]]
            position = [target[axis] + VIEWS[item["view"]][axis] for axis in range(3)]
            task["_environment"].prepare(item["frame"], position, target)
            task["_expected_frame"] = item["frame"]
            task["_phase_started"] = now
            task["phase"] = "settle"
        elif task["phase"] == "settle" and now - task["_phase_started"] >= 0.1:
            path = Path(item["path"])
            if path.exists():
                raise RuntimeError("A render image already exists; it was not overwritten")
            task["_environment"].capture(path)
            task["_phase_started"] = now
            task["_signature"] = None
            task["phase"] = "waiting"
        elif task["phase"] == "waiting":
            ready, task["_signature"] = _complete_png(item["path"], task.get("_signature"))
            if ready:
                task["completed"] += 1
                task["phase"] = "prepare"
                _write(task)
            elif now - task["_phase_started"] > 15:
                raise RuntimeError("Native rendering did not finish a complete PNG")
    except Exception as error:
        _finish(task, "failed", error)


def start(payload):
    previous = getattr(builtins, SLOT, None)
    if previous and previous["state"] == "running":
        raise RuntimeError("A Kimodo native render task is already active")
    environment = _Native()
    original = environment.inspect()
    scene = str(Path(payload["scene"]).resolve(strict=True))
    roots = payload["roots"]
    step = payload.get("frame_step", 2)
    timeout = payload.get("timeout", 900)
    if type(step) is not int or not 1 <= step <= 30 or type(timeout) is not int or not 1 <= timeout <= 900:
        raise ValueError("Render step or timeout is outside the supported range")
    if (original["scene"] != scene or not isinstance(roots, list) or not 2 <= len(roots) <= min(120, original["frame_count"])
            or any(not isinstance(p, list) or len(p) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in p) for p in roots)):
        raise ValueError("Render scene or root sample sequence is invalid")
    output = environment.validate_output(payload["output"])
    receipt = output / "native-render.json"
    if receipt.exists():
        raise FileExistsError("Render receipt already exists; inspect the previous outcome")
    frames = list(range(0, len(roots), step))
    if frames[-1] != len(roots) - 1:
        frames.append(len(roots) - 1)
    images = []
    for label in VIEWS:
        directory = output / ("frames-" + label)
        directory.mkdir(exist_ok=False)
        images.extend({"view": label, "frame": frame, "path": str(directory / f"{index:04d}.png")}
                      for index, frame in enumerate(frames))
    task = {"id": uuid.uuid4().hex, "pid": os.getpid(), "scene": scene,
            "fingerprint": original["fingerprint"], "file_stamp": original["file_stamp"],
            "original": original, "roots": roots, "images": images, "frame_indices": frames,
            "frame_count": len(roots), "fps": 30, "frame_step": step, "state": "running", "phase": "prepare",
            "completed": 0, "total": len(images), "receipt": str(receipt), "error": None, "restoration": None,
            "timeout": timeout, "started_at": time.time(), "_started": time.monotonic(),
            "_expected_frame": original["frame"], "_environment": environment}
    task["_callback"] = lambda: _tick(task)
    task["_shutdown"] = lambda: _finish(task, "cancelled", "Cascadeur is shutting down")
    task["_timer"] = environment.timer(task["_callback"])
    setattr(builtins, SLOT, task)
    _write(task)
    environment.connect_shutdown(task["_shutdown"])
    task["_timer"].start()
    return _public(task)


def status(identifier, receipt):
    task = getattr(builtins, SLOT, None)
    if task and task["id"] == identifier and task["receipt"] == str(Path(receipt).resolve()):
        return _public(task)
    record = json.loads(Path(receipt).read_text(encoding="utf-8"))
    if record["id"] != identifier:
        raise ValueError("Render task receipt differs from its ID")
    if record["state"] == "running":
        record.update(state="interrupted", error="Native render owner is absent; the task was not restarted")
    return _public(record)


def cancel(identifier, receipt):
    task = getattr(builtins, SLOT, None)
    if not task or task["id"] != identifier or task["receipt"] != str(Path(receipt).resolve()):
        return status(identifier, receipt)
    if task["state"] == "running":
        _finish(task, "cancelled", "Cancelled by the owning host workflow")
    return _public(task)
