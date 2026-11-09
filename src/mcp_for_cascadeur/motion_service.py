"""Shared, durable Smart Motion jobs for the native panel and MCP clients.

The broker owns external workers. Cascadeur's UI only submits bounded requests
and observes receipts; neither diffusion nor HTTP inference runs on its Qt thread.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .atomic import replace_file
from .review_settings import configure_review, public_review_status
from .settings import Config, data_home, load_config

SERVICE_PORT = 9560
MAX_JSON = 4 * 1024 * 1024
ACTIVE_STATES = {"queued", "launching", "running", "cancel_requested", "orphaned"}
PATH_FIELDS = {"checkout", "calibration", "template_motion", "template_scene"}
REVIEW_FIELDS = {"provider", "model", "base_url", "media_transport", "sampling_fps",
                 "max_total_frames", "max_frames_per_video", "frame_width"}
DEFAULTS = {"frames": 120, "samples": 3, "seed": 42, "steps": 50, "port": 9552,
            "timeout": 1200, "text_weight": 2.0, "constraint_weight": 2.0,
            "sample_index": 0, "render_sample_step": 2, "no_cfg": False,
            "no_review": False, "no_activate": False, "build": False}
OPTIONS = set(DEFAULTS) | REVIEW_FIELDS | {"prompt", "source_scene", "keyframes", "output", "generate_only",
                                         "constraints", "generation_only", "cfg_enabled", "review_enabled", "auto_activate",
                                         "target_scene", "character_id", "rig_mapping", "first_frame", "no_render"}
RESUME_OPTIONS = REVIEW_FIELDS | {"no_activate", "no_review", "no_render", "render_sample_step", "review_enabled", "auto_activate"}
UUID = re.compile(r"^[0-9a-f]{32}$")


class MotionError(RuntimeError):
    def __init__(self, message, *, code="SMART_MOTION_ERROR", execution_state="not_started"):
        super().__init__(message)
        self.code, self.execution_state = code, execution_state


def _safe(value):
    text = str(value)
    text = re.sub(r"(?:AIza[\w-]{20,}|AQ\.[\w-]{20,}|sk-[\w-]{20,})", "[redacted]", text)
    return text[:2000]


def _read(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON:
        raise ValueError("Smart Motion metadata exceeds 4 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with contextlib.suppress(OSError):
            temporary.chmod(0o600)
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _private_path(name):
    root = data_home().resolve()
    if any((ancestor / ".git").exists() for ancestor in (root, *root.parents)):
        raise ValueError("Smart Motion private configuration must stay outside Git checkouts")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root / name


def _alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _number(value, name, low, high, *, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'} from {low} to {high}")


def _file(value, name, suffix=None):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(name + " must be an absolute file path")
    path = Path(value).resolve()
    if not path.is_file() or suffix and path.suffix.lower() != suffix:
        raise ValueError(name + " is missing or has the wrong file type")
    return str(path)


def validate_options(values, *, partial=False):
    if not isinstance(values, dict) or set(values) - OPTIONS:
        raise ValueError("Unsupported Smart Motion options; credentials are configured separately")
    result = dict(values) if partial else {**DEFAULTS, **values}
    if "generation_only" in result:
        if "generate_only" in result and result["generation_only"] != result["generate_only"]:
            raise ValueError("Conflicting raw generation options")
        result["generate_only"] = result.pop("generation_only")
    for positive, negative in (("cfg_enabled", "no_cfg"), ("review_enabled", "no_review"),
                               ("auto_activate", "no_activate")):
        if positive in result:
            if type(result[positive]) is not bool:
                raise ValueError(positive + " must be a boolean")
            if negative in values and values[negative] == result[positive]:
                raise ValueError("Conflicting Smart Motion boolean options")
            result[negative] = not result.pop(positive)
    if "frames" in result and (type(result["frames"]) is not int or result["frames"] < 2):
        raise ValueError("frames must be an integer of at least two")
    if "first_frame" in result and (type(result["first_frame"]) is not int or result["first_frame"] < 0):
        raise ValueError("first_frame must be a nonnegative integer")
    for name, low, high in (("samples", 1, 4), ("seed", 0, 2**32 - 1),
                            ("steps", 1, 200), ("port", 1024, 65535), ("timeout", 30, 86400),
                            ("sample_index", 0, 3), ("render_sample_step", 1, 30),
                            ("max_total_frames", 3, 1024), ("max_frames_per_video", 3, 512),
                            ("frame_width", 224, 1280)):
        if name in result:
            _number(result[name], name, low, high, integer=True)
    for name, low, high in (("text_weight", 0, 10), ("constraint_weight", 0, 10), ("sampling_fps", .01, 24)):
        if name in result:
            _number(result[name], name, low, high)
    for name in ("no_cfg", "no_review", "no_render", "no_activate", "build", "generate_only"):
        if name in result and type(result[name]) is not bool:
            raise ValueError(name + " must be a boolean")
    if "prompt" in result or not partial:
        prompt = result.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
            raise ValueError("Provide a nonempty prompt of at most 4000 characters")
        result["prompt"] = prompt.strip()
    if "samples" in result and result.get("sample_index", 0) >= result["samples"]:
        raise ValueError("sample_index must identify a requested candidate")
    if result.get("provider") not in (None, "gemini", "openai-compatible"):
        raise ValueError("Unknown visual review provider")
    if result.get("media_transport") not in (None, "video", "frames"):
        raise ValueError("media_transport must be video or frames")
    if "model" in result and (not isinstance(result["model"], str)
                               or not re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,240}", result["model"])):
        raise ValueError("Provide an exact visual model identifier")
    if "base_url" in result:
        from .review_settings import _validate
        _validate({"base_url": result["base_url"]})
    if result.get("source_scene"):
        result["source_scene"] = _file(result["source_scene"], "source_scene", ".casc")
    if result.get("target_scene"):
        result["target_scene"] = _file(result["target_scene"], "target_scene", ".casc")
    if "character_id" in result and (not isinstance(result["character_id"], str)
                                     or not 1 <= len(result["character_id"]) <= 4096):
        raise ValueError("character_id must identify one native RigInfo character")
    if "rig_mapping" in result:
        mapping = result["rig_mapping"]
        if (not isinstance(mapping, dict) or not mapping or len(mapping) > 64
                or any(not isinstance(key, str) or not 1 <= len(key) <= 128
                       or not isinstance(value, str) or not 1 <= len(value) <= 4096
                       for key, value in mapping.items())):
            raise ValueError("rig_mapping must bind body roles to native Joint IDs or names")
    if result.get("constraints"):
        result["constraints"] = _file(result["constraints"], "constraints", ".npz")
        if not partial and not result.get("generate_only"):
            raise ValueError("Prepared constraint NPZ input is supported only for raw generation")
    keyframes = result.get("keyframes")
    if isinstance(keyframes, str):
        try:
            keyframes = [int(value.strip()) for value in keyframes.split(",") if value.strip()]
        except ValueError:
            raise ValueError("keyframes must be increasing integer frame indices") from None
    if keyframes is not None:
        if (not isinstance(keyframes, list) or len(keyframes) < 2
                or any(type(value) is not int or value < 0 for value in keyframes)
                or any(a >= b for a, b in zip(keyframes, keyframes[1:]))):
            raise ValueError("Provide two or more increasing nonnegative keyframes")
        result["keyframes"] = keyframes
    if not partial:
        if bool(result.get("source_scene")) != bool(keyframes):
            raise ValueError("source_scene and keyframes must be supplied together")
        if keyframes and keyframes[-1] - keyframes[0] + 1 != result["frames"]:
            raise ValueError("Keyframes must span the requested generation duration")
        if result.get("generate_only") and (keyframes or result.get("source_scene")):
            raise ValueError("Raw generation uses optional prepared constraints, without native pose capture")
        if result.get("generate_only") and any(result.get(name) for name in ("target_scene", "character_id", "rig_mapping")):
            raise ValueError("Raw generation does not use a native target character")
        if any(result.get(name) for name in ("character_id", "rig_mapping")) and not result.get("target_scene"):
            raise ValueError("A native target_scene is required for character selection")
        if result.get("no_render") and not result.get("no_review"):
            raise ValueError("Visual review requires rendered evidence; use no_review with no_render")
    return result


def _tail(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, Path(path).stat().st_size - 8192))
            return _safe(stream.read().decode("utf-8", errors="replace"))
    except OSError:
        return ""


class Controller:
    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.exchange = (self.config.workspace / "kimodo").resolve()
        self.directory = self.exchange / "smart-motion-jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self._binding_cache = None

    def _settings(self):
        path = _private_path("smart-motion.json")
        return _read(path) if path.is_file() else {}

    def settings(self):
        value = self._settings()
        return {**{name: value.get(name, "") for name in PATH_FIELDS},
                "defaults": {**DEFAULTS, **value.get("defaults", {})}, "review": public_review_status()}

    def configure(self, values):
        if not isinstance(values, dict) or set(values) - (PATH_FIELDS | {"defaults", "review"}):
            raise ValueError("Provide Smart Motion paths, defaults or nonsecret review settings")
        with self.lock:
            if self._active():
                raise MotionError("Wait for the active job before changing its configuration", code="JOB_ACTIVE")
            settings = self._settings()
            for name in PATH_FIELDS & values.keys():
                value = values[name]
                if not isinstance(value, str) or not Path(value).is_absolute():
                    raise ValueError(name + " must be an absolute path")
                if name == "checkout":
                    root = Path(value).resolve()
                    if not (root / "integrations/kimodo/pipeline.py").is_file():
                        raise ValueError("checkout must contain integrations/kimodo/pipeline.py")
                    settings[name] = str(root)
                else:
                    settings[name] = _file(value, name, ".casc" if name == "template_scene" else ".json")
            if "defaults" in values:
                defaults = validate_options(values["defaults"], partial=True)
                if set(defaults) & {"source_scene", "keyframes", "target_scene", "character_id", "rig_mapping", "first_frame", "output"}:
                    raise ValueError("Source poses and output paths belong to individual jobs")
                settings["defaults"] = {**settings.get("defaults", {}), **defaults}
            # Validate both updates before changing either saved configuration.
            if "review" in values:
                from .review_settings import _validate
                _validate(values["review"])
                configure_review(**values["review"])
            _write(_private_path("smart-motion.json"), settings)
            if "defaults" in values and "port" in values["defaults"]:
                self._remember_runtime(values["defaults"]["port"], source="configuration")
            return self.settings()

    def capabilities(self):
        return {"model": "Kimodo-SOMA-RP-v1.2", "fps": 30, "frames": [2, None], "samples": [1, 4],
                "generation_options": sorted(OPTIONS), "review_options": sorted(REVIEW_FIELDS),
                "raw_generation_frames": [2, None], "generation_chunk_frames": 240,
                "target_character_selection": True,
                "cooperative_cancel": True, "automatic_generation_retry": False,
                "local_visual_quality_parity_verified": False, "single_active_job": True}

    def status(self):
        with self.lock:
            active = self._active()
            settings = self.settings()
        configured = bool(settings.get("checkout"))
        return {"service": "smart-motion", "pid": os.getpid(), "workspace": str(self.config.workspace),
                "configured": configured, "raw_configured": bool(settings.get("checkout")),
                "active_job": active, "review": settings["review"],
                "runtime": self._runtime_status(),
                "capabilities": self.capabilities()}

    def _receipt(self, job_id):
        if not isinstance(job_id, str) or not UUID.fullmatch(job_id):
            raise ValueError("Invalid Smart Motion job ID")
        path = self.directory / (job_id + ".json")
        if path.is_symlink() or path.resolve().parent != self.directory:
            raise ValueError("Job metadata was replaced")
        return path

    def _owned_output(self, value, *, existing=True):
        original = Path(value)
        if not original.is_absolute():
            raise ValueError("Run output must be an absolute workspace path")
        path = original.resolve()
        if original.is_symlink() or not path.is_relative_to(self.exchange) or path == self.exchange:
            raise ValueError("Run output must stay inside the configured workspace/kimodo")
        if path.is_relative_to(self.directory):
            raise ValueError("Run output cannot replace the Smart Motion job journal")
        if existing and not path.is_dir():
            raise ValueError("The run directory does not exist")
        return path

    def _load(self, job_id):
        record = _read(self._receipt(job_id))
        if record.get("job_id") != job_id or record.get("schema") != "smart-motion-job-v1":
            raise ValueError("Invalid Smart Motion job receipt")
        self._owned_output(record["output"], existing=False)
        if record.get("log_path") != str(self.directory / (job_id + ".log")):
            raise ValueError("The job log differs from its owned journal path")
        return record

    def _refresh(self, record):
        # Callers may have read a running receipt before another observer records
        # completion. Serialize the transition and reload its durable authority
        # inside the lock, so a stale observer cannot erase a known exit code.
        with self.lock:
            current = self._load(record["job_id"])
            return self._refresh_locked(current)

    def _refresh_locked(self, record):
        if record["state"] not in ACTIVE_STATES:
            return record
        process = self.processes.get(record["job_id"])
        if process is not None:
            code = process.poll()
            if code is None:
                return record
            record.update(exit_code=code, updated_at=time.time())
            run_path = Path(record["output"]) / "run.json"
            run = _read(run_path) if run_path.is_file() else {}
            progress_path = Path(record["output"]) / "progress.json"
            progress = _read(progress_path) if progress_path.is_file() else {}
            cancelled = run.get("state") == "cancelled" or progress.get("state") == "cancelled"
            if code == 0:
                record["state"] = "cancelled" if cancelled else "completed"
            else:
                record["state"] = "cancelled" if cancelled else "failed"
                record["error"] = _safe(run.get("error") or "Worker failed; inspect the retained log and receipts")
            self.processes.pop(record["job_id"], None)
            _write(self._receipt(record["job_id"]), record)
        elif _alive(record.get("pid")):
            record["state"] = "orphaned"
            record["error"] = "The previous broker lost its worker handle. Wait for that worker to stop; no operation was replayed."
        else:
            record.update(state="interrupted", updated_at=time.time(),
                          error="The worker stopped without a broker completion receipt. Inspect and explicitly resume retained stages.")
            _write(self._receipt(record["job_id"]), record)
        return record

    def _public(self, record):
        record = self._refresh(dict(record))
        output = self._owned_output(record["output"], existing=False)
        result = dict(record)
        run_path = output / "run.json"
        run = _read(run_path) if run_path.is_file() and run_path.resolve().is_relative_to(output) else {}
        progress_path = output / "progress.json"
        progress = _read(progress_path) if progress_path.is_file() and progress_path.resolve().is_relative_to(output) else {}
        if record["state"] in ACTIVE_STATES and progress.get("updated_at", 0) < record.get("started_at", 0):
            progress = {}
        if not progress:
            active = record["state"] in ACTIVE_STATES
            stage = record["action"] if active and record["action"] != "run" else run.get("stage", record.get("action"))
            progress = {"stage": stage, "state": record["state"], "fraction": None,
                        "message": "Waiting for the current stage" if active else run.get("error", "Stage finished")}
        result.update(run=run, candidates=run.get("candidates", []), selection=run.get("selection", {}),
                      progress=progress, stage=progress.get("stage"), review=None,
                      run_path=str(run_path), log_tail=_tail(record.get("log_path", "")))
        if run.get("review"):
            review = Path(run["review"]).resolve()
            if review.is_relative_to(output) and review.is_file():
                result["review"] = _read(review)
        return result

    def list_jobs(self):
        with self.lock:
            records = []
            for path in sorted(self.directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:100]:
                job = self._public(self._load(path.stem))
                records.append({**{name: job.get(name) for name in
                                    ("job_id", "action", "state", "output", "stage", "progress", "created_at", "updated_at")},
                                "prompt": job.get("run", {}).get("prompt", job.get("options", {}).get("prompt", ""))})
            return {"jobs": records}

    def job(self, job_id):
        with self.lock:
            return self._public(self._load(job_id))

    def _active(self):
        for path in self.directory.glob("*.json"):
            record = self._refresh(self._load(path.stem))
            if record["state"] in ACTIVE_STATES:
                return self._public(record)
        return None

    def _ready(self, *, native=True, target=False):
        settings = self._settings()
        required = PATH_FIELDS if native and not target else {"checkout"}
        if not all(settings.get(name) for name in required):
            raise MotionError("Configure the Kimodo checkout and target inputs first",
                              code="NOT_CONFIGURED")
        for name in required - {"checkout"}:
            _file(settings[name], name, ".casc" if name == "template_scene" else ".json")
        if not (Path(settings["checkout"]) / "integrations/kimodo/pipeline.py").is_file():
            raise ValueError("The configured pipeline checkout is unavailable")
        return settings

    def _command(self, action, options, settings, output, cancel_path):
        command = [sys.executable, "-u", "-B", str(Path(settings["checkout"]) / "integrations/kimodo/pipeline.py"),
                   action, "--output", str(output), "--cancel-file", str(cancel_path)]
        if action == "run" and not options.get("generate_only"):
            names = ("template_motion",) if options.get("target_scene") else ("calibration", "template_motion", "template_scene")
            for name in names:
                if settings.get(name):
                    command.extend(["--" + name.replace("_", "-"), settings[name]])
        for name, value in options.items():
            if name == "output" or value is None or action != "run" and name not in RESUME_OPTIONS:
                continue
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                serialized = (json.dumps(value, allow_nan=False) if name == "rig_mapping" else
                              ",".join(str(item) for item in value) if name == "keyframes" else str(value))
                command.extend([flag, serialized])
        return command

    def _launch(self, record, command, settings):
        record.update(state="launching", updated_at=time.time(), started_at=time.time(), pid=None, exit_code=None)
        _write(self._receipt(record["job_id"]), record)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONUNBUFFERED"] = "1"
        # Child credentials resolve from the same per-user private store. They
        # are never CLI arguments or copied into the durable job options.
        try:
            with Path(record["log_path"]).open("ab") as log:
                process = subprocess.Popen(command, cwd=settings["checkout"], env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            record.update(state="failed", error="The external worker could not start", updated_at=time.time())
            _write(self._receipt(record["job_id"]), record)
            raise MotionError(record["error"]) from None
        self.processes[record["job_id"]] = process
        record.update(state="running", pid=process.pid, updated_at=time.time())
        _write(self._receipt(record["job_id"]), record)
        return self._public(record)

    def start(self, options):
        with self.lock:
            if self._active():
                raise MotionError("A Smart Motion job is already active", code="JOB_ACTIVE")
            settings = self._settings()
            options = validate_options({**settings.get("defaults", {}), **validate_options(options, partial=True)})
            settings = self._ready(native=not options.get("generate_only"), target=bool(options.get("target_scene")))
            if options.get("source_scene") and not Path(options["source_scene"]).is_relative_to(self.config.workspace):
                raise ValueError("The source scene must be saved inside the configured workspace")
            if options.get("target_scene") and not Path(options["target_scene"]).is_relative_to(self.config.workspace):
                raise ValueError("The target scene must be saved inside the configured workspace")
            if options.get("constraints") and not Path(options["constraints"]).is_relative_to(self.exchange):
                raise ValueError("Constraint NPZ input must be inside the Kimodo workspace exchange")
            job_id = uuid.uuid4().hex
            output = self._owned_output(options.pop("output", str(self.exchange / ("motion-" + job_id))), existing=False)
            if output.exists():
                raise FileExistsError("Choose a new output directory; existing animation will not be overwritten")
            cancel_path = self.directory / (job_id + ".cancel")
            record = {"schema": "smart-motion-job-v1", "job_id": job_id, "action": "run", "state": "queued",
                      "options": options, "output": str(output), "cancel_path": str(cancel_path),
                      "log_path": str(self.directory / (job_id + ".log")), "created_at": time.time(),
                      "updated_at": time.time()}
            result = self._launch(record, self._command("run", options, settings, output, cancel_path), settings)
            self._remember_runtime(options["port"], job_id=job_id)
            return result

    def cancel(self, job_id):
        with self.lock:
            record = self._refresh(self._load(job_id))
            if record["state"] not in ACTIVE_STATES:
                raise MotionError("The job is no longer active; no worker was stopped", code="JOB_NOT_ACTIVE")
            if record["action"].startswith("runtime_") or record["action"] == "select":
                raise MotionError("This operation cannot be cancelled safely; wait for its receipt", code="NOT_CANCELLABLE")
            path = self.directory / (job_id + ".cancel")
            with path.open("a", encoding="utf-8") as stream:
                stream.write("Cancellation requested at " + str(time.time()) + "\n")
            record.update(state="cancel_requested", updated_at=time.time())
            _write(self._receipt(job_id), record)
            return self._public(record)

    def resume(self, job_id, options=None, *, review_only=False):
        with self.lock:
            if self._active():
                raise MotionError("Wait for the active worker before resuming or reviewing", code="JOB_ACTIVE")
            record = self._load(job_id)
            options = {} if options is None else options
            if not isinstance(options, dict) or set(options) - RESUME_OPTIONS:
                raise ValueError("Resume/review accepts only review, activation and render options")
            checked = validate_options(options, partial=True)
            if review_only and checked.get("no_review"):
                raise ValueError("A review operation cannot disable visual review")
            if review_only and "render_sample_step" in checked:
                raise ValueError("Visual review reuses saved videos; render sampling cannot be changed here")
            settings = self._ready(native=False)
            output = self._owned_output(record["output"])
            if not (output / "run.json").is_file():
                raise MotionError("No durable pipeline run exists; inspect the failed worker before a new run")
            run = _read(output / "run.json")
            if run.get("mode") == "generated_only":
                raise MotionError("Raw generation has no native stages to resume or review")
            allowed_states = {"completed"} if review_only else {"running", "failed", "interrupted", "cancelled"}
            if run.get("state") not in allowed_states:
                raise MotionError("Visual review requires a completed native run" if review_only else
                                  "Completed native runs use visual review; resume requires an interrupted or failed stage")
            if not review_only:
                inherited = {name: value for name, value in (run.get("review_identity") or {}).items()
                             if name in REVIEW_FIELDS}
                for source in (record.get("options", {}), record.get("resume_options", {}), checked):
                    layer = {name: value for name, value in source.items() if name in RESUME_OPTIONS}
                    if layer.get("provider") and layer["provider"] != inherited.get("provider"):
                        inherited = {name: value for name, value in inherited.items() if name not in REVIEW_FIELDS}
                    inherited.update(layer)
                checked = validate_options(inherited, partial=True)
                record["resume_options"] = checked
            action = "review" if review_only else "resume"
            cancel_path = self.directory / (job_id + ".cancel")
            cancel_path.unlink(missing_ok=True)
            record.setdefault("attempts", []).append({name: record.get(name) for name in
                                                     ("action", "state", "pid", "exit_code", "updated_at")})
            record.update(action=action, error=None)
            return self._launch(record, self._command(action, checked, settings, output, cancel_path), settings)

    def list_runs(self):
        runs = []
        for path in sorted(self.exchange.glob("*/run.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:100]:
            if path.is_symlink() or not path.resolve().is_relative_to(self.exchange):
                continue
            try:
                value = _read(path)
                if value.get("format") == "kimodo-cascadeur-run-v1":
                    runs.append({"output": str(path.parent.resolve()), "prompt": value.get("prompt", ""),
                                 "state": value.get("state"), "candidates": len(value.get("candidates", []))})
            except (ValueError, OSError):
                continue
        return {"runs": runs}

    def import_run(self, output):
        with self.lock:
            output = self._owned_output(output)
            run = _read(output / "run.json")
            if run.get("format") != "kimodo-cascadeur-run-v1":
                raise ValueError("This directory is not a Kimodo pipeline run")
            for path in self.directory.glob("*.json"):
                record = self._load(path.stem)
                if record["output"] == str(output):
                    return self._public(record)
            job_id = uuid.uuid4().hex
            record = {"schema": "smart-motion-job-v1", "job_id": job_id, "action": "import", "state": "completed",
                      "options": {}, "output": str(output), "created_at": time.time(), "updated_at": time.time(),
                      "log_path": str(self.directory / (job_id + ".log")), "pid": None, "exit_code": None}
            _write(self._receipt(job_id), record)
            return self._public(record)

    def select(self, job_id, candidate_id):
        with self.lock:
            if self._active():
                raise MotionError("Wait for the active worker before activating a candidate", code="JOB_ACTIVE")
            record = self._load(job_id)
            output = self._owned_output(record["output"])
            run = _read(output / "run.json")
            matches = [item for item in run.get("candidates", []) if item.get("id") == candidate_id]
            if len(matches) != 1:
                raise ValueError("Choose an exact retained candidate ID")
            candidate = matches[0]
            scene = Path(_file(candidate.get("scene"), "candidate scene", ".casc"))
            if not scene.is_relative_to(output):
                raise ValueError("The candidate scene is outside the selected run")
            receipt = Path(_file(candidate.get("receipt"), "native receipt", ".json"))
            if not receipt.is_relative_to(output):
                raise ValueError("The candidate receipt is outside the selected run")
            native = _read(receipt)
            digest = hashlib.sha256(scene.read_bytes()).hexdigest()
            if native.get("phase") != "completed" or native.get("result_sha256") != digest:
                raise MotionError("The saved native candidate differs from its completion receipt")
            settings = self._ready(native=False)
            pipeline = Path(settings["checkout"]) / "integrations/kimodo/pipeline.py"
            code = ("import importlib.util; from mcp_for_cascadeur.runner import RunnerClient; "
                    f"s=importlib.util.spec_from_file_location('smart_motion_selection',{str(pipeline)!r}); "
                    "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                    f"c=RunnerClient({self.config.endpoint!r},{str(self.config.workspace)!r},"
                    f"allow_python={self.config.allow_python!r}); m.activate_scene(c,{str(scene)!r})")
            record.update(action="select", manual_selection=candidate_id, error=None)
            return self._launch(record, [sys.executable, "-u", "-c", code], settings)

    def runtime(self, action):
        if action not in {"start", "stop", "build", "status", "logs"}:
            raise ValueError("Unknown Kimodo runtime action")
        settings = self._ready(native=False)
        if action == "status":
            return self._runtime_status()
        with self.lock:
            if self._active():
                raise MotionError("A Smart Motion worker owns the runtime", code="JOB_ACTIVE")
            port = self._requested_port()
            if action == "stop":
                binding = self._docker_binding(refresh=True)
                if not binding.get("verified") or binding.get("state") != "running" or binding.get("port") is None:
                    raise MotionError("The integration's actual Docker port could not be verified; no containers were stopped")
                port = binding["port"]
                health = self._runtime_health(port)
                if not health.get("reachable") or health.get("health", {}).get("busy") is not False:
                    raise MotionError("Kimodo idle state is unknown or busy; no containers were stopped")
            elif action == "logs":
                binding = self._docker_binding()
                if binding.get("verified") and binding.get("port") is not None:
                    port = binding["port"]
            job_id = uuid.uuid4().hex
            record = {"schema": "smart-motion-job-v1", "job_id": job_id, "action": "runtime_" + action,
                      "state": "queued", "options": {"port": port}, "output": str(self.exchange / ("runtime-" + job_id)),
                      "log_path": str(self.directory / (job_id + ".log")), "created_at": time.time(),
                      "updated_at": time.time()}
            command = ["powershell", "-NoProfile", "-File", str(Path(settings["checkout"]) / "integrations/kimodo/kimodo.ps1"),
                       "-Action", action, "-ExchangePath", str(self.exchange), "-Port", str(port)]
            result = self._launch(record, command, settings)
            if action == "start":
                self._remember_runtime(port, job_id=job_id)
            return result

    def _requested_port(self):
        path = _private_path("smart-motion-runtime.json")
        if path.is_file():
            state = _read(path)
            _number(state.get("requested_port"), "requested runtime port", 1024, 65535, integer=True)
            return state["requested_port"]
        return self._settings().get("defaults", {}).get("port", 9552)

    def _remember_runtime(self, port, *, job_id=None, source="job"):
        _write(_private_path("smart-motion-runtime.json"), {
            "requested_port": port, "job_id": job_id, "source": source, "updated_at": time.time(),
        })

    def _runtime_status(self):
        requested = self._requested_port()
        binding = self._docker_binding()
        actual = binding.get("port") if binding.get("verified") and binding.get("state") == "running" else None
        status = self._runtime_health(actual if actual is not None else requested)
        status.update(requested_port=requested, requested_endpoint=f"http://127.0.0.1:{requested}",
                      bound_port=actual, binding=binding, endpoint_source="docker_binding" if actual else "requested")
        return status

    def _docker_binding(self, *, refresh=False):
        """Inspect only this integration's fixed backend, never infer a binding from intent."""
        if not refresh and self._binding_cache and time.monotonic() - self._binding_cache[0] < 2:
            return dict(self._binding_cache[1])
        value = {"verified": False, "state": "unknown", "port": None}
        command = ["docker", "inspect", "--type", "container", "--format",
                   "[{{json .Config.Labels}},{{json .NetworkSettings.Ports}},{{json .State.Running}}]",
                   "cascadeur-kimodo-backend-1"]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if len(result.stdout) > 64 * 1024:
                raise ValueError("Unexpected Docker inspection size")
            labels, ports, running = json.loads(result.stdout)
            if (labels.get("com.docker.compose.project") != "cascadeur-kimodo"
                    or labels.get("com.docker.compose.service") != "backend" or type(running) is not bool):
                raise ValueError("Container ownership did not verify")
            if not running:
                value = {"verified": True, "state": "stopped", "port": None}
            else:
                mappings = ports.get("9552/tcp")
                if not isinstance(mappings, list) or len(mappings) != 1 or mappings[0].get("HostIp") != "127.0.0.1":
                    raise ValueError("Backend must have one loopback-only published port")
                port = int(mappings[0]["HostPort"])
                _number(port, "Docker runtime port", 1024, 65535, integer=True)
                value = {"verified": True, "state": "running", "port": port}
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, AttributeError):
            value["message"] = "The integration backend container binding could not be verified"
        self._binding_cache = (time.monotonic(), value)
        return dict(value)

    @staticmethod
    def _runtime_health(port):
        endpoint = f"http://127.0.0.1:{port}"
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
            with opener.open(endpoint + "/health", timeout=.5) as response:
                raw = response.read(128 * 1024 + 1)
            if len(raw) > 128 * 1024:
                raise ValueError("oversized health response")
            health = json.loads(raw)
            if not isinstance(health, dict) or health.get("model_checkpoint", {}).get("name") != "Kimodo-SOMA-RP-v1.2":
                raise ValueError("The endpoint is not the configured Kimodo model")
            return {"endpoint": endpoint, "reachable": True, "health": health}
        except (OSError, ValueError, TypeError, AttributeError):
            return {"endpoint": endpoint, "reachable": False, "health": None}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _endpoint(value):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("Invalid private Smart Motion endpoint")
    return value.rstrip("/")


def _request(info, method, path, payload=None, *, timeout=15):
    endpoint = _endpoint(info["endpoint"])
    if not path.startswith("/") or "?" in path or "#" in path:
        raise ValueError("Invalid Smart Motion request route")
    request = urllib.request.Request(endpoint + path, method=method,
                                     data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
                                     headers={"Content-Type": "application/json", "X-Smart-Motion-Token": info["token"]})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_JSON + 1)
    except urllib.error.HTTPError as error:
        raw = error.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise MotionError("Smart Motion response exceeds 4 MiB")
    result = json.loads(raw)
    if result.get("ok") is not True:
        error = result.get("error", {})
        raise MotionError(error.get("message", "Smart Motion request failed"), code=error.get("code", "SMART_MOTION_ERROR"),
                          execution_state=error.get("execution_state", "unknown"))
    return result["result"]


def ensure_service(config=None):
    config = config or load_config()
    info_path = _private_path("smart-motion-service.json")
    lock_path = _private_path("smart-motion-bootstrap.lock")
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if info_path.is_file():
            try:
                info = _read(info_path)
                status = _request(info, "GET", "/status", timeout=5)
                if status.get("workspace") != str(config.workspace):
                    raise MotionError("The active Smart Motion broker uses a different workspace; stop it before reconfiguration")
                return status
            except (OSError, ValueError, KeyError, MotionError) as error:
                if isinstance(error, MotionError) and "different workspace" in str(error):
                    raise
        try:
            with lock_path.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "created_at": time.time()}, stream)
        except FileExistsError:
            try:
                owner = _read(lock_path)
                if not _alive(owner.get("pid")):
                    lock_path.unlink(missing_ok=True)
                    continue
            except (OSError, ValueError):
                pass
            time.sleep(.15)
            continue
        try:
            command = [sys.executable, "-m", "mcp_for_cascadeur.motion_service", "--workspace", str(config.workspace),
                       "--native-endpoint", config.endpoint]
            if config.allow_python:
                command.append("--allow-python")
            with _private_path("smart-motion-service.log").open("ab") as log:
                subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while time.monotonic() < deadline:
                if info_path.is_file():
                    try:
                        status = _request(_read(info_path), "GET", "/status", timeout=5)
                        if status.get("workspace") == str(config.workspace):
                            return status
                    except (OSError, ValueError, KeyError, MotionError):
                        pass
                time.sleep(.1)
            raise MotionError("Smart Motion could not start; inspect its private service log")
        finally:
            lock_path.unlink(missing_ok=True)
    raise MotionError("Smart Motion startup is still owned by another client")


def call_service(method, path, payload=None, *, config=None, timeout=15):
    ensure_service(config)
    info = _read(_private_path("smart-motion-service.json"))
    try:
        return _request(info, method, path, payload, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
        raise MotionError("Smart Motion request outcome is unknown; inspect job status before retrying",
                          execution_state="unknown" if method == "POST" else "not_started") from error


def make_server(controller, token, *, port=SERVICE_PORT):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, format, *args):
            pass

        def _authorize(self):
            try:
                local = ipaddress.ip_address(self.client_address[0]).is_loopback
                host = urllib.parse.urlsplit("http://" + self.headers.get("Host", ""))
                correct_host = host.hostname in {"127.0.0.1", "localhost"} and host.port == self.server.server_port
                origin = self.headers.get("Origin")
                correct_origin = origin is None or origin in {
                    f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
                supplied = self.headers.get("X-Smart-Motion-Token", "")
                return local and correct_host and correct_origin and secrets.compare_digest(supplied, token)
            except ValueError:
                return False

        def _reply(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _dispatch(self, method):
            if not self._authorize():
                self._reply(403, {"ok": False, "error": {"code": "FORBIDDEN", "message": "Local authorization required"}})
                return
            try:
                payload = None
                if method == "POST":
                    if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                        raise ValueError("Use an application/json request")
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= MAX_JSON:
                        raise ValueError("Request body is missing or exceeds 4 MiB")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be an object")
                path = self.path.split("/")
                if method == "GET" and self.path == "/status":
                    result = controller.status()
                elif method == "GET" and self.path == "/settings":
                    result = controller.settings()
                elif method == "POST" and self.path == "/settings":
                    result = controller.configure(payload)
                elif method == "GET" and self.path == "/jobs":
                    result = controller.list_jobs()
                elif method == "GET" and self.path == "/runs":
                    result = controller.list_runs()
                elif method == "POST" and self.path == "/jobs/import":
                    if set(payload) != {"output"}:
                        raise ValueError("Import requires exactly one output directory")
                    result = controller.import_run(payload["output"])
                elif method == "POST" and self.path == "/jobs":
                    result = controller.start(payload)
                elif method == "GET" and len(path) == 3 and path[1] == "jobs":
                    result = controller.job(path[2])
                elif method == "POST" and len(path) == 4 and path[1] == "jobs":
                    if path[3] == "cancel" and not payload:
                        result = controller.cancel(path[2])
                    elif path[3] in {"resume", "review"}:
                        result = controller.resume(path[2], payload, review_only=path[3] == "review")
                    elif path[3] == "select" and set(payload) == {"candidate_id"}:
                        result = controller.select(path[2], payload["candidate_id"])
                    else:
                        raise ValueError("Unknown job operation or extra arguments")
                elif method == "POST" and self.path == "/runtime" and set(payload) == {"action"}:
                    result = controller.runtime(payload["action"])
                else:
                    raise ValueError("Unknown Smart Motion route")
                self._reply(200, {"ok": True, "result": result})
            except Exception as error:
                self._reply(400, {"ok": False, "error": {"code": getattr(error, "code", type(error).__name__),
                                                         "message": _safe(error),
                                                         "execution_state": getattr(error, "execution_state", "not_started")}})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


@contextlib.contextmanager
def _broker_lock():
    path = _private_path("smart-motion-broker.lock")
    for attempt in range(2):
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "created_at": time.time()}, stream)
            break
        except FileExistsError:
            owner = _read(path)
            if attempt == 0 and not _alive(owner.get("pid")):
                path.unlink(missing_ok=True)
                continue
            raise MotionError("Another Smart Motion broker already owns this private configuration") from None
    try:
        yield
    finally:
        with contextlib.suppress(OSError, ValueError):
            if _read(path).get("pid") == os.getpid():
                path.unlink(missing_ok=True)


def _serve(config, port):
    token = secrets.token_urlsafe(32)
    server = make_server(Controller(config), token, port=port)
    info_path = _private_path("smart-motion-service.json")
    _write(info_path, {"pid": os.getpid(), "endpoint": f"http://127.0.0.1:{server.server_port}", "token": token,
                       "workspace": str(config.workspace), "created_at": time.time()})
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
        with contextlib.suppress(OSError, ValueError):
            if _read(info_path).get("pid") == os.getpid():
                info_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--native-endpoint")
    parser.add_argument("--allow-python", action="store_true", default=None)
    parser.add_argument("--port", type=int, default=SERVICE_PORT)
    args = parser.parse_args(argv)
    config = load_config()
    config = Config(endpoint=args.native_endpoint or config.endpoint,
                    workspace=args.workspace or config.workspace,
                    allow_python=config.allow_python if args.allow_python is None else args.allow_python,
                    cascadeur_home=config.cascadeur_home)
    with _broker_lock():
        _serve(config, args.port)


if __name__ == "__main__":
    main()
