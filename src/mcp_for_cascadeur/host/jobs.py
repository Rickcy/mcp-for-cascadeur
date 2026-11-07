"""Durable native jobs with a recovery receipt held through asynchronous work."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import Tool
from . import common, native_operations, recovery, state

MAX_JOB_BYTES = 8 * 1024 * 1024
TERMINAL = {"completed", "cancelled", "failed", "recovered"}
START_TOOLS = {"mocap_start", "motion_generation_start", "unbaking_start", "autophysics_start"}
CONTROL_TOOLS = {"native_job_cancel", "native_job_recover"}
_request_id = ContextVar("native_job_request_id", default=None)
_session = None
_shutdown_connected = False


class JobError(RuntimeError):
    def __init__(self, code, message, *, execution_state="not_started", details=None):
        super().__init__(message)
        self.code = code
        self.execution_state = execution_state
        self.details = details


@dataclass
class Session:
    record: dict
    receipt: object
    view: object
    scene: object
    timer: object = None
    subscription: object = None
    busy: bool = False


def _now():
    return datetime.now(timezone.utc).isoformat()


def _identifier(job_id):
    if type(job_id) is not str or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("job_id must be a 32-character lowercase hexadecimal ID")
    return job_id


def _path(job_id):
    return recovery._record_path("jobs", _identifier(job_id))


def _write(record):
    record["updated_at"] = _now()
    if len(state.canonical_bytes(record)) > MAX_JOB_BYTES:
        raise JobError("JOB_STORAGE_LIMIT", "Native job metadata exceeds its durable byte limit")
    recovery._write(_path(record["job_id"]), record)


def _read(job_id):
    record = recovery._read(_path(job_id))
    if record is None or record.get("job_id") != job_id:
        raise JobError("JOB_NOT_FOUND", "No native job has this ID")
    return record


def _public(record, transaction=None, *, connected=True):
    result = {
        key: copy.deepcopy(record[key])
        for key in (
            "job_id",
            "request_id",
            "operation",
            "status",
            "phase",
            "progress",
            "created_at",
            "updated_at",
            "host_pid",
            "original_scene",
            "working_scene",
            "source_copy",
            "output_path",
            "cancel_requested",
            "message",
            "result",
            "error",
            "step_inflight",
            "recovery",
            "native_error",
            "native_error_count",
        )
        if key in record
    }
    result["connected"] = connected
    if result["status"] == "succeeded":
        result["status"], result["phase"] = "verifying", "committing"
    result["events"] = copy.deepcopy(record.get("events", [])[-25:])
    if transaction is not None:
        result["transaction_status"] = transaction.get("status")
        if transaction.get("status") == "committed":
            result["status"], result["phase"] = "completed", "completed"
        elif transaction.get("status") in {"rolled_back", "recovered_to_checkpoint"}:
            if result["status"] not in TERMINAL:
                result["status"], result["phase"] = "recovered", "recovered"
        elif result["status"] in {"completed", "committing"}:
            result["status"] = "verifying" if connected else "recovery_required"
        elif result["status"] in {"cancelled", "failed", "recovered"}:
            result["status"], result["phase"] = "recovering", "rolling_back"
    if not connected and result["status"] not in TERMINAL:
        result["execution_state"] = "unknown"
    result["poll_after_ms"] = 500 if result["status"] not in TERMINAL else None
    return result


def read_durable_status(workspace, job_id):
    """Read a checksummed job when the native process or transport is unavailable."""
    workspace = Path(workspace).resolve()
    root = workspace / ".cascadeur-mcp" / "recovery"
    if not root.resolve().is_relative_to(workspace):
        raise JobError("JOB_STORAGE_UNSAFE", "Native job storage escaped the workspace")

    def read(path):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise JobError("JOB_STORAGE_UNSAFE", "Native job metadata escaped its managed directory")
        with path.open("rb") as stream:
            raw = stream.read(recovery.MAX_RECORD_BYTES + 1)
        if len(raw) > recovery.MAX_RECORD_BYTES:
            raise JobError("JOB_STORAGE_LIMIT", "Native job metadata exceeds its byte limit")
        record = json.loads(raw)
        digest = record.pop("record_sha256", None)
        if (
            record.get("schema_version") != 1
            or digest != hashlib.sha256(state.canonical_bytes(record)).hexdigest()
        ):
            raise JobError("INVALID_JOB_RECORD", "Native job metadata could not be verified")
        return record

    identifier = _identifier(job_id)
    record = read(root / "jobs" / f"{identifier}.json")
    if record.get("job_id") != identifier:
        raise JobError("INVALID_JOB_RECORD", "Native job identity differs from its filename")
    try:
        transaction = read(root / "transactions" / f"{_identifier(record['request_id'])}.json")
    except FileNotFoundError:
        if record.get("status") not in TERMINAL:
            raise JobError(
                "JOB_RECOVERY_MISSING", "The unfinished job's recovery receipt is unavailable"
            ) from None
        transaction = None  # Terminal receipts may be explicitly released through transaction_release.
    return _public(record, transaction, connected=False)


def read_durable_list(workspace, limit=20):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    workspace = Path(workspace).resolve()
    root = workspace / ".cascadeur-mcp" / "recovery" / "jobs"
    if root.is_symlink() or not root.resolve().is_relative_to(workspace):
        raise JobError("JOB_STORAGE_UNSAFE", "Native job storage escaped the workspace")
    paths = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return {"jobs": [read_durable_status(workspace, path.stem) for path in paths[:limit]], "connected": False}


@contextmanager
def request_context(request_id):
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def require_idle(operation):
    if operation in CONTROL_TOOLS or operation in {"scene_save", "playback_stop"}:
        return
    if _session is not None and _session.record["status"] not in TERMINAL:
        raise JobError(
            "NATIVE_JOB_ACTIVE",
            "A native job owns the scene until completion or verified recovery",
            details={"job_id": _session.record["job_id"], "phase": _session.record["phase"]},
        )
    pending = recovery._pending()
    if pending and pending.get("job_id") and pending.get("host_pid") == os.getpid():
        if operation.startswith("recovery_"):
            raise JobError(
                "NATIVE_JOB_ACTIVE", "Use native_job_recover to quiesce the owned native operation"
            )


def _sha(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise JobError("NATIVE_SAVE_UNVERIFIED", "A fresh regular native scene file is required")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _backup(session):
    record = session.record
    output = Path(record["output_path"])
    working = output.with_name(f".mcp-native-{record['job_id']}-working.casc")
    with working.open("xb"):
        pass
    session.view.save(str(working))
    output_hash = _sha(working)
    source = output.with_name(f".mcp-native-{record['job_id']}-source.casc")
    with source.open("xb") as destination, working.open("rb") as origin:
        shutil.copyfileobj(origin, destination)
        destination.flush()
        os.fsync(destination.fileno())
    if _sha(source) != output_hash:
        raise JobError("NATIVE_BACKUP_MISMATCH", "The native source copy differs from the saved model")
    record["source_copy"] = {"path": str(source), "sha256": output_hash}
    record["working_scene"] = {
        "path": str(Path(session.view.get_path_name()).resolve()),
        "name": session.view.name(),
    }
    record["plan"]["scene_identity"] = copy.deepcopy(record["working_scene"])
    session.receipt.record.update(
        scene=copy.deepcopy(record["working_scene"]), source_copy=record["source_copy"]
    )
    recovery._write(session.receipt.path, session.receipt.record)


def _subscribe(session):
    import csc
    from events import event_message_manager

    def observe(native_scene, full_message, level, _simple_message):
        if _session is not session or session.record["status"] in TERMINAL:
            return
        if native_scene is not None and native_scene != session.scene:
            return
        event = {
            "at": _now(),
            "level": "Error" if int(level) == int(csc.MessageLevel.Error) else str(level),
            "text": str(full_message)[-4000:],
        }
        if event["level"] == "Error":
            session.record["native_error"] = event
            session.record["native_error_count"] = session.record.get("native_error_count", 0) + 1
        session.record.setdefault("events", []).append(event)
        session.record["events"] = session.record["events"][-80:]

    session.subscription = event_message_manager.subscribe(observe)


def _start_timer(session):
    global _shutdown_connected
    from PySide6.QtCore import QCoreApplication, QThread, QTimer

    application = QCoreApplication.instance()
    if application is None or QThread.currentThread() != application.thread():
        raise JobError("NATIVE_THREAD_REQUIRED", "Native jobs must start on the application thread")
    if not _shutdown_connected:
        application.aboutToQuit.connect(release_native_references)
        _shutdown_connected = True
    timer = QTimer(application)
    timer.setInterval(500)
    timer.timeout.connect(lambda: _tick(session))
    session.timer = timer
    timer.start()


def _release(session):
    if session.timer is not None:
        try:
            session.timer.stop()
            session.timer.deleteLater()
        except RuntimeError:
            pass  # Qt may already be tearing down its application-owned objects.
        session.timer = None
    if session.subscription is not None:
        from events import event_message_manager

        event_message_manager.unsubscribe(session.subscription)
        session.subscription = None
    session.view = session.scene = None
    session.receipt.view = session.receipt.scene = None


def _reconcile(session):
    transaction = recovery._read(session.receipt.path)
    if transaction is None or transaction.get("status") not in {
        "committed",
        "rolled_back",
        "recovered_to_checkpoint",
    }:
        return False
    session.receipt.record = transaction
    session.receipt.committed = transaction["status"] == "committed"
    session.record.update(
        status="completed" if session.receipt.committed else "recovered",
        phase="completed" if session.receipt.committed else "recovered",
        step_inflight=False,
    )
    _release(session)
    return True


def release_native_references():
    if _session is None:
        return
    if _session.record["status"] not in TERMINAL:
        _session.record.update(status="recovery_required", phase="interrupted", step_inflight=False)
        try:
            _write(_session.record)
        except Exception:
            pass  # The pre-write recovery receipt remains authoritative on storage failure.
    _release(_session)


def _restore(session, *, cancelled=False, error=None):
    record = session.record
    record.update(status="recovering", phase="rolling_back", step_inflight=False)
    _write(record)
    source = record.get("source_copy")
    if source is not None:
        path = common.workspace_path(source["path"], must_exist=True, suffixes=(".casc",))
        if _sha(path) != source["sha256"]:
            raise JobError("NATIVE_BACKUP_CHANGED", "The native recovery source copy changed")
        proof = recovery.recovery_open(str(path))
        session.receipt.record = recovery._read(session.receipt.path)
    else:
        try:
            with recovery.receipt_context(session.receipt):
                session.receipt.rollback(error or RuntimeError("Native job cancelled"))
        except recovery.RecoveryError as failure:
            if not failure.rollback_verified:
                raise
            proof = {"verified": True, "rollback_verified": True, "details": failure.details}
    record.update(
        status="cancelled" if cancelled else "failed",
        phase="rolled_back",
        step_inflight=False,
        recovery=proof,
    )
    if error is not None:
        record["error"] = {
            "code": getattr(error, "code", type(error).__name__),
            "message": str(error)[-8000:],
            "details": getattr(error, "details", None),
        }
    try:
        _write(record)
    finally:
        _release(session)


def _complete(session, result):
    from .job_verification import protected_report

    record = session.record
    errors = [event for event in record.get("events", []) if event.get("level") == "Error"]
    if errors or record.get("native_error"):
        raise JobError("NATIVE_REPORTED_ERROR", "Cascadeur reported an error during the job", details=errors)
    after = state.capture(session.scene)
    protected = protected_report(session.receipt.checkpoint, after, record["plan"], result)
    if not protected["matches"]:
        samples = []
        for failure in protected.get("errors", [])[:10]:
            parts = failure.get("path", "").split("/")
            if len(parts) != 4 or parts[0] != "channels" or parts[2] != "values" or not parts[3].isdigit():
                continue
            key, frame = parts[1], int(parts[3])
            previous = session.receipt.checkpoint["channels"].get(key, {})
            current = after["channels"].get(key, {})
            old_values, new_values = previous.get("values", []), current.get("values", [])
            sample = {
                "channel": key,
                "name": previous.get("name"),
                "owner": previous.get("owner"),
                "frame": frame,
                "before": old_values[min(frame, len(old_values) - 1)] if old_values else None,
                "after": new_values[frame] if frame < len(new_values) else None,
            }
            if len(state.canonical_bytes(sample)) <= 4096:
                samples.append(sample)
        protected["samples"] = samples
        raise JobError(
            "NATIVE_PROTECTED_STATE_CHANGED", "The job changed protected scene state", details=protected
        )
    output = Path(record["output_path"])
    with output.open("xb"):
        pass
    session.view.save(record["output_path"])
    if record.get("native_error") or any(event.get("level") == "Error" for event in record.get("events", [])):
        raise JobError("NATIVE_SAVE_ERROR", "Cascadeur reported an error while saving the result")
    result = {
        **result,
        "path": str(output),
        "sha256": _sha(output),
        "bytes": output.stat().st_size,
        "protected_state": protected,
    }
    record.update(status="committing", phase="committing", result=result, step_inflight=False)
    _write(record)
    with recovery.receipt_context(session.receipt):
        session.receipt.verified_after = after
        proof = session.receipt.commit(result)
    record.update(status="completed", phase="completed", progress={"fraction": 1.0}, transaction=proof)
    try:
        _write(record)
    finally:
        _release(session)


def _fail(session, error):
    record = session.record
    if _reconcile(session):
        return
    driver_not_started = (
        isinstance(error, native_operations.NativeOperationError) and error.execution_state == "not_started"
    )
    if record.get("phase") != "rolling_back" and (
        record.get("safe_to_rollback") is True
        or getattr(error, "safe_to_rollback", False)
        or driver_not_started
    ):
        try:
            _restore(session, cancelled=record.get("cancel_requested", False), error=error)
            return
        except Exception as recovery_error:
            if _reconcile(session):
                return
            record["recovery_error"] = str(recovery_error)[-8000:]
    record.update(
        status="recovery_required",
        phase="failed",
        step_inflight=False,
        error={
            "code": getattr(error, "code", type(error).__name__),
            "message": str(error)[-8000:],
            "details": getattr(error, "details", None),
        },
    )
    try:
        _write(record)
    finally:
        if session.timer is not None:
            session.timer.stop()


def _tick(session):
    if _session is not session or session.busy or session.record["status"] in TERMINAL:
        return
    session.busy = True
    record = session.record
    try:
        _, _, view, scene = common.current_context()
        if view != session.view or scene != session.scene:
            raise JobError("NATIVE_JOB_SCENE_CHANGED", "The active scene changed during the native job")
        if time.time() > record["deadline"]:
            record["cancel_requested"] = True
            record["message"] = "The job deadline expired; waiting for a safe native stop"
        record["step_inflight"] = True
        record["safe_to_rollback"] = False
        record["step_number"] = record.get("step_number", 0) + 1
        _write(record)
        with recovery.receipt_context(session.receipt):
            if record.get("cancel_requested"):
                patch = native_operations.cancel(record)
            else:
                patch = native_operations.step(record)
        if type(patch) is not dict or patch.get("status") not in {
            "running",
            "awaiting_input",
            "verifying",
            "cancelling",
            "succeeded",
            "cancelled",
        }:
            raise JobError("NATIVE_JOB_DRIVER_INVALID", "The native driver returned no status")
        for key in ("phase", "progress", "driver", "message", "result", "safe_to_rollback"):
            if key in patch:
                record[key] = patch[key]
        record["effects"] = (record.get("effects", []) + patch.get("effects", []))[-100:]
        record["status"] = "recovering" if patch["status"] == "cancelled" else patch["status"]
        if patch["status"] == "cancelled":
            record["phase"] = "rolling_back"
        record["step_inflight"] = False
        _write(record)
        if (
            record.get("native_error")
            and not record.get("cancel_requested")
            and not record.get("safe_to_rollback")
        ):
            raise JobError(
                "NATIVE_REPORTED_ERROR",
                "Cascadeur reported an error; worker quiescence must be resolved before recovery",
                execution_state="unknown",
                details=record["native_error"],
            )
        if patch["status"] == "succeeded":
            if record.get("cancel_requested"):
                _restore(session, cancelled=True)
            else:
                _complete(session, patch.get("result", {}))
        elif patch["status"] == "cancelled":
            if patch.get("safe_to_rollback") is not True:
                raise JobError("NATIVE_STOP_UNVERIFIED", "The native worker's stop has not been verified")
            _restore(session, cancelled=True)
    except Exception as error:
        _fail(session, error)
    finally:
        session.busy = False


def _start(operation, arguments, output_path=None, timeout=900):
    global _session
    job_id = _request_id.get() or uuid.uuid4().hex
    if _path(job_id).exists():
        return native_job_status(job_id)
    require_idle("native_job_start")
    _, _, view, scene = common.current_context()
    recovery.guard("native_job_start", view, scene)
    if type(timeout) not in (int, float) or not 5 <= timeout <= 3600:
        raise ValueError("timeout must be between 5 and 3600 seconds")
    plan = native_operations.prepare(operation, arguments)
    original = {"path": str(Path(view.get_path_name()).resolve()), "name": view.name()}
    target = (
        Path(output_path)
        if output_path
        else Path(original["path"]).with_name(f"{Path(original['path']).stem}-{operation}-{job_id[:8]}.casc")
    )
    target = common.workspace_path(str(target), suffixes=(".casc",))
    if target.exists():
        raise JobError("NATIVE_OUTPUT_EXISTS", "Choose a new native job output path")
    receipt = recovery.Receipt(view, scene, operation, job_id)
    receipt.record["job_id"] = job_id
    recovery._write(receipt.path, receipt.record)
    record = {
        "schema_version": 1,
        "job_id": job_id,
        "request_id": job_id,
        "operation": operation,
        "status": "queued",
        "phase": "preparing",
        "progress": {"indeterminate": True},
        "created_at": _now(),
        "host_pid": os.getpid(),
        "original_scene": original,
        "output_path": str(target),
        "plan": plan,
        "driver": {},
        "events": [],
        "effects": [],
        "step_inflight": False,
        "deadline": time.time() + float(timeout),
        "cancel_requested": False,
    }
    session = Session(record, receipt, view, scene)
    _session = session
    try:
        _write(record)
        _backup(session)
        _subscribe(session)
        record.update(phase=plan["phase"], status="running")
        _write(record)
        _start_timer(session)
    except Exception as error:
        _restore(session, error=error)
    return _public(record, receipt.record)


def native_job_status(job_id):
    record = _read(job_id)
    transaction = recovery._read(recovery._record_path("transactions", record["request_id"]))
    if (transaction or {}).get("status") not in {
        "committed",
        "rolled_back",
        "recovered_to_checkpoint",
    } and record["host_pid"] != os.getpid():
        record.update(status="recovery_required", phase="interrupted")
        _write(record)
    return _public(record, transaction)


def native_jobs_list(limit=20):
    paths = list((recovery._root() / "jobs").glob("*.json"))
    paths.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return {"jobs": [native_job_status(path.stem) for path in paths[:limit]]}


def native_job_cancel(job_id):
    record = _read(job_id)
    if record["status"] in TERMINAL:
        return native_job_status(job_id)
    if _session is None or _session.record["job_id"] != job_id or record["host_pid"] != os.getpid():
        return native_job_recover(job_id)
    _session.record["cancel_requested"] = True
    _write(_session.record)
    if _session.timer is not None:
        _session.timer.start()
    return _public(_session.record)


def native_job_recover(job_id):
    record = _read(job_id)
    transaction = recovery._read(recovery._record_path("transactions", record["request_id"]))
    if transaction is None:
        raise JobError("JOB_RECOVERY_MISSING", "The native recovery receipt is unavailable")
    if transaction["status"] in {"committed", "rolled_back", "recovered_to_checkpoint"}:
        if _session is not None and _session.record["job_id"] == job_id:
            _reconcile(_session)
        return _public(record, transaction)
    if record["host_pid"] == os.getpid():
        if _session is not None and _session.record["job_id"] == job_id:
            _, app, _, _ = common.current_context()
            app.get_scene_manager().set_current_scene(_session.view)
            _session.record["cancel_requested"] = True
            _session.record["status"] = "cancelling"
            _write(_session.record)
            _start = _session.timer
            if _start is not None:
                _start.start()
            return _public(_session.record)
        raise JobError("NATIVE_RESTART_REQUIRED", "Restart Cascadeur to quiesce an orphaned native worker")
    source = record.get("source_copy")
    if not source:
        raise JobError("JOB_BACKUP_UNAVAILABLE", "The job stopped before its native source copy was verified")
    path = common.workspace_path(source["path"], must_exist=True, suffixes=(".casc",))
    if _sha(path) != source["sha256"]:
        raise JobError("NATIVE_BACKUP_CHANGED", "The native recovery source copy changed")
    pending = recovery._pending()
    if pending is None or pending["request_id"] != record["request_id"]:
        raise JobError("JOB_RECOVERY_SCOPE", "Another recovery receipt owns the write fence")
    proof = recovery.recovery_open(str(path))
    record.update(status="recovered", phase="recovered", recovery=proof, step_inflight=False)
    _write(record)
    return _public(record)


def mocap_start(
    first,
    last,
    character_id=None,
    video_path=None,
    reference_object=None,
    video_first=0,
    allow_model_download=False,
    output_path=None,
    timeout=900,
):
    arguments = {
        "first": first,
        "last": last,
        "character_id": character_id,
        "video_first": video_first,
        "allow_model_download": allow_model_download,
    }
    if video_path is not None:
        arguments["video_path"] = video_path
    if reference_object is not None:
        arguments["reference_object"] = reference_object
    return _start("mocap", arguments, output_path, timeout)


def motion_generation_start(first, last, character_id=None, output_path=None, timeout=900):
    return _start(
        "motion_generation",
        {"first": first, "last": last, "character_id": character_id},
        output_path,
        timeout,
    )


def unbaking_start(first, last, character_id=None, position_tolerance=0.01, output_path=None, timeout=900):
    return _start(
        "unbake",
        {
            "first": first,
            "last": last,
            "character_id": character_id,
            "position_tolerance": position_tolerance,
        },
        output_path,
        timeout,
    )


def autophysics_start(first, last, character_id=None, apply_mode="all_frames", output_path=None, timeout=900):
    return _start(
        "autophysics",
        {"first": first, "last": last, "character_id": character_id, "apply_mode": apply_mode},
        output_path,
        timeout,
    )


_ID = {"type": "string", "minLength": 32, "maxLength": 32}
_BASE = {
    "first": {"type": "integer", "minimum": 0},
    "last": {"type": "integer", "minimum": 1},
    "character_id": {"type": ["string", "null"]},
    "output_path": {"type": "string", "minLength": 1},
    "timeout": {"type": "number", "minimum": 5, "maximum": 3600, "default": 900},
}
TOOLS = [
    Tool(
        "mocap_start",
        "Start a durable native video-mocap job with an owned scene copy and verified completion.",
        mocap_start,
        {
            **_BASE,
            "video_path": {"type": "string"},
            "reference_object": {"type": "string"},
            "video_first": {"type": "integer", "minimum": 0},
            "allow_model_download": {"type": "boolean", "default": False},
        },
        ("first", "last"),
        False,
    ),
    Tool(
        "motion_generation_start",
        "Generate native motion between existing poses as a durable checked job.",
        motion_generation_start,
        dict(_BASE),
        ("first", "last"),
        False,
    ),
    Tool(
        "unbaking_start",
        "Convert native baked character animation to editable keys as a durable checked job.",
        unbaking_start,
        {
            **_BASE,
            "position_tolerance": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 1,
                "default": 0.01,
                "description": "Tolerance for protected Point readback outside the interval, in scene units. It does not set native Unbaking simplification accuracy.",
            },
        },
        ("first", "last"),
        False,
    ),
    Tool(
        "autophysics_start",
        "Calculate and apply native AutoPhysics through a durable job with measured results.",
        autophysics_start,
        {
            **_BASE,
            "apply_mode": {"type": "string", "enum": ["all_frames", "only_keys"], "default": "all_frames"},
        },
        ("first", "last"),
        False,
    ),
    Tool(
        "native_job_status",
        "Read durable native progress, events, result and recovery state.",
        native_job_status,
        {"job_id": _ID},
        ("job_id",),
    ),
    Tool(
        "native_jobs_list",
        "List recent durable native animation jobs and their observed outcomes.",
        native_jobs_list,
        {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}},
    ),
    Tool(
        "native_job_cancel",
        "Request a native stop and retain the scene fence until rollback is verified.",
        native_job_cancel,
        {"job_id": _ID},
        ("job_id",),
        False,
    ),
    Tool(
        "native_job_recover",
        "Recover an interrupted job from its verified native source copy after restart.",
        native_job_recover,
        {"job_id": _ID},
        ("job_id",),
        False,
    ),
]
