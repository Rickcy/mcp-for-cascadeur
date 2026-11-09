"""Native operations executed by Cascadeur's own main-thread script runner."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from .. import __version__
from ..contracts import Tool
from .common import configure, current_context, get_settings, json_value


def app_status() -> dict:
    csc, application, view, scene = current_context(require_scene=False)
    return {
        "application": "Cascadeur",
        "adapter_version": __version__,
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "thread_id": threading.get_ident(),
        "scene": view.name() if view is not None else None,
        "frame": scene.get_current_frame() if scene is not None else None,
        "workspace": str(get_settings()["workspace"]),
        "python_enabled": get_settings()["allow_python"],
    }


def python_execute(code: str) -> dict:
    if not get_settings()["allow_python"]:
        raise PermissionError("Python execution is disabled; run setup --allow-python to enable it")
    csc, application, view, scene = current_context(require_scene=False)
    namespace = {
        "csc": csc,
        "app": application,
        "view": view,
        "scene": scene,
        "workspace": get_settings()["workspace"],
    }
    try:
        compiled = compile(code, "<cascadeur-python>", "eval")
    except SyntaxError:
        exec(compile(code, "<cascadeur-python>", "exec"), namespace, namespace)
        result = namespace.get("result")
    else:
        result = eval(compiled, namespace, namespace)
    return {
        "value": json_value(result),
        "execution_state": "completed",
        "note": "Model changes use the bounded recovery transaction; external Python side effects are not reversible.",
    }


def registry() -> dict[str, Tool]:
    from . import (
        animation,
        character_edit,
        characters,
        data_channels,
        exchange,
        inspection,
        jobs,
        playback,
        profiles,
        recovery,
        scene,
        smart_motion,
        subtrees,
        workflows,
    )

    definitions = [
        Tool(
            "app_status",
            "Check the live Cascadeur process, current scene, adapter and workspace.",
            app_status,
        ),
        Tool(
            "python_execute",
            "Execute Python on Cascadeur's main thread with csc, app, view, scene and workspace. "
            "Statements return the variable result; expressions return their value. Full application privileges; "
            "bounded model recovery excludes external side effects. Requires setup --allow-python.",
            python_execute,
            {"code": {"type": "string", "minLength": 1, "maxLength": 200_000}},
            ("code",),
            False,
        ),
        *scene.TOOLS,
        *inspection.TOOLS,
        *animation.TOOLS,
        *profiles.TOOLS,
        *characters.TOOLS,
        *character_edit.TOOLS,
        *playback.TOOLS,
        *recovery.TOOLS,
        *workflows.TOOLS,
        *exchange.TOOLS,
        *subtrees.TOOLS,
        *data_channels.TOOLS,
        *jobs.TOOLS,
        Tool("smart_motion_panel", "Open the native Smart Motion panel for Kimodo generation, body anchors, retained variants and visual review. Does not edit animation when opening.",
             smart_motion.show_panel),
    ]
    result = {tool.name: tool for tool in definitions}
    if len(result) != len(definitions):
        raise RuntimeError("Duplicate tool names in the native registry")
    return result


SESSION_OPERATIONS = {
    "scene_new",
    "scene_open",
    "scene_activate",
    "selection_set",
    "timeline_set_frame",
    "timeline_set_range",
    "tracks_select_frames",
    "camera_set",
    "camera_frame",
    "viewport_mode_set",
    "playback_start",
    "playback_stop",
}
OUTPUT_OPERATIONS = {
    "scene_save",
    "pose_snapshot",
    "pose_release_snapshot",
    "rig_profile_create",
    "viewport_capture",
    "fbx_export",
    "fbx_export_verified",
    "checkpoint_create",
    "checkpoint_release",
    "transaction_release",
    "character_pose_snapshot",
    "subtree_snapshot",
    "subtree_release_snapshot",
}


class WriteVerificationError(RuntimeError):
    code = "WRITE_VERIFICATION_FAILED"

    def __init__(self, verification):
        super().__init__("The committed native values did not satisfy the requested postconditions")
        self.details = verification


class NativeMessageError(RuntimeError):
    code = "NATIVE_EVENT_ERROR"

    def __init__(self, messages):
        super().__init__("; ".join(item["text"] for item in messages)[-8000:])
        self.details = {"messages": messages[-50:]}


def _call_handler(tool, arguments):
    """Observe error events before the transaction can be committed."""
    messages = []
    manager = None
    subscription = None
    if "csc" in sys.modules:
        import csc
        from events import event_message_manager

        manager = event_message_manager

        def observe(_scene, full_message, level, _simple_message):
            if int(level) == int(csc.MessageLevel.Error):
                messages.append({"level": "Error", "text": str(full_message)})

        subscription = manager.subscribe(observe)
    try:
        value = json_value(tool.handler(**arguments))
        if messages:
            raise NativeMessageError(messages)
        return value
    finally:
        if manager is not None:
            manager.unsubscribe(subscription)


def _verify_result(value):
    if not isinstance(value, dict):
        return
    report = value.get("verification")
    if isinstance(report, dict) and report.get("matches") is False:
        raise WriteVerificationError(report)
    if value.get("key_mapping_verified") is False or value.get("channel_values_verified") is False:
        raise WriteVerificationError(
            {key: item for key, item in value.items() if "verif" in key or "mismatch" in key}
        )


def _execute_write(tool, arguments, request_id):
    from . import jobs, playback, recovery

    if tool.name in jobs.START_TOOLS and jobs._path(request_id).exists():
        recorded = jobs._read(request_id)
        operation = "unbake" if tool.name == "unbaking_start" else tool.name.removesuffix("_start")
        if recorded["operation"] != operation:
            raise jobs.JobError(
                "JOB_REQUEST_REUSED",
                "This request ID already belongs to another native operation",
                details={"job_id": request_id, "operation": recorded["operation"]},
            )
        return jobs.native_job_status(request_id), None

    _, _, view, scene = current_context(require_scene=False)
    recovery.guard(tool.name, view, scene)
    jobs.require_idle(tool.name)
    if tool.name == "scene_save" and recovery.status()["writes_blocked"]:
        target = Path(arguments["path"]).expanduser().resolve()
        if arguments.get("overwrite", False) or target.exists():
            raise recovery.RecoveryError(
                "RECOVERY_REQUIRED", "A quarantined scene can only be saved to a new file"
            )
    if tool.name not in {
        "playback_start",
        "playback_stop",
        "camera_set",
        "camera_frame",
        "viewport_capture",
        "viewport_mode_set",
    }:
        playback.require_idle()
    managed = tool.name.startswith("recovery_") or tool.name in (
        SESSION_OPERATIONS | OUTPUT_OPERATIONS | jobs.START_TOOLS | jobs.CONTROL_TOOLS
    )
    if managed:
        with jobs.request_context(request_id):
            value = _call_handler(tool, arguments)
        _verify_result(value)
        return value, None
    if scene is None:
        raise RuntimeError("Open a scene before editing")
    with recovery.transaction(view, scene, tool.name, request_id) as receipt:
        value = _call_handler(tool, arguments)
        _verify_result(value)
        summary = receipt.commit(value)
    return value, summary


def dispatch(
    name: str,
    arguments: dict,
    *,
    workspace: str | None = None,
    allow_python: bool = False,
    version: str = __version__,
    request_id: str | None = None,
) -> dict:
    started = time.monotonic()
    phase = "not_started"
    output = io.StringIO()
    try:
        if version != __version__:
            raise RuntimeError("Adapter versions differ; run setup again and restart Cascadeur")
        request_id = request_id or uuid.uuid4().hex
        if type(request_id) is not str or not __import__("re").fullmatch(r"[0-9a-f]{32}", request_id):
            raise ValueError("request_id must be a 32-character lowercase UUID hex string")
        tools = registry()
        if name not in tools:
            raise ValueError(f"Unknown tool: {name}")
        tool = tools[name]
        tool.validate(arguments)
        if workspace is not None and Path(workspace).resolve() != get_settings()["workspace"].resolve():
            raise ValueError("Client and Cascadeur use different workspaces; run setup and restart Cascadeur")
        if name == "python_execute" and not allow_python:
            raise PermissionError("Python execution is disabled in this client")
        phase = "running"
        transaction_summary = None
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            if tool.read_only:
                result = json_value(tool.handler(**arguments))
            else:
                result, transaction_summary = _execute_write(tool, arguments, request_id)
        phase = "completed"
        return {
            "ok": True,
            "result": result,
            "execution_state": phase,
            "adapter_version": __version__,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "log": output.getvalue()[-16_000:],
            "request_id": request_id,
            **({"transaction": transaction_summary} if transaction_summary else {}),
        }
    except Exception as error:
        return {
            "ok": False,
            "error": {
                "code": getattr(error, "code", type(error).__name__),
                "message": str(error),
                "execution_state": getattr(
                    error, "execution_state", "unknown" if phase == "running" else phase
                ),
                "request_id": request_id,
                **(
                    {"rollback_verified": error.rollback_verified}
                    if hasattr(error, "rollback_verified")
                    else {}
                ),
                **({"details": error.details} if getattr(error, "details", None) is not None else {}),
            },
            "adapter_version": __version__,
            "log": output.getvalue()[-16_000:],
        }


def start_bridge(workspace: str | Path, allow_python: bool = False, port: int = 8765) -> dict:
    """Attach our tools to the installed script runner; no bundled native binaries."""
    from PySide6.QtCore import QCoreApplication, QThread
    from scripts.mcp.script_server import server

    application = QCoreApplication.instance()
    if application is None or QThread.currentThread() != application.thread():
        raise RuntimeError("Start the bridge on Cascadeur's Qt application thread")
    configure(workspace, allow_python)
    server.start(None, host="127.0.0.1", port=port)
    return {"started": server.is_running(), "adapter_version": __version__, "port": port}
