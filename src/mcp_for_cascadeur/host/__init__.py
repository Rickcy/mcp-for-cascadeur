"""Native operations executed by Cascadeur's own main-thread script runner."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
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
        "note": "Python has the same privileges as Cascadeur; no automatic Undo or rollback is implied.",
    }


def registry() -> dict[str, Tool]:
    from . import animation, inspection, profiles, scene

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
            "no automatic Undo. Requires setup --allow-python.",
            python_execute,
            {"code": {"type": "string", "minLength": 1, "maxLength": 200_000}},
            ("code",),
            False,
        ),
        *scene.TOOLS,
        *inspection.TOOLS,
        *animation.TOOLS,
        *profiles.TOOLS,
    ]
    result = {tool.name: tool for tool in definitions}
    if len(result) != len(definitions):
        raise RuntimeError("Duplicate tool names in the native registry")
    return result


def dispatch(
    name: str,
    arguments: dict,
    *,
    workspace: str | None = None,
    allow_python: bool = False,
    version: str = __version__,
) -> dict:
    started = time.monotonic()
    phase = "not_started"
    output = io.StringIO()
    try:
        if version != __version__:
            raise RuntimeError("Adapter versions differ; run setup again and restart Cascadeur")
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
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = tool.handler(**arguments)
        phase = "completed"
        return {
            "ok": True,
            "result": json_value(result),
            "execution_state": phase,
            "adapter_version": __version__,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "log": output.getvalue()[-16_000:],
        }
    except Exception as error:
        return {
            "ok": False,
            "error": {
                "code": type(error).__name__,
                "message": str(error),
                "execution_state": getattr(
                    error, "execution_state", "unknown" if phase == "running" else phase
                ),
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
