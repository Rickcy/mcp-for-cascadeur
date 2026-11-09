"""Operate the staged Kimodo bridge through Cascadeur's native script runner.

This entry point deliberately uses /run rather than the model-editing
python_execute MCP tool: a scene switch cannot share a transaction belonging to
the previous scene. Bridge-owned scene copies and receipts provide recovery.
Every write still checks the adapter's recovery, job and playback guards on the
application thread. No native operation is retried after an unknown outcome.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

BRIDGE_PATH = Path(__file__).with_name("cascadeur_bridge.py").resolve()
STAGES = {
    "prepared": ("open_target", "target_open"),
    "target_open": ("open_source", "source_open"),
    "source_open": ("import_fbx", "fbx_imported"),
    "fbx_imported": ("retarget_copy", "retarget_copied"),
    "retarget_copied": ("prepare_target_interval", "target_ready"),
    "target_ready": ("retarget_paste", "retargeted"),
    "retargeted": ("save_result", "saved"),
    "saved": ("reopen_prepare", "reopen_ready"),
    "reopen_ready": ("reopen_result", "verified"),
    "verified": ("cleanup", "completed"),
    "points_prepared": ("point_attach", "points_ready:0"),
    "points_applied": ("point_save", "points_saved"),
    "points_saved": ("point_reopen_prepare", "points_reopen_ready"),
    "points_reopen_ready": ("point_reopen", "points_verified"),
    "points_verified": ("point_cleanup", "completed"),
}
OPERATIONS = {stage for stage, _ in STAGES.values()} | {"point_apply"}


class ImportFailure(RuntimeError):
    def __init__(self, message, *, execution_state="not_started", details=None):
        super().__init__(message)
        self.execution_state = execution_state
        self.details = details


def _load_bridge():
    spec = importlib.util.spec_from_file_location("kimodo_cascadeur_bridge", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_points():
    path = Path(__file__).with_name("point_retarget.py")
    spec = importlib.util.spec_from_file_location("kimodo_point_retarget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt_bridge(receipt_path):
    schema = json.loads(Path(receipt_path).read_text(encoding="utf-8")).get("schema")
    if schema == "kimodo-point-session-v1":
        return _load_points()
    if schema == "kimodo-cascadeur-import-v1":
        return _load_bridge()
    raise ImportFailure("Unsupported native import session schema")


def _next_stage(record):
    if record["phase"].startswith("points_ready:"):
        first = int(record["phase"].split(":", 1)[1])
        if first != record["next_frame"]:
            raise ImportFailure("Point session frame cursor differs from its durable phase")
        last = min(first + 20, record["num_frames"])
        return "point_apply", f"points_ready:{last}" if last < record["num_frames"] else "points_applied"
    return STAGES.get(record["phase"], (None, None))


def _native_preflight():
    """Read guards without creating or restoring a core model checkpoint."""
    from mcp_for_cascadeur.host import common, jobs, playback, recovery

    _, _, view, scene = common.current_context(require_scene=False)
    recovery_state = recovery.status()
    blockers = []
    if recovery_state.get("writes_blocked"):
        blockers.append("Adapter recovery blocks writes")
    if recovery.active_transaction() is not None:
        blockers.append("A core model transaction is active")
    if not common.get_settings().get("allow_python"):
        blockers.append("Trusted Python integration is disabled in the installed adapter")
    for guard in (lambda: jobs.require_idle("kimodo_native_import"), playback.require_idle):
        try:
            guard()
        except Exception as error:
            blockers.append(str(error))
    return {
        "ready": not blockers,
        "blockers": blockers,
        "pid": os.getpid(),
        "frame": int(scene.get_current_frame(False)) if scene is not None else None,
        "view_path": view.get_path_name() if view is not None else None,
        "view_name": view.name() if view is not None else None,
        "recovery": recovery_state,
    }


def _witness(state):
    return {key: state[key] for key in ("pid", "frame", "view_path", "view_name")}


def _native_dispatch(
    operation, receipt_path=None, expected_pid=None, idle_witness=None, *, callback_scene=None
):
    """Fixed native entry point; called only by the CLI's /run expression."""
    dispatched = False
    try:
        state = _native_preflight()
        if expected_pid is not None and state["pid"] != expected_pid:
            raise RuntimeError("Cascadeur restarted; inspect the session before continuing")
        if operation == "preflight":
            return {"ok": True, "state": state}
        bridge = _receipt_bridge(receipt_path)
        if operation == "inspect":
            return {"ok": True, "state": state, "result": bridge.inspect(receipt_path)}
        if operation not in OPERATIONS:
            raise ValueError("Unknown bridge stage")
        if not state["ready"]:
            raise RuntimeError("Native preflight blocked the stage: " + "; ".join(state["blockers"]))
        if idle_witness is None or _witness(state) != idle_witness:
            raise RuntimeError("The active scene or playhead changed after the idle check")
        dispatched = True
        function_name = operation.removeprefix("point_") if operation.startswith("point_") else operation
        options = (
            {"callback_scene": callback_scene}
            if function_name in {"reopen", "reopen_result", "cleanup"}
            else {}
        )
        result = getattr(bridge, function_name)(receipt_path, **options)
        return {"ok": True, "state": state, "result": result}
    except Exception as error:
        return {
            "ok": False,
            "error": {
                "message": str(error),
                "type": type(error).__name__,
                "execution_state": "unknown" if dispatched else "not_started",
                "traceback": traceback.format_exc(),
            },
        }


def _runner_client(endpoint, timeout):
    if not 0 < timeout <= 35:
        raise ValueError("Native import requests must use a timeout between 0 and 35 seconds")
    try:
        from mcp_for_cascadeur.runner import RunnerClient
        from mcp_for_cascadeur.settings import load_config
    except ModuleNotFoundError:
        source = Path(__file__).resolve().parents[2] / "src"
        if not (source / "mcp_for_cascadeur" / "runner.py").is_file():
            raise
        sys.path.insert(0, str(source))
        from mcp_for_cascadeur.runner import RunnerClient
        from mcp_for_cascadeur.settings import load_config

    config = load_config()
    return RunnerClient(endpoint or config.endpoint, config.workspace, timeout=timeout), config


def _expression(operation, receipt_path=None, expected_pid=None, idle_witness=None):
    # repr produces Python literals, not shell commands. User data is never eval'd.
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('kimodo_native_import', {str(Path(__file__).resolve())!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"result = module._native_dispatch({operation!r}, {receipt_path!r}, {expected_pid!r}, {idle_witness!r}, callback_scene=native_callback_scene)\n"
    )
    return (
        "__import__('json').dumps("
        f"(lambda namespace: (exec({code!r}, namespace, namespace), namespace['result'])[1])({{'native_callback_scene': scene}}),"
        "allow_nan=False)"
    )


def _raw(client, operation, receipt_path=None, expected_pid=None, idle_witness=None):
    envelope = client._request(
        "/run", {"code": _expression(operation, receipt_path, expected_pid, idle_witness)}
    )
    if envelope.get("ok") is not True:
        raise ImportFailure("Native script runner failed", execution_state="unknown", details=envelope)
    try:
        result = json.loads(ast.literal_eval(envelope["value"]))
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            raise ValueError("Missing native bridge response envelope")
    except (KeyError, ValueError, TypeError, SyntaxError) as error:
        raise ImportFailure("Invalid native bridge response", execution_state="unknown") from error
    if not result["ok"]:
        error = result.get("error", {})
        raise ImportFailure(
            error.get("message", "Native bridge failed"),
            execution_state=error.get("execution_state", "unknown"),
            details={"native_error": error, "messages": envelope.get("messages", [])[-100:]},
        )
    messages = envelope.get("messages", [])
    errors = [item for item in messages if str(item.get("level", "")).casefold() == "error"]
    if errors:
        raise ImportFailure("Cascadeur reported native errors", execution_state="unknown", details=errors)
    result["native_messages"] = messages[-100:]
    return result


def _transport_paths(receipt_path):
    receipt, _ = _receipt_bridge(receipt_path)._read(receipt_path)
    marker = receipt.with_name(".native-transport.json")
    lock = receipt.with_name(".native-import.lock")
    if marker.is_symlink() or lock.is_symlink():
        raise ImportFailure("Native transport paths were replaced by symbolic links")
    return receipt, marker, lock


def _transport_record(marker):
    return json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {"status": "not_submitted"}


def _write_json(path, record):
    _load_bridge()._write(path, record)


def _idle_preflight(client, expected_pid=None):
    health = client.health()
    if health.get("queued", health.get("queued_requests", 0)):
        raise ImportFailure("Other native requests are queued; wait before starting the import stage")
    first = _raw(client, "preflight", expected_pid=expected_pid)["state"]
    if not first["ready"]:
        raise ImportFailure("Native preflight blocked writes", details=first)
    time.sleep(0.2)
    second = _raw(client, "preflight", expected_pid=expected_pid)["state"]
    if not second["ready"] or _witness(first) != _witness(second):
        raise ImportFailure("Cascadeur is not idle; the scene or playhead changed", details=second)
    return second


def step(client, receipt_path, stage=None, expected_pid=None):
    bridge = _receipt_bridge(receipt_path)
    receipt, record = bridge._read(receipt_path)
    _, marker, lock = _transport_paths(receipt)
    transport = _transport_record(marker)
    if transport["status"] in {"submitted", "unknown"}:
        raise ImportFailure(
            "Previous native outcome is unknown; use status --native and inspect before continuing"
        )
    operation, completed = _next_stage(record)
    if operation is None:
        raise ImportFailure(f"No automatic next stage for {record['phase']!r}; inspect the receipt")
    if stage is not None and stage != operation:
        raise ImportFailure(f"The next allowed stage is {operation}, not {stage}")
    if lock.exists():
        raise ImportFailure("Another CLI request is active or interrupted; inspect the native import lock")
    pinned_pid = expected_pid or record.get("native_pid")
    state = _idle_preflight(client, pinned_pid)
    token = uuid.uuid4().hex
    lock_record = {"pid": os.getpid(), "token": token, "stage": operation, "started_ns": time.time_ns()}
    try:
        with lock.open("x", encoding="utf-8") as handle:
            json.dump(lock_record, handle)
    except FileExistsError as error:
        raise ImportFailure("Another native import request acquired the session") from error
    try:
        # Recheck after acquiring the lock; another CLI may have finished meanwhile.
        receipt, current = bridge._read(receipt)
        if current["phase"] != record["phase"]:
            raise ImportFailure("The import phase changed during preflight; inspect before continuing")
        current["native_pid"] = state["pid"]
        bridge._write(receipt, current)
        transport = {
            "status": "submitted",
            "stage": operation,
            "expected_phase": completed,
            "native_pid": state["pid"],
            "submitted_ns": time.time_ns(),
            "token": token,
        }
        _write_json(marker, transport)
        try:
            response = _raw(client, operation, str(receipt), state["pid"], _witness(state))
        except Exception as error:
            execution_state = getattr(error, "execution_state", "unknown")
            transport.update(
                status="not_started" if execution_state == "not_started" else "unknown",
                error=str(error),
                execution_state=execution_state,
                details=getattr(error, "details", None),
            )
            _write_json(marker, transport)
            raise
        transport.update(
            status="completed",
            completed_ns=time.time_ns(),
            native_messages=response.get("native_messages", []),
        )
        _write_json(marker, transport)
        if completed == "completed":
            marker.unlink()
        return {**response["result"], "native_messages": response.get("native_messages", [])}
    finally:
        if lock.is_file() and json.loads(lock.read_text(encoding="utf-8")).get("token") == token:
            lock.unlink()


def status(client, receipt_path, *, native=False, acknowledge_completed=False, expected_pid=None):
    bridge = _receipt_bridge(receipt_path)
    receipt, record = bridge._read(receipt_path)
    _, marker, lock = _transport_paths(receipt)
    transport = _transport_record(marker)
    result = {
        "receipt_path": str(receipt),
        "phase": record["phase"],
        "error": record.get("error"),
        "transport": transport,
        "lock": json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else None,
        "next_stage": _next_stage(record)[0],
        "before_path": str(Path(record["directory"]) / "before.casc"),
        "result_path": str(Path(record["directory"]) / "result.casc"),
    }
    if native or acknowledge_completed:
        response = _raw(client, "inspect", str(receipt), expected_pid or record.get("native_pid"))
        result["native"] = response
    if acknowledge_completed:
        if lock.exists():
            raise ImportFailure("A CLI lock remains; inspect its owning process before removing it")
        actual_phase = result["native"]["result"]["phase"]
        if transport["status"] not in {"unknown", "submitted"} or actual_phase != transport.get(
            "expected_phase"
        ):
            raise ImportFailure("Native receipt does not prove completion of the uncertain stage")
        transport.update(status="acknowledged_completed", inspected_ns=time.time_ns())
        _write_json(marker, transport)
        result["transport"] = transport
        if actual_phase == "completed":
            marker.unlink()
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="Validate generation and create isolated scene copies"
    )
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--fbx", required=True)
    prepare_parser.add_argument("--workspace", required=True)
    prepare_parser.add_argument("--template", required=True, help="Kimodo.casc source template")
    prepare_parser.add_argument(
        "--cascy-template", help="Cascy.casc target template; defaults to configured installation"
    )
    for command in ("step", "status", "complete"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--receipt", required=True)
        subparser.add_argument("--endpoint")
        subparser.add_argument("--timeout", type=float, default=35)
        subparser.add_argument("--expected-pid", type=int)
        if command == "step":
            subparser.add_argument("--stage", choices=sorted(OPERATIONS))
        if command == "status":
            subparser.add_argument("--native", action="store_true")
            subparser.add_argument("--acknowledge-completed", action="store_true")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            _, config = _runner_client(None, 35)
            cascy = arguments.cascy_template
            if cascy is None:
                if config.cascadeur_home is None:
                    raise ValueError("Pass --cascy-template or configure the Cascadeur installation path")
                cascy = config.cascadeur_home / "samples" / "Cascy.casc"
            result = _load_bridge().prepare(
                arguments.workspace, cascy, arguments.template, arguments.manifest, arguments.fbx
            )
        else:
            client, _ = _runner_client(arguments.endpoint, arguments.timeout)
            if arguments.command == "status":
                result = status(
                    client,
                    arguments.receipt,
                    native=arguments.native,
                    acknowledge_completed=arguments.acknowledge_completed,
                    expected_pid=arguments.expected_pid,
                )
            elif arguments.command == "step":
                result = step(client, arguments.receipt, arguments.stage, arguments.expected_pid)
            else:
                while True:
                    result = status(client, arguments.receipt)
                    if result["transport"]["status"] in {"submitted", "unknown"}:
                        raise ImportFailure("Previous native outcome is unknown; inspect status --native")
                    if result["phase"] == "completed":
                        break
                    completed = step(client, arguments.receipt, expected_pid=arguments.expected_pid)
                    print(json.dumps(completed, allow_nan=False), flush=True)
        print(json.dumps({"ok": True, "result": result}, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(error),
                    "execution_state": getattr(error, "execution_state", "not_started"),
                    "details": getattr(error, "details", None),
                    "retry_allowed": False,
                },
                indent=2,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
