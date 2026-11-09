"""Exercise one native job on a fresh scene copy and keep its acceptance evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from mcp_for_cascadeur.host.jobs import read_durable_status
from mcp_for_cascadeur.runner import RunnerClient, RunnerError
from mcp_for_cascadeur.settings import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("motion_generation", "mocap", "unbaking", "autophysics"))
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=30)
    parser.add_argument(
        "--tolerance", type=float, default=0.01, help="Protected Point readback tolerance in scene units."
    )
    args = parser.parse_args()
    config = load_config()
    client = RunnerClient(config.endpoint, config.workspace, allow_python=config.allow_python)
    directory = config.workspace / "validation" / f"job-{args.operation}-{time.time_ns()}"
    directory.mkdir(parents=True)
    source = directory / "source.casc"
    shutil.copy2(args.scene, source)
    # Samples can contain relative texture references.
    for name in ("textures", "Textures"):
        textures = args.scene.parent / name
        if textures.is_dir() and not (directory / name).exists():
            shutil.copytree(textures, directory / name)
    report = {"operation": args.operation, "directory": str(directory), "events": []}

    def call(name, arguments):
        try:
            response = client.call(name, arguments)
        except RunnerError as error:
            unavailable = error.code in {"NOT_CONNECTED", "CONNECTION_LOST", "TIMEOUT", "HTTP_ERROR"}
            unavailable |= error.code == "NATIVE_SCRIPT_FAILED" and "idle event" in str(error).lower()
            if name != "native_job_status" or not unavailable:
                raise
            report.setdefault("transport_events", []).append({"code": error.code, "at": time.time()})
            return read_durable_status(config.workspace, arguments["job_id"])
        if not response.get("ok"):
            raise RuntimeError(response)
        return response["result"]

    call("scene_open", {"path": str(source)})
    arguments = {"first": args.first, "last": args.last, "output_path": str(directory / "result.casc")}
    if args.operation == "mocap":
        if args.video is None:
            parser.error("mocap requires --video")
        arguments["video_path"] = str(args.video.resolve())
    if args.operation == "unbaking":
        arguments["position_tolerance"] = args.tolerance
    started = time.monotonic()
    status = call(args.operation + "_start", arguments)
    identifier = status["job_id"]
    previous = None
    while True:
        report["result"] = status
        (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        observed = (status["status"], status["phase"], status.get("progress"))
        if observed != previous:
            print(
                json.dumps(
                    {
                        "job_id": identifier,
                        "status": status["status"],
                        "phase": status["phase"],
                        "progress": status.get("progress"),
                        "elapsed": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
            report["events"].append(
                {"elapsed": time.monotonic() - started, "status": status["status"], "phase": status["phase"]}
            )
            previous = observed
        if status["status"] in {"completed", "failed", "cancelled", "recovered", "recovery_required"}:
            print(
                json.dumps(
                    {
                        "report": str(directory / "report.json"),
                        "job_id": identifier,
                        "status": status["status"],
                        "error": status.get("error", {}).get("code"),
                        "output_path": status.get("result", {}).get("path"),
                    }
                ),
                flush=True,
            )
            return 0 if status["status"] == "completed" else 1
        if time.monotonic() - started > 1000:
            raise RuntimeError(f"Acceptance polling stopped; inspect native_job_status({identifier!r})")
        time.sleep(1)
        status = call("native_job_status", {"job_id": identifier})


if __name__ == "__main__":
    raise SystemExit(main())
