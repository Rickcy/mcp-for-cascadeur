"""Exercise real MCP tools on a disposable copy of an installed sample."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_for_cascadeur.settings import load_config


async def check_sample(sample: str, render: bool) -> dict:
    config = load_config()
    if config.cascadeur_home is None:
        raise RuntimeError("Run mcp-for-cascadeur setup first")
    source = config.cascadeur_home / "samples" / f"{sample}.casc"
    output = config.workspace / "validation" / f"{sample.lower()}-{time.time_ns()}"
    output.mkdir(parents=True)
    input_path = output / "input.casc"
    shutil.copy2(source, input_path)
    report = {"sample": sample, "output": str(output), "checks": []}
    position_tolerance = 0.002 if sample == "Cascy" else 0.001
    parameters = StdioServerParameters(command=sys.executable, args=["-m", "mcp_for_cascadeur"])
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            initialized = await session.initialize()
            report["server"] = initialized.serverInfo.model_dump()

            async def call(name: str, **arguments):
                response = await session.call_tool(name, arguments)
                payload = response.structuredContent
                if response.isError:
                    raise RuntimeError(f"{name}: {payload}")
                report["checks"].append(name)
                print(name, flush=True)
                return payload["result"]

            await call("app_status")
            await call("scene_open", path=str(input_path))
            objects = await call("objects_list", object_type="Point", limit=128)
            controls = [record["id"] for record in objects["objects"]]
            if not controls:
                raise RuntimeError("The sample contains no Point controls")
            pose = await call("pose_read", objects=controls, frame=0)
            snapshot = await call(
                "pose_snapshot", objects=controls, frame=0, path=str(output / "initial-pose.json")
            )
            keyframes = []
            for frame, shift in ((0, 0.0), (12, 15.0), (24, 0.0)):
                keyframes.append(
                    {
                        "frame": frame,
                        "poses": [
                            {
                                "object": item["object"],
                                "position": [
                                    item["position"][0] + shift,
                                    item["position"][1],
                                    item["position"][2],
                                ],
                            }
                            for item in pose["poses"]
                        ],
                    }
                )
            plan = await call("motion_sample", keyframes=keyframes, step=4, easing="smoothstep")
            applied = await call(
                "pose_apply_sequence",
                keyframes=plan["keyframes"],
                extend_timeline=True,
                verify=True,
                position_tolerance=position_tolerance,
            )
            report["pose_verification"] = applied["verification"]
            if not applied["verification"]["matches"]:
                raise RuntimeError("Native pose readback did not match the requested targets")
            restored = await call(
                "pose_restore",
                snapshot_id=snapshot["snapshot_id"],
                frame=12,
                position_tolerance=position_tolerance,
            )
            if not restored["verification"]["matches"]:
                raise RuntimeError("Snapshot restore did not match its captured targets")
            await call(
                "pose_apply_sequence",
                keyframes=plan["keyframes"],
                verify=True,
                position_tolerance=position_tolerance,
            )
            await call("pose_read_sequence", objects=controls, frames=[0, 6, 12, 18, 24])
            await call("timeline_set_range", first=0, last=24)
            if render:
                await call("timeline_set_frame", frame=12)
                await call("camera_frame", objects=controls, distance_scale=4)
                image = await call("viewport_capture", path=str(output / "motion.png"), width=960, height=720)
                report["image"] = image
            saved = await call("scene_save", path=str(output / "motion.casc"))
            report["saved"] = saved
            profile_name = "smoke-" + str(time.time_ns())
            await call("rig_profile_create", name=profile_name, roles={"body": controls})
            await call("rig_pose_read", profile=profile_name, roles=["body"], frame=12)
            await call("scene_open", path=str(output / "motion.casc"))
            reopened = await call("pose_read_sequence", objects=controls, frames=[0, 12, 24])
            report["reopened_frames"] = [entry["frame"] for entry in reopened["keyframes"]]
            expected_by_frame = {
                entry["frame"]: {item["object"]: item["position"] for item in entry["poses"]}
                for entry in keyframes
            }
            saved_error = max(
                math.dist(item["position"], expected_by_frame[entry["frame"]][item["object"]])
                for entry in reopened["keyframes"]
                for item in entry["poses"]
            )
            report["reopened_max_position_error"] = saved_error
            if saved_error > position_tolerance:
                raise RuntimeError("Saved/reopened pose values differ from requested positions")
            await call("rig_profile_read", name=profile_name)
            report["pose_snapshot"] = snapshot["path"]
    destination = output / "report.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("Cube", "Cascy"), default="Cube")
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(asyncio.run(check_sample(arguments.sample, not arguments.no_render)), indent=2))


if __name__ == "__main__":
    main()
