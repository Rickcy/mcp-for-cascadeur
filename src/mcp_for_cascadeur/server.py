"""MCP stdio surface; all native execution stays in Cascadeur."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server

from . import __version__
from .credentials import CredentialError, load_secret
from .host import registry
from .runner import RunnerClient, RunnerError
from .settings import Config, load_config

GUIDE = """Read cascadeur://scene for current native scene context. Work with a saved scene
copy in the configured workspace. Start with app_status, app_capabilities, recovery_status,
objects_list and tracks_list. Use exact IDs for ambiguous names. Inspect actual native
behavior channels and constraints before editing. scene_save creates a native .casc backup;
checkpoint_create captures bounded native recovery state. These differ from pose_snapshot
and durable character_pose_snapshot, which preserve a selected pose rather than an entire clip.
Durable character_pose_restore restores Point targets and validated finger-local rotations
while retaining existing curve metadata and untouched authored keys. Independent linear-FK
Joint subtrees use subtree_pose_read, subtree_snapshot, subtree_pose_apply and subtree_restore;
their whole-clip snapshots preserve current unrelated edits. object_data_read and object_data_set
expose exact owned native data IDs/display names, including data without behavior-property bindings.

Discover RigInfo characters with characters_list and character_skeleton. rig_semantics
validates the complete native Cascy Point profile; arbitrary rig_profile groups do not imply
that validation. Body Joints are driven read-only outputs. Use character Point targets for
body motion and hand_pose_read/hand_pose_apply_sequence for validated finger-local rotations.
Palm Point movement does not curl fingers. Read character_edit_readiness before semantic
editing; it is advisory and allocates no history. Ordinary semantic writes create LINEAR
character keys. Use character_offset_preserving_curves or character_pose_preserving_curves
for supported existing-key corrections that must retain authored key/curve metadata.
character_retime handles native synchronized character keys; tracks_retime has a narrower
independent-transform scope. Checked character writes need an allocated successor frame;
the final stored terminal frame is read-only in the current native edit contract.

motion_sample and motion_mirror create plans without native edits. Apply suitable batches
with pose_apply_sequence, transforms_animate or the checked character tools. motion_measure
reads actual native trajectories. native_mirror invokes rig-aware native mirroring. Native
actions report dispatch, not successful completion. Select real track/frame intervals before
interval physics or mirror actions. Solvers can change neighboring controls: read back poses,
inspect edit-impact reports, sample inbetweens and review body/hand PNGs before claiming quality.

Use mocap_start, motion_generation_start, unbaking_start and autophysics_start for
checked native jobs on supported builds and rigs. Motion Generation fills existing
endpoint poses; it does not accept text prompts. These tools return a job_id before
native completion. Poll native_job_status/native_jobs_list for observed stages and
verified output. Progress can be indeterminate. CLAMPED_BEZIER is supported in the
classic curve/retiming contracts. A running job owns the scene; native_job_cancel
requests cancellation and retains the fence until a safe stop and rollback are
proved. After a process failure, native_job_recover restores the verified source.
Offline status reads the durable journal and reports unknown execution state;
never treat a lost connection or a dispatched native action as completed work.

Eligible model edits persist a request_id and recovery checkpoint before mutation. Inspect
transaction_status after an uncertain response; never blindly retry relative edits or toggles.
False native postconditions trigger checked rollback. recovery_status reports a persistent
failed-recovery fence; reads and saving a NEW quarantine .casc remain available. Restarting
does not unlock a changed quarantine. recovery_verify compares the complete recorded state;
recovery_restore/checkpoint_restore must verify restoration before writes can continue.
Full model recovery excludes opaque asset payloads and arbitrary Python's external side effects.

Use playback_start/status/stop for bounded observed native playback. A timer observes frames,
it never animates by repeated frame-setting. Stop only owned playback and verify the measured
stop before editing. All file paths must be absolute and inside the configured workspace.
Python is opt-in and has Cascadeur's full application privileges. Successful tool execution
does not establish physical plausibility, contact quality or professional animation quality.

Uthana tools call an external cloud service from the MCP process; they do not edit Cascadeur.
Read uthana_account for current account/rate information before paid requests. Submit text
or local video with an explicit cost cap and unique request_id; retain the durable job_id.
Poll uthana_job_status and inspect uncertain submissions before any retry. uthana_download
saves verified provider output inside the configured workspace; importing or retargeting that
motion is a separate native step. An accepted cloud request is not a completed animation.
Configure API keys through the hidden credentials CLI or provider environment variables,
never in tool arguments, scene scripts, prompts, or settings files.
"""


def _verify_file(result: dict, config: Config, wait: float = 8.0) -> bytes | None:
    value = result.get("result")
    if not isinstance(value, dict) or "path" not in value or "verification" not in value:
        return None
    path = Path(value["path"]).resolve()
    if not path.is_relative_to(config.workspace.resolve()):
        raise RunnerError("FILE_SCOPE_ERROR", "The native result points outside the workspace", "completed")
    previous = value.get("previous_stamp")
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            stat = path.stat()
            fresh = stat.st_size > 0 and [stat.st_mtime_ns, stat.st_size] != previous
        except FileNotFoundError:
            fresh = False
        if fresh:
            value.update(verification="fresh_file", bytes=stat.st_size)
            if value.get("mime_type") == "image/png":
                if stat.st_size > 8 * 1024 * 1024:
                    value["image_note"] = "PNG exceeds the inline image limit; inspect the saved file."
                    return None
                data = path.read_bytes()
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise RunnerError("INVALID_PNG", "Rendered output is not a PNG image", "completed")
                return data
            return None
        time.sleep(0.1)
    value["verification"] = "pending"
    value["note"] = "The operation returned before fresh output was observed; inspect before retrying."
    return None


def _call_with_job_fallback(connection, definition, arguments, config):
    """Read durable job observations when the native transport cannot answer.

    Only these two read operations have a local substitute. Native validation
    and recovery failures retain their normal error behavior.
    """
    try:
        return connection.call(definition.name, arguments)
    except RunnerError as error:
        unavailable = error.code in {"NOT_CONNECTED", "CONNECTION_LOST", "TIMEOUT", "HTTP_ERROR"}
        unavailable = unavailable or (
            error.code == "NATIVE_SCRIPT_FAILED"
            and "timed out waiting for cascadeur idle event" in str(error).casefold()
        )
        if definition.name not in {"native_job_status", "native_jobs_list"} or not unavailable:
            raise
        try:
            definition.validate(arguments)
        except ValueError as invalid:
            raise RunnerError("INVALID_ARGUMENTS", str(invalid)) from invalid
        # Import lazily: the external server never needs csc or Qt to inspect a
        # persisted job. The helper verifies managed paths and record checksums.
        from .host import jobs

        try:
            if definition.name == "native_job_status":
                result = jobs.read_durable_status(config.workspace, arguments["job_id"])
            else:
                result = jobs.read_durable_list(config.workspace, arguments.get("limit", 20))
        except Exception as failure:
            raise RunnerError(
                getattr(failure, "code", "JOB_METADATA_UNREADABLE"),
                str(failure),
                getattr(failure, "execution_state", "not_started"),
                details={"transport_error": error.as_dict()},
            ) from failure
        return {
            "ok": True,
            "result": result,
            "execution_state": "completed",
            "adapter_version": __version__,
            "executor": "durable_journal",
            "connected": False,
            "transport_error": error.as_dict(),
            "log": "",
            "messages": [],
        }


def create_server(config: Config, client: RunnerClient | None = None, cloud=None) -> Server:
    from .cloud import CloudError, CloudJobs

    connection = client or RunnerClient(config.endpoint, config.workspace, allow_python=config.allow_python)
    tools = registry()
    if not config.allow_python:
        tools.pop("python_execute", None)
    cloud_jobs = cloud if cloud is not None else CloudJobs(config.workspace, credential=load_secret)
    cloud_tools = cloud_jobs.tools()
    if set(tools) & set(cloud_tools):
        raise ValueError("Cloud and native tool names must be distinct")
    available = {**tools, **cloud_tools}

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield {}
        finally:
            try:
                await cloud_jobs.aclose()
            except Exception:
                logging.getLogger(__name__).warning(
                    "Cloud client cleanup failed; provider details were withheld."
                )

    server = Server(
        "mcp-for-cascadeur",
        version=__version__,
        instructions=GUIDE,
        website_url="https://github.com/Rickcy/mcp-for-cascadeur",
        lifespan=lifespan,
    )

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=definition.name,
                description=definition.description,
                inputSchema=definition.input_schema,
                annotations=types.ToolAnnotations(
                    readOnlyHint=definition.read_only,
                    destructiveHint=not definition.read_only,
                    openWorldHint=definition.name in cloud_tools,
                ),
            )
            for definition in available.values()
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        try:
            if name not in available:
                raise RunnerError("UNKNOWN_TOOL", f"Tool is unavailable: {name}")
            if name in cloud_tools:
                try:
                    cloud_tools[name].validate({} if arguments is None else arguments)
                except ValueError as invalid:
                    raise RunnerError("INVALID_ARGUMENTS", str(invalid)) from invalid
                result = await cloud_jobs.call(name, {} if arguments is None else arguments)
                payload = {
                    "ok": True,
                    "result": result,
                    "execution_state": result.get("execution_state", "completed"),
                    "adapter_version": __version__,
                    "executor": "uthana",
                }
                png = None
            else:
                payload = await asyncio.to_thread(
                    _call_with_job_fallback,
                    connection,
                    tools[name],
                    {} if arguments is None else arguments,
                    config,
                )
                png = await asyncio.to_thread(_verify_file, payload, config)
            content = [
                types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, allow_nan=False))
            ]
            if png is not None:
                content.append(
                    types.ImageContent(
                        type="image", mimeType="image/png", data=base64.b64encode(png).decode("ascii")
                    )
                )
            return types.CallToolResult(content=content, structuredContent=payload, isError=False)
        except RunnerError as error:
            payload = {"error": error.as_dict()}
        except CloudError as error:
            payload = {"error": error.as_dict()}
        except CredentialError as error:
            payload = {
                "error": {"code": error.code, "message": str(error), "execution_state": error.execution_state}
            }
        except Exception as error:
            payload = {
                "error": {
                    "code": "CLOUD_OPERATION_FAILED" if name in cloud_tools else type(error).__name__,
                    "message": "The cloud operation failed; inspect its durable job before retrying."
                    if name in cloud_tools
                    else str(error),
                    "execution_state": "unknown",
                }
            }
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))],
            structuredContent=payload,
            isError=True,
        )

    @server.list_resources()
    async def list_resources():
        return [
            types.Resource(
                uri="cascadeur://guide",
                name="Animation workflow",
                description="How to inspect, edit and verify native animation.",
                mimeType="text/plain",
            ),
            types.Resource(uri="cascadeur://tools", name="Tool contracts", mimeType="application/json"),
            types.Resource(
                uri="cascadeur://scene",
                name="Current native scene",
                description="Fresh active native scene, frame, bounds and selection through the local runner.",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri):
        if str(uri) == "cascadeur://guide":
            return [ReadResourceContents(content=GUIDE, mime_type="text/plain")]
        if str(uri) == "cascadeur://tools":
            manifest = [
                {
                    "name": item.name,
                    "read_only": item.read_only,
                    "input_schema": item.input_schema,
                    "open_world": item.name in cloud_tools,
                }
                for item in available.values()
            ]
            return [ReadResourceContents(content=json.dumps(manifest), mime_type="application/json")]
        if str(uri) == "cascadeur://scene":
            payload = await asyncio.to_thread(connection.call, "scene_info", {})
            return [
                ReadResourceContents(
                    content=json.dumps(payload["result"], allow_nan=False), mime_type="application/json"
                )
            ]
        raise ValueError("Unknown Cascadeur resource")

    @server.list_prompts()
    async def list_prompts():
        return [
            types.Prompt(
                name="animate_character",
                description="Inspect, checkpoint, author and verify native character animation from a brief.",
                arguments=[
                    types.PromptArgument(
                        name="brief",
                        required=True,
                        description="Character, action, duration/FPS, reference and output requirements.",
                    )
                ],
            )
        ]

    @server.get_prompt()
    async def get_prompt(name, arguments):
        if name != "animate_character":
            raise ValueError("Unknown Cascadeur prompt")
        if not isinstance(arguments, dict) or set(arguments) != {"brief"}:
            raise ValueError("animate_character requires exactly the brief argument")
        brief = arguments["brief"]
        if not isinstance(brief, str) or not brief.strip() or len(brief) > 12000:
            raise ValueError("brief must contain 1-12000 characters")
        text = (
            "Read cascadeur://guide and cascadeur://scene. Discover native capabilities, recovery state, "
            "the character's actual RigInfo/Point/finger bindings and timeline. Save a new native scene "
            "copy and capture a recovery checkpoint before editing. Plan key poses and timing for this brief:\n\n"
            + brief
            + "\n\n"
            "Check readiness, author body Points first and validated finger-local poses separately. "
            "Preserve authored curves when requested. Read actual solved poses, measure inbetweens, "
            "observe native playback and review full-body and hand PNGs. Verify native save/reopen and "
            "licensed exports. Report measured results and remaining visual limitations; retain request IDs "
            "and inspect uncertain outcomes before retrying."
        )
        return types.GetPromptResult(
            description="Native character animation with checked recovery and visual verification",
            messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))],
        )

    return server


async def run(config: Config) -> None:
    server = create_server(config)
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main(config: Config | None = None) -> None:
    asyncio.run(run(config or load_config()))
