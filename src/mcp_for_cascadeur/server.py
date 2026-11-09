"""MCP stdio surface; all native execution stays in Cascadeur."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server

from . import __version__
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

Read cascadeur://workflow for the full Smart MCP operating sequence and
cascadeur://capabilities for declared route boundaries and limits. Native tools
are distinct from the optional checkout-based Kimodo pipeline. Kimodo supports
text and captured body-keypose generation; source-video capture uses native
mocap_start. Gemini and a configured local Qwen3-VL-compatible endpoint serve as
visual reviewers, not motion generators. Require timestamped multi-view motion
evidence, native geometry/save/reopen checks, and explicit uncertainty. An equal
review schema does not prove equal model quality. Never claim that a local
reviewer was tested or that an animation was visually improved without evidence.
"""

WORKFLOW = """Smart MCP animation workflow

1. Establish the native scene and scope.
Read cascadeur://scene, app_status, app_capabilities, recovery_status,
characters_list, character_skeleton and tracks_list. Resolve exact identities,
units, FPS, existing frames, native license and build. Inspect properties and
bindings rather than guessing names or invoking arbitrary menu actions.
Use rig_semantics for validated Cascy body roles; a user rig_profile is not that
validation. Use character_edit_readiness before checked character writes.

2. Choose the route that matches the input.
- Existing animation: read/measure it and use a bounded native edit.
- Text: optional integrations/kimodo/pipeline.py run generates new motion through
  the external Kimodo Docker service, then transfers it to a separate Cascy scene.
- Body key poses: the same pipeline captures selected existing native frames,
  converts their skeleton/units/floor/first-root/heading to SOMA, generates the
  interval, reverses the coordinate conversion and checks the returned anchors.
- Video: mocap_start uses Cascadeur's native Mocap model and a workspace MP4 or
  existing reference plane. Kimodo's integration is not a video-mocap endpoint.
- Existing endpoint poses: motion_generation_start invokes Cascadeur's native
  pose-based Motion Generation; it does not accept text prompts.

3. Protect authored work and plan meaningful poses.
Save a new .casc copy and create a bounded checkpoint before a sequence of model
edits. Durable character_pose_snapshot captures one pose, not the whole clip.
Plan pose/timing constraints from motion events: contact, weight acceptance,
passing, push-off, takeoff or landing. Uniform frame spacing is a transport test,
not an artistic key-selection rule. Define the intended root path, root-motion
or in-place result and loop requirement; seamless looping is not automatic.
Body Joints are solved read-only outputs. Author body motion with Point controls;
author validated finger-local rotations separately. Captured Kimodo constraints
cover the body, not finger articulation, facial motion or attached armor physics.

4. Generate, author or run a native operation.
The optional Kimodo route supports 2-120 transferred frames at 30 FPS and 1-4
retained candidates. Calculate candidates sequentially on a 12 GB GPU; release
the idle generator/encoder service before native review or a local visual model.
Do not start heavyweight inference on Cascadeur's main thread. Record the prompt,
seed, checkpoint, input hashes and output paths. Preserve all generated candidates.
Use the pipeline's run.json and stage receipts; there is no registered
kimodo_generate MCP tool in this release.
Use unbaking_start to obtain classic editable intervals from baked motion when
needed. Some approximation displacement is expected: measure it against the
source and chosen tolerance. Use CLAMPED_BEZIER through native interpolation
tools; preserving its enum does not preserve the shape after a changed pose.
Use autophysics_start only with an explicit interval and all_frames/only_keys
choice. Its application changes motion and requires a fresh comparison.
Poll native_job_status until verified completion. Dispatch is not completion.

5. Verify the actual motion before judging or correcting it.
Read solved native poses, connection geometry and authored curve metadata.
Measure root travel, contacts and foot speed; identify which contact mask and
coordinate space produced each diagnostic. Predicted contacts, low toe height,
Joint-center speed and mesh penetration are different observations. None alone
certifies foot planting. Preserve protected channels and animation outside the
declared edit interval. Use curve-preserving Point tools for existing keys and
character_retime for its supported synchronized 2-8-key timing edits.

6. Perform provider-neutral visual review.
Render the resulting Cascy motion from front, side and three-quarter views at
consistent speed and framing. Use native frame rendering with scene updates
between captures, then CPU video encoding. Declare sampling/held-frame cadence;
still images establish pose evidence, not temporal smoothness.
The reviewer can be Gemini or a configured local Qwen3-VL-compatible service.
Use visual_review_status to inspect its public configuration and animation_review
to review a workspace JSON request through the external MCP process. This tool
does not execute a native script; supply verified native checks from the editing
workflow rather than assuming a video proves geometry or persistence.
Both return the same validated candidate IDs, usability, confidence, scores,
observations, limitations and preferred candidate. Record provider/model and
SHA256 of every reviewed medium. Use one comparative request for the candidates;
repeat after a meaningful correction or when a specific unresolved issue needs
new evidence, not after every tool call. Local frame-sequence transport must keep
order and timestamps and disclose lower sampling. Do not silently reduce evidence
to fit VRAM or silently switch providers. Equal API shape is not quality parity:
benchmark a replacement on the same native clips before trusting its rankings.
The visual reviewer proposes findings; it does not modify animation or certify
physical correctness. Missing/stale evidence, failed native/anchor/persistence
checks, insufficient confidence or unsuitable candidates return needs_revision.

7. Refine only a diagnosed problem.
Choose the smallest relevant pose/interval or additional generation constraint.
Protect the accepted root path, unrelated controls, authored curves and source.
After the correction, read native results again and review the changed motion.
Do not edit blindly from a single screenshot or assume a lower numerical proxy
means more natural motion. Automatic candidate selection is separate from an
automatic repair loop; a generic autonomous stylistic repair loop is not provided.

8. Save, reopen, play and export.
Save a new result, reopen it and verify the relevant stored data/anchors. Observe
bounded native playback with playback_start/status/stop and confirm the stop.
Timeline progression does not establish display frame-time stability. Use
fbx_export_status and fbx_export_verified when licensed; verify the receiving
application's skeleton, axes, scale and clip range. Preserve useful native scenes,
candidate artifacts, review evidence and recovery records; remove only owned
temporary files after their workers are proven stopped.

Recovery and credentials
After an uncertain write inspect transaction_status or its stage receipt; never
replay relative edits or toggles. A native job owns its working scene; cancellation
retains its fence until stop/rollback is verified. A restart is not an unlock.
Use native_job_recover or recovery_verify/recovery_restore as appropriate.
Kimodo pipeline resume reuses verified artifacts and native/review stages; it does
not resubmit generation or resume an interrupted diffusion pass.
Keep provider credentials in the application's private local settings/key store
or process environment, never in prompts, source, Docker images, scenes or run
records. Read the current provider configuration and capabilities before use.
"""


def _capabilities(tools):
    return {
        "schema": "smart-cascadeur-capabilities-v1",
        "package_version": __version__,
        "scope": "Declared contracts; query app_capabilities for the running installation.",
        "registered_mcp_tools": sorted(tools),
        "native_job_target": {"version": "2026.2.3", "build": "44356", "rig": "validated Cascy"},
        "routes": {
            "native_scene_editing": {"interface": "MCP", "entry": "character_edit_readiness"},
            "native_video_mocap": {"interface": "MCP", "entry": "mocap_start"},
            "native_pose_inbetweening": {"interface": "MCP", "entry": "motion_generation_start"},
            "native_unbaking": {"interface": "MCP", "entry": "unbaking_start"},
            "native_physics": {"interface": "MCP", "entry": "autophysics_start"},
            "text_or_body_keypose_generation": {
                "interface": "optional checkout CLI",
                "entry": "integrations/kimodo/pipeline.py run",
                "registered_mcp_tool": False,
                "checkpoint": "Kimodo-SOMA-RP-v1.2",
                "finger_articulation": False,
                "automatic_seamless_loop": False,
            },
        },
        "limits": {
            "checked_character_stored_frames": 121,
            "terminal_guard_writable": False,
            "character_retime_synchronized_keys": [2, 8],
            "native_job_stored_frames_including_guard": 512,
            "kimodo_transferred_frames": [2, 120],
            "kimodo_fps": 30,
            "kimodo_retained_candidates": [1, 4],
            "native_playback_seconds": 20,
        },
        "visual_review": {
            "interface": "packaged MCP",
            "tools": ["visual_review_status", "animation_review"],
            "role": "analysis and candidate selection, not motion generation or direct editing",
            "providers": ["gemini", "openai-compatible"],
            "local_example": "Qwen3-VL through a compatible local multimodal server",
            "same_validated_response_contract": True,
            "quality_parity_verified": False,
            "required_evidence": ["timestamped motion", "multiple views", "media SHA256"],
            "acceptance": ["native geometry", "save/reopen", "required anchors", "visual evidence"],
            "failure_outcome": "needs_revision",
        },
        "instruction_resources": ["cascadeur://guide", "cascadeur://workflow", "cascadeur://tools"],
    }


OFFHOST_TOOLS = frozenset({"animation_review", "visual_review_status"})


def tool_definitions(config: Config):
    """Combine native contracts with external visual analysis without importing csc."""
    from .review_service import tools as review_tools

    definitions = registry()
    for definition in review_tools(config):
        if definition.name in definitions:
            raise RuntimeError("Duplicate native and visual-review tool name")
        definitions[definition.name] = definition
    if not config.allow_python:
        definitions.pop("python_execute", None)
    return definitions


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


def create_server(config: Config, client: RunnerClient | None = None) -> Server:
    connection = client or RunnerClient(config.endpoint, config.workspace, allow_python=config.allow_python)
    tools = tool_definitions(config)
    server = Server(
        "mcp-for-cascadeur",
        version=__version__,
        instructions=GUIDE,
        website_url="https://github.com/Rickcy/mcp-for-cascadeur",
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
                    destructiveHint=not definition.read_only and definition.name not in OFFHOST_TOOLS,
                    openWorldHint=definition.name == "animation_review",
                ),
            )
            for definition in tools.values()
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        try:
            if name not in tools:
                raise RunnerError("UNKNOWN_TOOL", f"Tool is unavailable: {name}")
            values = {} if arguments is None else arguments
            if name in OFFHOST_TOOLS:
                try:
                    tools[name].validate(values)
                except ValueError as error:
                    raise RunnerError("INVALID_ARGUMENTS", str(error)) from error
                result = await asyncio.to_thread(tools[name].handler, **values)
                payload = {
                    "ok": True, "result": result, "execution_state": "completed",
                    "executor": "visual_review", "adapter_version": __version__,
                }
                png = None
            else:
                payload = await asyncio.to_thread(_call_with_job_fallback, connection, tools[name], values, config)
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
        except Exception as error:
            payload = {
                "error": {"code": type(error).__name__, "message": str(error), "execution_state": "unknown"}
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
                uri="cascadeur://workflow",
                name="Smart MCP workflow",
                description="Input routes, generation, provider-neutral visual review and acceptance.",
                mimeType="text/plain",
            ),
            types.Resource(
                uri="cascadeur://capabilities",
                name="Capability boundaries",
                description="Declared native and optional integration routes, limits and review contract.",
                mimeType="application/json",
            ),
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
        if str(uri) == "cascadeur://workflow":
            return [ReadResourceContents(content=WORKFLOW, mime_type="text/plain")]
        if str(uri) == "cascadeur://capabilities":
            return [ReadResourceContents(content=json.dumps(_capabilities(tools)), mime_type="application/json")]
        if str(uri) == "cascadeur://tools":
            manifest = [
                {"name": item.name, "read_only": item.read_only, "input_schema": item.input_schema}
                for item in tools.values()
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
            "Read cascadeur://guide, cascadeur://workflow and cascadeur://scene. "
            "Discover native capabilities, recovery state, "
            "the character's actual RigInfo/Point/finger bindings and timeline. Save a new native scene "
            "copy and capture a recovery checkpoint before editing. Plan key poses and timing for this brief:\n\n"
            + brief
            + "\n\n"
            "Check readiness, author body Points first and validated finger-local poses separately. "
            "Preserve authored curves when requested. Read actual solved poses, measure inbetweens, "
            "observe native playback and review timestamped multi-view motion with the configured visual "
            "reviewer. Body/hand PNGs supplement pose evidence. Do not infer temporal quality from stills. "
            "Verify native save/reopen and "
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
