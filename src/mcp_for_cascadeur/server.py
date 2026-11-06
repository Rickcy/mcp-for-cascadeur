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

GUIDE = """Work with a saved scene copy in the configured workspace. Start with app_status,
app_capabilities, objects_list and tracks_list. Use IDs for ambiguous names. Inspect native
Transform/Point/Box and finger channels before writing. Capture pose_snapshot for selected
controls and scene_save for a whole native checkpoint; a pose snapshot is not a clip backup.
motion_sample and motion_mirror create editable plans without modifying the scene. Apply
plans with pose_apply_sequence. All frames and targets are validated before a native edit;
solvers can change neighboring controls. Read back poses, sample inbetweens, and review PNG
captures before claiming animation quality. File operations require absolute workspace paths.
An uncertain write result must be inspected before retrying: relative offsets accumulate,
native actions can toggle state, and Python has full application privileges. No automatic
rollback or professional motion-quality guarantee is implied by a successful tool call.
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


def create_server(config: Config, client: RunnerClient | None = None) -> Server:
    connection = client or RunnerClient(config.endpoint, config.workspace, allow_python=config.allow_python)
    tools = registry()
    if not config.allow_python:
        tools.pop("python_execute", None)
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
                    destructiveHint=not definition.read_only,
                    openWorldHint=False,
                ),
            )
            for definition in tools.values()
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        try:
            if name not in tools:
                raise RunnerError("UNKNOWN_TOOL", f"Tool is unavailable: {name}")
            payload = await asyncio.to_thread(connection.call, name, {} if arguments is None else arguments)
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
        ]

    @server.read_resource()
    async def read_resource(uri):
        if str(uri) == "cascadeur://guide":
            return [ReadResourceContents(content=GUIDE, mime_type="text/plain")]
        if str(uri) == "cascadeur://tools":
            manifest = [
                {"name": item.name, "read_only": item.read_only, "input_schema": item.input_schema}
                for item in tools.values()
            ]
            return [ReadResourceContents(content=json.dumps(manifest), mime_type="application/json")]
        raise ValueError("Unknown Cascadeur resource")

    return server


async def run(config: Config) -> None:
    server = create_server(config)
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main(config: Config | None = None) -> None:
    asyncio.run(run(config or load_config()))
