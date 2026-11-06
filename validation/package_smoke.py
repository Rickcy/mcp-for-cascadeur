"""Validate an installed wheel with the official MCP SDK, without Cascadeur.

Run this script with a fresh wheel environment's Python, outside the checkout.
Only a disposable HTTP fixture is contacted; native application calls are absent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import mcp_for_cascadeur


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tools", type=int, default=96)
    arguments = parser.parse_args()
    imported = Path(mcp_for_cascadeur.__file__).resolve()
    if not imported.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            "Use a fresh wheel environment; the package was imported from outside its environment"
        )
    requests = []

    class Fixture(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/run"
            identifier = re.search(r"request_id='([0-9a-f]{32})'", body["code"]).group(1)
            requests.append(identifier)
            if ".dispatch('scene_info', {}" in body["code"]:
                result = {
                    "ok": True,
                    "result": {"name": "Isolated package fixture", "current_frame": 0},
                    "execution_state": "completed",
                    "request_id": identifier,
                }
                reply = {"ok": True, "value": repr(json.dumps(result)), "messages": []}
            else:
                assert ".dispatch('pose_offset', " in body["code"]
                reply = {"ok": False, "error": "Timed out waiting for Cascadeur idle event", "messages": []}
            encoded = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    async def check(environment):
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "mcp_for_cascadeur"], env=environment
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "mcp-for-cascadeur"
                tools = await session.list_tools()
                assert len(tools.tools) == arguments.expected_tools - 1
                names = {tool.name for tool in tools.tools}
                assert "python_execute" not in names
                assert {
                    "recovery_verify",
                    "character_retime",
                    "fbx_export_verified",
                    "subtree_restore",
                    "object_data_read",
                    "object_data_set",
                } <= names
                resources = await session.list_resources()
                assert {str(item.uri) for item in resources.resources} == {
                    "cascadeur://guide",
                    "cascadeur://tools",
                    "cascadeur://scene",
                }
                guide = await session.read_resource("cascadeur://guide")
                assert "pose_apply_sequence" in guide.contents[0].text
                contracts = await session.read_resource("cascadeur://tools")
                assert len(json.loads(contracts.contents[0].text)) == arguments.expected_tools - 1
                scene = await session.read_resource("cascadeur://scene")
                assert json.loads(scene.contents[0].text)["name"] == "Isolated package fixture"
                invalid = await session.call_tool("timeline_set_frame", {"frame": True})
                assert invalid.isError and invalid.structuredContent["error"]["code"] == "INVALID_ARGUMENTS"
                prompts = await session.list_prompts()
                prompt = await session.get_prompt(
                    "animate_character", {"brief": "Isolated package acceptance."}
                )
                assert (
                    len(prompts.prompts) == 1
                    and "Isolated package acceptance." in prompt.messages[0].content.text
                )
                timeout = await session.call_tool(
                    "pose_offset", {"objects": ["fixture"], "frames": [0], "delta": [1, 0, 0]}
                )
                error = timeout.structuredContent["error"]
                assert (
                    timeout.isError
                    and error["code"] == "NATIVE_SCRIPT_FAILED"
                    and error["execution_state"] == "unknown"
                )
                assert error["request_id"] == requests[-1] and "transaction_status" in error["message"]
                assert len(requests) == 2
                return {
                    "version": initialized.serverInfo.version,
                    "enabled_tools": len(tools.tools),
                    "resources": len(resources.resources),
                    "prompts": len(prompts.prompts),
                    "fixture_requests": len(requests),
                    "native_calls": 0,
                    "timeout_request_id_preserved": True,
                    "automatic_retries": 0,
                    "package_import": str(imported),
                }

    fixture = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    worker = threading.Thread(target=fixture.serve_forever, daemon=True)
    worker.start()
    try:
        with tempfile.TemporaryDirectory(prefix="mcp-for-cascadeur-package-") as directory:
            home = Path(directory).resolve()
            assert home.is_relative_to(Path(tempfile.gettempdir()).resolve())
            (home / "config.json").write_text(
                json.dumps(
                    {
                        "endpoint": f"http://127.0.0.1:{fixture.server_port}",
                        "workspace": str(home / "workspace"),
                        "allow_python": False,
                    }
                ),
                encoding="utf-8",
            )
            environment = dict(os.environ, MCP_FOR_CASCADEUR_HOME=str(home))
            environment.pop("PYTHONPATH", None)
            result = asyncio.run(check(environment))
        print(json.dumps(result, indent=2))
    finally:
        fixture.shutdown()
        fixture.server_close()
        worker.join()


if __name__ == "__main__":
    main()
