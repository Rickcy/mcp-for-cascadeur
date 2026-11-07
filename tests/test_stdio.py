import os
import sys

import pytest
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_for_cascadeur.cloud import CLOUD_TOOLS
from mcp_for_cascadeur.host import registry


def test_all_published_tool_schemas_are_valid():
    definitions = registry()
    assert len(definitions) >= 35
    assert not set(definitions) & set(CLOUD_TOOLS)
    for definition in [*definitions.values(), *CLOUD_TOOLS.values()]:
        Draft202012Validator.check_schema(definition.input_schema)


@pytest.mark.asyncio
async def test_official_sdk_stdio_discovery_resources_and_validation(tmp_path):
    environment = dict(os.environ, MCP_FOR_CASCADEUR_HOME=str(tmp_path))
    parameters = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_for_cascadeur"], env=environment
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "mcp-for-cascadeur"
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"pose_apply_sequence", "scene_new", "app_status", "motion_sample"} <= names
            assert "python_execute" not in names
            assert {"uthana_account", "uthana_text_start", "uthana_jobs_list"} <= names
            external = next(tool for tool in tools.tools if tool.name == "uthana_account")
            assert external.annotations.openWorldHint is True
            resources = await session.list_resources()
            assert "cascadeur://guide" in {str(resource.uri) for resource in resources.resources}
            guide = await session.read_resource("cascadeur://guide")
            assert "pose_apply_sequence" in guide.contents[0].text
            invalid = await session.call_tool("timeline_set_frame", {"frame": True})
            assert invalid.isError
            assert invalid.structuredContent["error"]["code"] == "INVALID_ARGUMENTS"
