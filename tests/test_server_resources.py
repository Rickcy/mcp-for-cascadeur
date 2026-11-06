import json

import pytest
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_for_cascadeur.runner import RunnerError
from mcp_for_cascadeur.server import create_server
from mcp_for_cascadeur.settings import Config


class SceneConnection:
    def __init__(self):
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": True, "result": {"scene": "native", "frame": len(self.calls)}}


@pytest.mark.asyncio
async def test_scene_resource_reads_fresh_native_context_each_time(tmp_path):
    connection = SceneConnection()
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        listed = await session.list_resources()
        assert "cascadeur://scene" in {str(resource.uri) for resource in listed.resources}
        first = await session.read_resource("cascadeur://scene")
        second = await session.read_resource("cascadeur://scene")
        assert json.loads(first.contents[0].text) == {"scene": "native", "frame": 1}
        assert json.loads(second.contents[0].text) == {"scene": "native", "frame": 2}
    assert connection.calls == [("scene_info", {}), ("scene_info", {})]


@pytest.mark.asyncio
async def test_guide_and_character_prompt_expose_checked_workflow(tmp_path):
    connection = SceneConnection()
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        guide = (await session.read_resource("cascadeur://guide")).contents[0].text
        for term in (
            "recovery_status",
            "request_id",
            "checkpoint_create",
            "character_edit_readiness",
            "character_pose_snapshot",
            "semantic",
            "playback",
        ):
            assert term in guide
        prompts = await session.list_prompts()
        prompt = next(item for item in prompts.prompts if item.name == "animate_character")
        assert prompt.arguments[0].name == "brief"
        assert prompt.arguments[0].required is True
        rendered = await session.get_prompt(
            "animate_character", {"brief": "A two-second wave with curled fingers."}
        )
        text = rendered.messages[0].content.text
        assert "A two-second wave with curled fingers." in text
        assert "cascadeur://scene" in text
        assert "checkpoint" in text
        assert "finger-local" in text
    assert connection.calls == []


@pytest.mark.asyncio
async def test_character_prompt_rejects_missing_or_extra_arguments(tmp_path):
    server = create_server(Config(workspace=tmp_path), SceneConnection())
    async with create_connected_server_and_client_session(server) as session:
        with pytest.raises(McpError):
            await session.get_prompt("animate_character", {})
        with pytest.raises(McpError):
            await session.get_prompt("animate_character", {"brief": "wave", "unexpected": "value"})


@pytest.mark.asyncio
async def test_typed_recovery_metadata_survives_official_mcp_roundtrip(tmp_path):
    request_id = "c" * 32

    class FailedConnection:
        def call(self, name, arguments):
            raise RunnerError(
                "RECOVERY_REQUIRED",
                "Native recovery could not be verified",
                "recovery_required",
                details={"comparison": {"equivalent": False}},
                request_id=request_id,
                rollback_verified=False,
            )

    server = create_server(Config(workspace=tmp_path), FailedConnection())
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("pose_apply", {})
        assert result.isError is True
        error = result.structuredContent["error"]
        assert error["code"] == "RECOVERY_REQUIRED"
        assert error["request_id"] == request_id
        assert error["rollback_verified"] is False
        assert error["execution_state"] == "recovery_required"
        assert error["details"]["comparison"]["equivalent"] is False
