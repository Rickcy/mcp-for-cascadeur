"""The MCP surface submits the same asynchronous jobs as the native panel."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_for_cascadeur import cli, motion_service, motion_tools
from mcp_for_cascadeur.server import create_server
from mcp_for_cascadeur.settings import Config


class NoNativeCalls:
    def call(self, *args):
        raise AssertionError("An asynchronous Kimodo submission must not use the native runner")


@pytest.mark.asyncio
async def test_generation_returns_the_broker_job_and_rejects_secret_arguments(tmp_path, monkeypatch):
    requests = []

    def call(method, path, payload=None, **options):
        requests.append((method, path, payload))
        return {"job_id": "a" * 32, "state": "running", "stage": "queued"}

    monkeypatch.setattr(motion_service, "call_service", call)
    server = create_server(Config(workspace=tmp_path), NoNativeCalls())
    async with create_connected_server_and_client_session(server) as session:
        listed = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert listed["kimodo_generate"].annotations.openWorldHint is True
        assert listed["kimodo_generate"].annotations.destructiveHint is False
        result = await session.call_tool("kimodo_generate", {
            "prompt": "Walk forward", "frames": 120, "samples": 3,
            "source_scene": str(tmp_path / "source.casc"), "keyframes": [0, 60, 119],
            "auto_activate": False,
        })
        invalid = await session.call_tool("kimodo_generate", {"prompt": "Walk", "api_key": "never-submit-this"})
    assert result.isError is False
    assert result.structuredContent["executor"] == "smart_motion"
    assert result.structuredContent["result"]["job_id"] == "a" * 32
    assert requests[0][:2] == ("POST", "/jobs")
    assert requests[0][2]["keyframes"] == [0, 60, 119]
    assert len(requests) == 1 and invalid.isError
    assert "never-submit-this" not in json.dumps(invalid.structuredContent)


@pytest.mark.parametrize("arguments", [
    {"prompt": "Walk", "frames": 1}, {"prompt": "Walk", "seed": -1},
    {"prompt": "Walk", "samples": 5}, {"prompt": "Walk", "text_weight": float("nan")},
    {"prompt": "Walk", "keyframes": [0, 0]},
])
def test_invalid_generation_is_rejected_before_service_start(arguments, tmp_path):
    definitions = {tool.name: tool for tool in motion_tools.tools(Config(workspace=tmp_path))}
    with pytest.raises(ValueError):
        definitions["kimodo_generate"].validate(arguments)


def test_review_configuration_cannot_contain_keys(tmp_path):
    definitions = {tool.name: tool for tool in motion_tools.tools(Config(workspace=tmp_path))}
    with pytest.raises(ValueError, match="unknown"):
        definitions["kimodo_configure"].validate({"review": {"provider": "gemini", "api_key": "private"}})


def test_long_clip_and_explicit_custom_target_have_no_fixed_duration_cap(tmp_path):
    definitions = {tool.name: tool for tool in motion_tools.tools(Config(workspace=tmp_path))}
    arguments = {"prompt": "Walk", "frames": 12000, "target_scene": str(tmp_path / "custom.casc"),
                 "character_id": "selected-rig", "rig_mapping": {"pelvis": "custom-root"},
                 "first_frame": 40, "no_render": True, "review_enabled": False,
                 "keyframes": list(range(121))}
    assert definitions["kimodo_generate"].validate(arguments) == arguments


@pytest.mark.asyncio
async def test_broker_preflight_errors_retain_their_recovery_state(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise motion_service.MotionError("Configure the model paths first", code="NOT_CONFIGURED",
                                         execution_state="not_started")

    monkeypatch.setattr(motion_service, "call_service", unavailable)
    async with create_connected_server_and_client_session(create_server(Config(workspace=tmp_path), NoNativeCalls())) as session:
        result = await session.call_tool("kimodo_generate", {"prompt": "Walk"})
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "NOT_CONFIGURED"
    assert result.structuredContent["error"]["execution_state"] == "not_started"


def test_motion_cli_validates_and_submits_a_json_request(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(motion_service, "call_service", lambda method, path, payload, **options:
                        calls.append((method, path, payload)) or {"job_id": "b" * 32, "state": "running"})
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"prompt": "Walk", "generate_only": True, "frames": 240}), encoding="utf-8")
    assert cli.main(["motion", "generate", "--request", str(request), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["job_id"] == "b" * 32
    assert calls == [("POST", "/jobs", {"prompt": "Walk", "generate_only": True, "frames": 240})]
