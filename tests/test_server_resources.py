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
        assert "cascadeur://workflow" in text
        assert "checkpoint" in text
        assert "finger-local" in text
    assert connection.calls == []


@pytest.mark.asyncio
async def test_workflow_and_capability_resources_ship_without_native_connection(tmp_path):
    connection = SceneConnection()
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        listed = {str(resource.uri) for resource in (await session.list_resources()).resources}
        assert {"cascadeur://workflow", "cascadeur://capabilities"} <= listed
        workflow = (await session.read_resource("cascadeur://workflow")).contents[0].text
        capabilities = json.loads((await session.read_resource("cascadeur://capabilities")).contents[0].text)
        contracts = json.loads((await session.read_resource("cascadeur://tools")).contents[0].text)
        assert capabilities["registered_mcp_tools"] == sorted(item["name"] for item in contracts)
        assert capabilities["routes"]["text_or_body_keypose_generation"]["registered_mcp_tool"] is True
        assert capabilities["routes"]["text_or_body_keypose_generation"]["entry"] == "kimodo_generate"
        assert capabilities["visual_review"]["quality_parity_verified"] is False
        assert capabilities["limits"]["kimodo_transferred_frames"] == [2, None]
        assert capabilities["limits"]["smart_motion_panel_visual_review"] is False
        assert "Qwen3-VL" in workflow and "Gemini" in workflow
        assert "needs_revision" in workflow
        assert "not a video-mocap endpoint" in workflow
        assert "not an artistic key-selection rule" in workflow
        assert "not finger articulation" in workflow
    assert connection.calls == []


@pytest.mark.asyncio
async def test_capability_manifest_respects_optional_python_configuration(tmp_path):
    for allowed in (False, True):
        server = create_server(Config(workspace=tmp_path, allow_python=allowed), SceneConnection())
        async with create_connected_server_and_client_session(server) as session:
            capabilities = json.loads((await session.read_resource("cascadeur://capabilities")).contents[0].text)
        assert ("python_execute" in capabilities["registered_mcp_tools"]) is allowed


@pytest.mark.asyncio
async def test_visual_review_tools_execute_outside_the_native_runner(tmp_path, monkeypatch):
    from mcp_for_cascadeur import review_service
    from mcp_for_cascadeur.contracts import Tool

    connection = SceneConnection()
    calls = []

    def review(request_path):
        calls.append(request_path)
        return {"report": {"provider": "fixture"}, "selection": {"state": "needs_revision"}}

    monkeypatch.setattr(review_service, "tools", lambda config: [
        Tool("animation_review", "Fixture review", review,
             {"request_path": {"type": "string"}}, ("request_path",), False),
        Tool("visual_review_status", "Fixture status", lambda: {"provider": "fixture", "key_present": False}),
    ])
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        listed = {item.name: item for item in (await session.list_tools()).tools}
        assert listed["animation_review"].annotations.openWorldHint is True
        assert listed["animation_review"].annotations.destructiveHint is False
        assert listed["visual_review_status"].annotations.readOnlyHint is True
        status = await session.call_tool("visual_review_status", {})
        result = await session.call_tool("animation_review", {"request_path": str(tmp_path / "review.json")})
    assert status.structuredContent["executor"] == "visual_review"
    assert result.isError is False
    assert result.structuredContent["result"]["selection"]["state"] == "needs_revision"
    assert calls == [str(tmp_path / "review.json")]
    assert connection.calls == []


@pytest.mark.asyncio
async def test_visual_review_schema_rejects_keys_before_handler_or_native_call(tmp_path, monkeypatch):
    from mcp_for_cascadeur import review_service
    from mcp_for_cascadeur.contracts import Tool

    connection, calls = SceneConnection(), []
    monkeypatch.setattr(review_service, "tools", lambda config: [
        Tool("animation_review", "Fixture review", lambda **kwargs: calls.append(kwargs),
             {"request_path": {"type": "string"}}, ("request_path",), False),
    ])
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("animation_review", {
            "request_path": str(tmp_path / "review.json"), "api_key": "fixture-secret",
        })
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "INVALID_ARGUMENTS"
    assert "fixture-secret" not in json.dumps(result.structuredContent)
    assert calls == connection.calls == []


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


def _durable_job(workspace, *, status="running", transaction_status="pending", job_id="a" * 32):
    import hashlib

    from mcp_for_cascadeur.host import state

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {**value, "record_sha256": hashlib.sha256(state.canonical_bytes(value)).hexdigest()}
        path.write_bytes(state.canonical_bytes(body))

    root = workspace / ".cascadeur-mcp" / "recovery"
    request_id = ("b" if job_id == "a" * 32 else "e") * 32
    write(
        root / "jobs" / f"{job_id}.json",
        {
            "schema_version": 1,
            "job_id": job_id,
            "request_id": request_id,
            "operation": "motion_generation",
            "status": status,
            "phase": "wait",
            "events": [{"level": "Info", "text": "Native dispatch queued"}],
        },
    )
    write(
        root / "transactions" / f"{request_id}.json",
        {
            "schema_version": 1,
            "request_id": request_id,
            "status": transaction_status,
        },
    )
    return job_id


class UnavailableConnection:
    def __init__(self, code="NOT_CONNECTED", message="Native process unavailable"):
        self.code, self.message, self.calls = code, message, []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        raise RunnerError(self.code, self.message, "unknown", request_id="d" * 32)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["NOT_CONNECTED", "CONNECTION_LOST", "TIMEOUT", "HTTP_ERROR"])
async def test_job_status_falls_back_to_verified_journal_after_transport_failure(tmp_path, code):
    job_id = _durable_job(tmp_path)
    connection = UnavailableConnection(code)
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        response = await session.call_tool("native_job_status", {"job_id": job_id})
    assert response.isError is False
    payload = response.structuredContent
    assert payload["executor"] == "durable_journal"
    assert payload["connected"] is False
    assert payload["result"]["connected"] is False
    assert payload["result"]["execution_state"] == "unknown"
    assert payload["result"]["status"] == "running"
    assert payload["transport_error"]["code"] == code
    assert payload["transport_error"]["request_id"] == "d" * 32
    assert connection.calls == [("native_job_status", {"job_id": job_id})]


@pytest.mark.asyncio
async def test_job_list_offline_preserves_committed_outcome_and_unknown_pending_jobs(tmp_path):
    first = _durable_job(tmp_path, job_id="a" * 32, status="completed", transaction_status="committed")
    second = _durable_job(tmp_path, job_id="c" * 32, status="running", transaction_status="pending")
    server = create_server(Config(workspace=tmp_path), UnavailableConnection())
    async with create_connected_server_and_client_session(server) as session:
        response = await session.call_tool("native_jobs_list", {"limit": 1})
        complete = await session.call_tool("native_jobs_list", {"limit": 2})
    assert response.isError is False
    payload = response.structuredContent
    assert payload["executor"] == "durable_journal"
    assert len(payload["result"]["jobs"]) == 1
    assert payload["result"]["jobs"][0]["job_id"] in {first, second}
    assert payload["result"]["connected"] is False
    by_id = {item["job_id"]: item for item in complete.structuredContent["result"]["jobs"]}
    assert by_id[first]["status"] == "completed"
    assert by_id[second]["status"] == "running"
    assert by_id[second]["execution_state"] == "unknown"


@pytest.mark.asyncio
async def test_status_journal_fallback_handles_vendor_idle_deadline(tmp_path):
    job_id = _durable_job(tmp_path)
    connection = UnavailableConnection("NATIVE_SCRIPT_FAILED", "Timed out waiting for Cascadeur idle event")
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        response = await session.call_tool("native_job_status", {"job_id": job_id})
    assert response.isError is False
    assert response.structuredContent["executor"] == "durable_journal"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["RECOVERY_REQUIRED", "INVALID_ARGUMENTS", "NATIVE_SCRIPT_FAILED"])
async def test_native_job_errors_are_not_replaced_with_journal_results(tmp_path, code):
    job_id = _durable_job(tmp_path)
    server = create_server(Config(workspace=tmp_path), UnavailableConnection(code))
    async with create_connected_server_and_client_session(server) as session:
        response = await session.call_tool("native_job_status", {"job_id": job_id})
    assert response.isError is True
    assert response.structuredContent["error"]["code"] == code


@pytest.mark.asyncio
async def test_offline_job_status_rejects_corrupt_record_instead_of_claiming_success(tmp_path):
    job_id = _durable_job(tmp_path)
    path = tmp_path / ".cascadeur-mcp" / "recovery" / "jobs" / f"{job_id}.json"
    value = json.loads(path.read_text())
    value["status"] = "completed"
    path.write_text(json.dumps(value))
    server = create_server(Config(workspace=tmp_path), UnavailableConnection())
    async with create_connected_server_and_client_session(server) as session:
        response = await session.call_tool("native_job_status", {"job_id": job_id})
    assert response.isError is True
    assert response.structuredContent["error"]["code"] == "INVALID_JOB_RECORD"


@pytest.mark.asyncio
async def test_offline_fallback_revalidates_arguments_and_never_replays_writes(tmp_path):
    connection = UnavailableConnection()
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        invalid = await session.call_tool("native_jobs_list", {"limit": True})
        write = await session.call_tool("native_job_cancel", {"job_id": "a" * 32})
    assert invalid.isError is True
    assert invalid.structuredContent["error"]["code"] == "INVALID_ARGUMENTS"
    assert write.isError is True
    assert write.structuredContent["error"]["code"] == "NOT_CONNECTED"
    assert len(connection.calls) == 2


@pytest.mark.asyncio
async def test_connected_job_status_keeps_native_result(tmp_path):
    _durable_job(tmp_path)
    connection = SceneConnection()
    server = create_server(Config(workspace=tmp_path), connection)
    async with create_connected_server_and_client_session(server) as session:
        response = await session.call_tool("native_job_status", {"job_id": "a" * 32})
    assert response.isError is False
    assert response.structuredContent == {"ok": True, "result": {"scene": "native", "frame": 1}}
