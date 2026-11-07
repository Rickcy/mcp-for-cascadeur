import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_for_cascadeur.contracts import Tool
from mcp_for_cascadeur.server import create_server
from mcp_for_cascadeur.settings import Config


class NoNativeCalls:
    def call(self, *args, **kwargs):
        pytest.fail("Cloud tools must not enter the native runner")


class FakeCloud:
    def __init__(self, error=None):
        self.calls = []
        self.closed = False
        self.error = error

    def tools(self):
        return {
            "uthana_text_start": Tool(
                "uthana_text_start",
                "External cloud generation",
                self.call,
                {"prompt": {"type": "string", "minLength": 1}, "length": {"type": "number", "minimum": 1}},
                ("prompt", "length"),
                read_only=False,
            )
        }

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if self.error:
            raise self.error
        return {"job_id": "test-job", "status": "submitted"}

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_cloud_tools_have_open_world_contract_and_async_lifecycle(tmp_path):
    cloud = FakeCloud()
    server = create_server(Config(workspace=tmp_path), NoNativeCalls(), cloud)
    async with create_connected_server_and_client_session(server) as session:
        definitions = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert definitions["uthana_text_start"].annotations.openWorldHint is True
        assert definitions["uthana_text_start"].annotations.readOnlyHint is False
        assert definitions["app_status"].annotations.openWorldHint is False
        resource = await session.read_resource("cascadeur://tools")
        manifest = {item["name"]: item for item in json.loads(resource.contents[0].text)}
        assert manifest["uthana_text_start"]["open_world"] is True
        result = await session.call_tool("uthana_text_start", {"prompt": "A wave", "length": 3})
        assert not result.isError
        assert result.structuredContent["executor"] == "uthana"
        assert result.structuredContent["result"]["job_id"] == "test-job"
    assert cloud.calls == [("uthana_text_start", {"prompt": "A wave", "length": 3})]
    assert cloud.closed


@pytest.mark.asyncio
async def test_cloud_schema_is_validated_before_any_provider_call(tmp_path):
    cloud = FakeCloud()
    server = create_server(Config(workspace=tmp_path), NoNativeCalls(), cloud)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("uthana_text_start", {"prompt": "wave", "length": True})
        assert result.isError
        assert result.structuredContent["error"]["code"] == "INVALID_ARGUMENTS"
    assert cloud.calls == []
    assert cloud.closed


@pytest.mark.asyncio
async def test_unexpected_cloud_error_does_not_echo_sdk_secrets(tmp_path):
    cloud = FakeCloud(RuntimeError("Authorization: test-fixture-sensitive-token"))
    server = create_server(Config(workspace=tmp_path), NoNativeCalls(), cloud)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("uthana_text_start", {"prompt": "wave", "length": 3})
        assert result.isError
        assert result.structuredContent["error"]["code"] == "CLOUD_OPERATION_FAILED"
        assert result.structuredContent["error"]["execution_state"] == "unknown"
        assert "test-fixture-sensitive-token" not in result.model_dump_json()
    assert cloud.closed


@pytest.mark.asyncio
async def test_cloud_typed_unknown_submission_preserves_job_correlation(tmp_path):
    from mcp_for_cascadeur.cloud import CloudError

    error = CloudError(
        "SUBMISSION_UNKNOWN",
        "Inspect the durable job before retrying.",
        execution_state="unknown",
        details={"job_id": "managed-job", "request_id": "managed-request"},
    )
    cloud = FakeCloud(error)
    server = create_server(Config(workspace=tmp_path), NoNativeCalls(), cloud)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("uthana_text_start", {"prompt": "wave", "length": 3})
        assert result.isError
        assert result.structuredContent["error"] == error.as_dict()
        assert result.structuredContent["error"]["details"]["job_id"] == "managed-job"


@pytest.mark.asyncio
async def test_real_cloud_offline_list_does_not_load_credentials_or_call_native(tmp_path):
    from mcp_for_cascadeur.cloud import CloudJobs

    def forbidden_credential():
        pytest.fail("A local receipt list must not load credentials or call Uthana")

    cloud = CloudJobs(tmp_path, credential=forbidden_credential)
    server = create_server(Config(workspace=tmp_path), NoNativeCalls(), cloud)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("uthana_jobs_list", {"limit": 1})
        assert not result.isError
        assert result.structuredContent["executor"] == "uthana"
        assert result.structuredContent["result"]["jobs"] == []
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_successful_receipt_read_preserves_unknown_job_execution_state(tmp_path):
    class UnknownCloud(FakeCloud):
        async def call(self, name, arguments):
            return {"job_id": "test-job", "status": "unknown", "execution_state": "unknown"}

    server = create_server(Config(workspace=tmp_path), NoNativeCalls(), UnknownCloud())
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("uthana_text_start", {"prompt": "wave", "length": 3})
        assert not result.isError
        assert result.structuredContent["execution_state"] == "unknown"


@pytest.mark.asyncio
async def test_cloud_cleanup_failure_does_not_echo_sdk_exception(tmp_path, caplog):
    class FailingClose(FakeCloud):
        async def aclose(self):
            raise RuntimeError("test-fixture-sensitive-cleanup-token")

    server = create_server(Config(workspace=tmp_path), NoNativeCalls(), FailingClose())
    async with create_connected_server_and_client_session(server) as session:
        await session.list_tools()
    assert "Cloud client cleanup failed" in caplog.text
    assert "test-fixture-sensitive-cleanup-token" not in caplog.text
