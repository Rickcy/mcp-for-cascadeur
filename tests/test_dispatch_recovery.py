"""Cross-route fencing and typed recovery errors at the public dispatch boundary."""

import json
import os
import re
import sys
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import mcp_for_cascadeur.host as host
from mcp_for_cascadeur.contracts import Tool
from mcp_for_cascadeur.host import jobs, playback, recovery
from mcp_for_cascadeur.runner import RunnerClient, RunnerError

REQUEST_ID = "1" * 32
WRITE_ROUTES = sorted(name for name, tool in host.registry().items() if not tool.read_only)


@pytest.fixture
def isolated_dispatch(monkeypatch, tmp_path):
    view, scene = object(), object()
    monkeypatch.setattr(host, "current_context", lambda require_scene=False: (None, None, view, scene))
    monkeypatch.setattr(playback, "require_idle", Mock())
    monkeypatch.setattr(recovery, "_active", ContextVar("test_dispatch_receipt", default=None))
    monkeypatch.setattr(recovery, "_root", lambda: tmp_path / "recovery")
    return view, scene


@pytest.fixture
def durable_native_job(isolated_dispatch, monkeypatch):
    def create(operation):
        record = {
            "schema_version": 1,
            "job_id": REQUEST_ID,
            "request_id": REQUEST_ID,
            "operation": operation,
            "status": "running",
            "phase": "wait",
            "host_pid": os.getpid(),
        }
        recovery._write(jobs._path(REQUEST_ID), record)
        recovery._write(
            recovery._record_path("transactions", REQUEST_ID),
            {"schema_version": 1, "request_id": REQUEST_ID, "status": "pending"},
        )
        monkeypatch.setattr(jobs, "_session", SimpleNamespace(record=record))
        return record

    return create


@pytest.mark.parametrize(
    "name,operation",
    [
        ("mocap_start", "mocap"),
        ("motion_generation_start", "motion_generation"),
        ("unbaking_start", "unbake"),
        ("autophysics_start", "autophysics"),
    ],
)
def test_public_native_start_retry_returns_active_job_before_write_guards(
    name, operation, durable_native_job, monkeypatch
):
    durable_native_job(operation)
    with pytest.raises(jobs.JobError, match="owns the scene"):
        jobs.require_idle(name)
    fence = Mock(side_effect=recovery.RecoveryError("RECOVERY_REQUIRED", "Active job owns the receipt"))
    idle = Mock(wraps=jobs.require_idle)
    context = Mock(side_effect=AssertionError("A retry only needs the durable journal"))
    handler = Mock(side_effect=AssertionError("The native action must not be replayed"))
    monkeypatch.setattr(recovery, "guard", fence)
    monkeypatch.setattr(jobs, "require_idle", idle)
    monkeypatch.setattr(host, "current_context", context)
    monkeypatch.setattr(host, "_call_handler", handler)

    response = host.dispatch(name, {"first": 0, "last": 30}, request_id=REQUEST_ID)

    assert response["ok"] is True
    assert response["result"]["job_id"] == REQUEST_ID
    assert response["result"]["operation"] == operation
    assert response["result"]["status"] == "running"
    assert response["result"]["transaction_status"] == "pending"
    assert jobs._read(REQUEST_ID)["status"] == "running"
    fence.assert_not_called()
    idle.assert_not_called()
    context.assert_not_called()
    handler.assert_not_called()
    playback.require_idle.assert_not_called()


def test_public_native_start_rejects_request_id_reused_for_another_operation(durable_native_job, monkeypatch):
    durable_native_job("motion_generation")
    fence = Mock(side_effect=AssertionError("Identity reuse must be rejected before scene guards"))
    handler = Mock()
    monkeypatch.setattr(recovery, "guard", fence)
    monkeypatch.setattr(host, "_call_handler", handler)

    response = host.dispatch("unbaking_start", {"first": 0, "last": 30}, request_id=REQUEST_ID)

    assert response["ok"] is False
    assert response["error"]["code"] == "JOB_REQUEST_REUSED"
    assert response["error"]["request_id"] == REQUEST_ID
    assert response["error"]["execution_state"] == "not_started"
    assert response["error"]["details"]["operation"] == "motion_generation"
    assert jobs._read(REQUEST_ID)["operation"] == "motion_generation"
    fence.assert_not_called()
    handler.assert_not_called()


def test_new_native_start_keeps_global_fence_when_another_request_owns_the_job(
    isolated_dispatch, durable_native_job, monkeypatch
):
    durable_native_job("motion_generation")
    fence = Mock(side_effect=recovery.RecoveryError("RECOVERY_REQUIRED", "Active job owns the receipt"))
    handler = Mock()
    monkeypatch.setattr(recovery, "guard", fence)
    monkeypatch.setattr(host, "_call_handler", handler)
    request_id = "2" * 32

    response = host.dispatch("motion_generation_start", {"first": 0, "last": 30}, request_id=request_id)

    assert response["ok"] is False
    assert response["error"]["code"] == "RECOVERY_REQUIRED"
    fence.assert_called_once_with("motion_generation_start", *isolated_dispatch)
    assert not jobs._path(request_id).exists()
    handler.assert_not_called()


@pytest.mark.parametrize("name", WRITE_ROUTES)
def test_every_declared_write_route_reaches_the_global_fence_before_handler(
    name, isolated_dispatch, monkeypatch
):
    handler = Mock(return_value={})
    definition = Tool(name, "Route probe", handler, read_only=False)
    monkeypatch.setattr(host, "registry", lambda: {name: definition})
    denied = Mock(side_effect=recovery.RecoveryError("RECOVERY_REQUIRED", "Checkpoint mismatch"))
    monkeypatch.setattr(recovery, "guard", denied)
    transaction = Mock()
    monkeypatch.setattr(recovery, "transaction", transaction)
    response = host.dispatch(name, {}, request_id=REQUEST_ID, allow_python=True)
    assert response["ok"] is False
    assert response["error"]["code"] == "RECOVERY_REQUIRED"
    assert response["error"]["request_id"] == REQUEST_ID
    denied.assert_called_once_with(name, *isolated_dispatch)
    handler.assert_not_called()
    transaction.assert_not_called()


@pytest.mark.parametrize(
    "name", ["python_execute", "action_run", "timeline_set_frame", "fbx_export", "camera_set"]
)
def test_real_fence_blocks_representative_native_and_nontransactional_routes(
    name, isolated_dispatch, monkeypatch
):
    handler = Mock(return_value={})
    monkeypatch.setattr(host, "registry", lambda: {name: Tool(name, "Probe", handler, read_only=False)})
    monkeypatch.setattr(
        recovery, "status", lambda **_: {"writes_blocked": True, "status": "recovery_required"}
    )
    response = host.dispatch(name, {}, request_id=REQUEST_ID, allow_python=True)
    assert response["ok"] is False
    assert response["error"]["code"] == "RECOVERY_REQUIRED"
    handler.assert_not_called()


def test_readonly_diagnostics_still_work_with_quarantined_scene(monkeypatch):
    handler = Mock(return_value={"state": "readable"})
    monkeypatch.setattr(host, "registry", lambda: {"read_probe": Tool("read_probe", "Probe", handler)})
    fence = Mock(side_effect=AssertionError("Readonly diagnostics must not be fenced"))
    monkeypatch.setattr(recovery, "guard", fence)
    response = host.dispatch("read_probe", {}, request_id=REQUEST_ID)
    assert response["ok"] is True
    assert response["result"] == {"state": "readable"}
    fence.assert_not_called()


@pytest.mark.parametrize("exists,overwrite", [(True, False), (True, True), (False, True)])
def test_quarantine_save_refuses_existing_or_overwrite_paths(
    isolated_dispatch, monkeypatch, tmp_path, exists, overwrite
):
    target = tmp_path / "quarantine.casc"
    if exists:
        target.write_bytes(b"existing scene")
    handler = Mock()
    definition = Tool(
        "scene_save",
        "Probe",
        handler,
        {"path": {"type": "string"}, "overwrite": {"type": "boolean"}},
        ("path",),
        False,
    )
    monkeypatch.setattr(host, "registry", lambda: {"scene_save": definition})
    monkeypatch.setattr(recovery, "status", lambda **_: {"writes_blocked": True})
    result = host.dispatch("scene_save", {"path": str(target), "overwrite": overwrite}, request_id=REQUEST_ID)
    assert result["ok"] is False
    assert result["error"]["code"] == "RECOVERY_REQUIRED"
    handler.assert_not_called()
    if exists:
        assert target.read_bytes() == b"existing scene"


def test_quarantine_save_allows_only_new_file(isolated_dispatch, monkeypatch, tmp_path):
    target = tmp_path / "new-quarantine.casc"

    def save(path, overwrite=False):
        with Path(path).open("xb") as stream:
            stream.write(b"quarantine copy")
        return {"path": path}

    definition = Tool(
        "scene_save",
        "Probe",
        save,
        {"path": {"type": "string"}, "overwrite": {"type": "boolean"}},
        ("path",),
        False,
    )
    monkeypatch.setattr(host, "registry", lambda: {"scene_save": definition})
    monkeypatch.setattr(recovery, "status", lambda **_: {"writes_blocked": True})
    transaction = Mock(side_effect=AssertionError("Quarantine copy must not edit native model history"))
    monkeypatch.setattr(recovery, "transaction", transaction)
    response = host.dispatch("scene_save", {"path": str(target)}, request_id=REQUEST_ID)
    assert response["ok"] is True
    assert target.read_bytes() == b"quarantine copy"
    transaction.assert_not_called()


def test_durable_character_pose_output_is_not_mistaken_for_native_edit(isolated_dispatch, monkeypatch):
    handler = Mock(return_value={"path": "pose.json"})
    monkeypatch.setattr(
        host,
        "registry",
        lambda: {
            "character_pose_snapshot": Tool("character_pose_snapshot", "Probe", handler, read_only=False)
        },
    )
    monkeypatch.setattr(recovery, "guard", Mock())
    transaction = Mock(side_effect=AssertionError("Saving pose JSON must not allocate a native receipt"))
    monkeypatch.setattr(recovery, "transaction", transaction)
    result = host.dispatch("character_pose_snapshot", {}, request_id=REQUEST_ID)
    assert result["ok"] is True
    transaction.assert_not_called()


@pytest.fixture
def fake_receipts(monkeypatch):
    model = {"value": "before"}
    receipts = []

    class Receipt:
        def __init__(self, view, scene, operation, request_id):
            self.committed = False
            self.before = model["value"]
            self.request_id = request_id
            self.commit_calls = 0
            self.rollback_calls = 0
            receipts.append(self)

        def commit(self, result):
            self.commit_calls += 1
            self.committed = True
            return {"request_id": self.request_id, "status": "committed"}

        def rollback(self, error):
            self.rollback_calls += 1
            self.failure = error
            model["value"] = self.before
            raise recovery.RecoveryError(
                "TRANSACTION_ROLLED_BACK",
                str(error),
                execution_state="rolled_back",
                rollback_verified=True,
                details={"request_id": self.request_id},
            ) from error

    monkeypatch.setattr(recovery, "Receipt", Receipt)
    monkeypatch.setattr(recovery, "guard", Mock())
    return model, receipts


@pytest.mark.parametrize(
    "result",
    [
        {"verification": {"matches": False, "max_position_error": 1}},
        {"key_mapping_verified": False},
        {"channel_values_verified": False},
    ],
)
def test_false_native_postcondition_rolls_back_instead_of_reporting_success(
    isolated_dispatch, fake_receipts, monkeypatch, result
):
    model, receipts = fake_receipts

    def edit():
        model["value"] = "changed"
        return result

    monkeypatch.setattr(
        host, "registry", lambda: {"pose_apply": Tool("pose_apply", "Probe", edit, read_only=False)}
    )
    response = host.dispatch("pose_apply", {}, request_id=REQUEST_ID)
    assert response["ok"] is False
    assert response["error"]["execution_state"] == "rolled_back"
    assert response["error"]["rollback_verified"] is True
    assert response["error"]["request_id"] == REQUEST_ID
    assert model["value"] == "before"
    assert receipts[0].rollback_calls == 1
    assert receipts[0].commit_calls == 0
    assert recovery.active_transaction() is None


def test_typed_rollback_error_and_request_id_survive_runner_json_rpc(
    isolated_dispatch, fake_receipts, monkeypatch
):
    model, _ = fake_receipts

    def edit():
        model["value"] = "changed"
        return {"verification": {"matches": False}}

    monkeypatch.setattr(
        host, "registry", lambda: {"pose_apply": Tool("pose_apply", "Probe", edit, read_only=False)}
    )
    captured = {}
    client = RunnerClient()

    def native_envelope(route, payload):
        assert route == "/run"
        identifier = re.search(r"request_id='([0-9a-f]{32})'", payload["code"]).group(1)
        captured["request_id"] = identifier
        response = host.dispatch("pose_apply", {}, request_id=identifier)
        return {"ok": True, "value": repr(json.dumps(response)), "messages": []}

    monkeypatch.setattr(client, "_request", native_envelope)
    with pytest.raises(RunnerError) as caught:
        client.call("pose_apply", {})
    error = caught.value.as_dict()
    assert error["code"] == "TRANSACTION_ROLLED_BACK"
    assert error["execution_state"] == "rolled_back"
    assert error["rollback_verified"] is True
    assert error["request_id"] == captured["request_id"]
    assert error["details"]["request_id"] == captured["request_id"]
    assert model["value"] == "before"


@pytest.mark.parametrize("case", ["invalid_result", "native_event"])
def test_unknown_submitted_outcomes_retain_request_id(monkeypatch, case):
    monkeypatch.setattr(host, "registry", lambda: {"probe": Tool("probe", "Probe", lambda: {})})
    client = RunnerClient()
    submitted = {}

    def envelope(route, payload):
        submitted["request_id"] = re.search(r"request_id='([0-9a-f]{32})'", payload["code"]).group(1)
        if case == "invalid_result":
            return {"ok": True, "value": "not a Python string literal", "messages": []}
        return {
            "ok": True,
            "value": repr(json.dumps({"ok": True, "result": {}})),
            "messages": [{"level": "Error", "text": "Native failure after execution"}],
        }

    monkeypatch.setattr(client, "_request", envelope)
    with pytest.raises(RunnerError) as caught:
        client.call("probe", {})
    assert caught.value.execution_state == "unknown"
    assert caught.value.request_id == submitted["request_id"]


def test_native_error_message_rolls_back_before_receipt_commit_and_unsubscribes(
    isolated_dispatch, fake_receipts, monkeypatch
):
    model, receipts = fake_receipts
    callbacks, removed = {}, []

    def subscribe(callback):
        callbacks[7] = callback
        return 7

    def unsubscribe(identifier):
        removed.append(identifier)
        del callbacks[identifier]

    manager = SimpleNamespace(subscribe=subscribe, unsubscribe=unsubscribe)
    monkeypatch.setitem(sys.modules, "csc", SimpleNamespace(MessageLevel=SimpleNamespace(Error=3)))
    monkeypatch.setitem(sys.modules, "events", SimpleNamespace(event_message_manager=manager))

    def edit():
        model["value"] = "changed"
        callbacks[7](isolated_dispatch[1], "Native solver rejected its update", 3, "Solver error")
        return {"verification": {"matches": True}}

    monkeypatch.setattr(
        host, "registry", lambda: {"pose_apply": Tool("pose_apply", "Probe", edit, read_only=False)}
    )
    result = host.dispatch("pose_apply", {}, request_id=REQUEST_ID)
    assert result["ok"] is False
    assert result["error"]["code"] == "TRANSACTION_ROLLED_BACK"
    assert result["error"]["execution_state"] == "rolled_back"
    assert result["error"]["rollback_verified"] is True
    assert result["error"]["request_id"] == REQUEST_ID
    assert receipts[0].commit_calls == 0
    assert receipts[0].rollback_calls == 1
    assert receipts[0].failure.code == "NATIVE_EVENT_ERROR"
    assert receipts[0].failure.details["messages"] == [
        {"level": "Error", "text": "Native solver rejected its update"}
    ]
    assert model["value"] == "before"
    assert removed == [7]
    assert callbacks == {}
