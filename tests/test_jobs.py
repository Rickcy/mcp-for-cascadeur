import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import common, jobs, recovery, state


@pytest.fixture
def session(tmp_path, monkeypatch):
    common.configure(tmp_path)
    monkeypatch.setattr(jobs, "_session", None)
    monkeypatch.setattr(recovery, "_in_memory_failure", None)
    snapshot = {
        "format_version": 1,
        "count": 2,
        "objects": {},
        "channels": {},
        "tracks": {},
        "auxiliary": {},
        "external": {},
    }
    scene = SimpleNamespace(model=copy.deepcopy(snapshot), frame=0)
    scene.get_current_frame = lambda: scene.frame
    scene.set_current_frame = lambda value: setattr(scene, "frame", value)
    scene.model_viewer = lambda: SimpleNamespace(get_objects=lambda: [])
    scene.selector = lambda: SimpleNamespace(
        selected=lambda: SimpleNamespace(ids=set()), select=lambda value: None
    )
    current_path = [str(tmp_path / "original.casc")]

    def save(path):
        Path(path).write_text(json.dumps(scene.model))
        current_path[0] = path

    view = SimpleNamespace(
        save=save,
        get_path_name=lambda: current_path[0],
        name=lambda: Path(current_path[0]).name,
        animation_boundary=lambda: SimpleNamespace(
            first_frame=0, last_frame=1, first_visible_frame=0, last_visible_frame=1
        ),
    )
    view.save(current_path[0])
    monkeypatch.setattr(common, "current_context", lambda **kwargs: (None, None, view, scene))
    monkeypatch.setattr(state, "capture", lambda native: copy.deepcopy(native.model))
    monkeypatch.setattr(
        jobs.native_operations,
        "prepare",
        lambda operation, arguments: {
            "operation": operation,
            "arguments": arguments,
            "phase": "scope",
            "scene_identity": {"path": current_path[0], "name": view.name()},
        },
    )
    subscribe = jobs._subscribe
    monkeypatch.setattr(jobs, "_subscribe", lambda session: None)
    monkeypatch.setattr(jobs, "_start_timer", lambda session: None)
    identifier = "a" * 32
    with jobs.request_context(identifier):
        result = jobs.motion_generation_start(0, 1)
    monkeypatch.setattr(jobs, "_subscribe", subscribe)
    assert result["status"] == "running"
    return jobs._session


def test_start_reserves_source_copy_without_creating_final_output(session):
    record = session.record
    assert not Path(record["output_path"]).exists()
    assert jobs._sha(Path(record["source_copy"]["path"])) == record["source_copy"]["sha256"]
    assert recovery._pending()["request_id"] == record["job_id"]
    with pytest.raises(jobs.JobError, match="owns the scene"):
        jobs.require_idle("transforms_animate")


def test_repeated_request_returns_original_job(session):
    with jobs.request_context(session.record["job_id"]):
        # A retried request must be recognized before the ordinary active-job gate.
        result = jobs._start("motion_generation", {"first": 0, "last": 1})
    assert result["job_id"] == session.record["job_id"]


def test_tick_persists_intent_before_driver(session, monkeypatch):
    def step(record):
        stored = jobs._read(record["job_id"])
        assert stored["step_inflight"] is True
        assert recovery.active_transaction() is session.receipt
        return {"status": "running", "phase": "wait", "driver": {"phase": "wait"}}

    monkeypatch.setattr(jobs.native_operations, "step", step)
    jobs._tick(session)
    assert jobs._read(session.record["job_id"])["step_inflight"] is False
    assert recovery.active_transaction() is None


def test_unproven_cancel_keeps_fence(session, monkeypatch):
    session.record["cancel_requested"] = True
    monkeypatch.setattr(
        jobs.native_operations,
        "cancel",
        lambda record: {"status": "awaiting_input", "phase": "cancelling", "safe_to_rollback": False},
    )
    jobs._tick(session)
    assert recovery._pending() is not None
    assert session.record["status"] == "awaiting_input"


def test_cancel_cannot_claim_stop_without_proof(session, monkeypatch):
    session.record["cancel_requested"] = True
    monkeypatch.setattr(
        jobs.native_operations, "cancel", lambda record: {"status": "cancelled", "phase": "cancelled"}
    )
    jobs._tick(session)
    assert session.record["status"] == "recovery_required"
    assert session.record["error"]["code"] == "NATIVE_STOP_UNVERIFIED"
    assert recovery._pending() is not None


def test_offline_status_verifies_checksum_and_reports_disconnect(session):
    result = jobs.read_durable_status(common._settings["workspace"], session.record["job_id"])
    assert result["connected"] is False
    assert result["execution_state"] == "unknown"
    path = jobs._path(session.record["job_id"])
    record = json.loads(path.read_text())
    record["status"] = "completed"
    path.write_text(json.dumps(record))
    with pytest.raises(jobs.JobError, match="verified"):
        jobs.read_durable_status(common._settings["workspace"], session.record["job_id"])


def test_committed_receipt_is_authoritative_after_job_write_failure(session):
    session.receipt.record["status"] = "committed"
    recovery._write(session.receipt.path, session.receipt.record)
    jobs._fail(session, OSError("job write failed"))
    assert session.record["status"] == "completed"
    assert session.scene is None
    result = jobs.read_durable_status(common._settings["workspace"], session.record["job_id"])
    assert result["status"] == "completed"


def test_foreign_process_status_requires_recovery(session):
    session.record["host_pid"] = os.getpid() + 1
    jobs._write(session.record)
    result = jobs.native_job_status(session.record["job_id"])
    assert result["status"] == "recovery_required"
    assert recovery._pending() is not None


def test_offline_list_validates_limit(session):
    workspace = common._settings["workspace"]
    assert len(jobs.read_durable_list(workspace)["jobs"]) == 1
    with pytest.raises(ValueError):
        jobs.read_durable_list(workspace, True)


def completed_driver(monkeypatch):
    from mcp_for_cascadeur.host import job_verification

    monkeypatch.setattr(job_verification, "protected_report", lambda *args: {"matches": True})
    monkeypatch.setattr(
        jobs.native_operations,
        "step",
        lambda record: {"status": "succeeded", "phase": "complete", "safe_to_rollback": True, "result": {}},
    )


def test_silent_final_save_failure_never_commits_stale_source(session, monkeypatch):
    from mcp_for_cascadeur.host import job_verification

    monkeypatch.setattr(job_verification, "protected_report", lambda *args: {"matches": True})
    session.view.save = lambda path: None
    with pytest.raises(jobs.JobError, match="fresh regular"):
        jobs._complete(session, {})
    assert recovery._read(session.receipt.path)["status"] == "pending"
    assert Path(session.record["source_copy"]["path"]).stat().st_size > 0
    assert Path(session.record["output_path"]).stat().st_size == 0


def test_marker_clear_failure_keeps_committed_result(session, monkeypatch):
    completed_driver(monkeypatch)

    def fail(request_id):
        raise OSError("marker clear failed")

    monkeypatch.setattr(recovery, "_clear_marker", fail)
    jobs._tick(session)
    assert session.record["status"] == "completed"
    assert session.receipt.committed
    assert (
        jobs.read_durable_status(common._settings["workspace"], session.record["job_id"])["status"]
        == "completed"
    )
    assert not recovery._pending()


def test_final_job_write_failure_reconciles_committed_receipt(session, monkeypatch):
    completed_driver(monkeypatch)
    original_write = jobs._write

    def fail_final(record):
        if record["status"] == "completed":
            raise OSError("job storage full")
        original_write(record)

    monkeypatch.setattr(jobs, "_write", fail_final)
    jobs._tick(session)
    assert session.record["status"] == "completed"
    assert session.scene is None
    assert (
        jobs.read_durable_status(common._settings["workspace"], session.record["job_id"])["status"]
        == "completed"
    )


def test_recovery_refreshes_authoritative_in_memory_receipt(session, monkeypatch):
    def recover(path):
        record = recovery._read(session.receipt.path)
        record["status"] = "recovered_to_checkpoint"
        recovery._write(session.receipt.path, record)
        recovery._clear_marker(record["request_id"])
        return {"verified": True}

    monkeypatch.setattr(recovery, "recovery_open", recover)
    jobs._restore(session, cancelled=True)
    assert session.receipt.record["status"] == "recovered_to_checkpoint"
    assert session.record["status"] == "cancelled"
    assert session.scene is None


def test_native_error_survives_event_ring_eviction(session, monkeypatch):
    callbacks = []
    monkeypatch.setitem(sys.modules, "csc", SimpleNamespace(MessageLevel=SimpleNamespace(Error=2)))
    monkeypatch.setitem(
        sys.modules,
        "events",
        SimpleNamespace(
            event_message_manager=SimpleNamespace(
                subscribe=lambda callback: callbacks.append(callback) or "subscription"
            )
        ),
    )
    jobs._subscribe(session)
    callbacks[0](session.scene, "native failed", 2, "")
    for _ in range(100):
        callbacks[0](session.scene, "information", 1, "")
    assert not any(event["level"] == "Error" for event in session.record["events"])
    assert session.record["native_error_count"] == 1
    with pytest.raises(jobs.JobError, match="reported an error"):
        jobs._complete(session, {})
    assert recovery._read(session.receipt.path)["status"] == "pending"


def test_released_terminal_receipt_does_not_break_offline_status_or_list(session):
    session.receipt.record["status"] = "committed"
    recovery._write(session.receipt.path, session.receipt.record)
    session.record["status"] = "completed"
    jobs._write(session.record)
    recovery.transaction_release(session.record["request_id"])
    workspace = common._settings["workspace"]
    assert jobs.read_durable_status(workspace, session.record["job_id"])["status"] == "completed"
    assert len(jobs.read_durable_list(workspace)["jobs"]) == 1


def test_missing_unfinished_receipt_is_an_explicit_recovery_error(session):
    session.receipt.path.unlink()
    with pytest.raises(jobs.JobError) as caught:
        jobs.read_durable_status(common._settings["workspace"], session.record["job_id"])
    assert caught.value.code == "JOB_RECOVERY_MISSING"


def test_native_error_is_reported_promptly_without_releasing_unproven_worker(session, monkeypatch):
    session.record["native_error"] = {"level": "Error", "text": "worker failed"}
    monkeypatch.setattr(jobs.native_operations, "step", lambda record: {"status": "running", "phase": "wait"})
    jobs._tick(session)
    assert session.record["status"] == "recovery_required"
    assert session.record["error"]["code"] == "NATIVE_REPORTED_ERROR"
    assert recovery._pending() is not None


def test_recovery_can_keep_observing_a_cancelled_worker_after_native_error(session, monkeypatch):
    session.record.update(native_error={"level": "Error"}, cancel_requested=True)
    monkeypatch.setattr(
        jobs.native_operations, "cancel", lambda record: {"status": "cancelling", "phase": "wait"}
    )
    jobs._tick(session)
    assert session.record["status"] == "cancelling"
    assert recovery._pending() is not None


def test_cancel_reports_recovering_until_the_checkpoint_is_verified(session, monkeypatch):
    session.record["cancel_requested"] = True
    monkeypatch.setattr(
        jobs.native_operations,
        "cancel",
        lambda record: {"status": "cancelled", "phase": "cancelled", "safe_to_rollback": True},
    )

    def restore(current, **kwargs):
        assert current is session
        status = jobs.native_job_status(session.record["job_id"])
        assert status["status"] == "recovering"
        assert status["poll_after_ms"] == 500
        assert recovery._pending() is not None

    monkeypatch.setattr(jobs, "_restore", restore)
    jobs._tick(session)
    assert session.record["status"] == "recovering"


def test_old_cancelled_driver_status_is_not_a_verified_rollback(session):
    session.record.update(status="cancelled", phase="cancelled")
    result = jobs._public(session.record, session.receipt.record)
    assert result["status"] == "recovering"
    assert result["poll_after_ms"] == 500


def test_failed_source_restore_is_not_replayed_automatically(session, monkeypatch):
    attempts = []

    def failed(path):
        attempts.append(path)
        raise RuntimeError("source load failed")

    monkeypatch.setattr(recovery, "recovery_open", failed)
    monkeypatch.setattr(
        jobs.native_operations,
        "cancel",
        lambda record: {"status": "cancelled", "phase": "cancelled", "safe_to_rollback": True},
    )
    session.record["cancel_requested"] = True
    jobs._tick(session)
    assert len(attempts) == 1
    assert session.record["status"] == "recovery_required"
    assert recovery._pending() is not None
