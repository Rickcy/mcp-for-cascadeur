import copy
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import common, recovery, state


def model(value=1.0):
    return {
        "format_version": 1,
        "count": 2,
        "objects": {"object": {"name": "Point", "type": "Point", "behaviours": {}}},
        "channels": {"data:value": {"kind": "data", "mode": "Animation", "values": [value, value]}},
        "tracks": {},
        "auxiliary": {},
        "external": {},
    }


@pytest.fixture
def context(tmp_path, monkeypatch):
    common.configure(tmp_path)
    recovery._in_memory_failure = None
    scene = SimpleNamespace(model=model(), frame=0)
    selection = SimpleNamespace(ids={"object"})
    scene.get_current_frame = lambda: scene.frame
    scene.set_current_frame = lambda frame: setattr(scene, "frame", frame)
    scene.model_viewer = lambda: SimpleNamespace(get_objects=lambda: ["object"])
    scene.selector = lambda: SimpleNamespace(
        selected=lambda: selection, select=lambda items: setattr(selection, "ids", items)
    )
    boundary = SimpleNamespace(first_frame=0, last_frame=1, first_visible_frame=0, last_visible_frame=1)
    view = SimpleNamespace(
        name=lambda: "scene",
        get_path_name=lambda: str(tmp_path / "scene.casc"),
        animation_boundary=lambda: boundary,
    )
    monkeypatch.setattr(state, "capture", lambda scene: copy.deepcopy(scene.model))

    def restore(scene, checkpoint):
        scene.model = copy.deepcopy(checkpoint)
        return {"restored": True}

    monkeypatch.setattr(state, "restore", restore)
    monkeypatch.setattr(common, "current_context", lambda **kwargs: (None, None, view, scene))
    undo = []

    def call_action(name):
        assert name == "Scene.Undo"
        if not undo:
            raise RuntimeError("No recorded Undo state")
        scene.model = copy.deepcopy(undo.pop(0))

    monkeypatch.setitem(
        sys.modules,
        "csc",
        SimpleNamespace(
            app=SimpleNamespace(
                get_application=lambda: SimpleNamespace(
                    get_action_manager=lambda: SimpleNamespace(call_action=call_action)
                )
            )
        ),
    )
    return view, scene, undo


def test_checkpoint_is_durable_before_first_mutation_and_receipt_commits(context):
    view, scene, _ = context
    with recovery.transaction(view, scene, "pose_apply", "request_one") as receipt:
        record = recovery._read(receipt.path)
        assert record["status"] == "pending"
        assert record["checkpoint"] == model()
        assert recovery._pending()["request_id"] == "request_one"
        assert recovery.active_checkpoint(scene) == model()
        scene.model = model(3.0)
        recovery.note_native_commit(scene)
        proof = receipt.commit()
    assert proof["execution_state"] == "completed"
    assert recovery.status_by_id("request_one")["status"] == "committed"
    assert not recovery.status()["writes_blocked"]
    assert recovery.active_transaction() is None


def test_failure_undoes_only_recorded_states_and_allows_unchanged_ui_history(context):
    view, scene, undo = context
    before = model()
    with pytest.raises(recovery.RecoveryError) as caught:
        with recovery.transaction(view, scene, "pose_apply", "undo_request"):
            scene.model = model(5.0)
            recovery.note_native_commit(scene)
            undo.extend([model(5.0), before])
            raise RuntimeError("postcondition failed")
    assert caught.value.rollback_verified
    assert caught.value.execution_state == "rolled_back"
    assert caught.value.details["rollback"]["undo_actions"] == 2
    assert scene.model == before
    assert not recovery.status()["writes_blocked"]


def test_prevalidation_failure_never_undos_unrelated_history(context):
    view, scene, undo = context
    undo.append(model(99.0))
    with pytest.raises(recovery.RecoveryError) as caught:
        with recovery.transaction(view, scene, "pose_apply", "prevalidation"):
            raise ValueError("bad frame")
    assert caught.value.rollback_verified
    assert caught.value.details["rollback"]["undo_actions"] == 0
    assert len(undo) == 1


def test_unrecorded_undo_state_stops_history_walk_and_uses_manual_restore(context):
    view, scene, undo = context
    with pytest.raises(recovery.RecoveryError) as caught:
        with recovery.transaction(view, scene, "pose_apply", "diverged"):
            scene.model = model(5.0)
            recovery.note_native_commit(scene)
            undo.extend([model(99.0), model()])
            raise RuntimeError("fail after native write")
    assert caught.value.rollback_verified
    assert caught.value.details["rollback"]["undo_actions"] == 1
    assert caught.value.details["rollback"]["manual_restore"]
    assert len(undo) == 1


def test_failed_restore_is_persistent_and_changed_quarantine_does_not_unlock(context, monkeypatch):
    view, scene, _ = context

    def failed_restore(*args):
        raise RuntimeError("Injected restore failure")

    monkeypatch.setattr(state, "restore", failed_restore)
    with pytest.raises(recovery.RecoveryError) as caught:
        with recovery.transaction(view, scene, "pose_apply", "failed_recovery"):
            scene.model = model(7.0)
            raise RuntimeError("Injected partial write")
    assert not caught.value.rollback_verified
    recovery_module = importlib.reload(recovery)
    assert recovery_module.status(scene=scene)["writes_blocked"]
    assert recovery_module.status(scene=scene)["checkpoint_matches"] is False
    for operation in ("python_execute", "action_run", "timeline_set_frame", "camera_set", "fbx_export"):
        with pytest.raises(recovery_module.RecoveryError):
            recovery_module.guard(operation)
    with pytest.raises(recovery_module.RecoveryError, match="not been restored"):
        recovery_module.recovery_verify()
    scene.model = model()
    assert recovery_module.recovery_verify()["verified"]
    assert not recovery_module.status()["writes_blocked"]


def test_interrupted_pending_transaction_blocks_after_module_reload(context):
    view, scene, _ = context
    receipt = recovery.Receipt(view, scene, "pose_apply", "interrupted")
    scene.model = model(9.0)
    assert receipt.path.is_file()
    recovered_module = importlib.reload(recovery)
    assert recovered_module.status(scene=scene)["status"] == "pending"
    with pytest.raises(recovered_module.RecoveryError):
        recovered_module.guard("pose_apply")


def test_corrupt_journal_or_checksum_is_fail_closed(context):
    view, scene, _ = context
    receipt = recovery.Receipt(view, scene, "pose_apply", "corrupt")
    data = json.loads(receipt.path.read_text())
    data["checkpoint"]["channels"]["data:value"]["values"][1] = 42.0
    receipt.path.write_text(json.dumps(data))
    assert recovery.status(scene=scene)["writes_blocked"]
    assert recovery.status()["status"] == "invalid_or_unverifiable_record"


def test_failure_to_persist_checkpoint_prevents_mutation(context, monkeypatch):
    view, scene, _ = context

    def no_disk(*args):
        raise OSError("Disk unavailable")

    monkeypatch.setattr(recovery, "_write", no_disk)
    with pytest.raises(recovery.RecoveryError) as caught:
        with recovery.transaction(view, scene, "pose_apply", "no_disk"):
            pytest.fail("Mutation must not start without durable journal")
    assert caught.value.execution_state == "not_started"


def test_replayed_request_id_is_rejected_without_execution(context):
    view, scene, _ = context
    with recovery.transaction(view, scene, "pose_apply", "once") as receipt:
        receipt.commit()
    with pytest.raises(recovery.RecoveryError) as caught:
        with recovery.transaction(view, scene, "pose_apply", "once"):
            pytest.fail("A recorded relative edit must not be replayed")
    assert caught.value.code == "REQUEST_ALREADY_RECORDED"


def test_callback_observation_can_defer_to_mandatory_postcommit_capture(context, monkeypatch):
    view, scene, _ = context
    original_capture = state.capture
    with recovery.transaction(view, scene, "pose_apply", "deferred") as receipt:
        monkeypatch.setattr(
            state, "capture", lambda _: (_ for _ in ()).throw(RuntimeError("deferred metadata"))
        )
        recovery.observe_callback(scene)
        assert receipt.record["observer_errors"]
        monkeypatch.setattr(state, "capture", original_capture)
        receipt.commit()


def test_checkpoint_restore_rejects_external_edits_and_protects_recorded_history(context):
    view, scene, _ = context
    saved = recovery.checkpoint_create("known scene")
    with recovery.transaction(view, scene, "pose_apply", "owned_edit") as receipt:
        scene.model = model(4.0)
        recovery.note_native_commit(scene)
        receipt.commit()
    scene.model = model(8.0)
    with pytest.raises(recovery.RecoveryError) as caught:
        recovery.checkpoint_restore(saved["checkpoint_id"])
    assert caught.value.code == "EXTERNAL_STATE_CHANGED"
    with pytest.raises(recovery.RecoveryError, match="dependent checkpoint"):
        recovery.transaction_release("owned_edit")


def test_history_owned_checkpoint_restores_complete_state(context):
    view, scene, undo = context
    saved = recovery.checkpoint_create()
    with recovery.transaction(view, scene, "pose_apply", "edit") as receipt:
        scene.model = model(3.0)
        recovery.note_native_commit(scene)
        receipt.commit()
    undo.append(model())
    assert recovery.checkpoint_restore(saved["checkpoint_id"])["restored"]
    assert scene.model == model()


def test_verified_postcommit_state_can_be_reused_but_next_mutation_invalidates_it(context, monkeypatch):
    view, scene, _ = context
    with recovery.transaction(view, scene, "pose_apply", "cache") as receipt:
        scene.model = model(4.0)
        recovery.note_native_commit(scene)
        cached = recovery.latest_committed_checkpoint(scene)
        recovery.accept_verified_checkpoint(scene, cached)
        assert receipt.verified_after is cached
        recovery.note_native_attempt(scene)
        assert recovery.latest_committed_checkpoint(scene) is None
        assert receipt.verified_after is None
        scene.model = model(5.0)
        receipt.commit()
    assert recovery._read(receipt.path)["after"] == model(5.0)


def test_fresh_complete_model_must_match_after_view_restoration_before_fence_can_clear(context, monkeypatch):
    view, scene, _ = context
    recovery.Receipt(view, scene, "pose_apply", "view_changes_model")

    def bad_view(*args):
        scene.model = model(55.0)

    monkeypatch.setattr(recovery, "_restore_view", bad_view)
    with pytest.raises(recovery.RecoveryError, match="View restoration changed"):
        recovery.recovery_verify()
    assert recovery.status()["writes_blocked"]


def test_checkpoint_restore_reuses_outer_before_and_final_verified_capture(context, monkeypatch):
    view, scene, undo = context
    saved = recovery.checkpoint_create()
    with recovery.transaction(view, scene, "pose_apply", "edit_for_restore") as receipt:
        scene.model = model(3.0)
        recovery.note_native_commit(scene)
        receipt.commit()
    undo.append(model())
    captures = []
    original = state.capture
    monkeypatch.setattr(state, "capture", lambda scene: captures.append(1) or original(scene))
    monkeypatch.setattr(
        state,
        "restore",
        lambda *args: pytest.fail("Already verified Undo must not recapture through manual restore"),
    )
    with recovery.transaction(view, scene, "checkpoint_restore", "restore") as receipt:
        result = recovery.checkpoint_restore(saved["checkpoint_id"])
        receipt.commit()
    assert result["method"] == "checked_native_undo"
    assert len(captures) == 3  # outer before, post-Undo, mandatory final after view restoration


def test_matching_full_checkpoint_can_verify_a_renamed_native_copy(context):
    view, scene, _ = context
    recovery.Receipt(view, scene, "pose_apply", "renamed_copy")
    view.get_path_name = lambda: "different-quarantine-path.casc"
    view.name = lambda: "quarantine copy"
    status = recovery.status(scene=scene, view=view)
    assert status["checkpoint_matches"]
    assert status["candidate_scene"] != status["recorded_scene"]
    assert recovery.recovery_verify()["verified"]


def test_camera_restore_accepts_native_float32_roundoff_but_keeps_other_view_fields_exact():
    saved = {
        "frame": 12,
        "bounds": {"first_frame": 0, "last_frame": 25},
        "selection": ["point"],
        "camera": {
            "type": 0,
            "mode": 1,
            "position": [107.134346, 192.165405, 0.0],
            "target": [0.0, 1.0, 0.0],
        },
    }
    actual = copy.deepcopy(saved)
    actual["camera"]["position"] = [107.1343307, 192.165466, 0.0]
    assert recovery._view_comparison(saved, actual)["matches"]
    actual["frame"] = 24
    report = recovery._view_comparison(saved, actual)
    assert not report["matches"]
    assert report["differences"] == [{"field": "frame", "expected": 12, "actual": 24}]
    actual["frame"] = 12
    actual["camera"]["target"][0] = 0.002
    report = recovery._view_comparison(saved, actual)
    assert not report["matches"]
    assert report["differences"][0]["field"] == "camera.target"
    assert report["camera_tolerance"] == {"absolute": 0.001, "relative": 1e-6}


def test_restore_does_not_reset_already_equivalent_camera(context):
    view, scene, _ = context
    camera = SimpleNamespace(position=[107.134346, 192.165405, 0.0], target=[0.0, 1.0, 0.0], type=0)
    viewport = SimpleNamespace(
        camera_struct=lambda: camera,
        mode_visualizers=lambda: 1,
        set_camera_struct=lambda _: pytest.fail("No setter is needed for float32-equivalent camera"),
        set_mode_visualizers=lambda _: pytest.fail("Mode already matches"),
    )
    view.active_viewport = lambda: SimpleNamespace(domain_viewport=lambda: viewport)
    saved = recovery._view_state(view, scene)
    camera.position = [107.1343307, 192.165466, 0.0]
    recovery._restore_view(view, scene, saved)


def test_restore_failure_reports_expected_and_actual_view_values(context):
    view, scene, _ = context
    saved = recovery._view_state(view, scene)
    scene.set_current_frame = lambda _: setattr(scene, "frame", 24)
    with pytest.raises(recovery.RecoveryError) as caught:
        recovery._restore_view(view, scene, saved)
    assert caught.value.code == "VIEW_RESTORE_FAILED"
    assert caught.value.details["differences"] == [{"field": "frame", "expected": 0, "actual": 24}]


def test_explicit_older_checkpoint_recovery_preserves_failed_evidence_and_is_not_reported_as_rollback(
    context, monkeypatch
):
    view, scene, _ = context
    saved = recovery.checkpoint_create("trusted original")
    scene.model = model(3.0)
    receipt = recovery.Receipt(view, scene, "checkpoint_restore", "failed_old_state")
    receipt.record.update(
        status="recovery_required",
        execution_state="recovery_required",
        rollback_verified=False,
        failure={"code": "Injected", "message": "original failed rollback"},
        rollback={"verified": False, "old_evidence": True},
        after=model(9.0),
        after_sha256=state.digest(model(9.0)),
    )
    recovery._write(receipt.path, receipt.record)
    original = copy.deepcopy(receipt.record)
    scene.model = model()  # The user deliberately opened/reached the known older state.
    monkeypatch.setattr(
        state,
        "restore",
        lambda *args: pytest.fail("A matching candidate requires proof, not native rewriting"),
    )
    result = recovery.recovery_restore(saved["checkpoint_id"])
    record = recovery._read(receipt.path)
    assert result["status"] == "recovered_to_checkpoint" and result["verified"]
    assert result["rollback_verified"] is False
    assert record["recovery_verified"] and record["rollback_verified"] is False
    assert record["checkpoint"] == original["checkpoint"] == model(3.0)
    assert record["after"] == original["after"] == model(9.0)
    assert record["failure"] == original["failure"] and record["rollback"] == original["rollback"]
    assert record["recovered_after"] == model()
    assert record["recovery_target"]["checkpoint_id"] == saved["checkpoint_id"]
    assert recovery.status_by_id("failed_old_state")["status"] == "recovered_to_checkpoint"
    assert not recovery.status()["writes_blocked"]


@pytest.mark.parametrize("invalid", ["newer", "scene", "topology", "corrupt"])
def test_recovery_target_provenance_is_checked_before_intent_or_native_mutation(
    context, monkeypatch, invalid
):
    view, scene, _ = context
    saved = recovery.checkpoint_create()
    receipt = recovery.Receipt(view, scene, "pose_apply", "scope_failure")
    path = recovery._record_path("checkpoints", saved["checkpoint_id"])
    record = recovery._read(path)
    if invalid == "newer":
        record["created_at"] = "2999-01-01T00:00:00+00:00"
    elif invalid == "scene":
        record["scene"] = {"name": "other", "path": "other.casc"}
    elif invalid == "topology":
        record["checkpoint"]["objects"]["other"] = record["checkpoint"]["objects"].pop("object")
        record["checkpoint_sha256"] = state.digest(record["checkpoint"])
    recovery._write(path, record)
    if invalid == "corrupt":
        path.write_text(path.read_text().replace('"label":""', '"label":"tampered"'))
    monkeypatch.setattr(
        state, "restore", lambda *args: pytest.fail("An invalid target must not cause native mutation")
    )
    with pytest.raises(recovery.RecoveryError):
        recovery.recovery_restore(saved["checkpoint_id"])
    assert "recovery_target" not in recovery._read(receipt.path)
    assert recovery.status()["writes_blocked"]


def test_selected_target_is_durable_before_mutation_and_survives_interrupted_restore(context, monkeypatch):
    view, scene, _ = context
    saved = recovery.checkpoint_create()
    scene.model = model(3.0)
    receipt = recovery.Receipt(view, scene, "pose_apply", "selected_interrupt")
    scene.model = model(9.0)

    def interrupted(scene, target):
        assert recovery._read(receipt.path)["recovery_target"]["checkpoint_id"] == saved["checkpoint_id"]
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(state, "restore", interrupted)
    with pytest.raises(RuntimeError, match="process interruption"):
        recovery.recovery_restore(saved["checkpoint_id"])
    resumed = importlib.reload(recovery)
    with pytest.raises(resumed.RecoveryError):
        resumed.recovery_verify()
    scene.model = model()
    result = resumed.recovery_verify()
    assert result["status"] == "recovered_to_checkpoint"
    assert not result["rollback_verified"]


def test_recovered_transition_is_part_of_owned_checkpoint_history(context):
    view, scene, undo = context
    saved = recovery.checkpoint_create()
    with recovery.transaction(view, scene, "pose_apply", "first_edit") as receipt:
        scene.model = model(3.0)
        recovery.note_native_commit(scene)
        receipt.commit()
    failed = recovery.Receipt(view, scene, "checkpoint_restore", "failed_restore")
    scene.model = model()
    recovery.recovery_restore(saved["checkpoint_id"])
    with recovery.transaction(view, scene, "pose_apply", "later_edit") as receipt:
        scene.model = model(5.0)
        recovery.note_native_commit(scene)
        receipt.commit()
    undo.append(model())
    assert recovery.checkpoint_restore(saved["checkpoint_id"])["restored"]
    recovery.checkpoint_release(saved["checkpoint_id"])
    assert recovery.transaction_release(failed.record["request_id"])["released"]


def test_owned_undo_is_reversed_by_redo_if_later_verification_fails(context, monkeypatch):
    view, scene, _ = context
    scene.model = model(3.0)
    redo = []

    def redo_action():
        redo.append(1)
        scene.model = model(3.0)

    monkeypatch.setattr(recovery, "_redo", redo_action)
    monkeypatch.setattr(
        recovery, "_undo", lambda: pytest.fail("Undoing an owned Undo would traverse unrelated history")
    )
    with pytest.raises(recovery.RecoveryError) as caught:
        with recovery.transaction(view, scene, "checkpoint_restore", "inverse_undo"):
            recovery.note_native_attempt(scene)
            scene.model = model()
            recovery.note_native_commit(scene, checkpoint=copy.deepcopy(scene.model), kind="undo")
            raise RuntimeError("later view verification failed")
    assert caught.value.rollback_verified
    assert redo == [1] and scene.model == model(3.0)
