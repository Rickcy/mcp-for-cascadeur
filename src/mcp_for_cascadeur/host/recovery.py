"""Durable transactions and verified recovery for the bounded native model.

A pending marker is fsynced before user mutation starts. Unknown or corrupt
records block writes across process restarts. Clearing a fence requires a full
comparison with its recorded checkpoint; no force-unlock operation exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import Tool
from . import common, state

MAX_CHECKPOINTS = 16
MAX_TRANSACTIONS = 32
MAX_RECORD_BYTES = 2 * state.MAX_BYTES + 1024 * 1024
_active = ContextVar("cascadeur_model_transaction", default=None)
_in_memory_failure = None
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")


class RecoveryError(RuntimeError):
    def __init__(
        self, code, message, *, execution_state="recovery_required", rollback_verified=False, details=None
    ):
        super().__init__(message)
        self.code = code
        self.execution_state = execution_state
        self.rollback_verified = rollback_verified
        self.details = details


def _root():
    workspace = Path(common.get_settings()["workspace"]).resolve()
    target = workspace / ".cascadeur-mcp" / "recovery"
    if not target.resolve().is_relative_to(workspace):
        raise RecoveryError("RECOVERY_STORAGE_UNSAFE", "Recovery storage points outside the workspace")
    return target


def _record_path(kind, identifier):
    if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
        raise RecoveryError(
            "INVALID_RECOVERY_ID",
            "Use an identifier returned by recovery tools",
            execution_state="not_started",
        )
    return _root() / kind / f"{identifier}.json"


def _sync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write(path, value):
    body = dict(value)
    body["record_sha256"] = hashlib.sha256(state.canonical_bytes(value)).hexdigest()
    raw = state.canonical_bytes(body)
    if len(raw) > MAX_RECORD_BYTES:
        raise RecoveryError(
            "RECOVERY_STORAGE_LIMIT",
            "The recovery record exceeds its durable byte limit",
            execution_state="not_started",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.resolve().is_relative_to(_root().resolve()):
        raise RecoveryError(
            "RECOVERY_STORAGE_UNSAFE", "Recovery records must remain inside their managed directory"
        )
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path):
    if path.is_symlink() or not path.resolve().is_relative_to(_root().resolve()):
        raise RecoveryError("RECOVERY_STORAGE_UNSAFE", "Unsafe recovery record path")
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_RECORD_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(raw) > MAX_RECORD_BYTES:
        raise RecoveryError("INVALID_RECOVERY_RECORD", "Recovery record exceeds the byte limit")
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Record is not a JSON object")
        checksum = value.pop("record_sha256")
        if (
            type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or checksum != hashlib.sha256(state.canonical_bytes(value)).hexdigest()
        ):
            raise ValueError("Recovery checksum/version mismatch")
        if "checkpoint" in value:
            state.validate(value["checkpoint"])
            if value.get("checkpoint_sha256") != state.digest(value["checkpoint"]):
                raise ValueError("Model checkpoint checksum mismatch")
        if "after" in value:
            state.validate(value["after"])
            if value.get("after_sha256") != state.digest(value["after"]):
                raise ValueError("Committed model checksum mismatch")
        if "recovered_after" in value:
            state.validate(value["recovered_after"])
            if value.get("recovered_after_sha256") != state.digest(value["recovered_after"]):
                raise ValueError("Recovered model checksum mismatch")
        if path.parent.name == "transactions":
            if value.get("request_id") != path.stem or value.get("status") not in {
                "pending",
                "committed",
                "rolled_back",
                "recovery_required",
                "recovered_to_checkpoint",
            }:
                raise ValueError("Transaction identity/status mismatch")
        if path.parent.name == "checkpoints" and value.get("checkpoint_id") != path.stem:
            raise ValueError("Checkpoint identity mismatch")
        return value
    except (ValueError, KeyError, TypeError, state.StateError) as error:
        raise RecoveryError(
            "INVALID_RECOVERY_RECORD", f"Recovery record could not be verified: {error}"
        ) from error


def _pending():
    marker = _read(_root() / "active.json")
    if marker is None:
        return None
    record = _read(_record_path("transactions", marker.get("request_id")))
    if record is None or record.get("request_id") != marker.get("request_id"):
        raise RecoveryError("INVALID_RECOVERY_RECORD", "Active transaction record is missing or differs")
    return record if record.get("status") in {"pending", "recovery_required"} else None


def _clear_marker(request_id):
    marker = _read(_root() / "active.json")
    if marker is not None:
        if marker.get("request_id") != request_id:
            raise RecoveryError(
                "RECOVERY_RECORD_CHANGED", "The active recovery record changed during verification"
            )
        (_root() / "active.json").unlink()
        _sync_directory(_root())


def _scene_identity(view):
    return {"path": view.get_path_name(), "name": view.name()}


def _view_state(view, scene):
    bounds = view.animation_boundary()
    result = {
        "frame": scene.get_current_frame(),
        "bounds": {
            key: int(getattr(bounds, key))
            for key in ("first_frame", "last_frame", "first_visible_frame", "last_visible_frame")
        },
        "selection": sorted(state._identifier(item) for item in scene.selector().selected().ids),
    }
    viewport = view.active_viewport() if hasattr(view, "active_viewport") else None
    if viewport is None:
        result["camera"] = None
    else:
        domain = viewport.domain_viewport()
        camera = domain.camera_struct()
        result["camera"] = {
            "position": state.encode(camera.position),
            "target": state.encode(camera.target),
            "type": int(camera.type),
            "mode": int(domain.mode_visualizers()),
        }
    return result


def _camera_value_equal(expected, actual):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            return False
        return all(
            expected[key] == actual[key]
            if key in {"__native__", "dtype", "shape"}
            else _camera_value_equal(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_camera_value_equal(left, right) for left, right in zip(expected, actual))
        )
    if type(expected) in (float, int) and type(actual) in (float, int):
        return math.isclose(expected, actual, abs_tol=0.001, rel_tol=1e-6)
    return type(expected) is type(actual) and expected == actual


def _view_comparison(expected, actual):
    differences = []
    for key in ("frame", "bounds", "selection"):
        if expected[key] != actual[key]:
            differences.append({"field": key, "expected": expected[key], "actual": actual[key]})
    left, right = expected.get("camera"), actual.get("camera")
    if left is None or right is None:
        if left != right:
            differences.append({"field": "camera", "expected": left, "actual": right})
    else:
        for key in ("type", "mode", "position", "target"):
            equal = (
                _camera_value_equal(left[key], right[key])
                if key in {"position", "target"}
                else left[key] == right[key]
            )
            if not equal:
                differences.append({"field": f"camera.{key}", "expected": left[key], "actual": right[key]})
    return {
        "matches": not differences,
        "differences": differences,
        "camera_tolerance": {"absolute": 0.001, "relative": 1e-6},
        "other_view_state": "exact",
    }


def _restore_view(view, scene, saved):
    boundary = view.animation_boundary()
    for key, value in saved["bounds"].items():
        setattr(boundary, key, value)
    scene.set_current_frame(saved["frame"])
    available = {state._identifier(item): item for item in scene.model_viewer().get_objects()}
    if sorted(state._identifier(item) for item in scene.selector().selected().ids) != saved["selection"]:
        if any(item not in available for item in saved["selection"]):
            raise RecoveryError("VIEW_RESTORE_FAILED", "A saved selection identity no longer exists")
        scene.selector().select({available[item] for item in saved["selection"]})
    current_camera = _view_state(view, scene).get("camera")
    if (
        saved.get("camera") is not None
        and not _view_comparison({**saved, "camera": current_camera}, saved)["matches"]
    ):
        import csc

        active = view.active_viewport()
        if active is None:
            raise RecoveryError("VIEW_RESTORE_FAILED", "The recorded viewport is no longer available")
        viewport = active.domain_viewport()
        camera = viewport.camera_struct()
        camera.position = state.decode(saved["camera"]["position"])
        camera.target = state.decode(saved["camera"]["target"])
        camera.type = csc.view.CameraType(saved["camera"]["type"])
        viewport.set_camera_struct(camera)
        viewport.set_mode_visualizers(csc.view.ViewportMode(saved["camera"]["mode"]))
    comparison = _view_comparison(saved, _view_state(view, scene))
    if not comparison["matches"]:
        raise RecoveryError(
            "VIEW_RESTORE_FAILED",
            "Timeline/playhead/selection/camera restoration could not be verified",
            details=comparison,
        )


def active_transaction():
    return _active.get()


def active_checkpoint(scene=None):
    receipt = active_transaction()
    return (
        receipt.checkpoint
        if receipt is not None and (scene is None or receipt.scene is scene or receipt.scene == scene)
        else None
    )


def latest_committed_checkpoint(scene=None):
    receipt = active_transaction()
    if receipt is None or (scene is not None and receipt.scene is not scene and receipt.scene != scene):
        return None
    return receipt.latest_committed


def note_native_attempt(scene=None):
    receipt = active_transaction()
    if receipt is not None:
        if scene is not None and receipt.scene is not scene and receipt.scene != scene:
            raise RecoveryError(
                "TRANSACTION_SCENE_CHANGED",
                "The modification targets another native scene",
                execution_state="not_started",
            )
        receipt.latest_committed = None
        receipt.verified_after = None


def accept_verified_checkpoint(scene, checkpoint, *, fresh=False):
    receipt = active_transaction()
    if receipt is None or (receipt.scene is not scene and receipt.scene != scene):
        return
    if fresh:
        state.validate(checkpoint)
        receipt.observed.append(checkpoint)
        receipt.latest_committed = checkpoint
    elif checkpoint is not receipt.latest_committed:
        raise RecoveryError(
            "VERIFICATION_STATE_MISMATCH", "Only the latest post-commit observation can be reused"
        )
    receipt.verified_after = checkpoint


def observe_callback(scene=None):
    receipt = active_transaction()
    if receipt is not None:
        receipt.latest_committed = None
        receipt.verified_after = None
        try:
            receipt.observe(scene or receipt.scene)
        except Exception as error:
            receipt.record.setdefault("observer_errors", []).append(str(error))


def note_native_commit(scene=None, *, checkpoint=None, kind="modify"):
    receipt = active_transaction()
    if receipt is not None:
        if kind not in {"modify", "undo", "redo"}:
            raise ValueError("Unsupported native history kind")
        target_scene = scene or receipt.scene
        if target_scene is not receipt.scene and target_scene != receipt.scene:
            raise RecoveryError("TRANSACTION_SCENE_CHANGED", "Native commit belongs to another scene")
        receipt.latest_committed = None
        receipt.verified_after = None
        receipt.record["native_commits"] += 1
        receipt.record.setdefault("native_history", []).append(kind)
        if kind == "undo":
            receipt.record["native_undo_commits"] = receipt.record.get("native_undo_commits", 0) + 1
        if checkpoint is None:
            receipt.observe(target_scene)
        else:
            state.validate(checkpoint)
            receipt.observed.append(checkpoint)
        receipt.latest_committed = receipt.observed[-1]


def _capacity():
    root = _root()
    checkpoints = len(list((root / "checkpoints").glob("*.json")))
    transactions = len(list((root / "transactions").glob("*.json")))
    return {
        "checkpoint_limit": MAX_CHECKPOINTS,
        "checkpoints_used": checkpoints,
        "checkpoints_remaining": max(0, MAX_CHECKPOINTS - checkpoints),
        "transaction_limit": MAX_TRANSACTIONS,
        "transactions_used": transactions,
        "transactions_remaining": max(0, MAX_TRANSACTIONS - transactions),
        "max_frames": state.MAX_FRAMES,
        "max_objects": state.MAX_OBJECTS,
        "max_channel_samples": state.MAX_CHANNEL_SAMPLES,
        "max_checkpoint_bytes": state.MAX_BYTES,
    }


def _validated_recovery_checkpoint(record, checkpoint_id):
    saved = _read(_record_path("checkpoints", checkpoint_id))
    if saved is None:
        raise RecoveryError(
            "CHECKPOINT_NOT_FOUND",
            "The selected managed recovery checkpoint does not exist",
            execution_state="not_started",
        )
    try:
        older = datetime.fromisoformat(saved["created_at"]) <= datetime.fromisoformat(record["created_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise RecoveryError(
            "INVALID_RECOVERY_CHECKPOINT",
            "The recovery checkpoint's provenance timestamp cannot be verified",
            execution_state="not_started",
        ) from error
    if not older:
        raise RecoveryError(
            "RECOVERY_CHECKPOINT_TOO_NEW",
            "A selected recovery checkpoint must predate the failed transaction",
            execution_state="not_started",
        )
    if saved["scene"] != record["scene"] or state._structural_identity(
        saved["checkpoint"]
    ) != state._structural_identity(record["checkpoint"]):
        raise RecoveryError(
            "RECOVERY_CHECKPOINT_SCOPE",
            "The selected checkpoint belongs to another saved scene or native identity/topology",
            execution_state="not_started",
        )
    return saved


def _recovery_target(record):
    choice = record.get("recovery_target")
    if choice is None:
        return record["checkpoint"], record["view_state"]
    saved = _validated_recovery_checkpoint(record, choice["checkpoint_id"])
    if saved["checkpoint_sha256"] != choice["checkpoint_sha256"]:
        raise RecoveryError(
            "RECOVERY_CHECKPOINT_CHANGED",
            "The explicitly selected recovery checkpoint changed after selection",
        )
    return saved["checkpoint"], saved["view_state"]


def _select_recovery_checkpoint(record, checkpoint_id):
    saved = _validated_recovery_checkpoint(record, checkpoint_id)
    if len(state.canonical_bytes(record)) + state.MAX_BYTES > MAX_RECORD_BYTES - 8192:
        raise RecoveryError(
            "RECOVERY_STORAGE_LIMIT",
            "The original evidence and a complete recovered state exceed the journal budget",
            execution_state="not_started",
        )
    choice = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": saved["checkpoint_sha256"],
        "scene": saved["scene"],
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    record.setdefault("original_status", record["status"])
    record.setdefault("original_execution_state", record["execution_state"])
    record.setdefault("original_rollback_verified", record.get("rollback_verified", False))
    record.setdefault("recovery_attempts", []).append(choice)
    record["recovery_target"] = choice
    # Preserve original checkpoint/error/rollback evidence. The selected target
    # is durable before any native mutation and remains explicit after restart.
    _write(_record_path("transactions", record["request_id"]), record)
    return saved["checkpoint"], saved["view_state"]


def status(scene=None, view=None, request_id=None):
    result = {
        "persistent": True,
        "writes_blocked": False,
        "status": "clear",
        "rollback_scope": "bounded native model and recorded timeline/selection; opaque asset payloads and arbitrary Python side effects excluded",
        "automatic_retry_allowed": False,
        "capacity": _capacity(),
    }
    if request_id is not None:
        result["transaction"] = status_by_id(request_id)
    try:
        record = _pending()
        if _in_memory_failure is not None:
            result.update(writes_blocked=True, status="persistence_failed", error=_in_memory_failure)
        if record is not None:
            result.update(
                writes_blocked=True,
                status=record["status"],
                request_id=record["request_id"],
                operation=record["operation"],
                next_step="Preserve a new quarantine scene; use recovery_verify or recovery_restore. No unchecked unlock is available.",
            )
            result["recorded_scene"] = record["scene"]
            result["candidate_scene"] = _scene_identity(view) if view is not None else None
            result["identity_policy"] = (
                "Complete native UUID/model state; candidate pathname may differ for a native source or quarantine copy"
            )
            target, _ = _recovery_target(record)
            if record.get("recovery_target"):
                result["selected_target"] = record["recovery_target"]
            if scene is not None:
                candidate = state.capture(scene)
                report = state.compare(target, candidate)
                result["checkpoint_matches"] = report["equivalent"]
                result["original_checkpoint_matches"] = state.compare(record["checkpoint"], candidate)[
                    "equivalent"
                ]
                result["comparison"] = report
    except Exception as error:
        result.update(writes_blocked=True, status="invalid_or_unverifiable_record", error=str(error))
    return result


def readiness(scene=None):
    return status(scene=scene)


def status_by_id(request_id):
    record = _read(_record_path("transactions", request_id))
    if record is None:
        return {"request_id": request_id, "status": "not_found", "execution_state": "not_started"}
    return {
        key: record[key]
        for key in (
            "request_id",
            "operation",
            "status",
            "native_commits",
            "execution_state",
            "rollback_verified",
            "failure",
            "rollback",
            "created_at",
            "recovery_target",
            "recovery_verified",
            "recovery_attempts",
            "original_status",
            "original_execution_state",
            "native_history",
        )
        if key in record
    }


def guard(operation, view=None, scene=None):
    # The caller must additionally enforce quarantine save-as (overwrite=False).
    if operation.startswith("recovery_") or operation in {"scene_save", "playback_stop"}:
        return
    if active_transaction() is not None:
        return
    result = status()
    if result["writes_blocked"]:
        raise RecoveryError(
            "RECOVERY_REQUIRED",
            "Writes are blocked until the complete recorded checkpoint is verified",
            details=result,
        )


def _undo():
    """One native Undo; trusted tests may replace this private seam."""
    import csc

    csc.app.get_application().get_action_manager().call_action("Scene.Undo")


def _redo():
    import csc

    csc.app.get_application().get_action_manager().call_action("Scene.Redo")


class Receipt:
    def __init__(self, view, scene, operation, request_id):
        self.view, self.scene = view, scene
        self.path = _record_path("transactions", request_id)
        if self.path.exists():
            raise RecoveryError(
                "REQUEST_ALREADY_RECORDED",
                "This request ID already exists; inspect transaction status instead of replaying",
                execution_state="not_started",
                details=status_by_id(request_id),
            )
        if _capacity()["transactions_remaining"] <= 0:
            raise RecoveryError(
                "TRANSACTION_CAPACITY",
                "Release completed transaction records before recording another edit",
                execution_state="not_started",
            )
        self.checkpoint = state.capture(scene)
        self.observed = [self.checkpoint]
        self.record = {
            "schema_version": 1,
            "request_id": request_id,
            "operation": operation,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "host_pid": os.getpid(),
            "scene": _scene_identity(view),
            "view_state": _view_state(view, scene),
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": state.digest(self.checkpoint),
            "native_commits": 0,
            "status": "pending",
            "execution_state": "unknown",
        }
        self.committed = False
        self.latest_committed = None
        self.verified_after = None
        try:
            _write(self.path, self.record)
            _write(_root() / "active.json", {"schema_version": 1, "request_id": request_id})
        except Exception as error:
            raise RecoveryError(
                "JOURNAL_PERSISTENCE_FAILED",
                "The pre-write checkpoint could not be persisted; no mutation was started",
                execution_state="not_started",
                details={"request_id": request_id},
            ) from error

    def observe(self, scene):
        if scene is not self.scene and scene != self.scene:
            raise RecoveryError(
                "TRANSACTION_SCENE_CHANGED", "The modification belongs to another native scene"
            )
        self.observed.append(state.capture(scene))

    def commit(self, result=None):
        after = self.verified_after if self.verified_after is not None else state.capture(self.scene)
        self.record.update(
            status="committed",
            execution_state="completed",
            after=after,
            after_sha256=state.digest(after),
            rollback_verified=False,
        )
        _write(self.path, self.record)
        _clear_marker(self.record["request_id"])
        self.committed = True
        return {
            "request_id": self.record["request_id"],
            "status": "committed",
            "execution_state": "completed",
            "checkpoint_sha256": self.record["checkpoint_sha256"],
            "after_sha256": self.record["after_sha256"],
        }

    def rollback(self, cause):
        global _in_memory_failure
        report = {"undo_actions": 0, "manual_restore": False, "attempt_errors": []}
        verified = False
        try:
            current = state.capture(self.scene)
            self.observed.append(current)
            verified = state.compare(self.checkpoint, current)["equivalent"]
            if not verified and self.record["native_commits"]:
                inverse = list(reversed(self.record.get("native_history", ["modify"])))
                inverse_index = 0
                for _ in range(12):
                    if not any(state.compare(known, current)["equivalent"] for known in self.observed):
                        raise RecoveryError(
                            "UNDO_HISTORY_DIVERGED", "Native history left the recorded transaction states"
                        )
                    if inverse[inverse_index] == "undo":
                        _redo()
                    else:
                        _undo()
                    report["undo_actions"] += 1
                    after = state.capture(self.scene)
                    if state.compare(self.checkpoint, after)["equivalent"]:
                        verified = True
                        break
                    if not state.compare(current, after)["equivalent"] and not any(
                        state.compare(known, after)["equivalent"] for known in self.observed
                    ):
                        raise RecoveryError(
                            "UNDO_HISTORY_DIVERGED",
                            "Native Undo produced an unrecorded model; no further Undo was attempted",
                        )
                    if not state.compare(current, after)["equivalent"]:
                        inverse_index = min(inverse_index + 1, len(inverse) - 1)
                    current = after
        except Exception as error:
            report["attempt_errors"].append(f"{type(error).__name__}: {error}")
        if not verified:
            try:
                restore = state.restore(self.scene, self.checkpoint)
                report["manual_restore"] = True
                verified = (
                    restore["restored"]
                    and state.compare(self.checkpoint, state.capture(self.scene))["equivalent"]
                )
            except Exception as error:
                report["attempt_errors"].append(f"{type(error).__name__}: {error}")
        if verified:
            try:
                _restore_view(self.view, self.scene, self.record["view_state"])
            except Exception as error:
                verified = False
                report["attempt_errors"].append(f"{type(error).__name__}: {error}")
        try:
            report["comparison"] = state.compare(self.checkpoint, state.capture(self.scene))
            verified = verified and report["comparison"]["equivalent"]
        except Exception as error:
            verified = False
            report["attempt_errors"].append(f"Final verification failed: {error}")
        self.record.update(
            status="rolled_back" if verified else "recovery_required",
            execution_state="rolled_back" if verified else "recovery_required",
            rollback_verified=verified,
            failure={"code": getattr(cause, "code", type(cause).__name__), "message": str(cause)},
            rollback=report,
        )
        try:
            _write(self.path, self.record)
            if verified:
                _clear_marker(self.record["request_id"])
        except Exception as error:
            _in_memory_failure = str(error)
            verified = False
            report["attempt_errors"].append(f"Persistence failed: {error}")
        raise RecoveryError(
            "TRANSACTION_ROLLED_BACK" if verified else "RECOVERY_REQUIRED",
            f"{cause}; rollback_verified={verified}",
            execution_state="rolled_back" if verified else "recovery_required",
            rollback_verified=verified,
            details={
                "request_id": self.record["request_id"],
                "rollback": report,
                "cause": self.record["failure"],
            },
        ) from cause


@contextmanager
def transaction(view, scene, operation, request_id=None):
    if active_transaction() is not None:
        raise RecoveryError(
            "NESTED_TRANSACTION",
            "Nested native transaction journals are not supported",
            execution_state="not_started",
        )
    guard(operation, view, scene)
    receipt = Receipt(view, scene, operation, request_id or uuid.uuid4().hex)
    token = _active.set(receipt)
    try:
        try:
            yield receipt
            if not receipt.committed:
                raise RuntimeError("The operation exited without explicit transaction verification/commit")
        except Exception as error:
            if receipt.committed:
                raise
            receipt.rollback(error)
    finally:
        _active.reset(token)


def checkpoint_create(label=""):
    _, _, view, scene = common.current_context()
    guard("checkpoint_create", view, scene)
    if _capacity()["checkpoints_remaining"] <= 0:
        raise RecoveryError(
            "CHECKPOINT_CAPACITY",
            "Release a checkpoint before creating another one",
            execution_state="not_started",
        )
    identifier = uuid.uuid4().hex
    model = state.capture(scene)
    record = {
        "schema_version": 1,
        "checkpoint_id": identifier,
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": _scene_identity(view),
        "view_state": _view_state(view, scene),
        "checkpoint": model,
        "checkpoint_sha256": state.digest(model),
        "history": sorted(path.stem for path in (_root() / "transactions").glob("*.json")),
    }
    _write(_record_path("checkpoints", identifier), record)
    return {
        "checkpoint_id": identifier,
        "label": label,
        "frames": model["count"],
        "objects": len(model["objects"]),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "scope": "full bounded native model; asset payload limitations described by recovery_status",
    }


def checkpoint_status(checkpoint_id):
    record = _read(_record_path("checkpoints", checkpoint_id))
    if record is None:
        raise RecoveryError(
            "CHECKPOINT_NOT_FOUND", "No managed checkpoint has this ID", execution_state="not_started"
        )
    _, _, view, scene = common.current_context()
    return {
        "checkpoint_id": checkpoint_id,
        "scene_matches": _scene_identity(view) == record["scene"],
        "comparison": state.compare(record["checkpoint"], state.capture(scene)),
    }


def checkpoint_restore(checkpoint_id):
    record = _read(_record_path("checkpoints", checkpoint_id))
    if record is None:
        raise RecoveryError(
            "CHECKPOINT_NOT_FOUND", "No managed checkpoint has this ID", execution_state="not_started"
        )
    _, _, view, scene = common.current_context()
    guard("checkpoint_restore", view, scene)
    if _scene_identity(view) != record["scene"]:
        raise RecoveryError(
            "CHECKPOINT_SCENE_MISMATCH",
            "The active saved scene differs from this checkpoint",
            execution_state="not_started",
        )
    current = active_checkpoint(scene)
    if current is None:
        current = state.capture(scene)
    known = [record["checkpoint"]]
    undo_actions = 0
    if not state.compare(record["checkpoint"], current)["equivalent"]:
        chain = []
        for path in (_root() / "transactions").glob("*.json"):
            if path.stem in record["history"]:
                continue
            if active_transaction() is not None and path.stem == active_transaction().record["request_id"]:
                continue
            item = _read(path)
            if item.get("scene") == record["scene"] and item.get("status") in {
                "committed",
                "recovered_to_checkpoint",
            }:
                chain.append(item)
        chain.sort(key=lambda item: item["created_at"])
        previous = record["checkpoint"]
        for item in chain:
            after_state = (
                item.get("recovered_after")
                if item["status"] == "recovered_to_checkpoint"
                else item.get("after")
            )
            if not state.compare(previous, item["checkpoint"])["equivalent"] or after_state is None:
                raise RecoveryError(
                    "CHECKPOINT_HISTORY_DIVERGED",
                    "A recorded edit is not connected to this checkpoint",
                    execution_state="not_started",
                )
            previous = after_state
            known.extend([item["checkpoint"], after_state])
        if not chain or not state.compare(previous, current)["equivalent"]:
            raise RecoveryError(
                "EXTERNAL_STATE_CHANGED",
                "The active model differs from the checkpoint's recorded edit history",
                execution_state="not_started",
            )
        for _ in range(12):
            if state.compare(record["checkpoint"], current)["equivalent"]:
                break
            if not any(state.compare(item, current)["equivalent"] for item in known):
                break
            note_native_attempt(scene)
            _undo()
            undo_actions += 1
            after = state.capture(scene)
            note_native_commit(scene, checkpoint=after, kind="undo")
            if not state.compare(current, after)["equivalent"] and not any(
                state.compare(item, after)["equivalent"] for item in known
            ):
                break
            current = after
    if state.compare(record["checkpoint"], current)["equivalent"]:
        result = {"restored": True, "method": "checked_native_undo" if undo_actions else "already_equal"}
    else:
        result = state.restore(scene, record["checkpoint"])
    _restore_view(view, scene, record["view_state"])
    final = state.capture(scene)
    report = state.compare(record["checkpoint"], final)
    if not report["equivalent"]:
        raise RecoveryError(
            "CHECKPOINT_RESTORE_MISMATCH",
            "Model state changed while restoring the checkpoint's view state",
            details=report,
        )
    result["verification"] = report
    result["native_undo_actions"] = undo_actions
    accept_verified_checkpoint(scene, final, fresh=True)
    return {"checkpoint_id": checkpoint_id, **result}


def recovery_status():
    try:
        _, _, view, scene = common.current_context(require_scene=False)
    except Exception:
        view = scene = None
    return status(scene=scene, view=view)


def _verify_recovery(record, target, saved_view, *, candidate=None):
    global _in_memory_failure
    _, _, view, scene = common.current_context()
    report = state.compare(target, candidate if candidate is not None else state.capture(scene))
    if not report["equivalent"]:
        raise RecoveryError(
            "CHECKPOINT_MISMATCH", "The complete recorded native model has not been restored", details=report
        )
    _restore_view(view, scene, saved_view)
    final = state.capture(scene)
    report = state.compare(target, final)
    if not report["equivalent"]:
        raise RecoveryError(
            "CHECKPOINT_MISMATCH", "View restoration changed the recovered native model", details=report
        )
    selected = record.get("recovery_target")
    if selected:
        record.update(
            status="recovered_to_checkpoint",
            execution_state="completed",
            recovery_verified=True,
            recovered_after=final,
            recovered_after_sha256=state.digest(final),
            rollback_verified=False,
        )
    else:
        record.update(
            status="rolled_back",
            execution_state="rolled_back",
            rollback_verified=True,
            recovery_verified=True,
        )
    _write(_record_path("transactions", record["request_id"]), record)
    _clear_marker(record["request_id"])
    _in_memory_failure = None
    return {
        "verified": True,
        "request_id": record["request_id"],
        "comparison": report,
        "status": record["status"],
        "rollback_verified": record["rollback_verified"],
        **({"recovered_to_checkpoint": selected["checkpoint_id"]} if selected else {}),
    }


def recovery_verify():
    record = _pending()
    if record is None:
        if _in_memory_failure is not None:
            raise RecoveryError(
                "RECOVERY_REQUIRED",
                "Durability could not be established; preserve records and restart before checking recovery",
            )
        return {"verified": True, "status": "clear"}
    target, saved_view = _recovery_target(record)
    return _verify_recovery(record, target, saved_view)


def recovery_restore(checkpoint_id=None):
    record = _pending()
    if record is None:
        if checkpoint_id is not None:
            raise RecoveryError(
                "NO_RECOVERY_REQUIRED",
                "Use checkpoint_restore when no recovery fence is active",
                execution_state="not_started",
            )
        return recovery_verify()
    target, saved_view = (
        _select_recovery_checkpoint(record, checkpoint_id)
        if checkpoint_id is not None
        else _recovery_target(record)
    )
    _, _, _, scene = common.current_context()
    candidate = state.capture(scene)
    if not state.compare(target, candidate)["equivalent"]:
        state.restore(scene, target)
        candidate = None
    return _verify_recovery(record, target, saved_view, candidate=candidate)


def recovery_open(path):
    if _pending() is None:
        raise RecoveryError(
            "NO_RECOVERY_REQUIRED",
            "Use ordinary scene_open when no recovery fence is active",
            execution_state="not_started",
        )
    source = common.workspace_path(path, must_exist=True, suffixes=(".casc",))
    _, app, _, _ = common.current_context(require_scene=False)
    if not app.get_data_source_manager().load_scene(str(source)):
        raise RecoveryError("RECOVERY_OPEN_FAILED", "Cascadeur declined to open the proposed recovery scene")
    # A different pathname is valid only when all recorded native identities,
    # settings, frames and metadata actually match. A changed quarantine stays fenced.
    return {"opened": str(source), **recovery_verify()}


def checkpoint_release(checkpoint_id):
    target = _record_path("checkpoints", checkpoint_id)
    if _read(target) is None:
        return {"released": False}
    target.unlink()
    _sync_directory(target.parent)
    return {"released": True, "checkpoint_id": checkpoint_id}


def transaction_release(request_id):
    target = _record_path("transactions", request_id)
    record = _read(target)
    if record is None:
        return {"released": False}
    if record.get("status") not in {"committed", "rolled_back", "recovered_to_checkpoint"}:
        raise RecoveryError("RECOVERY_REQUIRED", "Pending or failed transactions cannot be released")
    for path in (_root() / "checkpoints").glob("*.json"):
        checkpoint = _read(path)
        if record.get("scene") == checkpoint["scene"] and request_id not in checkpoint["history"]:
            raise RecoveryError(
                "CHECKPOINT_HISTORY_IN_USE",
                "Release the dependent checkpoint before releasing its recorded edit history",
                execution_state="not_started",
            )
    marker = _read(_root() / "active.json")
    if marker is not None and marker.get("request_id") == request_id:
        _clear_marker(request_id)
    target.unlink()
    _sync_directory(target.parent)
    return {"released": True, "request_id": request_id}


_ID = {"type": "string", "minLength": 1, "maxLength": 80}
TOOLS = [
    Tool(
        "checkpoint_create",
        "Capture every stored model channel and track into a bounded durable checkpoint; no native save-as or asset copying.",
        checkpoint_create,
        {"label": {"type": "string", "maxLength": 200}},
        (),
        False,
    ),
    Tool(
        "checkpoint_status",
        "Compare a managed checkpoint with the complete active native model.",
        checkpoint_status,
        {"checkpoint_id": _ID},
        ("checkpoint_id",),
    ),
    Tool(
        "checkpoint_restore",
        "Restore a managed whole-model checkpoint when native identities remain supported; verify every stored frame and metadata.",
        checkpoint_restore,
        {"checkpoint_id": _ID},
        ("checkpoint_id",),
        False,
    ),
    Tool(
        "checkpoint_release",
        "Release one managed checkpoint JSON without editing the scene.",
        checkpoint_release,
        {"checkpoint_id": _ID},
        ("checkpoint_id",),
        False,
    ),
    Tool(
        "recovery_status",
        "Read the persistent write fence, checkpoint match and remaining journal capacity.",
        recovery_status,
    ),
    Tool(
        "recovery_verify",
        "Clear a recovery fence only after the complete recorded native checkpoint and view state actually match.",
        recovery_verify,
        read_only=False,
    ),
    Tool(
        "recovery_restore",
        "Recover to the recorded pre-write model, or explicitly choose a checksummed managed checkpoint predating the failure in the same saved scene/topology. Clears the fence only after full verification; an alternative target is reported as recovery, never as rollback.",
        recovery_restore,
        {"checkpoint_id": _ID},
        (),
        False,
    ),
    Tool(
        "recovery_open",
        "Open a workspace native checkpoint without closing existing tabs; clear the fence only after its complete recorded model matches.",
        recovery_open,
        {"path": {"type": "string", "minLength": 1, "maxLength": 4096}},
        ("path",),
        False,
    ),
    Tool(
        "transaction_status",
        "Read a durable request receipt; never replay an uncertain relative edit automatically.",
        status_by_id,
        {"request_id": _ID},
        ("request_id",),
    ),
    Tool(
        "transaction_release",
        "Release a terminal durable receipt to free capacity; pending/recovery records cannot be removed.",
        transaction_release,
        {"request_id": _ID},
        ("request_id",),
        False,
    ),
]
