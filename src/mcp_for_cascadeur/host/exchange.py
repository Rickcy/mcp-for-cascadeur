"""Checked single-character FBX delivery through the installed native exporter."""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid

from ..contracts import Tool
from . import characters, common, state
from . import scene as scene_tools

MAX_EXPORT_BYTES = 512 * 1024 * 1024
SETTINGS = {"mode": "Ascii", "up_axis": "Y", "bake_animation": True, "apply_euler_filter": True}


class ExportError(RuntimeError):
    def __init__(self, code, message, *, execution_state="not_started", details=None):
        super().__init__(message)
        self.code = code
        self.execution_state = execution_state
        self.details = details


def _inspect(scene, view, app, character_id):
    blockers = []
    try:
        entitled = bool(app.is_export_available())
    except Exception as error:
        entitled = False
        blockers.append({"code": "EXPORT_ENTITLEMENT_UNVERIFIED", "message": str(error)})
    if not entitled:
        blockers.append(
            {
                "code": "EXPORT_UNAVAILABLE",
                "message": "The native installation does not report export entitlement",
            }
        )
    rig = None
    joints, meshes = set(), set()
    if len(scene.behaviour_viewer().get_behaviours("RigInfo")) != 1:
        blockers.append(
            {"code": "EXPORT_SCOPE", "message": "Verified export requires exactly one RigInfo character"}
        )
    else:
        try:
            rig = characters.resolve_character(scene, character_id)
            model = scene.model_viewer()
            joints = {item for item in model.get_objects() if model.get_object_type_name(item) == "Joint"}
            meshes = {
                scene.behaviour_viewer().get_behaviour_owner(item)
                for item in scene.behaviour_viewer().get_behaviours("MeshObject")
            }
            if not joints or not joints | meshes <= rig.owned or joints != set(rig.joints):
                blockers.append(
                    {
                        "code": "EXPORT_SCOPE",
                        "message": "All native Joint and Mesh owners must belong to the identified character",
                    }
                )
        except Exception as error:
            blockers.append({"code": "EXPORT_SCOPE", "message": str(error)})
    count = int(scene.data_viewer().get_animation_size())
    boundary = view.animation_boundary()
    playback = {"first": int(boundary.first_frame), "last": int(boundary.last_frame)}
    if not 1 <= count <= state.MAX_FRAMES:
        blockers.append(
            {
                "code": "EXPORT_RANGE",
                "message": f"Verified export supports 1-{state.MAX_FRAMES} stored frames",
            }
        )
    if playback != {"first": 0, "last": count - 1}:
        blockers.append(
            {
                "code": "EXPORT_RANGE",
                "message": "Set the native playback range to the complete stored range before export",
            }
        )
    return {
        "ready": not blockers,
        "entitled": entitled,
        "scope_verified": rig is not None and not any(item["code"] == "EXPORT_SCOPE" for item in blockers),
        "character_id": rig.id if rig else character_id,
        "joint_count": len(joints),
        "mesh_count": len(meshes),
        "stored_frames": count,
        "range": {"first": 0, "last": count - 1},
        "playback_range": playback,
        "settings": dict(SETTINGS),
        "blockers": blockers,
        "scope": "whole scene, exactly one character; all native Joint and Mesh owners verified; selection ignored",
        "asset_delivery": "Native FBX file; linked texture sidecars are not repackaged",
    }


def fbx_export_status(character_id=None):
    _, app, view, scene = common.current_context()
    return _inspect(scene, view, app, character_id)


def _sha(path):
    total = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        initial = os.fstat(stream.fileno())
        if initial.st_size > MAX_EXPORT_BYTES:
            raise ExportError(
                "EXPORT_FILE_LIMIT",
                "The output exceeds the verified file size limit",
                execution_state="unknown",
            )
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > MAX_EXPORT_BYTES:
                raise ExportError(
                    "EXPORT_FILE_LIMIT",
                    "The output grew beyond its verified file size limit",
                    execution_state="unknown",
                )
            digest.update(block)
        final = os.fstat(stream.fileno())
    if (initial.st_size, initial.st_mtime_ns, initial.st_ino) != (
        final.st_size,
        final.st_mtime_ns,
        final.st_ino,
    ):
        raise ExportError(
            "EXPORT_FILE_CHANGED", "The file changed while it was being hashed", execution_state="unknown"
        )
    return digest.hexdigest(), final


def _verify_file(path, started_ns, *, fbx):
    if not path.is_file() or path.is_symlink():
        raise ExportError(
            "EXPORT_FAILED", "Native export did not produce a regular output file", execution_state="unknown"
        )
    digest, stat = _sha(path)
    if stat.st_size <= (1024 if fbx else 0):
        raise ExportError(
            "EXPORT_FAILED",
            "Native output is empty or below the verified size threshold",
            execution_state="unknown",
        )
    if stat.st_mtime_ns < started_ns - 2_000_000_000 or stat.st_mtime_ns > time.time_ns() + 2_000_000_000:
        raise ExportError(
            "EXPORT_STALE_FILE",
            "Output modification time is outside this export attempt",
            execution_state="unknown",
        )
    if fbx:
        with path.open("rb") as stream:
            if not stream.read(128).startswith(b"; FBX "):
                raise ExportError(
                    "EXPORT_INVALID_HEADER",
                    "Native output is not the requested ASCII FBX",
                    execution_state="unknown",
                )
    return {"path": str(path), "bytes": stat.st_size, "sha256": digest, "mtime_ns": stat.st_mtime_ns}


def _expected_destination(destination, expected_sha256):
    if expected_sha256 is not None and (
        type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256)
    ):
        raise ExportError("EXPORT_INVALID_HASH", "expected_sha256 must be a complete SHA-256 hex digest")
    expected = expected_sha256.lower() if expected_sha256 else None
    if destination.exists():
        if (
            expected is None
            or not destination.is_file()
            or destination.is_symlink()
            or _sha(destination)[0] != expected
        ):
            raise ExportError(
                "EXPORT_CONFLICT", "The destination exists or its contents differ from expected_sha256"
            )
    elif expected is not None:
        raise ExportError("EXPORT_CONFLICT", "The expected destination is missing")
    return expected


def _publish(staged, destination, expected):
    if expected is None:
        try:
            os.link(staged, destination)
        except FileExistsError as error:
            raise ExportError(
                "EXPORT_CONFLICT",
                "Another writer created the destination during export",
                execution_state="unknown",
            ) from error
        staged.unlink()
        return "exclusive_new_file"
    if not destination.is_file() or destination.is_symlink():
        raise ExportError(
            "EXPORT_CONFLICT", "The expected destination disappeared during export", execution_state="unknown"
        )
    digest, observed = _sha(destination)
    current = destination.stat()
    if digest != expected or (current.st_size, current.st_mtime_ns, current.st_ino) != (
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ino,
    ):
        raise ExportError(
            "EXPORT_CONFLICT", "The destination changed during export", execution_state="unknown"
        )
    os.replace(staged, destination)
    return "atomic_replace_after_expected_sha256_recheck"


def fbx_export_verified(
    path, source_copy_path, character_id=None, expected_sha256=None, start_frame=0, end_frame=None
):
    destination = common.workspace_path(path, overwrite=True, suffixes=(".fbx",))
    source = common.workspace_path(source_copy_path, overwrite=True, suffixes=(".casc",))
    if source.exists():
        raise ExportError(
            "EXPORT_SOURCE_CONFLICT", "The source-copy path must be new and cannot overwrite a scene"
        )
    expected = _expected_destination(destination, expected_sha256)
    csc, app, view, scene = common.current_context()
    status = _inspect(scene, view, app, character_id)
    if not status["ready"]:
        blocker = status["blockers"][0]
        raise ExportError(blocker["code"], blocker["message"], details=status)
    last = status["stored_frames"] - 1
    if (
        type(start_frame) is not int
        or start_frame != 0
        or (end_frame is not None and (type(end_frame) is not int or end_frame != last))
    ):
        raise ExportError("EXPORT_RANGE", f"Verified export requires the full stored range 0..{last}")
    # Capture before save-as: a source copy is an intended pathname change, not a
    # permitted change to data/settings/topology/track metadata.
    before = state.capture(scene)
    source_started = time.time_ns()
    try:
        # Reserve only this new path; never let save-as overwrite a competing file.
        with source.open("xb"):
            pass
    except FileExistsError as error:
        raise ExportError(
            "EXPORT_SOURCE_CONFLICT", "The new native source-copy path was created by another writer"
        ) from error
    view.save(str(source))
    saved = _verify_file(source, source_started, fbx=False)
    stage = destination.parent / (".mcp-for-cascadeur-export-" + uuid.uuid4().hex)
    stage.mkdir()
    original_stage = stage.resolve()
    temporary = stage / destination.name
    cleanup_note = None
    try:
        settings = csc.fbx.FbxSettings()
        settings.mode = csc.fbx.FbxSettingsMode.Ascii
        settings.up_axis = csc.fbx.FbxSettingsAxis.Y
        settings.bake_animation = True
        settings.apply_euler_filter = True
        loader = scene_tools._native_tool(app, "FbxSceneLoader").get_fbx_loader(view)
        loader.set_settings(settings)
        export_started = time.time_ns()
        loader.export_all_objects(str(temporary))
        staged = _verify_file(temporary, export_started, fbx=True)
        report = state.compare(before, state.capture(scene))
        if not report["equivalent"]:
            raise ExportError(
                "EXPORT_STATE_CHANGED",
                "Native model changed during save/export; output was not published and the source copy was retained",
                execution_state="unknown",
                details={"source_copy": saved, "comparison": report},
            )
        publication = _publish(temporary, destination, expected)
        delivered = _verify_file(destination, export_started, fbx=True)
        if delivered["sha256"] != staged["sha256"]:
            raise ExportError(
                "EXPORT_PUBLICATION_CHANGED",
                "Published FBX was changed by another writer; inspect the destination",
                execution_state="unknown",
            )
        result = {
            "exported": True,
            **delivered,
            "source_copy": saved,
            "character_id": status["character_id"],
            "stored_frames": status["stored_frames"],
            "start_frame": 0,
            "end_frame": last,
            "settings": dict(SETTINGS),
            "publication": publication,
            "model_unchanged": True,
            "model_comparison": report,
            "scope": status["scope"],
        }
    finally:
        # Only remove our precise staged filename and the directory if empty.
        # Unexpected native sidecars remain available for review.
        if stage.resolve() == original_stage and original_stage.is_relative_to(
            common.get_settings()["workspace"].resolve()
        ):
            if temporary.is_file() or temporary.is_symlink():
                temporary.unlink(missing_ok=True)
            try:
                stage.rmdir()
            except OSError:
                cleanup_note = str(stage)
    if cleanup_note:
        result["staging_directory_retained"] = cleanup_note
    return result


_PATH = {"type": "string", "minLength": 1, "maxLength": 4096}
_CHARACTER = {"type": "string", "minLength": 1, "maxLength": 512}
TOOLS = [
    Tool(
        "fbx_export_status",
        "Read native export entitlement, exact single-character Joint/Mesh scope and whole stored-frame range blockers.",
        fbx_export_status,
        {"character_id": _CHARACTER},
    ),
    Tool(
        "fbx_export_verified",
        "Save a new native source copy, stage a whole single-character ASCII/Y baked FBX, verify model/file integrity and publish with destination conflict protection.",
        fbx_export_verified,
        {
            "path": _PATH,
            "source_copy_path": _PATH,
            "character_id": _CHARACTER,
            "expected_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
            "start_frame": {"type": "integer", "const": 0},
            "end_frame": {"type": "integer", "minimum": 0},
        },
        ("path", "source_copy_path"),
        False,
    ),
]
