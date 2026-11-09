"""Staged Kimodo FBX import into a new Cascy scene, on Cascadeur's main thread.

The bridge never starts a model or edits an existing user tab. Each native write
has a durable stage marker; an interrupted write must be inspected, not replayed.
Load this module inside Cascadeur with importlib and keep each stage a separate
native script-server call. Model generation belongs to the Docker HTTP service.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
import zipfile
from pathlib import Path

SCHEMA = "kimodo-cascadeur-import-v1"
MAX_FRAMES = 240
POSITION_TOLERANCE = 0.01
_FILENAMES = {
    "before": "before.casc",
    "target": "working.casc",
    "source": "source.casc",
    "result": "result.casc",
}


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _input_file(value, suffix):
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file() or path.suffix.lower() != suffix:
        raise ValueError(f"Expected an existing absolute {suffix} file: {value}")
    return path.resolve()


def _scene_json(path, member, maximum_size):
    """Read one bounded native JSON member, including Cascadeur's Zstandard ZIPs."""
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo(member).file_size > maximum_size:
            raise ValueError(f"Native scene member is too large: {member}")
        try:
            content = archive.read(member)
        except NotImplementedError:
            # Python 3.11/3.12 cannot read ZIP method 93. Windows' bundled libarchive can.
            executable = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"
            if not executable.is_file():
                raise RuntimeError(
                    "A Zstandard-capable Windows tar.exe is required to inspect .casc settings"
                )
            result = subprocess.run(
                [str(executable), "-xOf", str(path), member],
                capture_output=True,
                check=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            content = result.stdout
    if len(content) > maximum_size:
        raise ValueError(f"Native scene member is too large: {member}")
    return json.loads(content)


def scene_settings(path):
    """Read the saved native timebase, including Cascadeur's Zstandard ZIPs."""
    path = _input_file(path, ".casc")
    data = _scene_json(path, "scene/scene_settings.json", 64 * 1024)["Data"]
    if data.get("FPS") != 30 or data.get("UnitLength") != "cm":
        raise ValueError("The bridge requires native scenes with 30 FPS and centimetre units")
    return {"fps": data["FPS"], "unit_length": data["UnitLength"], "gravity": data.get("Gravity")}


def validate_generation(manifest_path, fbx_path):
    """Check the actual encoder, timebase and bytes before touching Cascadeur."""
    manifest = _input_file(manifest_path, ".json")
    fbx = _input_file(fbx_path, ".fbx")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    frames = data.get("num_frames")
    if type(frames) is not int or not 2 <= frames <= MAX_FRAMES:
        raise ValueError(f"Generation requires 2-{MAX_FRAMES} integer frames")
    if type(data.get("fps")) not in (int, float) or data["fps"] != 30:
        raise ValueError("The Cascadeur bridge currently accepts 30 fps clips only")
    encoder = data.get("text_encoder") or {}
    if encoder.get("kind") != "llm2vec":
        raise ValueError("Generation must use the real LLM2Vec encoder; hash fallback is unsupported")
    if data.get("root_motion") is False:
        raise ValueError("This import expects root-motion generation")
    digest = _sha(fbx)
    if digest != str(data.get("sha256", "")).lower():
        raise ValueError("FBX checksum differs from the generation manifest")
    if fbx.stat().st_size < 32:
        raise ValueError("Generated FBX is empty or truncated")
    return {
        "fbx_path": str(fbx),
        "manifest_path": str(manifest),
        "sha256": digest,
        "num_frames": frames,
        "fps": 30,
        "model_name": data.get("model_name"),
        "text_encoder": encoder,
    }


def _write(receipt_path, record):
    from mcp_for_cascadeur.atomic import replace_file

    destination = Path(receipt_path)
    temporary = destination.with_name(destination.name + ".pending")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    replace_file(temporary, destination)


def _read(receipt_path):
    receipt = _input_file(receipt_path, ".json")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA or record.get("directory") != str(receipt.parent):
        raise ValueError("Not a valid bridge session receipt")
    return receipt, record


def _owned_path(record, kind):
    root = Path(record["directory"]).resolve()
    candidate = root / _FILENAMES[kind]
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
        raise ValueError("A bridge-owned path was replaced or escaped its session")
    return candidate


def prepare(workspace_root, cascy_scene, kimodo_scene, generation_manifest, fbx_path):
    """Create isolated source and target copies; this stage performs no native calls."""
    generation = validate_generation(generation_manifest, fbx_path)
    cascy = _input_file(cascy_scene, ".casc")
    kimodo = _input_file(kimodo_scene, ".casc")
    native_settings = {"cascy": scene_settings(cascy), "kimodo": scene_settings(kimodo)}
    workspace = Path(workspace_root).expanduser()
    if not workspace.is_absolute():
        raise ValueError("The exchange workspace must be absolute")
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    directory = workspace / ("import-" + uuid.uuid4().hex)
    directory.mkdir()
    record = {
        "schema": SCHEMA,
        "directory": str(directory),
        "phase": "prepared",
        "generation": generation,
        "created_ns": time.time_ns(),
        "events": [],
        "active_target": "target",
        "protected_tabs": None,
        "native_settings": native_settings,
    }
    try:
        for source, kind in ((cascy, "before"), (cascy, "target"), (kimodo, "source")):
            shutil.copyfile(source, _owned_path(record, kind))
        record["before_sha256"] = _sha(_owned_path(record, "before"))
        receipt = directory / "session.json"
        _write(receipt, record)
    except Exception:
        # This directory was just created and contains only our exact file names.
        for filename in (*_FILENAMES.values(), "session.json", "session.json.pending"):
            candidate = directory / filename
            if candidate.is_file() and not candidate.is_symlink():
                candidate.unlink()
        directory.rmdir()
        raise
    return {"receipt_path": str(receipt), "phase": record["phase"], "before_sha256": record["before_sha256"]}


def _application():
    import csc

    application = csc.app.get_application()
    if application is None:
        raise RuntimeError("Run the bridge inside the initialized Cascadeur application")
    return csc, application, application.get_scene_manager()


def _tabs(manager):
    return [{"name": item.name(), "path": item.get_path_name()} for item in manager.scenes()]


def _find_view(record, kind):
    _, _, manager = _application()
    path = _owned_path(record, kind).resolve()
    matches = [
        view
        for view in manager.scenes()
        if view.get_path_name() and Path(view.get_path_name()).resolve() == path
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one owned {kind} tab; found {len(matches)}")
    return matches[0]


def _require_callback_anchor(anchor_view, callback_scene):
    """A scene-idle callback must retain its own domain scene until it returns."""
    _, _, manager = _application()
    if callback_scene is None:
        raise RuntimeError("Scene closure requires the actual native scene_idle callback context")
    if manager.current_scene() != anchor_view or callback_scene != anchor_view.domain_scene():
        raise RuntimeError(
            "The active tab and scene_idle callback must both belong to the retained anchor scene"
        )


def _close_inactive_view(view, callback_scene):
    """Close an owned, saved scene without expanding closure to linked scenes.

    Callers retain their scene-idle anchor in another tab and save any owned
    edits first. The native close can return before a dialog/queued callback
    completes; that outcome must stop the stage instead of reopening the file.
    """
    _, app, manager = _application()
    if callback_scene is None:
        raise RuntimeError("Scene closure requires an explicit native callback scene")
    current = manager.current_scene()
    before = list(manager.scenes())
    if view not in before:
        raise RuntimeError("The owned scene is no longer open; inspect before retrying closure")
    if current == view or view.domain_scene() == callback_scene:
        raise RuntimeError(
            "Refusing to close the active tab or the scene held by the current native callback"
        )
    if current is None or current not in before or not any(
        item.domain_scene() == callback_scene for item in before
    ):
        raise RuntimeError("The active tab and native callback anchor must remain open")
    path = _input_file(view.get_path_name(), ".casc")
    try:
        linked = _scene_json(path, "tool/linked_scenes.json", 1024 * 1024)
        isolated = isinstance(linked, dict) and isinstance(linked.get("Data"), dict) and not linked["Data"]
    except (KeyError, ValueError, OSError, zipfile.BadZipFile) as error:
        raise RuntimeError("Cannot confirm that the saved owned scene has no linked scenes") from error
    if not isolated:
        raise RuntimeError("Refusing automatic closure of a scene with linked scenes")
    app.get_data_source_manager().close_scene(view)
    after = list(manager.scenes())
    if view in after:
        raise RuntimeError(
            "Native scene close has not completed; inspect the pending close before retrying"
        )
    if manager.current_scene() != current or any(item not in after for item in before if item != view):
        raise RuntimeError("Native scene close changed another tab or its retained callback anchor")


def _stage(receipt_path, expected, completed, function):
    receipt, record = _read(receipt_path)
    if record["phase"] != expected:
        raise RuntimeError(f"Expected stage {expected!r}, found {record['phase']!r}; inspect before retrying")
    record["phase"] = "running:" + completed
    record["events"].append({"stage": completed, "started_ns": time.time_ns()})
    _write(receipt, record)
    try:
        result = function(record)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["phase"] = "needs_review:" + completed
        _write(receipt, record)
        raise
    record["phase"] = completed
    record.pop("error", None)
    record["events"][-1]["finished_ns"] = time.time_ns()
    _write(receipt, record)
    return {"receipt_path": str(receipt), "phase": completed, **(result or {})}


def _open(record, kind):
    _, app, manager = _application()
    path = _owned_path(record, kind)
    if any(
        view.get_path_name() and Path(view.get_path_name()).resolve() == path for view in manager.scenes()
    ):
        raise RuntimeError("The new bridge path is already open; inspect instead of loading it twice")
    if not app.get_data_source_manager().load_scene(str(path)):
        raise RuntimeError(f"Cascadeur declined to open {path}")
    view = _find_view(record, kind)
    manager.set_current_scene(view)
    return view


def _rig(scene):
    viewer, model, layers = (
        scene.model_viewer().behaviour_viewer(),
        scene.model_viewer(),
        scene.layers_viewer(),
    )
    rigs = list(viewer.get_behaviours("RigInfo"))
    if len(rigs) != 1:
        raise RuntimeError("The bridge requires exactly one rigged character in each owned scene")
    information = rigs[0]
    owner = viewer.get_behaviour_owner(information)
    objects = {owner}
    for field in ("related_joints", "rig_objects", "related_objects"):
        objects.update(viewer.get_behaviour_objects_range(information, field))
    container = viewer.get_behaviour_by_name(owner, "ObjectsContainer")
    if not container.is_null():
        objects.update(viewer.get_behaviour_objects_range(container, "ids"))
    tracks = {layers.layer_id_by_obj_id_or_null(obj) for obj in objects}
    tracks = [track for track in tracks if not track.is_null()]
    if not tracks:
        raise RuntimeError("The character has no animation tracks")
    return {"owner": owner, "objects": objects, "tracks": tracks, "model": model}


def _modify(scene, label, callback):
    errors = []

    def execute(model, update, updater, session):
        try:
            callback(model, update, updater, session)
        except Exception as error:
            errors.append(error)
            raise

    accepted = scene.modify_with_session(label, execute)
    if errors:
        raise RuntimeError(f"{label} failed: {errors[0]}") from errors[0]
    if not accepted:
        raise RuntimeError(f"Cascadeur declined {label}; native outcome requires inspection")


def _select(view, count):
    csc, _, manager = _application()
    manager.set_current_scene(view)
    scene = view.domain_scene()
    rig = _rig(scene)

    def choose(_model, _update, _updater, session):
        session.take_selector().select(rig["objects"], rig["owner"])
        frames = csc.layers.index.FramesIndices.from_range(0, count - 1)
        indices = csc.layers.index.IndicesContainer(rig["tracks"], frames)
        session.take_layers_selector().set_full_selection_by_parts(indices)

    _modify(scene, "Select Kimodo retarget interval", choose)


def _vector(value):
    values = value.tolist() if hasattr(value, "tolist") else list(value)
    result = [float(item[0] if isinstance(item, (tuple, list)) else item) for item in values]
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise RuntimeError("A native Point has a nonfinite or invalid position")
    return result


def _snapshot(view, count):
    from mcp_for_cascadeur.host.characters import resolve_character

    scene = view.domain_scene()
    character = resolve_character(scene, require_semantics=True)
    stored = int(scene.data_viewer().get_animation_size())
    if stored < count:
        raise RuntimeError(f"Expected {count} stored motion frames, found {stored}")
    data, model = scene.data_viewer(), scene.model_viewer()
    positions = {}
    for key, binding in character.bindings.items():
        name = model.get_object_name(character.controls[key])
        if name in positions:
            raise RuntimeError("Character control names must be unique for reopen verification")
        positions[name] = [
            _vector(data.get_data_value(binding.position_data, frame)) for frame in range(count)
        ]
    pelvis_key = character.roles["pelvis"]["center"]
    pelvis_name = model.get_object_name(character.controls[pelvis_key])
    pelvis = positions[pelvis_name]
    delta = [pelvis[-1][axis] - pelvis[0][axis] for axis in range(3)]
    return {
        "fingerprint": character.fingerprint,
        "native_points": len(positions),
        "roles": sorted(character.roles),
        "positions": positions,
        "stored_frames": stored,
        "root_delta": delta,
        "root_distance": math.sqrt(sum(value * value for value in delta)),
    }


def _summary(snapshot):
    return {key: value for key, value in snapshot.items() if key != "positions"}


def open_target(receipt_path):
    def operation(record):
        _, _, manager = _application()
        record["protected_tabs"] = _tabs(manager)
        view = _open(record, "target")
        from mcp_for_cascadeur.host.characters import resolve_character

        character = resolve_character(view.domain_scene(), require_semantics=True)
        record["target_fingerprint"] = character.fingerprint
        return {"native_points": len(character.controls), "roles": sorted(character.roles)}

    return _stage(receipt_path, "prepared", "target_open", operation)


def open_source(receipt_path):
    def operation(record):
        view = _open(record, "source")
        rig = _rig(view.domain_scene())
        return {"objects": len(rig["objects"]), "tracks": len(rig["tracks"])}

    return _stage(receipt_path, "target_open", "source_open", operation)


def import_fbx(receipt_path):
    def operation(record):
        generation = record["generation"]
        if _sha(generation["fbx_path"]) != generation["sha256"]:
            raise RuntimeError("Generated FBX changed since preparation")
        csc, app, manager = _application()
        view = _find_view(record, "source")
        manager.set_current_scene(view)
        loader = app.get_tools_manager().get_tool("FbxSceneLoader").get_fbx_loader(view)
        settings = csc.fbx.FbxSettings()
        settings.bake_animation = True
        settings.apply_euler_filter = True
        loader.set_settings(settings)
        loader.import_animation(generation["fbx_path"].replace("\\", "/"), 0)
        stored = int(view.domain_scene().data_viewer().get_animation_size())
        if stored < generation["num_frames"]:
            raise RuntimeError(
                f"FBX import stored {stored} frames, expected at least {generation['num_frames']}"
            )
        return {"stored_frames": stored, "fbx_sha256": generation["sha256"]}

    return _stage(receipt_path, "source_open", "fbx_imported", operation)


def retarget_copy(receipt_path):
    def operation(record):
        _, app, _ = _application()
        _select(_find_view(record, "source"), record["generation"]["num_frames"])
        app.get_action_manager().call_action("View.Retargeting_Copy")

    return _stage(receipt_path, "fbx_imported", "retarget_copied", operation)


def prepare_target_interval(receipt_path):
    def operation(record):
        csc, _, manager = _application()
        view = _find_view(record, "target")
        manager.set_current_scene(view)
        scene = view.domain_scene()
        rig = _rig(scene)
        count = record["generation"]["num_frames"]

        def allocate(model, _update, _updater, _session):
            editor = model.layers_editor()
            editor.set_section(csc.layers.layer.Section(), count, rig["tracks"][0])
            model.fit_animation_size_by_layers()
            for track in rig["tracks"]:
                for frame in range(count):
                    editor.set_fixed_interpolation_or_key_if_need(track, frame, True)
            editor.normalize_sections(scene)

        _modify(scene, "Prepare Cascy retarget keys", allocate)
        boundary = view.animation_boundary()
        boundary.first_frame, boundary.last_frame = 0, count - 1
        boundary.first_visible_frame, boundary.last_visible_frame = 0, count - 1
        record["baseline"] = _snapshot(view, count)
        return _summary(record["baseline"])

    return _stage(receipt_path, "retarget_copied", "target_ready", operation)


def retarget_paste(receipt_path):
    def operation(record):
        _, app, _ = _application()
        view = _find_view(record, "target")
        count = record["generation"]["num_frames"]
        _select(view, count)
        app.get_action_manager().call_action("View.Retargeting_Paste")
        snapshot = _snapshot(view, count)
        if snapshot["fingerprint"] != record["target_fingerprint"]:
            raise RuntimeError("Retargeting changed Cascy topology or semantic bindings")
        if snapshot["root_distance"] < 1.0:
            raise RuntimeError("Retargeted clip has no measurable forward root translation")
        record["expected"] = snapshot
        view.domain_scene().set_current_frame(0)
        return _summary(snapshot)

    return _stage(receipt_path, "target_ready", "retargeted", operation)


def save_result(receipt_path):
    def operation(record):
        view = _find_view(record, "target")
        output = _owned_path(record, "result")
        if output.exists():
            raise RuntimeError("The result path already exists; inspect instead of overwriting it")
        view.save(str(output))
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Cascadeur did not write a nonempty result")
        record["active_target"] = "result"
        record["result_sha256"] = _sha(output)
        record["result_settings"] = scene_settings(output)
        record["expected"] = _snapshot(view, record["generation"]["num_frames"])
        return {"path": str(output), "sha256": record["result_sha256"]}

    return _stage(receipt_path, "retargeted", "saved", operation)


def compare_snapshots(expected, observed, tolerance=POSITION_TOLERANCE):
    if (
        expected["fingerprint"] != observed["fingerprint"]
        or expected["positions"].keys() != observed["positions"].keys()
    ):
        raise RuntimeError("Reopened Cascy topology or Point bindings changed")
    maximum = 0.0
    for name, frames in expected["positions"].items():
        actual = observed["positions"][name]
        if len(frames) != len(actual):
            raise RuntimeError("Reopened Cascy has a different motion length")
        for before, after in zip(frames, actual):
            if len(before) != 3 or len(after) != 3:
                raise RuntimeError("Reopen comparison requires complete Point coordinates")
            for left, right in zip(before, after):
                if not math.isfinite(left) or not math.isfinite(right):
                    raise RuntimeError("Reopen comparison encountered a nonfinite coordinate")
                maximum = max(maximum, abs(left - right))
    if maximum > tolerance:
        raise RuntimeError(f"Reopened Point coordinates changed by {maximum:g}, tolerance {tolerance:g}")
    return {"point_coordinates_verified": True, "max_coordinate_error": maximum, "tolerance": tolerance}


def reopen_prepare(receipt_path):
    """Activate a retained source tab; do not close any scene in this callback."""

    def operation(record):
        _, _, manager = _application()
        source = _find_view(record, "source")
        manager.set_current_scene(source)
        return {
            "anchor_path": source.get_path_name(),
            "next_step": "reopen_result in a separate native request",
        }

    return _stage(receipt_path, "saved", "reopen_ready", operation)


def reopen_result(receipt_path, *, callback_scene=None):
    def operation(record):
        source = _find_view(record, "source")
        _require_callback_anchor(source, callback_scene)
        if _sha(_owned_path(record, "result")) != record["result_sha256"]:
            raise RuntimeError("Saved result changed before reopen verification")
        owned = _find_view(record, "result")
        _close_inactive_view(owned, callback_scene)
        observed = _snapshot(_open(record, "result"), record["generation"]["num_frames"])
        comparison = compare_snapshots(record["expected"], observed)
        record["reopen_verification"] = comparison
        return {**comparison, **_summary(observed)}

    return _stage(receipt_path, "reopen_ready", "verified", operation)


def inspect(receipt_path):
    receipt, record = _read(receipt_path)
    _, _, manager = _application()
    result = {
        "receipt_path": str(receipt),
        "phase": record["phase"],
        "error": record.get("error"),
        "tabs": _tabs(manager),
        "generation": record["generation"],
        "before_unchanged": _sha(_owned_path(record, "before")) == record["before_sha256"],
        "reopen_verification": record.get("reopen_verification"),
        "native_settings": record.get("native_settings"),
        "result_settings": record.get("result_settings"),
    }
    if record.get("expected"):
        result["motion"] = _summary(record["expected"])
    result["diagnostics"] = diagnose(receipt_path)
    return result


def diagnose(receipt_path):
    """Read source FK, target Point motion and UI/application contexts without writes."""
    _, record = _read(receipt_path)
    _, app, manager = _application()
    active = manager.current_scene()
    application_view = app.current_scene()
    result = {
        "pro_features_available": bool(app.is_pro_features_available()),
        "export_available": bool(app.is_export_available()),
        "manager_current": active.get_path_name() if active is not None else None,
        "application_current": application_view.get_path_name() if application_view is not None else None,
        "scenes": {},
    }
    count = record["generation"]["num_frames"]
    for kind in ("source", record.get("active_target", "target")):
        try:
            view = _find_view(record, kind)
            scene = view.domain_scene()
            model, data, behaviours = scene.model_viewer(), scene.data_viewer(), scene.behaviour_viewer()
            stored = int(data.get_animation_size())
            frames = sorted({0, min(count // 2, stored - 1), min(count - 1, stored - 1)})
            observations = []
            point_count = 0
            wanted = {"hips", "root", "pelvis", "leftfoot", "rightfoot", "foot_l", "foot_r"}
            for obj in model.get_objects():
                name, object_type = model.get_object_name(obj), model.get_object_type_name(obj)
                if object_type == "Point":
                    point_count += 1
                    keep = point_count <= 6 or any(token in name.lower() for token in ("pelvis", "foot"))
                else:
                    keep = object_type in {"Joint", "Transform"} and name.split(":")[-1].lower() in wanted
                if not keep:
                    continue
                transform = behaviours.get_behaviour_by_name(obj, "Transform")
                if transform.is_null():
                    continue
                channels = {}
                for channel in ("global_position", "local_position"):
                    identifier = behaviours.get_behaviour_data(transform, channel)
                    if not identifier.is_null() and data.has_data(identifier):
                        channels[channel] = [
                            _vector(data.get_data_value(identifier, frame)) for frame in frames
                        ]
                observations.append({"name": name, "type": object_type, "channels": channels})
            selection = scene.get_layers_selector().selection().frames_interval()
            selected = list(scene.selector().selected().ids)
            result["scenes"][kind] = {
                "path": view.get_path_name(),
                "stored_frames": stored,
                "frames": frames,
                "native_points": point_count,
                "observations": observations,
                "selected_count": len(selected),
                "selected_names": [model.get_object_name(obj) for obj in selected[:16]],
                "selected_frames": [int(selection.first), int(selection.last)],
                "viewport_count": len(view.viewports()),
            }
            if kind != "source":
                observed = _snapshot(view, count)
                baseline = record.get("baseline")
                if baseline:
                    maximum = max(
                        abs(a - b)
                        for name, rows in baseline["positions"].items()
                        for before, after in zip(rows, observed["positions"][name])
                        for a, b in zip(before, after)
                    )
                    result["scenes"][kind]["max_point_change_from_baseline"] = maximum
                result["scenes"][kind]["motion"] = _summary(observed)
        except Exception as error:
            result["scenes"][kind] = {"inspection_error": f"{type(error).__name__}: {error}"}
    return result


def cleanup(receipt_path, *, callback_scene=None):
    def operation(record):
        _, _, manager = _application()
        result = _find_view(record, "result")
        _require_callback_anchor(result, callback_scene)
        source = _find_view(record, "source")
        # FBX import edits this disposable source; save it so close_scene needs no save dialog.
        source.save(str(_owned_path(record, "source")))
        _close_inactive_view(source, callback_scene)
        deleted = []
        for kind in ("source", "target"):
            path = _owned_path(record, kind)
            if path.is_file():
                path.unlink()
                deleted.append(path.name)
        record.pop("baseline", None)
        record.pop("expected", None)
        return {"deleted": deleted, "retained": ["before.casc", "result.casc", "session.json"]}

    return _stage(receipt_path, "verified", "completed", operation)
