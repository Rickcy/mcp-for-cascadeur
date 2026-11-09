"""Build-specific, incremental drivers for the native animation tools.

The job scheduler owns persistence, the checkpoint and exclusive scene access.
No QObject or native scene reference is stored in a plan. A queued action is
never replayed when its in-process observation has been lost.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from collections import deque
from pathlib import Path

from . import animation, common, state

BUILD = {"git_version": "2026.2.3", "git_count": "44356"}
ROUTES = {
    "motion_generation": "View.Inbetweening_RunMotionGeneration",
    "mocap": "View.MocapTool",
    "unbake": "View.Animation unbaking",
    "autophysics": "AutoPhysicsTool.Switch Auto Physics",
}
COMPLETION = {
    "motion_generation": "Root motion: done",
    "mocap": "All frames have been processed",
    "unbake": "Auto keys: done",
    "autophysics": "Command: Snap to autophysics",
}
MODEL_SHA256 = "a151cd3a69f550efc28015ab8b7e62dd1c3b2b581dd0e1edb26c4b31e0690888"
MODEL_BYTES = 825_323_915
_callbacks: dict[str, dict] = {}


def _completed(operation, log_text):
    markers = [COMPLETION[operation]]
    if operation == "unbake":
        markers.append("Auto posing.Auto unlock: done")
    if operation == "autophysics":
        markers.append("Autophysics core finished")
    return all(marker in log_text for marker in markers)


class NativeOperationError(RuntimeError):
    execution_state = "not_started"

    def __init__(self, code, message, *, unknown=False, details=None, safe_to_rollback=False):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.safe_to_rollback = safe_to_rollback
        if safe_to_rollback:
            self.execution_state = "verification_failed"
        elif unknown:
            self.execution_state = "unknown"


def _fail(code, message, **kwargs):
    raise NativeOperationError(code, message, **kwargs)


def _native_build(csc):
    result = {key: str(getattr(csc.SystemVariables, key)()) for key in BUILD}
    if result != BUILD:
        _fail(
            "NATIVE_BUILD_UNVERIFIED",
            "Native workflow routes are accepted only on build 44356 (2026.2.3)",
            details={"expected": BUILD, "actual": result},
        )
    return result


def _identity(view):
    path = str(view.get_path_name())
    if not path:
        _fail("NATIVE_SCENE_UNSAVED", "Save the scene before starting a native operation")
    return {"path": str(Path(path).resolve()), "name": str(view.name())}


def _data_root():
    return (
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Nekki Limited/Cascadeur"
    )


def _log_cursor():
    path = _data_root() / "logs/cascadeur_log.log"
    if not path.is_file():
        _fail("NATIVE_LOG_UNAVAILABLE", "The native operation log is required to verify completion")
    return {"path": str(path), "offset": path.stat().st_size}


def _read_log(cursor):
    path = Path(cursor["path"])
    size = path.stat().st_size
    if size < cursor["offset"]:
        _fail(
            "NATIVE_LOG_REPLACED", "The native log was truncated; completion cannot be inferred", unknown=True
        )
    with path.open("rb") as stream:
        stream.seek(cursor["offset"])
        content = stream.read(262144)
        offset = stream.tell()
    return content.decode("utf8", errors="replace"), {**cursor, "offset": offset}


def _model_ready():
    path = _data_root() / "resources/mocap.pt"
    if not path.is_file() or path.stat().st_size != MODEL_BYTES:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256


def _snapshot(scene, rig):
    layers = scene.layers_viewer()
    return {
        "count": animation._size(scene),
        "positions": {
            identifier: [
                animation._read(scene, obj, frame, "global")["position"]
                for frame in range(animation._size(scene))
            ]
            for identifier, obj in sorted(rig.controls.items())
        },
        "tracks": {
            str(identifier): {
                "objects": sorted(str(obj) for obj in layers.layer(identifier).obj_ids),
                "sections": {
                    str(frame): state.section_state(section)
                    for frame, section in layers.layer(identifier).sections.items()
                },
            }
            for identifier in layers.all_layer_ids()
        },
    }


def _guard_plan(scene, rig, selected, baseline):
    """Stage missing finger-track terminal keys and exact authored rotations."""
    from .data_channels import read_data_value

    terminal = baseline["count"] - 1
    fingers = {}
    for hand in rig.fingers.values():
        for binding in hand.values():
            fingers.setdefault(str(binding.track_id), {})[str(binding.rotation_data)] = binding.rotation_data
    anchors, sections, channels = {}, {}, {}
    layers = scene.layers_viewer()
    for identifier in selected:
        if layers.layer(identifier).is_key(terminal):
            continue
        track = str(identifier)
        if track not in fingers:
            _fail(
                "NATIVE_GUARD_ANCHOR_UNSUPPORTED",
                "Unbaking requires existing terminal keys on non-finger tracks",
            )
        originals = baseline["tracks"][track]["sections"]
        keys = [int(frame) for frame, section in originals.items() if section["key"] is not None]
        if not keys:
            _fail(
                "NATIVE_GUARD_ANCHOR_UNSUPPORTED",
                "A finger guard needs an existing native key as its metadata source",
            )
        descriptor = copy.deepcopy(originals[str(max(keys))])
        descriptor.update(interpolation="STEP", ai_description="")
        anchors[track], sections[track] = terminal, descriptor
        data = scene.data_viewer()
        for key, reference in fingers[track].items():
            mode = getattr(data.get_data(reference).mode, "name", "")
            if mode != "Animation":
                _fail(
                    "NATIVE_GUARD_ANCHOR_UNSUPPORTED", "Finger guard inputs must be animated native rotations"
                )
            value = state.encode(read_data_value(data, reference, terminal))
            if (
                not isinstance(value, dict)
                or value.get("__native__") != "rotation"
                or "euler_radians" not in value
            ):
                _fail(
                    "NATIVE_GUARD_ANCHOR_UNSUPPORTED",
                    "Finger guard rotations must preserve their native Euler branch",
                )
            channels[key] = {"track": track, "frame": terminal, "value": value}
    return anchors, sections, channels


def _settings_padding(checkpoint, plan):
    """Plan source-held settings only for protected newly allocated frames."""
    old_count, count = checkpoint["count"], plan["expected_count"]
    owners = set(plan["owned_objects"])
    result = []
    for key, item in checkpoint["channels"].items():
        if item["kind"] != "setting" or item["mode"] != "Animation":
            continue
        value = item["values"][old_count - 1]
        if type(value) not in (bool, int):
            _fail(
                "NATIVE_SETTING_PADDING_UNSUPPORTED",
                "Animated settings must retain a native bool or integer value",
            )
        for frame in range(old_count, count):
            if item["owner"] not in owners or not plan["first"] <= frame <= plan["last"]:
                result.append(
                    {"key": key, "id": item["id"], "owner": item["owner"], "frame": frame, "value": value}
                )
    return result


def _hold_extended_settings(scene, plan, *, after_native=False):
    """Restore only planned padding settings; never solve the terminal frame."""
    from . import recovery
    from .data_channels import read_setting_value

    source = recovery.active_checkpoint(scene)
    if source is None:
        _fail(
            "NATIVE_SETTING_PADDING_CHECKPOINT_REQUIRED",
            "Timeline padding requires the job's original durable checkpoint",
        )
    if animation._size(scene) != plan["expected_count"]:
        _fail(
            "NATIVE_SETTING_PADDING_SIZE_CHANGED",
            "Timeline size changed before settings padding",
            unknown=True,
        )
    planned = _settings_padding(source, plan)
    data = scene.data_viewer()
    objects = {str(item): item for item in scene.model_viewer().get_objects()}
    references = {}
    for owner in {item["owner"] for item in planned}:
        if owner not in objects:
            _fail(
                "NATIVE_SETTING_PADDING_OWNER_CHANGED", "An original setting owner disappeared", unknown=True
            )
        references[owner] = {str(item): item for item in data.get_all_settings_id(objects[owner])}
    changed = []
    for item in planned:
        reference = references[item["owner"]].get(item["id"])
        if reference is None or getattr(data.get_setting(reference).mode, "name", "") != "Animation":
            _fail(
                "NATIVE_SETTING_PADDING_CHANNEL_CHANGED",
                "An original animated setting changed identity or mode",
                unknown=True,
            )
        value = read_setting_value(data, reference, item["frame"])
        if type(value) is not type(item["value"]) or value != item["value"]:
            changed.append((item, reference))
    if not changed:
        return {"verified": True, "planned_samples": len(planned), "changed_samples": 0}
    # A cached allocation snapshot is valid before native dispatch. After a
    # worker completes, read fresh storage rather than using that stale cache.
    before = None if after_native else recovery.latest_committed_checkpoint(scene)
    if before is None:
        before = state.capture(scene)
    expected = copy.deepcopy(before)
    for item, _reference in changed:
        expected["channels"][item["key"]]["values"][item["frame"]] = item["value"]

    def apply(model, _updater):
        editor = model.data_editor()
        for item, reference in changed:
            editor.set_setting_value(reference, item["frame"], item["value"])

    common.modify(scene, "Hold settings in extended timeline padding", apply)
    after = recovery.latest_committed_checkpoint(scene) or state.capture(scene)
    comparison = state.compare(expected, after)
    if not comparison["equivalent"]:
        _fail(
            "NATIVE_SETTING_PADDING_CHANGED_VALUES",
            "Restoring timeline settings changed other protected state",
            details=comparison,
            safe_to_rollback=True,
        )
    return {
        "verified": True,
        "planned_samples": len(planned),
        "changed_samples": len(changed),
        "comparison": comparison,
    }


def _interval_mode_context():
    """Resolve the timeline component that imports the actual native enum."""
    roots = [item for item in _roots("IntervalEditModeButton") if item.property("visible") is not False]
    if len(roots) != 1:
        _fail(
            "NATIVE_INTERVAL_MODE_UNAVAILABLE", "Expected one visible native IntervalEditModeButton context"
        )
    try:
        values = json.loads(
            str(
                _eval(
                    roots[0],
                    "JSON.stringify({mode:viewModel.domainState.editMode,selection_mode:DomainStateController.SelectionEditMode})",
                )
            )
        )
    except (NativeOperationError, ValueError, TypeError, RuntimeError) as error:
        _fail(
            "NATIVE_INTERVAL_MODE_UNAVAILABLE",
            "The native timeline edit-mode context could not be read before dispatch",
            details={"cause": str(error)},
        )
    if (
        not isinstance(values, dict)
        or type(values.get("mode")) is not int
        or type(values.get("selection_mode")) is not int
    ):
        _fail(
            "NATIVE_INTERVAL_MODE_UNAVAILABLE", "Native interval edit mode and enum could not be read exactly"
        )
    return roots[0], values


def _interval_mode_state():
    return _interval_mode_context()[1]


def _assert_physics_scope(plan):
    expected = {"first": plan["first"], "last": plan["last"]}
    timeline, mode = animation.timeline_get(), _interval_mode_state()
    if timeline.get("playback") != expected or timeline.get("visible") != expected:
        _fail(
            "NATIVE_PHYSICS_BOUNDS_MISMATCH",
            "Native AutoPhysics playback/visible bounds differ from the requested interval",
            details={
                "expected": expected,
                "playback": timeline.get("playback"),
                "visible": timeline.get("visible"),
            },
        )
    if mode["mode"] != mode["selection_mode"]:
        _fail(
            "NATIVE_PHYSICS_MODE_MISMATCH",
            "Native AutoPhysics requires the observed SelectionEditMode before dispatch",
            details=mode,
        )
    return {"verified": True, "playback": expected, "visible": expected, "interval_mode": mode}


def _prepare_physics_scope(plan):
    expected = {"first": plan["first"], "last": plan["last"]}
    actual = animation.timeline_set_range(plan["first"], plan["last"], visible=True)
    if actual.get("playback") != expected or actual.get("visible") != expected:
        _fail(
            "NATIVE_PHYSICS_BOUNDS_MISMATCH",
            "Setting native AutoPhysics bounds did not read back exactly",
            details={"expected": expected, "actual": actual},
        )
    item, mode = _interval_mode_context()
    if mode["mode"] != mode["selection_mode"]:
        # Use the installed timeline's slot only when needed. Reacquire the
        # visual context after the call instead of keeping a stale wrapper.
        try:
            _eval(item, "viewModel.domainState.toggleEditMode(DomainStateController.SelectionEditMode)")
        except (NativeOperationError, RuntimeError) as error:
            _fail(
                "NATIVE_INTERVAL_MODE_UNAVAILABLE",
                "The native timeline edit-mode slot failed before AutoPhysics dispatch",
                details={"cause": str(error)},
            )
    return _assert_physics_scope(plan)


def _anchor_guards(scene, rig, plan):
    """Insert planned guard keys, then prove every original model value survived.

    The terminal frame is storage, not a solver work frame: no run_update or
    interpolator invocation is made there. Only validated authored finger-local
    rotations are written; driven/global/local output channels are never mixed.
    """
    from . import recovery
    from .data_channels import read_data_value

    anchors = plan["guard_anchors"]
    identifiers = {str(item): item for item in animation._tracks(scene, list(anchors))}
    references = {
        str(binding.rotation_data): binding.rotation_data
        for hand in rig.fingers.values()
        for binding in hand.values()
    }
    staged = []
    for key, item in plan["guard_anchor_channels"].items():
        if key not in references:
            _fail("NATIVE_GUARD_INPUT_CHANGED", "A staged finger rotation no longer belongs to the character")
        reference = references[key]
        actual = state.encode(read_data_value(scene.data_viewer(), reference, item["frame"]))
        if not state.equivalent_values(item["value"], actual, exact=True):
            _fail("NATIVE_GUARD_INPUT_CHANGED", "A staged finger guard value changed before anchoring")
        staged.append((reference, item["frame"], state.decode(item["value"])))
    for descriptor in plan["guard_anchor_sections"].values():
        state.validate_section_metadata(descriptor)
    before = recovery.latest_committed_checkpoint(scene) or recovery.active_checkpoint(scene)
    if before is None:
        before = state.capture(scene)
    expected = copy.deepcopy(before)
    for track, frame in anchors.items():
        expected["tracks"][track]["keys"] = sorted(set(expected["tracks"][track]["keys"]) | {frame})
        expected["tracks"][track]["sections"][str(frame)] = copy.deepcopy(
            plan["guard_anchor_sections"][track]
        )

    def apply(model, updater):
        editor = model.layers_editor()
        for track, frame in anchors.items():
            editor.set_fixed_interpolation_or_key_if_need(identifiers[track], frame, True)
        editor.normalize_sections(scene)
        model.fit_animation_size_by_layers()
        updater.generate_update()
        for track, frame in anchors.items():
            # Preserve the old curve metadata exactly, including any previous
            # terminal section that the native key helper normalized.
            for old_frame, descriptor in before["tracks"][track]["sections"].items():
                if int(old_frame) == frame:
                    continue
                editor.change_section(
                    int(old_frame),
                    identifiers[track],
                    lambda section, value=descriptor: state.apply_section_metadata(section, value),
                )
            descriptor = plan["guard_anchor_sections"][track]
            editor.change_section(
                frame,
                identifiers[track],
                lambda section, value=descriptor: state.apply_section_metadata(section, value),
            )
        data_editor = model.data_editor()
        for reference, frame, value in staged:
            data_editor.set_data_value(reference, frame, value)

    common.modify(scene, "Anchor native unbaking guard keys", apply)
    after = recovery.latest_committed_checkpoint(scene) or state.capture(scene)
    comparison = state.compare(expected, after)
    if not comparison["equivalent"]:
        _fail(
            "NATIVE_GUARD_ANCHOR_CHANGED_VALUES",
            "Guard anchoring changed protected model values or metadata",
            details=comparison,
            safe_to_rollback=True,
        )
    return {
        "verified": True,
        "anchors": anchors,
        "channels_verified": len(before["channels"]),
        "stored_frames_verified": before["count"],
        "comparison": comparison,
    }


def prepare(operation, arguments):
    """Validate a native route and return a JSON-only immutable plan; no writes."""
    from . import characters, curves, playback

    if operation not in ROUTES or type(arguments) is not dict:
        _fail("NATIVE_OPERATION_INVALID", "Choose motion_generation, mocap, unbake or autophysics")
    allowed = {
        "character_id",
        "first",
        "last",
        "tracks",
        "video_path",
        "reference_object",
        "video_first",
        "allow_model_download",
        "apply_mode",
        "position_tolerance",
        "extend_timeline",
    }
    if set(arguments) - allowed:
        _fail("NATIVE_ARGUMENT_INVALID", f"Unknown native arguments: {sorted(set(arguments) - allowed)}")
    csc, _, view, scene = common.current_context()
    build = _native_build(csc)
    playback.require_idle()
    identity = _identity(view)
    count = animation._size(scene)
    if not 1 <= count <= state.MAX_FRAMES:
        _fail("NATIVE_FRAME_LIMIT", f"Native jobs require 1-{state.MAX_FRAMES} checkpointable stored frames")
    first, last = arguments.get("first"), arguments.get("last")
    extend = arguments.get("extend_timeline", operation == "mocap")
    if type(extend) is not bool or (extend and operation != "mocap"):
        _fail("NATIVE_ARGUMENT_INVALID", "Timeline extension is supported only for Mocap")
    if type(first) is not int or type(last) is not int or not 0 <= first < last < state.MAX_FRAMES - 1:
        _fail(
            "NATIVE_INTERVAL_INVALID",
            "Use first < last within the checkpoint frame budget, including a terminal guard",
        )
    if not extend and last >= count - 1:
        _fail("NATIVE_INTERVAL_INVALID", "The interval requires stored frames and a following terminal guard")
    expected_count = max(count, last + 2) if extend else count
    rig = characters.resolve_character(
        scene, arguments.get("character_id"), require_semantics=True, require_hands=operation == "unbake"
    )
    available = curves.character_tracks(scene, rig)
    requested = arguments.get("tracks")
    if requested is None:
        selected = list(available)
    else:
        selected = animation._tracks(scene, requested)
    if not selected or set(selected) - set(available):
        _fail("NATIVE_TRACK_SCOPE_INVALID", "Choose unlocked tracks owned exclusively by the character")
    point_tracks = [
        identifier
        for identifier in selected
        if set(map(str, available[identifier].obj_ids)).intersection(rig.controls)
    ]
    selected_controls = sorted(
        {str(obj) for identifier in selected for obj in available[identifier].obj_ids}.intersection(
            rig.controls
        )
    )
    if not selected_controls:
        _fail("NATIVE_TRACK_SCOPE_INVALID", "The operation needs native Point-control tracks")
    if operation == "motion_generation":
        for identifier in point_tracks:
            if not available[identifier].is_key(first) or not available[identifier].is_key(last):
                _fail(
                    "NATIVE_ENDPOINT_KEYS_REQUIRED",
                    "Motion Generation needs two existing keyed endpoint poses on each selected track",
                )
    if operation in {"mocap", "motion_generation", "autophysics"} and set(selected_controls) != set(
        rig.controls
    ):
        _fail(
            "NATIVE_PARTIAL_CHARACTER_UNVERIFIED",
            "This native operation requires the complete character Point-track scope",
        )
    tolerance = arguments.get("position_tolerance", 0.01)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or not 0 < tolerance <= 1
    ):
        _fail("NATIVE_ARGUMENT_INVALID", "position_tolerance must be finite and in (0, 1] scene units")
    if type(arguments.get("allow_model_download", False)) is not bool:
        _fail("NATIVE_ARGUMENT_INVALID", "allow_model_download must be boolean")
    if operation != "mocap" and set(arguments).intersection(
        {"video_path", "reference_object", "video_first", "allow_model_download"}
    ):
        _fail("NATIVE_ARGUMENT_INVALID", "Video and download arguments belong to mocap")
    if arguments.get("apply_mode", "all_frames") not in {"all_frames", "only_keys"}:
        _fail("NATIVE_ARGUMENT_INVALID", "apply_mode must be all_frames or only_keys")
    if operation != "autophysics" and "apply_mode" in arguments:
        _fail("NATIVE_ARGUMENT_INVALID", "apply_mode belongs to autophysics")
    args = copy.deepcopy(arguments)
    args.update(first=first, last=last, character_id=rig.id, position_tolerance=float(tolerance))
    if operation == "mocap":
        video = args.get("video_path")
        reference = args.get("reference_object")
        if bool(video) == bool(reference):
            _fail("NATIVE_VIDEO_REQUIRED", "Provide exactly one video_path or an existing reference_object")
        if video:
            path = common.workspace_path(video, must_exist=True, suffixes=(".mp4",))
            args["video_path"] = str(path)
            source_first = args.get("video_first", 0)
            if type(source_first) is not int or source_first < 0:
                _fail("NATIVE_ARGUMENT_INVALID", "video_first must be a nonnegative source-frame index")
            args["video_first"] = source_first
        else:
            args["reference_object"] = str(common.resolve_object(scene, reference))
        ready = _model_ready()
        if not ready and not args.get("allow_model_download", False):
            _fail(
                "NATIVE_MODEL_REQUIRED",
                "Install the official verified Mocap model, or explicitly allow its native download",
            )
    else:
        ready = None
    baseline = _snapshot(scene, rig)
    guard_anchors, guard_sections, guard_channels = {}, {}, {}
    if operation == "unbake" and not any(
        section["interpolation"] == "FIXED"
        for identifier in map(str, selected)
        for frame, section in baseline["tracks"][identifier]["sections"].items()
        if first <= int(frame) < last
    ):
        _fail("NATIVE_BAKED_INPUT_REQUIRED", "Unbaking requires a FIXED interval in the selected scope")
    if operation == "unbake":
        guard_anchors, guard_sections, guard_channels = _guard_plan(scene, rig, selected, baseline)
    # Refuse pre-existing workflow dialogs so a job cannot claim another request's UI.
    ui = _ui_snapshot()
    if ui["mocap"] or ui["video"] or ui["apply"] or ui["download"]:
        _fail("NATIVE_DIALOG_ALREADY_OPEN", "Close existing native workflow dialogs before starting a job")
    original_scope = None
    if operation == "autophysics":
        original_scope = {"interval_edit_mode": None, "timeline_bounds": None}
        try:
            original_scope["interval_edit_mode"] = _interval_mode_state()
        except NativeOperationError as error:
            original_scope["interval_mode_unavailable"] = str(error)
        if callable(getattr(view, "animation_boundary", None)):
            bounds = view.animation_boundary()
            original_scope["timeline_bounds"] = {
                "playback": {"first": bounds.first_frame, "last": bounds.last_frame},
                "visible": {"first": bounds.first_visible_frame, "last": bounds.last_visible_frame},
            }
    return {
        "schema_version": 1,
        "operation": operation,
        "arguments": args,
        "build": build,
        "scene_identity": identity,
        "host_pid": os.getpid(),
        "character_id": rig.id,
        "tracks": sorted(map(str, selected)),
        "point_tracks": sorted(map(str, point_tracks)),
        "owned_objects": sorted(map(str, rig.owned)),
        "controls": selected_controls,
        "first": first,
        "last": last,
        "phase": "allocate"
        if expected_count > count
        else ("physics_scope" if operation == "autophysics" else "scope"),
        "original_operational_state": original_scope,
        "expected_count": expected_count,
        "baseline": baseline,
        "guard_anchors": guard_anchors,
        "guard_anchor_sections": guard_sections,
        "guard_anchor_channels": guard_channels,
        "log_cursor": _log_cursor(),
        "model_ready": ready,
        "object_ids": sorted(str(obj) for obj in scene.model_viewer().get_objects()),
        "capabilities": {
            "cancel_before_dispatch": True,
            "cancel_running": False,
            "request_cancel_running": operation == "mocap",
            "cancellation_requires_observation": True,
            "native_video_encoder": False,
        },
    }


def _qt():
    from PySide6.QtCore import QMetaObject, Qt, QTimer
    from PySide6.QtQml import QQmlEngine, QQmlExpression
    from PySide6.QtWidgets import QApplication, QFileDialog

    return QApplication, QFileDialog, QMetaObject, Qt, QTimer, QQmlEngine, QQmlExpression


def _walk(item):
    pending, count = deque([item]), 0
    while pending and count < 10000:
        current = pending.popleft()
        count += 1
        yield current
        children = getattr(current, "childItems", None)
        pending.extend(children() if callable(children) else current.children())


def _windows():
    from PySide6.QtGui import QGuiApplication

    # Register QtQuick wrapper types before enumerating C++ windows. Otherwise
    # PySide6 can expose a real QQuickView as QWindow without contentItem().
    importlib.import_module("PySide6.QtQuick")
    # Embedded applications may expose instance() as a QCoreApplication Python
    # wrapper. The static GUI API does not depend on that wrapper's methods.
    return [window for window in QGuiApplication.allWindows() if window.isVisible()]


def _class(item):
    return str(item.metaObject().className())


def _roots(kind):
    found = []
    for window in _windows():
        content = getattr(window, "contentItem", None)
        if not callable(content):
            continue
        for item in _walk(content()):
            if kind in _class(item):
                found.append(item)
    return found


def _eval(item, expression):
    *_, engine, expression_type = _qt()
    context = engine.contextForObject(item)
    if context is None:
        _fail("NATIVE_QML_CONTEXT_MISSING", "The live native QML context is unavailable", unknown=True)
    expression_object = expression_type(context, item, expression)
    result = expression_object.evaluate()
    if expression_object.hasError():
        _fail("NATIVE_QML_ERROR", expression_object.error().toString(), unknown=True)
    return result[0] if isinstance(result, tuple) and len(result) == 2 else result


def _qml_read(item, names):
    entries = ",".join(f"{json.dumps(name)}:viewModel[{json.dumps(name)}]" for name in names)
    value = _eval(item, "JSON.stringify({" + entries + "})")
    return json.loads(str(value))


def _buttons(window):
    content = getattr(window, "contentItem", None)
    if not callable(content):
        return []
    return [
        (str(item.property("text")), item)
        for item in _walk(content())
        if "Button" in _class(item)
        and item.property("visible") is not False
        and item.property("enabled") is not False
        and item.property("text")
    ]


def _window_texts(window):
    content = getattr(window, "contentItem", None)
    if not callable(content):
        return []
    return [str(value) for item in _walk(content()) if (value := item.property("text"))]


def _ui_snapshot():
    import re

    mocap, video, apply, download = [], [], [], []
    confirmations = []
    for item in _roots("MocapDialog"):
        mocap.append(_qml_read(item, ("state", "received", "total")))
    for item in _roots("LoadVideoDialog"):
        video.append(
            _qml_read(
                item, ("sourceFile", "from", "to", "nbFrames", "isInProgress", "progress", "canDecompress")
            )
        )
    for window in _windows():
        title = str(window.title())
        texts = [text for text, _ in _buttons(window)]
        if title == "Apply AutoPhysics":
            apply.append({"title": title, "buttons": texts})
        content = " ".join([title, *_window_texts(window)]).casefold()
        confirmation = re.search(
            r"\bvideo[- ]mocap\s+for\s+the\s+selected\s+interval\s+of\s+(\d+)\s+frames?"
            r"\s+will\s+take\s+approximately\s+[^?]{1,120}\?\s*",
            content,
        )
        if confirmation and "proceed?" in confirmation.group(0):
            confirmations.append({"title": title, "frames": int(confirmation.group(1)), "buttons": texts})
        if "download" in content and ("mocap" in content or "787" in content):
            download.append({"title": title, "buttons": texts})
    return {
        "mocap": mocap,
        "video": video,
        "apply": apply,
        "download": download,
        "mocap_confirmations": confirmations,
    }


def _queued_click(title, text):
    _, _, meta, qt, *_ = _qt()
    matches = [
        item
        for window in _windows()
        if str(window.title()) == title
        for label, item in _buttons(window)
        if label.strip() == text
    ]
    if len(matches) != 1:
        _fail(
            "NATIVE_DIALOG_AMBIGUOUS",
            f"Expected exactly one visible {text!r} button in {title!r}",
            unknown=True,
        )
    if not meta.invokeMethod(matches[0], "clicked", qt.QueuedConnection):
        _fail("NATIVE_QUEUE_REJECTED", "Qt rejected the queued dialog confirmation", unknown=True)
    # Do not inspect the wrapper after scheduling a callback that may destroy it.


def _job_id(record):
    identifier = record.get("job_id") or record.get("id") or record.get("request_id")
    if type(identifier) is not str or not identifier:
        _fail("NATIVE_JOB_INVALID", "The scheduler must provide a stable job_id")
    return identifier


def _context(plan):
    csc, app, view, scene = common.current_context()
    _native_build(csc)
    if plan["host_pid"] != os.getpid() or plan["scene_identity"] != _identity(view):
        _fail(
            "NATIVE_JOB_CONTEXT_LOST",
            "The native process or saved scene changed; resolve the durable receipt",
            unknown=True,
        )
    return app, view, scene


def _queue(record, token, callback):
    identifier = _job_id(record) + ":" + token
    if identifier in _callbacks:
        _fail("NATIVE_DISPATCH_DUPLICATE", "A native stage must not be replayed", unknown=True)
    observation = {"status": "queued", "token": token}
    _callbacks[identifier] = observation

    def execute():
        observation["status"] = "started"
        try:
            app, view, scene = _context(record["plan"])
            callback(app, view, scene)
        except Exception as error:
            observation.update(
                status="failed", error=str(error), code=getattr(error, "code", "NATIVE_CALLBACK_FAILED")
            )
        else:
            observation["status"] = "returned"

    _qt()[4].singleShot(0, execute)
    return identifier


def _callback(driver):
    identifier = driver.get("callback_id")
    if identifier is None:
        return None
    observed = _callbacks.get(identifier)
    if observed is None:
        _fail(
            "NATIVE_DISPATCH_OUTCOME_UNKNOWN",
            "A queued native stage lost its observation; do not replay",
            unknown=True,
        )
    if observed["status"] == "failed":
        _fail(observed["code"], observed["error"], unknown=True)
    return observed


def _patch(driver, phase, *, status="running", progress=None, result=None, effects=()):
    driver["phase"] = phase
    result_patch = {
        "phase": phase,
        "status": status,
        "driver": driver,
        "progress": progress or {"indeterminate": True},
        "effects": list(effects),
        "safe_to_rollback": status in {"verifying", "succeeded", "cancelled"},
    }
    if result is not None:
        result_patch["result"] = result
    return result_patch


def _action(record, driver, phase, action):
    driver["callback_id"] = _queue(
        record, phase, lambda app, _view, _scene: app.get_action_manager().call_action(action)
    )
    return _patch(driver, phase, effects=({"kind": "native_action_queued", "action": action},))


def _physics_switch(record, driver, phase):
    """Queue the installed toolbar's real frontend action route."""

    def release(_application, _view, _scene):
        matches = [
            item
            for root in _roots("AutoPhysics")
            for item in _walk(root)
            if str(item.property("actionId")) == ROUTES["autophysics"]
            and item.isVisible()
            and item.isEnabled()
        ]
        if len(matches) != 1:
            _fail(
                "NATIVE_PHYSICS_BUTTON_AMBIGUOUS",
                "Expected one effectively visible, enabled AutoPhysics Switch toolbar button",
                unknown=True,
                details={"matches": len(matches)},
            )
        _, _, meta, qt, *_ = _qt()
        if not meta.invokeMethod(matches[0], "released", qt.QueuedConnection):
            _fail("NATIVE_QUEUE_REJECTED", "Qt rejected the queued AutoPhysics toolbar release", unknown=True)
        # A release may destroy/rebuild toolbar delegates. Do not inspect the
        # selected native wrapper after queuing its signal.

    driver["callback_id"] = _queue(record, phase, release)
    return _patch(
        driver,
        phase,
        effects=(
            {"kind": "native_frontend_action_queued", "action": ROUTES["autophysics"], "signal": "released"},
        ),
    )


def _qml_call(record, driver, phase, kind, expression):
    def call(_app, _view, _scene):
        roots = _roots(kind)
        if len(roots) != 1:
            _fail("NATIVE_DIALOG_AMBIGUOUS", f"Expected one owned {kind} dialog", unknown=True)
        _eval(roots[0], expression)

    driver["callback_id"] = _queue(record, phase, call)
    return _patch(driver, phase, effects=({"kind": "qml_method_queued", "dialog": kind},))


def _verify(plan, after, log_text, *, reference_object=None):
    before, first, last = plan["baseline"], plan["first"], plan["last"]
    selected, controls = set(plan["tracks"]), set(plan["controls"])
    errors, maximum, changed = [], 0.0, 0
    if plan.get("expected_count", before["count"]) != after["count"] or set(before["positions"]) != set(
        after["positions"]
    ):
        errors.append("stored frame count or character membership changed")
    for identifier, positions in before["positions"].items():
        target = after["positions"].get(identifier, [])
        for frame, (left, right) in enumerate(zip(positions, target)):
            distance = math.dist(left, right)
            if not math.isfinite(distance):
                errors.append("nonfinite native position")
            maximum = max(maximum, distance)
            if identifier not in controls or not first <= frame <= last:
                if distance > plan["arguments"]["position_tolerance"]:
                    errors.append(f"protected Point {identifier} changed at frame {frame}")
            elif distance > 1e-5:
                changed += 1
    for identifier, track in before["tracks"].items():
        if identifier not in selected and after["tracks"].get(identifier) != track:
            errors.append(f"unselected track metadata changed: {identifier}")
        elif identifier in selected:
            current_sections = after["tracks"].get(identifier, {}).get("sections", {})
            protected = {
                frame: section
                for frame, section in track["sections"].items()
                if not first <= int(frame) <= last
            }
            actual_protected = {
                frame: section
                for frame, section in current_sections.items()
                if not first <= int(frame) <= last and int(frame) < before["count"]
            }
            anchor = plan.get("guard_anchors", {}).get(identifier)
            if anchor is not None:
                key = str(anchor)
                expected_anchor = plan["guard_anchor_sections"][identifier]
                if current_sections.get(key) != expected_anchor:
                    errors.append(f"planned terminal anchor metadata differs: {identifier}")
                elif key in protected:
                    actual_protected[key] = protected[key]
                else:
                    actual_protected.pop(key, None)
            if protected != actual_protected:
                errors.append(f"track metadata outside the requested interval changed: {identifier}")
    removed = set(before["tracks"]) - set(after["tracks"])
    added = set(after["tracks"]) - set(before["tracks"])
    allowed_reference_tracks = {
        identifier
        for identifier in added
        if plan["operation"] == "mocap"
        and reference_object
        and set(after["tracks"][identifier]["objects"]) == {reference_object}
    }
    if removed or added - allowed_reference_tracks:
        errors.append("native track membership changed")
    operation = plan["operation"]
    metadata_changed = any(
        before["tracks"][identifier] != after["tracks"].get(identifier) for identifier in selected
    )
    unchanged = not changed and not metadata_changed
    if unchanged and operation != "autophysics":
        errors.append("the scoped native result is unchanged")
    intervals = [
        section["interpolation"]
        for identifier in selected
        for frame, section in after["tracks"][identifier]["sections"].items()
        if first <= int(frame) < last
    ]
    if operation == "motion_generation" and (not intervals or "FIXED" not in intervals):
        errors.append("Motion Generation did not produce a FIXED interval")
    if operation == "unbake" and (
        not intervals or any(value in {"FIXED", "AI", "NONE"} for value in intervals)
    ):
        errors.append("Unbaking did not produce classic editable intervals")
    if operation == "mocap":
        for identifier in plan.get("point_tracks", selected):
            keys = after["tracks"][identifier]["sections"]
            if any(
                str(frame) not in keys or keys[str(frame)]["key"] is None for frame in range(first, last + 1)
            ):
                errors.append(f"Mocap did not author every requested frame on {identifier}")
    if not _completed(operation, log_text):
        errors.append("the operation-specific native completion marker is absent")
    return {
        "matches": not errors,
        "errors": errors[:50],
        "changed_point_samples": changed,
        "maximum_point_displacement": maximum,
        "track_metadata_changed": metadata_changed,
        "unchanged": unchanged,
        "interpolation": sorted(set(intervals)),
        "artistic_quality_verified": False,
        "physical_plausibility_verified": False,
    }


def step(job_record):
    """Perform one stage, returning a JSON patch for a durable scheduler."""
    from . import characters, workflows

    plan = job_record["plan"]
    _context(plan)
    driver = copy.deepcopy(job_record.get("driver") or {})
    phase = driver.get("phase", plan["phase"])
    operation = plan["operation"]
    if phase in {"complete", "cancelled"}:
        return _patch(
            driver,
            phase,
            status="succeeded" if phase == "complete" else "cancelled",
            result=job_record.get("result"),
        )
    observed = _callback(driver)
    if observed is not None and observed["status"] in {"queued", "started"}:
        if phase == "video_pick" and observed["status"] == "started":
            # The picker may run a nested event loop inside the Choose call.
            # Service only its owned filename prompt while that call is live.
            return _pick_video(job_record, driver)
        return _patch(driver, phase, progress={"indeterminate": True, "native_callback": observed["status"]})
    app, _, scene = _context(plan)
    rig = characters.resolve_character(
        scene, plan["character_id"], require_semantics=True, require_hands=operation == "unbake"
    )
    if phase == "physics_scope":
        report = _prepare_physics_scope(plan)
        driver["physics_scope_verification"] = report
        return _patch(
            driver,
            "scope",
            effects=(
                {
                    "kind": "native_autophysics_scope",
                    "scope": report,
                    "original": plan.get("original_operational_state"),
                },
            ),
        )
    if phase == "allocate":
        animation.tracks_set_keys(plan["tracks"], [plan["first"], plan["last"]], extend_timeline=True)
        if animation._size(scene) != plan["expected_count"]:
            _fail(
                "NATIVE_ALLOCATION_FAILED",
                "The allocated native timeline differs from the planned count",
                unknown=True,
            )
        return _patch(
            driver,
            "allocation_settings",
            effects=({"kind": "native_timeline_extended", "stored_frames": plan["expected_count"]},),
        )
    if phase == "allocation_settings":
        driver["allocation_settings_verification"] = _hold_extended_settings(scene, plan)
        return _patch(driver, "scope", effects=({"kind": "native_setting_padding_held"},))
    if phase == "scope":
        selected = workflows.tracks_select_frames(plan["tracks"], plan["first"], plan["last"])
        if not selected["verification"]["matches"]:
            _fail("NATIVE_SCOPE_VERIFICATION_FAILED", "Native selected tracks/interval differ from the plan")
        objects = set(rig.controls.values())
        if plan["arguments"].get("reference_object"):
            objects.add(common.resolve_object(scene, plan["arguments"]["reference_object"]))
        scene.selector().select(objects)
        animation.timeline_set_frame(plan["first"])
        next_phase = "guard_anchor" if plan.get("guard_anchors") else "dispatch"
        return _patch(
            driver,
            "video_open" if operation == "mocap" and plan["arguments"].get("video_path") else next_phase,
            effects=({"kind": "native_selection", "tracks": plan["tracks"]},),
        )
    if phase == "guard_anchor":
        report = _anchor_guards(scene, rig, plan)
        driver["guard_anchor_verification"] = report
        return _patch(
            driver, "dispatch", effects=({"kind": "native_guard_anchors", "anchors": plan["guard_anchors"]},)
        )
    if phase == "video_open":
        return _action(job_record, driver, "video_choose", "View.Bind video")
    if phase == "video_choose":
        roots = _roots("LoadVideoDialog")
        if not roots:
            # In this build Bind video may open the OS picker before creating
            # its QML importer. Handle that order as well as Choose in QML.
            return _pick_video(job_record, driver)
        if len(roots) != 1:
            return _patch(
                driver, phase, status="awaiting_input", result={"required": "one native video-import dialog"}
            )
        source = _qml_read(roots[0], ("sourceFile",))["sourceFile"]
        if source and _same_path(source, plan["arguments"]["video_path"]):
            return _patch(driver, "video_configure")
        return _qml_call(
            job_record, driver, "video_pick", "LoadVideoDialog", "viewModel.onChooseFileClicked()"
        )
    if phase == "video_pick":
        return _pick_video(job_record, driver)
    if phase == "video_pick_wait":
        roots = _roots("LoadVideoDialog")
        if len(roots) == 1:
            source = _qml_read(roots[0], ("sourceFile",)).get("sourceFile")
            if source and _same_path(source, plan["arguments"]["video_path"]):
                return _patch(driver, "video_configure")
        return _patch(
            driver,
            phase,
            status="awaiting_input",
            result={"required": "native importer path readback", "path": plan["arguments"]["video_path"]},
        )
    if phase == "video_configure":
        first = plan["arguments"]["video_first"]
        last = first + plan["last"] - plan["first"]
        roots = _roots("LoadVideoDialog")
        if len(roots) != 1:
            _fail("NATIVE_VIDEO_DIALOG_LOST", "The owned video-import dialog disappeared", unknown=True)
        values = _qml_read(roots[0], ("sourceFile", "nbFrames"))
        if (
            not _same_path(values["sourceFile"], plan["arguments"]["video_path"])
            or values["nbFrames"] <= last
        ):
            _fail(
                "NATIVE_VIDEO_RANGE_INVALID",
                "The selected native video/path or source frame range differs from the plan",
                unknown=True,
            )
        _eval(
            roots[0],
            f"viewModel.from={first};viewModel.to={last};JSON.stringify([viewModel.from,viewModel.to])",
        )
        actual = _qml_read(roots[0], ("from", "to", "canDecompress"))
        if actual["from"] != first or actual["to"] != last or not actual["canDecompress"]:
            _fail(
                "NATIVE_VIDEO_CONFIGURATION_FAILED",
                "Native video import parameters did not read back",
                unknown=True,
            )
        return _patch(driver, "video_start")
    if phase == "video_start":
        return _qml_call(
            job_record, driver, "video_wait", "LoadVideoDialog", "viewModel.onDecompressClicked()"
        )
    if phase == "video_wait":
        ui = _ui_snapshot()
        if ui["video"]:
            values = ui["video"][0]
            if values.get("isInProgress"):
                return _patch(
                    driver, phase, progress={"fraction": values.get("progress"), "stage": "reference_import"}
                )
        added = set(map(str, scene.model_viewer().get_objects())) - set(plan["object_ids"])
        if len(added) != 1:
            return _patch(
                driver,
                phase,
                status="awaiting_input",
                result={"required": "a single imported reference plane", "new_objects": sorted(added)},
            )
        driver["reference_object"] = next(iter(added))
        scene.selector().select(
            set(rig.controls.values()) | {common.resolve_object(scene, driver["reference_object"])}
        )
        return _patch(driver, "dispatch")
    if phase == "dispatch":
        # The marker cursor is refreshed immediately before this job's dispatch.
        driver["log_cursor"] = _log_cursor()
        driver["log_text"] = ""
        if operation == "autophysics":
            _assert_physics_scope(plan)
            enabled = _physics_enabled()
            if enabled is None:
                _fail(
                    "NATIVE_PHYSICS_STATE_UNKNOWN",
                    "Cannot safely toggle AutoPhysics without its current native state",
                )
            if enabled:
                # The public editor.turn_off() does not synchronize the native
                # enabled flag in this build. Toggle only after reading True.
                return _physics_switch(job_record, driver, "physics_disabled")
        if operation == "autophysics":
            return _physics_switch(job_record, driver, "physics_solve")
        return _action(job_record, driver, "wait", ROUTES[operation])
    if phase == "physics_disabled":
        _assert_physics_scope(plan)
        if _physics_enabled() is not False:
            _fail(
                "NATIVE_PHYSICS_STOP_UNVERIFIED",
                "Native AutoPhysics did not report disabled after Switch OFF; no new solve was dispatched",
                unknown=True,
            )
        # Discard every marker from the old preview. The next solve and apply
        # must both be observed after the verified off state.
        driver["log_cursor"] = _log_cursor()
        driver["log_text"] = ""
        return _physics_switch(job_record, driver, "physics_solve")
    if phase == "physics_cancel_disabled":
        if _physics_enabled() is not False:
            _fail(
                "NATIVE_PHYSICS_STOP_UNVERIFIED",
                "Native AutoPhysics did not report disabled after the cancellation switch",
                unknown=True,
            )
        return _patch(
            driver,
            "cancelled",
            status="cancelled",
            result={"native_calculation_started": False, "native_physics_disabled": True},
        )
    if phase == "physics_solve":
        return _poll_log(driver, "physics_apply", "Autophysics core finished")
    if phase == "physics_apply":
        return _action(job_record, driver, "physics_confirm", "AutoPhysicsTool.Snap to Auto Physics")
    if phase == "physics_confirm":
        dialogs = _ui_snapshot()["apply"]
        if len(dialogs) != 1:
            return _patch(
                driver,
                phase,
                status="awaiting_input",
                result={"required": "one Apply AutoPhysics dialog", "count": len(dialogs)},
            )
        text = (
            "All Frames" if plan["arguments"].get("apply_mode", "all_frames") == "all_frames" else "Only keys"
        )
        _queued_click("Apply AutoPhysics", text)
        return _patch(driver, "wait", effects=({"kind": "native_dialog_confirmation", "choice": text},))
    if phase in {"wait", "cancelling"}:
        return _wait(job_record, driver, phase)
    if phase == "verify":
        if (
            operation == "mocap"
            and plan["expected_count"] > plan["baseline"]["count"]
            and not driver.get("padding_settings_restored")
        ):
            driver["padding_settings_verification"] = _hold_extended_settings(scene, plan, after_native=True)
            driver["padding_settings_restored"] = True
            return _patch(
                driver, "verify", status="verifying", effects=({"kind": "native_setting_padding_verified"},)
            )
        current = _snapshot(scene, rig)
        report = _verify(
            plan, current, driver.get("log_text", ""), reference_object=driver.get("reference_object")
        )
        if not report["matches"]:
            _fail(
                "NATIVE_RESULT_VERIFICATION_FAILED",
                "Native output did not satisfy its scoped postconditions",
                unknown=True,
                details=report,
                safe_to_rollback=True,
            )
        return _patch(
            driver,
            "complete",
            status="succeeded",
            result={
                "operation": operation,
                "verification": report,
                "reference_object": driver.get("reference_object", plan["arguments"].get("reference_object")),
                "requires_full_checkpoint_commit": True,
            },
            progress={"fraction": 1.0},
        )
    _fail("NATIVE_PHASE_INVALID", f"Unsupported driver phase {phase!r}", unknown=True)


def _same_path(left, right):
    from urllib.parse import unquote, urlparse

    left = str(left)
    if left.startswith("file:"):
        left = unquote(urlparse(left).path).lstrip("/") if os.name == "nt" else unquote(urlparse(left).path)
    return bool(left) and Path(left).resolve() == Path(right).resolve()


def _pick_video(record, driver):
    path = record["plan"]["arguments"]["video_path"]
    roots = _roots("LoadVideoDialog")
    if len(roots) == 1:
        value = _qml_read(roots[0], ("sourceFile",)).get("sourceFile")
        if value and _same_path(value, path):
            return _patch(driver, "video_configure")
    application, file_dialog, meta, qt, *_ = _qt()
    dialogs = [
        widget
        for widget in application.topLevelWidgets()
        if isinstance(widget, file_dialog) and widget.isVisible()
    ]
    if len(dialogs) != 1:
        native = _win32_pick_video(path)
        if native.get("queued"):
            return _patch(
                driver,
                "video_pick_wait",
                effects=(
                    {"kind": "native_file_selection", "path": path, "route": "win32_owned_file_dialog"},
                ),
            )
        return _patch(
            driver,
            "video_choose" if not driver.get("callback_id", "").endswith(":video_pick") else "video_pick",
            status="awaiting_input",
            result={
                "required": "select the requested file in the native video picker",
                "path": path,
                "reason": native.get("reason", "No unique Qt QFileDialog is exposed"),
            },
        )
    dialog = dialogs[0]
    dialog.selectFile(path)
    selected = dialog.selectedFiles()
    if len(selected) != 1 or not _same_path(selected[0], path):
        _fail(
            "NATIVE_FILE_SELECTION_FAILED",
            "The native picker did not select the exact requested video",
            unknown=True,
        )
    if not meta.invokeMethod(dialog, "accept", qt.QueuedConnection):
        _fail("NATIVE_QUEUE_REJECTED", "Qt rejected the queued file selection", unknown=True)
    return _patch(driver, "video_pick_wait", effects=({"kind": "native_file_selection", "path": path},))


def _win32_pick_video(path):
    """Use only the observed current-process, owned Windows Load video picker.

    No foreground changes, keyboard input, global matching or cross-process
    messages. Readback in the QML importer remains the final path check.
    """
    if os.name != "nt":
        return {"queued": False, "reason": "The native Windows picker is unavailable on this platform"}
    import ctypes
    from ctypes import wintypes

    user = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.EnumChildWindows.argtypes = [wintypes.HWND, callback_type, wintypes.LPARAM]
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.GetWindow.argtypes, user.GetWindow.restype = [wintypes.HWND, wintypes.UINT], wintypes.HWND
    user.GetDlgItem.argtypes, user.GetDlgItem.restype = [wintypes.HWND, ctypes.c_int], wintypes.HWND
    user.GetDlgCtrlID.argtypes = [wintypes.HWND]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.SendMessageTimeoutW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    user.SendMessageTimeoutW.restype = ctypes.c_ssize_t
    user.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

    def text(handle, *, class_name=False):
        buffer = ctypes.create_unicode_buffer(4096)
        (user.GetClassNameW if class_name else user.GetWindowTextW)(handle, buffer, len(buffer))
        return buffer.value

    def own_process(handle):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        return pid.value == os.getpid()

    def has_main_owner(handle):
        seen = set()
        for _ in range(16):
            handle = user.GetWindow(handle, 4)  # GW_OWNER
            if not handle or handle in seen or not own_process(handle):
                return False
            seen.add(handle)
            if "cascadeur" in text(handle).casefold() and text(handle, class_name=True).startswith("Qt"):
                return True
        return False

    dialogs = []

    @callback_type
    def observe(handle, _):
        if (
            own_process(handle)
            and user.IsWindowVisible(handle)
            and text(handle, class_name=True) == "#32770"
            and text(handle).strip().casefold() == "load video"
            and has_main_owner(handle)
        ):
            dialogs.append(handle)
        return True

    user.EnumWindows(observe, 0)
    if len(dialogs) != 1:
        return {
            "queued": False,
            "reason": "No unique current-process Load video picker with a Cascadeur owner",
        }
    dialog = dialogs[0]
    candidates = []

    @callback_type
    def find_combo(handle, _):
        if user.GetDlgCtrlID(handle) == 1148 and own_process(handle) and user.IsWindowVisible(handle):
            candidates.append(handle)
        return True

    user.EnumChildWindows(dialog, find_combo, 0)
    # Common dialogs can give both ComboBoxEx32 and its inner ComboBox ID
    # 1148. Resolve those aliases to the actual Edit before checking uniqueness.
    resolved_edits = set()
    for candidate in candidates:
        if text(candidate, class_name=True) == "Edit":
            resolved_edits.add(candidate)
            continue
        edits = set()

        @callback_type
        def find_edit(handle, _):
            if (
                text(handle, class_name=True) == "Edit"
                and own_process(handle)
                and user.IsWindowVisible(handle)
            ):
                edits.add(handle)
            return True

        user.EnumChildWindows(candidate, find_edit, 0)
        resolved_edits.update(edits)
    if len(resolved_edits) != 1:
        return {
            "queued": False,
            "reason": "Filename controls do not resolve to one visible current-process Edit",
        }
    filename = next(iter(resolved_edits))
    button = user.GetDlgItem(dialog, 1)
    if (
        not button
        or not own_process(button)
        or not user.IsWindowVisible(button)
        or text(button, class_name=True) != "Button"
        or text(button).replace("&", "").strip() != "Open"
    ):
        return {"queued": False, "reason": "The native Open button does not match the tested route"}
    buffer, result = ctypes.create_unicode_buffer(path), ctypes.c_size_t()
    accepted = user.SendMessageTimeoutW(
        filename, 0x000C, 0, ctypes.cast(buffer, ctypes.c_void_p).value, 0x0002, 1000, ctypes.byref(result)
    )  # WM_SETTEXT / SMTO_ABORTIFHUNG
    if not accepted or not _same_path(text(filename), path):
        _fail(
            "NATIVE_FILE_SELECTION_FAILED",
            "The bounded Windows picker did not read back the exact path",
            unknown=True,
        )
    if not user.PostMessageW(dialog, 0x0111, 1, button):  # WM_COMMAND / IDOK / BN_CLICKED
        _fail("NATIVE_QUEUE_REJECTED", "Windows rejected the queued Open command", unknown=True)
    return {"queued": True}


def _physics_enabled():
    candidates = list(_roots("AutoPhysics"))
    for window in _windows():
        content = getattr(window, "contentItem", None)
        if callable(content):
            root = content()
            candidates.extend(
                item
                for item in _walk(root)
                if str(item.property("actionId")) == "AutoPhysicsTool.Switch Auto Physics"
            )
            candidates.append(root)
    for item in candidates:
        try:
            value = _eval(
                item,
                "typeof toolManager !== 'undefined' ? JSON.stringify(toolManager.isAutoPhysicsEnabled) : 'null'",
            )
            decoded = json.loads(str(value))
            if type(decoded) is bool:
                return decoded
        except (NativeOperationError, ValueError, RuntimeError):
            continue
    return None


def _poll_log(driver, next_phase, marker):
    text, cursor = _read_log(driver["log_cursor"])
    driver["log_cursor"] = cursor
    driver["log_text"] = (driver.get("log_text", "") + text)[-524288:]
    return _patch(driver, next_phase if marker in driver["log_text"] else driver["phase"])


def _wait(record, driver, phase):
    plan = record["plan"]
    text, cursor = _read_log(driver["log_cursor"])
    driver["log_cursor"] = cursor
    driver["log_text"] = (driver.get("log_text", "") + text)[-524288:]
    ui = _ui_snapshot()
    confirmations = ui.get("mocap_confirmations", [])
    if confirmations:
        if plan["operation"] != "mocap" or len(confirmations) != 1:
            _fail(
                "NATIVE_MOCAP_CONFIRMATION_AMBIGUOUS",
                "Expected one owned Mocap processing confirmation",
                unknown=True,
            )
        confirmation = confirmations[0]
        expected_frames = plan["last"] - plan["first"] + 1
        if confirmation["frames"] != expected_frames:
            _fail(
                "NATIVE_MOCAP_CONFIRMATION_MISMATCH",
                "The native Mocap confirmation has a different frame count",
                unknown=True,
                details={"expected": expected_frames, "actual": confirmation["frames"]},
            )
        cancelling = bool(record.get("cancel_requested")) or phase == "cancelling"
        if cancelling and driver.get("mocap_processing_confirmed"):
            return _patch(
                driver,
                phase,
                status="awaiting_input",
                result={
                    "required": "observe the already-queued Mocap start before requesting worker cancellation",
                    "safe_to_rollback": False,
                },
            )
        flag = "mocap_processing_declined" if cancelling else "mocap_processing_confirmed"
        if not driver.get(flag):
            _queued_click(confirmation["title"], "No" if cancelling else "Yes")
            driver[flag] = True
            return _patch(
                driver,
                "cancelling" if cancelling else phase,
                effects=(
                    {
                        "kind": "mocap_processing_confirmation",
                        "choice": "No" if cancelling else "Yes",
                        "frames": expected_frames,
                    },
                ),
            )
        return _patch(
            driver,
            "cancelling" if cancelling else phase,
            status="awaiting_input",
            result={
                "required": "resolve the queued Mocap processing confirmation",
                "safe_to_rollback": False,
            },
        )
    if driver.get("mocap_processing_declined"):
        # No was queued before inference. Wait until both native windows close;
        # a closed confirmation alone is insufficient if the progress UI lives.
        if not ui["mocap"] and "start mocap" not in driver["log_text"].casefold():
            return _patch(
                driver,
                "cancelled",
                status="cancelled",
                result={"cancel_requested": True, "native_processing_declined": True},
            )
        return _patch(driver, "cancelling", progress={"indeterminate": True, "cancel_requested": True})
    if ui["download"]:
        if record.get("cancel_requested") or phase == "cancelling":
            return _patch(
                driver,
                phase,
                status="awaiting_input",
                result={
                    "required": "decline or resolve the native model-download prompt",
                    "cancel_requested": True,
                    "safe_to_rollback": False,
                },
            )
        if not plan["arguments"].get("allow_model_download", False):
            return _patch(
                driver,
                phase,
                status="awaiting_input",
                result={"required": "explicit model-download permission"},
            )
        if len(ui["download"]) != 1:
            _fail(
                "NATIVE_DIALOG_AMBIGUOUS",
                "More than one model-download confirmation is visible",
                unknown=True,
            )
        if driver.get("download_confirmed"):
            return _patch(
                driver,
                phase,
                status="awaiting_input",
                result={"required": "resolve the already-confirmed native model-download dialog"},
            )
        _queued_click(ui["download"][0]["title"], "Yes")
        driver["download_confirmed"] = True
        return _patch(driver, phase, effects=({"kind": "model_download_confirmation"},))
    if phase == "cancelling":
        if ui["mocap"]:
            return _patch(driver, phase, progress={"indeterminate": True, "cancel_requested": True})
        if _completed("mocap", driver["log_text"]):
            return _patch(
                driver,
                "cancelled",
                status="cancelled",
                result={"cancel_requested": True, "native_completion_observed": True},
            )
        # Closing a progress window does not prove that a worker has stopped.
        return _patch(
            driver,
            phase,
            status="awaiting_input",
            result={
                "required": "verified native worker quiescence before recovery",
                "cancel_requested": True,
                "safe_to_rollback": False,
            },
        )
    if ui["mocap"]:
        values = ui["mocap"][0]
        progress = {
            "stage": "mocap",
            "native_state": values.get("state"),
            "indeterminate": True,
            "received_bytes": values.get("received"),
            "total_bytes": values.get("total"),
        }
        total, received = values.get("total"), values.get("received")
        if total and received:
            progress["download_fraction"] = min(1.0, received / total)
        return _patch(driver, phase, progress=progress)
    if _completed(plan["operation"], driver["log_text"]) and not ui["apply"]:
        return _patch(driver, "verify", status="verifying")
    return _patch(driver, phase)


def cancel(job_record):
    """Request supported cancellation; never imply that a live worker was stopped."""
    plan = job_record["plan"]
    _context(plan)
    driver = copy.deepcopy(job_record.get("driver") or {})
    phase = driver.get("phase", plan["phase"])
    mocap_dispatched = any(
        effect.get("kind") == "native_action_queued" and effect.get("action") == ROUTES["mocap"]
        for effect in job_record.get("effects", [])
    )
    if (
        phase
        in {
            "allocate",
            "allocation_settings",
            "guard_anchor",
            "physics_scope",
            "scope",
            "video_open",
            "dispatch",
        }
        and not driver.get("callback_id")
        and not mocap_dispatched
    ):
        return _patch(driver, "cancelled", status="cancelled", result={"native_operation_started": False})
    observed = _callback(driver)
    if plan["operation"] == "mocap" and phase.startswith("video_") and not mocap_dispatched:
        # Finish an already-owned importer, including nested native pickers,
        # before restoring its model. Never dispatch inference on this path.
        patch = step(job_record)
        next_driver = patch["driver"]
        next_observation = _callback(next_driver)
        if patch["phase"] == "dispatch" and (
            next_observation is None or next_observation["status"] == "returned"
        ):
            return _patch(
                next_driver,
                "cancelled",
                status="cancelled",
                result={"native_operation_started": False, "reference_import_completed": True},
                effects=patch.get("effects", ()),
            )
        patch["message"] = (
            "Finishing the owned video import before restoring the checkpoint; Mocap inference will not start"
        )
        return patch
    if (
        plan["operation"] == "mocap"
        and phase == "dispatch"
        and not mocap_dispatched
        and (observed is None or observed["status"] == "returned")
    ):
        return _patch(
            driver,
            "cancelled",
            status="cancelled",
            result={"native_operation_started": False, "reference_import_completed": True},
        )
    if observed and observed["status"] in {"queued", "started"}:
        return _patch(
            driver,
            phase,
            status="awaiting_input",
            result={
                "cancel_available": False,
                "reason": "A queued or executing native callback must resolve first",
                "safe_to_rollback": False,
            },
        )
    if plan["operation"] == "autophysics" and phase in {"physics_disabled", "physics_cancel_disabled"}:
        fresh_solve = _callbacks.get(_job_id(job_record) + ":physics_solve")
        if fresh_solve is None:
            enabled = _physics_enabled()
            if enabled is False:
                return _patch(
                    driver,
                    "cancelled",
                    status="cancelled",
                    result={"native_calculation_started": False, "native_physics_disabled": True},
                )
            if phase == "physics_disabled" and enabled is True:
                # Unique token: never replay the older failed stop callback.
                return _physics_switch(job_record, driver, "physics_cancel_disabled")
            _fail(
                "NATIVE_PHYSICS_STOP_UNVERIFIED",
                "Native AutoPhysics cancellation could not verify its disabled state",
                unknown=True,
            )
    if phase == "cancelling":
        return _wait(job_record, driver, phase)
    if phase == "verify":
        return step(job_record)
    if plan["operation"] != "mocap":
        # No verified stop API: observe normal completion, then let the
        # scheduler restore the checkpoint instead of publishing this result.
        patch = step(job_record)
        patch["message"] = (
            "Native cancellation is unavailable; waiting for completion before restoring the checkpoint"
        )
        return patch
    if phase == "wait":
        observation = _wait(job_record, driver, phase)
        if observation["phase"] != "wait":
            return observation
    roots = _roots("MocapDialog")
    if len(roots) != 1:
        return _patch(
            driver,
            phase,
            status="awaiting_input",
            result={
                "cancel_available": False,
                "reason": "No unique owned Mocap progress dialog",
                "safe_to_rollback": False,
            },
        )
    # The installed QML Cancel button closes root.window. Queue the equivalent
    # observed button event, then wait for independent worker-stop evidence.
    matches = [
        (str(window.title()), text)
        for window in _windows()
        for text, _ in _buttons(window)
        if text.strip() == "Cancel"
        and callable(getattr(window, "contentItem", None))
        and any("MocapDialog" in _class(item) for item in _walk(window.contentItem()))
    ]
    if len(matches) != 1:
        return _patch(
            driver,
            phase,
            status="awaiting_input",
            result={
                "cancel_available": False,
                "reason": "No unique native Mocap Cancel button",
                "safe_to_rollback": False,
            },
        )
    _queued_click(*matches[0])
    return _patch(
        driver,
        "cancelling",
        result={"cancel_requested": True, "safe_to_rollback": False},
        effects=({"kind": "native_cancel_queued"},),
    )
