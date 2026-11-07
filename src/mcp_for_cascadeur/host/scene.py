"""Application scene, viewport and exchange operations through the native API."""

from __future__ import annotations

import math

from ..contracts import Tool
from . import common
from .action_catalog import ACTIONS


def _scene_record(view, index=None):
    result = {"name": view.name(), "path": view.get_path_name()}
    if index is not None:
        result["index"] = index
    return result


def scene_list():
    _, app, current, _ = common.current_context(require_scene=False)
    return {
        "scenes": [
            dict(_scene_record(view, index), active=view == current)
            for index, view in enumerate(app.get_scene_manager().scenes())
        ]
    }


def selection_get():
    _, _, _, scene = common.current_context()
    viewer = scene.model_viewer()
    records = []
    for object_id in scene.selector().selected().ids:
        record = {"id": str(object_id)}
        try:
            record["name"] = viewer.get_object_name(object_id)
        except (TypeError, RuntimeError):
            record["name"] = None  # Native tool selections need not be model objects.
        records.append(record)
    return {"objects": records}


def scene_info():
    _, _, view, scene = common.current_context()
    boundary = view.animation_boundary()
    return dict(
        _scene_record(view),
        current_frame=scene.get_current_frame(),
        frame_count=scene.layers_viewer().frames_count(),
        object_count=len(scene.model_viewer().get_objects()),
        selection=selection_get()["objects"],
        playback={
            name: getattr(boundary, name)
            for name in ("first_frame", "last_frame", "first_visible_frame", "last_visible_frame")
        },
    )


def scene_new():
    _, app, _, _ = common.current_context(require_scene=False)
    manager = app.get_scene_manager()
    view = manager.create_application_scene()
    manager.set_current_scene(view)
    return scene_info()


def scene_activate(index):
    _, app, _, _ = common.current_context(require_scene=False)
    scenes = app.get_scene_manager().scenes()
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(scenes):
        raise ValueError("Scene index is out of range; refresh scene_list first.")
    app.get_scene_manager().set_current_scene(scenes[index])
    return scene_info()


def scene_open(path):
    destination = common.workspace_path(path, must_exist=True, suffixes=(".casc",))
    _, app, _, _ = common.current_context(require_scene=False)
    if not app.get_data_source_manager().load_scene(str(destination)):
        raise RuntimeError("Cascadeur declined to open the scene.")
    return scene_info()


def _stamp(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return [stat.st_mtime_ns, stat.st_size]


def _file_result(path, before, mime_type=None):
    after = _stamp(path)
    result = {"path": str(path), "verification": "pending", "previous_stamp": before}
    if after is not None and after != before and after[1] > 0:
        result.update(verification="fresh_file", bytes=after[1])
    if mime_type:
        result["mime_type"] = mime_type
    return result


def scene_save(path, overwrite=False):
    import hashlib

    destination = common.workspace_path(path, overwrite=overwrite, suffixes=(".casc",))
    _, _, view, _ = common.current_context()
    before = _stamp(destination)
    view.save(str(destination))
    result = _file_result(destination, before)
    if result["verification"] == "fresh_file":
        with destination.open("rb") as stream:
            result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def selection_set(objects):
    if not isinstance(objects, list) or len(objects) > 500:
        raise ValueError("Provide a list containing at most 500 object identifiers.")
    _, _, _, scene = common.current_context()
    resolved = [common.resolve_object(scene, item) for item in objects]
    scene.selector().select(set(resolved))
    return selection_get()


def _vector(value):
    converted = common.json_value(value)
    if isinstance(converted, (tuple, list)):
        converted = [
            item[0] if isinstance(item, (tuple, list)) and len(item) == 1 else item for item in converted
        ]
    if not isinstance(converted, (tuple, list)) or len(converted) != 3:
        raise ValueError("Expected three finite coordinates.")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
        for item in converted
    ):
        raise ValueError("Expected three finite coordinates.")
    return [float(item) for item in converted]


def _viewport(view):
    viewport = view.active_viewport()
    if viewport is None:
        raise RuntimeError("The active scene has no viewport.")
    return viewport.domain_viewport()


def camera_get():
    _, _, view, _ = common.current_context()
    viewport = _viewport(view)
    camera = viewport.camera_struct()
    return {
        "position": _vector(camera.position),
        "target": _vector(camera.target),
        "projection": str(camera.type),
        "mode": str(viewport.mode_visualizers()),
    }


def camera_set(position=None, target=None):
    if position is None and target is None:
        raise ValueError("Provide a camera position or target.")
    checked_position = _vector(position) if position is not None else None
    checked_target = _vector(target) if target is not None else None
    _, _, view, _ = common.current_context()
    viewport = _viewport(view)
    camera = viewport.camera_struct()
    new_position = checked_position if checked_position is not None else _vector(camera.position)
    new_target = checked_target if checked_target is not None else _vector(camera.target)
    if new_position == new_target:
        raise ValueError("Camera position and target must differ.")
    import numpy as np

    if checked_position is not None:
        camera.position = np.asarray(checked_position, dtype=np.float32)
    if checked_target is not None:
        camera.target = np.asarray(checked_target, dtype=np.float32)
    viewport.set_camera_struct(camera)
    return camera_get()


def camera_frame(objects, distance_scale=2.5):
    if not isinstance(objects, list) or not 1 <= len(objects) <= 500:
        raise ValueError("Choose 1 to 500 objects to frame.")
    if (
        isinstance(distance_scale, bool)
        or not isinstance(distance_scale, (int, float))
        or not math.isfinite(distance_scale)
        or not 1 <= distance_scale <= 20
    ):
        raise ValueError("distance_scale must be between 1 and 20.")
    _, _, view, scene = common.current_context()
    model = scene.model_viewer()
    behaviours = model.behaviour_viewer()
    data = model.data_viewer()
    frame = scene.get_current_frame()
    positions = []
    for identifier in objects:
        object_id = common.resolve_object(scene, identifier)
        transform = behaviours.get_behaviour_by_name(object_id, "Transform")
        if transform.is_null():
            raise ValueError("Object has no native Transform: " + str(identifier))
        position_id = behaviours.get_behaviour_data(transform, "global_position")
        positions.append(_vector(data.get_data_value(position_id, frame)))
    low = [min(point[axis] for point in positions) for axis in range(3)]
    high = [max(point[axis] for point in positions) for axis in range(3)]
    center = [(left + right) / 2 for left, right in zip(low, high)]
    padding = distance_scale / 2.5
    points = [
        [center[axis] + (point[axis] - center[axis]) * padding for axis in range(3)] for point in positions
    ]
    import numpy as np

    _viewport(view).camera().zoom_to_points([np.asarray(point, dtype=np.float32) for point in points])
    result = camera_get()
    result["framing"] = "Native world-space framing of Transform positions; mesh bounds are not measured."
    result["distance_scale"] = distance_scale
    return result


def viewport_mode_set(mode):
    csc, _, view, _ = common.current_context()
    modes = getattr(csc.view.ViewportMode, "__members__", {})
    if mode not in modes:
        raise ValueError("Unknown viewport mode; discover actual ViewportMode members first.")
    _viewport(view).set_mode_visualizers(modes[mode])
    return camera_get()


def _native_tool(app, name):
    manager = app.get_tools_manager()
    for tool in manager.tools():
        if isinstance(tool, str):
            tool_name = tool
        else:
            native_name = getattr(tool, "name", None)
            tool_name = native_name() if callable(native_name) else type(tool).__name__
        if str(tool_name) == name:
            return manager.get_tool(tool) if isinstance(tool, str) else tool
    raise RuntimeError("Native tool is unavailable: " + name)


def viewport_capture(path, width=1280, height=720, samples=1, overwrite=False):
    for value, maximum, label in ((width, 3840, "width"), (height, 2160, "height"), (samples, 16, "samples")):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(label + " is outside the supported range.")
    destination = common.workspace_path(path, overwrite=overwrite, suffixes=(".png",))
    csc, app, view, _ = common.current_context()
    renderer = _native_tool(app, "RenderToFile")
    parameters = csc.tools.RenderParameters()
    parameters.width, parameters.height, parameters.samples = width, height, samples
    before = _stamp(destination)
    renderer.take_image(view, parameters, str(destination))
    return _file_result(destination, before, "image/png")


def fbx_import(path, mode="scene", take=-1, set_offset=True):
    from . import recovery

    methods = {
        "scene": "import_scene",
        "animation": "import_animation",
        "model": "import_model",
        "add_model": "add_model",
    }
    if mode not in methods:
        raise ValueError("Unsupported FBX import mode.")
    if isinstance(take, bool) or not isinstance(take, int) or take < -1:
        raise ValueError("take must be -1 or a nonnegative take index.")
    if type(set_offset) is not bool or (mode != "scene" and not set_offset):
        raise ValueError("set_offset must be a boolean; disabling it applies only to scene import.")
    destination = common.workspace_path(path, must_exist=True, suffixes=(".fbx",))
    _, app, view, native_scene = common.current_context()
    loader = _native_tool(app, "FbxSceneLoader").get_fbx_loader(view)
    method = getattr(loader, methods[mode])
    recovery.note_native_attempt(native_scene)
    if mode == "scene" and not set_offset:
        method(str(destination), take, False)
    elif mode in ("scene", "animation"):
        method(str(destination), take)
    else:
        method(str(destination))
    recovery.note_native_commit(native_scene)
    return {
        "path": str(destination),
        "mode": mode,
        "dispatched": True,
        "scene": scene_info(),
        "verification": "Inspect the imported rig, tracks and animation.",
    }


def fbx_export(path, mode="animation", overwrite=False):
    methods = {
        "animation": "export_joints",
        "scene": "export_all_objects",
        "selected_objects": "export_scene_selected_objects",
        "selected_joints": "export_joints_selected",
        "selected_frames": "export_joints_selected_frames",
        "model": "export_model",
    }
    if mode not in methods:
        raise ValueError("Unsupported FBX export mode.")
    destination = common.workspace_path(path, overwrite=overwrite, suffixes=(".fbx",))
    _, app, view, _ = common.current_context()
    if not app.is_export_available():
        raise PermissionError("This Cascadeur installation does not report export entitlement.")
    loader = _native_tool(app, "FbxSceneLoader").get_fbx_loader(view)
    before = _stamp(destination)
    getattr(loader, methods[mode])(str(destination))
    return dict(_file_result(destination, before), mode=mode)


# Exact IDs in the vendor's action-ID reference. Availability and completion are
# deliberately not inferred from a documented action name.
ACTION_CATALOG = {
    "Scene.Undo": "Undo the last native scene edit.",
    "Scene.Redo": "Redo a native scene edit.",
    "Timeline.Play": "Toggle native playback.",
    "AutoPhysicsTool.Snap to Auto Physics": "Apply the current AutoPhysics result.",
    "AutoPhysicsTool.Switch Auto Physics": "Toggle AutoPhysics.",
    "AutoPhysicsTool.Set priority frame": "Set the current priority frame.",
    "AutoPosingTool.AutoPosing": "Toggle AutoPosing.",
    "AutoPosingTool.Update": "Update AutoPosing with current context.",
    "MirrorTool.Mirror on current frame": "Mirror the current selection at this frame.",
    "MirrorTool.Mirror on interval": "Mirror the selection over the selected interval.",
    "View.Retargeting_Copy": "Copy the current retargeting source.",
    "View.Retargeting_Paste": "Paste retargeting into the current context.",
    "View.FixFoot": "Run fulcrum motion cleaning with current selection.",
    "View.Center camera": "Center the active camera using current selection.",
}


ACTION_CATALOG.update(
    {
        identifier: record["description"]
        for identifier, record in ACTIONS.items()
        if identifier not in ACTION_CATALOG
    }
)


def actions_list(query=""):
    if not isinstance(query, str) or len(query) > 200:
        raise ValueError("query must be a string of at most 200 characters.")
    return {
        "source": "https://cascadeur.com/help/category/301",
        "availability": "Documented IDs; this API does not enumerate runtime action availability.",
        "actions": [
            {"id": name, "description": description, "category": ACTIONS.get(name, {}).get("category")}
            for name, description in ACTION_CATALOG.items()
            if query.casefold() in (name + " " + description).casefold()
        ],
    }


def action_run(action_id, expected_scene):
    from . import recovery

    if action_id not in ACTION_CATALOG:
        raise ValueError("Choose an exact action ID from actions_list.")
    _, app, view, scene = common.current_context()
    actual_path = view.get_path_name()
    if not expected_scene or expected_scene not in (actual_path, view.name()):
        raise ValueError("Active scene differs from expected_scene; refresh scene_info.")
    recovery.note_native_attempt(scene)
    app.get_action_manager().call_action(action_id)
    history_kind = {"Scene.Undo": "undo", "Scene.Redo": "redo"}.get(action_id, "modify")
    recovery.note_native_commit(scene, kind=history_kind)
    return {
        "action_id": action_id,
        "dispatched": True,
        "completed": None,
        "frame_after": scene.get_current_frame(),
        "verification": "Check the native result before retrying; actions may toggle state or open dialogs.",
    }


_TEXT = {"type": "string"}
_FLAG = {"type": "boolean", "default": False}
_OBJECTS = {"type": "array", "items": _TEXT, "maxItems": 500}
_VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
TOOLS = [
    Tool("scene_info", "Read active scene, current frame, bounds, object count and selection.", scene_info),
    Tool("scene_list", "Read open scene tabs and fresh indices.", scene_list),
    Tool(
        "scene_new",
        "Create and activate a new scene tab, preserving existing tabs.",
        scene_new,
        read_only=False,
    ),
    Tool(
        "scene_activate",
        "Activate a scene using a fresh index from scene_list.",
        scene_activate,
        {"index": {"type": "integer", "minimum": 0}},
        ("index",),
        False,
    ),
    Tool(
        "scene_open",
        "Open a workspace .casc file; inspect the loaded scene before edits.",
        scene_open,
        {"path": _TEXT},
        ("path",),
        False,
    ),
    Tool(
        "scene_save",
        "Save the active scene to a workspace .casc path; verify fresh output and reopen to check persistence.",
        scene_save,
        {"path": _TEXT, "overwrite": _FLAG},
        ("path",),
        False,
    ),
    Tool("selection_get", "Read selected native object IDs and available names.", selection_get),
    Tool(
        "selection_set",
        "Replace selected objects by exact IDs or unambiguous names; an empty list clears selection.",
        selection_set,
        {"objects": _OBJECTS},
        ("objects",),
        False,
    ),
    Tool("camera_get", "Read active viewport camera and display mode.", camera_get),
    Tool(
        "camera_set",
        "Set active viewport camera position or target in scene units.",
        camera_set,
        {"position": _VECTOR, "target": _VECTOR},
        (),
        False,
    ),
    Tool(
        "camera_frame",
        "Frame actual Transform positions of named objects; inspect image framing afterwards.",
        camera_frame,
        {
            "objects": dict(_OBJECTS, minItems=1),
            "distance_scale": {"type": "number", "minimum": 1, "maximum": 20},
        },
        ("objects",),
        False,
    ),
    Tool(
        "viewport_mode_set",
        "Set an exact discovered native ViewportMode enum name.",
        viewport_mode_set,
        {"mode": _TEXT},
        ("mode",),
        False,
    ),
    Tool(
        "viewport_capture",
        "Capture a fresh workspace PNG and return its path for image review.",
        viewport_capture,
        {
            "path": _TEXT,
            "width": {"type": "integer", "minimum": 1, "maximum": 3840},
            "height": {"type": "integer", "minimum": 1, "maximum": 2160},
            "samples": {"type": "integer", "minimum": 1, "maximum": 16},
            "overwrite": _FLAG,
        },
        ("path",),
        False,
    ),
    Tool(
        "fbx_import",
        "Import FBX into the native scene; animation import replaces matching data. Checkpoint first.",
        fbx_import,
        {
            "path": _TEXT,
            "mode": {"enum": ["scene", "animation", "model", "add_model"]},
            "take": {"type": "integer", "minimum": -1},
            "set_offset": {
                "type": "boolean",
                "default": True,
                "description": "For scene imports, disable native placement offset when comparing original coordinates.",
            },
        },
        ("path",),
        False,
    ),
    Tool(
        "fbx_export",
        "Export a workspace FBX when the native license permits; verify rig and axes in the receiving app.",
        fbx_export,
        {
            "path": _TEXT,
            "mode": {
                "enum": [
                    "animation",
                    "scene",
                    "selected_objects",
                    "selected_joints",
                    "selected_frames",
                    "model",
                ]
            },
            "overwrite": _FLAG,
        },
        ("path",),
        False,
    ),
    Tool(
        "actions_list",
        "Read the documented native action allowlist; presence is not runtime verification.",
        actions_list,
        {"query": _TEXT},
    ),
    Tool(
        "action_run",
        "Dispatch an exact listed native action on the expected scene; report dispatch, then inspect completion.",
        action_run,
        {"action_id": _TEXT, "expected_scene": _TEXT},
        ("action_id", "expected_scene"),
        False,
    ),
]
