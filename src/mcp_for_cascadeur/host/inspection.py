"""Bounded read-only discovery of the installed application's public Python API."""

from __future__ import annotations

import inspect
import re
import sys

from ..contracts import Tool
from . import common
from .data_channels import read_data_value, read_setting_value


def _member_record(name, value):
    record = {"name": name, "type": type(value).__name__, "callable": callable(value)}
    if callable(value):
        try:
            record["signature"] = str(inspect.signature(value))[:1600]
        except (TypeError, ValueError):
            record["signature"] = None
    if value is None or type(value) in (str, int, float, bool):
        record["value"] = common.json_value(value)
    documentation = inspect.getdoc(value)
    if documentation:
        record["documentation"] = documentation[:2400]
    return record


def _public_path(root, path):
    if not isinstance(path, str) or len(path) > 400:
        raise ValueError("path must be a public dotted attribute path of at most 400 characters.")
    segments = path.split(".") if path else []
    if len(segments) > 8 or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", part) or part.startswith("_") for part in segments
    ):
        raise ValueError(
            "Use at most eight public attribute names; expressions and private members are forbidden."
        )
    value = root
    for part in segments:
        value = getattr(value, part)
    return value


def _describe(value, limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")
    names = sorted(name for name in dir(value) if not name.startswith("_"))
    result = _member_record("", value)
    members = []
    for name in names[:limit]:
        try:
            members.append(_member_record(name, getattr(value, name)))
        except Exception as error:
            members.append({"name": name, "unavailable": str(error)[:500]})
    result.update(members=members, total_members=len(names), truncated=len(names) > limit)
    return result


def api_inspect(root="csc", path="", limit=40):
    csc, app, view, scene = common.current_context(require_scene=False)
    roots = {"csc": csc, "app": app, "view": view, "scene": scene, "model": None, "layers": None}
    if root not in roots:
        raise ValueError("root must be csc, app, view, scene, model or layers.")
    if scene is not None and root in ("model", "layers"):
        roots[root] = scene.model_viewer() if root == "model" else scene.layers_viewer()
    if roots[root] is None:
        raise RuntimeError("Open a scene before inspecting this API root.")
    result = _describe(_public_path(roots[root], path), limit)
    result.update(
        root=root, path=path, note="Discovery reads public attributes; listed methods were not executed."
    )
    return result


def _tool_pairs(app):
    pairs = []
    for tool in app.get_tools_manager().tools():
        if isinstance(tool, str):
            name = tool
        else:
            native_name = getattr(tool, "name", None)
            name = native_name() if callable(native_name) else type(tool).__name__
        pairs.append((str(name), tool))
    return pairs


def _tools(app):
    records = [{"name": name, "type": type(tool).__name__} for name, tool in _tool_pairs(app)]
    return sorted(records, key=lambda item: item["name"])


def native_tools_list(query=""):
    if not isinstance(query, str) or len(query) > 200:
        raise ValueError("query must be a string of at most 200 characters.")
    _, app, _, _ = common.current_context(require_scene=False)
    return {
        "tools": [tool for tool in _tools(app) if query.casefold() in tool["name"].casefold()],
        "note": "Listed tools exist; their presence does not verify an operation or license.",
    }


def native_tool_inspect(name, editor=False, path="", limit=40):
    _, app, view, _ = common.current_context(require_scene=editor)
    matches = [tool for tool_name, tool in _tool_pairs(app) if tool_name == name]
    if len(matches) != 1:
        raise ValueError("Choose an exact tool name returned by native_tools_list.")
    tool = matches[0]
    if isinstance(tool, str):
        tool = app.get_tools_manager().get_tool(tool)
    if editor:
        tool = tool.editor(view)
    result = _describe(_public_path(tool, path), limit)
    result.update(tool=name, editor=editor, path=path)
    return result


def app_capabilities():
    csc, app, view, _ = common.current_context(require_scene=False)
    variables = csc.SystemVariables
    values = {}
    for name in ("git_version", "git_sha", "git_date", "git_count"):
        try:
            values[name] = common.json_value(getattr(variables, name)())
        except Exception as error:
            values[name] = {"unavailable": str(error)[:300]}
    return {
        "native_build": values,
        "embedded_python": sys.version,
        "scene_open": view is not None,
        "license": {
            "export_available": bool(app.is_export_available()),
            "pro_features_available": bool(app.is_pro_features_available()),
        },
        "native_tools": _tools(app),
        "viewport_modes": list(getattr(csc.view.ViewportMode, "__members__", {})),
        "interpolation_modes": list(getattr(csc.layers.layer.Interpolation, "__members__", {})),
        "workspace": str(common.get_settings()["workspace"]),
        "verification": "API discovery only; test scene edits, render and file roundtrips independently.",
    }


def _object_kind(behaviours, data, object_id):
    try:
        basic = behaviours.get_behaviour_by_name(object_id, "Basic")
        if basic.is_null():
            return None
        channel = behaviours.get_behaviour_data(basic, "type_selectable")
        return str(data.get_data_value(channel))
    except (TypeError, ValueError, RuntimeError):
        return None


def objects_list(query="", object_type=None, offset=0, limit=100):
    if not isinstance(query, str) or len(query) > 200:
        raise ValueError("query must be a string of at most 200 characters.")
    if object_type is not None and (not isinstance(object_type, str) or len(object_type) > 100):
        raise ValueError("object_type must be an exact discovered type string.")
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError("Use a nonnegative offset and limit between 1 and 500.")
    _, _, _, scene = common.current_context()
    model = scene.model_viewer()
    behaviours, data = model.behaviour_viewer(), model.data_viewer()
    rows = []
    for object_id in model.get_objects():
        name = model.get_object_name(object_id)
        if query.casefold() not in name.casefold():
            continue
        kind = _object_kind(behaviours, data, object_id)
        if object_type is None or kind == object_type:
            rows.append({"id": str(object_id), "name": name, "type": kind})
    rows.sort(key=lambda item: (item["name"].casefold(), item["id"]))
    return {
        "objects": rows[offset : offset + limit],
        "total": len(rows),
        "offset": offset,
        "next_offset": offset + limit if offset + limit < len(rows) else None,
    }


def _property_record(behaviours, data, behaviour_id, name, frame):
    property_type = behaviours.get_property_type(behaviour_id, name)
    kind = getattr(property_type, "name", str(property_type).split(".")[-1])
    result = {"name": name, "kind": kind}
    if kind == "DataType":
        channel = behaviours.get_behaviour_data(behaviour_id, name)
        result.update(
            data_id=str(channel),
            subtype=behaviours.get_behaviour_data_subtype(behaviour_id, name),
            value=common.json_value(read_data_value(data, channel, frame)),
        )
    elif kind == "DataRangeType":
        channels = behaviours.get_behaviour_data_range(behaviour_id, name)
        result.update(
            data=[
                {"id": str(channel), "value": common.json_value(read_data_value(data, channel, frame))}
                for channel in channels[:100]
            ],
            total=len(channels),
        )
    elif kind == "SettingType":
        setting = behaviours.get_behaviour_setting(behaviour_id, name)
        result.update(
            setting_id=str(setting),
            subtype=behaviours.get_behaviour_setting_subtype(behaviour_id, name),
            value=common.json_value(read_setting_value(data, setting, frame)),
        )
    elif kind == "SettingRangeType":
        settings = behaviours.get_behaviour_settings_range(behaviour_id, name)
        result.update(
            settings=[
                {"id": str(setting), "value": common.json_value(read_setting_value(data, setting, frame))}
                for setting in settings[:100]
            ],
            total=len(settings),
        )
    elif kind == "ObjectType":
        result["object_id"] = str(behaviours.get_behaviour_object(behaviour_id, name))
    elif kind == "ObjectRangeType":
        objects = behaviours.get_behaviour_objects_range(behaviour_id, name)
        result.update(object_ids=[str(item) for item in objects[:100]], total=len(objects))
    elif kind == "BehaviourType":
        result["behaviour_id"] = str(behaviours.get_behaviour_reference(behaviour_id, name))
    elif kind == "BehaviourRangeType":
        references = behaviours.get_behaviour_reference_range(behaviour_id, name)
        result.update(behaviour_ids=[str(item) for item in references[:100]], total=len(references))
    elif kind == "AssetType":
        result["asset_id"] = str(behaviours.get_behaviour_asset(behaviour_id, name))
    elif kind == "AssetRangeType":
        assets = behaviours.get_behaviour_asset_range(behaviour_id, name)
        result.update(asset_ids=[str(item) for item in assets[:100]], total=len(assets))
    elif kind == "StringType":
        result["value"] = behaviours.get_behaviour_string(behaviour_id, name)
    return result


def object_inspect(object, frame=None, behavior=None, limit=200):
    if type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500 properties.")
    if behavior is not None and (not isinstance(behavior, str) or not behavior or len(behavior) > 200):
        raise ValueError("behavior must be an exact discovered behavior name.")
    _, _, _, scene = common.current_context()
    model = scene.model_viewer()
    object_id = common.resolve_object(scene, object)
    frame = scene.get_current_frame() if frame is None else frame
    if type(frame) is not int or not 0 <= frame < scene.layers_viewer().frames_count():
        raise ValueError("frame must identify an existing animation frame.")
    behaviours, data = model.behaviour_viewer(), model.data_viewer()
    result = {
        "id": str(object_id),
        "name": model.get_object_name(object_id),
        "type": _object_kind(behaviours, data, object_id),
        "frame": frame,
        "behaviors": [],
    }
    remaining = limit
    for behaviour_id in list(behaviours.get_behaviours(object_id))[:100]:
        name = behaviours.get_behaviour_name(behaviour_id)
        if behavior is not None and name != behavior:
            continue
        names = behaviours.get_behaviour_property_names(behaviour_id)
        record = {"id": str(behaviour_id), "name": name, "properties": [], "total_properties": len(names)}
        for property_name in names[:remaining]:
            try:
                record["properties"].append(
                    _property_record(behaviours, data, behaviour_id, property_name, frame)
                )
            except Exception as error:
                record["properties"].append({"name": property_name, "unavailable": str(error)[:500]})
        remaining -= len(record["properties"])
        result["behaviors"].append(record)
        if not remaining:
            break
    result["property_limit"] = limit
    result["truncated"] = not remaining
    return result


def object_hierarchy(query="", limit=500):
    if not isinstance(query, str) or len(query) > 200 or type(limit) is not int or not 1 <= limit <= 2000:
        raise ValueError("Use a query up to 200 characters and limit between 1 and 2000.")
    _, _, _, scene = common.current_context()
    model = scene.model_viewer()
    behaviours = model.behaviour_viewer()
    rows = []
    for object_id in model.get_objects():
        name = model.get_object_name(object_id)
        if query.casefold() not in name.casefold():
            continue
        row = {"id": str(object_id), "name": name, "parent_id": None}
        try:
            basic = behaviours.get_behaviour_by_name(object_id, "Basic")
            if not basic.is_null():
                parent = behaviours.get_behaviour_object(basic, "parent")
                if not parent.is_null():
                    row["parent_id"] = str(parent)
        except (TypeError, RuntimeError, ValueError) as error:
            row["unavailable"] = str(error)[:300]
        rows.append(row)
    rows.sort(key=lambda item: (item["name"].casefold(), item["id"]))
    return {"objects": rows[:limit], "total": len(rows), "truncated": len(rows) > limit}


_PATH = {"type": "string", "maxLength": 400}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 100, "default": 40}
TOOLS = [
    Tool(
        "app_capabilities",
        "Read actual native build, Python version, license flags, tools and enum names without editing.",
        app_capabilities,
    ),
    Tool(
        "api_inspect",
        "Read public native API members, signatures and documentation; does not invoke discovered methods.",
        api_inspect,
        {
            "root": {"type": "string", "enum": ["csc", "app", "view", "scene", "model", "layers"]},
            "path": _PATH,
            "limit": _LIMIT,
        },
    ),
    Tool(
        "native_tools_list",
        "Read exact installed native tool names; availability does not verify an operation.",
        native_tools_list,
        {"query": {"type": "string", "maxLength": 200}},
    ),
    Tool(
        "native_tool_inspect",
        "Inspect an exact installed tool or its scene editor, without invoking discovered methods.",
        native_tool_inspect,
        {
            "name": {"type": "string", "minLength": 1, "maxLength": 200},
            "editor": {"type": "boolean", "default": False},
            "path": _PATH,
            "limit": _LIMIT,
        },
        ("name",),
    ),
    Tool(
        "objects_list",
        "Find objects by name substring and optional exact native type; returns IDs and pagination.",
        objects_list,
        {
            "query": {"type": "string", "maxLength": 200},
            "object_type": {"type": "string", "maxLength": 100},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        },
    ),
    Tool(
        "object_inspect",
        "Read an object's actual behaviors, data channels, settings and native references before editing.",
        object_inspect,
        {
            "object": {"type": "string", "minLength": 1},
            "frame": {"type": "integer", "minimum": 0},
            "behavior": {"type": "string", "minLength": 1, "maxLength": 200},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        ("object",),
    ),
    Tool(
        "object_hierarchy",
        "Read Basic parent links by exact native IDs, with a bounded output.",
        object_hierarchy,
        {
            "query": {"type": "string", "maxLength": 200},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
        },
    ),
]
