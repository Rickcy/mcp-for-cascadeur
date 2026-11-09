"""Exact native data-channel discovery and typed, checked batch writes."""

from __future__ import annotations

from ..contracts import Tool
from . import animation, common


def _read_native_value(viewer, reference, frame, *, setting=False):
    descriptor = viewer.get_setting(reference) if setting else viewer.get_data(reference)
    mode = str(getattr(descriptor.mode, "name", descriptor.mode)).lower()
    getter = viewer.get_setting_value if setting else viewer.get_data_value
    if mode == "static":
        return getter(reference)
    if mode in ("animation", "animated"):
        return getter(reference, frame)
    raise ValueError("Native reads require a supported Static or Animation mode.")


def read_data_value(viewer, reference, frame):
    """Use the actual native overload: Static has no frame argument."""
    return _read_native_value(viewer, reference, frame)


def read_setting_value(viewer, reference, frame):
    return _read_native_value(viewer, reference, frame, setting=True)


def _channels(scene, object_id):
    references = list(scene.data_viewer().get_all_data_id(object_id))
    if len(references) > 10000 or len(set(references)) != len(references):
        raise ValueError("Object data discovery exceeds 10000 unique native channels.")
    return references


def _resolve(scene, object_id, identifier):
    if type(identifier) is not str or not identifier.strip() or len(identifier) > 512:
        raise ValueError("data must be an exact native channel ID or nonempty display name.")
    viewer = scene.data_viewer()
    references = _channels(scene, object_id)
    matches = [reference for reference in references if str(reference) == identifier]
    if not matches:
        matches = [reference for reference in references if viewer.get_data_name(reference) == identifier]
    if len(matches) != 1:
        raise ValueError(
            f"Data channel {identifier!r} matched {len(matches)} channels on the specified object; use its exact ID."
        )
    reference = matches[0]
    if viewer.get_data(reference).object_id != object_id:
        raise ValueError("Native data channel owner differs from the requested object.")
    return reference


def _record(scene, reference, frame):
    data = scene.data_viewer()
    descriptor = data.get_data(reference)
    value = read_data_value(data, reference, frame)
    return {
        "id": str(reference),
        "name": data.get_data_name(reference),
        "owner": str(descriptor.object_id),
        "mode": descriptor.mode.name,
        "native_value_type": type(value).__name__,
        "value": common.json_value(value),
    }


def object_data_read(object, data=None, frame=None, query="", offset=0, limit=100):
    _, _, _, scene = common.current_context()
    object_id = common.resolve_object(scene, object)
    position = animation._existing_frame(scene, frame)
    if (
        type(query) is not str
        or len(query) > 200
        or type(offset) is not int
        or offset < 0
        or type(limit) is not int
        or not 1 <= limit <= 500
    ):
        raise ValueError("Use a query up to 200 characters, a nonnegative offset and limit 1-500.")
    viewer = scene.data_viewer()
    references = [_resolve(scene, object_id, data)] if data is not None else _channels(scene, object_id)
    references = [
        reference
        for reference in references
        if query.casefold() in viewer.get_data_name(reference).casefold()
    ]
    references.sort(key=lambda reference: (viewer.get_data_name(reference).casefold(), str(reference)))
    rows = []
    for reference in references[offset : offset + limit]:
        try:
            rows.append(_record(scene, reference, position))
        except Exception as error:
            rows.append(
                {
                    "id": str(reference),
                    "name": viewer.get_data_name(reference),
                    "unavailable": str(error)[:500],
                }
            )
    return {
        "object": str(object_id),
        "frame": position,
        "total": len(references),
        "channels": rows,
        "offset": offset,
        "limit": limit,
        "truncated": offset + limit < len(references),
        "scope": "All native data owned by this object, including channels without behavior-property bindings. Data can be authored inputs or derived outputs.",
    }


def object_data_set(changes, frame=None, create_keys=False):
    csc, _, _, scene = common.current_context()
    position = animation._existing_frame(scene, frame)
    if type(create_keys) is not bool:
        raise ValueError("create_keys must be a boolean.")
    if type(changes) is not list or not 1 <= len(changes) <= 200:
        raise ValueError("Provide 1-200 exact data-channel changes.")
    data, layers = scene.data_viewer(), scene.layers_viewer()
    plan, seen, keys_to_create = [], set(), set()
    for change in changes:
        if type(change) is not dict or set(change) != {"object", "data", "value"}:
            raise ValueError("Each change requires exactly object, data and value.")
        object_id = common.resolve_object(scene, change["object"])
        reference = _resolve(scene, object_id, change["data"])
        if reference in seen:
            raise ValueError("Each native data channel may be changed only once in a batch.")
        seen.add(reference)
        layer_id = layers.layer_id_by_obj_id_or_null(object_id)
        if layer_id.is_null() or layers.layer(layer_id).is_locked:
            raise ValueError("Data owner must belong to an unlocked native animation track.")
        mode = data.get_data(reference).mode
        if mode not in (csc.model.DataMode.Static, csc.model.DataMode.Animation):
            raise ValueError("Only native Static and Animation data modes are supported.")
        if mode == csc.model.DataMode.Animation and not layers.layer(layer_id).is_key(position):
            if not create_keys:
                raise ValueError("Animated data writes need an existing key or create_keys=true.")
            if position == animation._size(scene) - 1:
                raise ValueError("Creating a terminal key requires an allocated successor frame.")
            keys_to_create.add(layer_id)
        template = read_data_value(data, reference, position)
        value = animation._coerce_property_value(csc, template, change["value"])
        plan.append((reference, mode, value))

    def apply(model, updater):
        editor = model.data_editor()
        for layer_id in keys_to_create:
            model.layers_editor().set_fixed_interpolation_or_key_if_need(layer_id, position, True)
        for reference, mode, value in plan:
            if mode == csc.model.DataMode.Static:
                editor.set_data_value(reference, value)
            else:
                editor.set_data_value(reference, position, value)
        updater.run_update({reference for reference, _, _ in plan}, position)
        if keys_to_create:
            model.layers_editor().normalize_sections(scene)

    common.modify(scene, "Set exact native object data", apply)
    rows, mismatches = [], []
    for reference, mode, expected in plan:
        actual = read_data_value(data, reference, position)
        if not animation._native_values_match(expected, actual):
            mismatches.append(str(reference))
        rows.append(
            {
                **_record(scene, reference, position),
                "scope": "all frames" if mode == csc.model.DataMode.Static else "frame",
            }
        )
    if mismatches:
        raise RuntimeError(
            "Native data readback differs from requested values (derived outputs may be solver-owned): "
            + ", ".join(mismatches)
        )
    return {
        "execution_state": "completed",
        "frame": position,
        "channels": rows,
        "created_keys": sorted(map(str, keys_to_create)),
        "verification": {"matches": True},
        "note": "This is a generic data edit. Static channels affect all frames; use checked character tools for solver-aware body/hand editing.",
    }


_IDENTIFIER = {"type": "string", "minLength": 1, "maxLength": 512}
TOOLS = [
    Tool(
        "object_data_read",
        "Discover every native data channel on an object, including channels without behavior bindings; read exact IDs/display names and values at an existing frame.",
        object_data_read,
        {
            "object": _IDENTIFIER,
            "data": _IDENTIFIER,
            "frame": animation.FRAME,
            "query": {"type": "string", "maxLength": 200},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        ("object",),
    ),
    Tool(
        "object_data_set",
        "Write 1-200 exact native data IDs or unique display names with typed prevalidation, unlocked-track/key guards and actual readback. Includes data without behavior bindings; solver-owned outputs may reject edits.",
        object_data_set,
        {
            "changes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 200,
                "items": {
                    "type": "object",
                    "properties": {"object": _IDENTIFIER, "data": _IDENTIFIER, "value": {}},
                    "required": ["object", "data", "value"],
                    "additionalProperties": False,
                },
            },
            "frame": animation.FRAME,
            "create_keys": {"type": "boolean"},
        },
        ("changes",),
        False,
    ),
]
