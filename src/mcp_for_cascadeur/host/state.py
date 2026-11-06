"""Bounded native model checkpoints, with explicit comparison and restore scope.

No native wrappers survive a call. Assets are protected by identity and readable
payload fingerprints; opaque texture/blendshape contents are outside model Undo.
"""

from __future__ import annotations

import hashlib
import json
import math

MAX_FRAMES = 512
MAX_OBJECTS = 1024
MAX_BYTES = 64 * 1024 * 1024
MAX_NODES = 4_000_000
MAX_CHANNEL_SAMPLES = 2_000_000
POSITION_TOLERANCE = 0.01
ROTATION_TOLERANCE = 0.0001


class StateError(RuntimeError):
    def __init__(self, code, message, *, execution_state="not_started", details=None):
        super().__init__(message)
        self.code = code
        self.execution_state = execution_state
        self.details = details


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _identifier(value):
    return value.to_string() if hasattr(value, "to_string") else str(value)


def _attribute(value, name):
    found = getattr(value, name)
    return found() if callable(found) else found


def _euler_values(value):
    angles = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(angles, (list, tuple)) and len(angles) == 3:
        if all(isinstance(item, (list, tuple)) and len(item) == 1 for item in angles):
            angles = [item[0] for item in angles]
        if all(type(item) in (float, int) and math.isfinite(item) for item in angles):
            return [float(item) for item in angles]
    raise StateError("INVALID_ROTATION_STATE", "Native Euler rotation needs three finite radians")


def encode(value, _depth=0):
    """Preserve native numeric types/shapes instead of coercing unknown objects."""
    if _depth > 40:
        raise StateError("STATE_VALUE_LIMIT", "Native value nesting exceeds the checkpoint limit")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StateError("NONFINITE_STATE", "The native scene contains a nonfinite value")
        return value
    if hasattr(value, "to_string"):
        return {"__native__": "id", "value": value.to_string()}
    rotation = hasattr(value, "to_quaternion")
    quaternion = value.to_quaternion() if rotation else value
    if all(callable(getattr(quaternion, component, None)) for component in "wxyz"):
        values = [float(getattr(quaternion, component)()) for component in "wxyz"]
        if not all(math.isfinite(component) for component in values):
            raise StateError("NONFINITE_STATE", "The native scene contains a nonfinite rotation")
        sign = next((-1 if component < 0 else 1 for component in values if abs(component) > 1e-12), 1)
        result = {
            "__native__": "rotation" if rotation else "quaternion",
            "quaternion_wxyz": [component * sign for component in values],
        }
        if rotation and callable(getattr(value, "to_euler_angles", None)):
            result["euler_radians"] = _euler_values(value.to_euler_angles())
        return result
    if hasattr(value, "tolist"):
        items = encode(value.tolist(), _depth + 1)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return {
                "__native__": "ndarray",
                "dtype": str(value.dtype),
                "shape": [int(size) for size in value.shape],
                "items": items,
            }
        return items
    if isinstance(value, (list, tuple)):
        return [encode(item, _depth + 1) for item in value]
    if isinstance(value, set):
        return sorted((encode(item, _depth + 1) for item in value), key=canonical_bytes)
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: encode(item, _depth + 1) for key, item in value.items()}
    raise StateError("UNSUPPORTED_STATE_VALUE", f"Checkpoint cannot encode native {type(value).__name__}")


def decode(value):
    if isinstance(value, dict) and "__native__" in value:
        kind = value["__native__"]
        if kind == "ndarray":
            import numpy as np

            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "biuf":
                raise StateError("UNSUPPORTED_STATE_VALUE", "Only numeric native arrays can be restored")
            result = np.asarray(value["items"], dtype=dtype)
            if list(result.shape) != value["shape"]:
                raise StateError("INVALID_CHECKPOINT", "Native array shape differs from its checkpoint")
            return result
        import csc

        if kind == "rotation":
            if "euler_radians" in value:
                return csc.math.Rotation.from_euler(*_euler_values(value["euler_radians"]))
            return csc.math.Rotation.from_quaternion(*value["quaternion_wxyz"])
        if kind == "quaternion":
            return csc.math.Quaternion(*value["quaternion_wxyz"])
        if kind == "id":
            return csc.Guid(value["value"])
        raise StateError("UNSUPPORTED_STATE_VALUE", f"Unknown checkpoint value type: {kind}")
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def section_state(section):
    def common(part):
        return {"ik_fk": part.ik_fk.name, "fixation": part.fixation.name}

    key = section.key
    return {
        "interpolation": section.interval.interpolation.name,
        "interval_common": common(section.interval.common),
        "ai_description": section.interval.description_for_ai_interpolation,
        "key": None
        if key is None
        else {
            "common": common(key.common),
            "label": encode(key.label),
            "tangents": key.tangents.name,
            "left_weight_velocity": encode(float(key.left_weight_velocity)),
            "right_weight_velocity": encode(float(key.right_weight_velocity)),
        },
    }


def apply_section_metadata(result, value):
    """Edit an existing native section; never build detached C++ section wrappers."""
    import csc

    native = csc.layers.layer
    saved = value["key"]
    if (saved is None) != (result.key is None):
        raise StateError(
            "UNSUPPORTED_SECTION_RESTORE",
            "Native key/fixed-section creation did not produce the required key type",
        )
    result.interval.interpolation = native.Interpolation.__members__[value["interpolation"]]
    result.interval.common.ik_fk = native.IkFk.__members__[value["interval_common"]["ik_fk"]]
    result.interval.common.fixation = native.Fixation.__members__[value["interval_common"]["fixation"]]
    result.interval.description_for_ai_interpolation = value["ai_description"]
    if saved is not None:
        result.key.common.ik_fk = native.IkFk.__members__[saved["common"]["ik_fk"]]
        result.key.common.fixation = native.Fixation.__members__[saved["common"]["fixation"]]
        result.key.label = decode(saved["label"])
        result.key.tangents = native.Tangents.__members__[saved["tangents"]]
        result.key.left_weight_velocity = saved["left_weight_velocity"]
        result.key.right_weight_velocity = saved["right_weight_velocity"]


_PROPERTY_GETTERS = {
    "ObjectType": "get_behaviour_object",
    "ObjectRangeType": "get_behaviour_objects_range",
    "BehaviourType": "get_behaviour_reference",
    "BehaviourRangeType": "get_behaviour_reference_range",
    "DataType": "get_behaviour_data",
    "DataRangeType": "get_behaviour_data_range",
    "SettingType": "get_behaviour_setting",
    "SettingRangeType": "get_behaviour_settings_range",
    "AssetType": "get_behaviour_asset",
    "AssetRangeType": "get_behaviour_asset_range",
    "StringType": "get_behaviour_string",
}


def _assets(scene):
    manager = scene.assets_manager()
    result = {}
    for identifier in manager.all_ids():
        asset = manager.at(identifier)
        name = type(asset).__name__
        info = {"type": name, "link": manager.find_link(identifier)}
        if name == "Mesh":
            payload = {key: encode(getattr(asset, key)()) for key in ("positions", "normals", "uvs")}
            payload["vertices"] = [
                [item.position_index, item.normal_index, item.uv_index] for item in asset.vertices()
            ]
            payload["vertices_in_polygon"] = asset.vertices_in_polygon()
            info["payload_sha256"] = digest(payload)
        elif name == "MeshDependency":
            payload = {
                "matrices": encode(asset.inverse_bind_matrices()),
                "weights": [
                    [[item.index, encode(float(item.weight))] for item in row]
                    for row in asset.vertices_weights()
                ],
            }
            info["payload_sha256"] = digest(payload)
        else:
            info["payload_scope"] = "opaque; identity/reference protection only"
        result[_identifier(identifier)] = info
    return result


def _auxiliary(scene, layers, data):
    import csc

    folders = {
        _identifier(key): {
            "name": folder.header.name,
            "parent": _identifier(folder.header.parent),
            "children": [_identifier(child) for child in folder.children_ordered()],
        }
        for key, folder in layers.folders_map().items()
    }
    clusters = data.cluster_viewer()
    cluster_data = {
        str(index): {
            "name": clusters.cluster_name(index),
            "data": sorted(_identifier(item) for item in clusters.cluster_datas(index)),
        }
        for index in clusters.clusters()
    }
    manager = csc.additive_layers.get_manager(scene)
    stacked = manager.get_stacked_layers_count() if manager is not None else 0
    return {
        "folders": folders,
        "root_folder": _identifier(layers.root_id()),
        "clusters": cluster_data,
        "cluster_bindings": sorted(
            [[_attribute(edge, "a"), _attribute(edge, "b")] for edge in clusters.clusters_bindings()]
        ),
        "descriptions": {
            name: data.get_any_description_value_by_name(name)
            for name in sorted(data.get_description_names())
        },
        "additive": {
            "stacked_layers": stacked,
            "scope": "base model only" if stacked else "no additive stack",
        },
    }


def capture(scene, *, max_frames=MAX_FRAMES, max_objects=MAX_OBJECTS):
    """Read every stored model channel; never move the playhead or modify the scene."""
    data, model = scene.data_viewer(), scene.model_viewer()
    behaviours, layers = scene.behaviour_viewer(), scene.layers_viewer()
    count = int(data.get_animation_size())
    objects = list(model.get_objects())
    if not 1 <= count <= max_frames or len(objects) > max_objects:
        raise StateError(
            "STATE_CAPTURE_LIMIT", f"Checkpoint supports at most {max_frames} frames/{max_objects} objects"
        )
    result = {
        "format_version": 1,
        "count": count,
        "objects": {},
        "channels": {},
        "tracks": {},
        "auxiliary": {},
        "external": {},
    }
    samples = 0
    for obj in sorted(objects, key=_identifier):
        oid = _identifier(obj)
        record = {
            "name": model.get_object_name(obj),
            "type": model.get_object_type_name(obj),
            "behaviours": {},
        }
        for behaviour in sorted(behaviours.get_behaviours(obj), key=_identifier):
            props = {}
            for name in sorted(behaviours.get_behaviour_property_names(behaviour)):
                kind = behaviours.get_property_type(behaviour, name).name
                if kind not in _PROPERTY_GETTERS:
                    raise StateError(
                        "UNSUPPORTED_STATE_PROPERTY", f"No complete checkpoint reader for {kind}"
                    )
                value = getattr(behaviours, _PROPERTY_GETTERS[kind])(behaviour, name)
                props[name] = {"kind": kind, "value": encode(value)}
            record["behaviours"][_identifier(behaviour)] = {
                "name": behaviours.get_behaviour_name(behaviour),
                "hidden": bool(behaviours.is_hidden(behaviour)),
                "properties": props,
            }
        result["objects"][oid] = record
        for kind in ("data", "setting"):
            identifiers = data.get_all_data_id(obj) if kind == "data" else data.get_all_settings_id(obj)
            for identifier in identifiers:
                info = (data.get_data if kind == "data" else data.get_setting)(identifier)
                getter = data.get_data_value if kind == "data" else data.get_setting_value
                mode = info.mode.name
                if mode not in {"Static", "Animation"}:
                    raise StateError("UNSUPPORTED_STATE_MODE", f"No complete reader for {mode} channels")
                item = {
                    "owner": oid,
                    "id": _identifier(identifier),
                    "name": info.name,
                    "kind": kind,
                    "mode": mode,
                }
                if kind == "setting":
                    item["setting_type"] = int(info.type)
                description = data.get_description_id_by_linked_id(identifier)
                item["description"] = (
                    None
                    if description.is_null()
                    else {
                        "id": _identifier(description),
                        "name": data.get_description_name(description),
                        "value": data.get_description_value(description),
                    }
                )
                if mode == "Animation":
                    samples += count
                    if samples > MAX_CHANNEL_SAMPLES:
                        raise StateError(
                            "STATE_CAPTURE_LIMIT", "Checkpoint channel/frame sample budget exceeded"
                        )
                    item["values"] = [encode(getter(identifier, frame)) for frame in range(count)]
                else:
                    item["value"] = encode(getter(identifier))
                result["channels"][f"{kind}:{_identifier(identifier)}"] = item
    import csc

    for identifier, track in layers.layers_map().items():
        cycles = csc.layers.CyclesViewer(track).get_cycles_in_frames(0, count - 1)
        result["tracks"][_identifier(identifier)] = {
            "name": track.header.name,
            "parent": _identifier(track.header.parent),
            "objects": sorted(_identifier(item) for item in track.obj_ids),
            "locked": bool(track.is_locked),
            "visible": bool(track.is_visible),
            "keys": list(track.key_frame_indices()),
            "sections": {str(frame): section_state(section) for frame, section in track.sections.items()},
            "cycles": [
                {
                    name: encode(_attribute(cycle, name))
                    for name in (
                        "left_frame_index",
                        "right_frame_index",
                        "first_active_frame_index",
                        "last_active_frame_index",
                        "left_inactive_frame_index",
                        "right_inactive_frame_index",
                    )
                }
                for cycle in cycles
            ],
        }
    result["auxiliary"] = _auxiliary(scene, layers, data)
    result["external"] = {
        "assets": _assets(scene),
        "scope": "asset references and readable payload fingerprints; opaque payloads excluded",
    }
    validate(result)
    return result


def validate(checkpoint):
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint)
        != {"format_version", "count", "objects", "channels", "tracks", "auxiliary", "external"}
        or type(checkpoint["format_version"]) is not int
        or checkpoint["format_version"] != 1
    ):
        raise StateError("INVALID_CHECKPOINT", "Unsupported native checkpoint format")
    if type(checkpoint["count"]) is not int or not 1 <= checkpoint["count"] <= MAX_FRAMES:
        raise StateError("INVALID_CHECKPOINT", "Checkpoint animation bounds are invalid")
    if any(
        type(checkpoint[key]) is not dict
        for key in ("objects", "channels", "tracks", "auxiliary", "external")
    ):
        raise StateError("INVALID_CHECKPOINT", "Checkpoint model sections must be objects")
    if len(checkpoint["objects"]) > MAX_OBJECTS:
        raise StateError("INVALID_CHECKPOINT", "Checkpoint object count exceeds its bound")
    for channel in checkpoint["channels"].values():
        if not isinstance(channel, dict):
            raise StateError("INVALID_CHECKPOINT", "Native channel records must be JSON objects")
        if channel.get("kind") not in {"data", "setting"} or channel.get("mode") not in {
            "Static",
            "Animation",
        }:
            raise StateError("INVALID_CHECKPOINT", "Invalid native channel metadata")
        if channel["mode"] == "Animation" and len(channel.get("values", [])) != checkpoint["count"]:
            raise StateError("INVALID_CHECKPOINT", "Animated channel is missing stored frames")
    budget = [MAX_NODES]

    def visit(value, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 48:
            raise StateError("STATE_CAPTURE_LIMIT", "Checkpoint exceeds its nesting/node budget")
        if isinstance(value, dict):
            if value.get("__native__") == "rotation" and "euler_radians" in value:
                _euler_values(value["euler_radians"])
            for key, item in value.items():
                if type(key) is not str:
                    raise StateError("INVALID_CHECKPOINT", "Checkpoint keys must be strings")
                visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                visit(item, depth + 1)
        elif type(value) is float and not math.isfinite(value):
            raise StateError("INVALID_CHECKPOINT", "Checkpoint contains nonfinite numbers")
        elif value is not None and type(value) not in (str, int, float, bool):
            raise StateError("INVALID_CHECKPOINT", "Checkpoint contains a non-JSON value")

    visit(checkpoint)
    if len(canonical_bytes(checkpoint)) > MAX_BYTES:
        raise StateError("STATE_CAPTURE_LIMIT", "Checkpoint exceeds the byte budget")


def equivalent_values(first, second, *, exact=False):
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        if first.get("__native__") in {"rotation", "quaternion"}:
            if first.get("__native__") != second.get("__native__"):
                return False
            optional = {"euler_radians"} if first["__native__"] == "rotation" else set()
            if first.keys() - optional != second.keys() - optional:
                return False

            def same_components(key, count):
                left, right = first.get(key), second.get(key)
                if (
                    type(left) is not list
                    or type(right) is not list
                    or len(left) != count
                    or len(right) != count
                ):
                    return False
                return (
                    left == right
                    if exact
                    else all(
                        math.isclose(
                            a,
                            b,
                            abs_tol=ROTATION_TOLERANCE,
                            rel_tol=0.0 if key == "euler_radians" else 1e-6,
                        )
                        for a, b in zip(left, right)
                    )
                )

            if not same_components("quaternion_wxyz", 4):
                return False
            if (
                "euler_radians" in first
                and "euler_radians" in second
                and not same_components("euler_radians", 3)
            ):
                return False
            return all(
                equivalent_values(value, second[key], exact=exact)
                for key, value in first.items()
                if key not in {"quaternion_wxyz", "euler_radians"}
            )
        if first.keys() != second.keys():
            return False
        if not exact and first.get("__native__") == "ndarray":
            if any(first[key] != second[key] for key in ("__native__", "dtype", "shape")):
                return False
            if first["shape"] in ([3, 3], [4, 4]):
                return all(
                    math.isclose(
                        a, b, rel_tol=1e-6, abs_tol=POSITION_TOLERANCE if column == 3 else ROTATION_TOLERANCE
                    )
                    for row_a, row_b in zip(first["items"], second["items"])
                    for column, (a, b) in enumerate(zip(row_a, row_b))
                )
        return all(equivalent_values(value, second[key], exact=exact) for key, value in first.items())
    if isinstance(first, list):
        return len(first) == len(second) and all(
            equivalent_values(a, b, exact=exact) for a, b in zip(first, second)
        )
    if type(first) is float and not exact:
        return math.isclose(first, second, abs_tol=POSITION_TOLERANCE, rel_tol=1e-6)
    return first == second


def channel_equal(first, second):
    values = "values" if first.get("mode") == "Animation" else "value"
    if {key: value for key, value in first.items() if key != values} != {
        key: value for key, value in second.items() if key != values
    }:
        return False
    exact = first.get("kind") == "setting" or first.get("mode") != "Animation"
    return equivalent_values(first.get(values), second.get(values), exact=exact)


def compare(expected, actual):
    differences = []
    for key in sorted(set(expected) | set(actual)):
        if key == "channels":
            for identifier in sorted(set(expected.get(key, {})) | set(actual.get(key, {}))):
                left, right = expected.get(key, {}).get(identifier), actual.get(key, {}).get(identifier)
                if left is None or right is None or not channel_equal(left, right):
                    differences.append({"path": f"channels/{identifier}", "reason": "channel state differs"})
        elif not equivalent_values(expected.get(key), actual.get(key), exact=True):
            differences.append({"path": key, "reason": "exact metadata differs"})
    return {
        "equivalent": not differences,
        "difference_count": len(differences),
        "differences": differences[:100],
        "numeric_policy": {
            "animated_data_absolute": POSITION_TOLERANCE,
            "rotation_absolute": ROTATION_TOLERANCE,
            "native_euler_absolute": ROTATION_TOLERANCE,
            "native_euler_relative": 0.0,
            "relative": 1e-6,
            "static_settings_metadata": "exact",
        },
    }


def _structural_identity(checkpoint):
    objects = {}
    for oid, item in checkpoint["objects"].items():
        objects[oid] = {
            "type": item["type"],
            "behaviours": {
                bid: {
                    "name": behaviour["name"],
                    "properties": {
                        name: value
                        for name, value in behaviour["properties"].items()
                        if value["kind"] != "StringType"
                    },
                }
                for bid, behaviour in item["behaviours"].items()
            },
        }
    return {
        "objects": objects,
        "channels": {
            key: {name: value for name, value in item.items() if name not in {"value", "values"}}
            for key, item in checkpoint["channels"].items()
        },
        "tracks": {key: {"objects": item["objects"]} for key, item in checkpoint["tracks"].items()},
        "auxiliary": checkpoint["auxiliary"],
        "external": checkpoint["external"],
    }


def _data_property(item, name):
    for behaviour in item["behaviours"].values():
        if behaviour["name"] != "Transform":
            continue
        prop = behaviour["properties"].get(name)
        if prop is not None and prop["kind"] == "DataType":
            value = prop["value"]
            return "data:" + (value["value"] if isinstance(value, dict) else value)
    return None


def _restore_authoring_plan(scene, checkpoint, before):
    """Separate authored inputs from solver outputs before touching native data."""
    channels = checkpoint["channels"]
    owner_frames = {}
    static_owners = set()
    for key, item in channels.items():
        old = before["channels"][key]
        if item["mode"] == "Static":
            if not channel_equal(item, old):
                static_owners.add(item["owner"])
            continue
        for frame, value in enumerate(item["values"]):
            previous = old["values"][frame] if frame < len(old["values"]) else None
            if not equivalent_values(value, previous, exact=item["kind"] == "setting"):
                owner_frames.setdefault(item["owner"], set()).add(frame)
    for identifier, track in checkpoint["tracks"].items():
        if track != before["tracks"][identifier]:
            for owner in track["objects"]:
                owner_frames.setdefault(owner, set()).update(range(checkpoint["count"]))
    for owner in static_owners:
        owner_frames.setdefault(owner, set()).update(range(checkpoint["count"]))

    groups, rig_owned = [], set()
    rig_ids = [
        identifier
        for identifier, item in checkpoint["objects"].items()
        if any(behaviour["name"] == "RigInfo" for behaviour in item["behaviours"].values())
    ]
    if rig_ids:
        from . import curves

        try:
            curves.verify_supported(scene)
        except Exception as error:
            raise StateError(
                "UNSUPPORTED_AUTHORED_RESTORE",
                "Current native curves are outside the verified character recovery contract",
            ) from error
        if not 1 <= checkpoint["count"] <= 121:
            raise StateError(
                "UNSUPPORTED_AUTHORED_RESTORE", "Character restoration accepts 1-121 stored frames"
            )
        for track in checkpoint["tracks"].values():
            for section in track["sections"].values():
                if section["interpolation"] == "CLAMPED_BEZIER" or (
                    section["key"] is not None and section["key"]["tangents"] != "Continuous"
                ):
                    raise StateError(
                        "UNSUPPORTED_AUTHORED_RESTORE",
                        "Checkpoint curves require unverified native interpolation or custom tangents",
                    )
    for identifier in rig_ids:
        from .characters import resolve_character

        try:
            rig = resolve_character(scene, identifier, require_hands=True)
        except Exception as error:
            raise StateError(
                "UNSUPPORTED_AUTHORED_RESTORE",
                "Native character authored channels could not be verified; reopen the checkpoint scene",
            ) from error
        owned = {_identifier(obj) for obj in rig.owned}
        if rig_owned.intersection(owned):
            raise StateError(
                "UNSUPPORTED_AUTHORED_RESTORE", "Character ownership overlaps during restoration"
            )
        rig_owned.update(owned)
        frames = set().union(*(owner_frames.get(owner, set()) for owner in owned))
        point_keys = {"data:" + _identifier(binding.position_data) for binding in rig.bindings.values()}
        finger_keys = {
            "data:" + _identifier(binding.rotation_data)
            for hand in rig.fingers.values()
            for binding in hand.values()
        }
        if point_keys.intersection(finger_keys) or not point_keys:
            raise StateError(
                "UNSUPPORTED_AUTHORED_RESTORE", "Character authored input identities are ambiguous"
            )
        keys = point_keys | finger_keys
        if keys - channels.keys() or any(channels[key]["mode"] != "Animation" for key in keys):
            raise StateError(
                "UNSUPPORTED_AUTHORED_RESTORE", "Character authored inputs differ from the checkpoint"
            )
        controls = {_identifier(obj) for obj in rig.controls.values()}
        for snapshot in (checkpoint, before):
            for track in snapshot["tracks"].values():
                members = set(track["objects"])
                if not members.intersection(owned):
                    continue
                if members - owned:
                    raise StateError(
                        "UNSUPPORTED_AUTHORED_RESTORE", "A character track contains unrelated objects"
                    )
                for section in track["sections"].values():
                    if section["interpolation"] == "AI":
                        raise StateError(
                            "UNSUPPORTED_AUTHORED_RESTORE", "Character AI curve restoration is unverified"
                        )
                    if members.intersection(controls) and (
                        section["interval_common"]["ik_fk"] != "IK"
                        or (section["key"] is not None and section["key"]["common"]["ik_fk"] != "IK")
                    ):
                        raise StateError(
                            "UNSUPPORTED_AUTHORED_RESTORE",
                            "Character Point restoration requires native IK tracks",
                        )
        terminal = checkpoint["count"] - 1
        if terminal in frames:
            for key in keys:
                previous = before["channels"][key]["values"]
                if terminal >= len(previous) or not equivalent_values(
                    channels[key]["values"][terminal], previous[terminal]
                ):
                    raise StateError(
                        "TERMINAL_FRAME_RESTORE_UNSUPPORTED",
                        "A changed terminal character input needs a following solver interval; reopen the checkpoint scene",
                    )
            frames.remove(terminal)
        groups.append(
            {"kind": "character", "owners": owned, "frames": frames, "passes": [point_keys, finger_keys]}
        )

    allowed = {"Basic", "Transform", "Joint", "Node3d", "FbxExport", "JointExtraAttributes"}
    for owner, frames in owner_frames.items():
        if owner in rig_owned:
            continue
        item = checkpoint["objects"][owner]
        kinds = {behaviour["name"] for behaviour in item["behaviours"].values()}
        animated = {
            key
            for key, channel in channels.items()
            if channel["owner"] == owner and channel["kind"] == "data" and channel["mode"] == "Animation"
        }
        if not animated:
            continue
        if kinds - allowed:
            raise StateError(
                "UNSUPPORTED_AUTHORED_RESTORE",
                "Unknown constrained object has changed animated data; reopen a matching checkpoint scene",
            )
        inputs = set()
        for suffix in ("position", "rotation"):
            key = _data_property(item, "local_" + suffix) or _data_property(item, "global_" + suffix)
            if key in animated:
                inputs.add(key)
        if not inputs and len(animated) == 1 and kinds <= {"Basic", "Transform"}:
            inputs = set(animated)
        if not inputs:
            raise StateError(
                "UNSUPPORTED_AUTHORED_RESTORE", "No independent authored Transform inputs were identified"
            )
        groups.append(
            {"kind": "independent_transform", "owners": {owner}, "frames": frames, "passes": [inputs]}
        )
    return groups


def restore(scene, checkpoint):
    """Restore existing identities and verify all stored frames after native commit.

    Object/channel/asset creation and opaque external edits require reopening a
    matching native scene. This method never fabricates missing identities.
    """
    validate(checkpoint)
    before = capture(scene)
    if compare(checkpoint, before)["equivalent"]:
        return {"restored": True, "method": "already_equal", "verification": compare(checkpoint, before)}
    if not equivalent_values(_structural_identity(checkpoint), _structural_identity(before), exact=True):
        raise StateError(
            "UNSUPPORTED_STATE_RESTORE",
            "Topology, channel identity, assets or auxiliary structures differ; reopen a matching native checkpoint",
        )
    if checkpoint["auxiliary"].get("additive", {}).get("stacked_layers") or any(
        item.get("cycles") for item in checkpoint["tracks"].values()
    ):
        raise StateError(
            "UNSUPPORTED_STATE_RESTORE",
            "Manual restoration of additive stacks or cycles is not supported; checked native Undo is required",
        )
    authoring = _restore_authoring_plan(scene, checkpoint, before)
    model, data = scene.model_viewer(), scene.data_viewer()
    objects = {_identifier(item): item for item in model.get_objects()}
    handles = {}
    for obj in objects.values():
        for kind, getter in (("data", data.get_all_data_id), ("setting", data.get_all_settings_id)):
            handles.update({f"{kind}:{_identifier(item)}": item for item in getter(obj)})
    layers = {_identifier(item): item for item in scene.layers_viewer().all_layer_ids()}
    failures = []

    def apply(editor, _updates, updater):
        try:
            layer_editor = editor.layers_editor()
            changed_tracks = {
                identifier: item
                for identifier, item in checkpoint["tracks"].items()
                if item != before["tracks"][identifier]
            }
            layout_changed = False
            for identifier, item in changed_tracks.items():
                handle = layers[identifier]
                layer_editor.set_locked_for_layer(False, handle)
                old_sections = before["tracks"][identifier]["sections"]
                desired = item["sections"]
                for frame, saved in desired.items():
                    old = old_sections.get(frame)
                    different_key_kind = old is not None and ((old["key"] is None) != (saved["key"] is None))
                    if different_key_kind:
                        layer_editor.unset_section(int(frame), handle)
                    if old is None or different_key_kind:
                        layer_editor.set_fixed_interpolation_or_key_if_need(
                            handle, int(frame), saved["key"] is not None
                        )
                        layout_changed = True
                for frame in set(old_sections) - set(desired):
                    layer_editor.unset_section(int(frame), handle)
                    layout_changed = True
            if layout_changed:
                layer_editor.normalize_sections(scene)
                updater.generate_update()
            for identifier, item in checkpoint["tracks"].items():
                if identifier in changed_tracks:
                    for frame, metadata in item["sections"].items():
                        layer_editor.change_section(
                            int(frame),
                            layers[identifier],
                            lambda section, saved=metadata: apply_section_metadata(section, saved),
                        )
                    layer_editor.set_locked_for_layer(item["locked"], layers[identifier])
                    layer_editor.set_visible_for_layer(item["visible"], layers[identifier])
                    layer_editor.set_name(item["name"], layers[identifier])
            if before["count"] != checkpoint["count"]:
                editor.fit_animation_size_by_layers()
                updater.generate_update()
            for oid, item in checkpoint["objects"].items():
                if item["name"] != before["objects"][oid]["name"]:
                    editor.set_object_name(objects[oid], item["name"])
                for bid, behaviour in item["behaviours"].items():
                    old = before["objects"][oid]["behaviours"][bid]
                    import csc

                    handle = csc.model.BehaviourId(bid)
                    if behaviour["hidden"] != old["hidden"]:
                        editor.behaviour_editor().hide_behaviour(handle, behaviour["hidden"])
                    for prop, value in behaviour["properties"].items():
                        if value["kind"] == "StringType" and value != old["properties"][prop]:
                            editor.behaviour_editor().set_behaviour_string(handle, prop, value["value"])
            changed = {
                key: item
                for key, item in checkpoint["channels"].items()
                if not channel_equal(item, before["channels"][key])
            }
            de = editor.data_editor()
            for key, item in changed.items():
                setter = de.set_data_value if item["kind"] == "data" else de.set_setting_value
                if item["mode"] == "Static":
                    setter(handles[key], decode(item["value"]))
            updater.generate_update()
            for frame in range(checkpoint["count"]):
                for key, item in changed.items():
                    if item["mode"] != "Animation" or item["kind"] != "setting":
                        continue
                    previous = before["channels"][key]["values"]
                    if frame >= len(previous) or item["values"][frame] != previous[frame]:
                        de.set_setting_value(handles[key], frame, decode(item["values"][frame]))
                for group in authoring:
                    if frame not in group["frames"]:
                        continue
                    for keys in group["passes"]:
                        for key in keys:
                            item, previous = checkpoint["channels"][key], before["channels"][key]["values"]
                            if frame >= len(previous) or item["values"][frame] != previous[frame]:
                                de.set_data_value(handles[key], frame, decode(item["values"][frame]))
                    for keys in group["passes"]:
                        if keys:
                            updater.run_update({handles[key] for key in keys}, frame)
            interpolator = updater.get_interpolator()
            interpolator.reload()
            interpolator.interpolate()
        except Exception as error:
            failures.append(error)
            raise

    from . import recovery

    recovery.note_native_attempt(scene)
    accepted = scene.modify_update("Restore verified MCP model checkpoint", apply)
    if failures or not accepted:
        raise StateError(
            "STATE_RESTORE_FAILED",
            str(failures[0]) if failures else "Native restore was rejected",
            execution_state="recovery_required",
        )
    after = capture(scene)
    recovery.note_native_commit(scene, checkpoint=after)
    report = compare(checkpoint, after)
    if not report["equivalent"]:
        raise StateError(
            "STATE_RESTORE_MISMATCH",
            "Restored model differs from the complete recorded checkpoint",
            execution_state="recovery_required",
            details=report,
        )
    recovery.accept_verified_checkpoint(scene, after)
    return {
        "restored": True,
        "method": "native_authored_input_restore",
        "verification": report,
        "authoring_groups": [
            {
                "kind": group["kind"],
                "frames": sorted(group["frames"]),
                "input_channels": sum(map(len, group["passes"])),
            }
            for group in authoring
        ],
    }
