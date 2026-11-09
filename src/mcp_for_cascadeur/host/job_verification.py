"""Protect the complete native model outside a durable job's declared scope."""

from __future__ import annotations

import copy
import re

from . import state

MAX_REPORTED_ERRORS = 100
_NULL_ID = "00000000-0000-0000-0000-000000000000"
_REFERENCE_BEHAVIOURS = {
    "Basic",
    "Node3d",
    "Transform",
    "CustomTransformData",
    "MeshObject",
    "TextureContainer",
    "Material",
    "FbxExport",
}


def _metadata(channel):
    return {key: value for key, value in channel.items() if key not in {"value", "values"}}


def protected_report(before, after, plan, result):
    """Compare actual checkpoints; authorized samples are not quality-checked here.

    Animated data and settings of the declared owners may change only inside
    the inclusive job interval. Static data, identities and structural metadata
    remain exact. Timeline growth holds protected channels at their last stored
    value, rather than silently excluding newly allocated frames from checks.
    """
    errors = []
    error_count = 0
    counts = {
        "objects": 0,
        "channels": 0,
        "static_channels": 0,
        "animated_samples": 0,
        "editable_samples": 0,
        "tracks": 0,
        "reference_objects": 0,
        "reference_channels": 0,
        "reference_assets": 0,
        "guard_anchors": 0,
        "periodic_euler_samples": 0,
    }

    def reject(path, reason):
        nonlocal error_count
        error_count += 1
        if len(errors) < MAX_REPORTED_ERRORS:
            errors.append({"path": path, "reason": reason})

    def report():
        return {"matches": not error_count, "error_count": error_count, "errors": errors, "counts": counts}

    for name, snapshot in (("before", before), ("after", after)):
        try:
            state.validate(snapshot)
        except (state.StateError, KeyError, TypeError, ValueError) as error:
            reject(name, f"Invalid complete model checkpoint: {error}")
    if error_count:
        return report()
    if type(plan) is not dict or type(result) is not dict:
        reject("plan", "A native job plan and result are required")
        return report()

    first, last = plan.get("first"), plan.get("last")
    expected_count = plan.get("expected_count", before["count"])
    if (
        type(first) is not int
        or type(last) is not int
        or type(expected_count) is not int
        or not 0 <= first <= last < expected_count <= state.MAX_FRAMES
        or expected_count < before["count"]
    ):
        reject("plan", "Invalid inclusive frame scope or expected stored-frame count")
        return report()
    if after["count"] != expected_count:
        reject("count", "The actual stored-frame count differs from the native job plan")

    selected = plan.get("tracks")
    if type(selected) is not list or not selected or any(type(item) is not str for item in selected):
        reject("plan/tracks", "The native job needs explicit original track IDs")
        return report()
    if len(set(selected)) != len(selected) or set(selected) - before["tracks"].keys():
        reject("plan/tracks", "Selected track identities are duplicated or absent from the original model")
        return report()
    owners = plan.get("owned_objects")
    if owners is None:
        owners = sorted(
            {owner for identifier in selected for owner in before["tracks"][identifier]["objects"]}
        )
    if (
        type(owners) is not list
        or not owners
        or any(type(item) is not str for item in owners)
        or len(set(owners)) != len(owners)
        or set(owners) - before["objects"].keys()
    ):
        reject("plan/owned_objects", "Editable owners must be distinct existing native object IDs")
        return report()
    owners = set(owners)
    if not {owner for identifier in selected for owner in before["tracks"][identifier]["objects"]}.issubset(
        owners
    ):
        reject("plan/owned_objects", "The editable owner scope excludes a selected track member")
    guard_anchors = _guard_anchors(before, after, plan, selected, reject)
    counts["guard_anchors"] = len(guard_anchors)

    for identifier, old in before["objects"].items():
        counts["objects"] += 1
        if not state.equivalent_values(old, after["objects"].get(identifier), exact=True):
            reject(f"objects/{identifier}", "An existing object's identity or behaviour metadata changed")
    added_objects = set(after["objects"]) - before["objects"].keys()
    reference, reference_assets = _reference_addition(added_objects, after, plan, result, reject)
    if reference is not None:
        counts["reference_objects"] = 1

    for key, old in before["channels"].items():
        counts["channels"] += 1
        new = after["channels"].get(key)
        if new is None:
            reject(f"channels/{key}", "An original channel was removed")
            continue
        if not state.equivalent_values(_metadata(old), _metadata(new), exact=True):
            reject(f"channels/{key}", "Native channel identity, mode or description changed")
            continue
        if old["mode"] == "Static":
            counts["static_channels"] += 1
            if not state.equivalent_values(old.get("value"), new.get("value"), exact=True):
                reject(f"channels/{key}/value", "Static native data or settings changed")
            continue
        editable = old["owner"] in owners
        exact = old["kind"] == "setting"
        for frame, value in enumerate(new["values"]):
            if editable and first <= frame <= last:
                counts["editable_samples"] += 1
                continue
            counts["animated_samples"] += 1
            expected = old["values"][min(frame, before["count"] - 1)]
            if not state.equivalent_values(expected, value, exact=exact):
                if _periodic_derived_guard(before, after, plan, owners, old, frame, expected, value):
                    counts["periodic_euler_samples"] += 1
                    continue
                reject(f"channels/{key}/values/{frame}", "A protected native sample changed")
    for key in after["channels"].keys() - before["channels"].keys():
        item = after["channels"][key]
        if reference is None or item.get("owner") != reference or key != f"{item['kind']}:{item.get('id')}":
            reject(f"channels/{key}", "A channel was added outside the single new reference object")
        else:
            counts["reference_channels"] += 1

    added_tracks = set(after["tracks"]) - before["tracks"].keys()
    reference_tracks = set()
    for identifier in added_tracks:
        members = after["tracks"][identifier].get("objects")
        if reference is None or members != [reference]:
            reject(f"tracks/{identifier}", "A track was added outside the single new reference object")
        else:
            reference_tracks.add(identifier)
    if reference is not None and len(reference_tracks) != 1:
        reject("tracks", "A single imported reference plane needs one exclusive native track")
    for identifier in reference_tracks:
        parent = after["tracks"][identifier].get("parent")
        folders = after["auxiliary"].get("folders", {})
        memberships = [
            folder_id
            for folder_id, folder in folders.items()
            for child in folder.get("children", [])
            if child == identifier
        ]
        if memberships != [parent]:
            reject(
                f"tracks/{identifier}/parent",
                "The reference track is not uniquely attached to its native folder",
            )
    for identifier, old in before["tracks"].items():
        counts["tracks"] += 1
        new = after["tracks"].get(identifier)
        if new is None:
            reject(f"tracks/{identifier}", "An original track was removed")
            continue
        if identifier not in selected:
            if not state.equivalent_values(old, new, exact=True):
                reject(f"tracks/{identifier}", "Metadata of an unselected track changed")
            continue
        if not state.equivalent_values(
            {key: value for key, value in old.items() if key not in {"keys", "sections"}},
            {key: value for key, value in new.items() if key not in {"keys", "sections"}},
            exact=True,
        ):
            reject(f"tracks/{identifier}", "A selected track's identity, membership or visibility changed")
        anchor = guard_anchors.get(identifier)
        old_sections = {
            key: value
            for key, value in old["sections"].items()
            if not first <= int(key) <= last and int(key) != anchor
        }
        new_sections = {
            key: value
            for key, value in new["sections"].items()
            if not first <= int(key) <= last and int(key) < before["count"] and int(key) != anchor
        }
        if not state.equivalent_values(old_sections, new_sections, exact=True):
            reject(f"tracks/{identifier}/sections", "Curve metadata outside the job interval changed")
        if [frame for frame in old["keys"] if not first <= frame <= last and frame != anchor] != [
            frame
            for frame in new["keys"]
            if not first <= frame <= last and frame < before["count"] and frame != anchor
        ]:
            reject(f"tracks/{identifier}/keys", "Keys outside the job interval changed")

    if before["format_version"] != after["format_version"]:
        reject("format_version", "Native checkpoint format changed")
    auxiliary = copy.deepcopy(after["auxiliary"])
    # A new plane is inserted into an existing folder. Remove only its known
    # new track IDs; original siblings, order, folders and all other aux data
    # must match the complete pre-job model.
    for folder in auxiliary.get("folders", {}).values():
        if "children" in folder:
            folder["children"] = [child for child in folder["children"] if child not in reference_tracks]
    if not state.equivalent_values(before["auxiliary"], auxiliary, exact=True):
        reject("auxiliary", "Protected folders, clusters or other auxiliary metadata changed")
    external = copy.deepcopy(after["external"])
    old_assets = before["external"].get("assets", {})
    assets = external.get("assets", {})
    for identifier in set(assets) - old_assets.keys():
        item = assets[identifier]
        if (
            reference_assets.get(identifier) != item.get("type")
            or type(item.get("payload_sha256")) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", item.get("payload_sha256", ""))
        ):
            reject(
                f"external/assets/{identifier}",
                "An asset was added outside the reference plane's mesh/dependency",
            )
        else:
            counts["reference_assets"] += 1
            assets.pop(identifier)
    if not state.equivalent_values(before["external"], external, exact=True):
        reject("external", "Protected external assets or payload fingerprints changed")
    return report()


def _periodic_derived_guard(before, after, plan, owners, channel, frame, expected, actual):
    """Accept native angle wrapping only on an unbound Unbaking guard output."""
    if (
        plan.get("operation") != "unbake"
        or not state.unbound_angle_output(channel, before["objects"])
        or channel["owner"] not in owners
        or before["count"] != after["count"]
        or frame != before["count"] - 1
        or frame <= plan["last"]
    ):
        return False
    return state.periodic_rotation_equal(expected, actual)


def _guard_anchors(before, after, plan, selected, reject):
    """Permit only planned Unbaking key metadata; guard samples stay protected."""
    anchors = plan.get("guard_anchors", {})
    sections = plan.get("guard_anchor_sections", {})
    if type(anchors) is not dict or type(sections) is not dict or anchors.keys() != sections.keys():
        reject("plan/guard_anchors", "Guard anchors require one exact planned section per track")
        return {}
    if not anchors:
        return {}
    if plan.get("operation") != "unbake" or after["count"] != before["count"]:
        reject(
            "plan/guard_anchors", "Terminal guard anchoring is limited to Unbaking without timeline growth"
        )
        return {}
    verified = {}
    for identifier, frame in anchors.items():
        path = f"tracks/{identifier}/sections/{frame}"
        if (
            identifier not in selected
            or type(frame) is not int
            or frame != before["count"] - 1
            or frame <= plan["last"]
        ):
            reject(path, "An anchor must be the original terminal guard on an explicitly selected track")
            continue
        old = before["tracks"][identifier]
        previous = old["sections"].get(str(frame))
        if frame in old["keys"] or (previous is not None and previous.get("key") is not None):
            reject(path, "A guard anchor cannot replace an existing native key")
            continue
        expected = sections[identifier]
        source_keys = [
            key
            for key in old["keys"]
            if key < frame and old["sections"].get(str(key), {}).get("key") is not None
        ]
        if not source_keys:
            reject(path, "A terminal anchor needs an original preceding native key")
            continue
        recipe = copy.deepcopy(old["sections"][str(max(source_keys))])
        recipe.update(interpolation="STEP", ai_description="")
        if not state.equivalent_values(expected, recipe, exact=True):
            reject(
                path, "The planned anchor must retain the preceding key's metadata with a STEP guard interval"
            )
            continue
        new = after["tracks"].get(identifier, {})
        if (
            type(expected) is not dict
            or expected.get("key") is None
            or frame not in new.get("keys", [])
            or not state.equivalent_values(expected, new.get("sections", {}).get(str(frame)), exact=True)
        ):
            reject(path, "The terminal key does not match its exact predeclared native section metadata")
            continue
        verified[identifier] = frame
    return verified


def _native_id(value):
    if type(value) is not dict or set(value) != {"__native__", "value"} or value["__native__"] != "id":
        raise ValueError("Invalid native reference identity")
    if type(value["value"]) is not str or not value["value"]:
        raise ValueError("Missing native reference identity")
    return value["value"]


def _reference_addition(added, after, plan, result, reject):
    if not added:
        return None, {}
    identifier = result.get("reference_object")
    if plan.get("operation") != "mocap" or type(identifier) is not str or added != {identifier}:
        reject("objects", "Only the explicitly returned single Mocap reference object may be added")
        return None, {}
    obj = after["objects"][identifier]
    try:
        behaviours = obj["behaviours"]
        named = {item["name"]: (key, item["properties"]) for key, item in behaviours.items()}
        if (
            obj["type"] != "Mesh Object"
            or len(named) != len(behaviours)
            or named.keys() != _REFERENCE_BEHAVIOURS
        ):
            raise ValueError("Unexpected native video-plane type or behaviour set")
        channels = after["channels"]
        for _name, (_bid, properties) in named.items():
            for prop in properties.values():
                kind, value = prop["kind"], prop["value"]
                if kind in {"DataType", "SettingType", "DataRangeType", "SettingRangeType"}:
                    prefix = "setting" if kind.startswith("Setting") else "data"
                    values = value if "Range" in kind else [value]
                    for reference in values:
                        channel = channels[f"{prefix}:{_native_id(reference)}"]
                        if channel["owner"] != identifier:
                            raise ValueError("The imported reference points at another object's channel")
                elif kind in {"BehaviourType", "BehaviourRangeType"}:
                    for reference in value if "Range" in kind else [value]:
                        target = _native_id(reference)
                        if target != _NULL_ID and target not in behaviours:
                            raise ValueError("The imported reference points at another object's behaviour")
                elif kind in {"ObjectType", "ObjectRangeType"}:
                    for reference in value if "Range" in kind else [value]:
                        if _native_id(reference) not in {identifier, _NULL_ID}:
                            raise ValueError("The imported reference is linked to an existing foreign object")
        texture = named["TextureContainer"][1]
        start = texture["start_frame"]
        paths = texture["texture_paths"]
        if start["kind"] != "DataType" or paths["kind"] != "DataRangeType" or not paths["value"]:
            raise ValueError("Native video-plane texture metadata is missing")
        start_channel = channels["data:" + _native_id(start["value"])]
        if start_channel["mode"] != "Static" or type(start_channel["value"]) is not int:
            raise ValueError("Native video-plane start frame is not a static integer")
        for value in paths["value"]:
            channel = channels["data:" + _native_id(value)]
            if channel["mode"] != "Static" or type(channel["value"]) is not str or not channel["value"]:
                raise ValueError("Native video-plane texture paths are not static filenames")
        mesh = named["MeshObject"][1]
        assets = {}
        for name, expected_type in (("mesh", "Mesh"), ("mesh_dependency", "MeshDependency")):
            prop = mesh[name]
            if prop["kind"] != "AssetType":
                raise ValueError("Native video-plane mesh assets are missing")
            asset_id = _native_id(prop["value"])
            if after["external"]["assets"][asset_id]["type"] != expected_type:
                raise ValueError("Native video-plane asset types do not match their bindings")
            assets[asset_id] = expected_type
        if len(assets) != 2:
            raise ValueError("The reference mesh and dependency need distinct asset identities")
    except (KeyError, TypeError, ValueError) as error:
        reject(f"objects/{identifier}", str(error))
        return None, {}
    return identifier, assets
