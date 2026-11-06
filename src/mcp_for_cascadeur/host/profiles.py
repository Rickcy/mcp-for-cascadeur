"""User-defined semantic controller groups for any inspected native rig."""

from __future__ import annotations

import json
import re

from ..contracts import Tool
from . import animation
from .common import current_context, get_settings, resolve_object


def _profile_path(name):
    if type(name) is not str or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", name):
        raise ValueError("Profile names use 1-80 letters, digits, underscores or hyphens")
    workspace = get_settings()["workspace"].resolve()
    path = (workspace / ".cascadeur-mcp" / "profiles" / (name + ".json")).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError("Profile directory resolves outside the workspace")
    return path


def rig_profile_create(name, roles, overwrite=False):
    _, _, view, scene = current_context()
    destination = _profile_path(name)
    if destination.exists() and not overwrite:
        raise ValueError("Profile exists; set overwrite=true to replace it")
    if type(roles) is not dict or not 1 <= len(roles) <= 128:
        raise ValueError("Provide 1-128 named roles")
    resolved = {}
    for role, objects in roles.items():
        if type(role) is not str or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{0,79}", role):
            raise ValueError("Role names must be short public identifiers")
        if type(objects) is not list or not 1 <= len(objects) <= 128:
            raise ValueError("Each role must contain 1-128 object IDs or unique names")
        ids = [str(resolve_object(scene, identifier)) for identifier in objects]
        if len(set(ids)) != len(ids):
            raise ValueError("An object may occur only once in each role")
        resolved[role] = ids
    profile = {
        "schema_version": 1,
        "name": name,
        "scene_path": view.get_path_name(),
        "topology_sha256": animation._signature(scene),
        "roles": resolved,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = destination.with_suffix(".tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"profile": profile, "path": str(destination)}


def rig_profiles_list():
    directory = get_settings()["workspace"] / ".cascadeur-mcp" / "profiles"
    return {"profiles": sorted(path.stem for path in directory.glob("*.json"))}


def rig_profile_read(name):
    path = _profile_path(name)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Rig profile exceeds 1 MiB")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (
        type(profile) is not dict
        or profile.get("schema_version") != 1
        or profile.get("name") != name
        or type(profile.get("roles")) is not dict
    ):
        raise ValueError("Invalid rig profile")
    _, _, view, scene = current_context()
    if profile.get("scene_path") != view.get_path_name() or profile.get(
        "topology_sha256"
    ) != animation._signature(scene):
        raise ValueError("Rig profile belongs to a different scene or topology; bind a new profile")
    for role, objects in profile["roles"].items():
        if not isinstance(role, str) or type(objects) is not list or not objects:
            raise ValueError("Invalid role in rig profile")
        for identifier in objects:
            resolve_object(scene, identifier)
    return profile


def _groups(profile, roles=None):
    record = rig_profile_read(profile)
    selected = list(record["roles"]) if roles is None else roles
    if type(selected) is not list or not selected or any(role not in record["roles"] for role in selected):
        raise ValueError("Choose one or more exact roles from rig_profile_read")
    return {role: record["roles"][role] for role in selected}


def rig_pose_read(profile, roles=None, frame=None, space="global"):
    groups = _groups(profile, roles)
    identifiers = list(dict.fromkeys(identifier for group in groups.values() for identifier in group))
    result = animation.pose_read(identifiers, frame, space)
    by_id = {pose["object"]: pose for pose in result["poses"]}
    return {
        "frame": result["frame"],
        "space": space,
        "profile": profile,
        "roles": {role: [by_id[identifier] for identifier in group] for role, group in groups.items()},
    }


def rig_pose_apply(profile, targets, frame=None, space="global", create_keys=True, verify=True):
    groups = _groups(profile, list(targets))
    poses, assigned = [], set()
    for role, changes in targets.items():
        if type(changes) is not list or len(changes) != len(groups[role]):
            raise ValueError(f"Role {role!r} needs one patch per bound control")
        for identifier, patch in zip(groups[role], changes):
            if identifier in assigned:
                raise ValueError("Selected roles overlap; write each control only once")
            if type(patch) is not dict or set(patch) - {"position", "rotation_wxyz"}:
                raise ValueError("Role patches accept position and/or rotation_wxyz")
            poses.append({"object": identifier, **patch})
            assigned.add(identifier)
    return animation.pose_apply(poses, frame, space, create_keys, verify)


def rig_pose_offset(profile, roles, frames, delta, space="global", create_keys=False):
    groups = _groups(profile, roles)
    identifiers = list(dict.fromkeys(identifier for group in groups.values() for identifier in group))
    return animation.pose_offset(identifiers, frames, delta, space, create_keys)


NAME = {"type": "string", "minLength": 1, "maxLength": 80}
ROLE_LIST = {"type": "array", "items": NAME, "minItems": 1, "maxItems": 128, "uniqueItems": True}
TOOLS = [
    Tool(
        "rig_profile_create",
        "Bind your own body, hand, finger or prop roles to exact native controls. "
        "Stores a workspace profile tied to this saved scene and topology.",
        rig_profile_create,
        {
            "name": NAME,
            "roles": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": 128,
                "additionalProperties": animation.OBJECTS,
            },
            "overwrite": {"type": "boolean"},
        },
        ("name", "roles"),
        False,
    ),
    Tool(
        "rig_profiles_list",
        "List saved semantic rig profiles in the configured workspace.",
        rig_profiles_list,
    ),
    Tool(
        "rig_profile_read",
        "Read a rig profile and verify that its scene, topology and control IDs still match.",
        rig_profile_read,
        {"name": NAME},
        ("name",),
    ),
    Tool(
        "rig_pose_read",
        "Read controller poses grouped by user-defined body or finger roles.",
        rig_pose_read,
        {"profile": NAME, "roles": ROLE_LIST, "frame": animation.FRAME, "space": animation.SPACE},
        ("profile",),
    ),
    Tool(
        "rig_pose_apply",
        "Write patches for named roles in profile slot order, using native pose preflight and readback.",
        rig_pose_apply,
        {
            "profile": NAME,
            "targets": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": 128,
                "additionalProperties": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 128,
                    "items": {
                        "type": "object",
                        "properties": {"position": animation.VECTOR, "rotation_wxyz": animation.QUATERNION},
                        "additionalProperties": False,
                    },
                },
            },
            "frame": animation.FRAME,
            "space": animation.SPACE,
            "create_keys": {"type": "boolean"},
            "verify": {"type": "boolean"},
        },
        ("profile", "targets"),
        False,
    ),
    Tool(
        "rig_pose_offset",
        "Translate complete named control groups across existing frames. Relative edits accumulate.",
        rig_pose_offset,
        {
            "profile": NAME,
            "roles": ROLE_LIST,
            "frames": animation.FRAMES,
            "delta": animation.VECTOR,
            "space": animation.SPACE,
            "create_keys": {"type": "boolean"},
        },
        ("profile", "roles", "frames", "delta"),
        False,
    ),
]
