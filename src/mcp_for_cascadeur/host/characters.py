"""Native RigInfo discovery, checked semantic groups and finger-local channels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..contracts import Tool
from . import common
from .motion import finite_vector, normalized_quaternion


@dataclass
class ControlBinding:
    object_id: Any
    position_data: Any
    rotation_data: Any = None
    role: str | None = None
    slot: str | None = None


@dataclass
class FingerBinding:
    object_id: Any
    rotation_data: Any
    track_id: Any


@dataclass
class CharacterRig:
    id: str
    name: str
    rig_info: Any
    owner: Any
    controls: dict[str, Any]
    joints: set
    owned: set
    tracks: set
    bindings: dict[str, ControlBinding]
    roles: dict[str, dict[str, str]] = field(default_factory=dict)
    role_joints: dict[str, Any] = field(default_factory=dict)
    fingers: dict[str, dict[str, FingerBinding]] = field(default_factory=dict)
    native_links: list[dict] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def identifier(self):
        return self.id


# This vocabulary identifies actual Point targets; palm groups do not curl fingers.
BODY_GROUPS = {
    "pelvis": {
        "center": "pelvis_MainPoint",
        "orientation": "pelvis_AdditionalPoint",
        "left_hip": "thigh_MainPoint_l",
        "right_hip": "thigh_MainPoint_r",
    },
    "chest": {
        "waist": "stomach_MainPoint",
        "waist_orientation": "stomach_AdditionalPoint",
        "center": "chest_MainPoint",
        "orientation": "chest_AdditionalPoint",
        "left_clavicle": "clavicle_MainPoint_l",
        "left_clavicle_orientation": "clavicle_AdditionalPoint_l",
        "left_shoulder": "arm_MainPoint_l",
        "right_clavicle": "clavicle_MainPoint_r",
        "right_clavicle_orientation": "clavicle_AdditionalPoint_r",
        "right_shoulder": "arm_MainPoint_r",
    },
    "head": {
        "neck": "neck_MainPoint",
        "neck_orientation": "neck_AdditionalPoint",
        "center": "head_MainPoint",
        "direction": "head_DirectionPoint",
        "orientation": "head_AdditionalPoint",
    },
}
BODY_JOINTS = {"pelvis": "pelvis", "chest": "chest", "head": "head"}
PARENT_NAMES = {"pelvis": None, "stomach": "pelvis", "chest": "stomach", "neck": "chest", "head": "neck"}
for _side, _suffix in (("left", "l"), ("right", "r")):
    for _part, _entries in (
        (
            "hand",
            (("center", "MainPoint"), ("direction", "DirectionPoint"), ("orientation", "AdditionalPoint")),
        ),
        ("elbow", (("center", "MainPoint"), ("bend", "AdditionalPoint"))),
        (
            "foot",
            (
                ("center", "MainPoint"),
                ("heel", "Self0Point"),
                ("toe", "toe_MainPoint"),
                ("toe_direction", "toe_DirectionPoint"),
                ("toe_orientation", "toe_AdditionalPoint"),
            ),
        ),
        ("knee", (("center", "MainPoint"), ("bend", "AdditionalPoint"))),
    ):
        _stem = {"hand": "hand", "elbow": "forearm", "foot": "foot", "knee": "calf"}[_part]
        _role = _side + "_" + _part
        BODY_GROUPS[_role] = {
            slot: (token if token.startswith("toe_") else _stem + "_" + token) + "_" + _suffix
            for slot, token in _entries
        }
        BODY_JOINTS[_role] = _stem + "_" + _suffix
    for _chain in (
        ("chest", "clavicle_" + _suffix, "arm_" + _suffix, "forearm_" + _suffix, "hand_" + _suffix),
        ("pelvis", "thigh_" + _suffix, "calf_" + _suffix, "foot_" + _suffix, "toe_" + _suffix),
    ):
        PARENT_NAMES.update({child: parent for parent, child in zip(_chain, _chain[1:])})
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _parent(scene, object_id):
    viewer = scene.behaviour_viewer()
    basic = viewer.get_behaviour_by_name(object_id, "Basic")
    if basic.is_null():
        raise ValueError("Rig object has no Basic relationship: " + str(object_id))
    parent = viewer.get_behaviour_object(basic, "parent")
    return None if parent.is_null() else parent


def _transform_data(scene, object_id, channel):
    viewer, data = scene.behaviour_viewer(), scene.data_viewer()
    transform = viewer.get_behaviour_by_name(object_id, "Transform")
    if transform.is_null():
        raise ValueError("Rig object lacks Transform: " + str(object_id))
    reference = viewer.get_behaviour_data(transform, channel)
    if reference.is_null() or not data.has_data(reference):
        raise ValueError("Rig object lacks stored " + channel + ": " + str(object_id))
    return reference


def _links(scene, owned):
    viewer = scene.behaviour_viewer()
    result = []
    for object_id in sorted(owned, key=str):
        information = viewer.get_behaviour_by_name(object_id, "RigAdditionalInfo")
        if information.is_null():
            continue
        record = {"box_id": str(object_id)}
        for name in (
            "joint",
            "main_point",
            "direction_point",
            "additional_point",
            "rigid_body",
            "limb_direction",
        ):
            reference = viewer.get_behaviour_reference(information, name)
            record[name] = None if reference.is_null() else str(viewer.get_behaviour_owner(reference))
        result.append(record)
    return result


def resolve_character(scene, identifier=None, *, require_semantics=False, require_hands=False):
    """Resolve native membership afresh; never infer a character from name fragments."""
    viewer, model, layers = scene.behaviour_viewer(), scene.model_viewer(), scene.layers_viewer()
    matches = []
    for information in viewer.get_behaviours("RigInfo"):
        owner = viewer.get_behaviour_owner(information)
        if identifier is None or identifier in (str(owner), str(information), model.get_object_name(owner)):
            matches.append((information, owner))
    if len(matches) != 1:
        raise ValueError(
            "Choose one exact RigInfo owner ID from characters_list; matched " + str(len(matches))
        )
    information, owner = matches[0]
    joints = set(viewer.get_behaviour_objects_range(information, "related_joints"))
    owned = joints | {owner}
    for name in ("rig_objects", "related_objects"):
        owned.update(viewer.get_behaviour_objects_range(information, name))
    if not 1 <= len(joints) <= 256 or len(owned) > 2048:
        raise ValueError("Character exceeds supported native membership limits.")
    for mesh in viewer.get_behaviours("MeshObject"):
        skin = set(viewer.get_behaviour_objects_range(mesh, "linked_objects"))
        if skin and skin <= joints:
            owned.add(viewer.get_behaviour_owner(mesh))
    controls = {str(item): item for item in owned if model.get_object_type_name(item) == "Point"}
    if not 1 <= len(controls) <= 128:
        raise ValueError("Expected 1-128 native Point controls in RigInfo membership.")
    bindings = {
        key: ControlBinding(item, _transform_data(scene, item, "global_position"))
        for key, item in controls.items()
    }
    tracks = set()
    for item in owned:
        track_id = layers.layer_id_by_obj_id_or_null(item)
        if not track_id.is_null():
            tracks.add(track_id)
    rig = CharacterRig(
        str(owner),
        model.get_object_name(owner),
        information,
        owner,
        controls,
        joints,
        owned,
        tracks,
        bindings,
        native_links=_links(scene, owned),
    )
    signature = {
        "owner": rig.id,
        "objects": sorted(
            (
                str(item),
                model.get_object_name(item),
                model.get_object_type_name(item),
                str(_parent(scene, item)),
            )
            for item in owned
        ),
        "bindings": rig.native_links,
        "tracks": sorted(map(str, tracks)),
    }
    rig.fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if require_semantics:
        _bind_body(scene, rig)
    if require_hands:
        _bind_hands(scene, rig)
    return rig


def _unique_named(scene, rig, name, kind):
    model = scene.model_viewer()
    matches = [item for item in rig.owned if model.get_object_name(item) == name]
    if len(matches) != 1 or model.get_object_type_name(matches[0]) != kind:
        raise ValueError("Unsupported semantic topology: expected one native " + kind + " named " + name)
    return matches[0]


def _bind_body(scene, rig):
    canonical_joints = {name: _unique_named(scene, rig, name, "Joint") for name in PARENT_NAMES}
    for name, expected in PARENT_NAMES.items():
        actual = _parent(scene, canonical_joints[name])
        if actual != (canonical_joints[expected] if expected is not None else None):
            raise ValueError("Semantic Joint hierarchy differs at " + name)
    joint_links = {}
    for link in rig.native_links:
        joint = link["joint"]
        if joint is not None:
            if joint in joint_links:
                raise ValueError("Ambiguous RigAdditionalInfo Joint binding: " + joint)
            joint_links[joint] = link
    assigned = set()
    for role, slots in BODY_GROUPS.items():
        group = {}
        for slot, name in slots.items():
            object_id = _unique_named(scene, rig, name, "Point")
            key = str(object_id)
            if key not in rig.controls or key in assigned:
                raise ValueError("Semantic Point membership is ambiguous: " + name)
            if "_Self0Point_" not in name:
                stem, point_suffix = name.split("_", 1)
                side = "_" + name[-1] if name.endswith(("_l", "_r")) else ""
                joint = canonical_joints[stem + side]
                point_kind = point_suffix[: -len(side)] if side else point_suffix
                native_property = {
                    "MainPoint": "main_point",
                    "DirectionPoint": "direction_point",
                    "AdditionalPoint": "additional_point",
                }[point_kind]
                if joint_links.get(str(joint), {}).get(native_property) != key:
                    raise ValueError("Native semantic binding differs for " + name)
            group[slot] = key
            rig.bindings[key].role, rig.bindings[key].slot = role, slot
            assigned.add(key)
        rig.roles[role] = group
        rig.role_joints[role] = canonical_joints[BODY_JOINTS[role]]
    if assigned != set(rig.controls):
        raise ValueError("Semantic profile does not cover the complete native Point set.")


def _bind_hands(scene, rig):
    csc = __import__("csc")
    layers, data = scene.layers_viewer(), scene.data_viewer()
    for side, suffix in (("left", "l"), ("right", "r")):
        wrist = _unique_named(scene, rig, "hand_" + suffix, "Joint")
        group = {}
        for finger in FINGERS:
            parent = wrist
            for index in range(1, 5):
                name = "f_" + finger + str(index) + "_" + suffix
                joint = _unique_named(scene, rig, name, "Joint")
                if joint not in rig.joints or _parent(scene, joint) != parent:
                    raise ValueError("Finger parent chain differs at " + name)
                parent = joint
                if index == 4:
                    continue
                channel = _transform_data(scene, joint, "local_rotation")
                if data.get_data(channel).mode != csc.model.DataMode.Animation:
                    raise ValueError("Finger local rotation is not animated: " + name)
                track = layers.layer_id_by_obj_id_or_null(joint)
                if track.is_null():
                    raise ValueError("Finger Joint has no native track: " + name)
                group[finger + str(index)] = FingerBinding(joint, channel, track)
        rig.fingers[side] = group


def _frame(scene, value):
    frame = scene.get_current_frame() if value is None else value
    if type(frame) is not int or not 0 <= frame < scene.data_viewer().get_animation_size():
        raise ValueError("Choose an existing stored animation frame.")
    return frame


def _editable_frame(scene, value):
    frame = _frame(scene, value)
    if frame == scene.data_viewer().get_animation_size() - 1:
        raise ValueError(
            "Native character writes require an allocated successor frame; the terminal stored frame is read-only."
        )
    return frame


def _position(value):
    encoded = common.json_value(value)
    if isinstance(encoded, list) and len(encoded) == 3:
        encoded = [item[0] if isinstance(item, list) and len(item) == 1 else item for item in encoded]
    return finite_vector(encoded, 3)


def _quaternion(value):
    quaternion = value.to_quaternion() if hasattr(value, "to_quaternion") else value
    components = normalized_quaternion([float(getattr(quaternion, axis)()) for axis in "wxyz"])
    first = next((item for item in components if abs(item) > 1e-9), 1)
    return [-item for item in components] if first < 0 else components


def _joint_pose(scene, object_id, frame):
    result = {"id": str(object_id), "name": scene.model_viewer().get_object_name(object_id)}
    for space in ("local", "global"):
        result[space] = {
            "position": _position(
                scene.data_viewer().get_data_value(
                    _transform_data(scene, object_id, space + "_position"), frame
                )
            ),
            "rotation_wxyz": _quaternion(
                scene.data_viewer().get_data_value(
                    _transform_data(scene, object_id, space + "_rotation"), frame
                )
            ),
        }
    return result


def characters_list():
    _, _, view, scene = common.current_context()
    model, viewer = scene.model_viewer(), scene.behaviour_viewer()
    result = []
    for information in viewer.get_behaviours("RigInfo"):
        owner = viewer.get_behaviour_owner(information)
        result.append(
            {
                "id": str(owner),
                "name": model.get_object_name(owner),
                "joint_count": len(viewer.get_behaviour_objects_range(information, "related_joints")),
            }
        )
    return {"scene_path": view.get_path_name(), "characters": result}


def character_skeleton(character=None, include_track_sections=True):
    _, _, _, scene = common.current_context()
    rig = resolve_character(scene, character)
    model, viewer, layers = scene.model_viewer(), scene.behaviour_viewer(), scene.layers_viewer()
    nodes = [
        {
            "id": str(item),
            "name": model.get_object_name(item),
            "type": model.get_object_type_name(item),
            "parent_id": str(parent) if (parent := _parent(scene, item)) is not None else None,
            "behaviors": sorted(viewer.get_behaviour_name(value) for value in viewer.get_behaviours(item)),
        }
        for item in sorted(rig.owned, key=str)
    ]
    tracks = []
    for identifier in sorted(rig.tracks, key=str):
        track = layers.layer(identifier)
        row = {
            "id": str(identifier),
            "name": track.header.name,
            "locked": track.is_locked,
            "objects": sorted(map(str, track.obj_ids)),
            "keys": list(track.key_frame_indices()),
            "contains_other_objects": bool(set(track.obj_ids) - rig.owned),
        }
        if include_track_sections:
            row["sections"] = {
                str(frame): {
                    "interpolation": section.interval.interpolation.name,
                    "ik_fk": section.interval.common.ik_fk.name,
                    "fixation": section.interval.common.fixation.name,
                    "tangents": section.key.tangents.name,
                    "left_weight": section.key.left_weight_velocity,
                    "right_weight": section.key.right_weight_velocity,
                }
                for frame, section in track.sections.items()
            }
        tracks.append(row)
    from .inspection import _property_record

    constraints = []
    for item in sorted(rig.owned, key=str):
        for behavior in viewer.get_behaviours(item):
            name = viewer.get_behaviour_name(behavior)
            if name not in {"AngleConstraints", "ConnectionPointTwoBody", "AttractionPoint"}:
                continue
            properties = []
            for property_name in viewer.get_behaviour_property_names(behavior)[:100]:
                try:
                    properties.append(
                        _property_record(
                            viewer, scene.data_viewer(), behavior, property_name, scene.get_current_frame()
                        )
                    )
                except Exception as error:
                    properties.append({"name": property_name, "unavailable": str(error)[:500]})
            constraints.append(
                {"object_id": str(item), "behavior_id": str(behavior), "name": name, "properties": properties}
            )
    return {
        "character": rig.id,
        "fingerprint": rig.fingerprint,
        "objects": nodes,
        "point_control_ids": sorted(rig.controls),
        "native_bindings": rig.native_links,
        "tracks": tracks,
        "constraints": constraints,
        "joint_writes": "Driven body Joints are read-only; use Point targets and validated finger channels.",
    }


def character_pose(character=None, frame=None, include_joint_state=True):
    _, _, _, scene = common.current_context()
    rig, position = resolve_character(scene, character), _frame(scene, frame)
    data = scene.data_viewer()
    result = {
        "character": rig.id,
        "fingerprint": rig.fingerprint,
        "frame": position,
        "pose": {
            key: {"position": _position(data.get_data_value(binding.position_data, position))}
            for key, binding in rig.bindings.items()
        },
    }
    if include_joint_state:
        result["joints"] = {
            str(item): _joint_pose(scene, item, position) for item in sorted(rig.joints, key=str)
        }
    return result


def rig_semantics(character=None):
    _, _, _, scene = common.current_context()
    rig = resolve_character(scene, character, require_semantics=True)
    return {
        "character": rig.id,
        "profile": "cascy-native-points-v1",
        "fingerprint": rig.fingerprint,
        "roles": {
            role: {
                "controls": {
                    slot: {"id": key, "name": scene.model_viewer().get_object_name(rig.controls[key])}
                    for slot, key in slots.items()
                },
                "joint_id": str(rig.role_joints[role]),
                "joint_writable": False,
            }
            for role, slots in rig.roles.items()
        },
        "space": "native world positions",
        "palm_targets_curl_fingers": False,
    }


def selected_roles(rig, roles):
    names = list(rig.roles) if roles is None else roles
    if (
        type(names) is not list
        or not names
        or len(names) != len(set(names))
        or any(name not in rig.roles for name in names)
    ):
        raise ValueError("Choose unique role names from rig_semantics.")
    return names


def semantic_read_from(scene, rig, frame, roles=None, include_joint_state=True):
    names, data = selected_roles(rig, roles), scene.data_viewer()
    result = {
        "character": rig.id,
        "fingerprint": rig.fingerprint,
        "frame": frame,
        "pose": {
            role: {
                slot: _position(data.get_data_value(rig.bindings[key].position_data, frame))
                for slot, key in rig.roles[role].items()
            }
            for role in names
        },
    }
    if include_joint_state:
        result["joint_state"] = {role: _joint_pose(scene, rig.role_joints[role], frame) for role in names}
    return result


def semantic_pose_read(character=None, frame=None, roles=None, include_joint_state=True):
    _, _, _, scene = common.current_context()
    rig = resolve_character(scene, character, require_semantics=True)
    return semantic_read_from(scene, rig, _frame(scene, frame), roles, include_joint_state)


def semantic_pose_read_sequence(frames, character=None, roles=None, include_joint_state=True):
    if type(frames) is not list or not 1 <= len(frames) <= 64 or len(frames) != len(set(frames)):
        raise ValueError("Provide 1-64 unique existing frames.")
    _, _, _, scene = common.current_context()
    positions = [_frame(scene, item) for item in frames]
    rig = resolve_character(scene, character, require_semantics=True)
    return {
        "character": rig.id,
        "fingerprint": rig.fingerprint,
        "keyframes": [
            semantic_read_from(scene, rig, position, roles, include_joint_state) for position in positions
        ],
    }


def hand_pose_read(character=None, frame=None, side="both"):
    if side not in ("left", "right", "both"):
        raise ValueError("side must be left, right or both.")
    _, _, _, scene = common.current_context()
    rig, position = resolve_character(scene, character, require_hands=True), _frame(scene, frame)
    return {
        "character": rig.id,
        "frame": position,
        "space": "finger local quaternion wxyz",
        "hands": {
            hand: {
                name: _quaternion(scene.data_viewer().get_data_value(binding.rotation_data, position))
                for name, binding in fingers.items()
            }
            for hand, fingers in rig.fingers.items()
            if side in (hand, "both")
        },
        "bindings": {
            hand: {
                name: {"joint_id": str(binding.object_id), "data_id": str(binding.rotation_data)}
                for name, binding in fingers.items()
            }
            for hand, fingers in rig.fingers.items()
            if side in (hand, "both")
        },
    }


def _character_preflight(scene, rig, frames):
    from . import curves, playback

    playback.require_idle()
    result = curves.verify_supported(scene)
    tracks = curves.character_tracks(scene, rig)
    for layer in tracks.values():
        if any(
            section.interval.common.ik_fk.name != "IK"
            or (section.key is not None and section.key.common.ik_fk.name != "IK")
            for section in layer.sections.values()
        ):
            raise ValueError("Character Point writes require native IK character tracks.")
    for binding in rig.bindings.values():
        if scene.data_viewer().get_data(binding.position_data).mode.name != "Animation":
            raise ValueError("All character Point targets must be animated native channels.")
    for position in frames:
        _editable_frame(scene, position)
    return {
        **result,
        "character_tracks": len(tracks),
        "frames": list(frames),
        "advisory_only": True,
        "history_allocated": False,
    }, tracks


def _write_identity():
    csc, app, view, scene = common.current_context()
    if not view.get_path_name():
        raise ValueError("Save the native character scene before a checked edit.")
    return csc, app, view, scene


def _state_after_character(scene, rig, before, allowed_tracks, *, edited_finger_channels=()):
    from . import curves, state

    after = curves._capture(scene)
    curves.verify_structure(before, after, preserve_tracks=False)
    for key, channel in before["channels"].items():
        if channel["kind"] == "setting" or channel["mode"] == "Static":
            if not state.channel_equal(channel, after["channels"][key]):
                raise curves.CharacterVerificationError(
                    "A character write changed a setting or static channel: " + key
                )
    for key, track in before["tracks"].items():
        if key not in allowed_tracks and track != after["tracks"][key]:
            raise curves.CharacterVerificationError("A character write changed an unrelated track: " + key)
    for hand in rig.fingers.values():
        for binding in hand.values():
            key = "data:" + str(binding.rotation_data)
            if str(binding.rotation_data) in edited_finger_channels:
                continue
            if not state.channel_equal(before["channels"][key], after["channels"][key]):
                raise curves.CharacterVerificationError(
                    "A body Point write changed authored finger rotation: " + key
                )
    return after


def _restore_existing_sections(scene, editor, tracks, before, edited_frames):
    """Retain existing metadata after native key insertion, without detached Sections."""
    from . import state

    for identifier in tracks:
        for position, saved in before["tracks"][str(identifier)]["sections"].items():
            frame = int(position)

            def retain(section, metadata=saved, position=frame):
                value = metadata
                if position in edited_frames and metadata["key"] is None and section.key is not None:
                    # A fixed section at the requested target may acquire a key.
                    # Its original interval remains intact; the new native key
                    # gets the host's normal insertion metadata.
                    value = {**metadata, "key": state.section_state(section)["key"]}
                state.apply_section_metadata(section, value)

            editor.change_section(frame, identifier, retain)


def _verify_preserved_pose_curves(scene, rig, before, after, edited_frames, tracks):
    """Allow only target key insertion; preserve metadata and other authored keys."""
    from . import curves, state

    edited_frames, tracks = set(edited_frames), {str(value) for value in tracks}
    owned = {str(value) for value in rig.owned}
    protected = 0
    for key, old in before["tracks"].items():
        new = after["tracks"][key]
        expected_keys = sorted(set(old["keys"]) | edited_frames) if key in tracks else old["keys"]
        if new["keys"] != expected_keys:
            raise curves.CharacterVerificationError(
                "Durable pose restore changed unrelated native keys: " + key
            )
        extra_sections = set(new["sections"]) - set(old["sections"])
        if extra_sections - {str(frame) for frame in edited_frames} or (extra_sections and key not in tracks):
            raise curves.CharacterVerificationError(
                "Durable pose restore added unrelated native sections: " + key
            )
        for position, section in old["sections"].items():
            observed = new["sections"].get(position)
            if (
                int(position) in edited_frames
                and section["key"] is None
                and observed is not None
                and observed["key"] is not None
                and key in tracks
            ):
                observed = {**observed, "key": None}
            if section != observed:
                raise curves.CharacterVerificationError(
                    "Durable pose restore changed existing curve metadata: " + key + "/" + position
                )
    authored = {
        "data:" + str(binding.position_data): str(binding.object_id) for binding in rig.bindings.values()
    }
    authored.update(
        {
            "data:" + str(binding.rotation_data): str(binding.object_id)
            for group in rig.fingers.values()
            for binding in group.values()
        }
    )
    for key, channel in before["channels"].items():
        if channel["owner"] not in owned or channel["mode"] == "Static" or channel["kind"] == "setting":
            if not state.channel_equal(channel, after["channels"][key]):
                raise curves.CharacterVerificationError(
                    "Durable pose restore changed protected native data: " + key
                )
            protected += 1
        elif key in authored:
            owner = authored[key]
            keyframes = next(
                (track["keys"] for track in before["tracks"].values() if owner in track["objects"]), None
            )
            if keyframes is None or channel["mode"] != "Animation":
                raise curves.CharacterVerificationError(
                    "Authored pose channel has no native animation track: " + key
                )
            for frame in keyframes:
                if frame not in edited_frames and not state.equivalent_values(
                    channel["values"][frame], after["channels"][key]["values"][frame]
                ):
                    raise curves.CharacterVerificationError(
                        "Durable pose restore changed an untouched authored key: " + key + "/" + str(frame)
                    )
    return {
        "existing_curve_metadata": "exact",
        "untouched_authored_keys_verified": True,
        "protected_channels": protected,
        "new_keys_allowed_only_at": sorted(edited_frames),
    }


def _point_plan(scene, rig, entries, sampled_motion, *, coordinate_limit=10000):
    limit = 64 if sampled_motion else 8
    if type(entries) is not list or not 1 <= len(entries) <= limit:
        raise ValueError("Provide 1-" + str(limit) + " existing character poses.")
    plan, previous = [], -1
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"frame", "pose"}:
            raise ValueError("Each character entry requires frame and complete Point pose.")
        frame = _frame(scene, entry["frame"])
        if frame <= previous:
            raise ValueError("Character pose frames must be distinct and increasing.")
        previous = frame
        pose = entry["pose"]
        if type(pose) is not dict or set(pose) != set(rig.controls):
            raise ValueError("Character poses must contain exactly every discovered native Point ID.")
        targets = {}
        for key, patch in pose.items():
            if type(patch) is not dict or set(patch) != {"position"}:
                raise ValueError("Character Point patches require exactly a world-space position.")
            target = finite_vector(patch["position"], 3)
            if coordinate_limit is not None and any(abs(component) > coordinate_limit for component in target):
                raise ValueError("Character Point targets exceed 10000 scene units.")
            targets[key] = target
        if sampled_motion:
            _validate_rigid_targets(scene, rig, frame, targets)
        plan.append((frame, targets))
    return plan


def _validate_rigid_targets(scene, rig, frame, targets):
    """A complete sampled plan must retain each native bound rigid Point cluster."""
    data = scene.data_viewer()
    for link in rig.native_links:
        keys = list(
            dict.fromkeys(
                link[name]
                for name in ("main_point", "direction_point", "additional_point")
                if link.get(name) in targets
            )
        )
        for index, first in enumerate(keys):
            for second in keys[index + 1 :]:
                old_first = _position(data.get_data_value(rig.bindings[first].position_data, frame))
                old_second = _position(data.get_data_value(rig.bindings[second].position_data, frame))
                baseline = math.dist(old_first, old_second)
                requested = math.dist(targets[first], targets[second])
                if abs(baseline - requested) > max(0.02, baseline * 0.001):
                    raise ValueError(
                        "Sampled Point plan changes a native rigid control cluster: " + link["box_id"]
                    )


def _apply_points(
    scene,
    rig,
    entries,
    *,
    sampled_motion=False,
    preserve_existing_curves=False,
    after_interpolation=None,
    edited_finger_channels=(),
    defer_checkpoint_acceptance=False,
):
    from . import curves

    csc = __import__("csc")
    plan = _point_plan(scene, rig, entries, sampled_motion)
    readiness, tracks = _character_preflight(scene, rig, [frame for frame, _ in plan])
    if sampled_motion:
        _, _, view, _ = common.current_context()
        boundary, end = view.animation_boundary(), scene.data_viewer().get_animation_size() - 1
        if (
            boundary.first_frame > 0
            or boundary.first_visible_frame > 0
            or boundary.last_frame < end
            or boundary.last_visible_frame < end
        ):
            raise ValueError("Sampled character writes require full native playback and visible boundaries.")
    before = curves.checkpoint_before(scene)

    def apply(model, updater):
        import numpy

        editor, layers = model.data_editor(), model.layers_editor()
        for identifier in tracks:
            for frame, _ in plan:
                if not preserve_existing_curves or not scene.layers_viewer().layer(identifier).is_key(frame):
                    layers.set_fixed_interpolation_or_key_if_need(identifier, frame, True)
            if not preserve_existing_curves:
                for frame in list(scene.layers_viewer().layer(identifier).sections):
                    if frame < readiness["stored_frames"] - 1:

                        def linear(section):
                            section.interval.interpolation = csc.layers.layer.Interpolation.LINEAR

                        layers.change_section(frame, identifier, linear)
        if preserve_existing_curves:
            _restore_existing_sections(scene, layers, tracks, before, {frame for frame, _ in plan})
        for frame, targets in plan:
            changed = set()
            for key in sorted(targets):
                channel = rig.bindings[key].position_data
                template = scene.data_viewer().get_data_value(channel, frame)
                native = numpy.asarray(targets[key], dtype=numpy.float32).reshape(template.shape)
                editor.set_data_value(channel, frame, native)
                changed.add(channel)
            updater.run_update(changed, frame)
        layers.normalize_sections(scene)
        interpolator = updater.get_interpolator()
        interpolator.reload()
        interpolator.interpolate()
        if after_interpolation is not None:
            after_interpolation(model, updater)

    common.modify(scene, "Set checked character Point poses", apply)
    after = _state_after_character(
        scene,
        rig,
        before,
        {str(identifier) for identifier in tracks},
        edited_finger_channels=edited_finger_channels,
    )
    curve_evidence = None
    if preserve_existing_curves:
        curve_evidence = _verify_preserved_pose_curves(
            scene, rig, before, after, [frame for frame, _ in plan], tracks
        )
    geometry = curves.verify_geometry(scene, rig)
    data = scene.data_viewer()
    observed, worst = [], 0.0
    for frame, targets in plan:
        actual = {
            key: {"position": _position(data.get_data_value(rig.bindings[key].position_data, frame))}
            for key in rig.controls
        }
        deviations = {key: math.dist(targets[key], actual[key]["position"]) for key in rig.controls}
        error = max(deviations.values())
        if error > 3:
            raise curves.CharacterVerificationError(
                "Native solver target deviation exceeds three scene units."
            )
        worst = max(worst, error)
        observed.append({"frame": frame, "pose": actual, "solver_adjustments": deviations})
    result = {
        "character": rig.id,
        "fingerprint": rig.fingerprint,
        "execution_state": "completed",
        "transaction_count": 1,
        "max_solver_adjustment": worst,
        "geometry": geometry,
        "curve_policy": (
            "Existing section metadata and untouched authored keys retained; native adjacent interpolation recomputed."
            if preserve_existing_curves
            else "Whole-character keys and LINEAR intervals; authored easing is not preserved."
        ),
    }
    if curve_evidence is not None:
        result["curve_preservation"] = curve_evidence
    if sampled_motion:
        result.update(
            sampled_motion=True,
            sample_count=len(observed),
            actual_point_hashes=[
                {
                    "frame": row["frame"],
                    "sha256": hashlib.sha256(json.dumps(row["pose"], sort_keys=True).encode()).hexdigest(),
                }
                for row in observed
            ],
            readback_tool="semantic_pose_read_sequence",
        )
    else:
        result["keyframes"] = observed
    if not defer_checkpoint_acceptance:
        curves.accept_verified_checkpoint(scene, after)
    return result


def character_pose_apply(pose, character=None, frame=None):
    _, _, _, scene = _write_identity()
    rig = resolve_character(scene, character)
    return _apply_points(scene, rig, [{"frame": _frame(scene, frame), "pose": pose}])


def _scene_path(view):
    from pathlib import Path

    path = view.get_path_name()
    if not path:
        raise ValueError("Save the character scene before capturing a durable pose.")
    resolved = str(Path(path).resolve())
    return os.path.normcase(resolved).replace("\\", "/")


def _pose_digest(record):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def character_pose_snapshot(path, character=None, frame=None):
    from . import state

    _, _, view, scene = common.current_context()
    destination = common.workspace_path(path, suffixes=(".json",))
    rig, position = resolve_character(scene, character, require_hands=True), _frame(scene, frame)
    pose = character_pose(rig.id, position)
    payload = {
        "format": "mcp-for-cascadeur.character-pose",
        "version": 2,
        "scene_path": _scene_path(view),
        "character": rig.id,
        "fingerprint": rig.fingerprint,
        "frame": position,
        "points": pose["pose"],
        "joints": pose["joints"],
        "finger_rotations": {
            str(binding.object_id): state.encode(
                scene.data_viewer().get_data_value(binding.rotation_data, position)
            )
            for group in rig.fingers.values()
            for binding in group.values()
        },
    }
    _validate_finger_rotation_records(payload["finger_rotations"], payload["joints"])
    record = {**payload, "sha256": _pose_digest(payload)}
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    if len(encoded) > 2 * 1024 * 1024:
        raise ValueError("Durable character pose exceeds two MiB.")
    temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive publication avoids overwriting a racing user-created file.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "character": rig.id,
        "frame": position,
        "point_count": len(rig.controls),
        "joint_count": len(rig.joints),
        "format_version": 2,
        "finger_euler_fidelity": True,
        "scope": "One complete Point pose and authored finger-local rotations, plus read-only Joint verification; no whole-clip backup.",
    }


def _load_character_pose(path):
    source = common.workspace_path(path, must_exist=True, suffixes=(".json",))
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Durable character pose exceeds two MiB.")
    record = json.loads(source.read_text(encoding="utf-8"))
    fields = {
        "format",
        "version",
        "scene_path",
        "character",
        "fingerprint",
        "frame",
        "points",
        "joints",
        "sha256",
    }
    if (
        type(record) is not dict
        or not fields.issubset(record)
        or record["format"] != "mcp-for-cascadeur.character-pose"
        or type(record["version"]) is not int
        or record["version"] not in (1, 2)
        or set(record) != (fields if record["version"] == 1 else fields | {"finger_rotations"})
    ):
        raise ValueError("Unsupported durable character pose format.")
    if any(type(record[key]) is not str for key in ("scene_path", "character", "fingerprint", "sha256")):
        raise ValueError("Invalid durable character pose identity.")
    if type(record["points"]) is not dict or type(record["joints"]) is not dict:
        raise ValueError("Durable character pose needs native Point and Joint records.")
    payload = {key: value for key, value in record.items() if key != "sha256"}
    if _pose_digest(payload) != record["sha256"]:
        raise ValueError("Durable character pose checksum differs; no edit started.")
    for identifier, joint in record["joints"].items():
        if (
            type(joint) is not dict
            or set(joint) != {"id", "name", "local", "global"}
            or joint["id"] != identifier
            or type(joint["name"]) is not str
        ):
            raise ValueError("Invalid durable Joint verification record.")
        for space in ("local", "global"):
            pose = joint[space]
            if type(pose) is not dict or set(pose) != {"position", "rotation_wxyz"}:
                raise ValueError("Invalid durable Joint transform record.")
            finite_vector(pose["position"], 3)
            normalized_quaternion(pose["rotation_wxyz"])
    if record["version"] == 2:
        _validate_finger_rotation_records(record["finger_rotations"], record["joints"])
    return record


def _validate_finger_rotation_records(rotations, joints):
    if type(rotations) is not dict or len(rotations) != 30 or set(rotations) - set(joints):
        raise ValueError("Durable v2 pose requires exactly 30 native finger Joint rotation records.")
    for identifier, rotation in rotations.items():
        if (
            type(identifier) is not str
            or type(rotation) is not dict
            or set(rotation) != {"__native__", "quaternion_wxyz", "euler_radians"}
            or rotation["__native__"] != "rotation"
        ):
            raise ValueError("Durable finger rotation requires native Euler and quaternion state.")
        finite_vector(rotation["euler_radians"], 3)
        quaternion = finite_vector(rotation["quaternion_wxyz"], 4)
        if abs(sum(value * value for value in quaternion) - 1) > 1e-4:
            raise ValueError("Durable finger rotation requires a unit quaternion.")
        if _rotation_error(quaternion, joints[identifier]["local"]["rotation_wxyz"]) > 1e-4:
            raise ValueError("Durable finger rotation differs from its recorded native Joint identity.")


def _rotation_error(first, second):
    left, right = normalized_quaternion(first), normalized_quaternion(second)
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return 2 * math.acos(min(1.0, dot))


def _verify_joint_poses(expected, actual):
    """Verify real Joint identity and geometry with explicit angular tolerances."""
    from . import curves

    if expected.keys() != actual.keys():
        raise curves.CharacterVerificationError("Restored Joint membership differs from the durable pose.")
    worst_position, worst_rotation = 0.0, 0.0
    for identifier, previous in expected.items():
        observed = actual[identifier]
        if any(previous[name] != observed[name] for name in ("id", "name")):
            raise curves.CharacterVerificationError("Restored Joint identity differs: " + identifier)
        for space in ("local", "global"):
            position = math.dist(previous[space]["position"], observed[space]["position"])
            angle = _rotation_error(previous[space]["rotation_wxyz"], observed[space]["rotation_wxyz"])
            worst_position, worst_rotation = max(worst_position, position), max(worst_rotation, angle)
            if position > 0.01 or angle > 0.001:
                raise curves.CharacterVerificationError(
                    "Restored native Joint pose differs from the durable snapshot: "
                    + identifier
                    + "/"
                    + space
                )
    return {
        "max_position_error": worst_position,
        "max_rotation_error_radians": worst_rotation,
        "position_tolerance": 0.01,
        "rotation_tolerance_radians": 0.001,
        "quaternion_sign_invariant": True,
    }


def character_pose_restore(path, frame=None):
    from . import curves, state

    csc, _, view, scene = _write_identity()
    record = _load_character_pose(path)
    rig = resolve_character(scene, record["character"], require_hands=True)
    if _scene_path(view) != record["scene_path"] or rig.fingerprint != record["fingerprint"]:
        raise ValueError("Durable pose belongs to another saved scene or native rig topology.")
    if set(record["joints"]) != {str(item) for item in rig.joints}:
        raise ValueError("Durable pose Joint membership differs from the active native character.")
    position = _frame(scene, record["frame"] if frame is None else frame)
    bindings = [binding for group in rig.fingers.values() for binding in group.values()]
    if record["version"] == 2 and set(record["finger_rotations"]) != {
        str(binding.object_id) for binding in bindings
    }:
        raise ValueError("Durable finger Joint membership differs from validated native hand bindings.")
    fingers = []
    for binding in bindings:
        quaternion = normalized_quaternion(record["joints"][str(binding.object_id)]["local"]["rotation_wxyz"])
        native = (
            state.decode(record["finger_rotations"][str(binding.object_id)])
            if record["version"] == 2
            else csc.math.Rotation.from_quaternion(*quaternion)
        )
        if _rotation_error(quaternion, _quaternion(native)) > 1e-4:
            raise ValueError("Durable Euler rotation does not reproduce its recorded quaternion.")
        fingers.append((binding, quaternion, native))

    def restore_fingers(model, updater):
        editor = model.data_editor()
        for binding, _, native in fingers:
            editor.set_data_value(binding.rotation_data, position, native)
        updater.run_update({binding.rotation_data for binding, _, _ in fingers}, position)
        interpolator = updater.get_interpolator()
        interpolator.reload()
        interpolator.interpolate()

    result = _apply_points(
        scene,
        rig,
        [{"frame": position, "pose": record["points"]}],
        preserve_existing_curves=True,
        after_interpolation=restore_fingers,
        edited_finger_channels={str(binding.rotation_data) for binding, _, _ in fingers},
        defer_checkpoint_acceptance=True,
    )
    finger_error = max(
        (
            _rotation_error(
                quaternion, _quaternion(scene.data_viewer().get_data_value(binding.rotation_data, position))
            )
            for binding, quaternion, _ in fingers
        ),
        default=0.0,
    )
    if finger_error > 0.0001:
        raise curves.CharacterVerificationError(
            "Restored authored finger-local rotation differs from the durable pose."
        )
    euler_error = None
    if record["version"] == 2:
        euler_error = max(
            abs(a - b)
            for binding, _, _ in fingers
            for a, b in zip(
                record["finger_rotations"][str(binding.object_id)]["euler_radians"],
                state.encode(scene.data_viewer().get_data_value(binding.rotation_data, position))[
                    "euler_radians"
                ],
            )
        )
        if euler_error > 1e-4:
            raise curves.CharacterVerificationError(
                "Restored finger Euler winding differs from the durable pose."
            )
    actual = {str(item): _joint_pose(scene, item, position) for item in rig.joints}
    joint_verification = _verify_joint_poses(record["joints"], actual)
    from . import recovery

    checkpoint = recovery.latest_committed_checkpoint(scene)
    if checkpoint is not None:
        curves.accept_verified_checkpoint(scene, checkpoint)
    result.update(
        restored_from=path,
        joint_pose_verified=True,
        authored_finger_channels_restored=len(fingers),
        authored_finger_max_rotation_error_radians=finger_error,
        joint_verification=joint_verification,
        durable_format_version=record["version"],
        finger_euler_fidelity=record["version"] == 2,
        finger_max_euler_error_radians=euler_error,
        compatibility_note=(
            "Native Euler winding restored and verified."
            if record["version"] == 2
            else "Legacy v1 contains quaternions only; original finger Euler winding cannot be recovered from that file."
        ),
        scope="One character Point/finger pose; current curve metadata and untouched authored keys retained. Native adjacent interpolation is recomputed.",
    )
    return result


def _semantic_entries(scene, rig, keyframes, *, sampled_motion=False):
    limit = 64 if sampled_motion else 8
    if type(keyframes) is not list or not 1 <= len(keyframes) <= limit:
        raise ValueError("Provide 1-" + str(limit) + " semantic keyframes.")
    result = []
    for entry in keyframes:
        if type(entry) is not dict or set(entry) != {"frame", "pose"}:
            raise ValueError("Each semantic keyframe requires frame and pose.")
        frame = _frame(scene, entry["frame"])
        patch = entry["pose"]
        if type(patch) is not dict or not patch or set(patch) - rig.roles.keys():
            raise ValueError("Choose known semantic roles from rig_semantics.")
        if sampled_motion and set(patch) != set(rig.roles):
            raise ValueError("Sampled semantic motion requires every role and every Point slot.")
        current = {
            key: {"position": _position(scene.data_viewer().get_data_value(binding.position_data, frame))}
            for key, binding in rig.bindings.items()
        }
        for role, slots in patch.items():
            if type(slots) is not dict or not slots or set(slots) - rig.roles[role].keys():
                raise ValueError("Semantic role contains unknown or empty Point slots.")
            if sampled_motion and set(slots) != set(rig.roles[role]):
                raise ValueError("Sampled semantic motion requires complete role slots.")
            for slot, value in slots.items():
                current[rig.roles[role][slot]] = {"position": finite_vector(value, 3)}
        result.append({"frame": frame, "pose": current})
    return result


def semantic_pose_apply_sequence(keyframes, character=None, sampled_motion=False):
    _, _, _, scene = _write_identity()
    rig = resolve_character(scene, character, require_semantics=True, require_hands=True)
    return _apply_points(
        scene,
        rig,
        _semantic_entries(scene, rig, keyframes, sampled_motion=sampled_motion),
        sampled_motion=sampled_motion,
    )


def semantic_pose_apply(pose, character=None, frame=None):
    _, _, _, scene = _write_identity()
    return semantic_pose_apply_sequence([{"frame": _frame(scene, frame), "pose": pose}], character)


def semantic_pose_offset(frames, offsets, character=None):
    _, _, _, scene = _write_identity()
    rig = resolve_character(scene, character, require_semantics=True, require_hands=True)
    if type(offsets) is not dict or not offsets or set(offsets) - rig.roles.keys():
        raise ValueError("Offsets require known nonempty semantic roles.")
    if type(frames) is not list or not 1 <= len(frames) <= 8:
        raise ValueError("Provide 1-8 distinct increasing frames.")
    translations = {role: finite_vector(value, 3) for role, value in offsets.items()}
    entries = []
    for frame in frames:
        pose = semantic_read_from(scene, rig, _frame(scene, frame), list(offsets), False)["pose"]
        entries.append(
            {
                "frame": frame,
                "pose": {
                    role: {
                        slot: [component + shift for component, shift in zip(position, translations[role])]
                        for slot, position in slots.items()
                    }
                    for role, slots in pose.items()
                },
            }
        )
    result = _apply_points(scene, rig, _semantic_entries(scene, rig, entries))
    result["note"] = "Relative offsets accumulate. Inspect actual state after a timeout before retrying."
    return result


def _finger_preflight(scene, rig, plan):
    from . import curves, playback

    playback.require_idle()
    curves.verify_supported(scene)
    curves.character_tracks(scene, rig)
    allowed = {binding.object_id for group in rig.fingers.values() for binding in group.values()}
    model = scene.model_viewer()
    allowed.update(
        item
        for item in rig.joints
        if model.get_object_name(item)
        in {"f_" + finger + "4_" + suffix for finger in FINGERS for suffix in ("l", "r")}
    )
    allowed.update(
        item
        for item in rig.owned
        if model.get_object_type_name(item) == "Box"
        and any(
            link["box_id"] == str(item) and link["joint"] in {str(value) for value in allowed}
            for link in rig.native_links
        )
    )
    tracks = set()
    for frame, entries in plan:
        _editable_frame(scene, frame)
        for binding, _ in entries:
            layer = scene.layers_viewer().layer(binding.track_id)
            if layer.is_locked or set(layer.obj_ids) - allowed:
                raise ValueError("Finger track is locked or contains non-finger objects.")
            tracks.add(str(binding.track_id))
    return tracks


def hand_pose_apply_sequence(keyframes, character=None):
    from . import curves, state

    csc, _, _, scene = _write_identity()
    rig = resolve_character(scene, character, require_hands=True)
    if type(keyframes) is not list or not 1 <= len(keyframes) <= 8:
        raise ValueError("Provide 1-8 finger keyframes.")
    plan, previous = [], -1
    for entry in keyframes:
        if type(entry) is not dict or set(entry) != {"frame", "hands"}:
            raise ValueError("Hand keyframes require frame and hands.")
        frame = _frame(scene, entry["frame"])
        if frame <= previous:
            raise ValueError("Finger frames must be distinct and increasing.")
        previous = frame
        hands, changes = entry["hands"], []
        if type(hands) is not dict or not hands or set(hands) - {"left", "right"}:
            raise ValueError("hands must contain left and/or right finger patches.")
        for side, patch in hands.items():
            if type(patch) is not dict or not patch or set(patch) - rig.fingers[side].keys():
                raise ValueError("Use exact finger channel names returned by hand_pose_read.")
            for name, quaternion in patch.items():
                components = finite_vector(quaternion, 4)
                if abs(sum(component * component for component in components) - 1) > 1e-4:
                    raise ValueError("Finger rotations require unit quaternions in wxyz order.")
                changes.append((rig.fingers[side][name], normalized_quaternion(components)))
        plan.append((frame, changes))
    edited_tracks = _finger_preflight(scene, rig, plan)
    before = curves.checkpoint_before(scene)
    excluded = {
        "f_" + finger + str(index) + "_" + suffix
        for finger in FINGERS
        for index in range(1, 5)
        for suffix in ("l", "r")
    }
    body_joints = sorted(
        (item for item in rig.joints if scene.model_viewer().get_object_name(item) not in excluded), key=str
    )
    count = scene.data_viewer().get_animation_size()
    body_before = [
        {str(item): _joint_pose(scene, item, frame) for item in body_joints} for frame in range(count)
    ]

    def apply(model, updater):
        editor, layers = model.data_editor(), model.layers_editor()
        for frame, changes in plan:
            channels = set()
            for binding, quaternion in changes:
                layers.set_fixed_interpolation_or_key_if_need(binding.track_id, frame, True)
                editor.set_data_value(
                    binding.rotation_data, frame, csc.math.Rotation.from_quaternion(*quaternion)
                )
                channels.add(binding.rotation_data)
            updater.run_update(channels, frame)
        layers.normalize_sections(scene)
        interpolator = updater.get_interpolator()
        interpolator.reload()
        interpolator.interpolate()

    common.modify(scene, "Set checked finger-local poses", apply)
    after = curves._capture(scene)
    curves.verify_structure(before, after, preserve_tracks=False)
    for key, track in before["tracks"].items():
        if key not in edited_tracks and track != after["tracks"][key]:
            raise curves.CharacterVerificationError("Hand edit changed a body track: " + key)
    for key, channel in before["channels"].items():
        if channel["kind"] == "setting" or channel["mode"] == "Static":
            if not state.channel_equal(channel, after["channels"][key]):
                raise curves.CharacterVerificationError(
                    "Hand edit changed a static or setting channel: " + key
                )
    for key, binding in rig.bindings.items():
        channel_key = "data:" + str(binding.position_data)
        if not state.channel_equal(before["channels"][channel_key], after["channels"][channel_key]):
            raise curves.CharacterVerificationError("Hand edit moved a body Point: " + key)
    for frame, previous_body in enumerate(body_before):
        actual = {str(item): _joint_pose(scene, item, frame) for item in body_joints}
        if not state.equivalent_values(previous_body, actual):
            raise curves.CharacterVerificationError(
                "Hand edit changed a non-finger Joint at frame " + str(frame)
            )
    for frame, changes in plan:
        for binding, quaternion in changes:
            actual = _quaternion(scene.data_viewer().get_data_value(binding.rotation_data, frame))
            distance = min(
                sum((a - b) ** 2 for a, b in zip(actual, quaternion)),
                sum((a + b) ** 2 for a, b in zip(actual, quaternion)),
            )
            if distance > 1e-7:
                raise curves.CharacterVerificationError(
                    "Native finger rotation readback differs from the requested value."
                )
    curves.accept_verified_checkpoint(scene, after)
    return {
        "character": rig.id,
        "execution_state": "completed",
        "transaction_count": 1,
        "body_preserved": True,
        "frames_verified": count,
        "keyframes": [hand_pose_read(rig.id, frame) for frame, _ in plan],
    }


def character_edit_readiness(frames, operation, roles=None, character=None):
    from . import curves, recovery

    _, _, view, scene = common.current_context()
    errors, details = [], {}
    try:
        if not view.get_path_name():
            raise ValueError("Save the character scene before a checked edit.")
        if type(frames) is not list or not 1 <= len(frames) <= 8 or frames != sorted(set(frames)):
            raise ValueError("Provide 1-8 increasing existing frames.")
        rig = resolve_character(scene, character, require_semantics=True, require_hands=True)
        positions = [_frame(scene, frame) for frame in frames]
        names = selected_roles(rig, roles)
        if operation == "character_offset_preserving_curves":
            selected = {key for name in names for key in rig.roles[name].values()}
            details = curves.curve_preflight(scene, rig, positions, selected)
        elif operation in ("semantic_pose_apply_sequence", "semantic_pose_offset"):
            details, _ = _character_preflight(scene, rig, positions)
        else:
            raise ValueError("Unsupported character readiness operation.")
        curves.verify_geometry(scene, rig)
    except Exception as error:
        errors.append({"code": type(error).__name__, "message": str(error)})
    capacity = recovery.readiness(scene)
    limits = capacity.get("capacity", {})
    if (
        capacity.get("writes_blocked", False)
        or limits.get("checkpoints_remaining", 0) < 1
        or limits.get("transactions_remaining", 0) < 1
    ):
        errors.append({"code": "RecoveryNotReady", "message": str(capacity)})
    return {
        "ready": not errors,
        "operation": operation,
        "blockers": errors,
        "native_preflight": details,
        "recovery": capacity,
        "advisory_only": True,
        "history_allocated": False,
        "note": "Target feasibility and external playback are rechecked by the actual write.",
    }


_CHARACTER = {"type": "string", "minLength": 1, "maxLength": 512}
_FRAME = {"type": "integer", "minimum": 0, "maximum": 100000}
_ROLES = {
    "type": "array",
    "items": {"type": "string", "enum": list(BODY_GROUPS)},
    "minItems": 1,
    "maxItems": len(BODY_GROUPS),
    "uniqueItems": True,
}
TOOLS = [
    Tool(
        "characters_list",
        "Discover actual RigInfo owner IDs and Joint membership counts in the current scene.",
        characters_list,
    ),
    Tool(
        "character_skeleton",
        "Read native character hierarchy, Point membership, RigAdditionalInfo links and tracks.",
        character_skeleton,
        {"character": _CHARACTER, "include_track_sections": {"type": "boolean"}},
    ),
    Tool(
        "character_pose",
        "Read every native Point target and optional read-only Joint transforms for one RigInfo character.",
        character_pose,
        {"character": _CHARACTER, "frame": _FRAME, "include_joint_state": {"type": "boolean"}},
    ),
    Tool(
        "rig_semantics",
        "Validate complete Cascy hierarchy, Point membership and native bindings, then expose semantic role slots.",
        rig_semantics,
        {"character": _CHARACTER},
    ),
    Tool(
        "semantic_pose_read",
        "Read selected semantic Point groups and optional Joint states after full native profile validation.",
        semantic_pose_read,
        {
            "character": _CHARACTER,
            "frame": _FRAME,
            "roles": _ROLES,
            "include_joint_state": {"type": "boolean"},
        },
    ),
    Tool(
        "semantic_pose_read_sequence",
        "Read semantic body targets at bounded existing frames without moving the playhead.",
        semantic_pose_read_sequence,
        {
            "frames": {"type": "array", "items": _FRAME, "minItems": 1, "maxItems": 64, "uniqueItems": True},
            "character": _CHARACTER,
            "roles": _ROLES,
            "include_joint_state": {"type": "boolean"},
        },
        ("frames",),
    ),
    Tool(
        "hand_pose_read",
        "Read validated Cascy finger chains through authored local rotation channels; palm targets are separate.",
        hand_pose_read,
        {
            "character": _CHARACTER,
            "frame": _FRAME,
            "side": {"type": "string", "enum": ["left", "right", "both"]},
        },
    ),
    Tool(
        "character_pose_apply",
        "Write a complete native Point target set in one checked transaction; driven body Joints remain read-only.",
        character_pose_apply,
        {"pose": {"type": "object", "maxProperties": 128}, "character": _CHARACTER, "frame": _FRAME},
        ("pose",),
        False,
    ),
    Tool(
        "character_pose_snapshot",
        "Save one complete native Point/finger pose and read-only Joint verification to a new workspace JSON for restoration across sessions.",
        character_pose_snapshot,
        {"path": {"type": "string", "minLength": 1}, "character": _CHARACTER, "frame": _FRAME},
        ("path",),
        False,
    ),
    Tool(
        "character_pose_restore",
        "Restore recorded Point targets and validated finger-local rotations after durable identity/checksum checks; preserve existing curve metadata and untouched authored keys, then verify all Joint poses.",
        character_pose_restore,
        {"path": {"type": "string", "minLength": 1}, "frame": _FRAME},
        ("path",),
        False,
    ),
    Tool(
        "semantic_pose_apply",
        "Patch named Point slots at one existing frame, merging omitted slots with real native targets. Creates LINEAR character keys.",
        semantic_pose_apply,
        {"pose": {"type": "object"}, "character": _CHARACTER, "frame": _FRAME},
        ("pose",),
        False,
    ),
    Tool(
        "semantic_pose_apply_sequence",
        "Apply 1-8 partial semantic poses or explicitly sampled_motion=true 1-64 complete rigid-control poses. Existing frames only; verified native geometry and full checkpoint recovery.",
        semantic_pose_apply_sequence,
        {
            "keyframes": {"type": "array", "minItems": 1, "maxItems": 64},
            "character": _CHARACTER,
            "sampled_motion": {"type": "boolean", "default": False},
        },
        ("keyframes",),
        False,
    ),
    Tool(
        "semantic_pose_offset",
        "Translate selected semantic groups from each existing frame's own targets in one checked transaction. Relative edits accumulate; easing is not preserved.",
        semantic_pose_offset,
        {
            "frames": {"type": "array", "items": _FRAME, "minItems": 1, "maxItems": 8, "uniqueItems": True},
            "offsets": {"type": "object"},
            "character": _CHARACTER,
        },
        ("frames", "offsets"),
        False,
    ),
    Tool(
        "hand_pose_apply_sequence",
        "Apply 1-8 validated finger-local quaternion patches in one checked transaction, preserving body Points, Joint poses and unrelated tracks.",
        hand_pose_apply_sequence,
        {"keyframes": {"type": "array", "minItems": 1, "maxItems": 8}, "character": _CHARACTER},
        ("keyframes",),
        False,
    ),
    Tool(
        "character_edit_readiness",
        "Read exact native semantic edit blockers, geometry and remaining recovery capacity without allocating snapshots or history.",
        character_edit_readiness,
        {
            "frames": {"type": "array", "items": _FRAME, "minItems": 1, "maxItems": 8, "uniqueItems": True},
            "operation": {
                "type": "string",
                "enum": [
                    "semantic_pose_apply_sequence",
                    "semantic_pose_offset",
                    "character_offset_preserving_curves",
                ],
            },
            "roles": _ROLES,
            "character": _CHARACTER,
        },
        ("frames", "operation"),
    ),
]
