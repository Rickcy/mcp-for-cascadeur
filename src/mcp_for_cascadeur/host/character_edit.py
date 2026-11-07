"""Character timing and existing-key edits with native readback postconditions."""

from __future__ import annotations

import copy
import math

from ..contracts import Tool
from . import animation, curves
from .common import current_context, modify, resolve_object
from .motion import checked_sequence, finite_vector

RETIME_INTERPOLATIONS = tuple(name for name in animation.CLASSIC_INTERPOLATIONS if name != "STEP")


def _resolve(scene, identifier, *, semantics=False, hands=False):
    from .characters import resolve_character

    return resolve_character(scene, identifier, require_semantics=semantics, require_hands=hands)


def _frames(scene, frames, maximum=8):
    if type(frames) is not list or not 1 <= len(frames) <= maximum or frames != sorted(set(frames)):
        raise ValueError(f"Provide 1-{maximum} distinct, increasing stored frames")
    return [animation._existing_frame(scene, frame) for frame in frames]


def _positions(scene, rig, frames):
    data, channels = scene.data_viewer(), curves.point_channels(rig)
    return {
        frame: {
            identifier: curves._as_vector(data.get_data_value(channel, frame))
            for identifier, channel in channels.items()
        }
        for frame in frames
    }


def _tolerance(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 3:
        raise ValueError("position_tolerance must be finite and in (0, 3] scene units")
    return float(value)


def _interpolate(updater):
    interpolator = updater.get_interpolator()
    interpolator.reload()
    interpolator.interpolate()


def _write_points(model, updater, channels, targets):
    import numpy

    editor = model.data_editor()
    for frame, positions in targets.items():
        for identifier, position in positions.items():
            with animation._native_step(f"write character Point {identifier} at frame {frame}"):
                editor.set_data_value(
                    channels[identifier], frame, numpy.asarray(position, dtype=numpy.float32)
                )
        with animation._native_step(f"solve character Points at frame {frame}"):
            updater.run_update({channels[identifier] for identifier in positions}, frame)


def _verify_targets(expected, actual, tolerance):
    worst = 0.0
    for frame, points in expected.items():
        for identifier, position in points.items():
            error = math.dist(position, actual[frame][identifier])
            worst = max(worst, error)
    if not math.isfinite(worst) or worst > tolerance:
        raise curves.CharacterVerificationError(
            f"Native Point readback exceeded the requested tolerance: {worst} > {tolerance}"
        )
    return worst


def _impact(rig, before, targets, actual):
    frames, largest = [], 0.0
    for frame, old_points in before.items():
        direct, coupled = [], []
        for identifier, old in old_points.items():
            observed = actual[frame][identifier]
            movement = math.dist(old, observed)
            requested = targets[frame].get(identifier)
            if requested is None and movement <= 0.001:
                continue
            binding = rig.bindings[identifier]
            entry = {
                "object": identifier,
                "role": binding.role,
                "slot": binding.slot,
                "before": old,
                "observed": observed,
                "movement": movement,
            }
            if requested is not None:
                entry.update(requested=requested, target_error=math.dist(requested, observed))
                direct.append(entry)
            else:
                coupled.append(entry)
            largest = max(largest, movement)
        frames.append({"frame": frame, "direct": direct, "solver_coupled": coupled})
    return {
        "frames": frames,
        "reporting_threshold": 0.001,
        "max_point_movement": largest,
        "evidence": "Native Point values before editing and after the committed solver/interpolation pass",
        "scope": "Point movement by semantic role/slot; no claim about visual motion quality",
    }


def character_pose_preserving_curves(character_id, keyframes, position_tolerance=0.01):
    _, _, _, scene = current_context()
    rig = _resolve(scene, character_id, hands=True)
    return _apply_existing_keys(scene, rig, keyframes, position_tolerance)


def _apply_existing_keys(scene, rig, keyframes, position_tolerance):
    tolerance = _tolerance(position_tolerance)
    sequence = checked_sequence(keyframes, max_frames=8, max_objects=128)
    indices = _frames(scene, [entry["frame"] for entry in sequence])
    targets, selected = {}, set()
    for entry in sequence:
        positions = {}
        for pose in entry["poses"]:
            if set(pose) != {"object", "position"}:
                raise ValueError("Curve-preserving character poses accept only Point positions")
            identifier = str(resolve_object(scene, pose["object"]))
            if identifier not in rig.controls or identifier in positions:
                raise ValueError("Each target must resolve to one distinct Point in this character")
            positions[identifier] = finite_vector(pose["position"], 3, name="position")
        targets[entry["frame"]] = positions
        selected.update(positions)
    guard = curves.CurveProtection.prepare(scene, rig, indices, selected)
    curves.verify_geometry(scene, rig)
    before_points = _positions(scene, rig, indices)
    channels = curves.point_channels(rig)

    def apply(model, updater):
        _write_points(model, updater, channels, targets)
        with animation._native_step("reload and interpolate existing character curves"):
            _interpolate(updater)

    modify(scene, "Edit existing character Point keys", apply)
    actual = _positions(scene, rig, indices)
    maximum = _verify_targets(targets, actual, tolerance)
    protection = guard.verify(scene)
    return {
        "character_id": rig.id,
        "frames": indices,
        "execution_state": "completed",
        "max_target_error": maximum,
        "position_tolerance": tolerance,
        "curve_preservation": protection,
        "edit_impact": _impact(rig, before_points, targets, actual),
    }


def character_offset_preserving_curves(
    character_id, frames, offsets=None, profile=None, objects=None, delta=None, position_tolerance=0.01
):
    _, _, _, scene = current_context()
    if objects is not None:
        if offsets is not None or profile is not None or delta is None:
            raise ValueError("Use objects with delta, or semantic offsets with an optional profile")
        rig = _resolve(scene, character_id, hands=True)
        identifiers = [str(item) for item in animation._objects(scene, objects)]
        displacement = finite_vector(delta, 3, name="delta")
        shifts = {identifier: displacement for identifier in identifiers}
    else:
        if delta is not None or type(offsets) is not dict or not offsets:
            raise ValueError("Provide semantic offsets or objects and delta")
        rig = _resolve(scene, character_id, semantics=profile is None, hands=True)
        if profile is None:
            groups = rig.roles
        else:
            from .profiles import _groups

            groups = _groups(profile, list(offsets))
        shifts = {}
        for role, value in offsets.items():
            if role not in groups:
                raise ValueError(f"Unknown semantic role: {role}")
            group = groups[role]
            identifiers = list(group.values()) if isinstance(group, dict) else list(group)
            displacement = finite_vector(value, 3, name=f"offsets.{role}")
            for identifier in identifiers:
                if identifier in shifts:
                    raise ValueError("Offset groups overlap; select each native Point only once")
                shifts[identifier] = displacement
    if set(shifts) - set(rig.controls):
        raise ValueError("Semantic groups must contain only Points from this character")
    indices = _frames(scene, frames)
    baseline = _positions(scene, rig, indices)
    sequence = [
        {
            "frame": frame,
            "poses": [
                {
                    "object": identifier,
                    "position": [
                        position + shift for position, shift in zip(baseline[frame][identifier], offset)
                    ],
                }
                for identifier, offset in shifts.items()
            ],
        }
        for frame in indices
    ]
    result = _apply_existing_keys(scene, rig, sequence, position_tolerance)
    result.update(
        offsets=offsets,
        coordinate_space="global native scene units",
        relative_edit=True,
        automatic_retry_allowed=False,
    )
    return result


def _key_map(scene, keys):
    if type(keys) is not list or not 2 <= len(keys) <= 8:
        raise ValueError("Character retiming accepts 2-8 synchronized keys")
    result = []
    fields = {"source", "target", "interpolation", "left_weight", "right_weight"}
    for key in keys:
        if type(key) is not dict or set(key) - fields or not {"source", "target"}.issubset(key):
            raise ValueError("Each retime key requires source and target plus optional interpolation/easing")
        source = animation._existing_frame(scene, key["source"])
        target = animation._existing_frame(scene, key["target"])
        if "interpolation" in key and key["interpolation"] not in RETIME_INTERPOLATIONS:
            raise ValueError("Retiming supports " + ", ".join(RETIME_INTERPOLATIONS) + " intervals")
        for field in ("left_weight", "right_weight"):
            if field in key and (
                type(key[field]) not in (int, float)
                or not math.isfinite(key[field])
                or not 0.01 <= key[field] <= 0.99
            ):
                raise ValueError("Easing weights must be finite and in [0.01, 0.99]")
        result.append({**key, "source": source, "target": target})
    source = [key["source"] for key in result]
    target = [key["target"] for key in result]
    if source != sorted(set(source)) or target != sorted(set(target)):
        raise ValueError("Source and target frames must be strictly increasing")
    if (source[0], source[-1]) != (target[0], target[-1]):
        raise ValueError("The first and last character keys must remain fixed")
    return result


def _retime_tracks(scene, rig, keys):
    curves.verify_supported(scene)
    tracks = curves.character_tracks(scene, rig)
    source = [key["source"] for key in keys]
    for identifier, layer in tracks.items():
        if list(layer.key_frame_indices()) != source or sorted(layer.sections) != source:
            raise curves.UnsupportedCurveState(
                "Every character track must contain exactly the supplied synchronized source keys"
            )
        if any(section.key is None for section in layer.sections.values()):
            raise curves.UnsupportedCurveState("Retimed sections must contain complete native key metadata")
        if any(
            curves._name(section.interval.interpolation) in {"FIXED", "AI", "NONE"}
            for section in layer.sections.values()
        ):
            raise curves.UnsupportedCurveState("Retiming cannot discard baked or AI character samples")
        if {str(obj) for obj in layer.obj_ids}.intersection(rig.controls):
            if any(
                curves._name(section.interval.common.ik_fk) != "IK"
                or (section.key is not None and curves._name(section.key.common.ik_fk) != "IK")
                for section in layer.sections.values()
            ):
                raise curves.UnsupportedCurveState("Body Point tracks require native IK mode for retiming")
    return tracks


def _contact_interval_mode(first, last):
    return "STEP" if last - first == 1 else "FIXED"


def _retime_expected_tracks(before, tracks, keys, foot_tracks):
    expected = copy.deepcopy(before["tracks"])
    for identifier in tracks:
        record = expected[str(identifier)]
        old_sections = record["sections"]
        record["keys"] = [key["target"] for key in keys]
        record["sections"] = {}
        for index, key in enumerate(keys):
            section = copy.deepcopy(old_sections[str(key["source"])])
            if index < len(keys) - 1:
                if "interpolation" in key:
                    section["interpolation"] = key["interpolation"]
                if identifier in foot_tracks:
                    section["interpolation"] = _contact_interval_mode(
                        key["target"], keys[index + 1]["target"]
                    )
            for field in ("left_weight", "right_weight"):
                if field in key:
                    section["key"][field + "_velocity"] = key[field]
            record["sections"][str(key["target"])] = section
    return expected


def _contact_targets(rig, source_poses):
    if not rig.roles:
        raise ValueError("Contact stabilization requires validated body semantics")
    groups = {
        "left_foot": ("center", "heel", "toe", "toe_direction", "toe_orientation"),
        "right_foot": ("toe", "toe_direction", "toe_orientation"),
    }
    anchors = {}
    first = next(iter(source_poses.values()))
    for role, slots in groups.items():
        for slot in slots:
            identifier = rig.roles[role][slot]
            anchor = first[identifier]
            if any(math.dist(pose[identifier], anchor) > 0.02 for pose in source_poses.values()):
                raise ValueError(
                    "Contact stabilization requires stationary left-foot and right-toe targets at every source key"
                )
            anchors[identifier] = anchor
    return anchors


def _joint_key_states(scene, rig, frames):
    return {
        frame: {
            str(identifier): {
                space: animation._read(scene, identifier, frame, space) for space in ("local", "global")
            }
            for identifier in rig.joints
        }
        for frame in frames
    }


def _verify_joint_keys(before, after, source, target):
    worst_position, worst_angle = 0.0, 0.0
    for original, destination in zip(source, target):
        for identifier, poses in before[original].items():
            for space, expected in poses.items():
                actual = after[destination][identifier][space]
                if "position" in expected:
                    worst_position = max(worst_position, math.dist(expected["position"], actual["position"]))
                if "rotation_wxyz" in expected:
                    dot = abs(sum(a * b for a, b in zip(expected["rotation_wxyz"], actual["rotation_wxyz"])))
                    worst_angle = max(worst_angle, 2 * math.acos(min(1.0, dot)))
    if worst_position > 0.01 or worst_angle > 0.001:
        raise curves.CharacterVerificationError(
            f"Retiming changed a source joint key pose: position={worst_position}, angle={worst_angle}"
        )
    return {"max_position_error": worst_position, "max_rotation_error_radians": worst_angle}


def character_retime(character_id, keys, stabilize_contacts=False):
    from . import state

    csc, _, _, scene = current_context()
    rig = _resolve(scene, character_id, semantics=stabilize_contacts, hands=True)
    mapping = _key_map(scene, keys)
    source, target = [key["source"] for key in mapping], [key["target"] for key in mapping]
    tracks = _retime_tracks(scene, rig, mapping)
    before = curves.checkpoint_before(scene)
    curves.verify_geometry(scene, rig)
    point_values = _positions(scene, rig, source)
    joint_values = _joint_key_states(scene, rig, source)
    data = scene.data_viewer()
    points = curves.point_channels(rig)
    fingers = curves.finger_channels(rig)
    finger_values = {
        identifier: [animation._copy_native(csc, data.get_data_value(reference, frame)) for frame in source]
        for identifier, reference in fingers.items()
    }
    settings = {
        key: csc.model.SettingId(record["id"])
        for key, record in before["channels"].items()
        if record["kind"] == "setting"
        and record["mode"] == "Animation"
        and record["owner"] in {str(obj) for obj in rig.owned}
    }
    setting_values = {
        key: [data.get_setting_value(reference, frame) for frame in source]
        for key, reference in settings.items()
    }
    anchors = _contact_targets(rig, point_values) if stabilize_contacts else {}
    foot_tracks = {
        scene.layers_viewer().layer_id_by_obj_id_or_null(rig.controls[identifier]) for identifier in anchors
    }
    expected_tracks = _retime_expected_tracks(before, tracks, mapping, foot_tracks)
    staged_tracks = _retime_expected_tracks(before, tracks, mapping, set())
    for identifier in tracks:
        for metadata in staged_tracks[str(identifier)]["sections"].values():
            state.validate_section_metadata(metadata)
    protected = {
        key
        for key, record in before["channels"].items()
        if record["mode"] == "Static" or record["owner"] not in {str(obj) for obj in rig.owned}
    }
    native_targets = {destination: point_values[original] for original, destination in zip(source, target)}
    terminal_frame = animation._size(scene) - 1
    skipped_terminal = terminal_frame if target[-1] == source[-1] == terminal_frame else None

    def apply(model, updater):
        layer_editor, editor = model.layers_editor(), model.data_editor()
        for identifier in tracks:
            for frame in source[1:-1]:
                with animation._native_step(
                    f"remove interior character key on track {identifier}, frame {frame}"
                ):
                    layer_editor.unset_section(frame, identifier)
            for frame in target:
                with animation._native_step(
                    f"create retimed character key on track {identifier}, frame {frame}"
                ):
                    layer_editor.set_fixed_interpolation_or_key_if_need(identifier, frame, True)
                metadata = staged_tracks[str(identifier)]["sections"][str(frame)]

                def configure(section, metadata=metadata):
                    state.apply_section_metadata(section, metadata)

                with animation._native_step(
                    f"restore retimed key metadata on track {identifier}, frame {frame}"
                ):
                    layer_editor.change_section(frame, identifier, configure)
        with animation._native_step("normalize and regenerate the retimed character update graph"):
            layer_editor.normalize_sections(scene)
            updater.generate_update()
            updater.get_interpolator().reload()
        for index, frame in enumerate(target):
            if frame == skipped_terminal:
                continue
            for identifier, reference in fingers.items():
                editor.set_data_value(reference, frame, finger_values[identifier][index])
            for key, reference in settings.items():
                editor.set_setting_value(reference, frame, setting_values[key][index])
            _write_points(model, updater, points, {frame: native_targets[frame]})
            if fingers:
                updater.run_update(set(fingers.values()), frame)
        _interpolate(updater)
        if anchors:
            for identifier in foot_tracks:
                for frame, following in zip(target, target[1:]):
                    interpolation = getattr(
                        csc.layers.layer.Interpolation, _contact_interval_mode(frame, following)
                    )

                    def bake(section, interpolation=interpolation):
                        section.interval.interpolation = interpolation

                    layer_editor.change_section(frame, identifier, bake)
            pinned = {frame: anchors for frame in range(animation._size(scene)) if frame not in target}
            _write_points(model, updater, points, pinned)
            updater.get_interpolator().reload()

    modify(scene, "Retime authored character controls", apply)
    actual = _positions(scene, rig, target)
    max_point_error = _verify_targets(native_targets, actual, 0.01)
    joint_report = _verify_joint_keys(joint_values, _joint_key_states(scene, rig, target), source, target)
    after = curves._capture(scene)
    curves.verify_structure(before, after, preserve_tracks=False)
    if after["tracks"] != expected_tracks:
        raise curves.CharacterVerificationError(
            "Retimed key layout, interpolation or easing differs from the staged metadata"
        )
    curves._protected_compare(before, after, protected)
    for identifier, reference in fingers.items():
        for index, frame in enumerate(target):
            if not state.equivalent_values(
                state.encode(finger_values[identifier][index]),
                state.encode(data.get_data_value(reference, frame)),
            ):
                raise curves.CharacterVerificationError(
                    "Retiming changed a source finger-local rotation or Euler branch"
                )
    for key, reference in settings.items():
        for index, frame in enumerate(target):
            if setting_values[key][index] != data.get_setting_value(reference, frame):
                raise curves.CharacterVerificationError("Retiming changed a source authored setting")
    metrics = curves.verify_geometry(scene, rig)
    contacts = {}
    if anchors:
        all_points = _positions(scene, rig, list(range(animation._size(scene))))
        error = max(
            math.dist(points_at_frame[identifier], anchor)
            for points_at_frame in all_points.values()
            for identifier, anchor in anchors.items()
        )
        if error > 0.03:
            raise curves.CharacterVerificationError(f"Planted contact verification failed: {error}")
        contacts = {"targets_pinned": len(anchors), "baked_tracks": len(foot_tracks), "max_error": error}
    curves.accept_verified_checkpoint(scene, after)
    return {
        "character_id": rig.id,
        "source_frames": source,
        "target_frames": target,
        "execution_state": "completed",
        "point_source_poses_verified": True,
        "max_point_error": max_point_error,
        "joint_key_verification": joint_report,
        "finger_local_channels_verified": len(fingers),
        "animated_settings_verified": len(settings),
        "unchanged_terminal_key_skipped": skipped_terminal,
        "curve_metadata_verified": True,
        "protected_channels_verified": len(protected),
        "rig_geometry": metrics,
        "contacts": contacts,
        "note": "Only authored Point positions, validated finger-local rotations and settings were written; body Joint/global/local outputs were solved natively.",
    }


def character_motion_inspect(character_id):
    from . import state

    _, _, _, scene = current_context()
    rig = _resolve(scene, character_id)
    count = animation._size(scene)
    if not 1 <= count <= 121:
        raise ValueError("Motion inspection is limited to 121 stored frames")
    viewer, data = scene.behaviour_viewer(), scene.data_viewer()
    bodies = [
        behaviour
        for behaviour in viewer.get_behaviours("RigidBody")
        if viewer.get_behaviour_owner(behaviour) in rig.owned
    ]
    centers = list(viewer.get_behaviours("CenterOfMass"))
    if not bodies:
        raise ValueError("No native rigid-body masses were found for this character")
    geometry = curves.connection_geometry(scene, rig)
    frames = []
    for frame in range(count):
        total, weighted = 0.0, [0.0, 0.0, 0.0]
        for behaviour in bodies:
            reference = viewer.get_behaviour_data(behaviour, "mass")
            mass = (
                data.get_data_value(reference, frame)
                if curves._name(data.get_data(reference).mode) == "Animation"
                else data.get_data_value(reference)
            )
            if type(mass) not in (int, float) or not math.isfinite(mass) or mass < 0:
                raise ValueError("A native rigid body has an invalid mass")
            owner = viewer.get_behaviour_owner(behaviour)
            position = animation._read(scene, owner, frame, "global")["position"]
            total += mass
            weighted = [value + mass * component for value, component in zip(weighted, position)]
        if total <= 0:
            raise ValueError("The character has no positive total rigid-body mass")
        native = []
        for behaviour in centers:
            owner = viewer.get_behaviour_owner(behaviour)
            reference = viewer.get_behaviour_data(behaviour, "pos_global_target")
            target = (
                data.get_data_value(reference, frame)
                if curves._name(data.get_data(reference).mode) == "Animation"
                else data.get_data_value(reference)
            )
            native.append(
                {
                    "object": str(owner),
                    "belongs_to_character": owner in rig.owned,
                    "position": animation._read(scene, owner, frame, "global")["position"],
                    "target": curves._as_vector(target),
                }
            )
        frames.append(
            {
                "frame": frame,
                "total_mass": total,
                "mass_weighted_center_of_mass": [value / total for value in weighted],
                "native_centers_of_mass": native,
                **geometry["frames"][frame],
            }
        )
    tracks = curves._layers(scene)
    return {
        "character_id": rig.id,
        "rigid_body_count": len(bodies),
        "frames": frames,
        "connection_count": geometry["connection_count"],
        "max_connection_anchor_error": geometry["max_connection_anchor_error"],
        "tracks": {
            str(identifier): {
                "name": layer.header.name,
                "parent": str(layer.header.parent),
                "objects": sorted(str(obj) for obj in layer.obj_ids),
                "locked": bool(layer.is_locked),
                "keys": list(layer.key_frame_indices()),
                "sections": {
                    str(frame): state.section_state(section) for frame, section in layer.sections.items()
                },
            }
            for identifier, layer in tracks.items()
            if identifier in rig.tracks
        },
    }


IDENTIFIER = animation.IDENTIFIER
KEY_MAPPING = {
    "type": "array",
    "minItems": 2,
    "maxItems": 8,
    "items": {
        "type": "object",
        "properties": {
            "source": animation.FRAME,
            "target": animation.FRAME,
            "interpolation": {"type": "string", "enum": list(RETIME_INTERPOLATIONS)},
            "left_weight": {"type": "number", "minimum": 0.01, "maximum": 0.99},
            "right_weight": {"type": "number", "minimum": 0.01, "maximum": 0.99},
        },
        "required": ["source", "target"],
        "additionalProperties": False,
    },
}
TOLERANCE = {"type": "number", "exclusiveMinimum": 0, "maximum": 3, "default": 0.01}
TOOLS = [
    Tool(
        "character_motion_inspect",
        "Read actual native mass-weighted COM, COM targets, rigid-body connection errors at every stored frame and character curve metadata. No playhead changes.",
        character_motion_inspect,
        {"character_id": IDENTIFIER},
        ("character_id",),
    ),
    Tool(
        "character_retime",
        "Retime 2-8 synchronized character keys using authored Point targets, validated finger-local rotations and settings. Preserves key metadata; verifies source poses, protected channels and full-clip body connections. Optional stationary-foot contact stabilization. Checked outer transaction/rollback required.",
        character_retime,
        {"character_id": IDENTIFIER, "keys": KEY_MAPPING, "stabilize_contacts": {"type": "boolean"}},
        ("character_id", "keys"),
        False,
    ),
    Tool(
        "character_pose_preserving_curves",
        "Edit existing Point position keys without changing key layout, interpolation, easing or tangent metadata. Protects untouched authored keys, finger channels, baked tracks, external data and settings; verifies full-clip rig geometry.",
        character_pose_preserving_curves,
        {
            "character_id": IDENTIFIER,
            "keyframes": {**animation.KEYFRAMES, "maxItems": 8},
            "position_tolerance": TOLERANCE,
        },
        ("character_id", "keyframes"),
        False,
    ),
    Tool(
        "character_offset_preserving_curves",
        "Translate semantic Point groups or explicit character Points at existing keys while preserving curve metadata and protected authored data. Relative offsets accumulate; inspect before retrying an unknown outcome.",
        character_offset_preserving_curves,
        {
            "character_id": IDENTIFIER,
            "frames": {**animation.FRAMES, "maxItems": 8},
            "offsets": {"type": "object", "additionalProperties": animation.VECTOR, "maxProperties": 128},
            "profile": {"type": "string", "minLength": 1, "maxLength": 80},
            "objects": animation.OBJECTS,
            "delta": animation.VECTOR,
            "position_tolerance": TOLERANCE,
        },
        ("character_id", "frames"),
        False,
    ),
]
