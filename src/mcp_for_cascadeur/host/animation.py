"""Native controller, track and pose editing using Cascadeur's public scene API."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from contextlib import contextmanager

from ..contracts import Tool
from .common import current_context, json_value, modify, resolve_object, workspace_path
from .motion import (
    checked_frame,
    checked_sequence,
    finite_vector,
    mirror_motion,
    normalized_quaternion,
    sample_motion,
)

_snapshots = {}
MAX_SNAPSHOTS = 32
INTERPOLATIONS = ("LINEAR", "BEZIER", "LOW_AMPLITUDE_BEZIER", "STEP", "FIXED", "CLAMPED_BEZIER")


@contextmanager
def _native_step(description):
    try:
        yield
    except Exception as error:
        raise RuntimeError(f"{description}: {type(error).__name__}: {error}") from error


def _size(scene):
    return int(scene.data_viewer().get_animation_size())


def _existing_frame(scene, frame):
    frame = scene.get_current_frame() if frame is None else checked_frame(frame)
    if not 0 <= frame < _size(scene):
        raise ValueError("Frame is outside the stored animation; add keys to extend the timeline")
    return frame


def _objects(scene, objects):
    if type(objects) is not list or not 1 <= len(objects) <= 128:
        raise ValueError("Provide 1-128 exact object names or IDs")
    resolved = [resolve_object(scene, identifier) for identifier in objects]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same object was supplied more than once")
    return resolved


def _channel(scene, object_id, channel):
    viewer = scene.behaviour_viewer()
    transform = viewer.get_behaviour_by_name(object_id, "Transform")
    if transform.is_null():
        raise ValueError(f"Object {object_id} has no Transform behaviour")
    reference = viewer.get_behaviour_data(transform, channel)
    if reference.is_null():
        raise ValueError(f"Object {object_id} has no {channel} channel")
    return reference


def _rotation(value):
    quaternion = value.to_quaternion() if hasattr(value, "to_quaternion") else value
    return normalized_quaternion([float(getattr(quaternion, axis)()) for axis in "wxyz"])


def _read(scene, object_id, frame, space):
    from .data_channels import read_data_value

    data = scene.data_viewer()
    item = {"object": str(object_id), "name": scene.model_viewer().get_object_name(object_id)}
    for field, suffix in (("position", "position"), ("rotation_wxyz", "rotation")):
        try:
            reference = _channel(scene, object_id, f"{space}_{suffix}")
        except ValueError:
            continue
        value = read_data_value(data, reference, frame)
        if field == "position":
            encoded = json_value(value)
            if (
                isinstance(encoded, list)
                and len(encoded) == 3
                and all(isinstance(component, list) and len(component) == 1 for component in encoded)
            ):
                encoded = [component[0] for component in encoded]
            item[field] = finite_vector(encoded, 3)
        else:
            item[field] = _rotation(value)
    if not {"position", "rotation_wxyz"}.intersection(item):
        raise ValueError(f"Object {object_id} has no readable Transform channels")
    return item


def _space(value):
    if value not in ("local", "global"):
        raise ValueError("space must be local or global")
    return value


def _signature(scene):
    model, layers = scene.model_viewer(), scene.layers_viewer()
    topology = sorted(
        (str(item), model.get_object_name(item), model.get_object_type_name(item))
        for item in model.get_objects()
    )
    tracks = sorted(str(item) for item in layers.all_layer_ids())
    encoded = json.dumps([topology, tracks], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _tracks(scene, identifiers):
    if type(identifiers) is not list or not 1 <= len(identifiers) <= 128:
        raise ValueError("Provide 1-128 track IDs or exact names")
    viewer, result = scene.layers_viewer(), []
    available = list(viewer.all_layer_ids())
    for identifier in identifiers:
        matches = [
            item for item in available if str(item) == identifier or viewer.header(item).name == identifier
        ]
        if len(matches) != 1:
            raise ValueError(f"Track {identifier!r} matched {len(matches)} tracks; use its ID")
        track = viewer.layer(matches[0])
        if track.is_locked:
            raise ValueError(f"Track {identifier!r} is locked")
        result.append(matches[0])
    if len(result) != len(set(result)):
        raise ValueError("The same track was supplied more than once")
    return result


def timeline_get():
    _, _, view, scene = current_context()
    bounds = view.animation_boundary()
    return {
        "frame": scene.get_current_frame(),
        "stored_frames": _size(scene),
        "playback": {"first": bounds.first_frame, "last": bounds.last_frame},
        "visible": {"first": bounds.first_visible_frame, "last": bounds.last_visible_frame},
    }


def timeline_set_frame(frame):
    _, _, _, scene = current_context()
    scene.set_current_frame(_existing_frame(scene, frame))
    return timeline_get()


def timeline_set_range(first, last, visible=True):
    _, _, view, scene = current_context()
    first, last = _existing_frame(scene, first), _existing_frame(scene, last)
    if first > last:
        raise ValueError("first must not exceed last")
    bounds = view.animation_boundary()
    bounds.first_frame, bounds.last_frame = first, last
    if visible:
        bounds.first_visible_frame, bounds.last_visible_frame = first, last
    return timeline_get()


def tracks_list(include_sections=True):
    _, _, _, scene = current_context()
    viewer, model = scene.layers_viewer(), scene.model_viewer()
    result = []
    for identifier in viewer.all_layer_ids():
        track = viewer.layer(identifier)
        item = {
            "id": str(identifier),
            "name": viewer.header(identifier).name,
            "locked": bool(track.is_locked),
            "visible": bool(track.is_visible),
            "objects": [{"id": str(obj), "name": model.get_object_name(obj)} for obj in track.obj_ids],
            "keys": list(track.key_frame_indices()),
        }
        if include_sections:
            item["sections"] = [
                {
                    "frame": frame,
                    "interpolation": section.interval.interpolation.name,
                    "ik_fk": section.interval.common.ik_fk.name,
                    "key_tangents": section.key.tangents.name,
                    "left_weight": section.key.left_weight_velocity,
                    "right_weight": section.key.right_weight_velocity,
                }
                for frame, section in track.sections.items()
            ]
        result.append(item)
    return {"tracks": result, "stored_frames": _size(scene)}


def _ensure_frame_storage(csc, scene, model, updater, layer_ids, last_frame):
    if last_frame < _size(scene):
        return
    terminal_frame = last_frame + 1
    editor = model.layers_editor()
    for identifier in set(layer_ids):
        with _native_step(f"allocate terminal section at frame {terminal_frame} on track {identifier}"):
            editor.set_section(csc.layers.layer.Section(), terminal_frame, identifier)
    with _native_step("resize stored animation to endpoint sections"):
        model.fit_animation_size_by_layers()
    with _native_step("generate native update after timeline allocation"):
        updater.generate_update()
    if _size(scene) <= terminal_frame:
        raise RuntimeError("Cascadeur did not allocate animation data for the requested endpoint")


def tracks_set_keys(tracks, frames, mode="insert", extend_timeline=False):
    csc, _, _, scene = current_context()
    identifiers = _tracks(scene, tracks)
    if type(frames) is not list or not 1 <= len(frames) <= 256:
        raise ValueError("Provide 1-256 frames")
    positions = sorted({checked_frame(frame) for frame in frames})
    if len(positions) != len(frames):
        raise ValueError("Keyframe positions must be unique")
    if mode not in ("insert", "delete"):
        raise ValueError("mode must be insert or delete")
    expected_deleted_keys = {}
    if mode == "delete":
        if 0 in positions:
            raise ValueError("The initial frame 0 is protected from key deletion")
        for identifier in identifiers:
            existing = list(scene.layers_viewer().layer(identifier).key_frame_indices())
            if any(frame not in existing for frame in positions):
                raise ValueError("Every deleted frame must be an existing key on every selected track")
            if 0 not in set(existing) - set(positions):
                raise ValueError("Deletion must retain an existing initial key at frame 0")
            expected_deleted_keys[str(identifier)] = sorted(set(existing) - set(positions))
    elif positions[-1] >= _size(scene) and not extend_timeline:
        raise ValueError("New keys extend the animation; set extend_timeline=true explicitly")

    def apply(model, updater):
        editor = model.layers_editor()
        if mode == "insert":
            _ensure_frame_storage(csc, scene, model, updater, identifiers, positions[-1])
        for identifier in identifiers:
            for frame in positions:
                if mode == "insert":
                    editor.set_fixed_interpolation_or_key_if_need(identifier, frame, True)
                else:
                    editor.unset_section(frame, identifier)
        editor.normalize_sections(scene)
        model.fit_animation_size_by_layers()
        updater.generate_update()

    modify(scene, "Insert animation keys" if mode == "insert" else "Delete animation keys", apply)
    if mode == "delete":
        actual = {
            str(identifier): list(scene.layers_viewer().layer(identifier).key_frame_indices())
            for identifier in identifiers
        }
        if actual != expected_deleted_keys:
            raise RuntimeError("Native key deletion did not retain the expected frame-0/key layout")
    return {
        "mode": mode,
        "tracks": [str(item) for item in identifiers],
        "frames": positions,
        "stored_frames": _size(scene),
        "execution_state": "completed",
    }


def tracks_set_interpolation(tracks, frames, interpolation):
    csc, _, _, scene = current_context()
    identifiers = _tracks(scene, tracks)
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
    if type(frames) is not list or not 1 <= len(frames) <= 256:
        raise ValueError("Provide 1-256 existing keys")
    positions = [checked_frame(frame) for frame in frames]
    for identifier in identifiers:
        track = scene.layers_viewer().layer(identifier)
        if any(not track.is_key(frame) for frame in positions):
            raise ValueError("Interpolation changes require an existing key on each selected track")
    mode = getattr(csc.layers.layer.Interpolation, interpolation)

    def apply(model, updater):
        editor = model.layers_editor()
        for identifier in identifiers:
            for frame in positions:

                def change(section):
                    section.interval.interpolation = mode

                editor.change_section(frame, identifier, change)
        editor.normalize_sections(scene)
        updater.generate_update()

    modify(scene, "Change track interpolation", apply)
    return {
        "tracks": [str(item) for item in identifiers],
        "frames": positions,
        "interpolation": interpolation,
        "execution_state": "completed",
        "note": "Curve modes can alter neighboring motion; inspect inbetweens and contacts.",
    }


def _copy_native(csc, value):
    if hasattr(value, "copy"):
        return value.copy()
    if hasattr(value, "to_quaternion"):
        if callable(getattr(value, "to_euler_angles", None)) and callable(
            getattr(csc.math.Rotation, "from_euler", None)
        ):
            angles = json_value(value.to_euler_angles())
            if (
                isinstance(angles, list)
                and len(angles) == 3
                and all(isinstance(item, list) and len(item) == 1 for item in angles)
            ):
                angles = [item[0] for item in angles]
            return csc.math.Rotation.from_euler(*finite_vector(angles, 3))
        return csc.math.Rotation.from_quaternion(*_rotation(value))
    if all(callable(getattr(value, axis, None)) for axis in "wxyz"):
        return csc.math.Quaternion(*_rotation(value))
    return value


def tracks_retime(tracks, source_frames, target_frames):
    """Move synchronized interior keys while retaining their section metadata."""
    from . import state

    csc, _, _, scene = current_context()
    identifiers = _tracks(scene, tracks)
    if (
        type(source_frames) is not list
        or type(target_frames) is not list
        or not 2 <= len(source_frames) <= 128
        or len(source_frames) != len(target_frames)
    ):
        raise ValueError("Provide equal source_frames/target_frames lists with 2-128 keys")
    source = [checked_frame(frame) for frame in source_frames]
    targets = [_existing_frame(scene, frame) for frame in target_frames]
    if any(a >= b for a, b in zip(source, source[1:])) or any(a >= b for a, b in zip(targets, targets[1:])):
        raise ValueError("Source and target keyframes must be strictly increasing")
    if (source[0], source[-1]) != (targets[0], targets[-1]):
        raise ValueError("Retiming preserves the first and last key; keep their target frames unchanged")
    viewer, data = scene.layers_viewer(), scene.data_viewer()
    behaviors = scene.behaviour_viewer()
    sections, values, settings, global_poses = {}, {}, {}, {}
    for identifier in identifiers:
        track = viewer.layer(identifier)
        if list(track.key_frame_indices()) != source:
            raise ValueError("source_frames must list every key on every selected track")
        if csc.layers.CyclesViewer(track).get_cycles_in_frames(source[0], source[-1]):
            raise ValueError("Retiming cyclic tracks requires a dedicated cycle edit")
        if sorted(track.sections) != source:
            raise ValueError(
                "Retiming requires only the listed key sections; fixed/non-key sections need explicit resampling"
            )
        if any(
            section.interval.interpolation.name not in ("LINEAR", "BEZIER", "LOW_AMPLITUDE_BEZIER", "STEP")
            for section in track.sections.values()
        ):
            raise ValueError(
                "Retiming supports LINEAR/BEZIER/LOW_AMPLITUDE_BEZIER/STEP; FIXED, AI and CLAMPED curves need a separately verified workflow"
            )
        if any(
            section.key is None or section.key.tangents.name != "Continuous"
            for section in track.sections.values()
        ):
            raise ValueError("Retiming custom spatial tangents or non-key sections is unsupported")
        sections[identifier] = {
            destination: state.section_state(track.section(original))
            for original, destination in zip(source, targets)
        }
        for obj in track.obj_ids:
            names = {behaviors.get_behaviour_name(item) for item in behaviors.get_behaviours(obj)}
            if "Transform" not in names or names - {"Basic", "Transform"}:
                raise ValueError(
                    "Retiming currently supports standalone Basic/Transform objects; use native actions for constrained rigs"
                )
            if "Basic" in names:
                basic = behaviors.get_behaviour_by_name(obj, "Basic")
                if not behaviors.get_behaviour_object(basic, "parent").is_null():
                    raise ValueError("Retiming parented transforms needs dependency-aware channel selection")
            # Standalone transforms expose both authored local inputs and
            # derived global/matrix outputs. Only re-author the local inputs;
            # the native update graph must produce their world outputs.
            for name in ("local_position", "local_rotation", "local_scale"):
                reference = _channel(scene, obj, name)
                if data.get_data(reference).mode == csc.model.DataMode.Animation:
                    values[reference] = [
                        _copy_native(csc, data.get_data_value(reference, frame)) for frame in source
                    ]
            global_poses[obj] = [_read(scene, obj, frame, "global") for frame in source]
            for reference in data.get_all_settings_id(obj):
                if data.get_setting(reference).mode == csc.model.SettingMode.Animation:
                    settings[reference] = [data.get_setting_value(reference, frame) for frame in source]
    if (len(values) + len(settings)) * len(source) > 16384:
        raise ValueError("Retiming exceeds 16384 channel-key samples; select fewer tracks")

    def apply(model, updater):
        editor, layer_editor = model.data_editor(), model.layers_editor()
        for identifier, replacements in sections.items():
            for frame in list(viewer.layer(identifier).sections):
                if frame not in replacements:
                    layer_editor.unset_section(frame, identifier)
            for frame, metadata in replacements.items():
                if not viewer.layer(identifier).is_key(frame):
                    layer_editor.set_fixed_interpolation_or_key_if_need(identifier, frame, True)

                def restore(section, metadata=metadata):
                    state.apply_section_metadata(section, metadata)

                layer_editor.change_section(frame, identifier, restore)
        layer_editor.normalize_sections(scene)
        updater.generate_update()
        interpolator = updater.get_interpolator()
        interpolator.reload()
        for index, destination in enumerate(targets):
            changed = set()
            setting_changed = False
            for reference, samples in values.items():
                if not _native_values_match(samples[index], data.get_data_value(reference, destination)):
                    editor.set_data_value(reference, destination, samples[index])
                    changed.add(reference)
            for reference, samples in settings.items():
                if samples[index] != data.get_setting_value(reference, destination):
                    editor.set_setting_value(reference, destination, samples[index])
                    setting_changed = True
            if changed or setting_changed:
                updater.run_update(changed or set(values), destination)
        interpolator.interpolate()

    modify(scene, "Retime synchronized track keys", apply)
    actual_keys = {
        str(identifier): list(viewer.layer(identifier).key_frame_indices()) for identifier in identifiers
    }
    metadata_verified = all(
        {
            position: state.section_state(section)
            for position, section in viewer.layer(identifier).sections.items()
        }
        == expected
        for identifier, expected in sections.items()
    )
    if not metadata_verified:
        raise RuntimeError("Retimed native section metadata differs from the staged original keys")
    mismatches = []
    for reference, samples in values.items():
        for frame, expected in zip(targets, samples):
            if not _native_values_match(expected, data.get_data_value(reference, frame)):
                mismatches.append({"data_id": str(reference), "frame": frame})
    for reference, samples in settings.items():
        for frame, expected in zip(targets, samples):
            if expected != data.get_setting_value(reference, frame):
                mismatches.append({"setting_id": str(reference), "frame": frame})
    world_mismatches = []
    for obj, samples in global_poses.items():
        for frame, expected in zip(targets, samples):
            actual = _read(scene, obj, frame, "global")
            if "position" in expected and math.dist(expected["position"], actual["position"]) > 0.01:
                world_mismatches.append({"object": str(obj), "frame": frame, "field": "position"})
            if "rotation_wxyz" in expected:
                dot = abs(sum(a * b for a, b in zip(expected["rotation_wxyz"], actual["rotation_wxyz"])))
                if 2 * math.acos(min(1.0, dot)) > 0.001:
                    world_mismatches.append({"object": str(obj), "frame": frame, "field": "rotation"})
    mismatches.extend(world_mismatches)
    return {
        "execution_state": "completed",
        "source_frames": source,
        "target_frames": targets,
        "actual_keys": actual_keys,
        "key_mapping_verified": all(keys == targets for keys in actual_keys.values()),
        "channel_values_verified": not mismatches,
        "section_metadata_verified": metadata_verified,
        "world_poses_verified": not world_mismatches,
        "authored_channels": [
            "Transform.local_position",
            "Transform.local_rotation",
            "Transform.local_scale",
            "animated settings",
        ],
        "channel_mismatches": mismatches[:32],
        "channel_mismatch_count": len(mismatches),
        "preserved": ["key labels", "tangents", "velocity weights", "IK/FK", "fixation", "interpolation"],
        "note": "Temporal changes alter trajectories and contact timing. Read sampled poses and inspect the animation.",
    }


def _native_values_match(expected, actual):
    if hasattr(expected, "to_quaternion") or all(callable(getattr(expected, axis, None)) for axis in "wxyz"):
        dot = abs(sum(a * b for a, b in zip(_rotation(expected), _rotation(actual))))
        return dot >= 1 - 1e-7
    left, right = json_value(expected), json_value(actual)

    def equivalent(first, last):
        if isinstance(first, list) and isinstance(last, list):
            return len(first) == len(last) and all(equivalent(a, b) for a, b in zip(first, last))
        if type(first) in (int, float) and type(last) in (int, float):
            return math.isclose(first, last, rel_tol=1e-5, abs_tol=1e-5)
        return type(first) is type(last) and first == last

    return equivalent(left, right)


def _coerce_property_value(csc, template, value):
    if hasattr(template, "to_quaternion") or all(callable(getattr(template, axis, None)) for axis in "wxyz"):
        if type(value) is not dict or set(value) not in ({"rotation_wxyz"}, {"quaternion_wxyz"}):
            raise ValueError("Rotation values require {quaternion_wxyz: [w,x,y,z]}")
        components = normalized_quaternion(next(iter(value.values())))
        if hasattr(template, "to_quaternion"):
            return csc.math.Rotation.from_quaternion(*components)
        return csc.math.Quaternion(*components)
    if type(template) in (bool, int, float, str):
        if type(template) is float:
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e9:
                raise ValueError("Property requires a finite scalar number")
            return float(value)
        if type(value) is not type(template):
            raise ValueError(f"Property requires {type(template).__name__}")
        if type(value) is str and len(value) > 8192:
            raise ValueError("String property exceeds 8192 characters")
        if type(value) is int and not -(2**31) <= value < 2**31:
            raise ValueError("Integer property must fit a signed 32-bit native value")
        return value
    if hasattr(template, "shape") and hasattr(template, "dtype"):
        import numpy

        if not isinstance(value, list):
            raise ValueError("Array property requires nested JSON arrays")
        array = numpy.asarray(value)
        if array.shape != template.shape:
            if template.shape == (3, 1) and array.shape == (3,):
                array = array.reshape(3, 1)
            else:
                raise ValueError(f"Property array requires shape {tuple(template.shape)}")
        if array.size > 16 or array.dtype.kind not in "biuf" or not numpy.isfinite(array).all():
            raise ValueError("Array property requires at most 16 finite numeric or boolean components")
        if template.dtype.kind == "b" and array.dtype.kind != "b":
            raise ValueError("Boolean array property requires boolean components")
        if template.dtype.kind != "b" and array.dtype.kind == "b":
            raise ValueError("Numeric array property does not accept boolean components")
        if template.dtype.kind in "iu":
            bounds = numpy.iinfo(template.dtype)
            if (array < bounds.min).any() or (array > bounds.max).any():
                raise ValueError("Integer array property exceeds its native range")
            if array.dtype.kind == "f" and (array != numpy.floor(array)).any():
                raise ValueError("Integer array property requires whole numbers")
        with numpy.errstate(over="ignore", invalid="ignore"):
            converted = array.astype(template.dtype)
        if not numpy.isfinite(converted).all():
            raise ValueError("Property array overflows its native type")
        return converted
    raise ValueError(f"Unsupported native property type: {type(template).__name__}")


def object_properties_set(changes, frame=None, create_keys=False):
    from .data_channels import read_data_value

    csc, _, _, scene = current_context()
    position = _existing_frame(scene, frame)
    if type(changes) is not list or not 1 <= len(changes) <= 256:
        raise ValueError("Provide 1-256 property changes")
    behaviour_viewer, data, layers = scene.behaviour_viewer(), scene.data_viewer(), scene.layers_viewer()
    plan, seen = [], set()
    for change in changes:
        if type(change) is not dict or set(change) != {"object", "behavior", "property", "value"}:
            raise ValueError("Each change requires exactly object, behavior, property and value")
        identifier = resolve_object(scene, change["object"])
        behavior = behaviour_viewer.get_behaviour_by_name(identifier, change["behavior"])
        if behavior.is_null():
            raise ValueError(f"Object has no {change['behavior']!r} behavior")
        reference = behaviour_viewer.get_behaviour_data(behavior, change["property"])
        if reference.is_null():
            raise ValueError("Property is not a data channel; inspect object properties first")
        if reference in seen:
            raise ValueError("The same native property was supplied more than once")
        seen.add(reference)
        layer_id = layers.layer_id_by_obj_id_or_null(identifier)
        if layer_id.is_null() or layers.layer(layer_id).is_locked:
            raise ValueError("Property owner must belong to an unlocked animation track")
        mode = data.get_data(reference).mode
        if (
            mode == csc.model.DataMode.Animation
            and not create_keys
            and not layers.layer(layer_id).is_key(position)
        ):
            raise ValueError("Animated property writes require an existing key or create_keys=true")
        template = read_data_value(data, reference, position)
        native = _coerce_property_value(csc, template, change["value"])
        plan.append((reference, layer_id, mode, native))

    def apply(model, updater):
        editor, layer_editor = model.data_editor(), model.layers_editor()
        actuals = set()
        for reference, layer_id, mode, value in plan:
            if mode == csc.model.DataMode.Animation:
                if create_keys:
                    layer_editor.set_fixed_interpolation_or_key_if_need(layer_id, position, True)
                editor.set_data_value(reference, position, value)
            else:
                editor.set_data_value(reference, value)
            actuals.add(reference)
        updater.run_update(actuals, position)
        layer_editor.normalize_sections(scene)

    modify(scene, "Set object behavior properties", apply)
    return {
        "execution_state": "completed",
        "frame": position,
        "properties": [
            {
                "data_id": str(reference),
                "value": json_value(read_data_value(data, reference, position)),
                "scope": "all frames" if mode == csc.model.DataMode.Static else "frame",
            }
            for reference, _, mode, _ in plan
        ],
        "note": "Static channels change for all frames. Solvers and dependent properties can be affected.",
    }


def pose_read(objects, frame=None, space="global"):
    _, _, _, scene = current_context()
    position, space = _existing_frame(scene, frame), _space(space)
    return {
        "frame": position,
        "space": space,
        "poses": [_read(scene, identifier, position, space) for identifier in _objects(scene, objects)],
    }


def pose_read_sequence(objects, frames, space="global"):
    _, _, _, scene = current_context()
    identifiers, space = _objects(scene, objects), _space(space)
    if type(frames) is not list or not 1 <= len(frames) <= 128:
        raise ValueError("Provide 1-128 stored frames")
    positions = [_existing_frame(scene, frame) for frame in frames]
    if len(positions) != len(set(positions)):
        raise ValueError("frames must be unique")
    return {
        "space": space,
        "keyframes": [
            {"frame": frame, "poses": [_read(scene, identifier, frame, space) for identifier in identifiers]}
            for frame in positions
        ],
    }


def _prepare_sequence(scene, csc, keyframes, space, extend_timeline, create_keys):
    sequence = checked_sequence(keyframes, max_frames=128, max_objects=128)
    if sum(len(entry["poses"]) for entry in sequence) > 4096:
        raise ValueError("One edit is limited to 4096 object-frame samples; split the sequence")
    if sequence[-1]["frame"] >= _size(scene) and not extend_timeline:
        raise ValueError("Sequence extends the animation; set extend_timeline=true explicitly")
    if extend_timeline and not create_keys:
        raise ValueError("Extending the animation requires create_keys=true")
    data, layers = scene.data_viewer(), scene.layers_viewer()
    plan = []
    for entry in sequence:
        references, seen = [], set()
        for pose in entry["poses"]:
            identifier = resolve_object(scene, pose["object"])
            if identifier in seen:
                raise ValueError("Different identifiers resolved to the same object in a keyframe")
            seen.add(identifier)
            layer_id = layers.layer_id_by_obj_id_or_null(identifier)
            if layer_id.is_null() or layers.layer(layer_id).is_locked:
                raise ValueError(f"Object {pose['object']!r} needs an unlocked animation track")
            if not create_keys and not layers.layer(layer_id).is_key(entry["frame"]):
                raise ValueError("create_keys=false requires existing keys for every edited object")
            channels = []
            for field, suffix in (("position", "position"), ("rotation_wxyz", "rotation")):
                if field not in pose:
                    continue
                reference = _channel(scene, identifier, f"{space}_{suffix}")
                if data.get_data(reference).mode != csc.model.DataMode.Animation:
                    raise ValueError(f"{pose['object']!r} {field} is static; change the rig explicitly first")
                current = data.get_data_value(reference, min(entry["frame"], _size(scene) - 1))
                channels.append((reference, field, pose[field], hasattr(current, "to_quaternion")))
            references.append((identifier, layer_id, channels))
        plan.append((entry["frame"], references))
    return sequence, plan


def _verification_tolerances(position_tolerance, rotation_tolerance):
    for name, value in (
        ("position_tolerance", position_tolerance),
        ("rotation_tolerance", rotation_tolerance),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite, greater than zero and at most 1")
    return float(position_tolerance), float(rotation_tolerance)


def pose_apply_sequence(
    keyframes,
    space="global",
    create_keys=True,
    extend_timeline=False,
    verify=True,
    position_tolerance=0.001,
    rotation_tolerance=0.001,
):
    csc, _, _, scene = current_context()
    space = _space(space)
    position_tolerance, rotation_tolerance = _verification_tolerances(position_tolerance, rotation_tolerance)
    sequence, plan = _prepare_sequence(scene, csc, keyframes, space, extend_timeline, create_keys)

    def apply(model, updater):
        import numpy

        editor, layer_editor = model.data_editor(), model.layers_editor()
        if create_keys:
            all_layers = {layer_id for _, entries in plan for _, layer_id, _ in entries}
            _ensure_frame_storage(csc, scene, model, updater, all_layers, plan[-1][0])
            for frame, entries in plan:
                for _, layer_id, _ in entries:
                    with _native_step(f"insert pose key at frame {frame} on track {layer_id}"):
                        layer_editor.set_fixed_interpolation_or_key_if_need(layer_id, frame, True)
            with _native_step("fit animation after pose keys"):
                model.fit_animation_size_by_layers()
        for frame, entries in plan:
            changed = set()
            for _, _, channels in entries:
                for reference, field, value, is_rotation in channels:
                    if field == "position":
                        native_value = numpy.asarray(value, dtype=numpy.float32)
                    elif is_rotation:
                        native_value = csc.math.Rotation.from_quaternion(*value)
                    else:
                        native_value = csc.math.Quaternion(*value)
                    with _native_step(f"write {field} at frame {frame} on data {reference}"):
                        editor.set_data_value(reference, frame, native_value)
                    changed.add(reference)
            with _native_step(f"solve changed controllers at frame {frame}"):
                updater.run_update(changed, frame)
        with _native_step("normalize pose track sections"):
            layer_editor.normalize_sections(scene)

    modify(scene, "Apply controller pose sequence", apply)
    verification = {
        "requested": bool(verify),
        "matches": None,
        "position_tolerance": position_tolerance,
        "rotation_tolerance_radians": rotation_tolerance,
    }
    observed = []
    if verify:
        max_position, max_rotation = 0.0, 0.0
        for (frame, entries), requested in zip(plan, sequence):
            actual_poses = []
            for (identifier, _, _), expected in zip(entries, requested["poses"]):
                actual = _read(scene, identifier, frame, space)
                actual_poses.append(actual)
                if "position" in expected:
                    max_position = max(max_position, math.dist(actual["position"], expected["position"]))
                if "rotation_wxyz" in expected:
                    dot = abs(sum(a * b for a, b in zip(actual["rotation_wxyz"], expected["rotation_wxyz"])))
                    max_rotation = max(max_rotation, 2 * math.acos(min(1.0, dot)))
            observed.append({"frame": frame, "poses": actual_poses})
        verification.update(
            matches=max_position <= position_tolerance and max_rotation <= rotation_tolerance,
            max_position_error=max_position,
            max_rotation_error_radians=max_rotation,
        )
    return {
        "execution_state": "completed",
        "space": space,
        "frames": [frame for frame, _ in plan],
        "object_samples": sum(len(entries) for _, entries in plan),
        "verification": verification,
        "observed": observed,
        "note": "One native modification inside the caller's checked model transaction. Solvers may affect unedited controls.",
    }


def pose_apply(
    poses,
    frame=None,
    space="global",
    create_keys=True,
    verify=True,
    position_tolerance=0.001,
    rotation_tolerance=0.001,
):
    _, _, _, scene = current_context()
    frame = _existing_frame(scene, frame)
    return pose_apply_sequence(
        [{"frame": frame, "poses": poses}],
        space,
        create_keys,
        False,
        verify,
        position_tolerance=position_tolerance,
        rotation_tolerance=rotation_tolerance,
    )


def pose_offset(objects, frames, delta, space="global", create_keys=False):
    offset = finite_vector(delta, 3, name="delta")
    sequence = pose_read_sequence(objects, frames, space)["keyframes"]
    for entry in sequence:
        for pose in entry["poses"]:
            pose.pop("name", None)
            pose.pop("rotation_wxyz", None)
            if "position" not in pose:
                raise ValueError("Every selected object needs a readable position channel")
            pose["position"] = [value + shift for value, shift in zip(pose["position"], offset)]
    sequence.sort(key=lambda entry: entry["frame"])
    result = pose_apply_sequence(sequence, space, create_keys, False, True)
    result["note"] = (
        "Offset is relative and accumulates when repeated. Read back after a timeout before retrying."
    )
    return result


def pose_snapshot(objects, frame=None, space="global", path=None, overwrite=False):
    _, _, view, scene = current_context()
    if len(_snapshots) >= MAX_SNAPSHOTS:
        raise ValueError("Snapshot capacity reached; release an old snapshot before capturing another")
    pose = pose_read(objects, frame, space)
    snapshot = {
        "schema_version": 1,
        "scene_path": view.get_path_name(),
        "scene_name": view.name(),
        "topology_sha256": _signature(scene),
        **pose,
    }
    if path is not None:
        destination = workspace_path(path, overwrite=overwrite, suffixes=(".json",))
        encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2)
        with destination.open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(encoded)
    identifier = uuid.uuid4().hex
    _snapshots[identifier] = snapshot
    return {
        "snapshot_id": identifier,
        "snapshot": snapshot,
        "path": path,
        "scope": "One pose on selected objects; no track, interpolation or whole-scene backup.",
    }


def _validate_snapshot(snapshot):
    fields = {"schema_version", "scene_path", "scene_name", "topology_sha256", "frame", "space", "poses"}
    if (
        type(snapshot) is not dict
        or set(snapshot) != fields
        or type(snapshot.get("schema_version")) is not int
        or snapshot["schema_version"] != 1
    ):
        raise ValueError("Unsupported pose snapshot format")
    if any(type(snapshot.get(key)) is not str for key in ("scene_path", "scene_name", "topology_sha256")):
        raise ValueError("Invalid pose snapshot identity")
    _space(snapshot["space"])
    if type(snapshot["poses"]) is not list or any(type(pose) is not dict for pose in snapshot["poses"]):
        raise ValueError("Pose snapshot entries must be an array of objects")
    poses = [{key: value for key, value in pose.items() if key != "name"} for pose in snapshot["poses"]]
    checked_sequence([{"frame": snapshot["frame"], "poses": poses}])
    return poses


def pose_restore(
    snapshot_id=None, path=None, frame=None, verify=True, position_tolerance=0.001, rotation_tolerance=0.001
):
    _, _, view, scene = current_context()
    if (snapshot_id is None) == (path is None):
        raise ValueError("Provide exactly one of snapshot_id or path")
    if path is not None:
        source = workspace_path(path, must_exist=True, suffixes=(".json",))
        if source.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Pose snapshot exceeds 2 MiB")
        snapshot = json.loads(source.read_text(encoding="utf-8"))
    else:
        if snapshot_id not in _snapshots:
            raise ValueError("Snapshot not found in this Cascadeur session")
        snapshot = _snapshots[snapshot_id]
    poses = _validate_snapshot(snapshot)
    if snapshot["topology_sha256"] != _signature(scene):
        raise ValueError("Snapshot topology does not match the active scene")
    if snapshot["scene_path"] != view.get_path_name():
        raise ValueError("Snapshot belongs to a different saved scene path")
    target_frame = snapshot["frame"] if frame is None else frame
    return pose_apply(
        poses,
        target_frame,
        snapshot["space"],
        True,
        verify,
        position_tolerance=position_tolerance,
        rotation_tolerance=rotation_tolerance,
    )


def pose_release_snapshot(snapshot_id):
    if snapshot_id not in _snapshots:
        raise ValueError("Snapshot not found in this Cascadeur session")
    del _snapshots[snapshot_id]
    return {"released": snapshot_id, "remaining": len(_snapshots)}


def motion_sample(keyframes, step=1, easing="smoothstep"):
    return sample_motion(keyframes, step, easing)


def motion_mirror(keyframes, axis="x", object_map=None):
    return mirror_motion(keyframes, axis, object_map)


IDENTIFIER = {"type": "string", "minLength": 1, "maxLength": 512}
FRAME = {"type": "integer", "minimum": 0, "maximum": 100000}
OBJECTS = {"type": "array", "items": IDENTIFIER, "minItems": 1, "maxItems": 128, "uniqueItems": True}
FRAMES = {"type": "array", "items": FRAME, "minItems": 1, "maxItems": 256, "uniqueItems": True}
SPACE = {"type": "string", "enum": ["local", "global"], "default": "global"}
VECTOR = {
    "type": "array",
    "items": {"type": "number", "minimum": -1e9, "maximum": 1e9},
    "minItems": 3,
    "maxItems": 3,
}
QUATERNION = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
POSE = {
    "type": "object",
    "properties": {"object": IDENTIFIER, "position": VECTOR, "rotation_wxyz": QUATERNION},
    "required": ["object"],
    "additionalProperties": False,
}
POSES = {"type": "array", "items": POSE, "minItems": 1, "maxItems": 128}
KEYFRAMES = {
    "type": "array",
    "minItems": 1,
    "maxItems": 128,
    "items": {
        "type": "object",
        "properties": {"frame": FRAME, "poses": POSES},
        "required": ["frame", "poses"],
        "additionalProperties": False,
    },
}
BOOLEAN = {"type": "boolean"}
PATH = {"type": "string", "minLength": 1, "maxLength": 4096}
POSITION_TOLERANCE = {
    "type": "number",
    "exclusiveMinimum": 0,
    "maximum": 1,
    "default": 0.001,
    "description": "Accepted native position readback error in scene units; does not alter writes.",
}
ROTATION_TOLERANCE = {
    "type": "number",
    "exclusiveMinimum": 0,
    "maximum": 1,
    "default": 0.001,
    "description": "Accepted native quaternion angular error in radians; does not alter writes.",
}

TOOLS = [
    Tool(
        "timeline_get",
        "Read the current frame, stored animation size and playback/visible boundaries.",
        timeline_get,
    ),
    Tool(
        "timeline_set_frame",
        "Move to an existing animation frame.",
        timeline_set_frame,
        {"frame": FRAME},
        ("frame",),
        False,
    ),
    Tool(
        "timeline_set_range",
        "Set playback boundaries within stored animation; optionally update visible boundaries.",
        timeline_set_range,
        {"first": FRAME, "last": FRAME, "visible": BOOLEAN},
        ("first", "last"),
        False,
    ),
    Tool(
        "tracks_list",
        "Inspect native tracks, object membership, keys and interpolation sections.",
        tracks_list,
        {"include_sections": BOOLEAN},
    ),
    Tool(
        "tracks_set_keys",
        "Insert keys on explicit unlocked tracks, or delete existing nonzero keys including the final key while retaining frame 0. Extending animation requires extend_timeline=true.",
        tracks_set_keys,
        {
            "tracks": OBJECTS,
            "frames": FRAMES,
            "mode": {"type": "string", "enum": ["insert", "delete"]},
            "extend_timeline": BOOLEAN,
        },
        ("tracks", "frames"),
        False,
    ),
    Tool(
        "tracks_set_interpolation",
        "Change interpolation at existing track keys; preserves each section's key metadata. Inspect neighboring motion afterward.",
        tracks_set_interpolation,
        {
            "tracks": OBJECTS,
            "frames": FRAMES,
            "interpolation": {"type": "string", "enum": list(INTERPOLATIONS)},
        },
        ("tracks", "frames", "interpolation"),
        False,
    ),
    Tool(
        "tracks_retime",
        "Retime standalone Basic/Transform tracks while preserving native key/section metadata and verifying moved data/settings. Requires unparented objects, all track keys, fixed endpoints, and no baked/AI intervals or cycles. Constrained rigs are unsupported.",
        tracks_retime,
        {"tracks": OBJECTS, "source_frames": FRAMES, "target_frames": FRAMES},
        ("tracks", "source_frames", "target_frames"),
        False,
    ),
    Tool(
        "object_properties_set",
        "Set exact behavior data properties discovered by object_inspect. Prevalidates types and unlocked tracks. Static channels affect all frames; animated writes need keys.",
        object_properties_set,
        {
            "changes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 256,
                "items": {
                    "type": "object",
                    "properties": {
                        "object": IDENTIFIER,
                        "behavior": IDENTIFIER,
                        "property": IDENTIFIER,
                        "value": {},
                    },
                    "required": ["object", "behavior", "property", "value"],
                    "additionalProperties": False,
                },
            },
            "frame": FRAME,
            "create_keys": BOOLEAN,
        },
        ("changes",),
        False,
    ),
    Tool(
        "pose_read",
        "Read actual position and quaternion channels for exact object IDs or unique names; works with arbitrary Transform rigs.",
        pose_read,
        {"objects": OBJECTS, "frame": FRAME, "space": SPACE},
        ("objects",),
    ),
    Tool(
        "pose_read_sequence",
        "Read native controller poses at up to 128 stored frames without changing the current frame.",
        pose_read_sequence,
        {"objects": OBJECTS, "frames": {**FRAMES, "maxItems": 128}, "space": SPACE},
        ("objects", "frames"),
    ),
    Tool(
        "pose_apply",
        "Write selected animated Transform channels in one native modification, then report observed values and target errors. Solvers may move other controls.",
        pose_apply,
        {
            "poses": POSES,
            "frame": FRAME,
            "space": SPACE,
            "create_keys": BOOLEAN,
            "verify": BOOLEAN,
            "position_tolerance": POSITION_TOLERANCE,
            "rotation_tolerance": ROTATION_TOLERANCE,
        },
        ("poses",),
        False,
    ),
    Tool(
        "pose_apply_sequence",
        "Prevalidate and apply up to 128 increasing pose frames (4096 object samples). Adds keys only when requested; timeline extension is explicit. Readback reports solver adjustments.",
        pose_apply_sequence,
        {
            "keyframes": KEYFRAMES,
            "space": SPACE,
            "create_keys": BOOLEAN,
            "extend_timeline": BOOLEAN,
            "verify": BOOLEAN,
            "position_tolerance": POSITION_TOLERANCE,
            "rotation_tolerance": ROTATION_TOLERANCE,
        },
        ("keyframes",),
        False,
    ),
    Tool(
        "pose_offset",
        "Offset selected positions across existing frames from their actual values. Repeated calls accumulate; inspect before retrying after timeouts.",
        pose_offset,
        {
            "objects": OBJECTS,
            "frames": {**FRAMES, "maxItems": 128},
            "delta": VECTOR,
            "space": SPACE,
            "create_keys": BOOLEAN,
        },
        ("objects", "frames", "delta"),
        False,
    ),
    Tool(
        "pose_snapshot",
        "Capture a selected-object pose in this host session, optionally to workspace JSON. This does not preserve whole-scene animation or curves.",
        pose_snapshot,
        {"objects": OBJECTS, "frame": FRAME, "space": SPACE, "path": PATH, "overwrite": BOOLEAN},
        ("objects",),
        False,
    ),
    Tool(
        "pose_restore",
        "Restore a pose snapshot after checking saved scene path and topology. Native interpolation and solvers can affect neighboring frames.",
        pose_restore,
        {
            "snapshot_id": IDENTIFIER,
            "path": PATH,
            "frame": FRAME,
            "verify": BOOLEAN,
            "position_tolerance": POSITION_TOLERANCE,
            "rotation_tolerance": ROTATION_TOLERANCE,
        },
        (),
        False,
    ),
    Tool(
        "pose_release_snapshot",
        "Release one session pose snapshot without editing the scene.",
        pose_release_snapshot,
        {"snapshot_id": IDENTIFIER},
        ("snapshot_id",),
        False,
    ),
    Tool(
        "motion_sample",
        "Build a pose sequence using easing and shortest-arc quaternion SLERP. Returns a plan without native edits; feed suitable batches to pose_apply_sequence.",
        motion_sample,
        {
            "keyframes": KEYFRAMES,
            "step": {"type": "integer", "minimum": 1, "maximum": 10000},
            "easing": {"type": "string", "enum": ["linear", "smoothstep", "smootherstep", "hold"]},
        },
        ("keyframes",),
    ),
    Tool(
        "motion_mirror",
        "Reflect pose position/rotation data across an axis plane with optional exact object-ID mapping. Returns a plan without changing the native scene.",
        motion_mirror,
        {
            "keyframes": KEYFRAMES,
            "axis": {"type": "string", "enum": ["x", "y", "z"]},
            "object_map": {"type": "object", "additionalProperties": IDENTIFIER, "maxProperties": 128},
        },
        ("keyframes",),
    ),
]
