"""Native timeline selection, motion measurements, mirroring and mixed-space authoring."""

from __future__ import annotations

import math

from ..contracts import Tool
from . import animation, common


def tracks_select_frames(tracks, first, last):
    _, _, _, scene = common.current_context()
    identifiers = animation._tracks(scene, tracks)
    first, last = animation._existing_frame(scene, first), animation._existing_frame(scene, last)
    if first > last:
        raise ValueError("first must not exceed last")
    selector = scene.get_layers_selector()
    selector.set_full_selection_by_parts(identifiers, first, last)
    selected = selector.selection()
    interval = selected.frames_interval()
    selected_ids = [str(item) for item in selected.item_ids()]
    layers = scene.layers_viewer()
    leaf_ids = {str(item) for item in layers.all_layer_ids()}
    group_ids, ancestors = set(), set()
    if callable(getattr(layers, "folders_map", None)):
        group_ids.update(str(item) for item in layers.folders_map())
    if callable(getattr(layers, "root_id", None)):
        root_id = str(layers.root_id())
        group_ids.add(root_id)
        ancestors.add(root_id)
    if callable(getattr(layers, "all_parent_ids", None)):
        for identifier in identifiers:
            parents = {str(item) for item in layers.all_parent_ids(identifier)}
            group_ids.update(parents)
            ancestors.update(parents)
    selected_tracks = [identifier for identifier in selected_ids if identifier in leaf_ids]
    expanded_groups = [identifier for identifier in selected_ids if identifier in group_ids]
    unknown_items = sorted(set(selected_ids) - leaf_ids - group_ids)
    unrelated_groups = sorted(set(expanded_groups) - ancestors)
    return {
        "tracks": selected_tracks,
        "all_selection_items": selected_ids,
        "expanded_groups": expanded_groups,
        "unknown_selection_items": unknown_items,
        "first": interval.first,
        "last": interval.last,
        "verification": {
            "matches": set(selected_tracks) == {str(item) for item in identifiers}
            and not unknown_items
            and not unrelated_groups
            and interval.first == first
            and interval.last == last,
            "unrelated_groups": unrelated_groups,
        },
    }


def motion_measure(objects, first, last, step=1):
    _, _, _, scene = common.current_context()
    identifiers = animation._objects(scene, objects)
    first, last = animation._existing_frame(scene, first), animation._existing_frame(scene, last)
    if first > last or type(step) is not int or not 1 <= step <= 10000:
        raise ValueError("Use an increasing frame range and a positive integer step")
    frames = list(range(first, last + 1, step))
    if len(frames) > 2001 or len(frames) * len(identifiers) > 64000:
        raise ValueError("Sampling exceeds 2001 frames or 64000 object samples")
    rows = []
    model = scene.model_viewer()
    for identifier in identifiers:
        positions = [animation._read(scene, identifier, frame, "global")["position"] for frame in frames]
        distances = [math.dist(left, right) for left, right in zip(positions, positions[1:])]
        rows.append(
            {
                "object": str(identifier),
                "name": model.get_object_name(identifier),
                "positions": positions,
                "path_length": sum(distances),
                "max_step_distance": max(distances, default=0.0),
            }
        )
    return {
        "frames": frames,
        "objects": rows,
        "scope": "Actual native positions in scene units; distances do not establish contact or motion quality.",
    }


def native_mirror(objects, interval=False):
    from . import recovery

    if type(interval) is not bool:
        raise ValueError("interval must be a boolean")
    _, app, view, scene = common.current_context()
    identifiers = set(animation._objects(scene, objects))
    core = app.get_tools_manager().get_tool("MirrorTool").editor(view).core()
    recovery.note_native_attempt(scene)
    if interval:
        core.mirror_interval(identifiers)
    else:
        core.mirror_frame(identifiers)
    recovery.note_native_commit(scene)
    frame = scene.get_current_frame()
    return {
        "mirrored_objects": sorted(str(item) for item in identifiers),
        "interval": interval,
        "observed": [animation._read(scene, identifier, frame, "global") for identifier in identifiers],
        "verification": "Native symmetry rules were invoked; inspect paired controls and rendered motion.",
    }


def _rotation(csc, value):
    return csc.math.Rotation.from_quaternion(*animation._rotation(value))


def _transform_rows(csc, scene, keyframes):
    from .data_channels import read_data_value

    if type(keyframes) is not list or not 1 <= len(keyframes) <= 500:
        raise ValueError("Provide 1-500 keyframes")
    data, layers = scene.data_viewer(), scene.layers_viewer()
    prepared, seen = [], set()
    for key in keyframes:
        if type(key) is not dict or set(key) != {"frame", "transforms"}:
            raise ValueError("Each keyframe needs exactly frame and transforms")
        frame = animation.checked_frame(key["frame"])
        if type(key["transforms"]) is not list or not 1 <= len(key["transforms"]) <= 200:
            raise ValueError("Each keyframe needs 1-200 transforms")
        for change in key["transforms"]:
            if type(change) is not dict or set(change) - {
                "object",
                "space",
                "position",
                "rotation_wxyz",
                "rotation_delta_degrees",
                "reference_frame",
            }:
                raise ValueError("Unknown transform fields")
            identifier = common.resolve_object(scene, change.get("object"))
            if (frame, identifier) in seen:
                raise ValueError("Each object can occur only once at a given frame")
            seen.add((frame, identifier))
            space = animation._space(change.get("space", "local"))
            track = layers.layer_id_by_obj_id_or_null(identifier)
            if track.is_null() or layers.layer(track).is_locked:
                raise ValueError("Each edited object requires an unlocked track")
            if "rotation_wxyz" in change and "rotation_delta_degrees" in change:
                raise ValueError("Choose an absolute quaternion or a rotation delta")
            values, expected = [], {"object": str(identifier), "space": space, "frame": frame}
            if "position" in change:
                position = animation.finite_vector(change["position"], 3)
                reference = animation._channel(scene, identifier, space + "_position")
                values.append((reference, position, "position"))
                expected["position"] = position
            rotation = None
            if "rotation_wxyz" in change:
                rotation = csc.math.Rotation.from_quaternion(
                    *animation.normalized_quaternion(change["rotation_wxyz"])
                )
            elif "rotation_delta_degrees" in change:
                reference_frame = animation._existing_frame(scene, change.get("reference_frame", 0))
                angles = animation.finite_vector(change["rotation_delta_degrees"], 3)
                reference = animation._channel(scene, identifier, space + "_rotation")
                base = _rotation(csc, read_data_value(data, reference, reference_frame))
                delta = csc.math.Rotation.from_euler(*[math.radians(angle) for angle in angles])
                rotation = base * delta
            if rotation is not None:
                reference = animation._channel(scene, identifier, space + "_rotation")
                expected["rotation_wxyz"] = animation._rotation(rotation)
                template = read_data_value(data, reference, min(frame, animation._size(scene) - 1))
                if not hasattr(template, "to_quaternion"):
                    rotation = csc.math.Quaternion(*expected["rotation_wxyz"])
                values.append((reference, rotation, "rotation"))
            if not values:
                raise ValueError("Each transform requires a position or rotation")
            if any(
                data.get_data(reference).mode != csc.model.DataMode.Animation for reference, _, _ in values
            ):
                raise ValueError(
                    "Transform authoring requires animated channels; edit static rig values explicitly"
                )
            prepared.append((frame, identifier, track, values, expected))
    if len(prepared) > 4096:
        raise ValueError("One authoring batch supports at most 4096 object-frame samples")
    return prepared


def transforms_animate(
    keyframes,
    interpolation="BEZIER",
    extend_timeline=False,
    position_tolerance=0.01,
    rotation_tolerance=0.001,
):
    csc, _, _, scene = common.current_context()
    position_tolerance, rotation_tolerance = animation._verification_tolerances(
        position_tolerance, rotation_tolerance
    )
    if interpolation not in animation.INTERPOLATIONS:
        raise ValueError("Unknown interpolation mode")
    prepared = _transform_rows(csc, scene, keyframes)
    last = max(row[0] for row in prepared)
    if last == animation._size(scene) - 1:
        raise ValueError(
            "The terminal stored frame has no following solver interval; allocate a successor before editing it"
        )
    if last >= animation._size(scene) and not extend_timeline:
        raise ValueError("Set extend_timeline=true to author new stored frames")
    mode = getattr(csc.layers.layer.Interpolation, interpolation)

    def edit(model, updater):
        import numpy

        layer_editor, editor = model.layers_editor(), model.data_editor()
        animation._ensure_frame_storage(csc, scene, model, updater, [row[2] for row in prepared], last)
        for frame, track in sorted({(row[0], row[2]) for row in prepared}, key=lambda pair: pair[0]):
            layer_editor.set_fixed_interpolation_or_key_if_need(track, frame, True)

            def change(section):
                section.interval.interpolation = mode

            layer_editor.change_section(frame, track, change)
        model.fit_animation_size_by_layers()
        updater.generate_update()
        for frame in sorted({row[0] for row in prepared}):
            actuals = set()
            for row_frame, _, _, values, _ in prepared:
                if row_frame != frame:
                    continue
                for reference, value, kind in values:
                    native_value = numpy.asarray(value, dtype=numpy.float32) if kind == "position" else value
                    editor.set_data_value(reference, frame, native_value)
                    actuals.add(reference)
            updater.run_update(actuals, frame)
        layer_editor.normalize_sections(scene)
        updater.generate_update()

    common.modify(scene, "Author mixed-space transform keys", edit)
    observed, position_error, rotation_error = [], 0.0, 0.0
    for frame, identifier, _, _, expected in prepared:
        actual = animation._read(scene, identifier, frame, expected["space"])
        observed.append({"frame": frame, "space": expected["space"], **actual})
        if "position" in expected:
            position_error = max(position_error, math.dist(actual["position"], expected["position"]))
        if "rotation_wxyz" in expected:
            dot = abs(sum(a * b for a, b in zip(actual["rotation_wxyz"], expected["rotation_wxyz"])))
            rotation_error = max(rotation_error, 2 * math.acos(min(1.0, dot)))
    return {
        "observed": observed,
        "frames": sorted({row[0] for row in prepared}),
        "verification": {
            "matches": position_error <= position_tolerance and rotation_error <= rotation_tolerance,
            "max_position_error": position_error,
            "max_rotation_error_radians": rotation_error,
            "position_tolerance": position_tolerance,
            "rotation_tolerance_radians": rotation_tolerance,
        },
    }


def pose_read_euler(objects, frame=None, space="local"):
    from .data_channels import read_data_value

    csc, _, _, scene = common.current_context()
    frame, space = animation._existing_frame(scene, frame), animation._space(space)
    rows = []
    for identifier in animation._objects(scene, objects):
        row = animation._read(scene, identifier, frame, space)
        reference = animation._channel(scene, identifier, space + "_rotation")
        native_rotation = read_data_value(scene.data_viewer(), reference, frame)
        rotation = (
            native_rotation
            if callable(getattr(native_rotation, "to_euler_angles", None))
            else _rotation(csc, native_rotation)
        )
        euler = common.json_value(rotation.to_euler_angles())
        if (
            isinstance(euler, list)
            and len(euler) == 3
            and all(isinstance(item, list) and len(item) == 1 for item in euler)
        ):
            euler = [item[0] for item in euler]
        row["rotation_euler_radians"] = animation.finite_vector(euler, 3)
        rows.append(row)
    return {"frame": frame, "space": space, "poses": rows}


TRANSFORM = {
    "type": "object",
    "required": ["object"],
    "additionalProperties": False,
    "properties": {
        "object": animation.IDENTIFIER,
        "space": {**animation.SPACE, "default": "local"},
        "position": animation.VECTOR,
        "rotation_wxyz": animation.QUATERNION,
        "rotation_delta_degrees": animation.VECTOR,
        "reference_frame": {**animation.FRAME, "default": 0},
    },
}
TOOLS = [
    Tool(
        "tracks_select_frames",
        "Set real native track/frame selection for interval edits, mirroring or physics.",
        tracks_select_frames,
        {"tracks": animation.OBJECTS, "first": animation.FRAME, "last": animation.FRAME},
        ("tracks", "first", "last"),
        False,
    ),
    Tool(
        "motion_measure",
        "Sample actual native trajectories, path length and largest sample-to-sample distance.",
        motion_measure,
        {
            "objects": animation.OBJECTS,
            "first": animation.FRAME,
            "last": animation.FRAME,
            "step": {"type": "integer", "minimum": 1, "maximum": 10000},
        },
        ("objects", "first", "last"),
    ),
    Tool(
        "native_mirror",
        "Invoke native rig-aware mirroring on explicit objects at this frame or selected interval.",
        native_mirror,
        {"objects": animation.OBJECTS, "interval": {"type": "boolean"}},
        ("objects",),
        False,
    ),
    Tool(
        "transforms_animate",
        "Author positions and absolute or reference-frame Euler-delta rotations, with per-object spaces and native interpolation.",
        transforms_animate,
        {
            "keyframes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 500,
                "items": {
                    "type": "object",
                    "properties": {
                        "frame": animation.FRAME,
                        "transforms": {"type": "array", "minItems": 1, "maxItems": 200, "items": TRANSFORM},
                    },
                    "required": ["frame", "transforms"],
                    "additionalProperties": False,
                },
            },
            "interpolation": {"type": "string", "enum": list(animation.INTERPOLATIONS)},
            "extend_timeline": {"type": "boolean"},
            "position_tolerance": animation.POSITION_TOLERANCE,
            "rotation_tolerance": animation.ROTATION_TOLERANCE,
        },
        ("keyframes",),
        False,
    ),
    Tool(
        "pose_read_euler",
        "Read actual native positions, quaternions and Euler angles in radians.",
        pose_read_euler,
        {"objects": animation.OBJECTS, "frame": animation.FRAME, "space": animation.SPACE},
        ("objects",),
    ),
]
