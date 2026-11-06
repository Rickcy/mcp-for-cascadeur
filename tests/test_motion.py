import math

import pytest

from mcp_for_cascadeur.host.motion import checked_sequence, mirror_motion, quaternion_slerp, sample_motion


def key(frame, position, quaternion=None, identifier="hand"):
    pose = {"object": identifier, "position": position}
    if quaternion is not None:
        pose["rotation_wxyz"] = quaternion
    return {"frame": frame, "poses": [pose]}


def test_slerp_takes_short_arc_and_preserves_rotation_for_antipodes():
    assert quaternion_slerp([1, 0, 0, 0], [-1, 0, 0, 0], 0.5) == [1, 0, 0, 0]
    middle = quaternion_slerp([1, 0, 0, 0], [0, 0, 1, 0], 0.5)
    assert middle == pytest.approx([math.sqrt(0.5), 0, math.sqrt(0.5), 0])
    assert sum(value * value for value in middle) == pytest.approx(1)


@pytest.mark.parametrize("value", [[0, 0, 0, 0], [1, float("nan"), 0, 0], [True, 0, 0, 0]])
def test_slerp_rejects_invalid_quaternions(value):
    with pytest.raises(ValueError):
        quaternion_slerp(value, [1, 0, 0, 0], 0.5)


def test_sampling_retains_authored_keys_even_when_step_does_not_divide_interval():
    plan = sample_motion([key(0, [0, 0, 0]), key(3, [3, 0, 0]), key(10, [10, 0, 0])], step=4, easing="linear")
    assert [entry["frame"] for entry in plan["keyframes"]] == [0, 3, 4, 8, 10]
    assert plan["keyframes"][2]["poses"][0]["position"] == pytest.approx([4, 0, 0])
    assert plan["native_scene_changed"] is False


def test_sampling_rejects_inconsistent_channels_instead_of_dropping_data():
    with pytest.raises(ValueError, match="consistent"):
        sample_motion([key(0, [0, 0, 0], [1, 0, 0, 0]), key(5, [3, 0, 0])])


def test_sequence_requires_distinct_objects_and_monotonic_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        checked_sequence([key(3, [0, 0, 0]), key(2, [1, 0, 0])])
    with pytest.raises(ValueError, match="only once"):
        checked_sequence([{"frame": 0, "poses": [key(0, [0, 0, 0])["poses"][0]] * 2}])


def test_mirror_is_an_involution_and_does_not_modify_source_data():
    source = [key(0, [1, 2, 3], [0.5, 0.5, 0.5, 0.5], "left_hand")]
    mirrored = mirror_motion(source, "x", {"left_hand": "right_hand"})
    pose = mirrored["keyframes"][0]["poses"][0]
    assert pose == {"object": "right_hand", "position": [-1, 2, 3], "rotation_wxyz": [0.5, 0.5, -0.5, -0.5]}
    restored = mirror_motion(mirrored["keyframes"], "x", {"right_hand": "left_hand"})
    assert restored["keyframes"] == source
    assert source[0]["poses"][0]["position"] == [1, 2, 3]


def test_mirror_rejects_identifier_collisions():
    with pytest.raises(ValueError, match="same target"):
        mirror_motion(
            [
                {
                    "frame": 0,
                    "poses": [{"object": "a", "position": [0, 0, 0]}, {"object": "b", "position": [1, 0, 0]}],
                }
            ],
            object_map={"a": "b"},
        )


def test_sparse_object_spans_do_not_emit_unusable_empty_pose_frames():
    plan = sample_motion([key(0, [0, 0, 0], identifier="a"), key(10, [1, 0, 0], identifier="b")])
    assert [entry["frame"] for entry in plan["keyframes"]] == [0, 10]
    checked_sequence(plan["keyframes"])
