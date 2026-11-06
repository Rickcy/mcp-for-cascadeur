"""Check semantic patches, authored fingers, durable poses and native playback."""

import json
import math
import time

from native_advanced import CLIENT, CONFIG


def main():
    directory = CONFIG.workspace / "validation" / f"character-{time.time_ns()}"
    directory.mkdir()
    records = []

    def run(name, **arguments):
        value = CLIENT.call(name, arguments)
        records.append({"tool": name, "arguments": arguments, **value})
        (directory / "receipts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(name, value.get("elapsed_ms"), json.dumps(value["result"])[-1400:], flush=True)
        return value["result"]

    snapshot = str(directory / "body-pose.json")
    run("character_pose_snapshot", path=snapshot, frame=12)
    hand = run("hand_pose_read", frame=12, side="left")["hands"]["left"]["index2"]
    w, x, y, z = hand
    c, s = math.cos(math.radians(3) / 2), math.sin(math.radians(3) / 2)
    rotated = [w * c - z * s, x * c + y * s, y * c - x * s, z * c + w * s]
    fingers = run(
        "hand_pose_apply_sequence", keyframes=[{"frame": 12, "hands": {"left": {"index2": rotated}}}]
    )
    assert fingers["body_preserved"]
    restored = run("character_pose_restore", path=snapshot)
    assert restored["authored_finger_channels_restored"] == 30 and restored["finger_euler_fidelity"]
    actual = run("hand_pose_read", frame=12, side="left")["hands"]["left"]["index2"]
    assert abs(sum(a * b for a, b in zip(hand, actual))) > 1 - 1e-8
    ready = run(
        "character_edit_readiness", frames=[12], operation="semantic_pose_offset", roles=["left_hand"]
    )
    assert ready["ready"], ready
    run("semantic_pose_offset", frames=[12], offsets={"left_hand": [0, 0.1, 0]})
    run("character_pose_restore", path=snapshot)
    run("playback_start", first=0, last=24, timeout=1.5)
    for _ in range(6):
        playback = run("playback_status")
        if playback["phase"] == "stopped":
            break
        time.sleep(0.3)
    assert playback["stop_verified"] and playback["continuous_playback_observed"], playback
    run("motion_measure", objects=["hand_MainPoint_l"], first=0, last=24, step=4)
    points = run("character_pose", frame=24)["pose"]
    run("camera_frame", objects=list(points), distance_scale=4)
    run("viewport_capture", path=str(directory / "native-review.png"))
    print("ARTIFACT", directory, flush=True)


if __name__ == "__main__":
    main()
