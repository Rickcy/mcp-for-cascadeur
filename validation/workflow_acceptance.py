"""Verify native mirror/review and generic mixed-space animation authoring."""

import json
import shutil
import time

from native_advanced import CLIENT, CONFIG


def main(cube_only=False):
    directory = CONFIG.workspace / "validation" / f"workflows-{time.time_ns()}"
    directory.mkdir()
    records = []

    def run(name, **arguments):
        value = CLIENT.call(name, arguments)
        records.append({"tool": name, "arguments": arguments, **value})
        (directory / "receipts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(name, value.get("elapsed_ms"), json.dumps(value["result"])[-1200:], flush=True)
        return value["result"]

    if not cube_only:
        run("motion_measure", objects=["hand_MainPoint_l"], first=0, last=24, step=4)
        run("camera_frame", objects=["pelvis", "head", "foot_MainPoint_l"])
        run("viewport_capture", path=str(directory / "native-review.png"))
        character = run("character_pose", frame=12)
        checkpoint = run("checkpoint_create", label="Before native character mirroring")
        run("timeline_set_frame", frame=12)
        run("native_mirror", objects=list(character["pose"]))
        run("checkpoint_restore", checkpoint_id=checkpoint["checkpoint_id"])
        assert run("checkpoint_status", checkpoint_id=checkpoint["checkpoint_id"])["comparison"]["equivalent"]
    fixture = directory / "cube.casc"
    shutil.copy2(CONFIG.cascadeur_home / "samples" / "Cube.casc", fixture)
    run("scene_open", path=str(fixture))
    transforms = run("objects_list", object_type="Point")["objects"]
    assert transforms, transforms
    identifier = next(item["id"] for item in transforms if item["name"] == "joint1_MainPoint")
    baseline = run("pose_read", objects=[item["id"] for item in transforms], frame=0)["poses"]
    run(
        "transforms_animate",
        keyframes=[
            {
                "frame": frame,
                "transforms": [
                    {
                        "object": item["object"],
                        "space": "global",
                        "position": [item["position"][0] + frame, *item["position"][1:]],
                    }
                    for item in baseline
                ],
            }
            for frame in [0, 12, 24]
        ],
        interpolation="LINEAR",
        extend_timeline=True,
    )
    run("pose_read_euler", objects=["joint1"], frame=12, space="global")
    tracks = run("tracks_list")["tracks"]
    track = next(item["id"] for item in tracks if any(obj["id"] == identifier for obj in item["objects"]))
    selection = run("tracks_select_frames", tracks=[track], first=0, last=24)
    assert selection["verification"]["matches"]
    run("motion_measure", objects=[identifier], first=0, last=24, step=4)
    channels = run("object_data_read", object=identifier, frame=12)["channels"]
    position = next(item for item in channels if item["name"] == "Position")
    observed = position["value"]
    observed = [float(value[0]) if isinstance(value, list) else float(value) for value in observed]
    run(
        "object_data_set",
        changes=[{"object": identifier, "data": position["id"], "value": [observed[0] + 1, *observed[1:]]}],
        frame=12,
    )
    run("tracks_set_keys", tracks=[track], frames=[12, 24, 25], mode="delete")
    assert run("tracks_list")["tracks"]
    print("ARTIFACT", directory, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-only", action="store_true")
    main(cube_only=parser.parse_args().cube_only)
