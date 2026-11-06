"""Read native Euler turns, author a delta and import the verified FBX output."""

import json
import time

from native_advanced import CLIENT, CONFIG


def main():
    directory = CONFIG.workspace / "validation" / f"fbx-import-{time.time_ns()}"
    directory.mkdir()
    records = []

    def run(name, **arguments):
        receipt = CLIENT.call(name, arguments)
        records.append({"tool": name, "arguments": arguments, **receipt})
        (directory / "receipts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(name, receipt.get("elapsed_ms"), json.dumps(receipt["result"])[-1800:], flush=True)
        return receipt["result"]

    euler = run("pose_read_euler", objects=["MCP_FK_Root"], frame=5, space="local")
    assert euler["poses"][0]["rotation_euler_radians"][2] > 6.28
    run(
        "transforms_animate",
        keyframes=[
            {
                "frame": 0,
                "transforms": [
                    {
                        "object": "MCP_Unrelated_Joint",
                        "space": "local",
                        "rotation_delta_degrees": [0, 0, 15],
                        "reference_frame": 0,
                    }
                ],
            }
        ],
        interpolation="LINEAR",
    )
    exported = max(
        (CONFIG.workspace / "validation").glob("fbx-verified-*/character.fbx"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    run("scene_new")
    run("scene_save", path=str(directory / "before-import.casc"))
    run("fbx_import", path=str(exported), mode="scene")
    joints = run("objects_list", object_type="Joint", limit=200)["objects"]
    assert len(joints) >= 66, joints
    run("scene_save", path=str(directory / "imported.casc"))
    channel = run("object_data_read", object=joints[0]["id"], query="Use custom color")["channels"][0]
    assert channel["mode"] == "Static"
    run(
        "object_data_set",
        changes=[{"object": joints[0]["id"], "data": channel["id"], "value": not channel["value"]}],
    )
    run(
        "object_data_set",
        changes=[{"object": joints[0]["id"], "data": channel["id"], "value": channel["value"]}],
    )
    print("ARTIFACT", directory, flush=True)


if __name__ == "__main__":
    main()
