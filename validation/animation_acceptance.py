"""Exercise character curves and durable checkpoints on a disposable Cascy copy."""

import json
import shutil
import time

from native_advanced import CLIENT, CONFIG


def main():
    directory = CONFIG.workspace / "validation" / f"advanced-{time.time_ns()}"
    directory.mkdir(parents=True)
    results = []

    def run(name, **arguments):
        started = time.monotonic()
        receipt = CLIENT.call(name, arguments)
        results.append({"tool": name, "arguments": arguments, **receipt})
        (directory / "receipts.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(name, round(time.monotonic() - started, 2), json.dumps(receipt["result"])[:2200], flush=True)
        return receipt["result"]

    current = run("scene_info")
    fixture = directory / "curve-fixture.casc"
    shutil.copy2(current["path"], fixture)
    run("scene_open", path=str(fixture))
    character = run("characters_list")["characters"][0]["id"]
    run("character_motion_inspect", character_id=character)
    checkpoint = run("checkpoint_create", label="Cascy before curve edit")
    run(
        "character_offset_preserving_curves",
        character_id=character,
        frames=[12],
        offsets={"left_hand": [0, 0.1, 0]},
        position_tolerance=0.01,
    )
    run("checkpoint_status", checkpoint_id=checkpoint["checkpoint_id"])
    run("checkpoint_restore", checkpoint_id=checkpoint["checkpoint_id"])
    restored = run("checkpoint_status", checkpoint_id=checkpoint["checkpoint_id"])
    assert restored["comparison"]["equivalent"]
    print("ARTIFACT", directory, flush=True)


if __name__ == "__main__":
    main()
