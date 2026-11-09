"""Check sampled semantic motion and stationary-foot retiming on current Cascy."""

import json
import time

from native_advanced import CLIENT, CONFIG, probe


def main():
    records = []
    destination = CONFIG.workspace / "validation" / f"contacts-native-{time.time_ns()}.json"

    def run(name, **arguments):
        receipt = CLIENT.call(name, arguments)
        records.append({"tool": name, "arguments": arguments, **receipt})
        destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(name, receipt["elapsed_ms"], json.dumps(receipt["result"])[-2200:], flush=True)
        return receipt["result"]

    run("timeline_set_range", first=0, last=25)
    reference = run("semantic_pose_read", frame=25, include_joint_state=False)
    sampled = run(
        "semantic_pose_apply_sequence",
        sampled_motion=True,
        keyframes=[{"frame": frame, "pose": reference["pose"]} for frame in [0, 8, 16, 24]],
    )
    assert sampled["sampled_motion"] and sampled["sample_count"] == 4
    facts = probe("""
from mcp_for_cascadeur.host import characters, character_edit
rig = characters.resolve_character(scene, require_hands=True, require_semantics=True)
anchors = character_edit._contact_targets(rig, character_edit._positions(scene,rig,[0,4,8,12,16,20,24,25]))
result = {'character': rig.id, 'stationary_targets': len(anchors)}
""")
    assert facts["stationary_targets"] == 8
    checkpoint = run(
        "checkpoint_create", label="Stationary contact retiming with native interval canonicalization"
    )
    result = run(
        "character_retime",
        character_id=facts["character"],
        stabilize_contacts=True,
        keys=[
            {"source": a, "target": b}
            for a, b in zip([0, 4, 8, 12, 16, 20, 24, 25], [0, 3, 6, 10, 14, 19, 24, 25])
        ],
    )
    assert result["contacts"]["targets_pinned"] == 8 and result["contacts"]["baked_tracks"] == 2
    run("checkpoint_restore", checkpoint_id=checkpoint["checkpoint_id"])
    assert run("checkpoint_status", checkpoint_id=checkpoint["checkpoint_id"])["comparison"]["equivalent"]
    run("checkpoint_release", checkpoint_id=checkpoint["checkpoint_id"])
    print("ARTIFACT", destination, flush=True)


if __name__ == "__main__":
    main()
