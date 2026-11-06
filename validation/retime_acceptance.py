"""Check native character key relocation and retained section metadata."""

import json

from native_advanced import CLIENT, CONFIG, probe


def main():
    records = []
    probe("""
import builtins, faulthandler
builtins._mcp_validation_fault_log = open(workspace / 'validation' / 'native-fault.log', 'a')
faulthandler.enable(file=builtins._mcp_validation_fault_log, all_threads=True)
result = True
""")

    def run(name, **arguments):
        value = CLIENT.call(name, arguments)
        records.append({"tool": name, "arguments": arguments, **value})
        print(name, json.dumps(value)[-4000:], flush=True)
        return value["result"]

    facts = probe("""
from mcp_for_cascadeur.host import characters, curves
rig = characters.resolve_character(scene, None, require_hands=True)
result = {'character': rig.id, 'tracks': {str(k): {'name': v.header.name,
          'keys': list(v.key_frame_indices()), 'objects': len(v.obj_ids)}
          for k, v in curves.character_tracks(scene, rig).items()}}
""")
    print(json.dumps(facts, indent=2), flush=True)
    keys = [0, 4, 8, 12, 16, 20, 24, 25]
    unsynchronized = [key for key, value in facts["tracks"].items() if value["keys"] != keys]
    if unsynchronized:
        run("tracks_set_keys", tracks=unsynchronized, frames=keys)
    checkpoint = run("checkpoint_create", label="Cascy synchronized character timing")
    run(
        "character_retime",
        character_id=facts["character"],
        keys=[
            {"source": source, "target": target}
            for source, target in zip(keys, [0, 3, 6, 10, 14, 19, 24, 25])
        ],
    )
    run("checkpoint_restore", checkpoint_id=checkpoint["checkpoint_id"])
    assert run("checkpoint_status", checkpoint_id=checkpoint["checkpoint_id"])["comparison"]["equivalent"]
    destination = CONFIG.workspace / "validation" / "retime-native-receipts.json"
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
