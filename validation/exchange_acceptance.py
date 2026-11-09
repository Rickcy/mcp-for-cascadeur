"""Check native verified FBX output and optimistic publication conflicts."""

import hashlib
import json
import time

from native_advanced import CLIENT, CONFIG

from mcp_for_cascadeur.runner import RunnerError


def main():
    directory = CONFIG.workspace / "validation" / f"fbx-verified-{time.time_ns()}"
    directory.mkdir()
    records = []

    def run(name, **arguments):
        value = CLIENT.call(name, arguments)
        records.append({"tool": name, "arguments": arguments, **value})
        (directory / "receipts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(name, json.dumps(value)[-2400:], flush=True)
        return value["result"]

    count = run("scene_info")["frame_count"]
    run("timeline_set_range", first=0, last=count - 1, visible=True)
    ready = run("fbx_export_status")
    assert ready["ready"], ready
    destination = directory / "character.fbx"
    first = run("fbx_export_verified", path=str(destination), source_copy_path=str(directory / "source.casc"))
    assert first["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert first["model_unchanged"]
    for expected in (None, "0" * 64):
        try:
            CLIENT.call(
                "fbx_export_verified",
                {
                    "path": str(destination),
                    "source_copy_path": str(directory / "must-not-save.casc"),
                    **({"expected_sha256": expected} if expected else {}),
                },
            )
        except RunnerError as error:
            assert error.code == "EXPORT_CONFLICT", error.as_dict()
            records.append({"expected_conflict": error.as_dict()})
        else:
            raise AssertionError("An existing destination was overwritten without the matching digest")
        assert not (directory / "must-not-save.casc").exists()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == first["sha256"]
    run(
        "fbx_export_verified",
        path=str(destination),
        source_copy_path=str(directory / "source-replace.casc"),
        expected_sha256=first["sha256"],
    )
    print("ARTIFACT", directory, flush=True)


if __name__ == "__main__":
    main()
