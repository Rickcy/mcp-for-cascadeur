"""Verify the durable fence after a real native process interruption."""

import json

from native_advanced import CLIENT, CONFIG

from mcp_for_cascadeur.runner import RunnerError


def main():
    records = []
    status = CLIENT.call("recovery_status", {})
    assert status["result"]["persistent"] and status["result"]["writes_blocked"]
    records.append({"tool": "recovery_status", **status})
    for name, arguments in (
        ("python_execute", {"code": "result = 1"}),
        ("timeline_set_frame", {"frame": 0}),
        ("action_run", {"action_id": "Scene.Undo", "expected_scene": "curve-fixture.casc"}),
        ("fbx_export", {"path": str(CONFIG.workspace / "validation" / "must-not-export.fbx")}),
    ):
        try:
            CLIENT.call(name, arguments)
        except RunnerError as error:
            assert error.code == "RECOVERY_REQUIRED", error.as_dict()
            records.append({"tool": name, "error": error.as_dict()})
            print(name, error.code, flush=True)
        else:
            raise AssertionError(f"Write bypassed the persisted fence: {name}")
    records.append({"tool": "scene_info", **CLIENT.call("scene_info", {})})
    destination = CONFIG.workspace / "validation" / "recovery-process-restart.json"
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
