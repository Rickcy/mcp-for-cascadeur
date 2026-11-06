"""Inject failures after a real native edit and check durable rollback outcomes."""

import json
import uuid

from native_advanced import CLIENT, CONFIG, probe


def fault(block_restore=False):
    identifier = uuid.uuid4().hex
    code = f"""
from mcp_for_cascadeur import host
from mcp_for_cascadeur.contracts import Tool
from mcp_for_cascadeur.host import common, recovery, state
original_registry, original_undo, original_restore = host.registry, recovery._undo, state.restore
def fail(*args, **kwargs):
    raise RuntimeError('Intentional recovery failure for the native acceptance case')
def edit_then_fail():
    obj = scene.model_viewer().get_objects()[0]
    old_name = scene.model_viewer().get_object_name(obj)
    common.modify(scene, 'Validate failure after a native edit',
                  lambda model, updater: model.set_object_name(obj, old_name + '_acceptance_failure'))
    raise RuntimeError('Intentional post-commit failure for the native acceptance case')
try:
    if {block_restore!r}:
        recovery._undo, state.restore = fail, fail
    host.registry = lambda: {{**original_registry(), 'validation_fault':
        Tool('validation_fault', 'Private native acceptance fault', edit_then_fail, read_only=False)}}
    result = host.dispatch('validation_fault', {{}}, request_id={identifier!r})
finally:
    host.registry, recovery._undo, state.restore = original_registry, original_undo, original_restore
"""
    return probe(code)


def main():
    records = []
    for blocked in (False, True):
        result = fault(blocked)
        assert not result["ok"]
        error = result["error"]
        assert error["rollback_verified"] is (not blocked), error
        assert error["execution_state"] == ("recovery_required" if blocked else "rolled_back")
        records.append({"forced_restore_failure": blocked, **result})
        print(json.dumps(result), flush=True)
        if blocked:
            status = CLIENT.call("recovery_status", {})
            assert status["result"]["writes_blocked"]
            records.append(status)
            restore = CLIENT.call("recovery_restore", {})
            assert restore["result"]["verified"]
            records.append(restore)
            assert not CLIENT.call("recovery_status", {})["result"]["writes_blocked"]
    (CONFIG.workspace / "validation" / "rollback-native-receipts.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
