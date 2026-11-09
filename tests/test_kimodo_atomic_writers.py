"""UI reads must not turn a transient Windows journal replacement into a job retry."""

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations/kimodo"


@pytest.mark.parametrize("name", ["pipeline", "workflow", "native_render", "cascadeur_bridge"])
def test_journal_writer_retries_only_a_transient_windows_replace(name, tmp_path):
    spec = importlib.util.spec_from_file_location("kimodo_atomic_test_" + name, INTEGRATION / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "progress.json"
    target.write_text('{"state":"previous"}', encoding="utf-8")
    actual_replace = os.replace
    attempts = []

    def temporary_sharing_conflict(source, destination):
        attempts.append((Path(source), Path(destination)))
        if len(attempts) == 1:
            error = PermissionError(13, "A foreground UI briefly holds the progress file")
            error.winerror = 5
            raise error
        return actual_replace(source, destination)

    with patch("os.replace", side_effect=temporary_sharing_conflict), patch("time.sleep"):
        if name == "native_render":
            with patch.object(module, "_Native", side_effect=AssertionError("A receipt update cannot restart rendering")):
                module._write({"receipt": str(target), "state": "running", "_private": "not serialized"})
        elif name == "cascadeur_bridge":
            with patch.object(module, "_application", side_effect=AssertionError("A receipt update cannot dispatch native work")):
                module._write(target, {"state": "running"})
        else:
            with patch.object(module, "module", side_effect=AssertionError("A journal update cannot dispatch native/model work")):
                module.write(target, {"state": "running"})

    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["state"] == "running"
    assert "_private" not in record
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert not attempts[0][0].exists()
