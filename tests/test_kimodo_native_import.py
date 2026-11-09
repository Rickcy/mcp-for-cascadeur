"""Check native import dispatch guards and transport uncertainty without Cascadeur."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "integrations" / "kimodo" / "native_import.py"
SPEC = importlib.util.spec_from_file_location("kimodo_native_import_test", MODULE_PATH)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class KimodoNativeImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kimodo-native-cli-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.receipt = self.directory / "session.json"
        self.record = {
            "schema": "kimodo-cascadeur-import-v1",
            "directory": str(self.directory),
            "phase": "prepared",
            "events": [],
        }
        self.write_record()
        self.state = {
            "ready": True,
            "blockers": [],
            "pid": 81860,
            "frame": 0,
            "view_path": "original.casc",
            "view_name": "Original",
        }
        self.client = SimpleNamespace(health=lambda: {"ok": True, "version": "test", "queued_requests": 0})

    def write_record(self):
        self.receipt.write_text(json.dumps(self.record), encoding="utf-8")

    def test_raw_response_is_decoded_without_eval(self):
        payload = {"ok": True, "result": {"phase": "target_open"}}
        client = SimpleNamespace(
            _request=lambda route, body: {"ok": True, "value": repr(json.dumps(payload))}
        )
        self.assertEqual(
            cli._raw(client, "open_target", str(self.receipt)), {**payload, "native_messages": []}
        )

    def test_native_warnings_are_preserved_for_retarget_diagnostics(self):
        payload = {"ok": True, "result": {"phase": "target_open"}}
        messages = [{"level": "Warning", "text": "No supported rig was selected"}]
        client = SimpleNamespace(
            _request=lambda route, body: {
                "ok": True,
                "value": repr(json.dumps(payload)),
                "messages": messages,
            }
        )
        self.assertEqual(cli._raw(client, "open_target", str(self.receipt))["native_messages"], messages)

    def test_failed_stage_retains_native_events_with_the_traceback(self):
        payload = {"ok": False, "error": {"message": "No root motion", "execution_state": "unknown"}}
        messages = [{"level": "Info", "text": "Retargeting: failed"}]
        client = SimpleNamespace(
            _request=lambda route, body: {
                "ok": True,
                "value": repr(json.dumps(payload)),
                "messages": messages,
            }
        )
        with self.assertRaises(cli.ImportFailure) as failure:
            cli._raw(client, "retarget_paste", str(self.receipt))
        self.assertEqual(failure.exception.details["messages"], messages)

    def test_malformed_native_response_is_an_unknown_outcome(self):
        client = SimpleNamespace(_request=lambda route, body: {"ok": True, "value": "not a Python literal"})
        with self.assertRaises(cli.ImportFailure) as failure:
            cli._raw(client, "open_target", str(self.receipt))
        self.assertEqual(failure.exception.execution_state, "unknown")

    def test_native_error_messages_do_not_become_success(self):
        payload = {"ok": True, "result": {"phase": "target_open"}}
        client = SimpleNamespace(
            _request=lambda route, body: {
                "ok": True,
                "value": repr(json.dumps(payload)),
                "messages": [{"level": "Error", "text": "Import failed"}],
            }
        )
        with self.assertRaisesRegex(cli.ImportFailure, "native errors"):
            cli._raw(client, "open_target", str(self.receipt))

    def test_core_recovery_guard_is_checked_before_native_stage(self):
        blocked = {**self.state, "ready": False, "blockers": ["Recovery required"]}
        calls = []
        bridge = SimpleNamespace(open_target=lambda receipt: calls.append(receipt))
        with (
            patch.object(cli, "_native_preflight", return_value=blocked),
            patch.object(cli, "_load_bridge", return_value=bridge),
        ):
            result = cli._native_dispatch("open_target", str(self.receipt), 81860, cli._witness(blocked))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["execution_state"], "not_started")
        self.assertEqual(calls, [])

    def test_restart_or_moving_playhead_blocks_native_stage(self):
        calls = []
        bridge = SimpleNamespace(open_target=lambda receipt: calls.append(receipt))
        with (
            patch.object(cli, "_native_preflight", return_value=self.state),
            patch.object(cli, "_load_bridge", return_value=bridge),
        ):
            changed_pid = cli._native_dispatch(
                "open_target", str(self.receipt), 1234, cli._witness(self.state)
            )
            witness = {**cli._witness(self.state), "frame": 10}
            changed_frame = cli._native_dispatch("open_target", str(self.receipt), 81860, witness)
        self.assertFalse(changed_pid["ok"])
        self.assertFalse(changed_frame["ok"])
        self.assertEqual(calls, [])

    def test_transport_timeout_fences_retry_even_if_native_phase_is_unchanged(self):
        with (
            patch.object(cli, "_idle_preflight", return_value=self.state),
            patch.object(cli, "_raw", side_effect=TimeoutError("connection lost")) as raw,
        ):
            with self.assertRaises(TimeoutError):
                cli.step(self.client, self.receipt)
            with self.assertRaisesRegex(cli.ImportFailure, "unknown"):
                cli.step(self.client, self.receipt)
        self.assertEqual(raw.call_count, 1)
        marker = json.loads((self.directory / ".native-transport.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["status"], "unknown")
        self.assertFalse((self.directory / ".native-import.lock").exists())

    def test_status_inspects_without_replaying_or_clearing_uncertainty(self):
        marker = self.directory / ".native-transport.json"
        marker.write_text(
            json.dumps({"status": "unknown", "expected_phase": "target_open"}), encoding="utf-8"
        )
        with patch.object(cli, "_raw", return_value={"ok": True, "result": {"phase": "target_open"}}) as raw:
            result = cli.status(self.client, self.receipt, native=True)
        self.assertEqual(result["transport"]["status"], "unknown")
        self.assertEqual(raw.call_args.args[1], "inspect")
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["status"], "unknown")

    def test_only_explicit_acknowledgement_accepts_proven_native_completion(self):
        marker = self.directory / ".native-transport.json"
        marker.write_text(
            json.dumps({"status": "unknown", "expected_phase": "target_open"}), encoding="utf-8"
        )
        with patch.object(cli, "_raw", return_value={"ok": True, "result": {"phase": "target_open"}}):
            result = cli.status(self.client, self.receipt, acknowledge_completed=True)
        self.assertEqual(result["transport"]["status"], "acknowledged_completed")

    def test_queued_work_is_not_mistaken_for_idle(self):
        for field in ("queued", "queued_requests"):
            with self.subTest(field=field):
                client = SimpleNamespace(health=lambda: {"ok": True, "version": "test", field: 1})
                with patch.object(cli, "_raw") as raw, self.assertRaisesRegex(cli.ImportFailure, "queued"):
                    cli._idle_preflight(client)
                raw.assert_not_called()

    def test_request_timeout_cannot_exceed_native_bound(self):
        with self.assertRaisesRegex(ValueError, "35 seconds"):
            cli._runner_client("http://127.0.0.1:8765", 36)

    def test_real_point_receipt_module_integrates_with_cli_before_native_submission(self):
        self.record.update(
            schema="kimodo-point-session-v1", phase="points_prepared", num_frames=120, next_frame=0
        )
        self.write_record()
        calls = []

        def native_reply(_client, operation, receipt_path, expected_pid, idle_witness):
            calls.append(operation)
            points = cli._receipt_bridge(receipt_path)
            path, record = points._read(receipt_path)
            self.assertEqual(record["native_pid"], self.state["pid"])
            record["phase"] = "points_ready:0"
            points._write(path, record)
            return {
                "ok": True,
                "result": {"phase": "points_ready:0", "native_points": 43},
                "native_messages": [],
            }

        with (
            patch.object(cli, "_idle_preflight", return_value=self.state),
            patch.object(cli, "_raw", side_effect=native_reply),
        ):
            result = cli.step(self.client, self.receipt, stage="point_attach", expected_pid=self.state["pid"])
        self.assertEqual(result["phase"], "points_ready:0")
        self.assertEqual(calls, ["point_attach"])
        persisted = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(persisted["phase"], "points_ready:0")
        marker = json.loads((self.directory / ".native-transport.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["status"], "completed")

    def test_callback_domain_scene_is_forwarded_to_closing_stage(self):
        callback_scene = object()
        observed = []
        bridge = SimpleNamespace(
            cleanup=lambda path, **options: (
                observed.append(options["callback_scene"]) or {"phase": "completed"}
            )
        )
        with (
            patch.object(cli, "_native_preflight", return_value=self.state),
            patch.object(cli, "_receipt_bridge", return_value=bridge),
        ):
            response = cli._native_dispatch(
                "point_cleanup",
                str(self.receipt),
                self.state["pid"],
                cli._witness(self.state),
                callback_scene=callback_scene,
            )
        self.assertTrue(response["ok"])
        self.assertEqual(observed, [callback_scene])


if __name__ == "__main__":
    unittest.main()
