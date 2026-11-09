"""Native workflow flags, scene guards and asynchronous render cleanup contracts."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "integrations/kimodo/workflow.py"
SPEC = importlib.util.spec_from_file_location("kimodo_workflow_test", MODULE_PATH)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100 + b"\x00\x00\x00\x00IEND\xaeB`\x82"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kimodo-workflow-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scene = self.root / "source.casc"
        self.scene.write_bytes(b"source")
        self.original = {"camera": {"position": [9, 9, 9], "target": [1, 1, 1],
                                    "mode": "ViewportMode.Points"}, "frame": 3}
        self.measurement = {"fps": 30, "units": "centimeters", "frames": [
            {"pelvis": [index, 100, 0]} for index in range(4)
        ]}
        self.calls = []

    def fake_native(self, client, operation, payload):
        self.calls.append((operation, payload))
        if operation == "render_start":
            output = Path(payload["output"])
            frames = list(range(0, len(payload["roots"]), payload["frame_step"]))
            if frames[-1] != len(payload["roots"]) - 1:
                frames.append(len(payload["roots"]) - 1)
            images = []
            for label in ("front", "side", "three-quarter"):
                directory = output / ("frames-" + label)
                directory.mkdir()
                for index, frame in enumerate(frames):
                    path = directory / f"{index:04d}.png"
                    path.write_bytes(PNG)
                    images.append({"view": label, "frame": frame, "path": str(path)})
            self.render_task = {"id": "render-id", "receipt": str(output / "native-render.json"),
                                "pid": 42, "scene": str(self.scene.resolve()),
                                "state": "completed", "completed": len(images), "total": len(images),
                                "frame_count": len(payload["roots"]), "frame_step": payload["frame_step"],
                                "frame_indices": frames, "fps": 30,
                                "images": images, "restoration": {"restored": True}, "error": None}
            workflow.write(self.render_task["receipt"], self.render_task)
            return self.render_task
        if operation == "render_status":
            return self.render_task
        if operation == "render_cancel":
            self.render_task["state"] = "cancelled"
            return self.render_task
        return {}

    @staticmethod
    def encode(command, **kwargs):
        Path(command[-1]).write_bytes(b"complete mp4")

    def render_patches(self, native=None, encode=None):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(workflow, "native", side_effect=native or self.fake_native))
        stack.enter_context(mock.patch.object(workflow.subprocess, "run", side_effect=encode or self.encode))
        stack.enter_context(mock.patch.object(workflow.time, "sleep"))
        stack.enter_context(mock.patch.dict(sys.modules, {
            "imageio_ffmpeg": types.SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg")
        }))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_native_transport_decodes_double_encoded_runner_value(self):
        client = mock.Mock()
        client._request.return_value = {"ok": True, "value": repr(json.dumps({"frame": 9}))}
        self.assertEqual(workflow.native(client, "frame", {"frame": 9}), {"frame": 9})
        self.assertEqual(client._request.call_args.args[0], "/run")

    def test_long_measurement_uses_bounded_contiguous_reads_including_the_last_frame(self):
        calls = []
        def measure(client, operation, payload):
            calls.append((operation, payload))
            return {"fps": 30, "units": "centimeters", "frames": list(range(
                payload["first_frame"], payload["first_frame"] + payload["count"]))}
        with mock.patch.object(workflow, "native", side_effect=measure):
            result = workflow.measure(None, "target.casc", 1801, calibration="target-calibration.json")
        self.assertEqual(result["frames"], list(range(1801)))
        self.assertTrue(all(row[1]["count"] <= 120 and row[1]["calibration"] == "target-calibration.json"
                            for row in calls))
        self.assertEqual(calls[-1][1]["count"], 1)

    def test_incomplete_native_measurement_is_rejected(self):
        with mock.patch.object(workflow, "native", return_value={"fps": 30, "units": "centimeters", "frames": []}):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                workflow.measure(None, "target.casc", 121)

    def test_native_transport_failure_and_native_error_are_not_success(self):
        client = mock.Mock()
        client._request.return_value = {"ok": False}
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            workflow.native(client, "frame", {})
        client._request.return_value = {"ok": True, "value": repr("{}"),
                                        "messages": [{"level": "error", "text": "solver failed"}]}
        with self.assertRaisesRegex(RuntimeError, "native error"):
            workflow.native(client, "frame", {})

    def test_native_transport_error_keeps_bounded_redacted_diagnostics_without_retrying(self):
        client = mock.Mock()
        key = "AQ." + "a" * 32
        client._request.return_value = {
            "ok": False,
            "error": "Traceback\n" + "context " * 1000
                     + "\nTimed out waiting for Cascadeur idle event; api_key='fixture-secret' " + key,
        }
        with self.assertRaises(RuntimeError) as raised:
            workflow.native(client, "render_status", {"id": "render-id"})
        message = str(raised.exception)
        self.assertIn("outcome is unknown", message)
        self.assertIn("Timed out waiting for Cascadeur idle event", message)
        self.assertIn("[redacted]", message)
        self.assertNotIn("fixture-secret", message)
        self.assertNotIn(key, message)
        self.assertLess(len(message), 1800)
        client._request.assert_called_once()

    def test_native_dispatch_requires_trusted_python_and_expected_scene(self):
        common = types.SimpleNamespace(
            current_context=lambda: (None, None, types.SimpleNamespace(get_path_name=lambda: str(self.scene)), None),
            get_settings=lambda: {"allow_python": False},
        )
        recovery = types.SimpleNamespace(status=lambda: {}, active_transaction=lambda: None)
        host = types.ModuleType("mcp_for_cascadeur.host")
        host.characters = mock.Mock()
        host.common = common
        host.jobs = mock.Mock()
        host.playback = mock.Mock()
        host.recovery = recovery
        host.scene = mock.Mock()
        with mock.patch.dict(sys.modules, {"mcp_for_cascadeur.host": host}):
            with self.assertRaisesRegex(RuntimeError, "Trusted Python"):
                workflow.native_dispatch("camera_read", {"scene": str(self.scene)})
            common.get_settings = lambda: {"allow_python": True}
            with self.assertRaisesRegex(ValueError, "active scene differs"):
                workflow.native_dispatch("camera_read", {"scene": str(self.root / "other.casc")})
            host.scene.camera_get.assert_not_called()

    def test_mapped_native_measurement_ignores_virtual_axes_and_reads_the_requested_offset(self):
        roles = ("pelvis", "foot_l", "toe_l", "foot_r", "toe_r")
        calibration = self.root / "calibration.json"
        workflow.write(calibration, {"joints": {**{name: {"id": str(i)} for i, name in enumerate(roles)},
                                                "f_middle1_l": {"position": [0, 0, 0]}}})
        data = types.SimpleNamespace(get_animation_size=lambda: 181,
                                     get_data_value=lambda channel, frame: [frame, 0, 0])
        scene = types.SimpleNamespace(model_viewer=lambda: mock.Mock(), data_viewer=lambda: data)
        host = types.ModuleType("mcp_for_cascadeur.host")
        host.common = types.SimpleNamespace(
            current_context=lambda: (None, None, types.SimpleNamespace(get_path_name=lambda: str(self.scene)), scene),
            get_settings=lambda: {"allow_python": True})
        host.characters = types.SimpleNamespace(_transform_data=lambda *args: "position")
        host.recovery = types.SimpleNamespace(status=lambda: {}, active_transaction=lambda: None)
        host.jobs, host.playback, host.scene = mock.Mock(), mock.Mock(), mock.Mock()
        points = types.SimpleNamespace(resolve_rig=lambda *args: types.SimpleNamespace(joints=list(map(str, range(5)))),
                                       _vector=lambda vector: vector)
        with mock.patch.dict(sys.modules, {"mcp_for_cascadeur.host": host}), \
                mock.patch.object(workflow, "module", return_value=points):
            result = workflow.native_dispatch("measure", {"scene": str(self.scene), "calibration": str(calibration),
                                                           "first_frame": 180, "count": 1})
        self.assertEqual(result["frames"], [{role: [180, 0, 0] for role in roles}])

    def test_capture_does_not_overwrite_existing_metadata(self):
        output = self.root / "constraints.npz"
        output.with_suffix(".json").write_text("keep", encoding="utf-8")
        with mock.patch.object(workflow, "native") as native:
            with self.assertRaises(FileExistsError):
                workflow.capture_constraints(None, self.scene, [0, 1], "calibration", "template", output)
            native.assert_not_called()
        self.assertEqual(output.with_suffix(".json").read_text(encoding="utf-8"), "keep")

    @staticmethod
    def constraint_fixture():
        names = ["Hips", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]
        first = [[0, 1, 0], [0.1, 0.9, 0], [-0.1, 0.9, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0],
                 [0.1, 0, 0.2], [-0.1, 0, 0.2]]
        return {"units": "meters", "up_axis": "Y", "joint_names": names, "frame_indices": [0, 30],
                "num_frames": 31, "global_positions": [first, [[x, y, z + 1] for x, y, z in first]],
                "global_rot_mats": []}

    def test_neural_input_preflight_rejects_pelvis_relative_below_floor_constraints(self):
        payload = self.constraint_fixture()
        self.assertTrue(workflow.constraint_preflight(payload)["accepted"])
        payload["global_positions"] = [[[x, y - 1, z] for x, y, z in row]
                                       for row in payload["global_positions"]]
        report = workflow.constraint_preflight(payload)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["minimum_foot_anchor_y_m"], -1)
        self.assertIn("below the canonical floor", report["reasons"][0])

    def test_neural_input_preflight_rejects_world_translation_and_heading_conflicts(self):
        payload = self.constraint_fixture()
        payload["global_positions"][0][0][0] = 2
        self.assertFalse(workflow.constraint_preflight(payload)["accepted"])
        payload = self.constraint_fixture()
        payload["global_positions"] = [[[-z, y, x] for x, y, z in row]
                                       for row in payload["global_positions"]]
        report = workflow.constraint_preflight(payload)
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(abs(report["first_heading_degrees"]), 90)

    def test_neural_input_preflight_allows_small_joint_floor_tolerance(self):
        payload = self.constraint_fixture()
        payload["global_positions"][0][-1][1] = -0.01
        report = workflow.constraint_preflight(payload)
        self.assertTrue(report["accepted"])
        self.assertIn("not mesh penetration", report["scope"])

    def capture_fixture(self, *, roundtrip=True, payload=None):
        calibration, template = self.root / "calibration.json", self.root / "template.json"
        workflow.write(calibration, {})
        workflow.write(template, {})
        keys = types.SimpleNamespace(
            conversion_roundtrip_report=mock.Mock(return_value={"accepted": roundtrip, "max_point_error_cm": 0}),
            convert=mock.Mock(return_value=payload or self.constraint_fixture()),
            write_npz=mock.Mock(return_value={"npz": "constraints.npz"}),
        )
        return calibration, template, keys

    def test_capture_runs_geometry_and_floor_preflight_before_publishing_npz(self):
        calibration, template, keys = self.capture_fixture()
        output = self.root / "constraints.npz"
        with mock.patch.object(workflow, "native", return_value={"snapshot": True}), \
                mock.patch.object(workflow, "module", return_value=keys):
            result = workflow.capture_constraints(None, self.scene, [0, 30], calibration, template, output)
        keys.conversion_roundtrip_report.assert_called_once()
        keys.write_npz.assert_called_once()
        self.assertTrue(result["neural_input_preflight"]["accepted"])
        metadata = json.loads(output.with_suffix(".json").read_text())
        self.assertTrue(metadata["conversion_roundtrip"]["accepted"])
        self.assertNotIn("global_positions", metadata)

    def test_capture_rejects_algebraic_drift_before_npz_or_sidecars(self):
        calibration, template, keys = self.capture_fixture(roundtrip=False)
        output = self.root / "constraints.npz"
        with mock.patch.object(workflow, "native", return_value={}), \
                mock.patch.object(workflow, "module", return_value=keys):
            with self.assertRaisesRegex(ValueError, "does not preserve"):
                workflow.capture_constraints(None, self.scene, [0, 30], calibration, template, output)
        keys.convert.assert_not_called()
        keys.write_npz.assert_not_called()
        self.assertFalse(output.with_suffix(".capture.json").exists())

    def test_capture_rejects_below_floor_model_input_even_if_algebra_matches(self):
        payload = self.constraint_fixture()
        payload["global_positions"][0][-1][1] = -1
        calibration, template, keys = self.capture_fixture(payload=payload)
        output = self.root / "constraints.npz"
        with mock.patch.object(workflow, "native", return_value={}), \
                mock.patch.object(workflow, "module", return_value=keys):
            with self.assertRaisesRegex(ValueError, "below the canonical floor"):
                workflow.capture_constraints(None, self.scene, [0, 30], calibration, template, output)
        keys.write_npz.assert_not_called()

    def transfer_fixture(self, verified=True):
        calibration, motion = self.root / "calibration.json", self.root / "motion.json"
        workflow.write(calibration, {})
        workflow.write(motion, {})
        session_dir = self.root / "native-session"
        session_dir.mkdir()
        receipt = session_dir / "session.json"
        workflow.write(receipt, {"directory": str(session_dir), "phase": "completed",
                                 "reopen_verification": {"point_coordinates_verified": verified,
                                                         "max_coordinate_error": 1e-6, "tolerance": 1e-5}})
        plan = {"native_cluster_validation": {"max_error": 1e-6}}
        points = types.SimpleNamespace(build_plan=mock.Mock(return_value=plan),
                                       prepare=mock.Mock(return_value={"receipt_path": str(receipt)}))
        importer = types.SimpleNamespace(status=mock.Mock(side_effect=[{"phase": "points_applied"},
                                                                       {"phase": "completed"}]), step=mock.Mock())
        return calibration, motion, points, importer

    def test_target_timebase_is_read_from_the_saved_scene_and_rejects_60_fps(self):
        for fps in (30, 60):
            with zipfile.ZipFile(self.scene, "w") as archive:
                archive.writestr("scene/scene_settings.json", json.dumps({"Data": {"FPS": fps, "UnitLength": "cm"}}))
            if fps == 30:
                self.assertEqual(workflow.target_settings(self.scene)["fps"], 30)
            else:
                with self.assertRaisesRegex(ValueError, "30 FPS"):
                    workflow.target_settings(self.scene)

    def test_transfer_reports_only_readback_confirmed_frame_batches_without_repeating_steps(self):
        calibration, motion, points, importer = self.transfer_fixture()
        points.build_plan.return_value["num_frames"] = 40
        receipt = Path(points.prepare.return_value["receipt_path"])
        phases = [("points_ready:0", 0), ("points_ready:20", 20), ("points_applied", 40), ("completed", 40)]
        cursor = 0

        def persist():
            record = workflow.read(receipt)
            record.update(phase=phases[cursor][0], next_frame=phases[cursor][1], num_frames=40)
            workflow.write(receipt, record)

        def status(*args):
            return {"phase": phases[cursor][0]}

        def step(*args):
            nonlocal cursor
            cursor += 1
            persist()

        persist()
        importer.status, importer.step = mock.Mock(side_effect=status), mock.Mock(side_effect=step)
        observations = []
        with mock.patch.object(workflow, "module", side_effect=lambda name: {
                "point_retarget": points, "native_import": importer}[name]), contextlib.redirect_stdout(io.StringIO()):
            workflow.transfer(None, motion, calibration, self.scene, self.root / "transfer",
                              progress_callback=observations.append)
        self.assertEqual([(row["completed"], row["total"]) for row in observations],
                         [(0, 40), (20, 40), (40, 40), (40, 40)])
        self.assertEqual(importer.step.call_count, 3)

    def test_transfer_rejects_false_completed_cursor_without_another_native_step(self):
        calibration, motion, points, importer = self.transfer_fixture()
        points.build_plan.return_value["num_frames"] = 40
        receipt = Path(points.prepare.return_value["receipt_path"])
        record = workflow.read(receipt)
        record.update(phase="completed", next_frame=20, num_frames=40)
        workflow.write(receipt, record)
        importer.status = mock.Mock(return_value={"phase": "completed"})
        with mock.patch.object(workflow, "module", side_effect=lambda name: {
                "point_retarget": points, "native_import": importer}[name]):
            with self.assertRaisesRegex(RuntimeError, "completion differs"):
                workflow.transfer(None, motion, calibration, self.scene, self.root / "transfer",
                                  progress_callback=lambda value: None)
        importer.step.assert_not_called()

    def test_transfer_uses_actual_point_coordinates_verified_persistence_flag(self):
        calibration, motion, points, importer = self.transfer_fixture()
        with mock.patch.object(workflow, "module", side_effect=lambda name: {
            "point_retarget": points, "native_import": importer
        }[name]), contextlib.redirect_stdout(io.StringIO()):
            result = workflow.transfer(None, motion, calibration, self.scene, self.root / "transfer")
        self.assertTrue(result["validation"]["persistence"])
        importer.step.assert_called_once()
        self.assertEqual(json.loads((self.root / "transfer/transfer.json").read_text())["validation"],
                         result["validation"])

    def test_failed_persistence_is_not_promoted_to_verified(self):
        calibration, motion, points, importer = self.transfer_fixture(verified=False)
        with mock.patch.object(workflow, "module", side_effect=lambda name: {
            "point_retarget": points, "native_import": importer
        }[name]), contextlib.redirect_stdout(io.StringIO()):
            result = workflow.transfer(None, motion, calibration, self.scene, self.root / "transfer")
        self.assertFalse(result["validation"]["persistence"])

    def test_transfer_checks_anchor_poses_after_reopen_at_relative_frames(self):
        calibration, motion, points, importer = self.transfer_fixture()
        snapshot = {"poses": [{"frame": 5}, {"frame": 15}]}
        snapshot_path = self.root / "snapshot.json"
        workflow.write(snapshot_path, snapshot)
        keyposes = types.SimpleNamespace(
            build_return_plan=mock.Mock(return_value={"native_cluster_validation": {"max_error": 1e-6}}),
            compare_captures=mock.Mock(return_value={"accepted": False, "max_point_error_cm": 3}),
        )
        with mock.patch.object(workflow, "module", side_effect=lambda name: {
            "point_retarget": points, "native_import": importer, "keypose_constraints": keyposes,
        }[name]), mock.patch.object(workflow, "native", return_value={"observed": True}) as native, \
                contextlib.redirect_stdout(io.StringIO()):
            result = workflow.transfer(None, motion, calibration, self.scene, self.root / "transfer", snapshot_path)
        self.assertEqual(native.call_args.args[2]["frames"], [0, 10])
        keyposes.compare_captures.assert_called_once_with(snapshot, {"observed": True})
        self.assertFalse(result["validation"]["constraints"])
        self.assertTrue(result["validation"]["constraints_required"])

    def test_transfer_never_replays_an_existing_plan(self):
        output = self.root / "transfer"
        output.mkdir()
        (output / "point-plan.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(workflow, "module") as load:
            with self.assertRaisesRegex(FileExistsError, "inspect its session"):
                workflow.transfer(None, "motion", "calibration", self.scene, output)
            load.assert_not_called()

    def test_native_metrics_contact_mask_and_no_coverage_are_distinct(self):
        rows = [{name: [index, 0, 0] for name in ("foot_l", "toe_l", "foot_r", "toe_r")}
                for index in range(3)]
        measured = workflow.native_metrics({"frames": rows, "fps": 30}, [[1] * 4] * 3)
        self.assertAlmostEqual(measured["native_contact_speed_m_s"], 0.3)
        self.assertEqual(measured["native_contact_coverage_ratio"], 1)
        empty = workflow.native_metrics({"frames": rows, "fps": 30}, [[0] * 4] * 3)
        self.assertIsNone(empty["native_contact_speed_m_s"])
        self.assertEqual(empty["native_contact_coverage_ratio"], 0)

    def test_render_retains_three_videos_removes_frames_and_restores_state(self):
        output = self.root / "render"
        with self.render_patches():
            videos = workflow.render(None, self.scene, self.measurement, output)
        self.assertEqual({path.name for path in output.iterdir()},
                         {"front.mp4", "side.mp4", "three-quarter.mp4", "native-render.json"})
        self.assertEqual(len(videos), 3)
        self.assertEqual([operation for operation, _ in self.calls], ["render_start", "render_status"])
        self.assertEqual([item["fps"] for item in videos], [30, 30, 30])
        self.assertTrue(self.render_task["restoration"]["restored"])

    def test_render_callback_reports_real_frame_and_encoded_view_counts(self):
        observations = []
        with self.render_patches():
            workflow.render(None, self.scene, self.measurement, self.root / "render",
                            progress_callback=observations.append)
        frames = [row for row in observations if row["stage"] == "native_render"]
        encoded = [row for row in observations if row["stage"] == "encoding_review"]
        self.assertEqual(frames[-1]["completed"], frames[-1]["total"])
        self.assertEqual(frames[-1]["state"], "completed")
        self.assertEqual([row["completed"] for row in encoded], [0, 1, 2])
        self.assertEqual([row["view"] for row in encoded], ["front", "side", "three-quarter"])

    def test_render_progress_uses_the_journal_until_the_native_owner_is_terminal(self):
        def foreground_panel(client, operation, payload):
            if operation == "render_status":
                self.assertEqual(self.render_task["state"], "completed",
                                 "An active renderer cannot depend on scene-idle HTTP polling")
            value = self.fake_native(client, operation, payload)
            if operation == "render_start":
                value.update(state="running", completed=0, restoration=None)
                workflow.write(value["receipt"], value)
            return value

        def native_timer_advances(delay):
            self.render_task["completed"] += 3
            if self.render_task["completed"] == self.render_task["total"]:
                self.render_task.update(state="completed", restoration={"restored": True})
            workflow.write(self.render_task["receipt"], self.render_task)

        with self.render_patches(native=foreground_panel), \
                mock.patch.object(workflow.time, "sleep", side_effect=native_timer_advances):
            videos = workflow.render(None, self.scene, self.measurement, self.root / "render")
        self.assertEqual(len(videos), 3)
        self.assertEqual([operation for operation, _ in self.calls], ["render_start", "render_status"])

    def test_changed_render_journal_cannot_select_an_owner_or_delete_scratch(self):
        def tampered(client, operation, payload):
            value = self.fake_native(client, operation, payload)
            if operation == "render_start":
                record = workflow.read(value["receipt"])
                record["pid"] += 1
                workflow.write(value["receipt"], record)
            return value

        output = self.root / "render"
        with self.render_patches(native=tampered):
            with self.assertRaisesRegex(RuntimeError, "receipt identity"):
                workflow.render(None, self.scene, self.measurement, output)
        self.assertEqual([operation for operation, _ in self.calls], ["render_start", "render_cancel"])
        self.assertEqual(len(list(output.glob("frames-*/*.png"))), 9)

    def test_renderer_error_after_writing_png_cleans_own_scratch_and_restores(self):
        def fail(client, operation, payload):
            value = self.fake_native(client, operation, payload)
            if operation == "render_status":
                return {**value, "state": "failed", "error": "native renderer failed after writing"}
            return value
        output = self.root / "render"
        with self.render_patches(native=fail):
            with self.assertRaisesRegex(RuntimeError, "renderer failed"):
                workflow.render(None, self.scene, self.measurement, output)
        self.assertEqual([path.name for path in output.iterdir()], ["native-render.json"])
        self.assertTrue(self.render_task["restoration"]["restored"])

    def test_encoder_failure_removes_incomplete_video(self):
        def fail(command, **kwargs):
            Path(command[-1]).write_bytes(b"incomplete")
            raise RuntimeError("encoding failed")
        output = self.root / "render"
        with self.render_patches(encode=fail):
            with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                workflow.render(None, self.scene, self.measurement, output)
        self.assertEqual([path.name for path in output.iterdir()], ["native-render.json"])

    def test_failed_native_restoration_is_not_promoted_to_review_media(self):
        def fail(client, operation, payload):
            value = self.fake_native(client, operation, payload)
            if operation == "render_status":
                return {**value, "state": "failed", "error": "original render error",
                        "restoration": {"restored": False, "errors": ["restore camera error"]}}
            return value
        with self.render_patches(native=fail):
            with self.assertRaisesRegex(RuntimeError, "original render error"):
                workflow.render(None, self.scene, self.measurement, self.root / "render")
        self.assertFalse(list((self.root / "render").glob("*.mp4")))

    def test_unknown_render_owner_keeps_pngs_and_receipt_for_inspection(self):
        def unavailable(client, operation, payload):
            if operation in {"render_status", "render_cancel"}:
                raise RuntimeError("native transport outcome unknown")
            return self.fake_native(client, operation, payload)

        output = self.root / "render"
        with self.render_patches(native=unavailable):
            with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
                workflow.render(None, self.scene, self.measurement, output)
        self.assertTrue((output / "native-render.json").is_file())
        self.assertEqual(len(list(output.glob("frames-*/*.png"))), 9)

    def test_render_rejects_bad_timebase_and_step_before_native_changes(self):
        with self.render_patches():
            for step in (0, -1, True, 1.5):
                with self.subTest(step=step), self.assertRaises(ValueError):
                    workflow.render(None, self.scene, self.measurement, self.root / "render", frame_step=step)
            measurement = {**self.measurement, "fps": 60}
            with self.assertRaisesRegex(ValueError, "30 FPS"):
                workflow.render(None, self.scene, measurement, self.root / "render")
        self.assertFalse(self.calls)

    def test_asynchronous_png_waits_for_complete_file_and_stable_stamp(self):
        path = self.root / "async.png"
        path.write_bytes(PNG[:20])
        def complete_once(delay):
            if path.stat().st_size < len(PNG):
                path.write_bytes(PNG)
        with mock.patch.object(workflow.time, "sleep", side_effect=complete_once) as sleep:
            workflow.wait_for_image(path)
        self.assertGreaterEqual(sleep.call_count, 2)

    def test_asynchronous_png_timeout_is_bounded(self):
        with mock.patch.object(workflow.time, "monotonic", side_effect=[0, 0.1, 11]), \
                mock.patch.object(workflow.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                workflow.wait_for_image(self.root / "missing.png")


if __name__ == "__main__":
    unittest.main()
