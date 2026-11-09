"""Source-motion diagnostics must distinguish contact coverage from low sliding."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "integrations/kimodo/compare_motion.py"
SPEC = importlib.util.spec_from_file_location("kimodo_compare_motion", MODULE_PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def motion(count=5, contact_slots=6):
    return {
        "posed_joints": np.zeros((count, 77, 3)),
        "root_positions": np.zeros((count, 3)),
        "foot_contacts": np.ones((count, contact_slots), dtype=bool),
        "fps": np.array(30.0),
    }


class MotionMeasurementTests(unittest.TestCase):
    def test_stationary_contact_has_coverage_and_zero_sliding(self):
        measured = compare.measure_motion(motion())
        self.assertEqual(measured["predicted_contact_feet"]["contact_slot_intervals"], 16)
        self.assertEqual(measured["height_mask_toes"]["contact_slot_intervals"], 8)
        self.assertEqual(measured["predicted_contact_feet"]["mean_horizontal_speed_m_s"], 0)

    def test_constant_contact_slide_is_measured_in_metres_per_second(self):
        data = motion()
        data["posed_joints"][:, :, 0] = np.arange(5)[:, None] * 0.01
        measured = compare.measure_motion(data)
        self.assertAlmostEqual(measured["predicted_contact_feet"]["mean_horizontal_speed_m_s"], 0.3)
        self.assertAlmostEqual(measured["height_mask_toes"]["mean_horizontal_speed_m_s"], 0.3)

    def test_soma77_toe_end_slots_do_not_contaminate_four_contact_measurement(self):
        data = motion()
        data["posed_joints"][:, (71, 76), 0] = np.arange(5)[:, None] * 10
        data["foot_contacts"][:, (2, 5)] = False
        measured = compare.measure_motion(data)
        self.assertEqual(measured["predicted_contact_feet"]["mean_horizontal_speed_m_s"], 0)
        self.assertEqual(measured["predicted_contact_feet"]["coverage_ratio"], 1)

    def test_no_predicted_contact_reports_unknown_slip_not_perfect_zero(self):
        data = motion()
        data["foot_contacts"][:] = False
        measured = compare.measure_motion(data)
        self.assertIsNone(measured["predicted_contact_feet"]["mean_horizontal_speed_m_s"])
        self.assertEqual(measured["predicted_contact_feet"]["coverage_ratio"], 0)
        self.assertEqual(measured["height_mask_toes"]["coverage_ratio"], 1)

    def test_swing_and_contact_transitions_are_excluded(self):
        data = motion()
        data["foot_contacts"][2:] = False
        data["posed_joints"][2:, :, 0] = np.arange(3)[:, None] + 1
        measured = compare.measure_motion(data)
        self.assertEqual(measured["predicted_contact_feet"]["contact_slot_intervals"], 4)
        self.assertEqual(measured["predicted_contact_feet"]["mean_horizontal_speed_m_s"], 0)

    def test_height_mask_uses_both_toe_endpoints_and_can_disagree_with_prediction(self):
        data = motion()
        data["posed_joints"][:, :, 1] = 0.1
        measured = compare.measure_motion(data)
        self.assertEqual(measured["height_mask_toes"]["coverage_ratio"], 0)
        self.assertEqual(measured["predicted_contact_feet"]["coverage_ratio"], 1)

    def test_path_distance_and_elapsed_sample_span_are_distinct_from_displacement(self):
        data = motion(3)
        data["root_positions"][:, 0] = [0, 1, 0]
        measured = compare.measure_motion(data)
        self.assertEqual(measured["root_planar_displacement_m"], 0)
        self.assertEqual(measured["root_planar_path_length_m"], 2)
        self.assertAlmostEqual(measured["root_mean_planar_speed_m_s"], 30)
        self.assertAlmostEqual(measured["duration_seconds"], 0.1)
        self.assertAlmostEqual(measured["sample_span_seconds"], 2 / 30)

    def test_invalid_arrays_are_rejected_before_measurement(self):
        for key in ("root_positions", "posed_joints", "foot_contacts", "fps"):
            data = motion()
            data[key] = data[key].astype(float)
            data[key].flat[0] = np.nan
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "finite"):
                compare.measure_motion(data)
        data = motion()
        data["posed_joints"] = np.zeros((5, 30, 3))
        with self.assertRaisesRegex(ValueError, "SOMA77"):
            compare.measure_motion(data)


class ComparisonInputTests(unittest.TestCase):
    def test_clip_load_reads_generation_log_and_does_not_assume_missing_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walk.npz"
            np.savez(path, **motion())
            path.with_suffix(".json").write_text(
                json.dumps({"prompt": "walk", "num_frames": 5}), encoding="utf-8"
            )
            path.with_suffix(".log").write_text(
                'STATUS: before\n{"seed":42,"diffusion_steps":50}\nSTATUS: done', encoding="utf-8"
            )
            clip = compare.inspect_clip(path)
            result = compare.compare_clips(clip, clip)
            self.assertEqual(clip["generation_settings"]["seed"], 42)
            self.assertIn("fps", result["comparison_conditions"]["missing_recorded_settings"])
            self.assertFalse(result["comparison_conditions"]["recorded_settings_match"])

    def test_changed_settings_are_reported_even_when_metrics_match(self):
        metrics = compare.measure_motion(motion())
        baseline = {"generation_settings": {key: 1 for key in compare.SETTINGS}, "metrics": metrics}
        candidate = {
            "generation_settings": {**baseline["generation_settings"], "seed": 2},
            "metrics": metrics,
        }
        result = compare.compare_clips(baseline, candidate)
        self.assertEqual(
            result["comparison_conditions"]["different_recorded_settings"],
            {"seed": {"baseline": 1, "candidate": 2}},
        )
        self.assertFalse(result["comparison_conditions"]["recorded_settings_match"])

    def test_incorrect_recorded_frame_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walk.npz"
            np.savez(path, **motion())
            path.with_suffix(".json").write_text('{"num_frames":120}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "num_frames"):
                compare.inspect_clip(path)

    def test_cli_emits_json_without_creating_report_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walk.npz"
            np.savez(path, **motion())
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(compare.main([str(path), str(path)]), 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["schema"], "kimodo-motion-comparison-v1")
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["walk.npz"])


if __name__ == "__main__":
    unittest.main()
