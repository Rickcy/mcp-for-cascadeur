"""Check anatomical keypose conversion and native world-space roundtrips."""

import copy
import importlib.util
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


keyposes = load("keypose_constraints_test", ROOT / "integrations/kimodo/keypose_constraints.py")
forward_tests = load("keypose_forward_fixtures", ROOT / "tests/test_kimodo_point_retarget.py")
points = keyposes._points()
IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
YAW = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]


def fixture():
    calibration, template = forward_tests.KimodoPointRetargetTests().fixture()
    names = template["joint_names"]
    parents = []
    inverse = {source: target for target, source in points.MAPPING.items()}
    for name in names:
        if name in inverse:
            parent = points.PARENTS[inverse[name]]
            parent = points.MAPPING[parent] if parent is not None else None
        elif name == "HeadEnd":
            parent = "Head"
        elif "HandMiddle1" in name:
            parent = name.replace("HandMiddle1", "Hand")
        elif "ToeEnd" in name:
            parent = name.replace("ToeEnd", "ToeBase")
        else:
            raise AssertionError(name)
        parents.append(names.index(parent) if parent else -1)
    template.update(parents=parents, root_index=names.index("Hips"), basis_from_kimodo=IDENTITY)
    for name, joint in calibration["joints"].items():
        joint["id"] = "native-" + name
    return calibration, template


def snapshot(calibration, *, rotations=None, offset=None, frames=None):
    rotations, offset = rotations or {}, offset or [0, 0, 0]
    frames = [0, 1] if frames is None else frames
    rows = []
    for frame in frames:
        joints = {
            name: {"id": value["id"], "position": points._add(value["position"], offset),
                   "rotation": points._mm(rotations.get(name, IDENTITY), value["rotation"])}
            for name, value in calibration["joints"].items()
        }
        rows.append({"frame": frame, "joints": joints,
                     "points": {name: points._add(value["position"], offset)
                                for name, value in calibration["points"].items()}})
    value = {"format": "cascy-keypose-snapshot-v1", "scene_path": "source.casc",
             "scene_sha256": "saved-file", "target_fingerprint": calibration["fingerprint"],
             "calibration_sha256": keyposes._digest(calibration), "units": "centimeters",
             "up_axis": "Y", "fps": 30, "stored_frames": max(frames) + 1,
             "captured_playhead": 0, "poses": rows}
    value["capture_sha256"] = keyposes._digest(value)
    return value


def motion_from_constraints(template, converted):
    motion = copy.deepcopy(template)
    basis = template["basis_from_kimodo"]
    motion["num_frames"] = len(converted["global_positions"])
    motion["positions"] = [[points._scale(points._mv(basis, p), 100) for p in frame]
                           for frame in converted["global_positions"]]
    motion["rotations"] = [[points._mm(points._mm(basis, r), points._transpose(basis)) for r in frame]
                          for frame in converted["global_rot_mats"]]
    return motion


class KeyposeConversionTests(unittest.TestCase):
    def test_neutral_pose_roundtrip_does_not_change_native_points(self):
        calibration, template = fixture()
        captured = snapshot(calibration)
        constraints = keyposes.convert(captured, calibration, template)
        plan = keyposes.build_return_plan(calibration, motion_from_constraints(template, constraints), captured)
        self.assertLess(plan["keypose_deviation"]["max_point_error_cm"], 1e-8)

    def test_absolute_root_offset_and_height_survive_without_floor_snap(self):
        calibration, template = fixture()
        captured = snapshot(calibration, offset=[30, 17, 205])
        constraints = keyposes.convert(captured, calibration, template)
        plan = keyposes.build_return_plan(calibration, motion_from_constraints(template, constraints), captured)
        self.assertLess(plan["keypose_deviation"]["max_point_error_cm"], 1e-8)
        self.assertEqual(plan["ground_shift"], 0)
        self.assertEqual([round(x, 6) for x in plan["world_root_offset_cm"]], [30, 17, 205])

    def test_different_body_proportions_scale_root_without_stretching_soma(self):
        calibration, template = fixture()
        for group in (calibration["joints"], calibration["points"]):
            for value in group.values():
                value["position"] = points._scale(value["position"], 2)
        captured = snapshot(calibration, offset=[0, 0, 40])
        constraints = keyposes.convert(captured, calibration, template)
        self.assertAlmostEqual(constraints["height_scale"], 2)
        self.assertAlmostEqual(constraints["canonicalization"]["root_origin_m"][2], 0.2)
        self.assertEqual(constraints["global_positions"][0][0][2], 0.0)
        plan = keyposes.build_return_plan(calibration, motion_from_constraints(template, constraints), captured)
        self.assertLess(plan["keypose_deviation"]["max_point_error_cm"], 1e-8)

    def test_world_turn_and_native_rest_rotation_are_not_confused(self):
        calibration, template = fixture()
        for value in calibration["joints"].values():
            value["rotation"] = YAW
        captured = snapshot(calibration, rotations={name: YAW for name in calibration["joints"]})
        center = calibration["joints"]["pelvis"]["position"]
        for pose in captured["poses"]:
            for group in (pose["joints"],):
                for value in group.values():
                    value["position"] = points._add(center, points._mv(YAW, points._sub(value["position"], center)))
            for name, value in pose["points"].items():
                pose["points"][name] = points._add(center, points._mv(YAW, points._sub(value, center)))
        captured["capture_sha256"] = keyposes._digest({k: v for k, v in captured.items() if k != "capture_sha256"})
        constraints = keyposes.convert(captured, calibration, template)
        self.assertAlmostEqual(constraints["canonicalization"]["first_heading_radians"], math.pi / 2)
        for matrix in constraints["global_rot_mats"][0]:
            self.assertLess(max(abs(matrix[i][j] - IDENTITY[i][j]) for i in range(3) for j in range(3)), 1e-8)
        plan = keyposes.build_return_plan(calibration, motion_from_constraints(template, constraints), captured)
        self.assertLess(plan["keypose_deviation"]["max_point_error_cm"], 1e-8)

    def test_wrist_twist_roundtrip_keeps_the_additional_point(self):
        calibration, template = fixture()
        roll = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        captured = snapshot(calibration, rotations={"hand_l": roll})
        for pose in captured["poses"]:
            center = pose["points"]["hand_MainPoint_l"]
            pose["points"]["hand_AdditionalPoint_l"] = points._add(center, [0, -3, 0])
        captured["capture_sha256"] = keyposes._digest({k: v for k, v in captured.items() if k != "capture_sha256"})
        constraints = keyposes.convert(captured, calibration, template)
        plan = keyposes.build_return_plan(calibration, motion_from_constraints(template, constraints), captured)
        self.assertLess(plan["keypose_deviation"]["max_point_error_cm"], 1e-8)

    def test_world_basis_roundtrip_keeps_native_world_positions(self):
        calibration, template = fixture()
        template["basis_from_kimodo"] = YAW
        template["rest_positions"] = [points._mv(YAW, value) for value in template["rest_positions"]]
        captured = snapshot(calibration, offset=[13, 2, 57])
        constraints = keyposes.convert(captured, calibration, template)
        plan = keyposes.build_return_plan(calibration, motion_from_constraints(template, constraints), captured)
        self.assertLess(plan["keypose_deviation"]["max_point_error_cm"], 1e-8)

    def test_a_pose_source_neutral_converts_native_t_pose_without_arm_offset(self):
        calibration, template = fixture()
        index = {name: i for i, name in enumerate(template["joint_names"])}
        center = template["rest_positions"][index["LeftArm"]]
        c = math.sqrt(0.5)
        tilt = [[c, c, 0], [-c, c, 0], [0, 0, 1]]
        for name in ("LeftForeArm", "LeftHand", "LeftHandMiddle1"):
            i = index[name]
            template["rest_positions"][i] = points._add(
                center, points._mv(tilt, points._sub(template["rest_positions"][i], center)))
        captured = snapshot(calibration)
        constraints = keyposes.convert(captured, calibration, template)
        plan = keyposes.build_return_plan(calibration, motion_from_constraints(template, constraints), captured)
        self.assertLess(plan["keypose_deviation"]["max_point_error_cm"], 1e-8)

    def test_pelvis_relative_soma_neutral_becomes_floor_relative_for_generation(self):
        calibration, template = fixture()
        template["rest_positions"] = [points._sub(value, [0, 100, 0]) for value in template["rest_positions"]]
        captured = snapshot(calibration)
        constraints = keyposes.convert(captured, calibration, template)
        self.assertAlmostEqual(constraints["canonicalization"]["canonical_floor_offset_m"], 1.02)
        foot_indices = [template["joint_names"].index(name) for name in
                        ("LeftFoot", "LeftToeBase", "LeftToeEnd", "RightFoot", "RightToeBase", "RightToeEnd")]
        minimum = min(constraints["global_positions"][0][i][1] for i in foot_indices)
        self.assertAlmostEqual(minimum, 0)
        self.assertAlmostEqual(constraints["global_positions"][0][0][1], 1.02)
        plan = keyposes.build_return_plan(calibration, motion_from_constraints(template, constraints), captured)
        self.assertLess(plan["keypose_deviation"]["max_point_error_cm"], 1e-8)

    def test_first_generation_frame_requires_a_captured_anchor(self):
        calibration, template = fixture()
        with self.assertRaisesRegex(ValueError, "anchor frame zero"):
            keyposes.convert(snapshot(calibration, frames=[20, 40]), calibration, template,
                              start_frame=0, num_frames=50)

    def test_npz_arrays_preserve_frames_and_refuse_overwriting(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is required only for the NPZ writer integration")
        calibration, template = fixture()
        payload = keyposes.convert(snapshot(calibration), calibration, template)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constraints.npz"
            result = keyposes.write_npz(payload, path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["frame_indices"].dtype, np.dtype("int64"))
                self.assertEqual(data["frame_indices"].tolist(), [0, 1])
                self.assertEqual(data["global_positions"].shape, (2, len(template["joint_names"]), 3))
                self.assertEqual(data["global_rot_mats"].shape, (2, len(template["joint_names"]), 3, 3))
            self.assertEqual(result["sha256"], keyposes._file_sha(path))
            with self.assertRaisesRegex(ValueError, "new"):
                keyposes.write_npz(payload, path)

    def test_frame_interval_is_rebased_without_changing_source_frames(self):
        calibration, template = fixture()
        captured = snapshot(calibration, frames=[20, 40, 79])
        constraints = keyposes.convert(captured, calibration, template)
        self.assertEqual(constraints["frame_indices"], [0, 20, 59])
        self.assertEqual(constraints["num_frames"], 60)
        self.assertEqual(constraints["source_frames"], [20, 40, 79])

    def test_sparse_keypose_conversion_roundtrip_reports_every_anchor(self):
        calibration, template = fixture()
        captured = snapshot(calibration, frames=[0, 30, 60, 90, 119])
        report = keyposes.conversion_roundtrip_report(captured, calibration, template)
        self.assertTrue(report["accepted"])
        self.assertEqual([row["frame"] for row in report["frames"]], [0, 30, 60, 90, 119])
        self.assertLess(report["max_point_error_cm"], 1e-8)

    def test_snapshot_tampering_and_changed_calibration_are_rejected(self):
        calibration, template = fixture()
        captured = snapshot(calibration)
        captured["poses"][0]["points"]["pelvis_MainPoint"][0] += 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            keyposes.convert(captured, calibration, template)
        captured = snapshot(calibration)
        calibration["joints"]["pelvis"]["position"][0] += 1
        with self.assertRaisesRegex(ValueError, "calibration changed"):
            keyposes.convert(captured, calibration, template)

    def test_invalid_frames_and_reflected_rotations_are_rejected(self):
        calibration, template = fixture()
        for frames in ([1, 0], [0, 0], [False, 1], [0, 120]):
            with self.subTest(frames=frames):
                captured = snapshot(calibration, frames=frames)
                captured["stored_frames"] = 120
                captured["capture_sha256"] = keyposes._digest({k: v for k, v in captured.items() if k != "capture_sha256"})
                with self.assertRaisesRegex(ValueError, "frames"):
                    keyposes.convert(captured, calibration, template)
        captured = snapshot(calibration)
        captured["poses"][0]["joints"]["pelvis"]["rotation"] = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        captured["capture_sha256"] = keyposes._digest({k: v for k, v in captured.items() if k != "capture_sha256"})
        with self.assertRaisesRegex(ValueError, "reflected"):
            keyposes.convert(captured, calibration, template)

    def test_inconsistent_soma_parent_tree_is_rejected(self):
        calibration, template = fixture()
        template["parents"][1] = len(template["parents"]) - 1
        with self.assertRaisesRegex(ValueError, "acyclic"):
            keyposes.convert(snapshot(calibration), calibration, template)

    def test_observed_comparison_reports_drift_instead_of_accepting_it(self):
        calibration, _ = fixture()
        expected = snapshot(calibration)
        actual = snapshot(calibration, offset=[0, 0, 2])
        report = keyposes.compare_captures(expected, actual, position_tolerance_cm=1)
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(report["max_point_error_cm"], 2)
        self.assertAlmostEqual(report["max_joint_error_cm"], 2)
        self.assertTrue(keyposes.compare_captures(expected, actual, position_tolerance_cm=2)["accepted"])

    def test_observed_comparison_measures_actual_joint_rotations(self):
        calibration, _ = fixture()
        expected = snapshot(calibration)
        actual = snapshot(calibration, rotations={"hand_l": YAW})
        report = keyposes.compare_captures(expected, actual)
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(report["max_joint_rotation_error_degrees"], 90)
        self.assertEqual(report["frames"][0]["worst_rotated_joint"], "hand_l")

    def test_finger_articulation_does_not_reject_preserved_body_keyposes(self):
        calibration, _ = fixture()
        expected = snapshot(calibration)
        actual = snapshot(calibration, rotations={"f_middle1_l": YAW})
        for pose in actual["poses"]:
            pose["joints"]["f_middle1_l"]["position"][1] -= 3
        actual["capture_sha256"] = keyposes._digest({k: v for k, v in actual.items() if k != "capture_sha256"})
        report = keyposes.compare_captures(expected, actual)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["max_joint_error_cm"], 0)
        self.assertEqual(report["max_joint_rotation_error_degrees"], 0)
        self.assertEqual(report["excluded_joints"], ["f_middle1_l", "f_middle1_r"])
        self.assertIn("finger articulation is excluded", report["scope"])


class NativeCaptureTests(unittest.TestCase):
    def test_capture_reads_stored_joint_frames_without_moving_playhead(self):
        calibration, _ = fixture()
        mapping = points.MAPPING
        class Rotation:
            def to_rotation_matrix(self):
                return self
            def tolist(self):
                return IDENTITY
        class Data:
            def get_animation_size(self):
                return 120
            def get_data_value(self, key, frame):
                name, channel = key
                if channel == "global_rotation":
                    return Rotation()
                group = calibration["points"] if channel == "point" else calibration["joints"]
                return points._add(group[name]["position"], [0, 0, frame])
        scene = SimpleNamespace(
            model_viewer=lambda: SimpleNamespace(get_object_name=lambda obj: obj),
            data_viewer=lambda: Data(), get_current_frame=lambda _flag: 7,
        )
        rig = SimpleNamespace(fingerprint=calibration["fingerprint"], joints=set(calibration["joints"]),
                              controls={name: name for name in calibration["points"]},
                              bindings={name: SimpleNamespace(position_data=(name, "point"))
                                        for name in calibration["points"]})
        characters = SimpleNamespace(resolve_character=lambda _scene, **_kwargs: rig,
                                     _parent=lambda _scene, obj: points.PARENTS[obj] if obj in mapping else None,
                                     _transform_data=lambda _scene, obj, channel: (obj, channel))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.casc"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("scene/scene_settings.json", json.dumps({"Data": {"FPS": 30, "UnitLength": "cm"}}))
            view = SimpleNamespace(get_path_name=lambda: str(path), domain_scene=lambda: scene)
            with patch.dict("sys.modules", {"mcp_for_cascadeur.host": SimpleNamespace(characters=characters)}):
                result = keyposes.capture(view, [0, 30, 119], calibration=calibration, expected_scene_path=path)
                self.assertEqual(result["captured_playhead"], 7)
                self.assertEqual(result["poses"][1]["joints"]["pelvis"]["position"], [0, 100, 30])
                self.assertEqual(result["scene_sha256"], keyposes._file_sha(path))
                with self.assertRaisesRegex(ValueError, "differs"):
                    keyposes.capture(view, [0, 30], calibration=calibration, expected_scene_path=path.with_name("other.casc"))


if __name__ == "__main__":
    unittest.main()
