"""Check pose transfer geometry and generated-motion fidelity without Cascadeur."""

import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "integrations" / "kimodo" / "point_retarget.py"
SPEC = importlib.util.spec_from_file_location("kimodo_point_retarget_test", MODULE_PATH)
retarget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retarget)
IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class KimodoPointRetargetTests(unittest.TestCase):
    def fixture(self):
        coordinates = {
            "pelvis": [0, 100, 0],
            "stomach": [0, 110, 0],
            "chest": [0, 125, 0],
            "neck": [0, 145, 0],
            "head": [0, 160, 0],
        }
        for side, sign in (("l", 1), ("r", -1)):
            for stem, position in {
                "clavicle": [5, 140, 0],
                "arm": [15, 140, 0],
                "forearm": [40, 140, 0],
                "hand": [65, 140, 0],
                "thigh": [10, 100, 0],
                "calf": [10, 55, 0],
                "foot": [10, 10, 0],
                "toe": [10, 4, 12],
                "f_middle1": [70, 140, 0],
            }.items():
                coordinates[stem + "_" + side] = [sign * position[0], position[1], position[2]]
        joints = {
            name: {"position": value, "parent": retarget.PARENTS.get(name), "rotation": IDENTITY}
            for name, value in coordinates.items()
        }
        points = {}
        for name in retarget.MAPPING:
            for suffix, offset in (("MainPoint", [0, 0, 0]), ("AdditionalPoint", [0, 0, 3])):
                point_name = (
                    name.split("_")[0]
                    + "_"
                    + suffix
                    + ("_" + name[-1] if name.endswith(("_l", "_r")) else "")
                )
                points[point_name] = {
                    "id": point_name,
                    "joint": name,
                    "position": retarget._add(coordinates[name], offset),
                }
        for side in ("l", "r"):
            name = "foot_Self0Point_" + side
            points[name] = {
                "id": name,
                "joint": "foot_" + side,
                "position": retarget._add(coordinates["foot_" + side], [0, -10, -3]),
            }
        calibration = {
            "format": "cascy-point-calibration-v1",
            "units": "centimeters",
            "up_axis": "Y",
            "fingerprint": "native-rig",
            "joints": joints,
            "points": points,
        }
        source = {retarget.MAPPING[name]: coordinates[name] for name in retarget.MAPPING}
        source["HeadEnd"] = [0, 170, 0]
        for side, prefix in (("l", "Left"), ("r", "Right")):
            source[prefix + "HandMiddle1"] = coordinates["f_middle1_" + side]
            source[prefix + "ToeEnd"] = retarget._add(
                coordinates["toe_" + side],
                retarget._sub(coordinates["toe_" + side], coordinates["foot_" + side]),
            )
        names = list(source)
        rest = [source[name] for name in names]
        motion = {
            "format": "kimodo-retarget-motion-v1",
            "fps": 30,
            "num_frames": 2,
            "units": "centimeters",
            "up_axis": "Y",
            "joint_names": names,
            "rest_positions": rest,
            "rest_global_rotations": [IDENTITY for _ in names],
            "positions": [copy.deepcopy(rest), [retarget._add(point, [0, 0, 20]) for point in rest]],
            "rotations": [[IDENTITY for _ in names] for _ in range(2)],
            "source_npz_sha256": "test-motion",
        }
        return calibration, motion

    def test_identical_bind_shapes_transfer_without_pose_distortion(self):
        calibration, motion = self.fixture()
        plan = retarget.build_plan(calibration, motion)
        for name, point in calibration["points"].items():
            self.assertLess(math.dist(plan["frames"][0]["points"][name], point["position"]), 1e-8)
        self.assertLess(plan["max_rigid_cluster_error"], 1e-8)

    def test_root_translation_is_generated_motion_not_a_new_gait(self):
        calibration, motion = self.fixture()
        plan = retarget.build_plan(calibration, motion)
        self.assertEqual(plan["root_displacement"], [0, 0, 20])
        self.assertEqual(plan["source_npz_sha256"], "test-motion")

    def test_neutral_root_at_zero_does_not_raise_character_by_a_metre(self):
        calibration, motion = self.fixture()
        motion["rest_positions"] = [retarget._sub(point, [0, 100, 0]) for point in motion["rest_positions"]]
        plan = retarget.build_plan(calibration, motion)
        self.assertAlmostEqual(plan["frames"][0]["points"]["pelvis_MainPoint"][1], 100)

    def test_world_turn_rotates_rigid_points_with_the_source_character(self):
        calibration, motion = self.fixture()
        rotation = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        root = motion["positions"][0][0]
        motion["positions"][1] = [
            retarget._add(root, retarget._mv(rotation, retarget._sub(point, root)))
            for point in motion["positions"][0]
        ]
        motion["rotations"][1] = [rotation for _ in motion["joint_names"]]
        plan = retarget.build_plan(calibration, motion)
        for name, point in calibration["points"].items():
            expected = retarget._add(root, retarget._mv(rotation, retarget._sub(point["position"], root)))
            self.assertLess(math.dist(plan["frames"][1]["points"][name], expected), 1e-8)

    def test_a_pose_source_does_not_leave_a_t_pose_offset_in_target_arms(self):
        calibration, motion = self.fixture()
        index = {name: i for i, name in enumerate(motion["joint_names"])}
        shoulder = motion["rest_positions"][index["LeftArm"]]
        cosine = math.sqrt(0.5)
        rotation = [[cosine, cosine, 0], [-cosine, cosine, 0], [0, 0, 1]]
        for name in ("LeftForeArm", "LeftHand", "LeftHandMiddle1"):
            i = index[name]
            motion["rest_positions"][i] = retarget._add(
                shoulder, retarget._mv(rotation, retarget._sub(motion["rest_positions"][i], shoulder))
            )
        motion["positions"] = [copy.deepcopy(motion["rest_positions"]) for _ in range(2)]
        plan = retarget.build_plan(calibration, motion)
        arm = plan["frames"][0]["points"]["arm_MainPoint_l"]
        elbow = plan["frames"][0]["points"]["forearm_MainPoint_l"]
        self.assertAlmostEqual(math.dist(arm, elbow), 25)
        self.assertAlmostEqual((arm[1] - elbow[1]) / (elbow[0] - arm[0]), 1)

    def test_source_wrist_roll_is_preserved_without_changing_cluster_size(self):
        calibration, motion = self.fixture()
        i = motion["joint_names"].index("LeftHand")
        motion["rotations"][1][i] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        plan = retarget.build_plan(calibration, motion)
        main = plan["frames"][1]["points"]["hand_MainPoint_l"]
        additional = plan["frames"][1]["points"]["hand_AdditionalPoint_l"]
        self.assertAlmostEqual(math.dist(main, additional), 3)
        self.assertAlmostEqual(additional[1] - main[1], -3)

    def test_reflected_or_invalid_rotation_input_is_rejected(self):
        calibration, motion = self.fixture()
        motion["rotations"][0][0] = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with self.assertRaisesRegex(ValueError, "reflected"):
            retarget.build_plan(calibration, motion)

    def test_hip_branch_uses_parent_rigid_frame_instead_of_source_hip_drop(self):
        calibration, motion = self.fixture()
        index = motion["joint_names"].index("LeftLeg")
        motion["rest_positions"][index][1] -= 8
        for frame in motion["positions"]:
            frame[index][1] -= 8
        plan = retarget.build_plan(calibration, motion, ground_to_floor=False)
        pelvis = plan["frames"][0]["points"]["pelvis_MainPoint"]
        hip = plan["frames"][0]["points"]["thigh_MainPoint_l"]
        self.assertAlmostEqual(hip[1], pelvis[1])
        self.assertAlmostEqual(math.dist(hip, pelvis), 10)

    def test_shared_elbow_anchor_satisfies_both_native_rigid_clusters(self):
        calibration, motion = self.fixture()
        points = calibration["points"]
        calibration["native_links"] = [
            {
                "box_id": "upper-arm",
                "main_point": points["arm_MainPoint_l"]["id"],
                "direction_point": points["forearm_MainPoint_l"]["id"],
                "additional_point": points["forearm_AdditionalPoint_l"]["id"],
            },
            {
                "box_id": "forearm",
                "main_point": points["forearm_MainPoint_l"]["id"],
                "direction_point": points["hand_MainPoint_l"]["id"],
                "additional_point": points["forearm_AdditionalPoint_l"]["id"],
            },
        ]
        index = {name: i for i, name in enumerate(motion["joint_names"])}
        motion["positions"][1][index["LeftHand"]] = [40, 115, 20]
        motion["positions"][1][index["LeftHandMiddle1"]] = [40, 110, 20]
        motion["rotations"][1][index["LeftForeArm"]] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        plan = retarget.build_plan(calibration, motion)
        self.assertEqual(plan["native_cluster_validation"]["native_clusters"], 2)
        self.assertLess(plan["max_rigid_cluster_error"], 1e-8)
        corrupted = copy.deepcopy(plan["frames"])
        center = corrupted[1]["points"]["forearm_MainPoint_l"]
        corrupted[1]["points"]["forearm_AdditionalPoint_l"] = retarget._add(center, [3, 0, 0])
        with self.assertRaisesRegex(ValueError, "Native rigid cluster"):
            retarget.validate_native_clusters(calibration, corrupted)

    def test_script_entry_point_defines_geometry_helpers_before_planning(self):
        calibration, motion = self.fixture()
        with tempfile.TemporaryDirectory(prefix="kimodo-point-cli-test-") as directory:
            directory = Path(directory)
            calibration_path, motion_path, output = (
                directory / name for name in ("calibration.json", "motion.json", "plan.json")
            )
            calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
            motion_path.write_text(json.dumps(motion), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(MODULE_PATH),
                    "plan",
                    "--calibration",
                    str(calibration_path),
                    "--motion",
                    str(motion_path),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(json.loads(completed.stdout)["ok"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["num_frames"], 2)


if __name__ == "__main__":
    unittest.main()
