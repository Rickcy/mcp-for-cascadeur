"""Check exchange validation and failure recovery without loading Cascadeur."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

BRIDGE_PATH = Path(__file__).resolve().parents[1] / "integrations" / "kimodo" / "cascadeur_bridge.py"
SPEC = importlib.util.spec_from_file_location("kimodo_cascadeur_bridge_test", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class KimodoCascadeurBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kimodo-bridge-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fbx = self.root / "generated.fbx"
        self.fbx.write_bytes(b"Kaydara FBX Binary  " + b"motion" * 100)
        self.manifest = self.root / "generated.json"
        self.metadata = {
            "fps": 30,
            "num_frames": 120,
            "sha256": hashlib.sha256(self.fbx.read_bytes()).hexdigest(),
            "text_encoder": {"kind": "llm2vec", "quantization": "4bit"},
            "root_motion": True,
        }
        self.save_manifest()
        self.cascy = self.root / "Cascy.casc"
        self.kimodo = self.root / "Kimodo.casc"
        for scene in (self.cascy, self.kimodo):
            with zipfile.ZipFile(scene, "w") as archive:
                archive.writestr(
                    "scene/scene_settings.json", json.dumps({"Data": {"FPS": 30, "UnitLength": "cm"}})
                )
        self.original_scene = self.cascy.read_bytes()

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.metadata), encoding="utf-8")

    def prepare(self):
        return bridge.prepare(self.root / "exchange", self.cascy, self.kimodo, self.manifest, self.fbx)

    def test_accepts_real_encoder_and_matching_generation_bytes(self):
        result = bridge.validate_generation(self.manifest, self.fbx)
        self.assertEqual(result["num_frames"], 120)
        self.assertEqual(result["text_encoder"]["kind"], "llm2vec")

    def test_hash_fallback_cannot_be_reported_as_text_generation(self):
        self.metadata["text_encoder"]["kind"] = "hash"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "real LLM2Vec"):
            self.prepare()
        self.assertFalse((self.root / "exchange").exists())

    def test_changed_fbx_is_rejected_before_creating_scene_copies(self):
        self.fbx.write_bytes(b"different animation" * 100)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.prepare()
        self.assertFalse((self.root / "exchange").exists())

    def test_timebase_and_frame_budget_are_explicit(self):
        for field, value in (("fps", 60), ("fps", True), ("num_frames", 120.0), ("num_frames", 241)):
            with self.subTest(field=field, value=value):
                metadata = copy.deepcopy(self.metadata)
                metadata[field] = value
                self.manifest.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ValueError):
                    bridge.validate_generation(self.manifest, self.fbx)

    def test_prepare_keeps_inputs_and_backup_separate_from_working_copy(self):
        prepared = self.prepare()
        directory = Path(prepared["receipt_path"]).parent
        self.assertEqual((directory / "before.casc").read_bytes(), self.cascy.read_bytes())
        (directory / "working.casc").write_bytes(b"a native edit")
        self.assertEqual((directory / "before.casc").read_bytes(), self.original_scene)
        self.assertEqual(self.cascy.read_bytes(), self.original_scene)
        self.assertEqual(self.kimodo.read_bytes(), self.original_scene)

    def test_native_timebase_must_match_generation(self):
        with zipfile.ZipFile(self.cascy, "w") as archive:
            archive.writestr(
                "scene/scene_settings.json", json.dumps({"Data": {"FPS": 60, "UnitLength": "cm"}})
            )
        with self.assertRaisesRegex(ValueError, "30 FPS"):
            self.prepare()
        self.assertFalse((self.root / "exchange").exists())

    def test_failed_native_write_is_not_replayed_on_retry(self):
        receipt = self.prepare()["receipt_path"]
        effects = []

        def uncertain_write(_record):
            effects.append("native write dispatched")
            raise TimeoutError("unknown completion")

        with self.assertRaises(TimeoutError):
            bridge._stage(receipt, "prepared", "imported", uncertain_write)
        record = json.loads(Path(receipt).read_text(encoding="utf-8"))
        self.assertEqual(record["phase"], "needs_review:imported")
        with self.assertRaisesRegex(RuntimeError, "inspect before retrying"):
            bridge._stage(receipt, "prepared", "imported", uncertain_write)
        self.assertEqual(effects, ["native write dispatched"])

    def test_receipt_cannot_redirect_owned_files_to_another_directory(self):
        receipt = Path(self.prepare()["receipt_path"])
        record = json.loads(receipt.read_text(encoding="utf-8"))
        record["directory"] = str(self.root)
        receipt.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "valid bridge"):
            bridge._read(receipt)

    def test_failed_preparation_cleans_its_exact_copies(self):
        with patch.object(bridge, "_sha", side_effect=[self.metadata["sha256"], OSError("disk failure")]):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.prepare()
        self.assertEqual(list((self.root / "exchange").iterdir()), [])
        self.assertEqual(self.cascy.read_bytes(), self.original_scene)

    def snapshot(self):
        return {"fingerprint": "same native rig", "positions": {"pelvis_MainPoint": [[0, 1, 0], [0, 1, 100]]}}

    def test_reopen_checks_motion_frames_not_only_first_pose(self):
        expected = self.snapshot()
        observed = copy.deepcopy(expected)
        observed["positions"]["pelvis_MainPoint"][1][2] += 0.2
        with self.assertRaisesRegex(RuntimeError, "coordinates changed"):
            bridge.compare_snapshots(expected, observed)

    def test_small_save_rounding_is_measured_and_accepted(self):
        expected = self.snapshot()
        observed = copy.deepcopy(expected)
        observed["positions"]["pelvis_MainPoint"][1][2] += 0.005
        result = bridge.compare_snapshots(expected, observed)
        self.assertTrue(result["point_coordinates_verified"])
        self.assertAlmostEqual(result["max_coordinate_error"], 0.005)

    def test_reopen_rejects_missing_coordinates_and_nonfinite_data(self):
        for values in ([0, 1], [0, 1, float("nan")]):
            with self.subTest(values=values):
                expected = self.snapshot()
                observed = copy.deepcopy(expected)
                observed["positions"]["pelvis_MainPoint"][1] = values
                with self.assertRaises(RuntimeError):
                    bridge.compare_snapshots(expected, observed)

    def test_reopen_rejects_changed_native_semantic_topology(self):
        expected = self.snapshot()
        observed = copy.deepcopy(expected)
        observed["fingerprint"] = "different rig"
        with self.assertRaisesRegex(RuntimeError, "topology"):
            bridge.compare_snapshots(expected, observed)


if __name__ == "__main__":
    unittest.main()
