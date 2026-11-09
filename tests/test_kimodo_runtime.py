"""Checks for the optional Kimodo integration without an inference environment."""

import ast
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations" / "kimodo"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, INTEGRATION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_module("runtime_service")
patcher = load_module("runtime_patch")
retarget = load_module("prepare_retarget_motion")


class GenerationRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def request(self, **kwargs):
        return runtime.validate_generation_request({"prompt": "A person walks forward.", **kwargs}, self.root)

    def test_text_only_request_has_safe_defaults(self):
        request = self.request()
        self.assertEqual(request["constraints"], "")
        self.assertEqual((request["num_frames"], request["fps"]), (120, 30.0))
        self.assertEqual(request["num_samples"], 1)
        self.assertEqual(request["model_name"], "Kimodo-SOMA-RP-v1.2")
        for key in ("output", "output_fbx", "log_path"):
            self.assertTrue(Path(request[key]).is_relative_to(self.root))

    def test_output_cannot_escape_exchange(self):
        with self.assertRaisesRegex(ValueError, "inside"):
            self.request(output_fbx="../outside.fbx")

    def test_existing_animation_is_not_overwritten(self):
        target = self.root / "existing.fbx"
        target.write_bytes(b"existing animation")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.request(output_fbx=str(target))
        self.assertEqual(target.read_bytes(), b"existing animation")

    def test_frame_rate_changes_require_resampling(self):
        with self.assertRaisesRegex(ValueError, "30 fps"):
            self.request(fps=60)

    def test_invalid_limits_and_samples_are_rejected(self):
        for changes in (
            {"num_frames": True},
            {"num_frames": 0},
            {"num_samples": 1, "sample_index": 1},
            {"text_weight": float("nan")},
            {"seed": -1},
            {"dataset": "unknown"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.request(**changes)

    def test_candidates_have_distinct_paths_and_seeds_including_uint32_wrap(self):
        request = self.request(num_samples=3, sample_index=1, seed=2**32 - 1)
        candidates = runtime.candidate_payloads(request)
        self.assertEqual([row["seed"] for row in candidates], [2**32 - 1, 0, 1])
        self.assertEqual([row["num_samples"] for row in candidates], [1, 1, 1])
        self.assertEqual(len({row["output"] for row in candidates}), 3)
        self.assertEqual(len({row["output_fbx"] for row in candidates}), 3)
        self.assertTrue(candidates[0]["output"].endswith(".candidate-01.npz"))

    def test_single_candidate_preserves_legacy_paths(self):
        request = self.request()
        candidate, = runtime.candidate_payloads(request)
        self.assertEqual(candidate["output"], request["output"])
        self.assertEqual(candidate["output_fbx"], request["output_fbx"])

    def test_random_seed_is_resolved_once_and_recorded(self):
        with patch.object(runtime.secrets, "randbits", return_value=123):
            request = self.request(seed=None, num_samples=2)
        self.assertEqual([row["seed"] for row in runtime.candidate_payloads(request)], [123, 124])

    def test_existing_candidate_does_not_get_overwritten(self):
        (self.root / "clip.candidate-02.npz").write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "Candidate.*already exists"):
            self.request(num_samples=2, output="clip.npz")

    def test_existing_manifest_does_not_get_overwritten(self):
        (self.root / "clip.json").write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "manifest already exists"):
            self.request(output_fbx="clip.fbx")

    def test_manifest_preserves_frame_rate_and_fbx_identity(self):
        request = self.request(num_frames=2)
        fbx = Path(request["output_fbx"])
        fbx.write_bytes(b"fbx animation")
        npz = Path(request["output"])
        npz.write_bytes(b"npz animation")
        motion = {"root_positions": [[0, 1, 0], [0, 1, 1]], "local_rot_mats": [[], []]}
        encoder = {"kind": "llm2vec", "quantization": "4bit"}
        checkpoint = {"name": "Kimodo-SOMA-RP-v1.2", "weights_sha256": runtime.V12_WEIGHTS_SHA256}
        with (
            patch.dict(os.environ, {"KIMODO_EXCHANGE_ROOT": str(self.root)}),
            patch.object(runtime, "checkpoint_details", return_value=checkpoint),
        ):
            artifacts = runtime.write_motion_manifest(request, motion, "kimodo-soma-rp-v1.2", encoder)
        manifest = json.loads((self.root / artifacts["manifest"]).read_text())
        self.assertEqual(manifest["sha256"], hashlib.sha256(fbx.read_bytes()).hexdigest())
        self.assertEqual(manifest["npz_sha256"], hashlib.sha256(npz.read_bytes()).hexdigest())
        self.assertIsNone(manifest["constraints_sha256"])
        self.assertEqual(manifest["fps"], 30.0)
        self.assertEqual(manifest["num_frames"], 2)
        self.assertTrue(manifest["root_motion"])
        self.assertEqual(manifest["root_displacement_m"], [0.0, 0.0, 1.0])
        self.assertEqual(manifest["model_name"], checkpoint["name"])
        self.assertEqual(manifest["model_checkpoint"], checkpoint)
        self.assertEqual(manifest["candidate_index"], 0)
        self.assertEqual(manifest["diffusion_steps"], request["diffusion_steps"])

    def test_request_cannot_silently_use_another_backend_version(self):
        with self.assertRaisesRegex(ValueError, "running backend"):
            self.request(model_name="Kimodo-SOMA-RP-v1")

    def test_legacy_environment_cannot_change_the_checkpoint(self):
        for version in ("v1", "v1.1", "latest", "../../v1"):
            with self.subTest(version=version), patch.dict(os.environ, {"KIMODO_MODEL_VERSION": version}):
                self.assertEqual(self.request()["model_name"], "Kimodo-SOMA-RP-v1.2")

    def test_only_the_fixed_rp_checkpoint_is_available(self):
        for dataset in ("SEED", "unknown"):
            with self.subTest(dataset=dataset), self.assertRaisesRegex(ValueError, "Only the RP dataset"):
                self.request(dataset=dataset)
        for name in ("Kimodo-SOMA-RP-v1", "Kimodo-SOMA-RP-v1.1", "Kimodo-G1-RP-v1"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "only Kimodo-SOMA-RP-v1.2"):
                self.request(model_name=name)

    def test_manifest_rejects_wrong_loaded_model(self):
        request = self.request(num_frames=2)
        motion = {"root_positions": [[0, 1, 0], [0, 1, 1]], "local_rot_mats": [[], []]}
        with patch.object(runtime, "checkpoint_details", return_value={"name": "Kimodo-SOMA-RP-v1.2"}):
            with self.assertRaisesRegex(RuntimeError, "loaded model"):
                runtime.write_motion_manifest(request, motion, "kimodo-soma-rp-v1", {})

    def test_manifest_rejects_truncated_generation(self):
        request = self.request(num_frames=120)
        motion = {"root_positions": [[0, 1, 0]], "local_rot_mats": [[]]}
        with self.assertRaisesRegex(RuntimeError, "frame count"):
            runtime.write_motion_manifest(request, motion, "model", {})


class CandidateSetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.environment = patch.dict(os.environ, {"KIMODO_EXCHANGE_ROOT": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.request = runtime.validate_generation_request({
            "prompt": "A person walks forward", "num_samples": 3, "sample_index": 1,
        }, self.root)
        self.job = SimpleNamespace(emit_status=lambda message: None)

    def save(self, candidate, motion):
        relative = {}
        for name in ("output", "output_fbx"):
            path = Path(candidate[name])
            path.write_bytes(str(motion).encode())
            relative[name] = path.relative_to(self.root).as_posix()
        manifest = Path(candidate["output_fbx"]).with_suffix(".json")
        manifest.write_text(json.dumps({"seed": candidate["seed"]}))
        relative["manifest"] = manifest.relative_to(self.root).as_posix()
        relative["log_path"] = "job.log"
        return relative

    def test_all_candidates_are_saved_and_default_alias_is_requested_index(self):
        sequence = []

        def generate(candidate):
            sequence.append(("generate", candidate["candidate_index"], candidate["num_samples"]))
            return candidate["seed"]

        def save(candidate, motion):
            sequence.append(("save", candidate["candidate_index"], candidate["num_samples"]))
            return self.save(candidate, motion)

        artifacts = runtime.generate_candidate_set(self.request, self.job, generate, save)
        self.assertEqual(sequence, [
            (action, index, 1) for index in range(3) for action in ("generate", "save")
        ])
        self.assertEqual(len(artifacts["candidates"]), 3)
        self.assertEqual(artifacts["output"], artifacts["candidates"][1]["output"])
        record = json.loads(Path(self.request["candidates_manifest"]).read_text())
        self.assertEqual(record["status"], "completed")
        self.assertEqual([row["seed"] for row in record["candidates"]], [42, 43, 44])
        self.assertEqual(record["selection_method"], "requested_index")
        for row in record["candidates"]:
            self.assertTrue((self.root / row["output"]).is_file())
            self.assertTrue((self.root / row["output_fbx"]).is_file())

    def test_failure_preserves_finished_candidate_and_durable_index(self):
        def generate(candidate):
            if candidate["candidate_index"] == 1:
                raise RuntimeError("GPU generation failed")
            return candidate["seed"]

        with self.assertRaisesRegex(RuntimeError, "GPU generation failed"):
            runtime.generate_candidate_set(self.request, self.job, generate, self.save)
        record = json.loads(Path(self.request["candidates_manifest"]).read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(len(record["candidates"]), 1)
        self.assertEqual(len(self.job.artifacts["candidates"]), 1)
        self.assertTrue((self.root / record["candidates"][0]["output"]).is_file())
        self.assertNotIn("output", self.job.artifacts)


@unittest.skipIf(np is None, "Full-body NPZ checks require NumPy")
class ConstraintValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name).resolve() / "constraints.npz"

    def check(self, **changes):
        data = {
            "frame_indices": np.array([0, 119], dtype=np.int64),
            "global_positions": np.zeros((2, 30, 3), dtype=np.float32),
            "global_rot_mats": np.tile(np.eye(3), (2, 30, 1, 1)),
        }
        data.update(changes)
        np.savez(self.path, **data)
        return runtime.validate_constraints(self.path, 120)

    def test_valid_full_body_pose_targets(self):
        self.check()

    def test_api_pins_the_validated_input_and_checks_the_capture_hash(self):
        digest = self.check()
        payload = runtime.validate_generation_request({
            "prompt": "walk", "constraints": str(self.path), "constraints_sha256": digest,
        }, self.path.parent)
        self.assertEqual(payload["constraints_sha256"], digest)
        with self.assertRaisesRegex(ValueError, "captured checksum"):
            runtime.validate_generation_request({
                "prompt": "walk", "constraints": str(self.path), "constraints_sha256": "0" * 64,
            }, self.path.parent)

    def test_model_load_rejects_mutation_after_acceptance(self):
        digest = self.check()
        self.check(global_positions=np.ones((2, 30, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "changed after"):
            runtime.load_constraints(self.path, expected_sha256=digest, num_frames=120)

    def test_model_load_parses_the_same_bytes_that_were_hashed(self):
        digest = self.check()
        parse = runtime._constraint_arrays
        def mutate_during_parse(raw, num_frames):
            self.path.write_bytes(b"changed after the bounded read")
            return parse(raw, num_frames)
        with patch.object(runtime, "_constraint_arrays", side_effect=mutate_during_parse):
            frames, positions, rotations = runtime.load_constraints(
                self.path, expected_sha256=digest, num_frames=120)
        self.assertEqual(frames.tolist(), [0, 119])
        self.assertTrue(np.all(positions == 0))
        self.assertEqual(rotations.shape, (2, 30, 3, 3))

    def test_constraints_reader_bounds_compressed_bytes_and_declared_arrays(self):
        self.check()
        with patch.object(runtime, "MAX_CONSTRAINT_BYTES", 10), self.assertRaisesRegex(ValueError, "64 MiB"):
            runtime.validate_constraints(self.path, 120)
        import zipfile
        with zipfile.ZipFile(self.path, "w") as archive:
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, {
                "shape": (2**32,), "fortran_order": False, "descr": "<f8",
            })
            archive.writestr("frame_indices.npy", header.getvalue())
        with self.assertRaisesRegex(ValueError, "bounded numeric"):
            runtime.validate_constraints(self.path, 120)

    def test_manifest_records_pinned_input_and_rejects_a_changed_constraint_file(self):
        digest = self.check()
        payload = runtime.validate_generation_request({"prompt": "walk", "constraints": str(self.path)},
                                                       self.path.parent)
        Path(payload["output"]).write_bytes(b"generated npz")
        Path(payload["output_fbx"]).write_bytes(b"generated fbx")
        motion = {"root_positions": [[0, 1, 0]] * 120, "local_rot_mats": [[]] * 120}
        with patch.dict(os.environ, {"KIMODO_EXCHANGE_ROOT": str(self.path.parent)}), \
                patch.object(runtime, "checkpoint_details", return_value={"name": runtime.MODEL_NAME}):
            result = runtime.write_motion_manifest(payload, motion, runtime.MODEL_NAME, {})
            manifest = json.loads((self.path.parent / result["manifest"]).read_text())
            self.assertEqual(manifest["constraints_sha256"], digest)
            self.path.write_bytes(b"changed input")
            with self.assertRaisesRegex(ValueError, "changed after"):
                runtime.write_motion_manifest(payload, motion, runtime.MODEL_NAME, {})

    def test_rejects_duplicate_out_of_range_or_noninteger_frames(self):
        for frames in ([0, 0], [0, 120], [-1, 119], [0.0, 119.0]):
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "frames"):
                self.check(frame_indices=np.array(frames))

    def test_rejects_incoherent_nonfinite_and_reflected_transforms(self):
        for changes in (
            {"global_positions": np.zeros((2, 43, 3))},
            {"global_positions": np.full((2, 30, 3), np.nan)},
            {"global_rot_mats": np.zeros((2, 30, 3, 3))},
            {"global_rot_mats": np.tile(np.diag([-1, 1, 1]), (2, 30, 1, 1))},
        ):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.check(**changes)


class EncoderHealthTests(unittest.TestCase):
    def response(self, encoder):
        return io.BytesIO(json.dumps({"ok": True, "service": "text-encoder", "encoder": encoder}).encode())

    def test_healthy_real_encoder_is_accepted(self):
        details = {
            "kind": "llm2vec",
            "fallback": False,
            "embedding_dim": 4096,
            "quantization": "4bit",
            "model_path": "/models/merged",
        }
        with patch.object(runtime.urllib.request, "urlopen", return_value=self.response(details)):
            self.assertEqual(runtime.verify_text_encoder_health("http://encoder:9550"), details)

    def test_hash_fallback_cannot_pass_health(self):
        details = {
            "kind": "hash",
            "fallback": True,
            "embedding_dim": 4096,
            "quantization": "4bit",
            "model_path": "/models/merged",
        }
        with patch.object(runtime.urllib.request, "urlopen", return_value=self.response(details)):
            with self.assertRaisesRegex(RuntimeError, "hash fallback"):
                runtime.verify_text_encoder_health("http://encoder:9550")

    def test_requested_quantization_must_be_really_loaded(self):
        cls = type("LLM2VecEncoder", (), {})
        encoder = cls()
        encoder.model = SimpleNamespace(model=SimpleNamespace(is_loaded_in_4bit=False))
        encoder.quantization = "4bit"
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            runtime.encoder_details(encoder)

    def test_random_vector_class_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "real LLM2Vec"):
            runtime.encoder_details(SimpleNamespace())


class CheckpointIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.environment = patch.dict(os.environ, {"CHECKPOINT_DIR": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(runtime._checkpoint_sha256.cache_clear)

    def checkpoint(self, name):
        folder = self.root / name
        files = [folder / "config.yaml", folder / "model.safetensors"]
        files.extend(
            folder / "stats/motion" / area / (stat + ".npy")
            for area in ("body", "global_root", "local_root")
            for stat in ("mean", "std")
        )
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        return folder

    def test_v12_weights_must_match_pinned_nvidia_identity(self):
        self.checkpoint("Kimodo-SOMA-RP-v1.2")
        with self.assertRaisesRegex(RuntimeError, "pinned NVIDIA"):
            runtime.checkpoint_details()

    def test_checkpoint_reports_verified_version_revision_and_digest(self):
        self.checkpoint("Kimodo-SOMA-RP-v1.2")
        digest = hashlib.sha256(b"fixture").hexdigest()
        with patch.object(runtime, "V12_WEIGHTS_SHA256", digest):
            details = runtime.checkpoint_details()
            self.assertEqual(details["name"], "Kimodo-SOMA-RP-v1.2")
            self.assertEqual(details["repo_id"], "nvidia/Kimodo-SOMA-RP-v1.2")
            self.assertEqual(details["revision"], runtime.V12_REVISION)
            self.assertEqual(details["weights_sha256"], digest)
            runtime.checkpoint_details()
        self.assertEqual(runtime._checkpoint_sha256.cache_info().hits, 1)

    def test_stats_cannot_be_omitted(self):
        folder = self.checkpoint("Kimodo-SOMA-RP-v1.2")
        (folder / "stats/motion/body/std.npy").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing or empty"):
            runtime.checkpoint_details()

    def test_legacy_weights_cannot_replace_the_required_v12_checkpoint(self):
        self.checkpoint("Kimodo-SOMA-RP-v1")
        with patch.dict(os.environ, {"KIMODO_MODEL_VERSION": "v1"}):
            with self.assertRaisesRegex(RuntimeError, "Kimodo-SOMA-RP-v1.2"):
                runtime.checkpoint_details()


class RuntimePatchTests(unittest.TestCase):
    def test_patched_backend_generates_and_exports_every_candidate_sequentially(self):
        source = '''import numpy as np
def load_constraints(path):
    pass
def parse_args():
    parser.add_argument("--preload-dataset", choices=["RP", "SEED", ""])
class Job:
    def snapshot(self):
        return {
                "log_path": self.log_path,
        }
class BackendState:
    def get_or_load_model(self, dataset, emit_status):
        if True:
            model_name = api["kimodo_short_key_for_skeleton_dataset"]("SOMA", dataset)
    def snapshot(self):
        return {
                "loaded_datasets": loaded_datasets,
        }
def run_generation_job(state, job):
    pass
class RequestHandler:
    def do_GET(self):
        pass
    def do_POST(self):
        pass
'''
        patched = ast.parse(patcher.patch_backend(source))
        loader = next(node for node in patched.body if isinstance(node, ast.FunctionDef)
                      and node.name == "load_constraints")
        self.assertEqual([argument.arg for argument in loader.args.kwonlyargs], ["expected_sha256", "num_frames"])
        function = next(node for node in patched.body if isinstance(node, ast.FunctionDef)
                        and node.name == "run_generation_job")
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"KIMODO_EXCHANGE_ROOT": folder}):
            payload = runtime.validate_generation_request({
                "prompt": "A person walks", "num_frames": 2, "num_samples": 3, "diffusion_steps": 2,
            }, folder)
            events = []
            errors = []
            progress = []

            class Model:
                skeleton = object()

                def __call__(self, prompt, frames, **options):
                    events.append(("generate", options["num_samples"]))
                    list(options["progress_bar"](range(2)))
                    return {"local_rot_mats": [[], []], "root_positions": [[0, 1, 0], [0, 1, 1]]}

            def save_npz(path, motion):
                events.append(("npz", Path(path).name))
                Path(path).write_text("motion")

            def save_fbx(path, *args, **options):
                events.append(("fbx", Path(path).name))
                Path(path).write_text("fbx")

            scope = {
                "_kimodo_api": lambda: {
                    "seed_everything": lambda seed: events.append(("seed", seed)),
                    "save_kimodo_npz": save_npz, "save_motion_fbx": save_fbx,
                },
                "verify_text_encoder_health": lambda: {"kind": "llm2vec"},
                "emit_memory_snapshot": lambda *args: None,
                "np": SimpleNamespace(empty=lambda *args, **kwargs: SimpleNamespace(tolist=lambda: []),
                                      int64="int64", float32="float32", asarray=lambda value, **kwargs: value),
                "torch": SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda value: value,
                                         cuda=SimpleNamespace(is_available=lambda: False)),
                "json": json,
                "traceback": SimpleNamespace(format_exc=lambda: "traceback"),
                "select_sample": lambda output, index, count: output,
                "resolve_fbx_skeleton": lambda *args: object(),
                "write_motion_manifest": runtime.write_motion_manifest,
                "generate_candidate_set": runtime.generate_candidate_set,
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), "backend fixture", "exec"), scope)
            job = SimpleNamespace(
                payload=payload, job_id=1, _write_log=lambda message: None, emit_status=lambda message: None,
                emit_progress=lambda step, total: progress.append((step, total)),
                succeed=lambda: events.append(("done", 1)), fail=lambda *error: errors.append(error),
            )
            state = SimpleNamespace(
                device="cpu", get_or_load_model=lambda *args: (Model(), "Kimodo-SOMA-RP-v1.2"),
                finish_job=lambda identifier: events.append(("finished", identifier)),
            )
            with patch.object(runtime, "checkpoint_details", return_value={"name": "Kimodo-SOMA-RP-v1.2"}):
                scope["run_generation_job"](state, job)
            self.assertEqual(errors, [])
            self.assertEqual([event for event in events if event[0] == "seed"], [("seed", 42), ("seed", 43), ("seed", 44)])
            self.assertEqual([event for event in events if event[0] == "generate"], [("generate", 1)] * 3)
            self.assertEqual([event[0] for event in events], ["seed", "generate", "npz", "fbx"] * 3 + ["done", "finished"])
            self.assertEqual(progress, [(step, 6) for step in range(1, 7)])
            self.assertEqual(len(job.artifacts["candidates"]), 3)

    def test_registry_advertises_only_v12_and_resolves_its_base_alias(self):
        source = r"""import re
from dataclasses import dataclass
KIMODO_REPO_IDS = [
    "nvidia/Kimodo-SOMA-RP-v1",
    "nvidia/Kimodo-SOMA-SEED-v1",
    "nvidia/Kimodo-G1-RP-v1",
]
TMR_REPO_IDS = ["nvidia/TMR-SOMA-RP-v1"]
_REPO_NAME_PATTERN = re.compile(r"^(Kimodo|TMR)-([A-Za-z0-9]+)-(RP|SEED)-v(\d+)$")
@dataclass
class ModelInfo:
    repo_id: str
    short_key: str
    version: str
def _parse_repo_id(repo_id):
    match = _REPO_NAME_PATTERN.fullmatch(repo_id.split('/')[-1])
    if match is None:
        return None
    family, skeleton, dataset, version = match.groups()
    return ModelInfo(repo_id, f"{family}-{skeleton}-{dataset}".lower(), f"v{version}")
def _build_registry():
    return [], {}, []
MODEL_INFOS, MODEL_NAMES, AVAILABLE_MODELS = _build_registry()
FULL = re.compile(r"v(\d+)")
PARTIAL = re.compile(r"v(\d+)")
"""
        sorter = """def sort_VERSION(v):
        if v.startswith("v") and v[1:].isdigit():
            return int(v[1:])
        return 0
"""
        source += "".join(sorter.replace("VERSION", str(i)) for i in range(3))
        # dataclasses needs the defining module to be visible during execution.
        import types

        module = types.ModuleType("registry_fixture")
        with patch.dict(sys.modules, {"registry_fixture": module}):
            exec(compile(patcher.patch_model_registry(source), "registry fixture", "exec"), module.__dict__)
        names = module.MODEL_NAMES
        self.assertEqual(module.KIMODO_REPO_IDS, ["nvidia/Kimodo-SOMA-RP-v1.2"])
        self.assertEqual(module.TMR_REPO_IDS, [])
        self.assertEqual(len(module.MODEL_INFOS), 1)
        self.assertEqual(
            names,
            {
                "kimodo-soma-rp": "nvidia/Kimodo-SOMA-RP-v1.2",
                "kimodo-soma-rp-v1.2": "nvidia/Kimodo-SOMA-RP-v1.2",
            },
        )
        self.assertTrue(module.FULL.fullmatch("v1.2"))
        self.assertLess(module.sort_0("v1.2"), module.sort_0("v1.10"))

    def test_registry_patch_refuses_ambiguous_upstream_model_lists(self):
        for source in ("", "KIMODO_REPO_IDS = []\nKIMODO_REPO_IDS = []\n"):
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "assignment"):
                patcher.patch_model_registry(source)

    def test_fbx_wrapper_keeps_centimeter_scale_and_axis_rotation(self):
        properties = [
            ["P", [name, "", "", "", *values], "", []]
            for name, values in (
                ("Lcl Translation", [0, 0, 0]),
                ("Lcl Rotation", [0, 0, 180]),
                ("Lcl Scaling", [100, 100, 100]),
            )
        ]
        template = [
            [
                "Objects",
                [],
                "",
                [
                    [
                        "Model",
                        [1, "KimodoArmature::Model", "Root"],
                        "",
                        [
                            ["Properties70", [], "", properties],
                        ],
                    ]
                ],
            ]
        ]
        location, rotation, scale = runtime.fbx_wrapper_transform(template)
        self.assertEqual(location, [0, 0, 0])
        self.assertEqual(rotation, [0, 0, 180])
        self.assertEqual(scale, [100, 100, 100])

    def test_retarget_yaw_keeps_up_and_rotates_forward_explicitly(self):
        identity = retarget.yaw_basis(0)
        self.assertEqual(identity, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        basis = retarget.yaw_basis(180)
        self.assertEqual(basis[1], [0, 1, 0])
        self.assertAlmostEqual(basis[0][0], -1)
        self.assertAlmostEqual(basis[2][2], -1)

    def test_patch_rejects_ambiguous_upstream_source(self):
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            patcher.replace_once("original original", "original", "replacement")

    def test_method_patch_does_not_replace_another_class_method(self):
        source = "class First:\n    def run(self):\n        return 1\n\nclass Second:\n    def run(self):\n        return 2\n"
        changed = patcher.replace_function(source, "run", "    def run(self):\n        return 3", "First")
        scope = {}
        exec(compile(changed, "fixture", "exec"), scope)
        self.assertEqual(scope["First"]().run(), 3)
        self.assertEqual(scope["Second"]().run(), 2)

    def test_quantized_loading_disables_async_and_restores_state_on_error(self):
        source = (
            "import os\nfrom contextlib import contextmanager\n"
            "@contextmanager\ndef _quantized_load_without_allocator_warmup(enabled):\n    yield\n"
        )
        scope = {}
        exec(compile(patcher.patch_llm_loader(source), "loader fixture", "exec"), scope)

        def original_warmup():
            return "warmup"

        modeling = SimpleNamespace(caching_allocator_warmup=original_warmup)
        transformer = SimpleNamespace(modeling_utils=modeling)
        with patch.dict(sys.modules, {"transformers": transformer}):
            with patch.dict(os.environ, {"HF_DEACTIVATE_ASYNC_LOAD": "0"}):
                with self.assertRaisesRegex(RuntimeError, "loading failed"):
                    with scope["_quantized_load_without_allocator_warmup"](True):
                        self.assertEqual(os.environ["HF_DEACTIVATE_ASYNC_LOAD"], "1")
                        self.assertIsNone(modeling.caching_allocator_warmup())
                        raise RuntimeError("loading failed")
                self.assertEqual(os.environ["HF_DEACTIVATE_ASYNC_LOAD"], "0")
        self.assertIs(modeling.caching_allocator_warmup, original_warmup)

    def test_unquantized_loading_keeps_async_settings(self):
        source = (
            "import os\nfrom contextlib import contextmanager\n"
            "@contextmanager\ndef _quantized_load_without_allocator_warmup(enabled):\n    yield\n"
        )
        scope = {}
        exec(compile(patcher.patch_llm_loader(source), "loader fixture", "exec"), scope)
        with patch.dict(os.environ, {"HF_DEACTIVATE_ASYNC_LOAD": "0"}):
            with scope["_quantized_load_without_allocator_warmup"](False):
                self.assertEqual(os.environ["HF_DEACTIVATE_ASYNC_LOAD"], "0")


if __name__ == "__main__":
    unittest.main()
