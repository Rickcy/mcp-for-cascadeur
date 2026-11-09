"""Outcome handling and orchestration checks without Docker, Gemini or native writes."""

import contextlib
import hashlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "integrations/kimodo/pipeline.py"
SPEC = importlib.util.spec_from_file_location("kimodo_pipeline_test", PATH)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def review_settings(**options):
    settings = {"provider": "gemini", "model": "gemini-test", "media_transport": "video", "sampling_fps": 12,
                "base_url": "http://127.0.0.1:8000/v1", "max_total_frames": 96,
                "max_frames_per_video": 48, "frame_width": 384, "api_key": "fixture-secret-never-persist"}
    settings.update(options.get("config", {}))
    settings.update({key: value for key, value in options.items() if key in settings and value is not None})
    if settings["provider"] == "openai-compatible":
        settings["api_key"] = None
    return settings


def review_identity(**options):
    settings = review_settings(**options)
    fields = ["provider", "model", "media_transport", "sampling_fps"]
    if settings["provider"] == "openai-compatible":
        fields.append("base_url")
    if settings["media_transport"] == "frames":
        fields.extend(("max_total_frames", "max_frames_per_video", "frame_width"))
    return {key: settings[key] for key in fields}


def review_report(candidates, prompt, **options):
    return {"schema": "kimodo-visual-review-v1", "motion_prompt": prompt,
            "review_identity": review_identity(**options),
            "evidence": [{"id": item["id"], "media": [
                {"view": media["view"], "kind": "video", "sha256": pipeline._sha(media["path"])}
                for media in item["media"]]} for item in candidates]}


class GenerationTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.record = Path(self.temporary.name).resolve() / "generation.json"

    def test_submission_is_recorded_before_the_single_post(self):
        def request(url, route, payload):
            self.assertEqual(pipeline.read(self.record)["state"], "submitted")
            return {"ok": True, "job_id": 7}

        with patch.object(pipeline, "request_json", side_effect=request) as mocked:
            result = pipeline.submit_generation("http://127.0.0.1:9552", {"prompt": "walk"}, self.record)
        self.assertEqual(result["job_id"], 7)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(pipeline.read(self.record)["state"], "accepted")

    def test_lost_submission_response_is_not_retried(self):
        with patch.object(pipeline, "request_json", side_effect=TimeoutError) as mocked:
            with self.assertRaisesRegex(pipeline.PipelineError, "not repeated"):
                pipeline.submit_generation("http://127.0.0.1:9552", {"prompt": "walk"}, self.record)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(pipeline.read(self.record)["state"], "unknown")
        with patch.object(pipeline, "request_json") as mocked:
            with self.assertRaises(FileExistsError):
                pipeline.submit_generation("http://127.0.0.1:9552", {"prompt": "walk"}, self.record)
            mocked.assert_not_called()

    def test_poll_failure_never_resubmits_and_finished_partial_job_remains_readable(self):
        with patch.object(pipeline, "request_json", side_effect=TimeoutError) as mocked:
            with self.assertRaisesRegex(pipeline.PipelineError, "not resubmitted"):
                pipeline.wait_generation("http://127.0.0.1:9552", 3, poll_interval=0)
            self.assertEqual(mocked.call_args.args[1], "/jobs/3")
            self.assertEqual(mocked.call_count, 1)
        partial = {"job_id": 3, "done": True, "error": "generation failed",
                   "artifacts": {"candidates": [{"index": 0}]}}
        with patch.object(pipeline, "request_json", return_value={"ok": True, "job": partial}):
            self.assertEqual(pipeline.wait_generation("http://127.0.0.1:9552", 3), partial)

    def test_remote_credentials_redirect_targets_and_proxy_paths_are_rejected(self):
        for endpoint in ("https://127.0.0.1:9552", "http://remote:9552", "http://user:key@localhost:9552",
                         "http://localhost:9552/path", "http://localhost:9552?key=secret"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                pipeline._base_url(endpoint)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def candidate(self, index):
        npz = self.root / f"candidate-{index}.npz"
        fbx = self.root / f"candidate-{index}.fbx"
        manifest = fbx.with_suffix(".json")
        npz.write_bytes(b"npz")
        fbx.write_bytes(b"fbx" + bytes([index]))
        pipeline.write(manifest, {
            "format": "cascadeur-kimodo-motion-v1", "num_frames": 120, "fps": 30,
            "model_name": "Kimodo-SOMA-RP-v1.2", "seed": 42 + index, "candidate_index": index,
            "sha256": hashlib.sha256(fbx.read_bytes()).hexdigest(),
            "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            "constraints_sha256": None,
            "text_encoder": {"kind": "llm2vec", "fallback": False},
        })
        return {"index": index, "seed": 42 + index, "output": npz.name,
                "output_fbx": fbx.name, "manifest": manifest.name}

    def test_every_retained_animation_is_verified(self):
        rows = [self.candidate(0), self.candidate(1)]
        job = {"error": "", "artifacts": {"candidates": rows}}
        result = pipeline.verify_candidates(job, self.root, count=2, frames=120)
        self.assertEqual([item["id"] for item in result], ["candidate-01", "candidate-02"])
        (self.root / rows[1]["output_fbx"]).write_bytes(b"changed")
        with self.assertRaisesRegex(pipeline.PipelineError, "manifest"):
            pipeline.verify_candidates(job, self.root, count=2, frames=120)

    def test_missing_duplicate_and_escaping_candidates_are_rejected(self):
        first = self.candidate(0)
        for rows in ([first], [first, first]):
            with self.subTest(rows=len(rows)), self.assertRaises(pipeline.PipelineError):
                pipeline.verify_candidates({"artifacts": {"candidates": rows}}, self.root, count=2, frames=120)
        (self.root.parent / (self.root.name + "-outside.npz")).write_bytes(b"outside")
        outside = self.root.parent / (self.root.name + "-outside.npz")
        self.addCleanup(outside.unlink)
        with self.assertRaises(ValueError):
            pipeline._artifact(self.root, "../" + outside.name)

    def test_busy_or_unobservable_runtime_is_not_stopped(self):
        with patch.object(pipeline, "request_json", return_value={"ready": True, "busy": True}), \
                patch.object(pipeline, "runtime_action") as stop:
            with self.assertRaisesRegex(pipeline.PipelineError, "left running"):
                pipeline.release_runtime("http://127.0.0.1:9552", self.root)
            stop.assert_not_called()

    def test_constraint_and_npz_hashes_are_independent_generation_contracts(self):
        row = self.candidate(0)
        manifest_path = self.root / row["manifest"]
        manifest = pipeline.read(manifest_path)
        manifest["constraints_sha256"] = "a" * 64
        pipeline.write(manifest_path, manifest)
        job = {"artifacts": {"candidates": [row]}}
        pipeline.verify_candidates(job, self.root, count=1, frames=120, constraints_sha256="a" * 64)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.verify_candidates(job, self.root, count=1, frames=120, constraints_sha256="b" * 64)
        (self.root / row["output"]).write_bytes(b"different npz")
        with self.assertRaisesRegex(pipeline.PipelineError, "NPZ"):
            pipeline.verify_candidates(job, self.root, count=1, frames=120, constraints_sha256="a" * 64)


class PipelineOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.client = SimpleNamespace(workspace=str(self.root))
        self.args = SimpleNamespace(
            prompt="A person walks forward", frames=120, samples=2, output=str(self.root / "kimodo/task"),
            calibration=str(self.root / "calibration.json"), template_motion=str(self.root / "template.json"),
            template_scene=str(self.root / "Cascy.casc"), source_scene=None, keyframes=None,
            seed=42, steps=50, port=9552, timeout=1200, model="gemini-test", build=False,
        )
        for name in ("calibration", "template_motion", "template_scene"):
            Path(getattr(self.args, name)).write_text("fixture")
        self.events = []
        self.candidates = [{"id": f"candidate-{i + 1:02d}", "index": i, "seed": 42 + i,
                            "output": str(self.root / f"generated-{i}.npz")} for i in range(2)]
        self.workflow = SimpleNamespace(
            transfer=self.transfer,
            native=lambda *args: {"fps": 30, "units": "centimeters", "frames": []},
            native_metrics=lambda *args: {"native_contact_speed_m_s": 0.04, "native_contact_coverage_ratio": 0.3},
            render=self.render, capture_constraints=self.capture,
        )
        self.reviewer = SimpleNamespace(
            review_candidates=lambda *args, **kwargs: self.events.append("gemini") or review_report(*args, **kwargs),
            rank_candidates=lambda *args: {"state": "selected", "selected_id": "candidate-02"},
            review_settings=review_settings, review_identity=review_identity,
            _evidence_matches=pipeline.module("motion_review")._evidence_matches,
        )

    def transfer(self, client, motion, calibration, template, output, snapshot):
        self.events.append("transfer")
        self.assertIn("release", self.events)
        self.assertEqual(bool(snapshot), bool(self.args.keyframes))
        return {"scene": str(Path(output) / "result.casc"),
                "validation": {"geometry": True, "persistence": True, "constraints_required": bool(snapshot)}}

    def render(self, *args):
        self.events.append("render")
        path = Path(args[3]) / "front.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"review video")
        return [{"path": str(path), "view": "front"}]

    def capture(self, client, scene, frames, calibration, template, output):
        self.events.append("capture")
        self.assertEqual(frames, [0, 30, 60, 90, 119])
        return {"num_frames": 120, "capture": str(Path(output).with_suffix(".capture.json")), "sha256": "c" * 64}

    def run_fixture(self):
        def convert(source, target, exchange, **kwargs):
            self.events.append("convert")
            Path(target).write_text("fixture")

        def submit(endpoint, payload, path):
            self.events.append("submit")
            self.assertEqual(bool(payload.get("constraints")), bool(self.args.keyframes))
            return {"job_id": 7}

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret-only-in-environment"}), \
                patch.object(pipeline, "module", side_effect=lambda name: self.workflow if name == "workflow" else self.reviewer), \
                patch.object(pipeline, "runtime_action", side_effect=lambda *args, **kwargs: self.events.append("start")), \
                patch.object(pipeline, "submit_generation", side_effect=submit), \
                patch.object(pipeline, "wait_generation", return_value={"done": True}), \
                patch.object(pipeline, "verify_candidates", return_value=self.candidates), \
                patch.object(pipeline, "convert_motion", side_effect=convert), \
                patch.object(pipeline, "source_metrics", return_value=({"source_contact_speed_m_s": 0.03}, [])), \
                patch.object(pipeline, "release_runtime", side_effect=lambda *args, **kwargs: self.events.append("release")), \
                patch.object(pipeline, "activate_scene", side_effect=lambda client, path: self.events.append(("activate", str(path)))):
            return pipeline.run_pipeline(self.args, client=self.client)

    def test_text_variants_release_gpu_before_native_then_review_all_once(self):
        result = self.run_fixture()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(self.events.count("submit"), 1)
        self.assertEqual(self.events.count("gemini"), 1)
        self.assertLess(self.events.index("release"), self.events.index("transfer"))
        self.assertEqual(self.events[-1], ("activate", result["selected_scene"]))
        record = Path(self.args.output) / "run.json"
        self.assertNotIn("test-secret-only-in-environment", record.read_text())
        with self.assertRaises(FileExistsError):
            self.run_fixture()

    def test_keypose_mode_captures_and_returns_every_candidate_with_anchor_snapshot(self):
        self.args.source_scene = str(self.root / "source.casc")
        self.args.keyframes = "0,30,60,90,119"
        result = self.run_fixture()
        self.assertEqual(result["mode"], "keyposes")
        self.assertLess(self.events.index("capture"), self.events.index("start"))
        self.assertTrue(all(row["validation"]["constraints_required"] for row in result["candidates"]))

    def test_invalid_intervals_and_outputs_fail_before_runtime_or_native_work(self):
        self.args.source_scene = "scene.casc"
        self.args.keyframes = "0,30,60"
        with self.assertRaisesRegex(ValueError, "span"):
            self.run_fixture()
        self.assertEqual(self.events, [])
        self.assertFalse(Path(self.args.output).exists())

    def test_gemini_failure_preserves_native_candidates_without_selecting(self):
        def unavailable(*args, **kwargs):
            self.events.append("gemini")
            raise RuntimeError("Gemini unavailable; no selection")

        self.reviewer.review_candidates = unavailable
        with self.assertRaisesRegex(RuntimeError, "Gemini unavailable"):
            self.run_fixture()
        record = pipeline.read(Path(self.args.output) / "run.json")
        self.assertEqual(record["state"], "failed")
        self.assertEqual(len(record["candidates"]), 2)
        self.assertIn("release", self.events)
        self.assertFalse(any(isinstance(event, tuple) and event[0] == "activate" for event in self.events))

    def test_local_review_does_not_require_a_gemini_key(self):
        self.args.provider = "openai-compatible"
        self.args.model = "Qwen/Qwen3-VL-8B-Instruct"
        self.args.base_url = "http://127.0.0.1:8000/v1"
        result = self.run_fixture()
        self.assertEqual(result["review_identity"]["provider"], "openai-compatible")
        self.assertIsNone(review_settings(provider=self.args.provider)["api_key"])
        self.assertNotIn("api_key", Path(result["review"]).read_text())

    def test_missing_gemini_credentials_fail_before_generation(self):
        self.reviewer.review_settings = lambda **kwargs: {"api_key": None}
        with self.assertRaisesRegex(ValueError, "Configure a Gemini key"):
            self.run_fixture()
        self.assertFalse(self.events)
        self.assertFalse(Path(self.args.output).exists())


class PipelineResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.root = self.workspace / "kimodo"
        self.root.mkdir()
        self.output = self.root / "task"
        self.output.mkdir()
        self.client = SimpleNamespace(workspace=str(self.workspace))
        calibration = self.workspace / "calibration.json"
        pipeline.write(calibration, {})
        template = self.workspace / "Cascy.casc"
        template.write_bytes(b"native template")
        self.args = SimpleNamespace(output=str(self.output), calibration=str(calibration),
                                    template_scene=str(template), model="gemini-test")
        rows = [ArtifactTests.candidate(self, index) for index in range(2)]
        self.job = {"done": True, "artifacts": {"candidates": rows}}
        self.candidates = pipeline.verify_candidates(self.job, self.root, count=2, frames=120)
        for candidate in self.candidates:
            directory = self.output / candidate["id"]
            directory.mkdir()
            converted = directory / "motion.retarget.json"
            pipeline.write(converted, {"source_npz_sha256": pipeline._sha(candidate["output"])})
            candidate["retarget_motion"] = str(converted)
            candidate["retarget_sha256"] = pipeline._sha(converted)
        self.record = {"format": "kimodo-cascadeur-run-v1", "state": "interrupted", "stage": "rendering_candidate-01",
                       "prompt": "A person walks forward", "frames": 120, "samples": 2, "mode": "text",
                       "generated_candidates": self.candidates, "candidates": []}
        pipeline.write(self.output / "run.json", self.record)
        pipeline.write(self.output / "generation.json", {"state": "finished", "last_observation": self.job})
        self.transfers, self.renders, self.reviews, self.activations, self.review_options = [], [], [], [], []
        self.workflow = SimpleNamespace(transfer=self.transfer, render=self.render, native=self.native,
                                        native_metrics=lambda *args: {"native_contact_speed_m_s": 0.03,
                                                                     "native_contact_coverage_ratio": 0.3})
        original_reviewer = pipeline.module("motion_review")
        self.reviewer = SimpleNamespace(review_candidates=self.review,
                                        rank_candidates=lambda *args: {"state": "selected", "selected_id": "candidate-02"},
                                        review_settings=review_settings, review_identity=review_identity,
                                        _evidence_matches=original_reviewer._evidence_matches)

    def complete_native(self, candidate):
        directory = self.output / candidate["id"] / "native"
        session = directory / "points-test"
        session.mkdir(parents=True)
        before = session / "before.casc"
        before.write_bytes(Path(self.args.template_scene).read_bytes())
        scene = session / "result.casc"
        scene.write_bytes(b"native result " + candidate["id"].encode())
        plan = directory / "point-plan.json"
        pipeline.write(plan, {"source_npz_sha256": pipeline._sha(candidate["output"])})
        receipt = session / "session.json"
        pipeline.write(receipt, {"phase": "completed", "directory": str(session), "plan_path": str(plan),
                                 "plan_sha256": pipeline._sha(plan), "result_sha256": pipeline._sha(scene),
                                 "before_sha256": pipeline._sha(before),
                                 "reopen_verification": {"point_coordinates_verified": True}})
        result = {"scene": str(scene), "receipt": str(receipt),
                  "validation": {"geometry": True, "persistence": True, "constraints_required": False}}
        pipeline.write(directory / "transfer.json", result)
        return result

    def transfer(self, client, motion, calibration, template, output, snapshot):
        self.transfers.append(str(output))
        candidate = next(item for item in self.candidates if item["retarget_motion"] == motion)
        return self.complete_native(candidate)

    def render(self, client, scene, measurement, output):
        self.renders.append(str(output))
        Path(output).mkdir(parents=True)
        media = []
        for view in ("front", "side", "three-quarter"):
            path = Path(output) / (view + ".mp4")
            path.write_bytes((str(scene) + view).encode())
            media.append({"path": str(path), "view": view})
        return media

    def native(self, client, operation, payload):
        if operation == "render_status":
            return pipeline.read(payload["receipt"])
        self.assertEqual(operation, "measure")
        return {"frames": [], "fps": 30, "units": "centimeters"}

    def review(self, candidates, prompt, **kwargs):
        self.reviews.append([item["id"] for item in candidates])
        self.review_options.append(kwargs)
        self.assertEqual(kwargs["max_output_tokens"], 8192)
        return review_report(candidates, prompt, **kwargs)

    def run_resume(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pipeline, "module", side_effect=lambda name: {
                "workflow": self.workflow, "motion_review": self.reviewer
            }[name]))
            forbidden = [stack.enter_context(patch.object(pipeline, name, side_effect=AssertionError(name)))
                         for name in ("runtime_action", "submit_generation", "wait_generation", "convert_motion",
                                      "request_json", "release_runtime")]
            stack.enter_context(patch.object(pipeline, "source_metrics", return_value=({}, [])))
            stack.enter_context(patch.object(pipeline, "activate_scene", side_effect=lambda client, path:
                                            self.activations.append(str(path))))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = pipeline.resume_pipeline(self.args, client=self.client)
            for call in forbidden:
                call.assert_not_called()
            return result

    def test_resume_reuses_completed_scene_and_only_transfers_missing_candidate(self):
        original = self.complete_native(self.candidates[0])
        before = pipeline._sha(original["scene"])
        result = self.run_resume()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(len(self.transfers), 1)
        self.assertEqual(len(self.reviews), 1)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(pipeline._sha(original["scene"]), before)
        self.assertEqual(self.activations[0], original["scene"])
        self.assertEqual(self.activations[-1], result["selected_scene"])
        self.assertFalse((self.output / ".resume.lock").exists())

    def test_resume_reuses_pinned_media_without_duplicate_render(self):
        result = self.complete_native(self.candidates[0])
        media = self.render(None, result["scene"], {}, self.output / "candidate-01/review")
        for item in media:
            item["sha256"] = pipeline._sha(item["path"])
        self.record["candidates"] = [{**self.candidates[0], **result, "media": media}]
        pipeline.write(self.output / "run.json", self.record)
        self.renders.clear()
        self.run_resume()
        self.assertEqual(len(self.renders), 1)
        self.assertTrue(self.renders[0].endswith(str(Path("candidate-02/review"))))

    def test_incomplete_or_uncertain_native_receipt_refuses_all_scene_changes(self):
        result = self.complete_native(self.candidates[1])
        receipt = pipeline.read(result["receipt"])
        receipt["phase"] = "points_ready:20"
        pipeline.write(result["receipt"], receipt)
        with self.assertRaisesRegex(pipeline.PipelineError, "incomplete"):
            self.run_resume()
        self.assertFalse(self.transfers or self.renders or self.activations or self.reviews)

    def test_changed_saved_scene_is_not_rewritten(self):
        result = self.complete_native(self.candidates[0])
        Path(result["scene"]).write_bytes(b"edited by user")
        with self.assertRaisesRegex(pipeline.PipelineError, "differs"):
            self.run_resume()
        self.assertFalse(self.transfers or self.activations)
        self.assertEqual(Path(result["scene"]).read_bytes(), b"edited by user")

    def test_changed_source_npz_or_retarget_checksum_prevents_resume(self):
        converted = Path(self.candidates[0]["retarget_motion"])
        pipeline.write(converted, {"source_npz_sha256": "wrong"})
        with self.assertRaisesRegex(pipeline.PipelineError, "Retarget source"):
            self.run_resume()
        self.assertFalse(self.transfers or self.activations)

    def test_legacy_npz_hash_is_disclosed_and_verified_against_retarget(self):
        for candidate in self.candidates:
            manifest = pipeline.read(candidate["manifest"])
            manifest.pop("npz_sha256")
            pipeline.write(candidate["manifest"], manifest)
        # This simulates a legacy run before artifact hashes were recorded.
        for candidate in self.record["generated_candidates"]:
            candidate.pop("artifact_sha256")
        pipeline.write(self.output / "run.json", self.record)
        result = self.run_resume()
        self.assertIn("Legacy generation manifests lack NPZ hashes", result["provenance_limitations"][0])
        self.assertEqual(len(result["resume_pins"]["candidates"]), 2)

    def test_live_or_orphaned_renderer_refuses_native_transfer(self):
        directory = self.output / "candidate-01/review"
        directory.mkdir()
        pipeline.write(directory / "native-render.json", {"id": "active", "state": "running", "restoration": None})
        with self.assertRaisesRegex(pipeline.PipelineError, "renderer is active"):
            self.run_resume()
        self.assertFalse(self.transfers or self.activations or self.renders)

    def test_partial_unmanaged_media_is_retained_and_new_render_does_not_overwrite(self):
        directory = self.output / "candidate-01/review"
        directory.mkdir()
        original = directory / "front.mp4"
        original.write_bytes(b"partial old video")
        self.run_resume()
        self.assertEqual(original.read_bytes(), b"partial old video")
        self.assertTrue(any(path.endswith("review-2") for path in self.renders))

    def test_existing_hash_bound_gemini_review_is_reused_without_an_api_call(self):
        entries = []
        for candidate in self.candidates:
            result = self.complete_native(candidate)
            media = self.render(None, result["scene"], {}, self.output / candidate["id"] / "review")
            for item in media:
                item["sha256"] = pipeline._sha(item["path"])
            entries.append({**candidate, **result, "media": media})
        report = self.review(entries, self.record["prompt"], max_output_tokens=8192)
        pipeline.write(self.output / "gemini-review.json", report)
        self.record["candidates"] = entries
        pipeline.write(self.output / "run.json", self.record)
        self.renders.clear()
        self.reviews.clear()
        self.run_resume()
        self.assertFalse(self.renders or self.reviews or self.transfers)

    def test_unfinished_generation_and_resume_lock_are_not_replayed(self):
        pipeline.write(self.output / "generation.json", {"state": "unknown"})
        with self.assertRaisesRegex(pipeline.PipelineError, "not observed or resubmitted"):
            self.run_resume()
        self.assertFalse(self.transfers or self.renders)
        pipeline.write(self.output / ".resume.lock", {"pid": "other"})
        with self.assertRaises(FileExistsError):
            self.run_resume()
        self.assertFalse((Path(self.args.output) / "selection.json").exists())

    def test_review_abstention_retains_candidates_without_activating_a_winner(self):
        self.reviewer.rank_candidates = lambda *args: {"state": "needs_revision", "selected_id": None}
        result = self.run_resume()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["selection"]["state"], "needs_revision")
        self.assertNotIn("selected_scene", result)
        self.assertFalse(self.activations)

    def complete_run(self, *, legacy_review=False):
        entries = []
        for candidate in self.candidates:
            result = self.complete_native(candidate)
            media = self.render(None, result["scene"], {}, self.output / candidate["id"] / "review")
            for item in media:
                item["sha256"] = pipeline._sha(item["path"])
            entries.append({**candidate, **result, "media": media, "metrics": {"native_contact_speed_m_s": 0.03}})
        report = review_report(entries, self.record["prompt"], model=self.args.model)
        if legacy_review:
            report.pop("review_identity")
            report.update(schema="kimodo-gemini-review-v1", requested_model=self.args.model)
            for candidate in report["evidence"]:
                for item in candidate["media"]:
                    item["sampling_fps"] = 12
        baseline = self.output / "gemini-review.json"
        pipeline.write(baseline, report)
        self.record.update(state="completed", stage="completed", candidates=entries, review=str(baseline),
                           review_identity=review_identity(model=self.args.model),
                           selection={"state": "selected", "selected_id": "candidate-02"},
                           selected_scene=entries[1]["scene"])
        pipeline.write(self.output / "run.json", self.record)
        pipeline.write(self.output / "selection.json", self.record["selection"])
        self.renders.clear()
        return baseline

    def run_review(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pipeline, "module", side_effect=lambda name: {
                "workflow": self.workflow, "motion_review": self.reviewer
            }[name]))
            for name in ("runtime_action", "submit_generation", "wait_generation", "convert_motion", "request_json",
                         "release_runtime", "source_metrics", "activate_scene"):
                stack.enter_context(patch.object(pipeline, name, side_effect=AssertionError(name)))
            for name in ("native", "transfer", "render"):
                stack.enter_context(patch.object(self.workflow, name, side_effect=AssertionError(name)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return pipeline.review_pipeline(self.args, client=self.client)

    def test_completed_run_local_review_preserves_gemini_and_never_touches_native(self):
        baseline = self.complete_run()
        original = baseline.read_bytes()
        scenes = {item["scene"]: pipeline._sha(item["scene"]) for item in self.record["candidates"]}
        self.args.provider = "openai-compatible"
        self.args.model = "Qwen/Qwen3-VL-8B-Instruct"
        self.args.base_url = "http://localhost:8000/v1"
        self.args.media_transport = "frames"
        self.args.sampling_fps = 2
        self.args.max_total_frames, self.args.max_frames_per_video, self.args.frame_width = 48, 8, 384
        result = self.run_review()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(len(self.reviews), 1)
        self.assertEqual(baseline.read_bytes(), original)
        self.assertEqual(result["review_history"][0]["review"], str(baseline))
        self.assertTrue(Path(result["review"]).name.startswith("openai-compatible-"))
        self.assertEqual(result["review_identity"]["max_total_frames"], 48)
        self.assertEqual(self.review_options[0]["config"]["max_frames_per_video"], 8)
        self.assertEqual(pipeline.read(self.output / "selection.json"), result["selection"])
        self.assertFalse(result["last_review_attempt"]["selection_activated"])
        self.assertEqual({path: pipeline._sha(path) for path in scenes}, scenes)
        self.assertFalse(self.transfers or self.renders or self.activations)
        self.assertFalse((self.output / ".resume.lock").exists())
        for path in (self.output / "run.json", Path(result["review"])):
            self.assertNotIn("fixture-secret-never-persist", path.read_text())

    def test_completed_review_cache_needs_neither_credentials_nor_another_request(self):
        self.complete_run()
        self.args.provider, self.args.model = "openai-compatible", "local-qwen"
        first = self.run_review()
        self.reviewer.review_settings = lambda **kwargs: self.fail("Cached review must not need credentials")
        second = self.run_review()
        self.assertEqual(len(self.reviews), 1)
        self.assertEqual(first["review"], second["review"])
        self.assertTrue(second["last_review_attempt"]["reused"])

    def test_model_endpoint_transport_sampling_and_budget_each_partition_review_cache(self):
        self.complete_run()
        self.args.provider, self.args.model = "openai-compatible", "local-qwen"
        paths = [self.run_review()["review"]]
        for name, value in (("model", "another-model"), ("base_url", "http://127.0.0.1:8001/v1"),
                            ("sampling_fps", 2), ("media_transport", "frames"), ("max_total_frames", 240)):
            setattr(self.args, name, value)
            paths.append(self.run_review()["review"])
        self.assertEqual(len(set(paths)), 6)
        self.assertEqual(len(self.reviews), 6)
        self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_known_legacy_gemini_identity_reuses_report_without_request(self):
        baseline = self.complete_run(legacy_review=True)
        self.reviewer.review_settings = lambda **kwargs: self.fail("Legacy cached review must not need credentials")
        result = self.run_review()
        self.assertEqual(result["review"], str(baseline))
        self.assertFalse(self.reviews)
        self.assertTrue(result["last_review_attempt"]["reused"])

    def test_legacy_unknown_model_cannot_be_assumed_to_match_current_provider(self):
        baseline = self.complete_run(legacy_review=True)
        report = pipeline.read(baseline)
        report.pop("requested_model")
        pipeline.write(baseline, report)
        original = baseline.read_bytes()
        result = self.run_review()
        self.assertEqual(len(self.reviews), 1)
        self.assertNotEqual(result["review"], str(baseline))
        self.assertEqual(baseline.read_bytes(), original)

    def test_changed_video_or_scene_refuses_completed_review_and_preserves_previous_selection(self):
        self.complete_run()
        candidate = self.record["candidates"][0]
        for target, message in ((candidate["media"][0]["path"], "checksum"), (candidate["scene"], "differs")):
            with self.subTest(target=Path(target).suffix):
                path = Path(target)
                original = path.read_bytes()
                path.write_bytes(b"user changed this artifact")
                with self.assertRaisesRegex(pipeline.PipelineError, message):
                    self.run_review()
                record = pipeline.read(self.output / "run.json")
                self.assertEqual(record["state"], "completed")
                self.assertEqual(record["selection"], self.record["selection"])
                self.assertEqual(record["last_review_attempt"]["state"], "failed")
                path.write_bytes(original)
        self.assertFalse(self.reviews)

    def test_failed_new_provider_preserves_completed_state_and_baseline_selection(self):
        baseline = self.complete_run()
        original_selection = (self.output / "selection.json").read_bytes()
        self.args.provider, self.args.model = "openai-compatible", "local-qwen"
        self.reviewer.review_candidates = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Local model unavailable"))
        with self.assertRaisesRegex(RuntimeError, "Local model unavailable"):
            self.run_review()
        record = pipeline.read(self.output / "run.json")
        self.assertEqual((record["state"], record["stage"]), ("completed", "completed"))
        self.assertEqual(record["review"], str(baseline))
        self.assertEqual((self.output / "selection.json").read_bytes(), original_selection)
        self.assertFalse((self.output / ".resume.lock").exists())

    def test_completed_review_requires_terminal_renderer_and_all_saved_native_candidates(self):
        self.complete_run()
        receipt = self.output / "candidate-01/review/native-render.json"
        pipeline.write(receipt, {"id": "unknown", "state": "running", "restoration": None})
        with self.assertRaisesRegex(pipeline.PipelineError, "renderer is active"):
            self.run_review()
        self.assertFalse(self.reviews)
        pipeline.write(receipt, {"id": "finished", "state": "completed", "restoration": {"restored": True}})
        pipeline.write(self.output / "candidate-02/native/transfer.json", {"validation": {"geometry": True}})
        with self.assertRaisesRegex(pipeline.PipelineError, "persistence verification"):
            self.run_review()
        self.assertFalse(self.reviews)


class PipelineCliTests(unittest.TestCase):
    def test_review_command_forwards_provider_and_bounded_sampling_overrides(self):
        result = {"state": "completed", "mode": "text", "candidates": []}
        with patch.object(pipeline, "review_pipeline", return_value=result) as review, \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(["review", "--output", "retained-run", "--provider", "openai-compatible",
                           "--model", "local-qwen", "--base-url", "http://localhost:8000/v1",
                           "--media-transport", "frames", "--sampling-fps", "2", "--max-total-frames", "48",
                           "--max-frames-per-video", "8", "--frame-width", "384"])
        options = pipeline._review_options(review.call_args.args[0])
        self.assertEqual(options["provider"], "openai-compatible")
        self.assertEqual(options["sampling_fps"], 2)
        self.assertEqual(options["config"], {"max_total_frames": 48, "max_frames_per_video": 8, "frame_width": 384})
