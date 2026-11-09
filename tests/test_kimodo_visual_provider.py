"""Local HTTP protocol, bounded frame sampling and shared review acceptance."""

import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1] / "integrations/kimodo"


def load(name):
    spec = importlib.util.spec_from_file_location("test_visual_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reviewer, backend = load("motion_review"), load("visual_provider")


class LocalReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cascadeur-provider-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        clean_environment = {key: value for key, value in os.environ.items()
                             if not key.startswith("VISUAL_REVIEW_") and key not in {
                                 "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"
                             }}
        self.environment = mock.patch.dict(os.environ, clean_environment, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        resolver = mock.patch.object(reviewer, "_resolve_settings", return_value={})
        resolver.start()
        self.addCleanup(resolver.stop)
        self.candidates = []
        for index in range(2):
            path = self.root / f"clip-{index}.mp4"
            path.write_bytes(b"video fixture " + str(index).encode())
            self.candidates.append({
                "id": f"candidate-{index}", "media": [{"path": str(path), "view": "side"}],
                "validation": {"geometry": True, "persistence": True},
                "metrics": {"native_contact_speed_m_s": 0.03, "native_contact_coverage_ratio": 0.5},
            })
        self.review = {
            "candidates": [{"id": item["id"], "usable": True, "confidence": 0.8,
                            "scores": {name: 85 for name in reviewer.SCORE_WEIGHTS},
                            "observations": ["Side view 0.25s: readable alternating steps."],
                            "limitations": ["Far foot is sometimes occluded."]} for item in self.candidates],
            "preferred_candidate_id": "candidate-0", "comparison_summary": "Comparable, first preferred.",
        }
        self.requests = []
        self.status, self.finish, self.usage = 200, "stop", {"total_tokens": 123}
        self.redirect = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append({"path": self.path, "headers": dict(self.headers), "payload": payload})
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                if owner.redirect:
                    self.send_header("Location", owner.base_url + "/credential-leak")
                self.end_headers()
                result = {"model": "local-qwen-tested-id", "choices": [{"finish_reason": owner.finish,
                    "message": {"content": json.dumps(owner.review)}}], "usage": owner.usage}
                self.wfile.write(json.dumps(result).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def call(self, **overrides):
        settings = {"provider": "openai-compatible", "model": "Qwen/Qwen3-VL-4B-Instruct",
                    "base_url": self.base_url}
        settings.update(overrides)
        return reviewer.review_candidates(self.candidates, "Walk forward naturally", **settings)

    def test_real_loopback_http_returns_same_contract_and_ranking(self):
        report = self.call(api_key="local-test-key")
        self.assertEqual(report["schema"], "kimodo-visual-review-v1")
        self.assertEqual(report["provider"], "openai-compatible")
        self.assertEqual(report["actual_model"], "local-qwen-tested-id")
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertEqual(selection["selected_id"], "candidate-0")
        self.assertEqual(selection["visual_preference"], "candidate-0")
        self.assertEqual(len(self.requests), 1)
        sent = self.requests[0]
        self.assertEqual(sent["path"], "/v1/chat/completions")
        self.assertEqual(sent["headers"]["Authorization"], "Bearer local-test-key")
        payload = sent["payload"]
        videos = [part for part in payload["messages"][0]["content"] if part["type"] == "video_url"]
        self.assertEqual(len(videos), 2)
        self.assertTrue(all(part["video_url"]["url"].startswith("data:video/mp4;base64,") for part in videos))
        self.assertEqual(payload["mm_processor_kwargs"], {"fps": 12, "do_sample_frames": True})
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["preferred_candidate_id"]["type"], ["string", "null"])
        self.assertNotIn("local-test-key", json.dumps(report))

    def test_local_provider_does_not_inherit_gemini_credential(self):
        with mock.patch.object(reviewer, "_resolve_settings", return_value={
            "provider": "gemini", "api_key_provider": "gemini", "api_key": "gemini-vault-secret"
        }), mock.patch.dict(os.environ, {"GEMINI_API_KEY": "another-gemini-secret"}):
            report = self.call()
        self.assertNotIn("Authorization", self.requests[0]["headers"])
        self.assertNotIn("gemini-vault-secret", json.dumps(report))

    def test_public_identity_excludes_resolved_credentials(self):
        identity = reviewer.review_identity(provider="openai-compatible", model="qwen-local",
            base_url=self.base_url, config={"api_key": "secret", "api_key_provider": "openai-compatible"})
        self.assertEqual(identity["provider"], "openai-compatible")
        self.assertNotIn("api_key", identity)
        self.assertNotIn("secret", json.dumps(identity))

    def test_environment_and_explicit_options_override_local_settings(self):
        with mock.patch.object(reviewer, "_resolve_settings", return_value={"provider": "gemini", "model": "old"}), \
                mock.patch.dict(os.environ, {"VISUAL_REVIEW_PROVIDER": "openai-compatible",
                    "VISUAL_REVIEW_MODEL": "qwen-env", "VISUAL_REVIEW_SAMPLING_FPS": "6"}):
            resolved = reviewer.review_settings(model="qwen-explicit", base_url=self.base_url)
        self.assertEqual(resolved["model"], "qwen-explicit")
        self.assertEqual(resolved["sampling_fps"], 6)

    def test_remote_or_credential_urls_are_rejected_before_submission(self):
        for url in ("https://example.com/v1", "http://localhost:8000/v1?key=secret",
                    "http://user:secret@127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1#fragment"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.call(base_url=url)
        self.assertFalse(self.requests)

    def test_http_failure_has_no_fallback_or_provider_body(self):
        self.status, self.usage = 503, {"echo": "local-test-key"}
        with self.assertRaises(reviewer.ReviewError) as caught:
            self.call(api_key="local-test-key")
        self.assertIn("503", str(caught.exception))
        self.assertNotIn("local-test-key", str(caught.exception))
        self.assertEqual(len(self.requests), 1)

    def test_redirect_is_not_followed_with_authentication(self):
        self.status, self.redirect = 307, True
        with self.assertRaises(reviewer.ReviewError):
            self.call(api_key="local-test-key")
        self.assertEqual(len(self.requests), 1)

    def test_truncation_and_invalid_review_cannot_qualify_a_candidate(self):
        self.finish = "length"
        with self.assertRaisesRegex(reviewer.ReviewError, "truncated"):
            self.call()
        self.finish = "stop"
        self.review["candidates"][0]["id"] = "wrong-candidate"
        with self.assertRaisesRegex(reviewer.ReviewError, "IDs"):
            self.call()

    def test_credential_reflected_in_usage_is_discarded(self):
        self.usage = {"unexpected_echo": "local-test-key"}
        with self.assertRaisesRegex(reviewer.ReviewError, "credential material"):
            self.call(api_key="local-test-key")

    def test_frame_sequence_labels_timestamps_and_source_hashes(self):
        sampled = [{"seconds": i / 12, "jpeg": b"jpeg" + bytes([i])} for i in range(3)]
        with mock.patch.object(reviewer, "_load_module", return_value=backend), \
                mock.patch.object(backend, "decode_frames", return_value=sampled):
            report = self.call(media_transport="frames")
        payload = self.requests[0]["payload"]
        self.assertNotIn("mm_processor_kwargs", payload)
        parts = payload["messages"][0]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in parts), 6)
        self.assertTrue(any("0.083333 seconds" in part.get("text", "") for part in parts))
        self.assertEqual(report["evidence"][0]["media"][0]["frame_count"], 3)
        self.assertTrue(report["evidence"][0]["media"][0]["temporal_evidence"])
        self.assertEqual(reviewer.rank_candidates(self.candidates, report)["selected_id"], "candidate-0")
        report["evidence"][0]["media"][0]["frame_count"] = 1
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertFalse(next(row for row in selection["ranking"] if row["id"] == "candidate-0")["eligible"])

    def test_frame_budget_never_silently_drops_candidates_or_frames(self):
        sampled = [{"seconds": i / 12, "jpeg": b"jpeg"} for i in range(3)]
        with mock.patch.object(reviewer, "_load_module", return_value=backend), \
                mock.patch.object(backend, "decode_frames", return_value=sampled):
            with self.assertRaisesRegex(reviewer.ReviewError, "max_total_frames"):
                self.call(media_transport="frames", config={"max_total_frames": 5})
        self.assertFalse(self.requests)

    def test_preview_sampling_metadata_is_part_of_review_evidence(self):
        report = self.call()
        self.candidates[0]["media"][0]["render_sample_step"] = 2
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertFalse(next(row for row in selection["ranking"] if row["id"] == "candidate-0")["eligible"])

    def test_changed_source_video_during_decoding_is_rejected(self):
        def change(path, *args, **kwargs):
            Path(path).write_bytes(b"changed")
            return [{"seconds": i / 12, "jpeg": b"jpeg"} for i in range(3)]
        with mock.patch.object(reviewer, "_load_module", return_value=backend), \
                mock.patch.object(backend, "decode_frames", side_effect=change):
            with self.assertRaisesRegex(reviewer.ReviewError, "changed during"):
                self.call(media_transport="frames")
        self.assertFalse(self.requests)

    def test_real_ffmpeg_frames_cross_the_local_http_contract(self):
        try:
            import imageio_ffmpeg
        except ImportError:
            self.skipTest("imageio-ffmpeg is optional; run with the animation dependency group")
        video = self.root / "real.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc2=size=128x96:rate=12", "-t", "0.5", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True, timeout=15)
        self.candidates = [copy.deepcopy(self.candidates[0])]
        self.candidates[0]["media"][0]["path"] = str(video)
        self.review["candidates"] = self.review["candidates"][:1]
        report = self.call(media_transport="frames", config={"max_total_frames": 12, "frame_width": 224})
        frames = report["evidence"][0]["media"][0]
        self.assertEqual(frames["frame_count"], 6)
        self.assertAlmostEqual(frames["timestamps_seconds"][-1], 5 / 12)
        self.assertEqual(len(list(self.root.glob("*.jpg"))), 0)
        self.assertEqual(reviewer.rank_candidates(self.candidates, report)["selected_id"], "candidate-0")


if __name__ == "__main__":
    unittest.main()
