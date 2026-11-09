"""Motion selection needs reviewed video, native validation and honest abstention."""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "integrations/kimodo/motion_review.py"
SPEC = importlib.util.spec_from_file_location("kimodo_motion_review", MODULE_PATH)
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)


class MotionReviewTests(unittest.TestCase):
    def setUp(self):
        resolver = mock.patch.object(reviewer, "_resolve_settings", return_value={})
        resolver.start()
        self.addCleanup(resolver.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix="kimodo-review-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.key = "unit-test-secret-never-store"
        self.candidates = []
        for index in range(2):
            media = self.root / f"candidate-{index}.mp4"
            media.write_bytes(b"test video " + str(index).encode())
            self.candidates.append({
                "id": f"candidate-{index}",
                "media": [{"path": str(media), "view": "front and side"}],
                "validation": {"geometry": True, "persistence": True},
                "metrics": {
                    "native_contact_speed_m_s": 0.02 + index * 0.01,
                    "native_contact_coverage_ratio": 0.3,
                },
            })
        self.reply = {
            "candidates": [{
                "id": item["id"], "usable": True, "confidence": 0.8,
                "scores": {name: 85 - index * 5 for name in reviewer.SCORE_WEIGHTS},
                "observations": ["Side view 00:01: slight foot sliding."],
                "limitations": ["Occluded far foot."],
            } for index, item in enumerate(self.candidates)],
            "preferred_candidate_id": "candidate-0", "comparison_summary": "First is preferable.",
        }

    def api_response(self, reply=None, finish="STOP"):
        return io.BytesIO(json.dumps({
            "candidates": [{
                "finishReason": finish,
                "content": {"parts": [{"text": json.dumps(reply or self.reply)}]},
            }],
            "modelVersion": "gemini-test-actual", "usageMetadata": {"totalTokenCount": 123},
        }).encode())

    def run_review(self):
        with mock.patch.object(reviewer.urllib.request, "urlopen", return_value=self.api_response()):
            return reviewer.review_candidates(self.candidates, "Walk forward", api_key=self.key, model="gemini-test")

    def test_single_request_contains_all_media_custom_fps_and_no_query_key(self):
        with mock.patch.object(reviewer.urllib.request, "urlopen", return_value=self.api_response()) as request:
            report = reviewer.review_candidates(
                self.candidates, "Walk forward", api_key=self.key, model="gemini-test", sampling_fps=12
            )
        request.assert_called_once()
        sent = request.call_args.args[0]
        payload = json.loads(sent.data)
        self.assertNotIn(self.key, sent.full_url)
        self.assertNotIn(self.key, sent.data.decode())
        self.assertEqual(sent.get_header("X-goog-api-key"), self.key)
        media = [part for part in payload["contents"][0]["parts"] if "inlineData" in part]
        self.assertEqual(len(media), 2)
        self.assertTrue(all(part["videoMetadata"]["fps"] == 12 for part in media))
        self.assertNotIn(self.key, json.dumps(report))
        self.assertEqual(report["actual_model"], "gemini-test-actual")
        self.assertEqual(len(report["evidence"]), 2)

    def test_missing_credentials_does_not_send_a_request(self):
        with mock.patch.dict(reviewer.os.environ, {}, clear=True), mock.patch.object(
            reviewer.urllib.request, "urlopen"
        ) as request:
            with self.assertRaisesRegex(reviewer.ReviewError, "outside the repository"):
                reviewer.review_candidates(self.candidates, "Walk")
            request.assert_not_called()

    def test_sampled_preview_cadence_is_disclosed_to_the_reviewer(self):
        self.candidates[0]["media"][0]["render_sample_step"] = 2
        parts, evidence = reviewer._media_parts(self.candidates, 12, 18 * 1024 * 1024)
        labels = " ".join(part.get("text", "") for part in parts)
        self.assertIn("samples every 2 source frames", labels)
        self.assertIn("not proof of animation jerk", labels)
        self.assertEqual(evidence[0]["media"][0]["render_sample_step"], 2)

    def test_invalid_preview_cadence_is_rejected_before_transmission(self):
        self.candidates[0]["media"][0]["render_sample_step"] = 0
        with mock.patch.object(reviewer.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "sample step"):
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key)
            request.assert_not_called()

    def test_http_error_does_not_reflect_provider_body_or_credential(self):
        failure = urllib.error.HTTPError(
            "https://provider/" + self.key, 403, self.key, {}, io.BytesIO(self.key.encode())
        )
        with mock.patch.object(reviewer.urllib.request, "urlopen", side_effect=failure):
            with self.assertRaises(reviewer.ReviewError) as raised:
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key)
        self.assertIn("403", str(raised.exception))
        self.assertNotIn(self.key, str(raised.exception))

    def test_network_timeout_does_not_retry(self):
        with mock.patch.object(reviewer.urllib.request, "urlopen", side_effect=TimeoutError(self.key)) as request:
            with self.assertRaisesRegex(reviewer.ReviewError, "no automatic retry"):
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key)
            request.assert_called_once()

    def test_unavailable_model_explains_resume_without_exposing_provider_body(self):
        failure = urllib.error.HTTPError("https://provider/", 404, "missing", {}, io.BytesIO(self.key.encode()))
        with mock.patch.object(reviewer.urllib.request, "urlopen", side_effect=failure) as request:
            with self.assertRaises(reviewer.ReviewError) as raised:
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key, model="retired-model")
        self.assertIn("retired-model", str(raised.exception))
        self.assertIn("resume the saved run", str(raised.exception))
        self.assertNotIn(self.key, str(raised.exception))
        request.assert_called_once()

    def test_partial_response_and_unknown_candidate_are_rejected(self):
        reply = copy.deepcopy(self.reply)
        reply["candidates"][0]["id"] = "invented"
        with mock.patch.object(reviewer.urllib.request, "urlopen", return_value=self.api_response(reply)):
            with self.assertRaisesRegex(reviewer.ReviewError, "IDs do not match"):
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key)
        with mock.patch.object(
            reviewer.urllib.request, "urlopen", return_value=self.api_response(finish="MAX_TOKENS")
        ):
            with self.assertRaisesRegex(reviewer.ReviewError, "truncated"):
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key)

    def test_invalid_scores_and_entries_are_rejected(self):
        for value in (101, -1, float("nan"), True):
            with self.subTest(value=value):
                reply = copy.deepcopy(self.reply)
                reply["candidates"][0]["scores"]["naturalness"] = value
                with self.assertRaises(reviewer.ReviewError):
                    reviewer.validate_review(reply, [item["id"] for item in self.candidates])
        reply = {**self.reply, "candidates": [None]}
        with self.assertRaises(reviewer.ReviewError):
            reviewer.validate_review(reply, ["candidate-0"])

    def test_reflected_credential_is_discarded(self):
        reply = copy.deepcopy(self.reply)
        reply["comparison_summary"] = self.key
        with mock.patch.object(reviewer.urllib.request, "urlopen", return_value=self.api_response(reply)):
            with self.assertRaisesRegex(reviewer.ReviewError, "credential material"):
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key)

    def test_media_budget_and_invalid_fps_fail_before_network(self):
        with mock.patch.object(reviewer.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "budget"):
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key, max_inline_bytes=5)
            with self.assertRaises(ValueError):
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key, sampling_fps=30)
            request.assert_not_called()

    def test_credential_fields_in_diagnostics_are_not_sent(self):
        self.candidates[0]["metrics"]["api_key"] = self.key
        with mock.patch.object(reviewer.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "credential fields"):
                reviewer.review_candidates(self.candidates, "Walk", api_key=self.key)
            request.assert_not_called()

    def test_selection_combines_review_with_contact_diagnostics(self):
        report = self.run_review()
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertEqual(selection["state"], "selected")
        self.assertEqual(selection["selected_id"], "candidate-0")
        first = selection["ranking"][0]
        self.assertGreater(first["contact_diagnostic_score"], 80)
        self.assertEqual(selection["rubric"]["combined_weights"]["visual"], 0.75)

    def test_failed_native_save_rejects_visually_preferred_candidate(self):
        report = self.run_review()
        self.candidates[0]["validation"]["persistence"] = False
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertEqual(selection["selected_id"], "candidate-1")
        rejected = next(item for item in selection["ranking"] if item["id"] == "candidate-0")
        self.assertIn("Native persistence verification did not pass", rejected["rejections"])

    def test_failed_required_anchors_are_not_overridden_by_visual_score(self):
        report = self.run_review()
        for candidate in self.candidates:
            candidate["validation"].update(constraints_required=True, constraints=False)
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertIsNone(selection["selected_id"])
        self.assertEqual(selection["state"], "needs_revision")

    def test_no_or_insufficient_contact_coverage_is_unknown(self):
        report = self.run_review()
        self.candidates[0]["metrics"]["native_contact_coverage_ratio"] = 0
        self.candidates[1]["metrics"] = {}
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertIsNone(selection["selected_id"])
        self.assertTrue(all(item["contact_diagnostic_score"] is None for item in selection["ranking"]))

    def test_model_abstention_or_all_unusable_does_not_select_a_winner(self):
        report = self.run_review()
        report["review"]["preferred_candidate_id"] = None
        self.assertIsNone(reviewer.rank_candidates(self.candidates, report)["selected_id"])
        report["review"]["preferred_candidate_id"] = "candidate-0"
        for entry in report["review"]["candidates"]:
            entry["usable"] = False
        self.assertIsNone(reviewer.rank_candidates(self.candidates, report)["selected_id"])

    def test_still_only_review_does_not_claim_temporal_motion_selection(self):
        for candidate in self.candidates:
            image = self.root / (candidate["id"] + ".png")
            image.write_bytes(b"test png")
            candidate["media"] = [{"path": str(image), "view": "contact sheet"}]
        report = self.run_review()
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertIsNone(selection["selected_id"])
        self.assertTrue(all("Temporal video evidence is missing" in item["rejections"] for item in selection["ranking"]))

    def test_changed_video_cannot_reuse_an_old_review(self):
        report = self.run_review()
        for candidate in self.candidates:
            Path(candidate["media"][0]["path"]).write_bytes(b"different motion")
        selection = reviewer.rank_candidates(self.candidates, report)
        self.assertIsNone(selection["selected_id"])
        self.assertTrue(all(not item["eligible"] for item in selection["ranking"]))

    def test_missing_review_returns_review_required(self):
        selection = reviewer.rank_candidates(self.candidates, None)
        self.assertIsNone(selection["selected_id"])
        self.assertEqual(selection["state"], "review_required")


if __name__ == "__main__":
    unittest.main()
