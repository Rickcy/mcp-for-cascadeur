import hashlib
import json
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur import review_service as service
from mcp_for_cascadeur.settings import Config


@pytest.fixture
def review_request(tmp_path, monkeypatch):
    media = tmp_path / "side.mp4"
    media.write_bytes(b"fixture-media")
    path = tmp_path / "review_request.json"
    value = {"prompt": "Walk forward", "candidates": [{"id": "candidate-01", "media": [{"path": str(media), "view": "side"}],
             "validation": {"geometry": True, "persistence": True}}]}
    path.write_text(json.dumps(value))
    seen = []
    backend = SimpleNamespace(
        review_candidates=lambda candidates, prompt, **kwargs: seen.append(candidates) or {"review": "fixture"},
        rank_candidates=lambda candidates, report: {"state": "selected" if candidates[0]["validation"]["persistence"] else "needs_revision"},
    )
    monkeypatch.setattr(service, "_reviewer", lambda: backend)
    return SimpleNamespace(config=Config(workspace=tmp_path), path=path, value=value, seen=seen, backend=backend)


def test_review_does_not_trust_unverified_native_flags(review_request):
    result = service.animation_review(review_request.config, str(review_request.path))
    assert result["selection"]["state"] == "needs_revision"
    assert result["scene_changes"] is False
    assert review_request.seen[0][0]["validation"]["geometry"] is False


def test_output_and_workspace_are_checked_before_external_request(review_request, tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text("preserve")
    with pytest.raises(ValueError, match="new review output"):
        service.animation_review(review_request.config, str(review_request.path), str(existing))
    assert existing.read_text() == "preserve" and review_request.seen == []
    with pytest.raises(ValueError, match="outside"):
        service.animation_review(review_request.config, str(tmp_path.parent / "outside.json"))
    assert review_request.seen == []


def test_receipts_and_saved_scene_gate_selection_and_are_rechecked(review_request, tmp_path):
    scene = tmp_path / "result.casc"
    scene.write_bytes(b"saved-scene")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"native_cluster_validation": {"frames_checked": 120, "max_error": 0.0}}))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"phase": "completed", "num_frames": 120, "plan_path": str(plan),
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "result_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "reopen_verification": {"point_coordinates_verified": True}}))
    review_request.value["candidates"][0].update(receipt=str(receipt), scene=str(scene))
    review_request.path.write_text(json.dumps(review_request.value))
    result = service.animation_review(review_request.config, str(review_request.path))
    assert result["selection"]["state"] == "selected"
    def changed(candidates, prompt, **kwargs):
        scene.write_bytes(b"changed-during-review")
        return {}
    review_request.backend.review_candidates = changed
    result = service.animation_review(review_request.config, str(review_request.path))
    assert result["selection"]["state"] == "needs_revision"


def test_output_is_explicit_and_contains_no_native_edits(review_request, tmp_path):
    output = tmp_path / "review.json"
    result = service.animation_review(review_request.config, str(review_request.path), str(output), provider="openai-compatible", model="Qwen3-VL")
    assert result["output_path"] == str(output)
    assert json.loads(output.read_text())["scene_changes"] is False


def test_tool_contracts_never_accept_api_keys(review_request):
    definitions = {tool.name: tool for tool in service.tools(review_request.config)}
    assert set(definitions) == {"animation_review", "visual_review_status"}
    assert definitions["visual_review_status"].read_only
    with pytest.raises(ValueError):
        definitions["animation_review"].validate({"request_path": str(review_request.path), "api_key": "do-not-pass-here"})
