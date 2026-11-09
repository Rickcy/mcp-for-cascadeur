"""Visual assessment in the MCP process; native execution stays in Cascadeur."""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from .contracts import Tool
from .review_settings import public_review_status


def _reviewer():
    try:
        from .review_backend import motion_review

        return motion_review
    except ModuleNotFoundError as error:
        if error.name not in {"mcp_for_cascadeur.review_backend", "mcp_for_cascadeur.review_backend.motion_review"}:
            raise
        path = Path(__file__).resolve().parents[2] / "integrations/kimodo/motion_review.py"
        if not path.is_file():
            raise RuntimeError("The visual reviewer is missing from this installation") from None
        spec = importlib.util.spec_from_file_location("cascadeur_checkout_review", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _path(value, workspace, *, existing=True):
    original = Path(value)
    if not original.is_absolute():
        raise ValueError("Review paths must be absolute and inside the configured workspace")
    path = original.resolve()
    if not path.is_relative_to(workspace) or original.is_symlink():
        raise ValueError("Review path is outside the configured workspace or replaced by a symlink")
    if existing and not path.exists():
        raise ValueError("Review input must be an existing file")
    if existing and not path.is_file():
        raise ValueError("Review input must be an existing file")
    if not existing and (path.exists() or not path.parent.is_dir()):
        raise ValueError("Choose a new review output in an existing workspace directory")
    return path


def _read(path):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Review request or receipt exceeds 4 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _native_evidence(candidate, workspace):
    supplied = candidate.get("validation", {})
    validation = {"geometry": False, "persistence": False,
                  "constraints_required": bool(supplied.get("constraints_required"))}
    if "constraints" in supplied:
        validation["constraints"] = False
    receipt_name = candidate.get("receipt")
    if not receipt_name:
        return validation, "No native receipt: visual findings are available; automatic acceptance is disabled."
    receipt = _read(_path(receipt_name, workspace))
    scene = _path(candidate.get("scene", ""), workspace)
    if (receipt.get("phase") != "completed"
            or receipt.get("reopen_verification", {}).get("point_coordinates_verified") is not True
            or _sha(scene) != receipt.get("result_sha256")):
        return validation, "Native completion or saved-scene integrity did not verify."
    plan = _path(receipt.get("plan_path", ""), workspace)
    if _sha(plan) != receipt.get("plan_sha256"):
        return validation, "The native Point plan differs from its receipt."
    layout = _read(plan)
    validation["constraints_required"] = bool(layout.get("keypose_capture_sha256"))
    geometry = layout.get("native_cluster_validation", {})
    error = geometry.get("max_error", 1)
    if (geometry.get("frames_checked") != receipt.get("num_frames") or isinstance(error, bool)
            or not isinstance(error, (float, int)) or not math.isfinite(error) or error > 0.01):
        return validation, "The full native geometry plan did not verify."
    validation.update(geometry=True, persistence=True)
    if validation["constraints_required"] or "constraints" in supplied:
        transfer_path = Path(receipt_name).resolve().parent.parent / "transfer.json"
        transfer = _read(_path(transfer_path, workspace))
        same_scene = Path(transfer.get("scene", "")).resolve() == scene
        validation["constraints"] = same_scene and transfer.get("keypose_deviation", {}).get("accepted") is True
    return validation, "Native receipt, geometry plan and saved scene checked; visual review is advisory."


def animation_review(config, request_path, output_path=None, provider=None, model=None,
                     base_url=None, media_transport=None):
    workspace = config.workspace.resolve()
    source = _path(request_path, workspace)
    destination = _path(output_path, workspace, existing=False) if output_path else None
    request = _read(source)
    if not isinstance(request, dict) or not isinstance(request.get("candidates"), list):
        raise ValueError("Review request needs a prompt and a candidate list")
    candidates = copy.deepcopy(request["candidates"])
    if not 1 <= len(candidates) <= 4:
        raise ValueError("Review one to four candidates")
    notes = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Every review candidate must be an object")
        for medium in candidate.get("media", []):
            medium["path"] = str(_path(medium["path"], workspace))
        candidate["validation"], note = _native_evidence(candidate, workspace)
        notes[candidate.get("id", "unknown")] = note
    reviewer = _reviewer()
    report = reviewer.review_candidates(candidates, request.get("prompt", ""), provider=provider,
                                        model=model, base_url=base_url, media_transport=media_transport,
                                        max_output_tokens=8192)
    for candidate in candidates:
        candidate["validation"], notes[candidate.get("id", "unknown")] = _native_evidence(candidate, workspace)
    result = {"report": report, "selection": reviewer.rank_candidates(candidates, report),
              "native_evidence": notes, "scene_changes": False}
    if destination:
        # Exclusive output creation prevents replacing a user's previous review.
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        result["output_path"] = str(destination)
    return result


def tools(config):
    return [
        Tool("visual_review_status", "Read the visual review provider, model and key availability; never returns credentials.",
             public_review_status),
        Tool("animation_review", "Review saved motion evidence with the configured visual provider. Requires an absolute workspace JSON request with prompt and candidates. Native receipts gate automatic selection; the tool never edits scenes. API keys are configured through the CLI or environment, never tool arguments.",
             functools.partial(animation_review, config),
             {"request_path": {"type": "string", "minLength": 1},
              "output_path": {"type": "string", "minLength": 1},
              "provider": {"type": "string", "enum": ["gemini", "openai-compatible"]},
              "model": {"type": "string", "minLength": 1, "maxLength": 240},
              "base_url": {"type": "string", "minLength": 1, "maxLength": 400},
              "media_transport": {"type": "string", "enum": ["video", "frames"]}},
             ("request_path",), False),
    ]
