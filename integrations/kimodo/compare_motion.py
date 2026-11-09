"""Compare two generated SOMA77 clips without loading model weights or editing a scene.

Run inside the Kimodo container, where NumPy is already installed. The output is
JSON on stdout; no report files are created. Contact masks are diagnostics, not
ground truth, and a comparison of two clips cannot rank the models in general.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FOOT_JOINTS = (69, 70, 74, 75)  # LeftFoot, LeftToeBase, RightFoot, RightToeBase in SOMA77.
SETTINGS = (
    "prompt",
    "num_frames",
    "fps",
    "seed",
    "diffusion_steps",
    "num_samples",
    "sample_index",
    "cfg_enabled",
    "text_weight",
    "constraint_weight",
    "dataset",
    "constraints",
    "constraint_frames",
    "text_encoder",
)


def _contact_stats(horizontal_steps, mask, fps):
    velocities = horizontal_steps[mask] * fps
    count = int(mask.sum())
    return {
        "contact_slot_intervals": count,
        "total_slot_intervals": int(mask.size),
        "coverage_ratio": count / mask.size if mask.size else 0.0,
        "summed_contact_slot_seconds": count / fps,
        "mean_horizontal_speed_m_s": float(velocities.mean()) if count else None,
        "p95_horizontal_speed_m_s": float(np.percentile(velocities, 95)) if count else None,
        "max_horizontal_speed_m_s": float(velocities.max()) if count else None,
        "summed_horizontal_distance_m": float(horizontal_steps[mask].sum()),
    }


def measure_motion(motion, height_threshold=0.05):
    """Measure world-space feet in metres; exclude transitions into/out of contact."""
    if not np.isfinite(height_threshold) or height_threshold <= 0:
        raise ValueError("height_threshold must be finite and positive")
    positions = np.asarray(motion["posed_joints"], dtype=np.float64)
    roots = np.asarray(motion["root_positions"], dtype=np.float64)
    contacts = np.asarray(motion["foot_contacts"])
    fps_array = np.asarray(motion["fps"])
    if fps_array.shape != () or not np.isfinite(fps_array).item() or float(fps_array) <= 0:
        raise ValueError("fps must be a finite positive scalar")
    fps = float(fps_array)
    if positions.ndim != 3 or positions.shape[1:] != (77, 3) or len(positions) < 2:
        raise ValueError("Expected an unbatched SOMA77 clip with at least two frames")
    count = len(positions)
    if roots.shape != (count, 3) or contacts.shape not in {(count, 4), (count, 6)}:
        raise ValueError("Root positions and four/six foot contacts must match the clip length")
    if not all(np.isfinite(value).all() for value in (positions, roots, contacts)):
        raise ValueError("Motion arrays contain nonfinite values")
    if np.any((contacts < 0) | (contacts > 1)):
        raise ValueError("Foot contact values must lie in [0, 1]")
    # SOMA77 adds a ToeEnd per side. Exclude those slots instead of accidentally
    # comparing six-contact SOMA77 data with a four-contact SOMA30 metric.
    if contacts.shape[1] == 6:
        contacts = contacts[:, (0, 1, 3, 4)]
    contacts = contacts >= 0.5
    feet = positions[:, FOOT_JOINTS]
    horizontal_steps = np.linalg.norm(np.diff(feet[:, :, (0, 2)], axis=0), axis=-1)
    predicted_mask = contacts[:-1] & contacts[1:]
    toes = feet[:, (1, 3)]
    height_mask = (toes[:-1, :, 1] <= height_threshold) & (toes[1:, :, 1] <= height_threshold)
    root_delta = roots[-1] - roots[0]
    root_steps = np.linalg.norm(np.diff(roots[:, (0, 2)], axis=0), axis=-1)
    span = (count - 1) / fps
    return {
        "frame_count": count,
        "fps": fps,
        "duration_seconds": count / fps,
        "sample_span_seconds": span,
        "root_displacement_m": root_delta.tolist(),
        "root_planar_displacement_m": float(np.linalg.norm(root_delta[[0, 2]])),
        "root_planar_path_length_m": float(root_steps.sum()),
        "root_mean_planar_speed_m_s": float(root_steps.sum() / span),
        "root_vertical_range_m": float(np.ptp(roots[:, 1])),
        "minimum_foot_joint_height_m": float(feet[:, :, 1].min()),
        "foot_joint_below_y0_ratio": float((feet[:, :, 1] < 0).mean()),
        "predicted_contact_feet": _contact_stats(horizontal_steps, predicted_mask, fps),
        "height_mask_toes": _contact_stats(horizontal_steps[:, (1, 3)], height_mask, fps),
        "measurement": {
            "units": "metres, seconds",
            "skeleton": "SOMA77",
            "foot_joint_indices": list(FOOT_JOINTS),
            "selected_contact_slots": [0, 1, 3, 4] if motion["foot_contacts"].shape[1] == 6 else [0, 1, 2, 3],
            "height_threshold_m": height_threshold,
            "predicted_contact_threshold": 0.5,
            "both_interval_endpoints_must_be_in_contact": True,
            "height_mask_scope": "Two toe-base joints; predicted mask uses four heel/toe joints.",
            "contact_duration_scope": "Slot-seconds sum heel/toe slots separately, not physical stance duration.",
            "floor_scope": "Y=0 joint-centre diagnostics; not mesh penetration or a fitted floor.",
        },
    }


def _metadata(manifest_path, log_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    settings = {}
    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8")
        start = log.find("{")
        if start >= 0:
            settings, _ = json.JSONDecoder().raw_decode(log[start:])
    combined = {**manifest, **settings}
    encoder = combined.get("text_encoder")
    if isinstance(encoder, dict):
        combined["text_encoder"] = {
            key: encoder[key]
            for key in ("kind", "fallback", "quantization", "embedding_dim", "model_path")
            if key in encoder
        }
    return combined


def inspect_clip(npz_path, *, manifest_path=None, log_path=None, height_threshold=0.05):
    path = Path(npz_path).resolve()
    with np.load(path, allow_pickle=False) as source:
        motion = {name: source[name] for name in ("posed_joints", "root_positions", "foot_contacts", "fps")}
    measured = measure_motion(motion, height_threshold)
    metadata = _metadata(
        Path(manifest_path) if manifest_path else path.with_suffix(".json"),
        Path(log_path) if log_path else path.with_suffix(".log"),
    )
    for field, actual in (("num_frames", measured["frame_count"]), ("fps", measured["fps"])):
        if field in metadata and metadata[field] != actual:
            raise ValueError(f"Recorded {field} differs from the actual NPZ in {path.name}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recorded_model": metadata.get("model_name"),
        "generation_settings": {key: metadata[key] for key in SETTINGS if key in metadata},
        "metrics": measured,
    }


def compare_clips(baseline, candidate):
    left, right = baseline["generation_settings"], candidate["generation_settings"]
    missing = [key for key in SETTINGS if key not in left or key not in right]
    different = {
        key: {"baseline": left[key], "candidate": right[key]}
        for key in SETTINGS
        if key in left and key in right and left[key] != right[key]
    }
    actual_match = all(
        baseline["metrics"][key] == candidate["metrics"][key] for key in ("frame_count", "fps", "measurement")
    )
    return {
        "schema": "kimodo-motion-comparison-v1",
        "baseline": baseline,
        "candidate": candidate,
        "comparison_conditions": {
            "recorded_settings_match": not missing and not different,
            "missing_recorded_settings": missing,
            "different_recorded_settings": different,
            "measurement_conditions_match": actual_match,
            "scope": "Recorded settings only; encoder weights and software versions are not proven identical.",
        },
        "limitations": [
            "This is a comparison of these two generated clips, not a model benchmark or a quality verdict.",
            "Predicted contacts can be wrong. Read coverage alongside slip speeds; no contacts yields null speed.",
            "Height-based contacts are a heuristic. Joint centres do not establish visible mesh contact.",
            "Metrics describe source motion before Cascy retargeting; native contacts require a separate check.",
            "These metrics do not establish text alignment, naturalness, loop continuity, or every displayed frame.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--candidate-log", type=Path)
    parser.add_argument("--height-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)
    clips = []
    for prefix in ("baseline", "candidate"):
        clips.append(
            inspect_clip(
                getattr(args, prefix),
                manifest_path=getattr(args, prefix + "_manifest"),
                log_path=getattr(args, prefix + "_log"),
                height_threshold=args.height_threshold,
            )
        )
    print(json.dumps(compare_clips(*clips), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
