"""Export exact SOMA FK data for Cascadeur retargeting without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

CASCADEUR_MAPPING_CANDIDATES = {
    "pelvis": ["Hips"],
    "stomach": ["Spine1", "Spine2"],
    "chest": ["Chest"],
    "neck": ["Neck1", "Neck2"],
    "head": ["Head"],
    "clavicle_l": ["LeftShoulder"],
    "clavicle_r": ["RightShoulder"],
    "arm_l": ["LeftArm"],
    "arm_r": ["RightArm"],
    "forearm_l": ["LeftForeArm"],
    "forearm_r": ["RightForeArm"],
    "hand_l": ["LeftHand"],
    "hand_r": ["RightHand"],
    "thigh_l": ["LeftLeg"],
    "thigh_r": ["RightLeg"],
    "calf_l": ["LeftShin"],
    "calf_r": ["RightShin"],
    "foot_l": ["LeftFoot"],
    "foot_r": ["RightFoot"],
    "toe_l": ["LeftToeBase"],
    "toe_r": ["RightToeBase"],
}


def yaw_basis(degrees):
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]


def prepare_motion(source, output, yaw_degrees=0.0):
    # Only this short-lived conversion process loses GPU visibility. The running
    # encoder and generation backend keep their CUDA devices and loaded models.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from kimodo.skeleton import SOMASkeleton77

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not math.isfinite(yaw_degrees):
        raise ValueError("Yaw must be finite")
    with np.load(source, allow_pickle=False) as data:
        local = np.asarray(data["local_rot_mats"], dtype=np.float32)
        roots = np.asarray(data["root_positions"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data else 30.0
        saved_positions = np.asarray(data["posed_joints"]) if "posed_joints" in data else None
        saved_rotations = np.asarray(data["global_rot_mats"]) if "global_rot_mats" in data else None
    if local.ndim == 5 and local.shape[0] == 1:
        local = local[0]
    if roots.ndim == 3 and roots.shape[0] == 1:
        roots = roots[0]
    if local.ndim != 4 or local.shape[1:] != (77, 3, 3):
        raise ValueError(f"Expected SOMA77 local rotations, got {local.shape}")
    if roots.shape != (len(local), 3) or len(local) < 2:
        raise ValueError("Root positions and local rotations must contain the same animation frames")
    if not np.isfinite(local).all() or not np.isfinite(roots).all():
        raise ValueError("The source motion contains nonfinite transforms")
    if fps != 30.0:
        raise ValueError(f"Expected the checkpoint's native 30 fps, got {fps}")
    skeleton = SOMASkeleton77().cpu()
    with torch.inference_mode():
        rotations, positions, _ = skeleton.fk(torch.from_numpy(local), torch.from_numpy(roots))
        identity_local = torch.eye(3).repeat(1, 77, 1, 1)
        neutral_root = skeleton.neutral_joints[skeleton.root_idx].reshape(1, 3)
        rest_rotations, rest_positions, _ = skeleton.fk(identity_local, neutral_root)
    positions = positions.cpu().numpy()
    rotations = rotations.cpu().numpy()
    rest_positions = rest_positions.cpu().numpy()[0]
    rest_rotations = rest_rotations.cpu().numpy()[0]
    consistency = {}
    for name, saved, recomputed in (
        ("posed_joints_max_error_m", saved_positions, positions),
        ("global_rotations_max_error", saved_rotations, rotations),
    ):
        if saved is not None:
            if saved.ndim == recomputed.ndim + 1 and saved.shape[0] == 1:
                saved = saved[0]
            if saved.shape != recomputed.shape:
                raise ValueError(f"Stored {name} has a different shape from forward kinematics")
            error = float(np.max(np.abs(saved - recomputed)))
            if not math.isfinite(error) or error > 1e-3:
                raise ValueError(f"Stored animation and recomputed FK disagree: {name}={error}")
            consistency[name] = error
    basis = np.asarray(yaw_basis(yaw_degrees), dtype=np.float64)
    positions_cm = positions @ basis.T * 100.0
    rest_positions_cm = rest_positions @ basis.T * 100.0
    rotations = basis @ rotations @ basis.T
    rest_rotations = basis @ rest_rotations @ basis.T
    root_index = int(skeleton.root_idx)
    root_delta = positions_cm[-1, root_index] - positions_cm[0, root_index]
    payload = {
        "format": "kimodo-retarget-motion-v1",
        "source_npz_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fps": fps,
        "num_frames": len(local),
        "units": "centimeters",
        "up_axis": "Y",
        "rotation_format": "matrix3x3_row_major",
        "source_coordinates": {"units": "meters", "up_axis": "Y", "scale_to_cm": 100.0},
        "basis_from_kimodo": basis.tolist(),
        "yaw_degrees": yaw_degrees,
        "joint_names": list(skeleton.bone_order_names),
        "parents": skeleton.joint_parents.cpu().tolist(),
        "root_index": root_index,
        "rest_pose": "SOMA neutral joints with identity local rotations",
        "rest_positions": rest_positions_cm.tolist(),
        "rest_global_rotations": rest_rotations.tolist(),
        "positions": positions_cm.tolist(),
        "rotations": rotations.tolist(),
        "root_displacement_cm": root_delta.tolist(),
        "mapping_candidates": CASCADEUR_MAPPING_CANDIDATES,
        "fk_consistency": consistency,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(output)
    return {
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "num_frames": len(local),
        "num_joints": 77,
        "root_displacement_cm": root_delta.tolist(),
        "fk_consistency": consistency,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Generated Kimodo NPZ animation")
    parser.add_argument("output", help="New JSON file for the Cascadeur bridge")
    parser.add_argument("--yaw-degrees", type=float, default=0.0)
    arguments = parser.parse_args()
    print(json.dumps(prepare_motion(arguments.source, arguments.output, arguments.yaw_degrees)))
