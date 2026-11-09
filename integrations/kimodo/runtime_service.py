"""Validation and artifact metadata for the isolated Kimodo runtime.

This module deliberately has no Torch dependency so requests and health responses
can be tested before installing the inference environment.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import secrets
import urllib.request
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path

MODEL_NAME = "Kimodo-SOMA-RP-v1.2"
V12_REVISION = "0e3c89024d896064b6851201373f7834784a386b"
V12_WEIGHTS_SHA256 = "fdcef624140a12c0a4840192ce0b75cafb4e73228a893da3864b4a990d3e919e"
MAX_CONSTRAINT_BYTES = 64 * 1024 * 1024
INFERENCE_SEGMENT_FRAMES = 240


def generation_segments(num_frames):
    """Bound attention memory, preserving the exact requested total duration.

    Long clips use Kimodo's native multiprompt continuation, including its body
    history, heading and root translation. Balanced segments avoid a tiny tail.
    This is an inference window, not a limit on an animation's duration.
    """
    if type(num_frames) is not int or num_frames < 2:
        raise ValueError("num_frames must be an integer of at least 2")
    count = (num_frames + INFERENCE_SEGMENT_FRAMES - 1) // INFERENCE_SEGMENT_FRAMES
    length, extra = divmod(num_frames, count)
    return [length + (index < extra) for index in range(count)]


def configured_model_name(dataset="RP"):
    if dataset != "RP":
        raise ValueError("Only the RP dataset for Kimodo-SOMA-RP-v1.2 is supported")
    return MODEL_NAME


@lru_cache(maxsize=4)
def _checkpoint_sha256(path, size, modified_ns):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_details(dataset="RP"):
    """Verify the selected local checkpoint once per file revision."""
    name = configured_model_name(dataset)
    folder = Path(os.environ.get("CHECKPOINT_DIR", "/models/checkpoints")) / name
    required = [folder / "config.yaml", folder / "model.safetensors"]
    required.extend(
        folder / "stats/motion" / area / (stat + ".npy")
        for area in ("body", "global_root", "local_root")
        for stat in ("mean", "std")
    )
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Required checkpoint file is missing or empty: {path}")
    weights = folder / "model.safetensors"
    info = weights.stat()
    sha256 = _checkpoint_sha256(str(weights), info.st_size, info.st_mtime_ns)
    if sha256 != V12_WEIGHTS_SHA256:
        raise RuntimeError("Kimodo-SOMA-RP-v1.2 weights do not match the pinned NVIDIA checkpoint")
    return {
        "name": name,
        "repo_id": "nvidia/" + name,
        "version": "v1.2",
        "revision": V12_REVISION,
        "weights_sha256": sha256,
    }


def encoder_details(encoder):
    """Describe loaded weights, rejecting the archive's random-vector fallback."""
    if type(encoder).__name__ != "LLM2VecEncoder":
        raise RuntimeError("The text encoder must be the real LLM2Vec encoder")
    inner = getattr(getattr(encoder, "model", None), "model", None)
    requested = getattr(encoder, "quantization", "none")
    loaded_4bit = bool(getattr(inner, "is_loaded_in_4bit", False))
    loaded_8bit = bool(getattr(inner, "is_loaded_in_8bit", False))
    if requested == "4bit" and not loaded_4bit:
        raise RuntimeError("Llama was requested in 4-bit mode but quantized weights are not loaded")
    if requested == "8bit" and not loaded_8bit:
        raise RuntimeError("Llama was requested in 8-bit mode but quantized weights are not loaded")
    return {
        "kind": "llm2vec",
        "fallback": False,
        "model_path": os.environ.get("TEXT_ENCODER_MODEL_PATH", ""),
        "quantization": "4bit" if loaded_4bit else "8bit" if loaded_8bit else "none",
        "embedding_dim": int(encoder.llm_dim),
        "device": str(encoder.get_device()),
    }


def fbx_wrapper_transform(template):
    """Keep the template's axis and meter-to-centimeter wrapper conversion."""
    objects = next(node[3] for node in template if node[0] == "Objects")
    wrapper = next(
        node for node in objects if node[0] == "Model" and node[1][1].split("::")[0] == "KimodoArmature"
    )
    properties = next(node[3] for node in wrapper[3] if node[0] == "Properties70")
    values = {node[1][0]: node[1][4:7] for node in properties if node[0] == "P"}
    return tuple(values[name] for name in ("Lcl Translation", "Lcl Rotation", "Lcl Scaling"))


def verify_text_encoder_health(url=None):
    endpoint = (url or os.environ.get("TEXT_ENCODER_URL", "http://text-encoder:9550/")).rstrip("/")
    with urllib.request.urlopen(endpoint + "/health", timeout=5) as response:
        payload = json.load(response)
    details = payload.get("encoder", {})
    if (
        payload.get("service") != "text-encoder"
        or not payload.get("ok")
        or details.get("kind") != "llm2vec"
        or details.get("fallback") is not False
        or details.get("embedding_dim") != 4096
        or not details.get("model_path")
    ):
        raise RuntimeError("A healthy local LLM2Vec encoder is required; hash fallback is disabled")
    expected = os.environ.get("TEXT_ENCODER_REQUIRED_QUANTIZATION", "4bit")
    if expected and details.get("quantization") != expected:
        raise RuntimeError(f"Expected a {expected} LLM2Vec encoder, got {details.get('quantization')}")
    return details


def _exchange_path(value, default, root):
    candidate = Path(value or default)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Input and output paths must stay inside KIMODO_EXCHANGE_ROOT")
    return candidate


def validate_generation_request(request, exchange_root=None):
    if not isinstance(request, dict):
        raise ValueError("The request must be a JSON object")
    prompt = request.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError("prompt must be a nonempty string of at most 4000 characters")
    root = Path(exchange_root or os.environ.get("KIMODO_EXCHANGE_ROOT", "/exchange")).resolve()
    stem = "motions/" + uuid.uuid4().hex
    payload = {
        "prompt": prompt.strip(),
        "num_frames": 120,
        "fps": 30.0,
        "num_samples": 1,
        "sample_index": 0,
        "diffusion_steps": 50,
        "dataset": "RP",
        "seed": 42,
        "cfg_enabled": True,
        "text_weight": 2.0,
        "constraint_weight": 2.0,
    }
    payload.update({key: request[key] for key in payload if key in request})
    generation_segments(payload["num_frames"])
    for name, minimum, maximum in (
        ("num_samples", 1, 4),
        ("diffusion_steps", 1, 200),
        ("sample_index", 0, 3),
    ):
        value = payload[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    if payload["sample_index"] >= payload["num_samples"]:
        raise ValueError("sample_index must identify one of the requested samples")
    if isinstance(payload["fps"], bool) or payload["fps"] != 30:
        raise ValueError("These Kimodo checkpoints generate at 30 fps; resampling must be explicit")
    payload["fps"] = 30.0
    expected_model = configured_model_name(payload["dataset"])
    if request.get("model_name", expected_model) != expected_model:
        raise ValueError(
            "Requested model_name differs from the running backend; only Kimodo-SOMA-RP-v1.2 is supported"
        )
    payload["model_name"] = expected_model
    if type(payload["cfg_enabled"]) is not bool:
        raise ValueError("cfg_enabled must be a boolean")
    if payload["seed"] is not None and (type(payload["seed"]) is not int or not 0 <= payload["seed"] < 2**32):
        raise ValueError("seed must be null or an unsigned 32-bit integer")
    for name in ("text_weight", "constraint_weight"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if not 0 <= value <= 10:
            raise ValueError(f"{name} must be between 0 and 10")
    for name, extension in (("output", ".npz"), ("output_fbx", ".fbx"), ("log_path", ".log")):
        path = _exchange_path(request.get(name), stem + extension, root)
        if path.suffix.lower() != extension:
            raise ValueError(f"{name} must use the {extension} extension")
        if path.exists():
            raise ValueError(f"{name} already exists; choose a new artifact path")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload[name] = str(path)
    payload["constraints"] = ""
    payload["constraints_sha256"] = None
    if request.get("constraints"):
        path = _exchange_path(request["constraints"], "", root)
        if path.suffix.lower() != ".npz" or not path.is_file():
            raise ValueError("constraints must name an existing NPZ inside the exchange directory")
        payload["constraints"] = str(path)
        payload["constraints_sha256"] = validate_constraints(path, payload["num_frames"])
    requested_hash = request.get("constraints_sha256")
    if requested_hash is not None and requested_hash != payload["constraints_sha256"]:
        raise ValueError("Constraint input differs from the caller's captured checksum")
    payload["candidate_seed_base"] = payload["seed"] if payload["seed"] is not None else secrets.randbits(32)
    payload["candidates_manifest"] = str(Path(payload["output_fbx"]).with_suffix(".candidates.json"))
    for candidate in candidate_payloads(payload):
        for name in ("output", "output_fbx"):
            if Path(candidate[name]).exists():
                raise ValueError(f"Candidate {name} already exists; choose a new artifact path")
        if Path(candidate["output_fbx"]).with_suffix(".json").exists():
            raise ValueError("A candidate manifest already exists; choose a new artifact path")
    if Path(payload["candidates_manifest"]).exists():
        raise ValueError("A candidates manifest already exists; choose a new artifact path")
    return payload


def _constraint_bytes(path, expected_sha256=None):
    """Read one bounded byte snapshot; hash and parsing use these same bytes."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CONSTRAINT_BYTES + 1)
    if not raw or len(raw) > MAX_CONSTRAINT_BYTES:
        raise ValueError("Constraint NPZ must contain at most 64 MiB")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Constraint input changed after it was accepted")
    return raw, digest


def _constraint_arrays(raw, num_frames):
    """Validate archive bounds and array headers before allocating NumPy arrays."""
    import numpy as np

    required = ("frame_indices", "global_positions", "global_rot_mats")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if (len(infos) > 64 or len({info.filename for info in infos}) != len(infos)
                or sum(info.file_size for info in infos) > MAX_CONSTRAINT_BYTES):
            raise ValueError("Expanded constraint NPZ exceeds supported bounds or has duplicate entries")
        for name in required:
            info = archive.getinfo(name + ".npy")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Constraint arrays require NPY version 1 or 2")
                size = math.prod(shape) * dtype.itemsize
                if dtype.hasobject or dtype.kind not in "iuf" or size > MAX_CONSTRAINT_BYTES:
                    raise ValueError("Constraint arrays must use bounded numeric data")
                if stream.tell() + size != info.file_size:
                    raise ValueError("Constraint array header and stored byte count disagree")
                if name == "frame_indices" and (len(shape) != 1 or not 0 < shape[0] <= num_frames):
                    raise ValueError("Constraint frames exceed the generated interval")
                if name == "global_positions" and (len(shape) != 3 or not 0 < shape[0] <= num_frames
                                                   or shape[1:] not in {(30, 3), (77, 3)}):
                    raise ValueError("Constraint positions must contain 30 or 77 SOMA joints per frame")
                if name == "global_rot_mats" and (len(shape) != 4 or not 0 < shape[0] <= num_frames
                                                  or shape[1:] not in {(30, 3, 3), (77, 3, 3)}):
                    raise ValueError("Constraint rotations exceed the supported SOMA skeleton")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        frames = np.asarray(archive["frame_indices"])
        positions = np.asarray(archive["global_positions"])
        rotations = np.asarray(archive["global_rot_mats"])
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu":
        raise ValueError("Constraint frames must be a nonempty integer vector")
    if np.any(frames < 0) or np.any(frames >= num_frames) or np.any(np.diff(frames.astype(np.int64)) <= 0):
        raise ValueError("Constraint frames must be unique, increasing and inside the generated interval")
    if positions.shape not in ((len(frames), 30, 3), (len(frames), 77, 3)):
        raise ValueError("Constraint positions must contain 30 or 77 SOMA joints per frame")
    if rotations.shape != positions.shape[:2] + (3, 3):
        raise ValueError("Constraint rotations do not match the position array")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("Constraint transforms must be finite")
    if not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-3, rtol=0):
        raise ValueError("Constraint rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3, rtol=0):
        raise ValueError("Constraint rotations must be proper rotations")
    return frames.astype(np.int64), positions.astype(np.float32), rotations.astype(np.float32)


def validate_constraints(path, num_frames):
    """Pin validated full-body input bytes before accepting a GPU job."""
    raw, digest = _constraint_bytes(path)
    _constraint_arrays(raw, num_frames)
    return digest


def load_constraints(path, *, expected_sha256, num_frames):
    """Parse the exact bytes whose pinned hash is attached to the accepted job."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)):
        raise ValueError("A pinned constraint checksum is required before model loading")
    raw, _digest = _constraint_bytes(path, expected_sha256)
    return _constraint_arrays(raw, num_frames)


def candidate_payloads(payload):
    """Use one-sample calls with explicit seeds; keep the legacy single-clip paths."""
    base_seed = payload["candidate_seed_base"]
    candidates = []
    for index in range(payload["num_samples"]):
        candidate = {
            **payload,
            "candidate_index": index,
            "candidate_count": payload["num_samples"],
            "seed": (base_seed + index) % 2**32,
            "num_samples": 1,
            "sample_index": 0,
        }
        if payload["num_samples"] > 1:
            for name in ("output", "output_fbx"):
                path = Path(payload[name])
                candidate[name] = str(path.with_name(f"{path.stem}.candidate-{index + 1:02d}{path.suffix}"))
        candidates.append(candidate)
    return candidates


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_candidate_set(payload, job, generate_one, save_one):
    """Persist each complete candidate before starting the next GPU calculation.

    The index survives a backend restart. It is an outcome record, not a promise
    to resume diffusion. Finished candidates remain usable after a later failure.
    """
    root = Path(os.environ.get("KIMODO_EXCHANGE_ROOT", "/exchange")).resolve()
    index_path = Path(payload["candidates_manifest"])
    record = {
        "format": "cascadeur-kimodo-candidates-v1",
        "status": "running",
        "prompt": payload["prompt"],
        "num_frames": payload["num_frames"],
        "fps": payload["fps"],
        "num_candidates": payload["num_samples"],
        "seed_base": payload["candidate_seed_base"],
        "seed_policy": "(seed_base + candidate_index) modulo 2**32; sequential single-sample calls",
        "selected_index": payload["sample_index"],
        "selection_method": "requested_index",
        "candidates": [],
    }
    job.artifacts = {
        "candidates_manifest": index_path.relative_to(root).as_posix(),
        "candidates": [],
    }
    _write_json(index_path, record)
    try:
        for candidate in candidate_payloads(payload):
            index = candidate["candidate_index"]
            job.emit_status(f"Candidate {index + 1}/{payload['num_samples']} (seed {candidate['seed']})")
            motion = generate_one(candidate)
            try:
                artifacts = save_one(candidate, motion)
            finally:
                del motion
            row = {"index": index, "seed": candidate["seed"], **artifacts}
            record["candidates"].append(row)
            # Replace the published list after the complete candidate is durable.
            _write_json(index_path, record)
            job.artifacts["candidates"] = list(record["candidates"])
            if index == payload["sample_index"]:
                job.artifacts.update(artifacts)
        record["status"] = "completed"
        _write_json(index_path, record)
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        _write_json(index_path, record)
        raise
    return job.artifacts


def write_motion_manifest(payload, motion, model_name, encoder):
    roots = motion["root_positions"]
    if len(roots) != payload["num_frames"] or len(motion["local_rot_mats"]) != len(roots):
        raise RuntimeError("Generated frame count does not match the request")
    checkpoint = checkpoint_details(payload["dataset"])
    if model_name.lower() != checkpoint["name"].lower():
        raise RuntimeError("The loaded model differs from the selected checkpoint")
    start = [float(value) for value in roots[0]]
    end = [float(value) for value in roots[-1]]
    if not all(math.isfinite(value) for value in start + end):
        raise RuntimeError("Generated root positions are not finite")
    distance = math.hypot(end[0] - start[0], end[2] - start[2])
    root = Path(os.environ.get("KIMODO_EXCHANGE_ROOT", "/exchange")).resolve()
    artifacts = {}
    for name in ("output", "output_fbx", "log_path"):
        path = Path(payload[name]).resolve()
        artifacts[name] = path.relative_to(root).as_posix()
    fbx = Path(payload["output_fbx"])
    if not fbx.is_file() or fbx.stat().st_size == 0:
        raise RuntimeError("The FBX exporter did not produce an animation file")
    npz = Path(payload["output"])
    if not npz.is_file() or npz.stat().st_size == 0:
        raise RuntimeError("The motion exporter did not produce an NPZ animation file")
    with npz.open("rb") as stream:
        npz_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    constraints_sha256 = payload["constraints_sha256"]
    if payload["constraints"]:
        # Fail if the caller changed the source while inference was running.
        # The manifest always uses the checksum pinned before inference.
        _constraint_bytes(payload["constraints"], constraints_sha256)
    sha256 = hashlib.sha256()
    with fbx.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
    manifest_path = fbx.with_suffix(".json")
    manifest = {
        "format": "cascadeur-kimodo-motion-v1",
        "fps": payload["fps"],
        "num_frames": len(roots),
        "duration_seconds": len(roots) / payload["fps"],
        "model_name": checkpoint["name"],
        "model_checkpoint": checkpoint,
        "prompt": payload["prompt"],
        "seed": payload["seed"],
        "candidate_index": payload.get("candidate_index", 0),
        "candidate_count": payload.get("candidate_count", 1),
        "diffusion_steps": payload["diffusion_steps"],
        "generation_segments": generation_segments(payload["num_frames"]),
        "continuation": "native body-history constraints with root/heading continuity",
        "guidance": {
            "enabled": payload["cfg_enabled"],
            "text_weight": payload["text_weight"],
            "constraint_weight": payload["constraint_weight"],
        },
        "constraints_sha256": constraints_sha256,
        "npz_sha256": npz_sha256,
        "text_encoder": encoder,
        "root_motion": distance > 0.01,
        "root_displacement_m": [end[index] - start[index] for index in range(3)],
        "root_planar_distance_m": distance,
        "sha256": sha256.hexdigest(),
        **artifacts,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    artifacts["manifest"] = manifest_path.relative_to(root).as_posix()
    return artifacts
