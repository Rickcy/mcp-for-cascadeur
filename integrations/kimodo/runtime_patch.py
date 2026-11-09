"""Apply the reviewed runtime changes to the supplied Kimodo source tree."""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

PATCH_VERSION = "cascadeur-runtime-v7"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Kimodo source changed: expected one occurrence of {old[:70]!r}")
    return source.replace(old, new, 1)


def replace_function(source, name, replacement, class_name=None):
    tree = ast.parse(source)
    scope = tree.body
    if class_name:
        scope = next(
            node.body for node in scope if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    matches = [node for node in scope if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Kimodo source changed: expected one {class_name or ''}.{name}")
    target = matches[0]
    lines = source.splitlines(keepends=True)
    lines[target.lineno - 1 : target.end_lineno] = [replacement.rstrip() + "\n"]
    result = "".join(lines)
    ast.parse(result)
    return result


def replace_assignment(source, name, value):
    matches = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Kimodo source changed: expected one assignment to {name}")
    target = matches[0]
    lines = source.splitlines(keepends=True)
    lines[target.lineno - 1 : target.end_lineno] = [f"{name} = {value}\n"]
    result = "".join(lines)
    ast.parse(result)
    return result


def patch_encoder(source):
    source = replace_function(
        source,
        "_resolve_premerged_text_encoder_path",
        """def _resolve_premerged_text_encoder_path(base_model_name_or_path, peft_model_name_or_path):
    configured = os.environ.get("TEXT_ENCODER_MODEL_PATH")
    if not configured:
        raise RuntimeError("TEXT_ENCODER_MODEL_PATH must name the local merged LLM2Vec directory")
    candidate = Path(configured).resolve()
    if not _has_model_weights(candidate) or not (candidate / "llm2vec_config.json").is_file():
        raise RuntimeError(f"The merged LLM2Vec encoder is incomplete: {candidate}")
    return candidate.as_posix()
""",
    )
    source = replace_once(
        source, '"cpu": "64GiB",', '"cpu": os.environ.get("TEXT_ENCODER_MAX_CPU_MEMORY", "8GiB"),'
    )
    source = replace_once(
        source,
        "encoded_text = self.model.encode(text, batch_size=len(text), show_progress_bar=False)",
        "encoded_text = self.model.encode(text, batch_size=1, show_progress_bar=False)",
    )
    return source


def patch_llm_loader(source):
    return replace_function(
        source,
        "_quantized_load_without_allocator_warmup",
        """def _quantized_load_without_allocator_warmup(enabled):
    if not enabled:
        yield
        return
    from transformers import modeling_utils

    # Transformers 5 queues materialized BF16 tensors before converting them to
    # 4-bit. Sequential loading bounds the peak to the current weight tensor.
    previous_async = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    original_warmup = getattr(modeling_utils, "caching_allocator_warmup", None)
    if original_warmup is not None:
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    try:
        yield
    finally:
        if original_warmup is not None:
            modeling_utils.caching_allocator_warmup = original_warmup
        if previous_async is None:
            os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
        else:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = previous_async
""",
    )


def patch_text_server(source):
    source = replace_once(
        source,
        "import numpy as np",
        "import numpy as np\n\nfrom kimodo.cascadeur_runtime import encoder_details",
    )
    source = replace_function(
        source,
        "_build_encoder_with_fallback",
        """def _build_encoder_with_fallback(text_encoder_name):
    if text_encoder_name != "llm2vec":
        raise RuntimeError("Only the real local LLM2Vec encoder is supported")
    if os.getenv("TEXT_ENCODER_FORCE_FALLBACK", "0").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("Hash fallback is disabled")
    text_encoder, display_name = _build_text_encoder(text_encoder_name)
    details = encoder_details(text_encoder)
    return text_encoder, f"{display_name} ({details['quantization']})"
""",
    )
    source = replace_once(
        source,
        '"display_name": self.server.display_name,',
        '"display_name": self.server.display_name,\n                    "encoder": encoder_details(self.server.text_encoder),\n                    "encoder_kind": encoder_details(self.server.text_encoder)["kind"],\n                    "quantization": encoder_details(self.server.text_encoder)["quantization"],',
    )
    return source


def patch_model_registry(source):
    source = replace_assignment(source, "KIMODO_REPO_IDS", '["nvidia/Kimodo-SOMA-RP-v1.2"]')
    source = replace_assignment(source, "TMR_REPO_IDS", "[]")
    source = replace_once(source, r"-(RP|SEED)-v(\d+)$", r"-(RP|SEED)-v(\d+(?:\.\d+)*)$")
    source = replace_function(
        source,
        "_build_registry",
        """def _build_registry():
    infos = []
    for repo_id in KIMODO_REPO_IDS + TMR_REPO_IDS:
        info = _parse_repo_id(repo_id)
        if info is None:
            raise ValueError(f"Invalid model repo ID: {repo_id}")
        infos.append(info)
    names = {}
    for info in infos:
        base_key = info.short_key
        info.short_key = f"{base_key}-{info.version}"
        names[info.short_key] = info.repo_id
        names[base_key] = info.repo_id
    return infos, names, list(names)
""",
    )
    # Every version sort uses numeric components, including demo/version helpers.
    old = 'if v.startswith("v") and v[1:].isdigit():\n            return int(v[1:])\n        return 0'
    new = 'if v.startswith("v") and all(p.isdigit() for p in v[1:].split(".")):\n            return tuple(int(p) for p in v[1:].split("."))\n        return (0,)'
    if source.count(old) != 3:
        raise RuntimeError("Kimodo registry version sorting changed")
    source = source.replace(old, new)
    if source.count(r"v(\d+)") != 2:
        raise RuntimeError("Kimodo registry version parsing changed")
    source = source.replace(r"v(\d+)", r"v(\d+(?:\.\d+)*)")
    return source


def patch_backend(source):
    source = replace_once(source, 'choices=["RP", "SEED", ""]', 'choices=["RP", ""]')
    source = replace_once(
        source,
        "import numpy as np",
        "import numpy as np\n\nfrom kimodo.cascadeur_runtime import (\n    validate_generation_request, verify_text_encoder_health, write_motion_manifest,\n    configured_model_name, checkpoint_details, generate_candidate_set,\n    load_constraints as load_verified_constraints,\n)",
    )
    source = replace_once(
        source,
        'model_name = api["kimodo_short_key_for_skeleton_dataset"]("SOMA", dataset)',
        "model_name = configured_model_name(dataset)\n            checkpoint_details(dataset)",
    )
    source = replace_once(
        source,
        '                "loaded_datasets": loaded_datasets,',
        '                "loaded_datasets": loaded_datasets,\n                "model_checkpoint": checkpoint_details(),',
    )
    source = replace_function(
        source,
        "load_constraints",
        """def load_constraints(path, *, expected_sha256, num_frames):
    return load_verified_constraints(path, expected_sha256=expected_sha256, num_frames=num_frames)
""",
    )
    source = replace_function(
        source,
        "run_generation_job",
        """def run_generation_job(state, job):
    api = _kimodo_api()
    payload = job.payload
    constraint = None
    try:
        encoder = verify_text_encoder_health()
        emit_memory_snapshot(job, "Before job")
        if payload["constraints"]:
            frame_indices, positions, rotations = load_constraints(
                payload["constraints"], expected_sha256=payload["constraints_sha256"],
                num_frames=payload["num_frames"],
            )
        else:
            frame_indices = np.empty(0, dtype=np.int64)
        job._write_log(json.dumps({
            **payload,
            "constraint_frames": frame_indices.tolist(),
            "execution": "sequential candidates, one sample per model call",
        }, indent=2))
        model, resolved_name = state.get_or_load_model(payload["dataset"], job.emit_status)
        if payload["constraints"]:
            positions, rotations = adapt_constraints_to_model_skeleton(model.skeleton, positions, rotations)
            constraint = api["FullBodyConstraintSet"](
                model.skeleton,
                torch.from_numpy(frame_indices).long().to(state.device),
                torch.from_numpy(positions).to(device=state.device, dtype=torch.float32),
                torch.from_numpy(rotations).to(device=state.device, dtype=torch.float32),
            )
        cfg_kwargs = {"cfg_type": "nocfg"}
        if payload["cfg_enabled"]:
            cfg_kwargs = {
                "cfg_type": "separated",
                "cfg_weight": [float(payload["text_weight"]), float(payload["constraint_weight"])],
            }

        def generate_one(candidate):
            api["seed_everything"](candidate["seed"])

            def progress_bar(iterable):
                total = len(iterable)
                for step, value in enumerate(iterable, 1):
                    job.emit_progress(candidate["candidate_index"] * total + step, payload["num_samples"] * total)
                    yield value

            with torch.inference_mode():
                output = model(
                    candidate["prompt"], candidate["num_frames"],
                    constraint_lst=[constraint] if constraint is not None else [],
                    num_denoising_steps=candidate["diffusion_steps"],
                    num_samples=1, multi_prompt=False, post_processing=True,
                    return_numpy=True, progress_bar=progress_bar, **cfg_kwargs,
                )
            single = select_sample(output, 0, 1)
            single["fps"] = candidate["fps"]
            return single

        def save_one(candidate, single):
            job.emit_status(f"Saving candidate {candidate['candidate_index'] + 1}/{payload['num_samples']}")
            api["save_kimodo_npz"](candidate["output"], single)
            local = np.asarray(single["local_rot_mats"], dtype=np.float32)
            roots = np.asarray(single["root_positions"], dtype=np.float32)
            api["save_motion_fbx"](
                candidate["output_fbx"], torch.from_numpy(local), torch.from_numpy(roots),
                skeleton=resolve_fbx_skeleton(model.skeleton, local),
                fps=candidate["fps"], include_mesh=False,
            )
            return write_motion_manifest(candidate, single, resolved_name, encoder)

        generate_candidate_set(payload, job, generate_one, save_one)
        emit_memory_snapshot(job, "After job")
        job.succeed()
    except Exception as error:
        job.fail(str(error), traceback.format_exc())
    finally:
        del constraint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state.finish_job(job.job_id)
""",
    )
    source = replace_once(
        source,
        '                "log_path": self.log_path,',
        '                "log_path": self.log_path,\n                "artifacts": getattr(self, "artifacts", {}),',
    )
    source = replace_function(
        source,
        "do_GET",
        """    def do_GET(self):
        if self.path == "/health":
            try:
                encoder = verify_text_encoder_health()
                snapshot = self.server.backend_state.snapshot()
                snapshot["text_encoder"] = encoder
                snapshot["ready"] = not snapshot["warming_up"] and not snapshot["warmup_error"]
                self._write_json(HTTPStatus.OK, snapshot)
            except Exception as error:
                self._write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "ready": False, "error": str(error)})
            return
        if self.path.startswith("/jobs/"):
            try:
                job_id = int(self.path.rsplit("/", 1)[-1])
            except ValueError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid job id"})
                return
            job = self.server.backend_state.jobs.get(job_id)
            if job is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown job"})
                return
            self._write_json(HTTPStatus.OK, {"ok": True, "job": job.snapshot()})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown endpoint"})
""",
        "RequestHandler",
    )
    source = replace_function(
        source,
        "do_POST",
        """    def do_POST(self):
        if self.path in {"/generate", "/jobs"}:
            try:
                payload = validate_generation_request(self._read_json())
                verify_text_encoder_health()
                job = self.server.backend_state.create_job(payload)
                threading.Thread(target=run_generation_job, args=(self.server.backend_state, job), daemon=True).start()
                self._write_json(HTTPStatus.ACCEPTED, {"ok": True, "job_id": job.job_id})
            except RuntimeError as error:
                self._write_json(HTTPStatus.CONFLICT, {"ok": False, "error": str(error)})
            except Exception as error:
                self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
            return
        if self.path == "/shutdown":
            self._write_json(HTTPStatus.OK, {"ok": True})
            self.server.backend_state.shutdown()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Unknown endpoint"})
""",
        "RequestHandler",
    )
    return source


def patch_fbx_export(source):
    source = replace_once(
        source, "import torch", "import torch\n\nfrom kimodo.cascadeur_runtime import fbx_wrapper_transform"
    )
    source = replace_once(
        source,
        "    transforms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {",
        "    wrapper = fbx_wrapper_transform(_fast_armature_template())\n    wrapper_channels = tuple(np.repeat(np.asarray(value, dtype=np.float32)[None], num_frames, axis=0) for value in wrapper)\n    transforms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {",
    )
    source = replace_once(
        source,
        '"KimodoArmature": (zeros.copy(), zeros.copy(), ones.copy()),',
        '"KimodoArmature": wrapper_channels,',
    )
    source = replace_once(
        source,
        "            curve_node_id = model_property_to_node[(model_name, prop_name)]",
        "            model_properties = _model_property_handles(model_objs[model_name_to_id[model_name]])\n            model_properties[prop_name][1][4:7] = values[0].astype(float).tolist()\n            curve_node_id = model_property_to_node[(model_name, prop_name)]",
    )
    return source


def apply_runtime_patch(root):
    root = Path(root).resolve()
    marker = root / ".cascadeur-runtime-patch"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != PATCH_VERSION:
            raise RuntimeError("A different Kimodo runtime patch is already installed")
        return
    changes = {}
    for relative, patch in (
        ("kimodo/model/registry.py", patch_model_registry),
        ("kimodo/model/llm2vec/llm2vec_wrapper.py", patch_encoder),
        ("kimodo/model/llm2vec/llm2vec.py", patch_llm_loader),
        ("kimodo/scripts/run_text_encoder_server.py", patch_text_server),
        ("scripts/cascadeur_backend_service.py", patch_backend),
        ("kimodo/exports/fbx.py", patch_fbx_export),
    ):
        path = root / relative
        changed = patch(path.read_text(encoding="utf-8"))
        ast.parse(changed, filename=relative)
        changes[path] = changed
    # Validate every input before changing any source file.
    helper = Path(__file__).with_name("runtime_service.py")
    ast.parse(helper.read_text(encoding="utf-8"))
    for path, changed in changes.items():
        path.write_text(changed, encoding="utf-8")
    shutil.copyfile(helper, root / "kimodo/cascadeur_runtime.py")
    marker.write_text(PATCH_VERSION + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python runtime_patch.py KIMODO_SOURCE_ROOT")
    apply_runtime_patch(sys.argv[1])
