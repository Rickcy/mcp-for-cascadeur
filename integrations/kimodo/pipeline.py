"""Generate retained motion candidates, transfer them to Cascy and review them once.

Run output is a durable outcome record. Reusing an existing output directory is
refused: inspect its generation/native receipts after an interrupted operation.
Visual review uses the configured cloud or local provider. Credentials are
resolved at request time and never copied into a run record.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PipelineError(RuntimeError):
    pass


def module(name):
    spec = importlib.util.spec_from_file_location("kimodo_pipeline_" + name, HERE / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _review_options(args):
    options = {name: getattr(args, name, None) for name in
               ("provider", "model", "base_url", "media_transport", "sampling_fps")}
    options = {name: value for name, value in options.items() if value is not None}
    budgets = {name: getattr(args, name, None) for name in
               ("max_total_frames", "max_frames_per_video", "frame_width")}
    budgets = {name: value for name, value in budgets.items() if value is not None}
    if budgets:
        options["config"] = budgets
    return options


def _review_context(args, reviewer, *, require_credentials=False):
    options = _review_options(args)
    identity = reviewer.review_identity(**options)
    if require_credentials:
        resolved = reviewer.review_settings(**options)
        if identity["provider"] == "gemini" and not resolved.get("api_key"):
            raise ValueError("Configure a Gemini key outside the repository before starting")
    return options, identity


def _report_identity(report):
    if isinstance(report.get("review_identity"), dict):
        return report["review_identity"]
    # Historical Gemini reports predate provider identities. Reuse requires an
    # explicit model and one recorded video sampling rate; missing data is not
    # inferred from the currently selected provider.
    if report.get("schema") != "kimodo-gemini-review-v1" or not report.get("requested_model"):
        return None
    rates = {medium.get("sampling_fps") for item in report.get("evidence", [])
             for medium in item.get("media", []) if medium.get("kind") == "video"}
    rate = report.get("sampling_fps")
    if rate is None and len(rates) == 1:
        rate = next(iter(rates))
    if rate is None:
        return None
    return {"provider": "gemini", "model": report["requested_model"],
            "media_transport": "video", "sampling_fps": rate}


def _review_matches(reviewer, report, candidates, prompt, identity):
    return (_report_identity(report) == identity and report.get("motion_prompt") == prompt
            and all(reviewer._evidence_matches(candidate, next(
                (item["media"] for item in report.get("evidence", []) if item.get("id") == candidate["id"]), []
            )) for candidate in candidates))


def _profile_paths(output, identity):
    name = identity["provider"]
    if name not in {"gemini", "openai-compatible"}:
        raise ValueError("Unknown visual review provider")
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()[:20]
    directory = output / "reviews"
    if directory.is_symlink() or directory.resolve() != directory:
        raise PipelineError("Visual review directory was replaced")
    return directory / f"{name}-{digest}.json", directory / f"{name}-{digest}.selection.json"


def _review_once(args, reviewer, output, record, candidates):
    """Reuse only this provider/profile and these exact media bytes."""
    options, identity = _review_context(args, reviewer)
    target, selection_path = _profile_paths(output, identity)
    possible = [target]
    if record.get("review"):
        possible.append(Path(record["review"]))
    possible.append(output / "gemini-review.json")
    report, report_path, reused = None, None, False
    for path in dict.fromkeys(possible):
        if not path.exists():
            continue
        existing = read(_owned_file(path, output))
        if _report_identity(existing) != identity:
            if path == target:
                raise PipelineError("The cached visual review belongs to a different provider profile")
            continue
        if not _review_matches(reviewer, existing, candidates, record["prompt"], identity):
            raise PipelineError("The cached visual review does not match the retained media; inspect before another request")
        report, report_path, reused = existing, path, True
        break
    if report is None:
        _review_context(args, reviewer, require_credentials=True)
        report = reviewer.review_candidates(candidates, record["prompt"], **options, max_output_tokens=8192)
        if not _review_matches(reviewer, report, candidates, record["prompt"], identity):
            raise PipelineError("The visual review identity or evidence differs from the submitted profile")
        target.parent.mkdir(exist_ok=True)
        write(target, report)
        report_path = target
    selection = reviewer.rank_candidates(candidates, report)
    selection_path.parent.mkdir(exist_ok=True)
    write(selection_path, selection)
    if record.get("review") and (record["review"] != str(report_path) or record.get("selection") != selection):
        previous = {name: record[name] for name in
                    ("review", "review_identity", "selection", "selected_scene") if name in record}
        history = record.setdefault("review_history", [])
        if previous not in history:
            history.append(previous)
    record.update(review=str(report_path), review_identity=identity, selection=selection)
    record.setdefault("review_profiles", {})[selection_path.stem.removesuffix(".selection")] = {
        "identity": identity, "review": str(report_path), "selection": str(selection_path),
    }
    write(output / "selection.json", selection)
    return selection, str(report_path), reused


def _create_record(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _base_url(value):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("Kimodo must use an HTTP loopback endpoint without a path or credentials")
    return value.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def request_json(base_url, route, payload=None, *, timeout=30):
    endpoint = _base_url(base_url)
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(endpoint + route, data=body, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise PipelineError("Kimodo response exceeds the JSON limit")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise PipelineError("Kimodo did not return a JSON object")
    return result


def submit_generation(base_url, payload, record_path):
    """Submit exactly once, retaining unknown outcomes instead of retrying POST."""
    endpoint = _base_url(base_url)
    record = {"state": "submitted", "endpoint": endpoint, "payload": payload, "submitted_at": time.time()}
    _create_record(record_path, record)
    try:
        result = request_json(endpoint, "/jobs", payload)
        identifier = result.get("job_id")
        if result.get("ok") is not True or type(identifier) is not int or identifier < 1:
            raise PipelineError("The backend did not return an accepted job identifier")
    except Exception:
        record["state"] = "unknown"
        record["message"] = "Submission was not repeated; inspect the backend and this run before another POST"
        write(record_path, record)
        raise PipelineError(record["message"]) from None
    record.update(state="accepted", job_id=identifier, job_url=endpoint + f"/jobs/{identifier}")
    write(record_path, record)
    return record


def wait_generation(base_url, job_id, *, timeout=1200, record_path=None, poll_interval=1):
    """Poll the accepted job; a transport failure never starts another generation."""
    if type(job_id) is not int or job_id < 1 or timeout <= 0 or poll_interval < 0:
        raise ValueError("A positive job ID, timeout and nonnegative poll interval are required")
    deadline = time.monotonic() + timeout
    last_progress = None
    while time.monotonic() < deadline:
        try:
            response = request_json(base_url, f"/jobs/{job_id}", timeout=15)
            job = response["job"]
            if response.get("ok") is not True or job.get("job_id") != job_id:
                raise ValueError("Job identity differs")
        except Exception:
            raise PipelineError("Job observation failed; the accepted generation was not resubmitted") from None
        if record_path:
            record = read(record_path)
            record.update(state="finished" if job.get("done") else "running", last_observation=job)
            write(record_path, record)
        progress = (job.get("progress"), job.get("status"))
        if progress != last_progress:
            print(json.dumps({"stage": "generation", "job_id": job_id,
                              "progress": progress[0], "status": progress[1]}), flush=True)
            last_progress = progress
        if job.get("done") is True:
            return job
        time.sleep(poll_interval)
    raise PipelineError("Stopped waiting for the accepted job; it was not cancelled or resubmitted")


def _artifact(exchange, value):
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("Backend artifacts must use relative exchange paths")
    path = (Path(exchange) / relative).resolve(strict=True)
    if not path.is_relative_to(Path(exchange).resolve()) or not path.is_file():
        raise ValueError("Backend artifact escaped the exchange directory or is missing")
    return path


def verify_candidates(job, exchange, *, count, frames, constraints_sha256=None, allow_legacy_npz=False):
    if job.get("error"):
        raise PipelineError("Generation failed; completed candidates and the durable job record were retained")
    rows = job.get("artifacts", {}).get("candidates")
    if not isinstance(rows, list) or len(rows) != count:
        raise PipelineError("The backend did not retain every requested candidate")
    indexes, paths = set(), set()
    result = []
    for row in rows:
        index = row.get("index")
        if type(index) is not int or not 0 <= index < count or index in indexes:
            raise PipelineError("Candidate indexes are missing or duplicated")
        indexes.add(index)
        resolved = {name: _artifact(exchange, row[name]) for name in ("output", "output_fbx", "manifest")}
        if resolved["output"] in paths or resolved["output_fbx"] in paths:
            raise PipelineError("Candidates share an animation file")
        paths.update((resolved["output"], resolved["output_fbx"]))
        manifest = read(resolved["manifest"])
        npz_sha256 = hashlib.sha256(resolved["output"].read_bytes()).hexdigest()
        if (manifest.get("format") != "cascadeur-kimodo-motion-v1" or manifest.get("num_frames") != frames
                or manifest.get("fps") != 30 or manifest.get("model_name") != "Kimodo-SOMA-RP-v1.2"
                or manifest.get("seed") != row.get("seed") or manifest.get("candidate_index") != index
                or manifest.get("constraints_sha256") != constraints_sha256
                or manifest.get("sha256") != hashlib.sha256(resolved["output_fbx"].read_bytes()).hexdigest()
                or manifest.get("text_encoder", {}).get("kind") != "llm2vec"
                or manifest.get("text_encoder", {}).get("fallback") is not False):
            raise PipelineError("A retained candidate does not match its manifest")
        declared_npz = manifest.get("npz_sha256")
        if declared_npz != npz_sha256 and not (allow_legacy_npz and declared_npz is None):
            raise PipelineError("A retained candidate NPZ does not match its generation checksum")
        result.append({"id": f"candidate-{index + 1:02d}", "index": index, "seed": row["seed"],
                       **{name: str(path) for name, path in resolved.items()},
                       "artifact_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                           for name, path in resolved.items()},
                       "npz_provenance": "generation_manifest" if declared_npz else "legacy_retarget_source_required"})
    return sorted(result, key=lambda row: row["index"])


def _process(command, *, timeout=120, environment=None):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout,
                                env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        raise PipelineError("A Kimodo runtime command failed; inspect Docker status and the run record") from None
    return result.stdout


def runtime_action(action, exchange, *, timeout=1200, build=False, port=9552):
    command = ["powershell", "-NoProfile", "-File", str(HERE / "kimodo.ps1"), "-Action", action,
               "-ExchangePath", str(exchange), "-Port", str(port), "-TimeoutSeconds", str(timeout)]
    if build:
        command.append("-Build")
    return _process(command, timeout=timeout + 30)


def release_runtime(base_url, exchange, *, port=9552):
    health = request_json(base_url, "/health", timeout=15)
    if health.get("ready") is not True or health.get("busy") is not False:
        raise PipelineError("Kimodo is busy or its idle state is unknown; services were left running")
    runtime_action("stop", exchange, timeout=30, port=port)


def convert_motion(npz, output, exchange, *, port=9552):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Retarget conversion output already exists")
    exchange = Path(exchange).resolve()
    source = Path(npz).resolve(strict=True)
    if not source.is_relative_to(exchange) or not output.is_relative_to(exchange):
        raise ValueError("Conversion paths must stay inside the mounted exchange directory")
    environment = dict(os.environ)
    environment.update(KIMODO_MODELS_DIR=str(HERE.parents[1] / ".local/kimodo/models"),
                       KIMODO_EXCHANGE_DIR=str(exchange), KIMODO_BACKEND_PORT=str(port))
    command = ["docker", "compose", "--project-name", "cascadeur-kimodo", "--file", str(HERE / "compose.yaml"),
               "exec", "-T", "backend", "python", "-B", "/opt/kimodo/prepare_retarget_motion.py",
               "/exchange/" + source.relative_to(exchange).as_posix(),
               "/exchange/" + output.relative_to(exchange).as_posix()]
    result = json.loads(_process(command, environment=environment))
    if not output.is_file() or result.get("num_frames") is None:
        raise PipelineError("SOMA conversion did not create its verified retarget output")
    return result


def source_metrics(npz):
    import numpy as np

    comparer = module("compare_motion")
    inspected = comparer.inspect_clip(npz)
    with np.load(npz, allow_pickle=False) as source:
        contacts = np.asarray(source["foot_contacts"])
    if contacts.ndim != 2 or contacts.shape[1] not in {4, 6}:
        raise ValueError("Generated clip lacks supported foot contacts")
    if contacts.shape[1] == 6:
        contacts = contacts[:, (0, 1, 3, 4)]
    predicted = inspected["metrics"]["predicted_contact_feet"]
    return {"source_contact_speed_m_s": predicted["mean_horizontal_speed_m_s"],
            "source_contact_coverage_ratio": predicted["coverage_ratio"],
            "source_motion": inspected["metrics"]}, contacts.tolist()


def native_activate_scene(path):
    """Fixed native operation preserving every open tab, including the source."""
    from mcp_for_cascadeur.host import common

    guard = module("native_import")._native_preflight()
    if not guard["ready"]:
        raise RuntimeError("Native guards block scene activation: " + "; ".join(guard["blockers"]))
    destination = common.workspace_path(path, must_exist=True, suffixes=(".casc",))
    _, app, _, _ = common.current_context(require_scene=False)
    manager = app.get_scene_manager()
    matches = [view for view in manager.scenes()
               if view.get_path_name() and Path(view.get_path_name()).resolve() == destination.resolve()]
    if len(matches) > 1:
        raise RuntimeError("Multiple open tabs match the requested scene; choose one explicitly")
    if matches:
        manager.set_current_scene(matches[0])
    elif not app.get_data_source_manager().load_scene(str(destination)):
        raise RuntimeError("Cascadeur declined to open the requested scene")
    _, _, view, _ = common.current_context()
    if Path(view.get_path_name()).resolve() != destination.resolve():
        raise RuntimeError("Cascadeur activated an unexpected scene")
    return {"scene": str(destination)}


def activate_scene(client, path):
    code = (
        "import importlib.util\n"
        f"s=importlib.util.spec_from_file_location('kimodo_pipeline_native',{str(Path(__file__).resolve())!r})\n"
        "m=importlib.util.module_from_spec(s)\ns.loader.exec_module(m)\n"
        f"result=m.native_activate_scene({str(path)!r})\n"
    )
    expression = "__import__('json').dumps((lambda n:(exec(" + repr(code) + ",n,n),n['result'])[1])({}))"
    response = client._request("/run", {"code": expression})
    if response.get("ok") is not True:
        raise PipelineError("Scene activation outcome is unknown; inspect before retrying")
    if any(str(item.get("level", "")).lower() == "error" for item in response.get("messages", [])):
        raise PipelineError("Cascadeur reported a scene activation error")
    return json.loads(ast.literal_eval(response["value"]))


def run_pipeline(args, *, client=None):
    workflow = module("workflow")
    if client is None:
        client, _ = module("native_import")._runner_client(None, 35)
    if not client.workspace:
        raise ValueError("A configured Cascadeur workspace is required")
    exchange = Path(client.workspace).resolve() / "kimodo"
    output = Path(args.output).resolve()
    if not output.is_relative_to(exchange) or output == exchange:
        raise ValueError("Run output must be a new directory inside the configured workspace/kimodo")
    if output.exists():
        raise FileExistsError("Run output already exists; inspect its receipts before another run")
    if not args.prompt.strip() or len(args.prompt) > 4000 or not 2 <= args.frames <= 120 or not 1 <= args.samples <= 4:
        raise ValueError("Provide a prompt, 2–120 frames and 1–4 candidates")
    if not 0 <= args.seed < 2**32 or not 1 <= args.steps <= 200 or not 1024 <= args.port <= 65535:
        raise ValueError("Provide a uint32 seed, 1–200 denoising steps and a local port from 1024 to 65535")
    if not 30 <= args.timeout <= 3600:
        raise ValueError("Runtime timeout must be between 30 and 3600 seconds")
    for name in ("calibration", "template_motion", "template_scene"):
        if not Path(getattr(args, name)).is_file():
            raise ValueError("Missing pipeline input: " + name)
    frames = [int(value) for value in args.keyframes.split(",")] if args.keyframes else None
    if bool(frames) != bool(args.source_scene):
        raise ValueError("Use --source-scene and --keyframes together")
    if frames and (len(frames) < 2 or frames[0] < 0 or any(a >= b for a, b in zip(frames, frames[1:]))
                   or frames[-1] - frames[0] + 1 != args.frames):
        raise ValueError("Increasing keyframes must span the complete requested generation interval")
    reviewer = module("motion_review")
    _, identity = _review_context(args, reviewer, require_credentials=True)
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "run.json"
    record = {"format": "kimodo-cascadeur-run-v1", "state": "running", "stage": "prepared",
              "prompt": args.prompt, "frames": args.frames, "samples": args.samples,
              "mode": "keyposes" if frames else "text", "candidates": [], "review_identity": identity,
              "inputs": {name: {"path": str(Path(getattr(args, name)).resolve()),
                                "sha256": hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()}
                         for name in ("calibration", "template_motion", "template_scene")}}
    _create_record(record_path, record)

    def stage(name, **values):
        record.update(stage=name, **values)
        write(record_path, record)
        print(json.dumps({"stage": name, "run": str(record_path)}), flush=True)

    endpoint = f"http://127.0.0.1:{args.port}"
    try:
        payload = {"prompt": args.prompt, "num_frames": args.frames, "num_samples": args.samples,
                   "seed": args.seed, "diffusion_steps": args.steps, "fps": 30}
        snapshot = None
        if frames:
            stage("capturing_keyposes")
            activate_scene(client, args.source_scene)
            constraints = workflow.capture_constraints(client, args.source_scene, frames, args.calibration,
                                                       args.template_motion, output / "constraints.npz")
            if constraints["num_frames"] != args.frames:
                raise ValueError("Captured keyframes differ from the requested frame interval")
            snapshot = constraints["capture"]
            payload["constraints"] = (output / "constraints.npz").relative_to(exchange).as_posix()
            payload["constraints_sha256"] = constraints["sha256"]
            stage("keyposes_captured", constraints=constraints)
        stage("starting_runtime")
        runtime_action("start", exchange, timeout=args.timeout, build=args.build, port=args.port)
        stage("submitting_generation")
        accepted = submit_generation(endpoint, payload, output / "generation.json")
        stage("generation_accepted", job_id=accepted["job_id"])
        job = wait_generation(endpoint, accepted["job_id"], timeout=args.timeout,
                              record_path=output / "generation.json")
        candidates = verify_candidates(job, exchange, count=args.samples, frames=args.frames,
                                       constraints_sha256=payload.get("constraints_sha256"))
        stage("converting_candidates", generated_candidates=candidates)
        for candidate in candidates:
            directory = output / candidate["id"]
            directory.mkdir()
            converted = directory / "motion.retarget.json"
            convert_motion(candidate["output"], converted, exchange, port=args.port)
            candidate["retarget_motion"] = str(converted)
            candidate["retarget_sha256"] = hashlib.sha256(converted.read_bytes()).hexdigest()
            stage("converting_candidates", generated_candidates=candidates)
        stage("releasing_gpu")
        release_runtime(endpoint, exchange, port=args.port)
        for candidate in candidates:
            directory = output / candidate["id"]
            stage("transferring_" + candidate["id"])
            result = workflow.transfer(client, candidate["retarget_motion"], args.calibration,
                                       args.template_scene, directory / "native", snapshot)
            candidate.update(result)
            measured = workflow.native(client, "measure", {"scene": result["scene"], "count": args.frames})
            metrics, contacts = source_metrics(candidate["output"])
            metrics.update(workflow.native_metrics(measured, contacts))
            candidate["metrics"] = metrics
            write(directory / "native-measurement.json", measured)
            stage("rendering_" + candidate["id"])
            candidate["media"] = workflow.render(client, result["scene"], measured, directory / "review")
            for media in candidate["media"]:
                media["sha256"] = hashlib.sha256(Path(media["path"]).read_bytes()).hexdigest()
            record["candidates"].append(candidate)
            stage("candidate_complete")
        stage("visual_review")
        selection, report_path, reused = _review_once(args, reviewer, output, record, record["candidates"])
        stage("reviewed", selection=selection, review=report_path, review_reused=reused)
        if selection.get("selected_id"):
            selected = next(row for row in record["candidates"] if row["id"] == selection["selected_id"])
            stage("activating_selection", selected_scene=selected["scene"])
            activate_scene(client, selected["scene"])
        record["state"] = "completed"
        stage("completed")
        return record
    except BaseException as error:
        record["state"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        record["error"] = str(error)
        write(record_path, record)
        raise


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _owned_file(path, directory):
    original = Path(path)
    try:
        resolved = original.resolve(strict=True)
    except OSError:
        raise PipelineError("A resume artifact is missing; inspect its owning run before continuing") from None
    if original.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(Path(directory).resolve()):
        raise PipelineError("A resume artifact is missing, replaced or outside its owning directory")
    return resolved


def _completed_transfer(candidate, directory, template_sha256):
    native_dir = directory / "native"
    if not native_dir.exists() or native_dir.is_dir() and not any(native_dir.iterdir()):
        return None
    result_path = _owned_file(native_dir / "transfer.json", native_dir)
    result = read(result_path)
    validation = result.get("validation", {})
    if validation.get("geometry") is not True or validation.get("persistence") is not True:
        raise PipelineError("Existing native transfer has not completed geometry and persistence verification")
    receipt_path = _owned_file(result["receipt"], native_dir)
    receipt = read(receipt_path)
    if receipt.get("phase") != "completed" or receipt.get("directory") != str(receipt_path.parent):
        raise PipelineError("Native import is incomplete; inspect its receipt instead of resuming writes")
    if receipt.get("reopen_verification", {}).get("point_coordinates_verified") is not True:
        raise PipelineError("Native receipt does not verify reopened Point coordinates")
    lock = receipt_path.with_name(".native-import.lock")
    transport = receipt_path.with_name(".native-transport.json")
    if lock.exists() or transport.exists() and read(transport).get("status") in {"submitted", "unknown"}:
        raise PipelineError("Native import transport outcome is uncertain; inspect before resuming")
    scene = _owned_file(result["scene"], receipt_path.parent)
    plan = _owned_file(receipt["plan_path"], native_dir)
    if (scene != receipt_path.parent / "result.casc" or _sha(scene) != receipt.get("result_sha256")
            or _sha(plan) != receipt.get("plan_sha256")
            or read(plan).get("source_npz_sha256") != candidate["artifact_sha256"]["output"]):
        raise PipelineError("Completed native scene or plan differs from its verified receipt")
    before = _owned_file(receipt_path.parent / "before.casc", receipt_path.parent)
    if _sha(before) != receipt.get("before_sha256") or _sha(before) != template_sha256:
        raise PipelineError("Native recovery template changed after transfer")
    return result


def _inspect_render_receipts(client, workflow, directory, *, read_only=False):
    """Read native ownership before any scene switch; never cancel an unknown job."""
    for receipt_path in directory.glob("review*/native-render.json"):
        path = _owned_file(receipt_path, directory)
        receipt = read(path)
        state = receipt if read_only else workflow.native(
            client, "render_status", {"id": receipt["id"], "receipt": str(path)})
        if (state.get("state") not in {"completed", "failed", "cancelled"}
                or state.get("restoration", {}).get("restored") is not True):
            raise PipelineError("A prior native renderer is active or requires inspection; no scene was changed")
        if read_only:
            continue
        # A terminal owner has stopped writing. Only its explicit PNG names can
        # be removed; unrelated files and completed videos remain untouched.
        parents = set()
        for image in receipt.get("images", []):
            image_path = Path(image["path"])
            if (image_path.parent.parent.resolve() != path.parent
                    or not image_path.parent.name.startswith("frames-")
                    or not image_path.name.endswith(".png")):
                raise PipelineError("Native render receipt contains an unexpected scratch path")
            if image_path.exists():
                owned = _owned_file(image_path, path.parent)
                owned.unlink()
            parents.add(image_path.parent)
        for parent in parents:
            if parent.is_dir() and not parent.is_symlink() and not any(parent.iterdir()):
                parent.rmdir()


def _reusable_media(candidate, directory, review=None):
    media = candidate.get("media", [])
    if len(media) != 3 or {item.get("view") for item in media} != {"front", "side", "three-quarter"}:
        return None
    reviewed = {}
    if review:
        reviewed = {entry["view"]: entry.get("sha256")
                    for item in review.get("evidence", []) if item.get("id") == candidate["id"]
                    for entry in item.get("media", [])}
    for item in media:
        path = _owned_file(item["path"], directory)
        expected = item.get("sha256") or reviewed.get(item["view"])
        if not expected or path.suffix.lower() != ".mp4" or _sha(path) != expected:
            raise PipelineError("Existing review media is unpinned or differs from its recorded checksum")
    return media


def _render_destination(directory):
    base = directory / "review"
    if not base.exists() or base.is_dir() and not any(base.iterdir()):
        return base
    for index in range(2, 100):
        candidate = directory / f"review-{index}"
        if not candidate.exists():
            return candidate
    raise PipelineError("Too many retained render attempts; inspect and clean the run before continuing")


def resume_pipeline(args, *, client=None, review_only=False):
    """Continue native transfer/review only; never start or call the motion service.

    Source model artifacts must already exist and validate against a finished
    generation receipt. Native stages with uncertain outcomes are refused. A
    completed saved scene is reused, not rewritten or regenerated.
    """
    workflow = module("workflow")
    if client is None:
        client, _ = module("native_import")._runner_client(None, 35)
    exchange = Path(client.workspace).resolve() / "kimodo"
    output = Path(args.output).resolve(strict=True)
    if not output.is_relative_to(exchange) or output == exchange or not output.is_dir():
        raise ValueError("Resume output must be an existing run inside the configured workspace/kimodo")
    record_path = _owned_file(output / "run.json", output)
    record = read(record_path)
    allowed_states = {"completed"} if review_only else {"running", "failed", "interrupted"}
    if record.get("format") != "kimodo-cascadeur-run-v1" or record.get("state") not in allowed_states:
        raise PipelineError("Visual re-evaluation requires a completed run" if review_only
                            else "Only running, failed or interrupted motion runs can be resumed")
    previous_state, previous_stage = record["state"], record.get("stage")
    lock = output / ".resume.lock"
    token = {"pid": os.getpid(), "started_at": time.time()}
    _create_record(lock, token)
    try:
        return _resume_locked(args, client, workflow, exchange, output, record_path, record,
                              review_only=review_only)
    except BaseException as error:
        if review_only:
            record.update(state=previous_state, stage=previous_stage,
                          last_review_attempt={"state": "failed", "error": str(error)})
        else:
            record.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                          error=str(error))
        write(record_path, record)
        raise
    finally:
        if lock.is_file() and not lock.is_symlink() and read(lock) == token:
            lock.unlink()


def _resume_locked(args, client, workflow, exchange, output, record_path, record, *, review_only=False):
    def stage(name, **values):
        record.update(state="completed" if review_only else "running", stage=name, **values)
        write(record_path, record)
        print(json.dumps({"stage": name, "run": str(record_path), "resumed": True}), flush=True)

    inputs = {}
    for name in ("calibration", "template_scene"):
        recorded = record.get("inputs", {}).get(name, {})
        supplied = getattr(args, name, None)
        path = Path(supplied or recorded.get("path", "")).resolve(strict=True)
        if not path.is_file():
            raise PipelineError("Resume requires the original " + name)
        digest = _sha(path)
        if recorded and (recorded.get("path") != str(path) or recorded.get("sha256") != digest):
            raise PipelineError("A pinned pipeline input changed: " + name)
        inputs[name] = {"path": str(path), "sha256": digest}
    constraints_sha, snapshot = None, None
    if record.get("mode") == "keyposes":
        constraint_record = record["constraints"]
        constraint_file = _owned_file(output / "constraints.npz", output)
        constraints_sha = constraint_record["sha256"]
        if _sha(constraint_file) != constraints_sha:
            raise PipelineError("Captured constraints changed after generation")
        snapshot = _owned_file(constraint_record["capture"], output)
        captured = read(snapshot)
        neutral = read(inputs["calibration"]["path"])
        neutral_digest = hashlib.sha256(json.dumps(neutral, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False).encode()).hexdigest()
        if captured.get("calibration_sha256") != neutral_digest:
            raise PipelineError("Resume calibration differs from the captured keypose calibration")
        captured_digest = hashlib.sha256(json.dumps(
            {key: value for key, value in captured.items() if key != "capture_sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        if captured.get("capture_sha256") != captured_digest:
            raise PipelineError("Captured keypose data changed after export")
    generation = read(_owned_file(output / "generation.json", output))
    job = generation.get("last_observation", {})
    if generation.get("state") != "finished" or job.get("done") is not True:
        raise PipelineError("Generation has no verified terminal receipt; it was not observed or resubmitted")
    candidates = verify_candidates(job, exchange, count=record["samples"], frames=record["frames"],
                                   constraints_sha256=constraints_sha, allow_legacy_npz=True)
    retained = {item["id"]: item for item in record.get("generated_candidates", [])}
    completed = {item["id"]: item for item in record.get("candidates", [])}
    pins = {"constraints_sha256": constraints_sha, "capture_sha256": _sha(snapshot) if snapshot else None,
            "inputs": inputs, "candidates": {}}
    for candidate in candidates:
        previous = retained.get(candidate["id"], {})
        if previous.get("artifact_sha256") and previous["artifact_sha256"] != candidate["artifact_sha256"]:
            raise PipelineError("Generated candidate artifacts changed after their original verification")
        converted = _owned_file(previous.get("retarget_motion") or output / candidate["id"] / "motion.retarget.json",
                                output / candidate["id"])
        converted_sha = _sha(converted)
        if previous.get("retarget_sha256") and previous["retarget_sha256"] != converted_sha:
            raise PipelineError("Retarget source data changed after conversion")
        if read(converted).get("source_npz_sha256") != candidate["artifact_sha256"]["output"]:
            raise PipelineError("Retarget source checksum differs from the retained NPZ")
        candidate.update(retarget_motion=str(converted), retarget_sha256=converted_sha)
        pins["candidates"][candidate["id"]] = {
            "artifacts": candidate["artifact_sha256"], "retarget": converted_sha,
        }
    if record.get("resume_pins") and record["resume_pins"] != pins:
        raise PipelineError("A pinned resume artifact or original input changed")
    record["resume_pins"] = pins
    record["inputs"] = {**record.get("inputs", {}), **inputs}
    legacy = [item["id"] for item in candidates if item["npz_provenance"] == "legacy_retarget_source_required"]
    if legacy:
        record["provenance_limitations"] = [
            "Legacy generation manifests lack NPZ hashes for " + ", ".join(legacy)
            + "; verified against retarget source checksums and pinned observed bytes on first resume."
        ]
    stage("resume_inputs_verified", generated_candidates=candidates)
    # Validate ALL prior native ownership before activating or writing any scene.
    for candidate in candidates:
        _inspect_render_receipts(client, workflow, output / candidate["id"], read_only=review_only)
    native_results = {
        candidate["id"]: _completed_transfer(candidate, output / candidate["id"], inputs["template_scene"]["sha256"])
        for candidate in candidates
    }
    if review_only and any(value is None for value in native_results.values()):
        raise PipelineError("Visual re-evaluation requires all native transfers to be complete")
    review_path = Path(record.get("review") or output / "gemini-review.json")
    old_review = read(_owned_file(review_path, output)) if review_path.exists() else None
    results = dict(completed)
    for candidate in candidates:
        directory = output / candidate["id"]
        result = native_results[candidate["id"]]
        if result is None:
            stage("transferring_" + candidate["id"])
            result = workflow.transfer(client, candidate["retarget_motion"], inputs["calibration"]["path"],
                                       inputs["template_scene"]["path"], directory / "native", str(snapshot) if snapshot else None)
        else:
            stage("reusing_" + candidate["id"])
            if not review_only:
                activate_scene(client, result["scene"])
        candidate.update(result)
        saved = completed.get(candidate["id"], {})
        if review_only:
            if not isinstance(saved.get("metrics"), dict):
                raise PipelineError("Visual re-evaluation requires the retained native diagnostics")
            candidate["metrics"] = saved["metrics"]
        else:
            measured = workflow.native(client, "measure", {"scene": result["scene"], "count": record["frames"]})
            metrics, contacts = source_metrics(candidate["output"])
            metrics.update(workflow.native_metrics(measured, contacts))
            candidate["metrics"] = metrics
            write(directory / "native-measurement.json", measured)
        media = _reusable_media(saved, directory, old_review)
        if media is None:
            if review_only:
                raise PipelineError("Visual re-evaluation requires three existing hash-bound native videos")
            stage("rendering_" + candidate["id"])
            media = workflow.render(client, result["scene"], measured, _render_destination(directory))
            for item in media:
                item["sha256"] = _sha(item["path"])
        candidate["media"] = media
        results[candidate["id"]] = candidate
        record["candidates"] = [results[item["id"]] for item in candidates if item["id"] in results]
        stage("candidate_complete")
    reviewer = module("motion_review")
    stage("visual_review")
    selection, review_path, reused = _review_once(args, reviewer, output, record, candidates)
    stage("reviewed", selection=selection, review=review_path, review_reused=reused)
    if selection.get("selected_id"):
        selected = next(row for row in candidates if row["id"] == selection["selected_id"])
        record["selected_scene"] = selected["scene"]
        if not review_only:
            stage("activating_selection")
            activate_scene(client, selected["scene"])
    else:
        record.pop("selected_scene", None)
    if review_only:
        record["last_review_attempt"] = {"state": "completed", "identity": record["review_identity"],
                                         "reused": reused, "selection_activated": False}
    record.update(state="completed", stage="completed")
    record.pop("error", None)
    write(record_path, record)
    return record


def review_pipeline(args, *, client=None):
    """Evaluate a completed run's saved videos without touching native scenes."""
    return resume_pipeline(args, client=client, review_only=True)


def _add_review_options(parser):
    parser.add_argument("--provider", choices=("gemini", "openai-compatible"))
    parser.add_argument("--model", help="Review model ID; defaults to the saved provider settings")
    parser.add_argument("--base-url", help="Loopback OpenAI-compatible review endpoint, including /v1")
    parser.add_argument("--media-transport", choices=("video", "frames"))
    parser.add_argument("--sampling-fps", type=float)
    parser.add_argument("--max-total-frames", type=int)
    parser.add_argument("--max-frames-per-video", type=int)
    parser.add_argument("--frame-width", type=int)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("prompt", "output", "calibration", "template-motion", "template-scene"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--source-scene")
    run.add_argument("--keyframes", help="Increasing zero-based source frames spanning --frames")
    run.add_argument("--frames", type=int, default=120)
    run.add_argument("--samples", type=int, default=3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--steps", type=int, default=50)
    run.add_argument("--port", type=int, default=9552)
    run.add_argument("--timeout", type=int, default=1200)
    _add_review_options(run)
    run.add_argument("--build", action="store_true")
    resume = commands.add_parser("resume", help="Resume saved native/review stages without generating motion again")
    resume.add_argument("--output", required=True)
    resume.add_argument("--calibration", help="Original calibration, required for legacy runs without pinned inputs")
    resume.add_argument("--template-scene", help="Original Cascy template, required for legacy runs")
    _add_review_options(resume)
    review = commands.add_parser("review", help="Re-evaluate completed native videos without generating or changing scenes")
    review.add_argument("--output", required=True)
    review.add_argument("--calibration", help="Original calibration, required for legacy runs without pinned inputs")
    review.add_argument("--template-scene", help="Original Cascy template, required for legacy runs")
    _add_review_options(review)
    args = parser.parse_args(argv)
    result = {"run": run_pipeline, "resume": resume_pipeline, "review": review_pipeline}[args.command](args)
    selection = result.get("selection", {})
    print(json.dumps({
        "state": result["state"], "mode": result["mode"],
        "run": str(Path(args.output).resolve() / "run.json"),
        "selected_id": selection.get("selected_id"),
        "selected_scene": result.get("selected_scene"),
        "review": result.get("review"),
        "candidates": [{"id": item["id"], "scene": item.get("scene"),
                        "validation": item.get("validation")} for item in result.get("candidates", [])],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
