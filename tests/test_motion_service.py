"""Shared job ownership, durable recovery and authenticated loopback transport."""

import hashlib
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur import motion_service as service
from mcp_for_cascadeur.settings import Config


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(tmp_path / "private"))
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    checkout = tmp_path / "checkout"
    integration = checkout / "integrations/kimodo"
    integration.mkdir(parents=True)
    (integration / "pipeline.py").write_text("# fixture pipeline")
    (integration / "kimodo.ps1").write_text("# fixture launcher")
    inputs = {}
    for name in ("calibration", "template_motion", "template_scene"):
        path = tmp_path / (name + (".casc" if name == "template_scene" else ".json"))
        path.write_text("{}")
        inputs[name] = str(path.resolve())
    result = service.Controller(Config(workspace=workspace))
    result.configure({"checkout": str(checkout.resolve()), **inputs})
    processes = []

    def spawn(command, **options):
        process = SimpleNamespace(pid=1234567, exit_code=None, command=command, options=options)
        process.poll = lambda: process.exit_code
        processes.append(process)
        return process

    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    monkeypatch.setattr(result, "_docker_binding", lambda **kwargs: {"verified": True, "state": "running", "port": 9552})
    monkeypatch.setattr(result, "_runtime_health", lambda port: {"endpoint": f"http://127.0.0.1:{port}",
                                                               "reachable": False, "health": None})
    result.test_processes = processes
    return result


def test_start_is_nonblocking_single_owner_and_never_uses_a_shell(controller):
    prompt = 'Walk forward; $(echo secret) "exactly"'
    job = controller.start({"prompt": prompt, "samples": 2, "review_enabled": False, "auto_activate": False})
    assert job["state"] == "running"
    process = controller.test_processes[0]
    assert process.command[process.command.index("--prompt") + 1] == prompt
    assert "--no-review" in process.command and "--no-activate" in process.command
    assert not process.options.get("shell")
    with pytest.raises(service.MotionError, match="already active"):
        controller.start({"prompt": "Another walk"})
    assert len(controller.test_processes) == 1
    assert controller.job(job["job_id"])["pid"] == process.pid


def test_positive_per_run_flags_override_saved_defaults(controller):
    controller.configure({"defaults": {"review_enabled": False, "auto_activate": False}})
    job = controller.start({"prompt": "walk", "review_enabled": True, "auto_activate": True})
    assert job["options"]["no_review"] is False and job["options"]["no_activate"] is False
    assert "--no-review" not in controller.test_processes[-1].command


def test_invalid_bounds_or_credentials_are_rejected_before_launch(controller):
    for options in ({"prompt": "walk", "frames": 1}, {"prompt": "walk", "api_key": "fixture-secret"},
                    {"prompt": "walk", "cfg_enabled": "yes"}, {"prompt": "walk", "samples": 1, "sample_index": 1},
                    {"prompt": "walk", "text_weight": float("nan")},
                    {"prompt": "walk", "base_url": "https://untrusted.example/v1"}):
        with pytest.raises(ValueError):
            controller.start(options)
    assert not controller.test_processes


def test_source_pose_span_and_workspace_are_checked(controller, tmp_path):
    scene = tmp_path / "outside.casc"
    scene.write_bytes(b"scene")
    with pytest.raises(ValueError, match="workspace"):
        controller.start({"prompt": "walk", "source_scene": str(scene), "keyframes": [0, 119]})
    inside = controller.config.workspace / "source.casc"
    inside.write_bytes(b"scene")
    with pytest.raises(ValueError, match="span"):
        controller.start({"prompt": "walk", "source_scene": str(inside), "keyframes": [0, 60]})
    job = controller.start({"prompt": "walk", "source_scene": str(inside), "keyframes": "0,30,60,119"})
    assert job["options"]["keyframes"] == [0, 30, 60, 119]
    command = controller.test_processes[0].command
    assert command[command.index("--keyframes") + 1] == "0,30,60,119"


def test_selected_target_and_long_interval_do_not_need_legacy_cascy_inputs(controller):
    service._write(service._private_path("smart-motion.json"), {"checkout": controller.settings()["checkout"]})
    target = controller.config.workspace / "UE5_Manny.casc"
    target.write_bytes(b"target")
    job = controller.start({"prompt": "walk", "target_scene": str(target), "character_id": "rig-id",
                            "rig_mapping": {"pelvis": "pelvis-id"}, "first_frame": 90,
                            "frames": 7200, "samples": 1, "no_render": True, "review_enabled": False})
    command = controller.test_processes[-1].command
    assert job["options"]["frames"] == 7200 and job["options"]["first_frame"] == 90
    assert command[command.index("--target-scene") + 1] == str(target)
    assert json.loads(command[command.index("--rig-mapping") + 1]) == {"pelvis": "pelvis-id"}
    assert "--calibration" not in command and "--template-scene" not in command
    assert "--no-review" in command and "--no-render" in command


def test_target_scope_and_no_render_evidence_are_checked_before_launch(controller, tmp_path):
    target = tmp_path / "outside.casc"
    target.write_bytes(b"target")
    for options in ({"target_scene": str(target)}, {"character_id": "missing-scene"},
                    {"no_render": True, "review_enabled": True}, {"first_frame": -1},
                    {"rig_mapping": {"pelvis": 42}}):
        with pytest.raises(ValueError):
            controller.start({"prompt": "walk", **options})
    assert not controller.test_processes


def test_more_than_120_anchor_frames_and_1800_duration_are_preserved(controller):
    target = controller.config.workspace / "source.casc"
    target.write_bytes(b"source")
    anchors = list(range(0, 2000, 10)) + [1999]
    job = controller.start({"prompt": "walk", "source_scene": str(target), "frames": 2000,
                            "keyframes": anchors, "review_enabled": False, "no_render": True})
    assert job["options"]["keyframes"] == anchors
    assert controller.capabilities()["frames"] == [2, None]


def test_cancel_requests_a_phase_boundary_without_killing_or_resubmitting(controller):
    job = controller.start({"prompt": "walk"})
    requested = controller.cancel(job["job_id"])
    assert requested["state"] == "cancel_requested"
    assert (controller.directory / (job["job_id"] + ".cancel")).is_file()
    assert controller.test_processes[0].exit_code is None
    assert len(controller.test_processes) == 1
    with pytest.raises(service.MotionError, match="active"):
        controller.resume(job["job_id"])


def test_completion_and_progress_are_read_from_retained_run(controller):
    job = controller.start({"prompt": "walk"})
    output = Path(job["output"])
    output.mkdir()
    service._write(output / "run.json", {"state": "completed", "stage": "completed", "candidates": [
        {"id": "candidate-01", "scene": "retained.casc"}], "selection": {"selected_id": "candidate-01"}})
    service._write(output / "progress.json", {"state": "completed", "stage": "completed", "fraction": 1})
    controller.test_processes[0].exit_code = 0
    completed = controller.job(job["job_id"])
    assert completed["state"] == "completed"
    assert completed["progress"]["fraction"] == 1
    assert completed["selection"]["selected_id"] == "candidate-01"
    assert completed["run"]["candidates"] == completed["candidates"]


def test_orphaned_worker_fences_new_jobs_and_dead_worker_requires_explicit_resume(controller, monkeypatch):
    job = controller.start({"prompt": "walk"})
    restarted = service.Controller(controller.config)
    monkeypatch.setattr(service, "_alive", lambda pid: True)
    assert restarted.job(job["job_id"])["state"] == "orphaned"
    with pytest.raises(service.MotionError, match="already active"):
        restarted.start({"prompt": "walk"})
    monkeypatch.setattr(service, "_alive", lambda pid: False)
    assert restarted.job(job["job_id"])["state"] == "interrupted"
    assert not restarted.processes


def test_concurrent_terminal_observers_cannot_replace_failed_exit_code_with_interrupted(controller):
    job = controller.start({"prompt": "walk"})
    stale_a = controller._load(job["job_id"])
    stale_b = dict(stale_a)
    entered, release = threading.Event(), threading.Event()
    process = controller.test_processes[-1]

    def terminal_poll():
        entered.set()
        assert release.wait(2), "The second observer did not reach the regression barrier"
        return 7

    process.poll = terminal_poll
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(controller._refresh, stale_a)
        assert entered.wait(2)
        second = executor.submit(controller._refresh, stale_b)
        release.set()
        observed = [first.result(timeout=2), second.result(timeout=2)]
    assert all(record["state"] == "failed" and record["exit_code"] == 7 for record in observed)
    repeated = controller._refresh(stale_b)
    assert repeated["state"] == "failed" and repeated["exit_code"] == 7
    persisted = controller._load(job["job_id"])
    assert persisted["state"] == "failed" and persisted["exit_code"] == 7


def test_resume_reuses_output_and_uses_resume_without_generation(controller):
    job = controller.start({"prompt": "walk"})
    output = Path(job["output"])
    output.mkdir()
    service._write(output / "run.json", {"state": "failed", "mode": "text"})
    controller.test_processes[0].exit_code = 1
    controller.job(job["job_id"])
    resumed = controller.resume(job["job_id"], {"provider": "openai-compatible", "model": "local-qwen", "auto_activate": False})
    command = controller.test_processes[-1].command
    assert resumed["output"] == job["output"]
    assert "resume" in command and "run" not in command
    assert "--no-activate" in command and "--provider" in command
    assert "--prompt" not in command


def test_resume_preserves_original_controls_and_resolved_review_profile(controller):
    job = controller.start({"prompt": "walk", "review_enabled": False, "auto_activate": False})
    output = Path(job["output"])
    output.mkdir()
    service._write(output / "run.json", {"state": "failed", "mode": "text", "review_identity": {
        "provider": "gemini", "model": "original-gemini", "media_transport": "video", "sampling_fps": 9}})
    controller.test_processes[-1].exit_code = 1
    controller.job(job["job_id"])
    resumed = controller.resume(job["job_id"])
    command = controller.test_processes[-1].command
    assert "--no-review" in command and "--no-activate" in command
    assert command[command.index("--model") + 1] == "original-gemini"
    assert resumed["resume_options"]["provider"] == "gemini"
    assert resumed["options"]["no_review"] is True


def test_resume_explicit_provider_and_controls_override_and_survive_another_attempt(controller):
    job = controller.start({"prompt": "walk", "provider": "gemini", "model": "old-gemini",
                            "sampling_fps": 9, "review_enabled": False, "auto_activate": False})
    output = Path(job["output"])
    output.mkdir()
    service._write(output / "run.json", {"state": "failed", "mode": "text", "review_identity": {
        "provider": "gemini", "model": "old-gemini", "media_transport": "video", "sampling_fps": 9}})
    controller.test_processes[-1].exit_code = 1
    controller.job(job["job_id"])
    first = controller.resume(job["job_id"], {"provider": "openai-compatible", "model": "local-qwen",
                                            "review_enabled": True, "auto_activate": True})
    command = controller.test_processes[-1].command
    assert "--no-review" not in command and "--no-activate" not in command
    assert "--sampling-fps" not in command and "old-gemini" not in command
    assert first["options"]["model"] == "old-gemini"
    controller.test_processes[-1].exit_code = 1
    controller.job(job["job_id"])
    again = controller.resume(job["job_id"])
    command = controller.test_processes[-1].command
    assert command[command.index("--model") + 1] == "local-qwen"
    assert again["resume_options"]["no_review"] is False and "--no-review" not in command


def test_review_only_keeps_current_provider_resolution_instead_of_inheriting_original(controller):
    job = controller.start({"prompt": "walk", "provider": "gemini", "model": "old-gemini", "auto_activate": False})
    output = Path(job["output"])
    output.mkdir()
    service._write(output / "run.json", {"state": "completed", "mode": "text", "review_identity": {
        "provider": "gemini", "model": "old-gemini", "media_transport": "video", "sampling_fps": 9}})
    controller.test_processes[-1].exit_code = 0
    controller.job(job["job_id"])
    reviewed = controller.resume(job["job_id"], review_only=True)
    command = controller.test_processes[-1].command
    assert "--provider" not in command and "--model" not in command and "--no-activate" not in command
    assert "resume_options" not in reviewed


@pytest.mark.parametrize("state,mode,review_only", [
    ("completed", "text", False), ("failed", "text", True), ("completed", "generated_only", False),
])
def test_invalid_resume_modes_are_rejected_before_another_child_starts(controller, state, mode, review_only):
    output = controller.exchange / "not-resumable"
    output.mkdir()
    service._write(output / "run.json", {"format": "kimodo-cascadeur-run-v1", "state": state, "mode": mode})
    job = controller.import_run(str(output))
    with pytest.raises(service.MotionError):
        controller.resume(job["job_id"], review_only=review_only)
    assert not controller.test_processes


def test_raw_generation_needs_only_checkout_and_accepts_1800_frames(controller):
    service._write(service._private_path("smart-motion.json"), {"checkout": controller.settings()["checkout"]})
    job = controller.start({"prompt": "walk", "generate_only": True, "frames": 1800})
    command = controller.test_processes[-1].command
    assert job["options"]["frames"] == 1800
    assert "--generate-only" in command and "--calibration" not in command
    assert "--template-scene" not in command


def test_raw_constraint_paths_are_bounded_to_exchange(controller, tmp_path):
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"npz")
    with pytest.raises(ValueError, match="exchange"):
        controller.start({"prompt": "walk", "generate_only": True, "constraints": str(outside)})
    with pytest.raises(ValueError, match="raw generation"):
        controller.start({"prompt": "walk", "constraints": str(outside)})


def test_import_existing_run_and_visual_review_keep_its_scene_bytes(controller):
    output = controller.exchange / "existing"
    output.mkdir()
    scene = output / "kept.casc"
    scene.write_bytes(b"scene-bytes")
    service._write(output / "run.json", {"format": "kimodo-cascadeur-run-v1", "state": "completed",
                                        "mode": "text", "prompt": "walk", "candidates": []})
    job = controller.import_run(str(output))
    assert controller.list_runs()["runs"][0]["output"] == str(output)
    assert controller.import_run(str(output))["job_id"] == job["job_id"]
    reviewed = controller.resume(job["job_id"], {"provider": "gemini"}, review_only=True)
    assert reviewed["action"] == "review"
    assert "review" in controller.test_processes[-1].command
    assert scene.read_bytes() == b"scene-bytes"


def test_review_does_not_accept_options_that_would_rerender(controller):
    output = controller.exchange / "review-only"
    output.mkdir()
    service._write(output / "run.json", {"format": "kimodo-cascadeur-run-v1", "state": "completed", "mode": "text"})
    job = controller.import_run(str(output))
    with pytest.raises(ValueError, match="sampling cannot"):
        controller.resume(job["job_id"], {"render_sample_step": 1}, review_only=True)


def test_select_requires_completed_hash_bound_native_receipt(controller):
    output = controller.exchange / "retained"
    output.mkdir()
    scene = output / "result.casc"
    scene.write_bytes(b"native")
    receipt = output / "session.json"
    service._write(receipt, {"phase": "completed", "result_sha256": hashlib.sha256(scene.read_bytes()).hexdigest()})
    service._write(output / "run.json", {"format": "kimodo-cascadeur-run-v1", "state": "completed", "mode": "text",
                                        "candidates": [{"id": "candidate-01", "scene": str(scene), "receipt": str(receipt)}]})
    job = controller.import_run(str(output))
    scene.write_bytes(b"changed")
    with pytest.raises(service.MotionError, match="differs"):
        controller.select(job["job_id"], "candidate-01")
    assert not controller.test_processes
    scene.write_bytes(b"native")
    selection = controller.select(job["job_id"], "candidate-01")
    assert selection["action"] == "select" and selection["manual_selection"] == "candidate-01"


def test_select_does_not_report_old_completed_progress_while_activation_is_running(controller):
    output = controller.exchange / "old-progress"
    output.mkdir()
    scene, receipt = output / "result.casc", output / "session.json"
    scene.write_bytes(b"native")
    service._write(receipt, {"phase": "completed", "result_sha256": hashlib.sha256(scene.read_bytes()).hexdigest()})
    service._write(output / "run.json", {"format": "kimodo-cascadeur-run-v1", "state": "completed", "mode": "text",
                                        "stage": "completed", "candidates": [
                                            {"id": "candidate-01", "scene": str(scene), "receipt": str(receipt)}]})
    service._write(output / "progress.json", {"state": "completed", "stage": "completed", "fraction": 1,
                                             "updated_at": 1})
    job = controller.import_run(str(output))
    selecting = controller.select(job["job_id"], "candidate-01")
    assert selecting["state"] == "running" and selecting["progress"]["state"] == "running"
    assert selecting["stage"] == "select" and selecting["progress"]["fraction"] is None
    assert selecting["run"]["stage"] == "completed"
    controller.test_processes[-1].exit_code = 0
    completed = controller.job(job["job_id"])
    assert completed["state"] == "completed" and completed["stage"] == "completed"
    assert completed["run"]["candidates"][0]["scene"] == str(scene)


@pytest.mark.parametrize("health", [{"reachable": True, "health": {"busy": True}}, {"reachable": False, "health": None}])
def test_runtime_stop_refuses_busy_or_unknown_backend(controller, monkeypatch, health):
    monkeypatch.setattr(controller, "_runtime_health", lambda port: health)
    with pytest.raises(service.MotionError, match="busy"):
        controller.runtime("stop")
    assert not controller.test_processes


def test_failed_port_override_does_not_mistake_requested_port_for_the_running_backend(controller, monkeypatch):
    job = controller.start({"prompt": "walk", "port": 9666})
    controller.test_processes[-1].exit_code = 1
    assert controller.job(job["job_id"])["state"] == "failed"
    seen = []

    def actual_health(port):
        seen.append(port)
        return {"endpoint": f"http://127.0.0.1:{port}", "reachable": True, "health": {"busy": True}}

    monkeypatch.setattr(controller, "_runtime_health", actual_health)
    status = controller.runtime("status")
    assert status["requested_port"] == 9666 and status["bound_port"] == 9552
    assert status["endpoint_source"] == "docker_binding"
    assert controller.settings()["defaults"]["port"] == 9552
    with pytest.raises(service.MotionError, match="busy"):
        controller.runtime("stop")
    assert seen == [9552, 9552]
    assert len(controller.test_processes) == 1


def test_explicit_port_default_changes_next_start_but_stop_uses_verified_binding(controller, monkeypatch):
    controller.configure({"defaults": {"port": 9667}})
    status = controller.runtime("status")
    assert status["requested_port"] == 9667 and status["bound_port"] == 9552
    monkeypatch.setattr(controller, "_runtime_health", lambda port: {"reachable": True, "health": {"busy": False}})
    stopped = controller.runtime("stop")
    command = controller.test_processes[-1].command
    assert command[command.index("-Port") + 1] == "9552"
    assert stopped["options"]["port"] == 9552
    controller.test_processes[-1].exit_code = 0
    controller.job(stopped["job_id"])
    controller.runtime("start")
    command = controller.test_processes[-1].command
    assert command[command.index("-Port") + 1] == "9667"


@pytest.mark.parametrize("labels,host", [
    ({"com.docker.compose.project": "cascadeur-kimodo", "com.docker.compose.service": "backend"}, "127.0.0.1"),
    ({"com.docker.compose.project": "someone-else", "com.docker.compose.service": "backend"}, "127.0.0.1"),
    ({"com.docker.compose.project": "cascadeur-kimodo", "com.docker.compose.service": "backend"}, "0.0.0.0"),
])
def test_docker_binding_requires_owned_container_and_loopback_port(controller, monkeypatch, labels, host):
    calls = []

    def inspect(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps([labels, {"9552/tcp": [{"HostIp": host, "HostPort": "9668"}]}, True]))

    monkeypatch.setattr(service.subprocess, "run", inspect)
    binding = service.Controller._docker_binding(controller, refresh=True)
    expected = labels["com.docker.compose.project"] == "cascadeur-kimodo" and host == "127.0.0.1"
    assert binding["verified"] is expected
    assert binding["port"] == (9668 if expected else None)
    command, options = calls[0]
    assert command[-1] == "cascadeur-kimodo-backend-1" and options["timeout"] == 3
    assert not options.get("shell")


def test_private_config_refuses_credentials_and_git_storage(controller, tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        controller.configure({"api_key": "fixture-secret"})
    with pytest.raises(ValueError):
        controller.configure({"review": {"api_key": "fixture-secret"}})
    forbidden = tmp_path / "git-checkout"
    (forbidden / ".git").mkdir(parents=True)
    monkeypatch.setenv("MCP_FOR_CASCADEUR_HOME", str(forbidden / "private"))
    with pytest.raises(ValueError, match="outside Git"):
        service._private_path("smart-motion-service.json")


def test_http_requires_token_local_origin_and_host_and_never_returns_token(controller):
    token = "fixture-local-token"
    server = service.make_server(controller, token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    info = {"endpoint": endpoint, "token": token}
    try:
        result = service._request(info, "GET", "/status")
        assert result["service"] == "smart-motion" and token not in json.dumps(result)
        for headers in ({}, {"X-Smart-Motion-Token": token, "Origin": "https://foreign.example"},
                        {"X-Smart-Motion-Token": token, "Host": "foreign.example"}):
            request = urllib.request.Request(endpoint + "/status", headers=headers)
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=2)
            assert caught.value.code == 403
        with pytest.raises(service.MotionError, match="credentials"):
            service._request(info, "POST", "/jobs", {"prompt": "walk", "api_key": "fixture-secret"})
        assert not controller.test_processes
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
