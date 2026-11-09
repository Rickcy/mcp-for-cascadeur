"""MCP contracts for the same durable jobs used by the Cascadeur panel."""

from __future__ import annotations

from .contracts import Tool

TEXT = {"type": "string", "minLength": 1, "maxLength": 4096}
REVIEW_PROPERTIES = {
    "provider": {"type": "string", "enum": ["gemini", "openai-compatible"]},
    "model": {"type": "string", "minLength": 1, "maxLength": 240},
    "base_url": TEXT,
    "media_transport": {"type": "string", "enum": ["video", "frames"]},
    "sampling_fps": {"type": "number", "minimum": 0.01, "maximum": 24},
    "max_total_frames": {"type": "integer", "minimum": 3, "maximum": 1024},
    "max_frames_per_video": {"type": "integer", "minimum": 3, "maximum": 512},
    "frame_width": {"type": "integer", "minimum": 224, "maximum": 1280},
}
GENERATION_PROPERTIES = {
    "prompt": {"type": "string", "minLength": 1, "maxLength": 4000},
    "frames": {"type": "integer", "minimum": 2},
    "samples": {"type": "integer", "minimum": 1, "maximum": 4},
    "seed": {"type": "integer", "minimum": 0, "maximum": 4294967295},
    "steps": {"type": "integer", "minimum": 1, "maximum": 200},
    "source_scene": TEXT,
    "target_scene": TEXT,
    "character_id": TEXT,
    "rig_mapping": {"type": "object", "maxProperties": 64, "additionalProperties": TEXT},
    "first_frame": {"type": "integer", "minimum": 0},
    "keyframes": {"type": "array", "minItems": 2, "uniqueItems": True,
                  "items": {"type": "integer", "minimum": 0}},
    "cfg_enabled": {"type": "boolean"},
    "text_weight": {"type": "number", "minimum": 0, "maximum": 10},
    "constraint_weight": {"type": "number", "minimum": 0, "maximum": 10},
    "sample_index": {"type": "integer", "minimum": 0, "maximum": 3},
    "review_enabled": {"type": "boolean"},
    "no_render": {"type": "boolean"},
    "auto_activate": {"type": "boolean"},
    "render_sample_step": {"type": "integer", "minimum": 1, "maximum": 30},
    "generate_only": {"type": "boolean"},
    "constraints": TEXT,
    "build": {"type": "boolean"},
    "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
    "timeout": {"type": "integer", "minimum": 30, "maximum": 86400},
    **REVIEW_PROPERTIES,
}
RESUME_PROPERTIES = {
    **REVIEW_PROPERTIES,
    "review_enabled": {"type": "boolean"},
    "no_render": {"type": "boolean"},
    "auto_activate": {"type": "boolean"},
    "render_sample_step": {"type": "integer", "minimum": 1, "maximum": 30},
}
JOB_ID = {"type": "string", "minLength": 32, "maxLength": 32}
TOOL_NAMES = frozenset({
    "kimodo_status", "kimodo_configure", "kimodo_generate", "kimodo_jobs", "kimodo_job_status",
    "kimodo_cancel", "kimodo_resume", "kimodo_review", "kimodo_select", "kimodo_runtime", "kimodo_import_run",
})


def tools(config):
    from .motion_service import call_service

    def call(method, path, payload=None):
        return call_service(method, path, payload, config=config)

    def configure(**settings):
        return call("POST", "/settings", settings)

    def generate(**options):
        return call("POST", "/jobs", options)

    def job_status(job_id):
        return call("GET", "/jobs/" + job_id)

    def cancel(job_id):
        return call("POST", f"/jobs/{job_id}/cancel", {})

    def resume(job_id, **options):
        return call("POST", f"/jobs/{job_id}/resume", options)

    def review(job_id, **options):
        return call("POST", f"/jobs/{job_id}/review", options)

    def select(job_id, candidate_id):
        return call("POST", f"/jobs/{job_id}/select", {"candidate_id": candidate_id})

    return [
        Tool("kimodo_status", "Read Smart Motion configuration, runtime readiness and visual reviewer availability.",
             lambda: call("GET", "/status")),
        Tool("kimodo_configure", "Configure prepared Kimodo paths and nonsecret generation/review defaults. Keys use the private review CLI.",
             configure, {**{name: TEXT for name in ("checkout", "calibration", "template_motion", "template_scene")},
                         "defaults": {"type": "object", "properties": GENERATION_PROPERTIES, "additionalProperties": False},
                         "review": {"type": "object", "properties": REVIEW_PROPERTIES, "additionalProperties": False}}, (), False),
        Tool("kimodo_generate", "Start a durable asynchronous Kimodo job. Poll kimodo_job_status. Long clips use sequential generation chunks on 12 GB VRAM, without a fixed clip-length ceiling. target_scene plus character_id identifies the native target rig; rig_mapping binds custom body roles. Native source_scene plus keyframes supplies anchors. no_render with review_enabled=false skips video/VLM. generate_only retains raw NPZ/FBX. Native output runs at 30 FPS.",
             generate, GENERATION_PROPERTIES, ("prompt",), False),
        Tool("kimodo_jobs", "List retained Smart Motion jobs without resubmitting work.", lambda: call("GET", "/jobs")),
        Tool("kimodo_job_status", "Read durable job progress, retained candidates, visual scores and errors. Unknown writes must be inspected before resume.",
             job_status, {"job_id": JOB_ID}, ("job_id",)),
        Tool("kimodo_cancel", "Request cooperative cancellation. An in-flight generation or native stage must finish before cancellation is confirmed; retained outputs are preserved.",
             cancel, {"job_id": JOB_ID}, ("job_id",), False),
        Tool("kimodo_resume", "Resume verified saved native/review stages without generating motion again or replaying uncertain writes.",
             resume, {"job_id": JOB_ID, **RESUME_PROPERTIES}, ("job_id",), False),
        Tool("kimodo_review", "Start another visual provider/profile review of completed native videos. Does not generate, transfer, render or activate a scene.",
             review, {"job_id": JOB_ID, **REVIEW_PROPERTIES}, ("job_id",), False),
        Tool("kimodo_select", "Open an explicitly chosen retained native candidate after its saved integrity checks. Does not overwrite the source scene.",
             select, {"job_id": JOB_ID, "candidate_id": {"type": "string", "enum": [f"candidate-{i:02d}" for i in range(1, 5)]}},
             ("job_id", "candidate_id"), False),
        Tool("kimodo_runtime", "Inspect or manage this integration's Docker runtime. Start, stop and build return asynchronous jobs; models and animations are retained.",
             lambda action: call("POST", "/runtime", {"action": action}),
             {"action": {"type": "string", "enum": ["status", "start", "stop", "build", "logs"]}}, ("action",), False),
        Tool("kimodo_import_run", "Register an existing workspace/kimodo run for panel browsing and provider re-review without generation.",
             lambda output: call("POST", "/jobs/import", {"output": output}), {"output": TEXT}, ("output",), False),
    ]
