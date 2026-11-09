"""Local OpenAI-compatible visual review transport and bounded video sampling.

The adapter consumes the same labelled media and review schema as Gemini. Native
video requests target a trusted local vLLM server. Frame sequences support
image-capable local servers without pretending a single image is a motion test.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import math
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class ProviderError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def local_endpoint(base_url):
    """Only loopback servers may receive private local review media."""
    try:
        parsed = urllib.parse.urlsplit(base_url)
        host = parsed.hostname
        loopback = host == "localhost" or bool(host and ipaddress.ip_address(host).is_loopback)
        port = parsed.port
    except (ValueError, TypeError):
        raise ValueError("Visual review base URL must use a loopback HTTP endpoint") from None
    if (parsed.scheme not in {"http", "https"} or not loopback or parsed.username or parsed.password
            or parsed.query or parsed.fragment or "%" in parsed.path or ".." in parsed.path.split("/")
            or port is not None and not 1 <= port <= 65535):
        raise ValueError("Visual review base URL must use a loopback HTTP endpoint without credentials")
    return base_url.rstrip("/") + "/chat/completions"


def json_schema(schema):
    """Translate Gemini's schema spelling to strict JSON Schema, preserving fields."""
    if isinstance(schema, list):
        return [json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: json_schema(value) for key, value in schema.items() if key != "nullable"}
    if isinstance(result.get("type"), str):
        result["type"] = result["type"].lower()
    if schema.get("nullable"):
        result["type"] = [result["type"], "null"]
    if result.get("type") == "object":
        result["additionalProperties"] = False
    return result


def _bounded_integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def decode_frames(path, sampling_fps, *, frame_width=960, max_frames=64):
    """Read a complete fixed-rate JPEG sequence in memory; never truncate to fit."""
    _bounded_integer(frame_width, "frame_width", 224, 1280)
    _bounded_integer(max_frames, "max_frames_per_video", 3, 512)
    if not math.isfinite(sampling_fps) or not 0 < sampling_fps <= 24:
        raise ValueError("sampling_fps must be positive and no greater than 24")
    try:
        import imageio_ffmpeg
    except ImportError:
        raise ProviderError("Frame transport needs imageio-ffmpeg in the host environment") from None
    # One extra frame detects a clip exceeding the explicit budget. Losing the
    # rest of a clip silently would make the report's temporal claim misleading.
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(Path(path).resolve(strict=True)),
        "-an", "-map_metadata", "-1", "-vf",
        f"fps={sampling_fps}:start_time=0,scale={frame_width}:{frame_width}:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-frames:v", str(max_frames + 1), "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "pipe:1",
    ]
    try:
        process = subprocess.run(command, capture_output=True, check=True, timeout=90,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        raise ProviderError("CPU video frame extraction failed; no visual review was submitted") from None
    raw = process.stdout
    images = re.findall(b"\xff\xd8.*?\xff\xd9", raw, re.DOTALL)
    if not images or b"".join(images) != raw:
        raise ProviderError("Video decoding did not produce a complete JPEG frame sequence")
    if len(images) > max_frames:
        raise ProviderError("Video exceeds max_frames_per_video; explicitly lower sampling FPS or increase its budget")
    if len(images) < 3:
        raise ProviderError("At least three ordered frames are required to assess motion")
    return [{"seconds": index / sampling_fps, "jpeg": image} for index, image in enumerate(images)]


def frame_parts(candidates, parts, evidence, sampling_fps, *, max_inline_bytes,
                max_total_frames=96, max_frames_per_video=64, frame_width=960):
    """Replace each video part with a labelled full-clip sequence for either API."""
    _bounded_integer(max_total_frames, "max_total_frames", 3, 4096)
    _bounded_integer(max_frames_per_video, "max_frames_per_video", 3, 512)
    _bounded_integer(frame_width, "frame_width", 224, 1280)
    output, frame_count, total_bytes, part_index = [], 0, 0, 0
    for candidate, candidate_evidence in zip(candidates, evidence, strict=True):
        for item, recorded in zip(candidate["media"], candidate_evidence["media"], strict=True):
            label, data = parts[part_index:part_index + 2]
            part_index += 2
            output.append(label)
            if recorded["kind"] != "video":
                total_bytes += recorded["bytes"]
                if total_bytes > max_inline_bytes:
                    raise ProviderError("Frame sequence exceeds the configured inline byte budget")
                output.append(data)
                recorded.update(transport="image", temporal_evidence=False)
                continue
            sampled = decode_frames(item["path"], sampling_fps, frame_width=frame_width,
                                    max_frames=max_frames_per_video)
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != recorded["sha256"]:
                raise ProviderError("Source video changed during frame extraction")
            frame_count += len(sampled)
            if frame_count > max_total_frames:
                raise ProviderError("Review exceeds max_total_frames; choose fewer clips, lower FPS or an explicit larger budget")
            timestamps = []
            for sample in sampled:
                total_bytes += len(sample["jpeg"])
                if total_bytes > max_inline_bytes:
                    raise ProviderError("Frame sequence exceeds the configured inline byte budget")
                timestamps.append(sample["seconds"])
                output.append({"text": f"Candidate {candidate['id']}; view {recorded['view']}; time {sample['seconds']:.6f} seconds."})
                output.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(sample["jpeg"]).decode("ascii")}})
            recorded.update(transport="frames", temporal_evidence=True, frame_count=len(sampled),
                            timestamps_seconds=timestamps, frame_width_bound=frame_width,
                            transmitted_frame_bytes=sum(len(frame["jpeg"]) for frame in sampled))
    return output, evidence


def openai_parts(parts):
    output = []
    for part in parts:
        if "text" in part:
            output.append({"type": "text", "text": part["text"]})
        else:
            data = part["inlineData"]
            url = f"data:{data['mimeType']};base64,{data['data']}"
            kind = "video_url" if data["mimeType"].startswith("video/") else "image_url"
            output.append({"type": kind, kind: {"url": url}})
    return output


def request_review(parts, schema, *, model, base_url, api_key, sampling_fps,
                   transport, timeout, max_output_tokens):
    """One local chat request. Errors and redirects never reflect credential data."""
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,199}", model):
        raise ValueError("Set VISUAL_REVIEW_MODEL to the model ID served by the local endpoint")
    endpoint = local_endpoint(base_url)
    payload = {
        "model": model, "messages": [{"role": "user", "content": openai_parts(parts)}],
        "temperature": 0.2, "max_tokens": max_output_tokens,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "cascadeur_motion_review", "strict": True, "schema": json_schema(schema),
        }},
    }
    if transport == "video":
        payload["mm_processor_kwargs"] = {"fps": sampling_fps, "do_sample_frames": True}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                                     headers=headers, method="POST")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderError("Local visual review response exceeds its byte limit")
        result = json.loads(raw)
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
        raise ProviderError(f"Local visual review failed with HTTP {status}; no retry or provider change was made") from None
    except (urllib.error.URLError, OSError):
        raise ProviderError("Local visual review did not complete; no retry or provider change was made") from None
    except (ValueError, UnicodeError):
        raise ProviderError("Local visual review returned invalid JSON") from None
    try:
        choice = result["choices"][0]
        if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
            raise ProviderError("Local visual review was refused or truncated")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Invalid message")
        review = json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError):
        raise ProviderError("Local model did not return a complete structured review") from None
    return {"review": review, "actual_model": result.get("model", model), "usage": result.get("usage", {})}
