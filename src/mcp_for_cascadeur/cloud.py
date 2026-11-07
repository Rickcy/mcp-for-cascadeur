"""Workspace-scoped cloud receipts; native scene ownership stays in the host."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import struct
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .contracts import Tool
from .providers.uthana import DEFAULT_CHARACTER, ProviderError, UthanaProvider

MAX_RECORD_BYTES = 1024 * 1024
MAX_VIDEO_BYTES = 128 * 1024 * 1024
MAX_CHARACTER_BYTES = 30 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024
MAX_JOBS = 256
_REQUEST = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_IDENTITY = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_JOB = re.compile(r"[0-9a-f]{32}\Z")
_POLICY = "https://github.com/Uthana/uthana-mcp/blob/main/src/uthana_mcp/WORKFLOW.md"


class CloudError(RuntimeError):
    def __init__(self, code, message, *, execution_state="not_started", details=None):
        super().__init__(message)
        self.code = code
        self.execution_state = execution_state
        self.details = details

    def as_dict(self):
        return {
            "code": self.code,
            "message": str(self),
            "execution_state": self.execution_state,
            **({"details": self.details} if self.details is not None else {}),
        }


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _identity(value):
    if type(value) is not str or not _IDENTITY.fullmatch(value):
        raise CloudError("CLOUD_ID_INVALID", "Use an asset identity returned by the provider")
    return value


def _budget(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
        raise CloudError("CLOUD_BUDGET_INVALID", "max_cost_usd must be a finite number between 0 and 100")
    return Decimal(str(value))


def _mp4_duration(content):
    """Bound costing to consistent, unfragmented movies with straight playback."""
    movies, tracks = [], []
    visited = 0

    def timed_header(first, last):
        version = content[first]
        if version == 0 and last - first >= 20:
            scale, duration = struct.unpack_from(">II", content, first + 12)
        elif version == 1 and last - first >= 32:
            scale = struct.unpack_from(">I", content, first + 20)[0]
            duration = struct.unpack_from(">Q", content, first + 24)[0]
        else:
            raise ValueError("Unsupported timing")
        if not scale or not duration or duration in {0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF}:
            raise ValueError("Unknown timing")
        return scale, duration

    def store(track, key, value):
        if track is None or key in track:
            raise ValueError("Ambiguous track timing")
        track[key] = value

    def boxes(start, end, parent=None, track=None, depth=0):
        nonlocal visited
        if depth > 6:
            raise ValueError("Box nesting")
        position = start
        while position < end:
            visited += 1
            if visited > 100000:
                raise ValueError("Box count")
            if position + 8 > end:
                raise ValueError("Truncated box")
            size, kind = struct.unpack_from(">I4s", content, position)
            header = 8
            if size == 1:
                if position + 16 > end:
                    raise ValueError("Truncated extended box")
                size = struct.unpack_from(">Q", content, position + 8)[0]
                header = 16
            if size == 0:
                size = end - position
            if size < header or position + size > end:
                raise ValueError("Invalid box length")
            first, last = position + header, position + size
            if kind in {b"moof", b"mvex", b"mehd"}:
                raise ValueError("Fragmented movie")
            if kind == b"trak" and parent == b"moov":
                item = {}
                tracks.append(item)
                boxes(first, last, kind, item, depth + 1)
            elif kind in {b"moov", b"mdia", b"minf", b"stbl", b"edts"}:
                boxes(first, last, kind, track, depth + 1)
            elif kind == b"mvhd" and parent == b"moov":
                movies.append(timed_header(first, last))
            elif kind == b"mdhd" and parent == b"mdia":
                store(track, "media", timed_header(first, last))
            elif kind == b"tkhd" and parent == b"trak":
                version = content[first]
                if version == 0 and last - first >= 24:
                    duration = struct.unpack_from(">I", content, first + 20)[0]
                elif version == 1 and last - first >= 36:
                    duration = struct.unpack_from(">Q", content, first + 28)[0]
                else:
                    raise ValueError("Unsupported track timing")
                store(track, "duration", duration)
            elif kind == b"hdlr" and parent == b"mdia":
                if last - first < 12:
                    raise ValueError("Missing handler")
                store(track, "handler", content[first + 8 : first + 12])
            elif kind == b"stts" and parent == b"stbl":
                if last - first < 8:
                    raise ValueError("Missing samples")
                count = struct.unpack_from(">I", content, first + 4)[0]
                if not 0 < count <= 10000 or last - first != 8 + count * 8:
                    raise ValueError("Unknown samples")
                samples = [
                    struct.unpack_from(">II", content, first + 8 + index * 8) for index in range(count)
                ]
                if any(not amount or not delta for amount, delta in samples):
                    raise ValueError("Invalid samples")
                store(track, "samples", samples)
            elif kind == b"elst" and parent == b"edts":
                if last - first < 8 or struct.unpack_from(">I", content, first + 4)[0] != 1:
                    raise ValueError("Nontrivial edit list")
                version = content[first]
                if version == 0 and last - first == 20:
                    segment, media_time, rate, fraction = struct.unpack_from(">Iihh", content, first + 8)
                elif version == 1 and last - first == 28:
                    segment, media_time, rate, fraction = struct.unpack_from(">Qqhh", content, first + 8)
                else:
                    raise ValueError("Unknown edit list")
                if rate != 1 or fraction != 0 or media_time < 0:
                    raise ValueError("Nonlinear edit list")
                store(track, "edit", (segment, media_time))
            position = last

    try:
        boxes(0, len(content))
        if len(movies) != 1 or not tracks or sum(track.get("handler") == b"vide" for track in tracks) != 1:
            raise ValueError("Missing unique video movie")
        movie_scale, movie_duration = movies[0]
        duration = Decimal(movie_duration) / movie_scale
        maximum = duration
        for track in tracks:
            if track.get("handler") not in {b"vide", b"soun"}:
                raise ValueError("Unsupported timed track")
            media_scale, media_duration = track["media"]
            sample_duration = sum(count * delta for count, delta in track["samples"])
            step = max(delta for _count, delta in track["samples"])
            if sample_duration != media_duration:
                raise ValueError("Media and sample timing disagree")
            media_seconds = Decimal(media_duration) / media_scale
            track_seconds = Decimal(track["duration"]) / movie_scale
            tolerance = max(Decimal(2) / movie_scale, Decimal(2 * step) / media_scale)
            if abs(track_seconds - duration) > tolerance or abs(media_seconds - track_seconds) > tolerance:
                raise ValueError("Contradictory track duration")
            if "edit" in track:
                segment, media_time = track["edit"]
                if (
                    abs(Decimal(segment) / movie_scale - track_seconds) > Decimal(1) / movie_scale
                    or media_time > 2 * step
                ):
                    raise ValueError("Nontrivial playback trim")
            if track["handler"] == b"vide":
                frame_count = sum(count for count, _delta in track["samples"])
                if not 24 <= Decimal(frame_count) / media_seconds <= 120:
                    raise ValueError("Video frame rate outside provider contract")
            maximum = max(maximum, media_seconds, track_seconds)
        if not Decimal(2) <= duration <= maximum <= Decimal(60):
            raise ValueError("Duration outside provider contract")
        return maximum
    except (IndexError, KeyError, struct.error, ValueError, InvalidOperation):
        raise CloudError(
            "CLOUD_VIDEO_TIMING_INVALID",
            "Use a consistent unfragmented MP4/MOV movie (2–60 seconds, 24–120 FPS) without nonlinear edits",
        ) from None


def _validate_animation(content, output_format):
    if type(content) is not bytes or not 0 < len(content) <= MAX_DOWNLOAD_BYTES:
        raise CloudError(
            "CLOUD_DOWNLOAD_INVALID", "The returned animation file is empty or exceeds its byte limit"
        )
    if output_format == "fbx":
        if content.startswith(b"Kaydara FBX Binary  \x00\x1a\x00") and len(content) >= 27:
            return
        if content.lstrip().startswith(b"; FBX") and b"FBXHeaderExtension" in content[:65536]:
            return
    elif output_format == "glb" and len(content) >= 20:
        magic, version, size, chunk_size, chunk_type = struct.unpack_from("<4sIIII", content)
        if magic == b"glTF" and version == 2 and size == len(content) and chunk_type == 0x4E4F534A:
            if 0 < chunk_size <= len(content) - 20:
                try:
                    document = json.loads(content[20 : 20 + chunk_size])
                    if document.get("asset", {}).get("version") == "2.0":
                        return
                except (ValueError, TypeError, AttributeError):
                    pass
    raise CloudError("CLOUD_DOWNLOAD_INVALID", "The returned bytes do not match the requested FBX/GLB format")


def _write_animation_stage(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(content).hexdigest()
    if path.stat().st_size != len(content) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise CloudError("CLOUD_DOWNLOAD_CHANGED", "The saved animation did not match the received bytes")
    return digest


class CloudJobs:
    """Persist intent before provider mutations and retain uncertain outcomes."""

    def __init__(self, workspace, *, provider=None, credential=None):
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = self.workspace / ".cascadeur-mcp" / "cloud"
        self.provider = provider if provider is not None else UthanaProvider(credential)

    def tools(self):
        return {name: replace(tool, handler=getattr(self, name)) for name, tool in CLOUD_TOOLS.items()}

    async def call(self, name, arguments):
        if name not in CLOUD_TOOLS:
            raise CloudError("CLOUD_TOOL_UNKNOWN", "Unknown cloud operation")
        try:
            CLOUD_TOOLS[name].validate(arguments)
        except ValueError:
            raise CloudError(
                "CLOUD_ARGUMENTS_INVALID", "Cloud arguments do not match the tool schema"
            ) from None
        return await getattr(self, name)(**arguments)

    async def aclose(self):
        close = getattr(self.provider, "aclose", None)
        if close is not None:
            await close()

    def _managed(self, path):
        if path.is_symlink() or not path.resolve().is_relative_to(self.workspace):
            raise CloudError("CLOUD_STORAGE_UNSAFE", "Cloud storage must remain inside the workspace")
        return path

    def _path(self, job_id):
        if type(job_id) is not str or not _JOB.fullmatch(job_id):
            raise CloudError("CLOUD_JOB_ID_INVALID", "Use a cloud job identity returned by this server")
        return self._managed(self.root / "jobs" / f"{job_id}.json")

    def _read(self, job_id):
        path = self._path(job_id)
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_RECORD_BYTES + 1)
        except FileNotFoundError:
            raise CloudError("CLOUD_JOB_NOT_FOUND", "No cloud job has this identity") from None
        try:
            if len(raw) > MAX_RECORD_BYTES:
                raise ValueError("Record size")
            value = json.loads(raw)
            digest = value.pop("record_sha256")
            if (
                value.get("schema_version") != 1
                or value.get("job_id") != job_id
                or digest != hashlib.sha256(_json(value)).hexdigest()
            ):
                raise ValueError("Record identity or checksum")
            return value
        except (ValueError, KeyError, TypeError, AttributeError):
            raise CloudError(
                "CLOUD_RECORD_INVALID",
                "The durable cloud receipt could not be verified",
                execution_state="unknown",
                details={"job_id": job_id},
            ) from None

    def _write(self, record):
        record["updated_at"] = _now()
        path = self._path(record["job_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._managed(path.parent)
        body = {**record, "record_sha256": hashlib.sha256(_json(record)).hexdigest()}
        raw = _json(body)
        if len(raw) > MAX_RECORD_BYTES:
            raise CloudError(
                "CLOUD_STORAGE_LIMIT", "The cloud receipt exceeds its byte limit", execution_state="unknown"
            )
        temporary = self._managed(path.with_name(path.name + f".{uuid.uuid4().hex}.tmp"))
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            if os.name != "nt":
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _lock(self, job_id):
        self._path(job_id)
        target = self._managed(self.root / "locks" / f"{job_id}.lock")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a+b") as stream:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise CloudError(
                    "CLOUD_JOB_BUSY",
                    "Another request currently owns this cloud receipt",
                    execution_state="unknown",
                    details={"job_id": job_id},
                ) from None
            try:
                yield
            finally:
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _input(self, path, suffixes, limit):
        target = Path(path).expanduser()
        if (
            not target.is_absolute()
            or not target.resolve().is_relative_to(self.workspace)
            or target.is_symlink()
        ):
            raise CloudError(
                "CLOUD_INPUT_SCOPE", "Select a regular input file inside the configured workspace"
            )
        target = target.resolve()
        if target.suffix.lower() not in suffixes or not target.is_file():
            raise CloudError(
                "CLOUD_INPUT_INVALID", "The selected input file has an unsupported format or does not exist"
            )
        with target.open("rb") as stream:
            content = stream.read(limit + 1)
        if not 0 < len(content) <= limit:
            raise CloudError(
                "CLOUD_INPUT_LIMIT", "The selected input file is empty or exceeds its byte limit"
            )
        return content, {
            "path": str(target),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def _output(self, path, output_format):
        target = Path(path).expanduser()
        if not target.is_absolute() or target.suffix.lower() != f".{output_format}":
            raise CloudError(
                "CLOUD_OUTPUT_INVALID", "Use an absolute output path with the requested format extension"
            )
        if target.is_symlink() or not target.resolve().is_relative_to(self.workspace):
            raise CloudError(
                "CLOUD_OUTPUT_SCOPE", "Animation downloads must remain inside the configured workspace"
            )
        return target.resolve()

    async def uthana_account(self):
        try:
            usage, prices = await self.provider.account(), await self.provider.prices()
        except Exception:
            raise CloudError(
                "CLOUD_ACCOUNT_UNAVAILABLE", "Uthana account or current prices could not be read"
            ) from None
        return {"provider": "uthana", "usage": usage, "prices": prices, "measured_debit": None}

    async def _quote(self, model, quantity, max_cost_usd):
        budget = _budget(max_cost_usd)
        requested = Decimal(quantity)
        text_reserve = model == "text-to-motion-3.0"
        if text_reserve:
            # Billing uses generated output duration, which can include the
            # endpoint sample. This is our allowance, not a provider guarantee.
            quantity = requested + Decimal(1)
        try:
            prices = await self.provider.prices()
        except Exception:
            raise CloudError(
                "CLOUD_PRICE_UNAVAILABLE", "Current Uthana pricing is required before submitting work"
            ) from None
        if model == "existing-rig-upload":
            # The official workflow prices unlisted platform operations as free.
            # This route explicitly disables auto-rigging and never requests rerig.
            rate, basis = Decimal(0), "unlisted platform operation; auto_rig=false"
            source = _POLICY
        else:
            rows = [row for row in prices if row.get("model_key") == model]
            if len(rows) != 1 or rows[0].get("billing_unit") != "second":
                raise CloudError(
                    "CLOUD_PRICE_UNKNOWN", "The requested model needs one current per-second USD rate"
                )
            try:
                rate = Decimal(str(rows[0]["unit_price"]))
                if not rate.is_finite() or rate < 0:
                    raise ValueError("Invalid rate")
            except (InvalidOperation, ValueError, KeyError):
                raise CloudError("CLOUD_PRICE_UNKNOWN", "The provider returned an invalid USD rate") from None
            basis = (
                "requested output seconds plus our one-second reserve; provider billing is not guaranteed"
                if text_reserve
                else "measured input seconds rounded up"
            )
            source = "provider.org.get_prices"
        estimate = rate * quantity
        if estimate > budget:
            raise CloudError(
                "CLOUD_BUDGET_EXCEEDED",
                "The current cost estimate exceeds max_cost_usd",
                details={"estimated_cost_usd": str(estimate), "max_cost_usd": str(budget)},
            )
        return {
            "model": model,
            "unit_price_usd": str(rate),
            "quantity": str(quantity),
            "estimated_cost_usd": str(estimate),
            "max_cost_usd": str(budget),
            "basis": basis,
            "source": source,
            "quoted_at": _now(),
            "measured_debit_usd": None,
            "provider_hard_cap": False,
            **(
                {"requested_seconds": str(requested), "reserved_seconds": str(quantity)}
                if text_reserve
                else {}
            ),
        }

    def _public(self, record):
        result = copy.deepcopy(record)
        result.pop("fingerprint", None)
        if result["status"] == "submitting":
            result["status"] = "unknown"
        result["can_resubmit"] = False
        result["can_poll"] = bool(result.get("upstream_job_id")) and result["status"] in {
            "submitted",
            "unknown",
        }
        result["poll_after_ms"] = 15000 if result["can_poll"] else None
        result["execution_state"] = (
            "unknown"
            if result["status"] in {"unknown", "submitted"}
            else "not_started"
            if result["status"] == "rejected"
            else result["status"]
        )
        return result

    async def _start(self, operation, model, arguments, max_cost_usd, request_id, quantity, submit):
        if type(request_id) is not str or not _REQUEST.fullmatch(request_id):
            raise CloudError(
                "CLOUD_REQUEST_ID_INVALID",
                "Choose a durable request_id containing at most 80 letters, digits, underscores or hyphens",
            )
        _budget(max_cost_usd)
        job_id = hashlib.sha256(request_id.encode()).hexdigest()[:32]
        fingerprint = hashlib.sha256(
            _json(
                {
                    "operation": operation,
                    "arguments": arguments,
                    "max_cost_usd": str(Decimal(str(max_cost_usd))),
                }
            )
        ).hexdigest()
        with self._lock(job_id):
            if self._path(job_id).exists():
                record = self._read(job_id)
                if record["request_id"] != request_id or record["fingerprint"] != fingerprint:
                    raise CloudError(
                        "CLOUD_REQUEST_REUSED",
                        "This request_id already identifies different cloud work",
                        details={"job_id": job_id},
                    )
                return self._public(record)
            if len(list((self.root / "jobs").glob("*.json"))) >= MAX_JOBS:
                raise CloudError("CLOUD_STORAGE_LIMIT", "The workspace has reached its cloud receipt limit")
            quote = await self._quote(model, quantity, max_cost_usd)
            record = {
                "schema_version": 1,
                "job_id": job_id,
                "request_id": request_id,
                "provider": "uthana",
                "operation": operation,
                "model": model,
                "arguments": arguments,
                "fingerprint": fingerprint,
                "quote": quote,
                "status": "submitting",
                "created_at": _now(),
                "exports": [],
            }
            self._write(record)
            try:
                response = await submit()
                if operation == "character_upload":
                    record.update(status="completed", character_id=_identity(response["character_id"]))
                    record["compatibility_warning"] = response.get("compatibility_warning") is True
                    confidence = response.get("auto_rig_confidence")
                    if (
                        type(confidence) in (int, float)
                        and math.isfinite(confidence)
                        and 0 <= confidence <= 1
                    ):
                        record["auto_rig_confidence"] = confidence
                else:
                    record.update(status="submitted", upstream_job_id=_identity(response["job_id"]))
                self._write(record)
            except BaseException as error:
                rejected = isinstance(error, ProviderError) and error.rejected
                record.update(
                    status="rejected" if rejected else "unknown",
                    error={
                        "code": "CLOUD_PROVIDER_REJECTED" if rejected else "CLOUD_SUBMISSION_UNKNOWN",
                        "message": "Provider submission was rejected"
                        if rejected
                        else "Submission may have completed; do not submit it again",
                    },
                )
                try:
                    self._write(record)
                except Exception:
                    pass  # The fsynced submitting intent already forbids replay.
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise CloudError(
                    record["error"]["code"],
                    record["error"]["message"],
                    execution_state="not_started" if rejected else "unknown",
                    details={"job_id": job_id, "can_resubmit": False},
                ) from None
            return self._public(record)

    async def uthana_text_start(
        self, prompt, length, max_cost_usd, request_id, character_id=DEFAULT_CHARACTER
    ):
        if (
            type(prompt) is not str
            or not 1 <= len(prompt.strip()) <= 1000
            or type(length) is not int
            or not 4 <= length <= 10
        ):
            raise CloudError(
                "CLOUD_TEXT_INVALID",
                "Use a nonempty prompt of at most 1000 characters and an integer length from 4 to 10 seconds",
            )
        character_id = _identity(character_id)
        arguments = {"prompt": prompt, "length": length, "character_id": character_id}
        return await self._start(
            "text_to_motion",
            "text-to-motion-3.0",
            arguments,
            max_cost_usd,
            request_id,
            Decimal(length),
            lambda: self.provider.submit_text(**arguments),
        )

    async def uthana_video_start(self, video_path, max_cost_usd, request_id, character_id=DEFAULT_CHARACTER):
        content, source = self._input(video_path, {".mp4", ".mov"}, MAX_VIDEO_BYTES)
        duration = _mp4_duration(content)
        source["duration_seconds"] = str(duration)
        character_id = _identity(character_id)
        arguments = {"input": source, "character_id": character_id}
        return await self._start(
            "video_to_motion",
            "video-to-motion-2.1",
            arguments,
            max_cost_usd,
            request_id,
            Decimal(math.ceil(duration)),
            lambda: self.provider.submit_video(
                content=content, filename=Path(source["path"]).name, character_id=character_id
            ),
        )

    async def uthana_character_upload(self, path, max_cost_usd, request_id):
        content, source = self._input(path, {".fbx", ".glb"}, MAX_CHARACTER_BYTES)
        _validate_animation(content, Path(source["path"]).suffix[1:].lower())
        return await self._start(
            "character_upload",
            "existing-rig-upload",
            {"input": source, "auto_rig": False},
            max_cost_usd,
            request_id,
            Decimal(1),
            lambda: self.provider.upload_character(content=content, filename=Path(source["path"]).name),
        )

    async def uthana_job_status(self, job_id, refresh=True):
        if type(refresh) is not bool:
            raise CloudError("CLOUD_ARGUMENTS_INVALID", "refresh must be a boolean")
        with self._lock(job_id):
            record = self._read(job_id)
            if (
                not refresh
                or record["status"] not in {"submitted", "unknown", "submitting"}
                or not record.get("upstream_job_id")
            ):
                return self._public(record)
            try:
                value = await self.provider.poll(record["upstream_job_id"])
                status = value["status"]
                if type(status) is not str or not re.fullmatch(r"[A-Z_]{1,40}", status):
                    raise ValueError("Unknown status")
                record["upstream_status"] = status
                if status not in {"FINISHED", "FAILED", "CANCELLED", "CANCELED"}:
                    record["status"] = "submitted"
                if status in {"FAILED", "CANCELLED", "CANCELED"}:
                    record.update(
                        status="failed",
                        error={
                            "code": "CLOUD_GENERATION_FAILED",
                            "message": "The provider reported a terminal generation failure",
                        },
                    )
                elif status == "FINISHED":
                    motions = value.get("motion_ids", [])
                    if not motions or len(motions) > 16:
                        raise ValueError("Missing motion identity")
                    record.update(status="completed", motion_ids=[_identity(item) for item in motions])
                    if len(motions) == 1:
                        record["motion_id"] = motions[0]
                self._write(record)
            except Exception:
                raise CloudError(
                    "CLOUD_POLL_UNRESOLVED",
                    "The provider job could not be resolved; poll the known job again without regenerating",
                    execution_state="unknown",
                    details={"job_id": job_id},
                ) from None
            return self._public(record)

    async def uthana_jobs_list(self, limit=20):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise CloudError("CLOUD_ARGUMENTS_INVALID", "limit must be an integer from 1 to 100")
        directory = self._managed(self.root / "jobs")
        paths = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return {
            "provider": "uthana",
            "jobs": [self._public(self._read(path.stem)) for path in paths[:limit]],
            "refreshed": False,
        }

    async def uthana_download(
        self, job_id, output_path, output_format="fbx", fps=30, character_id=None, motion_id=None
    ):
        if output_format not in {"fbx", "glb"} or type(fps) is not int or fps not in {24, 30, 60}:
            raise CloudError("CLOUD_EXPORT_INVALID", "Choose FBX/GLB and an output FPS of 24, 30 or 60")
        target = self._output(output_path, output_format)
        with self._lock(job_id):
            record = self._read(job_id)
            if record["status"] != "completed" or not record.get("motion_ids"):
                raise CloudError(
                    "CLOUD_MOTION_NOT_READY", "Poll the cloud generation until a finished motion is available"
                )
            motion_id = motion_id or record.get("motion_id")
            if motion_id not in record["motion_ids"]:
                raise CloudError("CLOUD_MOTION_ID_REQUIRED", "Select a motion identity returned by this job")
            character_id = _identity(
                character_id or record["arguments"].get("character_id", DEFAULT_CHARACTER)
            )
            identity = {
                "path": str(target),
                "format": output_format,
                "fps": fps,
                "character_id": character_id,
                "motion_id": motion_id,
            }
            for previous in record["exports"]:
                if previous["request"] == identity and previous["status"] in {"completed", "publishing"}:
                    if (
                        not target.is_file()
                        or target.is_symlink()
                        or target.stat().st_size != previous["bytes"]
                        or hashlib.sha256(target.read_bytes()).hexdigest() != previous["sha256"]
                    ):
                        raise CloudError(
                            "CLOUD_ARTIFACT_CHANGED",
                            "The previously exported file is missing or changed; it was not downloaded again",
                        )
                    if previous["status"] == "publishing":
                        previous["status"] = "completed"
                        self._write(record)
                    return {**previous, "replayed": True}
            if target.exists():
                raise CloudError(
                    "CLOUD_OUTPUT_EXISTS", "Choose a new output path; existing files are never replaced"
                )
            if len(record["exports"]) >= 32:
                raise CloudError("CLOUD_STORAGE_LIMIT", "This job has reached its saved export limit")
            export = {"request": identity, "status": "downloading", "created_at": _now()}
            record["exports"].append(export)
            self._write(record)
            stage = None
            published = False
            try:
                content = await self.provider.download(
                    character_id=character_id,
                    motion_id=motion_id,
                    output_format=output_format,
                    fps=fps,
                    max_bytes=MAX_DOWNLOAD_BYTES,
                )
                _validate_animation(content, output_format)
                self._managed(target.parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                stage = self._managed(target.with_name(f".{target.name}.{uuid.uuid4().hex}.download"))
                digest = _write_animation_stage(stage, content)
                export.update(
                    status="publishing",
                    path=str(target),
                    bytes=len(content),
                    sha256=digest,
                    verification="format_header_and_sha256",
                    animation_verified=False,
                )
                # Persist the expected bytes before exclusive publication so a
                # restarted server can recognize the completed file.
                self._write(record)
                try:
                    os.link(stage, target)
                except FileExistsError:
                    raise CloudError(
                        "CLOUD_OUTPUT_EXISTS", "Another file claimed the output path; it was not replaced"
                    ) from None
                published = True
                if os.name != "nt":
                    descriptor = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                export["status"] = "completed"
                self._write(record)
                return {**export, "replayed": False}
            except BaseException as error:
                export["status"] = "publishing" if published else "failed"
                try:
                    self._write(record)
                except Exception:
                    pass
                if isinstance(error, asyncio.CancelledError):
                    raise
                if isinstance(error, CloudError):
                    raise
                raise CloudError(
                    "CLOUD_DOWNLOAD_FAILED",
                    "The animation download failed; the generated motion remains recorded",
                    execution_state="unknown",
                    details={"job_id": job_id},
                ) from None
            finally:
                if stage is not None:
                    stage.unlink(missing_ok=True)


def _external_only(**_arguments):
    raise CloudError("CLOUD_EXTERNAL_ONLY", "Cloud tools run only in the external MCP server")


_ID_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 128}
_JOB_SCHEMA = {"type": "string", "minLength": 32, "maxLength": 32}
_COST_SCHEMA = {"type": "number", "minimum": 0, "maximum": 100}
_REQUEST_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 80}
CLOUD_TOOLS = {
    tool.name: tool
    for tool in (
        Tool(
            "uthana_account",
            "Read Uthana account usage and current USD model rates without generating work.",
            _external_only,
        ),
        Tool(
            "uthana_text_start",
            "Submit a budgeted durable text-to-motion 3.0 job. Preserve request_id; never blindly retry unknown submissions.",
            _external_only,
            {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 1000},
                "length": {"type": "integer", "minimum": 4, "maximum": 10},
                "max_cost_usd": _COST_SCHEMA,
                "request_id": _REQUEST_SCHEMA,
                "character_id": _ID_SCHEMA,
            },
            ("prompt", "length", "max_cost_usd", "request_id"),
            False,
        ),
        Tool(
            "uthana_video_start",
            "Submit a budgeted video-to-motion 2.1 job from a complete local MP4/MOV (2–60 seconds). Input bytes and timing are hashed before upload.",
            _external_only,
            {
                "video_path": {"type": "string"},
                "max_cost_usd": _COST_SCHEMA,
                "request_id": _REQUEST_SCHEMA,
                "character_id": _ID_SCHEMA,
            },
            ("video_path", "max_cost_usd", "request_id"),
            False,
        ),
        Tool(
            "uthana_character_upload",
            "Upload an existing FBX/GLB rig without auto-rigging. Returns the target character ID for retargeted downloads.",
            _external_only,
            {"path": {"type": "string"}, "max_cost_usd": _COST_SCHEMA, "request_id": _REQUEST_SCHEMA},
            ("path", "max_cost_usd", "request_id"),
            False,
        ),
        Tool(
            "uthana_job_status",
            "Read a durable cloud receipt and optionally poll its known provider job ID. Never resubmits generation.",
            _external_only,
            {"job_id": _JOB_SCHEMA, "refresh": {"type": "boolean", "default": True}},
            ("job_id",),
        ),
        Tool(
            "uthana_jobs_list",
            "List workspace cloud receipts without network access; unknown outcomes remain unresolved.",
            _external_only,
            {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}},
        ),
        Tool(
            "uthana_download",
            "Download a finished motion as verified FBX/GLB to a new workspace path. Character ID controls retargeting; native import must be checked separately.",
            _external_only,
            {
                "job_id": _JOB_SCHEMA,
                "output_path": {"type": "string"},
                "output_format": {"type": "string", "enum": ["fbx", "glb"], "default": "fbx"},
                "fps": {"type": "integer", "enum": [24, 30, 60], "default": 30},
                "character_id": _ID_SCHEMA,
                "motion_id": _ID_SCHEMA,
            },
            ("job_id", "output_path"),
            False,
        ),
    )
}
