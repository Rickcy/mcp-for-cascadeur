"""Cloud contracts tested without credentials, provider traffic, or native apps."""

import asyncio
import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_for_cascadeur import cloud
from mcp_for_cascadeur.cloud import CLOUD_TOOLS, CloudError, CloudJobs
from mcp_for_cascadeur.providers.uthana import ProviderError, UthanaProvider

FBX = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400) + b"test-animation" * 4


def movie(seconds=2.0667, *, track_seconds=None, fragment=False, edit=None):
    def box(kind, body):
        return struct.pack(">I4s", len(body) + 8, kind) + body

    mvhd = bytes(12) + struct.pack(">II", 10000, round(seconds * 10000))
    tkhd = bytes(20) + struct.pack(">I", round(seconds * 10000))
    frames = round((seconds if track_seconds is None else track_seconds) * 30)
    mdhd = bytes(12) + struct.pack(">II", 300000, frames * 10000)
    hdlr = bytes(8) + b"vide"
    stts = bytes(4) + struct.pack(">III", 1, frames, 10000)
    edits = box(b"edts", box(b"elst", edit)) if edit is not None else b""
    media = box(
        b"mdia", box(b"mdhd", mdhd) + box(b"hdlr", hdlr) + box(b"minf", box(b"stbl", box(b"stts", stts)))
    )
    track = box(b"trak", box(b"tkhd", tkhd) + edits + media)
    fragment = box(b"mvex", box(b"mehd", bytes(4) + struct.pack(">I", 600000))) if fragment else b""
    return box(b"ftyp", b"isom0000") + box(b"moov", box(b"mvhd", mvhd) + fragment + track)


def glb():
    document = json.dumps({"asset": {"version": "2.0"}, "animations": []}).encode()
    document += b" " * (-len(document) % 4)
    return struct.pack("<4sIIII", b"glTF", 2, 20 + len(document), len(document), 0x4E4F534A) + document


class FakeProvider:
    def __init__(self):
        self.rates = [
            {"model_key": "text-to-motion-3.0", "billing_unit": "second", "unit_price": "0.10"},
            {"model_key": "video-to-motion-2.1", "billing_unit": "second", "unit_price": "0.05"},
        ]
        self.calls = []
        self.on_submit = None
        self.error = None
        self.poll_result = {"status": "FINISHED", "motion_ids": ["motion-1"]}
        self.content = FBX

    async def prices(self):
        return copy.deepcopy(self.rates)

    async def account(self):
        return {"payg_enabled": True, "payg_total_usd": "4.90"}

    async def submit_text(self, **arguments):
        self.calls.append(("text", arguments))
        if self.on_submit:
            self.on_submit()
        if self.error:
            raise self.error
        return {"job_id": "provider-job-1"}

    async def submit_video(self, **arguments):
        self.calls.append(("video", arguments))
        if self.on_submit:
            self.on_submit()
        if self.error:
            raise self.error
        return {"job_id": "provider-job-1"}

    async def upload_character(self, **arguments):
        self.calls.append(("character", arguments))
        return {"character_id": "cascy-target"}

    async def poll(self, job_id):
        self.calls.append(("poll", job_id))
        if self.error:
            raise self.error
        return copy.deepcopy(self.poll_result)

    async def download(self, **arguments):
        self.calls.append(("download", arguments))
        if self.error:
            raise self.error
        return self.content


@pytest.fixture
def service(tmp_path):
    return CloudJobs(tmp_path, provider=FakeProvider())


async def start(service, *, request_id="walk-1", **changes):
    arguments = {
        "prompt": "Walk slowly under heavy armor",
        "length": 4,
        "max_cost_usd": 0.50,
        "request_id": request_id,
    }
    arguments.update(changes)
    return await service.uthana_text_start(**arguments)


async def completed(service):
    result = await start(service)
    return await service.uthana_job_status(result["job_id"])


@pytest.mark.asyncio
async def test_text_intent_and_exact_current_price_are_durable_before_provider_submission(service):
    def check():
        paths = list((service.root / "jobs").glob("*.json"))
        assert len(paths) == 1
        saved = service._read(paths[0].stem)
        assert saved["status"] == "submitting"
        assert saved["quote"]["estimated_cost_usd"] == "0.50"
        assert saved["quote"]["requested_seconds"] == "4"
        assert saved["quote"]["reserved_seconds"] == "5"
        assert saved["quote"]["provider_hard_cap"] is False
        assert saved["quote"]["measured_debit_usd"] is None

    service.provider.on_submit = check
    result = await start(service)
    assert result["status"] == "submitted"
    assert result["upstream_job_id"] == "provider-job-1"
    assert result["can_resubmit"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["budget", "missing_rate", "duplicate_rate", "wrong_unit", "negative", "nan"]
)
async def test_unknown_ambiguous_or_over_budget_prices_never_submit(service, case):
    changes = {}
    if case == "budget":
        changes["max_cost_usd"] = 0.49
    elif case == "missing_rate":
        service.provider.rates.clear()
    elif case == "duplicate_rate":
        service.provider.rates.append(service.provider.rates[0])
    elif case == "wrong_unit":
        service.provider.rates[0]["billing_unit"] = "credit"
    else:
        service.provider.rates[0]["unit_price"] = "-1" if case == "negative" else "NaN"
    with pytest.raises(CloudError):
        await start(service, **changes)
    assert service.provider.calls == []
    assert list((service.root / "jobs").glob("*.json")) == []


@pytest.mark.asyncio
async def test_restart_replay_and_changed_arguments_cannot_duplicate_paid_generation(service):
    result = await start(service)
    restarted = CloudJobs(service.workspace, provider=FakeProvider())
    assert (await start(restarted))["upstream_job_id"] == "provider-job-1"
    assert restarted.provider.calls == []
    with pytest.raises(CloudError, match="different cloud work"):
        await start(restarted, prompt="Jump instead")
    assert restarted.provider.calls == []
    assert (await restarted.uthana_job_status(result["job_id"]))["motion_id"] == "motion-1"
    assert restarted.provider.calls == [("poll", "provider-job-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_unknown_submission_survives_interruption_and_cannot_replay(service, cancelled):
    secret = "never-write-this-provider-secret"
    service.provider.error = asyncio.CancelledError() if cancelled else RuntimeError(secret)
    expected = asyncio.CancelledError if cancelled else CloudError
    with pytest.raises(expected):
        await start(service)
    restarted = CloudJobs(service.workspace, provider=FakeProvider())
    receipt = (await restarted.uthana_jobs_list())["jobs"][0]
    assert receipt["status"] == "unknown"
    assert secret not in json.dumps(receipt)
    assert secret not in next((service.root / "jobs").glob("*.json")).read_text()
    replay = await start(restarted)
    assert replay["status"] == "unknown"
    assert restarted.provider.calls == []


@pytest.mark.asyncio
async def test_submitting_intent_after_forced_process_loss_reads_as_unknown(service):
    result = await start(service)
    record = service._read(result["job_id"])
    record["status"] = "submitting"
    record.pop("upstream_job_id")
    service._write(record)
    restarted = CloudJobs(service.workspace, provider=FakeProvider())
    assert (await restarted.uthana_job_status(result["job_id"]))["status"] == "unknown"
    assert (await start(restarted))["status"] == "unknown"
    assert restarted.provider.calls == []


@pytest.mark.asyncio
async def test_confirmed_rejection_is_recorded_without_replaying_it(service):
    service.provider.error = ProviderError("UNAUTHENTICATED", "sensitive upstream body", rejected=True)
    with pytest.raises(CloudError) as caught:
        await start(service)
    assert caught.value.execution_state == "not_started"
    result = await start(service)
    assert result["status"] == "rejected"
    assert len(service.provider.calls) == 1
    assert "sensitive" not in json.dumps(result)


@pytest.mark.asyncio
async def test_persistence_failure_after_submission_retains_known_job_for_polling(service, monkeypatch):
    write = service._write
    failed = False

    def fail_receipt(record):
        nonlocal failed
        if record["status"] == "submitted" and not failed:
            failed = True
            raise OSError("disk busy")
        write(record)

    monkeypatch.setattr(service, "_write", fail_receipt)
    with pytest.raises(CloudError):
        await start(service)
    receipt = (await service.uthana_jobs_list())["jobs"][0]
    assert receipt["status"] == "unknown"
    assert receipt["upstream_job_id"] == "provider-job-1"
    result = await service.uthana_job_status(receipt["job_id"])
    assert result["status"] == "completed"
    assert [kind for kind, _ in service.provider.calls] == ["text", "poll"]


@pytest.mark.asyncio
async def test_missing_finished_motion_and_poll_failures_do_not_regenerate(service):
    result = await start(service)
    service.provider.poll_result = {"status": "FINISHED", "motion_ids": []}
    with pytest.raises(CloudError, match="without regenerating"):
        await service.uthana_job_status(result["job_id"])
    assert (await service.uthana_job_status(result["job_id"], refresh=False))["status"] == "submitted"
    service.provider.poll_result = {"status": "FAILED", "motion_ids": []}
    assert (await service.uthana_job_status(result["job_id"]))["status"] == "failed"
    assert [kind for kind, _ in service.provider.calls].count("text") == 1


@pytest.mark.asyncio
async def test_video_hash_and_measured_duration_define_budget_and_uploaded_bytes(service):
    path = service.workspace / "walk.mp4"
    source = movie()
    path.write_bytes(source)
    result = await service.uthana_video_start(
        str(path), max_cost_usd=0.15, request_id="reference-1", character_id="cascy"
    )
    assert result["quote"]["quantity"] == "3"
    assert result["quote"]["estimated_cost_usd"] == "0.15"
    assert result["arguments"]["input"]["sha256"] == hashlib.sha256(source).hexdigest()
    assert service.provider.calls[0][1]["content"] == source
    assert service.provider.calls[0][1]["character_id"] == "cascy"
    path.write_bytes(movie(3))
    with pytest.raises(CloudError, match="different cloud work"):
        await service.uthana_video_start(
            str(path), max_cost_usd=0.15, request_id="reference-1", character_id="cascy"
        )
    assert len(service.provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["short", "long", "unknown_duration", "corrupt", "outside", "url"])
async def test_unmeasured_or_out_of_scope_video_never_reaches_provider(service, case, tmp_path):
    path = service.workspace / "video.mp4"
    content = {
        "short": movie(1),
        "long": movie(61),
        "unknown_duration": movie(0),
        "corrupt": b"not a movie",
    }.get(case, movie())
    path.write_bytes(content)
    if case == "outside":
        path = tmp_path.parent / "outside-test-cloud.mp4"
        path.write_bytes(content)
    with pytest.raises(CloudError):
        await service.uthana_video_start(
            "https://example.com/a.mp4" if case == "url" else str(path), 1, "video-invalid"
        )
    assert service.provider.calls == []


@pytest.mark.asyncio
async def test_existing_rig_upload_returns_retarget_identity_with_explicit_no_autorig_policy(service):
    path = service.workspace / "Cascy.fbx"
    path.write_bytes(FBX)
    result = await service.uthana_character_upload(str(path), max_cost_usd=0, request_id="existing-rig")
    assert result["status"] == "completed"
    assert result["character_id"] == "cascy-target"
    assert result["arguments"]["auto_rig"] is False
    assert result["quote"]["estimated_cost_usd"] == "0"
    assert "auto_rig=false" in result["quote"]["basis"]


@pytest.mark.asyncio
async def test_download_retains_target_identity_checks_bytes_and_replays_only_unchanged_file(service):
    result = await completed(service)
    target = service.workspace / "exports" / "walking.fbx"
    artifact = await service.uthana_download(
        result["job_id"], str(target), character_id="cascy-target", fps=60
    )
    assert artifact["sha256"] == hashlib.sha256(FBX).hexdigest()
    assert artifact["animation_verified"] is False
    call = service.provider.calls[-1]
    assert call[1]["character_id"] == "cascy-target"
    assert call[1]["fps"] == 60
    assert (
        await service.uthana_download(result["job_id"], str(target), character_id="cascy-target", fps=60)
    )["replayed"]
    assert service.provider.calls[-1] == call
    target.write_bytes(FBX + b"modified")
    with pytest.raises(CloudError, match="missing or changed"):
        await service.uthana_download(result["job_id"], str(target), character_id="cascy-target", fps=60)
    assert target.read_bytes() == FBX + b"modified"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["existing", "outside", "url", "wrong_format", "provider_error"])
async def test_download_failure_never_overwrites_or_erases_completed_generation(service, case, tmp_path):
    result = await completed(service)
    target = service.workspace / "walking.fbx"
    if case == "existing":
        target.write_bytes(b"precious")
    elif case == "outside":
        target = tmp_path.parent / "outside-animation.fbx"
    elif case == "wrong_format":
        service.provider.content = b'{"error":"not animation"}'
    elif case == "provider_error":
        service.provider.error = RuntimeError("private-key-from-upstream")
    with pytest.raises(CloudError) as caught:
        await service.uthana_download(
            result["job_id"], "https://evil.test/out.fbx" if case == "url" else str(target)
        )
    assert "private-key" not in str(caught.value)
    assert (await service.uthana_job_status(result["job_id"], refresh=False))["status"] == "completed"
    if case == "existing":
        assert target.read_bytes() == b"precious"


@pytest.mark.asyncio
async def test_glb_format_and_length_are_verified_before_persisting(service):
    result = await completed(service)
    service.provider.content = glb()
    target = service.workspace / "walk.glb"
    assert (await service.uthana_download(result["job_id"], str(target), output_format="glb"))[
        "status"
    ] == "completed"
    service.provider.content = glb() + b"trailing unexpected bytes"
    invalid = service.workspace / "broken.glb"
    with pytest.raises(CloudError, match="do not match"):
        await service.uthana_download(result["job_id"], str(invalid), output_format="glb")
    assert not invalid.exists()


@pytest.mark.asyncio
async def test_storage_checksum_and_request_identity_are_enforced(service):
    result = await start(service)
    path = service._path(result["job_id"])
    raw = json.loads(path.read_text())
    raw["status"] = "completed"
    path.write_text(json.dumps(raw))
    with pytest.raises(CloudError, match="verified"):
        await service.uthana_job_status(result["job_id"], refresh=False)
    with pytest.raises(CloudError, match="identity"):
        await service.uthana_job_status("../outside")


@pytest.mark.asyncio
async def test_duplicate_concurrent_submit_is_blocked_by_receipt_lock(service):
    entered, release = asyncio.Event(), asyncio.Event()

    async def submit(**_arguments):
        entered.set()
        await release.wait()
        return {"job_id": "only-one-job"}

    service.provider.submit_text = submit
    first = asyncio.create_task(start(service))
    await entered.wait()
    try:
        with pytest.raises(CloudError, match="currently owns"):
            await start(service)
    finally:
        release.set()
    assert (await first)["upstream_job_id"] == "only-one-job"


@pytest.mark.asyncio
async def test_tool_discovery_and_invalid_arguments_do_not_require_credentials(service):
    assert service.tools().keys() == CLOUD_TOOLS.keys()
    with pytest.raises(CloudError):
        await service.call("uthana_text_start", {"api_key": "must-not-be-an-argument"})
    assert service.provider.calls == []


@pytest.mark.asyncio
async def test_provider_uses_official_sdk_shapes_disables_telemetry_and_closes_sync_client():
    calls = []

    async def text(prompt, **arguments):
        calls.append((prompt, arguments))
        return {"id": "remote-job", "status": "READY"}

    async def upload(filename, content, **arguments):
        calls.append((filename, content, arguments))
        return SimpleNamespace(character_id="target-cascy")

    client = SimpleNamespace(
        ttm=SimpleNamespace(create_job=text),
        characters=SimpleNamespace(create_from_bytes=upload),
        close=Mock(),
    )
    factory = Mock(return_value=client)
    provider = UthanaProvider(lambda: "private-api-key", client_factory=factory)
    assert await provider.submit_text(prompt="Heavy walk", length=4, character_id="cascy") == {
        "job_id": "remote-job"
    }
    assert calls[0][1] == {
        "model": "text-to-motion-3.0",
        "length": 4,
        "character_id": "cascy",
        "rewrite_prompt": True,
        "fast": False,
    }
    assert factory.call_args.kwargs["telemetry"] is False
    assert factory.call_args.kwargs["domain"] == "uthana.com"
    assert factory.call_args.kwargs["trust_env"] is False
    uploaded = await provider.upload_character(content=FBX, filename="cascy.fbx")
    assert uploaded["character_id"] == "target-cascy"
    assert uploaded["compatibility_warning"] is False
    assert calls[1][2]["auto_rig"] is False
    assert client.close.call_count == 2


@pytest.mark.asyncio
async def test_provider_filters_account_data_and_suppresses_secret_errors():
    async def usage():
        return {
            "org": {
                "payg_enabled": True,
                "payg_total_usd": 4.9,
                "motion_download_secs_per_month_remaining": 100,
                "characters_allowed_remaining": 3,
                "api_key": "private-secret",
            }
        }

    client = SimpleNamespace(org=SimpleNamespace(get_usage=usage), close=Mock())
    provider = UthanaProvider(lambda: "private-secret", client_factory=lambda *_args, **_kwargs: client)
    result = await provider.account()
    assert result["motion_download_secs_remaining"] == "100"
    assert result["characters_remaining"] == "3"
    assert "private-secret" not in json.dumps(result)

    async def fail(*_args, **_kwargs):
        raise RuntimeError("private-secret in SDK URL and response")

    client.jobs = SimpleNamespace(get=fail)
    with pytest.raises(ProviderError) as caught:
        await provider.poll("known-job")
    assert "private-secret" not in str(caught.value)
    assert caught.value.rejected is False


@pytest.mark.asyncio
async def test_provider_keeps_real_dotted_model_price_keys_for_budgeting(tmp_path):
    async def prices():
        return [
            {"model_key": "text-to-motion-3.0", "billing_unit": "second", "unit_price": 0.10},
            {"model_key": "video-to-motion-2.1", "billing_unit": "second", "unit_price": "0.05"},
        ]

    client = SimpleNamespace(org=SimpleNamespace(get_prices=prices), close=Mock())
    provider = UthanaProvider(lambda: "private-secret", client_factory=lambda *_args, **_kwargs: client)
    service = CloudJobs(tmp_path, provider=provider)
    assert (await service._quote("text-to-motion-3.0", 4, 0.5))["estimated_cost_usd"] == "0.5"
    assert (await service._quote("video-to-motion-2.1", 3, 0.15))["estimated_cost_usd"] == "0.15"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"result": {"id": "refined-motion"}}, ["refined-motion"]),
        (json.dumps({"result": json.dumps({"motion_id": "refined-motion"})}), ["refined-motion"]),
        ({"result": {"result": {"motion_id": "refined-motion"}}}, ["refined-motion"]),
        ({"character": {"id": "character-not-motion"}}, []),
    ],
)
async def test_provider_polls_known_motion_envelopes_without_confusing_character_ids(payload, expected):
    async def get(job_id):
        return {"id": job_id, "status": "FINISHED", "result": payload}

    client = SimpleNamespace(jobs=SimpleNamespace(get=get), close=Mock())
    provider = UthanaProvider(lambda: "private-secret", client_factory=lambda *_args, **_kwargs: client)
    result = await provider.poll("known-job")
    assert result["motion_ids"] == expected


@pytest.mark.asyncio
async def test_provider_video_uses_vtm21_byte_snapshot_and_retargets_only_at_download():
    calls = []

    async def create(filename, content, **arguments):
        calls.append((filename, content, arguments))
        return {"id": "video-job", "status": "RESERVED"}

    async def download(character_id, motion_id, **arguments):
        calls.append((character_id, motion_id, arguments))
        return FBX

    client = SimpleNamespace(
        vtm=SimpleNamespace(create_from_bytes=create),
        motions=SimpleNamespace(download=download),
        close=Mock(),
    )
    provider = UthanaProvider(lambda: "private-secret", client_factory=lambda *_args, **_kwargs: client)
    source = movie()
    assert await provider.submit_video(content=source, filename="walk.mp4", character_id="cascy") == {
        "job_id": "video-job"
    }
    assert calls[0][0:2] == ("walk.mp4", source)
    assert calls[0][2]["model"] == "video-to-motion-2.1"
    assert "character_id" not in calls[0][2]
    await provider.download(
        character_id="cascy", motion_id="refined-motion", output_format="fbx", fps=30, max_bytes=1024
    )
    assert calls[1][0:2] == ("cascy", "refined-motion")
    assert calls[1][2] == {"output_format": "fbx", "fps": 30, "no_mesh": True, "max_bytes": 1024}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["RUNNING", "PENDING", "QUEUED", "PROCESSING", "RESERVED", "READY"])
async def test_additional_pending_states_preserve_known_job_without_claiming_success(service, status):
    result = await start(service)
    service.provider.poll_result = {"status": status, "motion_ids": []}
    observed = await service.uthana_job_status(result["job_id"])
    assert observed["status"] == "submitted"
    assert observed["upstream_status"] == status
    assert observed["upstream_job_id"] == "provider-job-1"
    assert observed["can_poll"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["CANCELED", "CANCELLED"])
async def test_provider_terminal_cancellation_never_publishes_a_motion(service, status):
    result = await start(service)
    service.provider.poll_result = {"status": status, "motion_ids": []}
    observed = await service.uthana_job_status(result["job_id"])
    assert observed["status"] == "failed"
    assert "motion_ids" not in observed


@pytest.mark.asyncio
async def test_partial_stage_write_never_publishes_a_final_animation(service, monkeypatch):
    result = await completed(service)
    target = service.workspace / "exports" / "interrupted.fbx"

    def fail(path, content):
        path.write_bytes(content[:10])
        raise OSError("disk full midway through write")

    monkeypatch.setattr(cloud, "_write_animation_stage", fail)
    with pytest.raises(CloudError, match="download failed"):
        await service.uthana_download(result["job_id"], str(target))
    assert not target.exists()
    assert list(target.parent.glob("*.download")) == []
    assert (await service.uthana_job_status(result["job_id"], refresh=False))["status"] == "completed"


@pytest.mark.asyncio
async def test_publication_race_preserves_the_file_created_by_another_writer(service, monkeypatch):
    result = await completed(service)
    target = service.workspace / "raced.fbx"
    link = cloud.os.link

    def other_writer(stage, destination):
        Path(destination).write_bytes(b"another writer's file")
        link(stage, destination)

    monkeypatch.setattr(cloud.os, "link", other_writer)
    with pytest.raises(CloudError, match="not replaced"):
        await service.uthana_download(result["job_id"], str(target))
    assert target.read_bytes() == b"another writer's file"
    assert list(target.parent.glob("*.download")) == []


@pytest.mark.asyncio
async def test_restarted_export_recognizes_published_bytes_after_final_journal_write_failure(
    service, monkeypatch
):
    result = await completed(service)
    target = service.workspace / "published.fbx"
    write = service._write

    def fail_final(record):
        if record["exports"] and record["exports"][-1]["status"] == "completed":
            raise OSError("final journal write failed")
        write(record)

    monkeypatch.setattr(service, "_write", fail_final)
    with pytest.raises(CloudError):
        await service.uthana_download(result["job_id"], str(target))
    assert target.read_bytes() == FBX
    restarted = CloudJobs(service.workspace, provider=FakeProvider())
    recovered = await restarted.uthana_download(result["job_id"], str(target))
    assert recovered["status"] == "completed"
    assert recovered["replayed"] is True
    assert restarted.provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["fragmented", "contradictory_track", "edit_speed", "edit_duration", "edit_offset"]
)
async def test_fragmented_or_contradictory_video_timing_is_rejected_before_costing_or_upload(service, case):
    arguments = {}
    if case == "fragmented":
        arguments["fragment"] = True
    elif case == "contradictory_track":
        arguments["track_seconds"] = 60
    else:
        segment = 600000 if case == "edit_duration" else 20000
        offset = 900000 if case == "edit_offset" else 0
        rate = 2 if case == "edit_speed" else 1
        arguments["edit"] = bytes(4) + struct.pack(">IIihh", 1, segment, offset, rate, 0)
    path = service.workspace / "ambiguous.mp4"
    path.write_bytes(movie(2, **arguments))
    with pytest.raises(CloudError, match="unfragmented"):
        await service.uthana_video_start(str(path), max_cost_usd=1, request_id="invalid-timing")
    assert service.provider.calls == []


@pytest.mark.asyncio
async def test_standard_encoder_delay_edit_remains_supported_for_budgeted_video(service):
    path = service.workspace / "encoder-delay.mp4"
    edit = bytes(4) + struct.pack(">IIihh", 1, 20667, 20000, 1, 0)
    path.write_bytes(movie(edit=edit))
    result = await service.uthana_video_start(str(path), max_cost_usd=0.15, request_id="encoder-delay")
    assert result["quote"]["estimated_cost_usd"] == "0.15"


@pytest.mark.asyncio
async def test_upload_compatibility_warning_is_retained_without_provider_text(service):
    async def upload(**_arguments):
        return {"character_id": "cascy-target", "compatibility_warning": True, "auto_rig_confidence": 0.7}

    service.provider.upload_character = upload
    path = service.workspace / "rig.fbx"
    path.write_bytes(FBX)
    result = await service.uthana_character_upload(str(path), 0, "rig-warning")
    assert result["compatibility_warning"] is True
    assert result["auto_rig_confidence"] == 0.7


@pytest.mark.asyncio
@pytest.mark.parametrize("length,ceiling,expected", [(4, 0.50, "0.50"), (8, 0.90, "0.90")])
async def test_text_quote_reserves_one_second_without_changing_requested_generation_length(
    service, length, ceiling, expected
):
    result = await start(service, length=length, max_cost_usd=ceiling)
    assert result["quote"]["estimated_cost_usd"] == expected
    assert result["quote"]["requested_seconds"] == str(length)
    assert result["quote"]["reserved_seconds"] == str(length + 1)
    assert "our one-second reserve" in result["quote"]["basis"]
    assert result["quote"]["provider_hard_cap"] is False
    assert service.provider.calls[0][1]["length"] == length


@pytest.mark.asyncio
async def test_old_completed_receipt_keeps_its_pre_reserve_quote_without_reading_current_rates(
    service, monkeypatch
):
    async def old_quote(_model, _quantity, _max_cost):
        return {
            "model": "text-to-motion-3.0",
            "unit_price_usd": "0.10",
            "quantity": "4",
            "estimated_cost_usd": "0.40",
            "max_cost_usd": "0.4",
            "basis": "whole requested output seconds",
            "provider_hard_cap": False,
        }

    monkeypatch.setattr(service, "_quote", old_quote)
    submitted = await start(service, max_cost_usd=0.4)
    finished = await service.uthana_job_status(submitted["job_id"])
    restarted = CloudJobs(service.workspace, provider=FakeProvider())
    restarted.provider.rates.clear()
    replayed = await start(restarted, max_cost_usd=0.4)
    assert replayed["status"] == "completed"
    assert replayed["quote"] == finished["quote"]
    assert replayed["quote"]["estimated_cost_usd"] == "0.40"
    assert "reserved_seconds" not in replayed["quote"]
    assert restarted.provider.calls == []
