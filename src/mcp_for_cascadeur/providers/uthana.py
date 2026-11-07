"""Small, injectable boundary around the official Uthana Python client."""

from __future__ import annotations

import inspect
import json
import os
import re
from contextlib import asynccontextmanager
from decimal import Decimal, InvalidOperation

MAX_QUERY_BYTES = 2 * 1024 * 1024
DEFAULT_CHARACTER = "cXi2eAP19XwQ"
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MODEL = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")


class ProviderError(RuntimeError):
    """A provider failure with no provider response text or credentials attached."""

    def __init__(self, code, message, *, rejected=False):
        super().__init__(message)
        self.code = code
        self.rejected = rejected


def _identifier(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise ProviderError("UTHANA_RESPONSE_INVALID", "Uthana returned an invalid asset or job identity")
    return value


def _amount(value):
    if type(value) not in (str, int, float):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return str(number) if number.is_finite() and number >= 0 else None


def _rejected(error):
    kind, status = getattr(error, "kind", None), getattr(error, "status_code", None)
    if kind == "http" and status in {401, 403}:
        return True
    payload = getattr(error, "response_data", None)
    if kind != "graphql" or type(payload) is not dict or payload.get("data"):
        return False
    errors = payload.get("errors")
    codes = {
        "UNAUTHENTICATED",
        "FORBIDDEN",
        "INVALID_ARGUMENTS",
        "INVALID_MODEL_FORMAT",
        "RATE_LIMIT_EXCEEDED",
        "INSUFFICIENT_CREDITS",
        "INSUFFICIENT_BALANCE",
    }
    return bool(errors) and all(
        type(item) is dict and item.get("extensions", {}).get("code") in codes for item in errors
    )


class UthanaProvider:
    """Instantiate the SDK lazily; never load it in Cascadeur's interpreter."""

    def __init__(self, credential=None, *, client_factory=None):
        self.credential = credential or (lambda: os.environ.get("UTHANA_API_KEY", ""))
        self.client_factory = client_factory

    @asynccontextmanager
    async def _client(self):
        try:
            key = self.credential()
        except Exception:
            raise ProviderError(
                "UTHANA_CREDENTIAL_UNAVAILABLE", "The Uthana credential could not be loaded", rejected=True
            ) from None
        if type(key) is not str or not key.strip():
            raise ProviderError(
                "UTHANA_CREDENTIAL_REQUIRED",
                "Configure a Uthana credential before using cloud tools",
                rejected=True,
            )
        factory = self.client_factory
        if factory is None:
            try:
                from uthana import Uthana
            except ImportError:
                raise ProviderError(
                    "UTHANA_SDK_MISSING",
                    "Install the supported Uthana SDK with the external server",
                    rejected=True,
                ) from None
            factory = Uthana
        try:
            client = factory(
                key,
                domain="uthana.com",
                telemetry=False,
                trust_env=False,
                max_response_bytes=MAX_QUERY_BYTES,
                max_mutation_response_bytes=MAX_QUERY_BYTES,
            )
        except Exception:
            raise ProviderError(
                "UTHANA_CLIENT_UNAVAILABLE", "The Uthana client could not initialize", rejected=True
            ) from None
        try:
            yield client
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(
                "UTHANA_REQUEST_FAILED",
                "The Uthana request failed; provider response details were withheld",
                rejected=_rejected(error),
            ) from None
        finally:
            try:
                closed = client.close()
                if inspect.isawaitable(closed):
                    await closed
            except Exception:
                pass

    async def prices(self):
        async with self._client() as client:
            rows = await client.org.get_prices()
        if type(rows) is not list or len(rows) > 1000:
            raise ProviderError("UTHANA_PRICES_INVALID", "Uthana did not return a bounded price list")
        result = []
        for row in rows:
            if type(row) is not dict:
                continue
            key, unit, price = row.get("model_key"), row.get("billing_unit"), _amount(row.get("unit_price"))
            if (
                type(key) is str
                and _MODEL.fullmatch(key)
                and type(unit) is str
                and _ID.fullmatch(unit)
                and price is not None
            ):
                result.append({"model_key": key, "billing_unit": unit, "unit_price": price})
        return result

    async def account(self):
        async with self._client() as client:
            response = await client.org.get_usage()
        if type(response) is not dict:
            raise ProviderError("UTHANA_ACCOUNT_INVALID", "Uthana did not return account usage")
        org = response.get("org") or {}
        if type(org) is not dict:
            raise ProviderError("UTHANA_ACCOUNT_INVALID", "Uthana did not return account usage")
        return {
            "provider": "uthana",
            "payg_enabled": org.get("payg_enabled") if type(org.get("payg_enabled")) is bool else None,
            "payg_total_usd": _amount(org.get("payg_total_usd")),
            "motion_download_secs_remaining": _amount(org.get("motion_download_secs_per_month_remaining")),
            "characters_remaining": _amount(org.get("characters_allowed_remaining")),
        }

    async def submit_text(self, *, prompt, length, character_id):
        async with self._client() as client:
            value = await client.ttm.create_job(
                prompt,
                model="text-to-motion-3.0",
                length=length,
                character_id=_identifier(character_id),
                rewrite_prompt=True,
                fast=False,
            )
        return {"job_id": _identifier(value.get("id") if type(value) is dict else None)}

    async def submit_video(self, *, content, filename, character_id):
        # VTM's SDK method has no target-character argument. Retarget at export.
        _identifier(character_id)
        async with self._client() as client:
            value = await client.vtm.create_from_bytes(
                filename,
                content,
                motion_name="Cascadeur video reference",
                model="video-to-motion-2.1",
                max_bytes=128 * 1024 * 1024,
            )
        return {"job_id": _identifier(value.get("id") if type(value) is dict else None)}

    async def upload_character(self, *, content, filename):
        async with self._client() as client:
            value = await client.characters.create_from_bytes(
                filename,
                content,
                name="Cascadeur existing rig",
                auto_rig=False,
                max_bytes=30 * 1024 * 1024,
            )
        confidence = getattr(value, "auto_rig_confidence", None)
        return {
            "character_id": _identifier(getattr(value, "character_id", None)),
            "compatibility_warning": bool(getattr(value, "message", None)),
            "auto_rig_confidence": float(confidence)
            if type(confidence) in (float, int) and 0 <= confidence <= 1
            else None,
        }

    async def poll(self, job_id):
        async with self._client() as client:
            value = await client.jobs.get(_identifier(job_id))
        if type(value) is not dict or value.get("id") != job_id:
            raise ProviderError("UTHANA_JOB_INVALID", "Uthana did not return the requested job")
        status = value.get("status")
        if type(status) is not str or not re.fullmatch(r"[A-Z_]{1,40}", status):
            raise ProviderError("UTHANA_JOB_INVALID", "Uthana returned an unsupported job state")
        result = {"status": status, "motion_ids": []}
        if status == "FINISHED":
            payload = value.get("result")
            # GenericScalar result envelopes can contain JSON text. Only follow
            # the provider's result field, never arbitrary nested asset IDs.
            for _ in range(5):
                if type(payload) is str:
                    try:
                        payload = json.loads(payload)
                    except (ValueError, TypeError):
                        payload = None
                        break
                if type(payload) is dict and "result" in payload:
                    payload = payload["result"]
                else:
                    break
            if type(payload) is dict:
                motion = payload.get("motion_id", payload.get("id"))
                if motion is not None:
                    result["motion_ids"] = [_identifier(motion)]
        return result

    async def download(self, *, character_id, motion_id, output_format, fps, max_bytes):
        async with self._client() as client:
            value = await client.motions.download(
                _identifier(character_id),
                _identifier(motion_id),
                output_format=output_format,
                fps=fps,
                no_mesh=True,
                max_bytes=max_bytes,
            )
        if type(value) is not bytes or not 0 < len(value) <= max_bytes:
            raise ProviderError("UTHANA_DOWNLOAD_INVALID", "Uthana did not return a bounded animation file")
        return value

    async def aclose(self):
        """Each SDK client is closed by its individual operation."""
