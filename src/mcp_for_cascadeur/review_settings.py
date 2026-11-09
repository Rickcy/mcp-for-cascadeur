"""Visual review configuration and per-user credentials, outside the checkout."""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from .settings import data_home

PROVIDERS = ("gemini", "openai-compatible")
DEFAULTS = {
    "provider": "gemini", "model": "gemini-flash-latest",
    "base_url": "http://127.0.0.1:8000/v1", "media_transport": "video",
    "sampling_fps": 12.0, "max_total_frames": 96,
    "max_frames_per_video": 48, "frame_width": 384,
}
ENVIRONMENT = {name: "VISUAL_REVIEW_" + name.upper() for name in DEFAULTS}


def review_config_path() -> Path:
    return data_home() / "visual-review.json"


def _provider(value):
    if value not in PROVIDERS:
        raise ValueError("Visual review provider must be gemini or openai-compatible")
    return value


def _validate(values):
    if set(values) - DEFAULTS.keys():
        raise ValueError("Visual review configuration contains unsupported fields; store keys separately")
    checked = dict(values)
    if "provider" in checked:
        _provider(checked["provider"])
    if "model" in checked:
        model = checked["model"]
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,240}", model):
            raise ValueError("Provide an exact model identifier without credentials or whitespace")
    if "base_url" in checked:
        try:
            parts = urlsplit(checked["base_url"])
            port = parts.port
        except (TypeError, ValueError):
            raise ValueError("Invalid local visual review endpoint") from None
        if (parts.scheme != "http" or parts.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parts.username or parts.password or parts.query or parts.fragment
                or port is not None and not 1 <= port <= 65535):
            raise ValueError("Local visual review requires a loopback HTTP endpoint without credentials")
        checked["base_url"] = checked["base_url"].rstrip("/")
    if "media_transport" in checked and checked["media_transport"] not in {"video", "frames"}:
        raise ValueError("media_transport must be video or frames")
    for name, low, high in (("sampling_fps", 0.01, 24), ("max_total_frames", 3, 1024),
                            ("max_frames_per_video", 3, 512), ("frame_width", 224, 1280)):
        if name not in checked:
            continue
        value = checked[name]
        expected = (int, float) if name == "sampling_fps" else (int,)
        if isinstance(value, bool) or not isinstance(value, expected) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be within {low}–{high}")
    return checked


def load_review_config() -> dict:
    path = review_config_path()
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Visual review configuration must contain an object")
    return _validate(value)


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".review-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def configure_review(**values) -> dict:
    updates = _validate({key: value for key, value in values.items() if value is not None})
    current = load_review_config()
    if updates.get("provider", current.get("provider")) != current.get("provider"):
        current = {}
    current.update(updates)
    provider = current.get("provider", "gemini")
    if provider == "openai-compatible" and not current.get("model"):
        raise ValueError("Configure the exact locally served model with --model")
    _atomic_bytes(review_config_path(), (json.dumps(current, indent=2) + "\n").encode())
    return public_review_status()


class _Blob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, *, decrypt=False) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Encrypted credential storage requires Windows; use environment variables here")
    from ctypes import wintypes

    buffer = ctypes.create_string_buffer(data)
    source = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = _Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes, kernel.LocalFree.restype = [ctypes.c_void_p], ctypes.c_void_p
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p if decrypt else wintypes.LPCWSTR,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
    function.restype = wintypes.BOOL
    label = None if decrypt else "mcp-for-cascadeur visual review"
    if not function(ctypes.byref(source), label, None, None, None, 1, ctypes.byref(result)):
        raise RuntimeError("Credential encryption failed; use environment variables or set the key again")
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        kernel.LocalFree(result.data)


def _credential_path(provider) -> Path:
    _provider(provider)
    path = (data_home() / "review-credentials" / (provider + ".dpapi")).resolve()
    if any((parent / ".git").exists() for parent in (path.parent, *path.parents)):
        raise ValueError("Credential storage must be outside every Git checkout")
    return path


def _key(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\n" in value or "\r" in value:
        raise ValueError("Provide one nonempty API key without line breaks")
    return value.strip()


def set_review_key(provider, value) -> dict:
    path = _credential_path(provider)
    _atomic_bytes(path, b"MCPCR1\0" + _dpapi(_key(value).encode()))
    return {"provider": provider, "stored": True, "storage": "Windows DPAPI, current user"}


def delete_review_key(provider) -> dict:
    path = _credential_path(provider)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"provider": provider, "removed": existed}


def _stored_key(provider):
    path = _credential_path(provider)
    if not path.exists():
        return None
    raw = path.read_bytes()
    if not raw.startswith(b"MCPCR1\0"):
        raise RuntimeError("Stored credential is invalid; set the key again")
    try:
        return _key(_dpapi(raw[7:], decrypt=True).decode())
    except Exception:
        raise RuntimeError("Stored credential cannot be read; set the key again") from None


def resolve_review_config(*, provider=None) -> dict:
    saved = load_review_config()
    family = _provider(provider or os.environ.get("VISUAL_REVIEW_PROVIDER") or saved.get("provider", "gemini"))
    values = dict(DEFAULTS)
    if family != "gemini":
        values.pop("model")
    if saved.get("provider", "gemini") == family:
        values.update(saved)
    values["provider"] = family
    for name, variable in ENVIRONMENT.items():
        if name == "provider" or variable not in os.environ:
            continue
        value = os.environ[variable]
        try:
            if name == "sampling_fps":
                value = float(value)
            elif name in {"max_total_frames", "max_frames_per_video", "frame_width"}:
                value = int(value)
        except ValueError:
            raise ValueError(f"Invalid numeric configuration in {variable}") from None
        values[name] = value
    if family == "gemini" and not os.environ.get("VISUAL_REVIEW_MODEL") and os.environ.get("GEMINI_MODEL"):
        values["model"] = os.environ["GEMINI_MODEL"]
    values = _validate(values)
    key = os.environ.get("VISUAL_REVIEW_API_KEY")
    key_source = "environment" if key else None
    if not key and family == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        key_source = "environment" if key else None
    if not key:
        key = _stored_key(family)
        key_source = "local-encrypted" if key else "none"
    return {**values, "api_key": _key(key) if key else None,
            "api_key_provider": family, "credential_source": key_source}


def public_review_status() -> dict:
    resolved = resolve_review_config()
    return {**{name: value for name, value in resolved.items() if name in DEFAULTS},
            "config_path": str(review_config_path()), "key_configured": bool(resolved.get("api_key")),
            "credential_source": resolved["credential_source"],
            "credential_storage": "Windows DPAPI" if os.name == "nt" else "environment variables"}
