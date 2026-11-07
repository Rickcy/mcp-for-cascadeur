"""User-scoped cloud credentials, separate from settings and native runtime.

Windows persistence uses DPAPI CurrentUser. Other platforms use environment
variables; no plaintext credential fallback is written.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path

from .settings import data_home

ENVIRONMENT = {"uthana": "UTHANA_API_KEY", "gemini": "GEMINI_API_KEY"}
_MAX_SECRET_BYTES = 16384


class CredentialError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.execution_state = "not_started"


def _provider(provider: str) -> str:
    if provider not in ENVIRONMENT:
        raise CredentialError("UNKNOWN_CREDENTIAL", "Supported credential providers are uthana and gemini.")
    return provider


def _path(provider: str) -> Path:
    return data_home() / "credentials" / f"{_provider(provider)}.dpapi.json"


def _validated_secret(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CredentialError("EMPTY_CREDENTIAL", "The API key must be a nonempty string.")
    if value != value.strip() or any(char in value for char in "\r\n\x00"):
        raise CredentialError(
            "INVALID_CREDENTIAL", "The API key must not contain padding or control characters."
        )
    if len(value.encode("utf-8")) > _MAX_SECRET_BYTES:
        raise CredentialError("INVALID_CREDENTIAL", "The API key exceeds the supported length.")
    return value


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data)
    return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _dpapi(data: bytes, provider: str, *, decrypt: bool) -> bytes:
    if sys.platform != "win32":
        raise CredentialError(
            "SECURE_STORE_UNAVAILABLE",
            f"Encrypted local storage is Windows-only; set {ENVIRONMENT[provider]} in the server environment.",
        )
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    source, source_buffer = _blob(data)
    entropy, entropy_buffer = _blob(f"mcp-for-cascadeur:{provider}:v1".encode())
    output = _Blob()
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.restype = wintypes.BOOL
    function.argtypes = [
        ctypes.POINTER(_Blob),
        ctypes.c_void_p if decrypt else wintypes.LPCWSTR,
        ctypes.POINTER(_Blob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_Blob),
    ]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    # CRYPTPROTECT_UI_FORBIDDEN prevents an unattended stdio server opening UI.
    description = None if decrypt else "mcp-for-cascadeur cloud credential"
    succeeded = function(
        ctypes.byref(source), description, ctypes.byref(entropy), None, None, 1, ctypes.byref(output)
    )
    # The buffers remain alive until the native call completes.
    _ = source_buffer, entropy_buffer
    if not succeeded:
        raise CredentialError("CREDENTIAL_STORE_FAILED", "Windows could not access the encrypted credential.")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)


def store_secret(provider: str, value: str) -> dict:
    """Persist a secret for this Windows account; return metadata only."""
    provider = _provider(provider)
    ciphertext = _dpapi(_validated_secret(value).encode("utf-8"), provider, decrypt=False)
    target = _path(provider)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "version": 1,
        "provider": provider,
        "protection": "windows-dpapi-current-user",
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{provider}-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"provider": provider, "stored": True, "protection": record["protection"]}


def load_secret(provider: str = "uthana") -> str:
    """Read an environment credential first, then this user's encrypted store."""
    provider = _provider(provider)
    value = os.environ.get(ENVIRONMENT[provider])
    if value is not None:
        return _validated_secret(value)
    target = _path(provider)
    if not target.exists():
        raise CredentialError(
            "CREDENTIAL_REQUIRED",
            f"Configure {provider} with 'credentials set {provider}' or {ENVIRONMENT[provider]}.",
        )
    try:
        if target.stat().st_size > 65536:
            raise ValueError("oversized record")
        record = json.loads(target.read_text(encoding="utf-8"))
        if (
            record.get("version") != 1
            or record.get("provider") != provider
            or record.get("protection") != "windows-dpapi-current-user"
        ):
            raise ValueError("unsupported record")
        ciphertext = base64.b64decode(record["ciphertext"], validate=True)
        return _validated_secret(_dpapi(ciphertext, provider, decrypt=True).decode("utf-8"))
    except CredentialError:
        raise
    except Exception:
        raise CredentialError(
            "INVALID_CREDENTIAL_STORE", "The encrypted credential record is unreadable."
        ) from None


def credential_status(provider: str) -> dict:
    """Report credential presence without decrypting or exposing a value."""
    provider = _provider(provider)
    environment = ENVIRONMENT[provider]
    inherited = environment in os.environ
    stored = _path(provider).is_file()
    return {
        "provider": provider,
        "configured": inherited or stored,
        "source": "environment" if inherited else "windows-dpapi-current-user" if stored else None,
        "environment_variable": environment,
        "stored": stored,
    }


def delete_secret(provider: str) -> dict:
    """Delete only this project's encrypted credential; leave inherited env alone."""
    provider = _provider(provider)
    target = _path(provider)
    existed = target.is_file()
    target.unlink(missing_ok=True)
    return {"provider": provider, "deleted": existed, "environment_unchanged": True}
