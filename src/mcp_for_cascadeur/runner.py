"""Client for the script runner shipped with Cascadeur 2026.2.3 and newer."""

from __future__ import annotations

import ast
import json
import socket
import threading
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from . import __version__
from .settings import validate_endpoint

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 4 * 1024 * 1024


class RunnerError(RuntimeError):
    def __init__(self, code: str, message: str, execution_state: str = "not_started", details=None):
        super().__init__(message)
        self.code = code
        self.execution_state = execution_state
        self.details = details

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "execution_state": self.execution_state}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        # This transport only talks to the configured local runner.
        return None


class RunnerClient:
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8765",
        workspace: str | Path | None = None,
        *,
        allow_python: bool = False,
        timeout: float = 35.0,
    ):
        self.endpoint = validate_endpoint(endpoint)
        self.workspace = str(Path(workspace).expanduser().resolve()) if workspace is not None else None
        self.allow_python = bool(allow_python)
        if not 0 < timeout <= 120:
            raise ValueError("Timeout must be between zero and 120 seconds")
        self.timeout = timeout
        self._lock = threading.Lock()
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, route: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        if body is not None and len(body) > MAX_REQUEST_BYTES:
            raise RunnerError("REQUEST_TOO_LARGE", "Request exceeds 4 MiB")
        request = Request(self.endpoint + route, data=body, headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RunnerError("RESPONSE_TOO_LARGE", "Response exceeds 8 MiB", "unknown")
            decoded = json.loads(raw)
            if type(decoded) is not dict:
                raise ValueError("Expected a JSON object")
            return decoded
        except HTTPError as error:
            raise RunnerError("HTTP_ERROR", f"Cascadeur returned HTTP {error.code}", "unknown") from error
        except (TimeoutError, socket.timeout) as error:
            raise RunnerError(
                "TIMEOUT", "Cascadeur timed out; inspect the scene before retrying a write", "unknown"
            ) from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise RunnerError(
                    "TIMEOUT", "Cascadeur timed out; inspect before retrying", "unknown"
                ) from error
            if payload is not None and not isinstance(
                error.reason, (ConnectionRefusedError, socket.gaierror)
            ):
                raise RunnerError(
                    "CONNECTION_LOST",
                    "The connection failed after submission; the operation may "
                    "have executed. Inspect the scene before retrying.",
                    "unknown",
                ) from error
            raise RunnerError(
                "NOT_CONNECTED",
                "Cascadeur is not reachable. Open or create a scene, then run the generated "
                "start_bridge.py, or enable startup during setup.",
            ) from error
        except (OSError, HTTPException) as error:
            raise RunnerError(
                "CONNECTION_LOST",
                "The connection closed before a result was received; inspect the scene before retrying.",
                "unknown" if payload is not None else "not_started",
            ) from error
        except (ValueError, UnicodeError) as error:
            raise RunnerError(
                "INVALID_RESPONSE", "Cascadeur returned an invalid JSON response", "unknown"
            ) from error

    def health(self) -> dict:
        status = self._request("/health")
        if status.get("ok") is not True or "version" not in status:
            raise RunnerError("UNEXPECTED_SERVICE", "The endpoint is not a Cascadeur script runner")
        return status

    def call(self, name: str, arguments: dict) -> dict:
        from .host import registry

        tools = registry()
        if name not in tools:
            raise RunnerError("UNKNOWN_TOOL", f"Unknown tool: {name}")
        try:
            tools[name].validate(arguments)
        except ValueError as error:
            raise RunnerError("INVALID_ARGUMENTS", str(error)) from error
        if name == "python_execute" and not self.allow_python:
            raise RunnerError("PYTHON_DISABLED", "Run setup --allow-python to enable Python execution")
        expression = (
            "__import__('json').dumps(__import__('mcp_for_cascadeur.host', fromlist=['dispatch'])"
            f".dispatch({name!r}, {arguments!r}, workspace={self.workspace!r}, "
            f"allow_python={self.allow_python!r}, version={__version__!r}), allow_nan=False)"
        )
        with self._lock:
            envelope = self._request("/run", {"code": expression})
        if envelope.get("ok") is not True:
            raise RunnerError(
                "NATIVE_SCRIPT_FAILED",
                str(envelope.get("error", "Script execution failed")),
                "unknown",
                envelope.get("messages"),
            )
        try:
            payload = json.loads(ast.literal_eval(envelope["value"]))
            if type(payload) is not dict or type(payload.get("ok")) is not bool:
                raise ValueError("Missing adapter response envelope")
        except (KeyError, TypeError, ValueError, SyntaxError, RecursionError) as error:
            raise RunnerError(
                "INVALID_RESULT", "The native adapter returned an invalid result", "unknown"
            ) from error
        messages = envelope.get("messages", [])
        if not payload["ok"]:
            error = payload.get("error", {})
            raise RunnerError(
                error.get("code", "NATIVE_ERROR"),
                error.get("message", "Native operation failed"),
                error.get("execution_state", "unknown"),
                payload,
            )
        errors = [message for message in messages if message.get("level") == "Error"]
        if errors:
            raise RunnerError(
                "NATIVE_EVENT_ERROR",
                "; ".join(str(item.get("text", "")) for item in errors)[-8000:],
                "unknown",
                payload,
            )
        payload["messages"] = messages[-50:]
        return payload
