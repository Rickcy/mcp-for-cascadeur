"""Persistent settings shared by the CLI and MCP process."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .atomic import replace_file


def data_home() -> Path:
    override = os.environ.get("MCP_FOR_CASCADEUR_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "mcp-for-cascadeur"


def config_path() -> Path:
    return data_home() / "config.json"


def validate_endpoint(endpoint: str) -> str:
    try:
        parts = urlsplit(endpoint)
        port = parts.port
    except (ValueError, TypeError) as exc:
        raise ValueError("The Cascadeur endpoint must be a loopback HTTP URL.") from exc
    if (
        parts.scheme != "http"
        or parts.hostname not in {"127.0.0.1", "localhost"}
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
        or port is None
        or not 1 <= port <= 65535
    ):
        raise ValueError("The Cascadeur endpoint must be a loopback HTTP URL with a port.")
    return endpoint.rstrip("/")


@dataclass(frozen=True)
class Config:
    endpoint: str = "http://127.0.0.1:8765"
    workspace: Path = field(default_factory=lambda: Path.home() / "Documents" / "Cascadeur MCP")
    allow_python: bool = False
    cascadeur_home: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", validate_endpoint(self.endpoint))
        object.__setattr__(self, "workspace", Path(self.workspace).expanduser().resolve())
        if type(self.allow_python) is not bool:
            raise ValueError("allow_python must be a boolean.")
        if self.cascadeur_home is not None:
            object.__setattr__(self, "cascadeur_home", Path(self.cascadeur_home).expanduser().resolve())

    @property
    def port(self) -> int:
        return urlsplit(self.endpoint).port or 8765

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "workspace": str(self.workspace),
            "allow_python": self.allow_python,
            "cascadeur_home": str(self.cascadeur_home) if self.cascadeur_home else None,
        }


def load_config(path: Path | None = None) -> Config:
    source = Path(path) if path is not None else config_path()
    if not source.exists():
        return Config()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The settings file must contain a JSON object.")
    allowed = {"endpoint", "workspace", "allow_python", "cascadeur_home"}
    return Config(**{key: value for key, value in data.items() if key in allowed})


def save_config(config: Config, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    replace_file(temporary, target)
    return target
