"""Install our host code and client settings without depending on the uv cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from urllib.parse import quote, urlsplit

from .settings import Config, data_home

HOOK_NAME = "mcp_for_cascadeur.py"
CODEX_SERVER_NAME = "mcp_for_cascadeur"
DEFAULT_SOURCE = "git+https://github.com/Rickcy/mcp-for-cascadeur"


def default_package_source() -> str:
    """Reuse a reproducible remote install source, never an ephemeral tool path."""
    try:
        record = metadata.distribution("mcp-for-cascadeur").read_text("direct_url.json")
        direct = json.loads(record) if record else {}
    except (metadata.PackageNotFoundError, OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return DEFAULT_SOURCE
    if not isinstance(direct, dict):
        return DEFAULT_SOURCE
    directory = direct.get("dir_info", {})
    if isinstance(directory, dict) and directory.get("editable"):
        return DEFAULT_SOURCE
    url = direct.get("url")
    if not isinstance(url, str):
        return DEFAULT_SOURCE
    remote = url.removeprefix("git+")
    try:
        parsed = urlsplit(remote)
    except ValueError:
        return DEFAULT_SOURCE
    if parsed.scheme not in {"http", "https", "ssh", "git"} or not parsed.hostname:
        return DEFAULT_SOURCE
    vcs = direct.get("vcs_info", {})
    if isinstance(vcs, dict) and vcs.get("vcs") == "git":
        commit = vcs.get("commit_id")
        if (
            not isinstance(commit, str)
            or len(commit) not in {40, 64}
            or any(character not in "0123456789abcdefABCDEF" for character in commit)
        ):
            return DEFAULT_SOURCE
        source = f"git+{remote}@{commit}"
        subdirectory = direct.get("subdirectory")
        if isinstance(subdirectory, str) and subdirectory:
            source += "#subdirectory=" + quote(subdirectory, safe="/")
        return source
    if parsed.scheme in {"http", "https"} and parsed.path.lower().endswith(".whl"):
        return url
    return DEFAULT_SOURCE


class InstallationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def scripts_directory(home: Path) -> Path:
    home = Path(home).expanduser().resolve()
    for relative in (
        "resources/scripts/python",
        "Contents/Resources/scripts/python",
        "Contents/MacOS/resources/scripts/python",
    ):
        candidate = home / relative
        if (candidate / "scripts/mcp/script_server/server.py").is_file():
            return candidate
    raise InstallationError(
        "CASCADEUR_LAYOUT_UNSUPPORTED", f"The installed script runner was not found in {home}."
    )


def detect_cascadeur_home(path: str | Path | None = None) -> Path:
    if path is not None:
        selected = Path(path).expanduser().resolve()
        scripts_directory(selected)
        return selected
    candidates = []
    if sys.platform == "win32":
        for variable, suffix in (
            ("LOCALAPPDATA", "Cascadeur"),
            ("ProgramFiles", "Cascadeur"),
            ("ProgramFiles", "Nekki Limited/Cascadeur"),
        ):
            if os.environ.get(variable):
                candidates.append(Path(os.environ[variable]) / suffix)
    elif sys.platform == "darwin":
        candidates.extend((Path("/Applications/Cascadeur.app"), Path.home() / "Applications/Cascadeur.app"))
    else:
        candidates.extend(
            (Path("/opt/cascadeur"), Path("/opt/Cascadeur"), Path.home() / ".local/share/Cascadeur")
        )
    for candidate in candidates:
        try:
            scripts_directory(candidate)
        except InstallationError:
            continue
        return candidate.resolve()
    raise InstallationError(
        "CASCADEUR_NOT_FOUND",
        "Cascadeur was not found. Pass --cascadeur-home with its installation directory.",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def install_runtime(config: Config) -> Path:
    """Copy only this project's stdlib host code, never Cascadeur's files."""
    source = Path(__file__).resolve().parent
    python_root = data_home() / "runtime" / "python"
    package_root = python_root / "mcp_for_cascadeur"
    files = [source / "__init__.py", source / "contracts.py", *sorted((source / "host").rglob("*.py"))]
    if not files[0].is_file() or not files[1].is_file() or not (source / "host" / "__init__.py").is_file():
        raise InstallationError(
            "PACKAGE_INCOMPLETE", "The installed package does not contain the host runtime."
        )
    for current in files:
        destination = package_root / current.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, destination)
    start_script = data_home() / "start_bridge.py"
    _write_text(
        start_script,
        (
            '"""Start mcp-for-cascadeur from Cascadeur\'s Python console."""\n'
            "import sys\n"
            f"_runtime = {str(python_root)!r}\n"
            "if _runtime not in sys.path:\n"
            "    sys.path.insert(0, _runtime)\n"
            "from mcp_for_cascadeur.host import start_bridge\n"
            f"start_bridge(workspace={str(config.workspace)!r}, allow_python={config.allow_python!r}, port={config.port!r})\n"
        ),
    )
    return start_script


def _state_path() -> Path:
    return data_home() / "installation.json"


def _load_state() -> dict:
    if not _state_path().exists():
        return {}
    value = json.loads(_state_path().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InstallationError(
            "INSTALLATION_STATE_INVALID", "The installation record must contain a JSON object."
        )
    return value


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def startup_hook_content(bridge_script: Path) -> str:
    return (
        "# Managed by mcp-for-cascadeur.\n"
        "def _start_bridge():\n"
        f"    with open({str(bridge_script)!r}, encoding='utf-8') as _source:\n"
        f"        exec(compile(_source.read(), {str(bridge_script)!r}, 'exec'), {{'__name__': '__main__'}})\n"
        "\n"
        "def run():\n"
        "    from events import scene_idle_manager\n"
        "    scene_idle_manager.execute_when_idle(_start_bridge)\n"
    )


def install_startup(config: Config, bridge_script: Path) -> dict:
    if config.cascadeur_home is None:
        raise InstallationError("CASCADEUR_NOT_FOUND", "The Cascadeur installation directory is required.")
    target = scripts_directory(config.cascadeur_home) / "events/application_started" / HOOK_NAME
    if not target.parent.is_dir():
        raise InstallationError(
            "CASCADEUR_LAYOUT_UNSUPPORTED", "The application_started scripts directory does not exist."
        )
    expected = startup_hook_content(bridge_script).encode("utf-8")
    state = _load_state()
    if target.exists():
        current = target.read_bytes()
        owned = state.get("startup_hook", {})
        if current != expected and (
            owned.get("path") != str(target) or owned.get("sha256") != _digest(current)
        ):
            raise InstallationError(
                "STARTUP_HOOK_CONFLICT", f"An unrelated or modified startup hook exists: {target}"
            )
    _write_text(target, expected.decode("utf-8"))
    state["startup_hook"] = {"path": str(target), "sha256": _digest(expected)}
    _write_text(_state_path(), json.dumps(state, indent=2) + "\n")
    return {"installed": True, "path": str(target)}


def uninstall_startup(home: str | Path | None = None) -> dict:
    state = _load_state()
    owned = state.get("startup_hook")
    if not owned:
        return {"removed": False, "reason": "not_installed"}
    target = Path(owned["path"])
    if home is not None:
        requested = scripts_directory(Path(home)) / "events/application_started" / HOOK_NAME
        if requested != target:
            raise InstallationError(
                "STARTUP_HOOK_CONFLICT",
                "The recorded startup hook belongs to another Cascadeur installation.",
            )
    if target.name != HOOK_NAME or target.parent.name != "application_started":
        raise InstallationError("INSTALLATION_STATE_INVALID", "The recorded startup hook path is invalid.")
    if target.exists():
        if _digest(target.read_bytes()) != owned.get("sha256"):
            raise InstallationError(
                "STARTUP_HOOK_CONFLICT", f"The startup hook was modified and has been preserved: {target}"
            )
        target.unlink()
    state.pop("startup_hook", None)
    _write_text(_state_path(), json.dumps(state, indent=2) + "\n")
    return {"removed": True, "path": str(target)}


def codex_config_path() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"


def _is_our_codex_entry(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    command = entry.get("command", "")
    if not isinstance(command, str) or Path(command).name.lower() not in {"uvx", "uvx.exe"}:
        return False
    arguments = entry.get("args", [])
    return not isinstance(arguments, str) and "mcp-for-cascadeur" in arguments


def configure_codex(
    config_path: Path | None = None, *, remove: bool = False, source: str | None = DEFAULT_SOURCE
) -> dict:
    import tomlkit

    target = Path(config_path) if config_path is not None else codex_config_path()
    original = target.read_text(encoding="utf-8") if target.exists() else ""
    document = tomlkit.parse(original)
    if "mcp_servers" not in document:
        if remove:
            return {"changed": False, "path": str(target)}
        document["mcp_servers"] = tomlkit.table()
    servers = document["mcp_servers"]
    if not isinstance(servers, dict):
        raise InstallationError("CODEX_CONFIG_INVALID", "mcp_servers must be a TOML table.")
    if remove:
        if CODEX_SERVER_NAME not in servers:
            return {"changed": False, "path": str(target)}
        existing = servers[CODEX_SERVER_NAME]
        if not _is_our_codex_entry(existing):
            raise InstallationError(
                "CODEX_ENTRY_CONFLICT", "The Codex MCP entry uses another command and has been preserved."
            )
        del servers[CODEX_SERVER_NAME]
    else:
        entry = tomlkit.table()
        entry["command"] = shutil.which("uvx") or "uvx"
        entry["args"] = (["--from", source] if source else []) + ["mcp-for-cascadeur"]
        entry["startup_timeout_sec"] = 60
        entry["tool_timeout_sec"] = 45
        configured_home = os.environ.get("MCP_FOR_CASCADEUR_HOME")
        if configured_home:
            entry["env"] = {"MCP_FOR_CASCADEUR_HOME": str(Path(configured_home).expanduser().resolve())}
        if CODEX_SERVER_NAME in servers:
            existing = servers[CODEX_SERVER_NAME]
            if not _is_our_codex_entry(existing):
                raise InstallationError(
                    "CODEX_ENTRY_CONFLICT",
                    f"The Codex MCP entry {CODEX_SERVER_NAME!r} is already used by another command.",
                )
            # Preserve explicit per-server permissions and other user settings.
            for key, value in entry.items():
                if key == "env" and key in existing:
                    existing[key].update(value)
                else:
                    existing[key] = value
        else:
            servers[CODEX_SERVER_NAME] = entry
    content = tomlkit.dumps(document)
    if content == original:
        return {"changed": False, "path": str(target)}
    backup = None
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = target.with_name(f"{target.name}.mcp-for-cascadeur.{stamp}.{uuid.uuid4().hex[:8]}.bak")
        shutil.copy2(target, backup)
    _write_text(target, content)
    return {"changed": True, "path": str(target), "backup": str(backup) if backup else None}
