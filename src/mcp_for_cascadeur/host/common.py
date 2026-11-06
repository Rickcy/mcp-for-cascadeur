"""Small native API helpers. This module also imports outside Cascadeur."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

_settings = {"workspace": Path.home() / "Documents" / "Cascadeur MCP", "allow_python": False}


def configure(workspace: str | Path, allow_python: bool = False) -> None:
    _settings.update(workspace=Path(workspace).expanduser().resolve(), allow_python=bool(allow_python))


def get_settings() -> dict:
    return dict(_settings)


def current_context(require_scene: bool = True):
    import csc

    application = csc.app.get_application()
    if application is None:
        raise RuntimeError("Cascadeur has not initialized its application")
    view = application.get_scene_manager().current_scene()
    if require_scene and view is None:
        raise RuntimeError("Open or create a scene in Cascadeur first")
    return csc, application, view, view.domain_scene() if view is not None else None


def json_value(value: Any, _depth: int = 0) -> Any:
    if _depth > 24:
        raise ValueError("Native result exceeds the nesting limit")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Native result contains a nonfinite number")
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_value(item, _depth + 1) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist(), _depth + 1)
    if hasattr(value, "to_quaternion"):
        quaternion = value.to_quaternion()
        return json_value(
            {"quaternion_wxyz": [float(getattr(quaternion, axis)()) for axis in "wxyz"]}, _depth + 1
        )
    if all(callable(getattr(value, axis, None)) for axis in "wxyz"):
        return json_value([float(getattr(value, axis)()) for axis in "wxyz"], _depth + 1)
    return str(value)


def resolve_object(scene, identifier: str):
    if type(identifier) is not str or not identifier.strip():
        raise ValueError("An object ID or exact name is required")
    model = scene.model_viewer()
    matches = [item for item in model.get_objects() if str(item) == identifier]
    if not matches:
        matches = list(model.get_objects(identifier))
    if len(matches) != 1:
        raise ValueError(f"Object {identifier!r} matched {len(matches)} objects; use a unique ID")
    return matches[0]


def workspace_path(
    path: str | Path, *, must_exist: bool = False, overwrite: bool = False, suffixes=()
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("File paths must be absolute")
    resolved = candidate.resolve()
    workspace = _settings["workspace"].resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"File must be inside the configured workspace: {workspace}")
    if suffixes and resolved.suffix.lower() not in suffixes:
        raise ValueError(f"Expected a file with extension: {', '.join(suffixes)}")
    if must_exist:
        if not resolved.is_file():
            raise ValueError("Input file does not exist")
    elif resolved.exists():
        if not resolved.is_file():
            raise ValueError("Destination must be a regular file, not a directory")
        if not overwrite:
            raise ValueError("Destination exists; request overwrite explicitly")
    elif not resolved.parent.is_dir():
        raise ValueError("Destination directory does not exist")
    return resolved


class NativeEditError(RuntimeError):
    execution_state = "unknown"


def modify(scene, label: str, callback):
    """Check the native transaction result, including exceptions swallowed by C++."""
    captured = []
    results = []

    def apply(model, _update_editor, updater):
        try:
            results.append(callback(model, updater))
            from . import recovery

            recovery.observe_callback(scene)
        except Exception as error:
            captured.append(error)
            raise

    from . import recovery

    recovery.note_native_attempt(scene)
    accepted = scene.modify_update(label, apply)
    if accepted:
        from . import recovery

        recovery.note_native_commit(scene)
    if captured:
        raise NativeEditError(
            f"{label} failed: {captured[0]}; inspect the scene before retrying"
        ) from captured[0]
    if not accepted:
        raise NativeEditError(f"Cascadeur rejected {label}; inspect the scene before retrying")
    if not results:
        raise NativeEditError(f"Cascadeur did not execute {label}'s callback")
    return results[0]
