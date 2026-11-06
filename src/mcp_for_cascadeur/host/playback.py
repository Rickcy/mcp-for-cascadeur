"""Owned native playback with bounded Qt observation and measured stop evidence."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..contracts import Tool
from . import common


@dataclass
class PlaybackSession:
    id: str
    view: Any
    scene: Any
    first: int
    last: int
    timeout: float
    started: float
    initial_frame: int
    timer: Any = None
    phase: str = "checking"
    play_started: float | None = None
    stop_requested: float | None = None
    last_change: float = 0
    last_frame: int | None = None
    play_commands: int = 0
    stop_commands: int = 0
    frame_set_calls: int = 0
    stop_verified: bool = False
    samples: list[dict] = field(default_factory=list)
    error: str | None = None


_session: PlaybackSession | None = None
_shutdown_connected = False


def _dispose_timer(session):
    if session.timer is not None:
        try:
            session.timer.stop()
            session.timer.deleteLater()
        except RuntimeError:
            pass  # Qt may already have destroyed a parent-owned timer during shutdown.
        session.timer = None


def _release_scene_references(session):
    session.view = None
    session.scene = None


def release_native_references():
    """Drop native wrappers before Qt destroys scene objects at application exit."""
    if _session is not None:
        _dispose_timer(_session)
        _release_scene_references(_session)
        if _session.phase in ("checking", "playing", "stopping"):
            _session.phase = "closed"


def _start_observer(session, application):
    from PySide6.QtCore import QTimer

    session.timer = QTimer(application)
    session.timer.setInterval(10)
    session.timer.timeout.connect(lambda: _observe(session))
    session.timer.start()


def _public(session):
    moving = {sample["frame"] for sample in session.samples if sample["phase"] == "playing"}
    return {
        "playback_id": session.id,
        "phase": session.phase,
        "first": session.first,
        "last": session.last,
        "samples": list(session.samples),
        "stop_verified": session.stop_verified,
        "native_play_commands": session.play_commands,
        "native_stop_commands": session.stop_commands,
        "scripted_frame_set_calls": session.frame_set_calls,
        "continuous_playback_observed": len(moving) >= 3,
        "full_range_observed": bool(moving) and max(moving) >= session.last - 1,
        "error": session.error,
        "note": "A Qt timer observes native playback; it does not animate by setting frames.",
    }


def require_idle():
    if _session is None:
        return
    unresolved = _session.play_commands > 0 and not _session.stop_verified
    if _session.phase in ("checking", "playing", "stopping") or unresolved:
        raise RuntimeError(
            "Owned playback is active or its stop is unverified; inspect playback_status and stop it first."
        )


def _toggle_stop(session, app, now):
    if session.stop_commands:
        return
    app.get_action_manager().call_action("Timeline.Play")
    session.stop_commands += 1
    session.stop_requested = now
    session.phase = "stopping"


def _observe(session, *, now=None):
    """One observation on Cascadeur's Qt thread; also independently testable."""
    if session.phase not in ("checking", "playing", "stopping"):
        return
    now = time.monotonic() if now is None else now
    try:
        _, app, view, scene = common.current_context()
        if view != session.view or scene != session.scene:
            raise RuntimeError("Active scene changed; playback ownership cannot follow a different scene.")
        frame = int(scene.get_current_frame(False))
        if frame != session.last_frame:
            session.last_change = now
            if len(session.samples) < 512:
                session.samples.append(
                    {"seconds": round(now - session.started, 6), "frame": frame, "phase": session.phase}
                )
        prior = session.last_frame
        session.last_frame = frame
        if session.phase == "checking":
            if frame != session.initial_frame:
                raise RuntimeError(
                    "The playhead moved during the idle check; external playback or edits are active."
                )
            if now - session.started >= 0.3:
                boundary = view.animation_boundary()
                boundary.first_frame, boundary.last_frame = session.first, session.last
                boundary.first_visible_frame, boundary.last_visible_frame = session.first, session.last
                scene.set_current_frame(session.first)
                session.frame_set_calls += 1
                session.last_frame = session.first
                app.get_action_manager().call_action("Timeline.Play")
                session.play_commands += 1
                session.play_started = now
                session.phase = "playing"
        elif session.phase == "playing":
            if (
                frame >= session.last
                or (prior is not None and frame < prior)
                or now - session.play_started >= session.timeout
            ):
                _toggle_stop(session, app, now)
        elif session.phase == "stopping" and now - session.stop_requested >= 0.4:
            session.stop_verified = now - session.last_change >= 0.3
            session.phase = "stopped" if session.stop_verified else "failed"
            if not session.stop_verified:
                session.error = "Native frames continued changing after the stop toggle."
            _dispose_timer(session)
            if session.stop_verified:
                _release_scene_references(session)
    except Exception as error:
        session.phase = "failed"
        session.error = type(error).__name__ + ": " + str(error)
        _dispose_timer(session)
        if not session.play_commands:
            _release_scene_references(session)


def playback_start(first, last, timeout=8):
    global _session, _shutdown_connected
    require_idle()
    if type(first) is not int or type(last) is not int or first < 0 or last <= first:
        raise ValueError("Choose distinct increasing integer playback boundaries.")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0.5 <= timeout <= 20:
        raise ValueError("timeout must be between 0.5 and 20 seconds.")
    _, _, view, scene = common.current_context()
    if last >= scene.data_viewer().get_animation_size():
        raise ValueError("Playback requires existing stored frames.")
    from PySide6.QtCore import QCoreApplication, QThread

    application = QCoreApplication.instance()
    if application is None or QThread.currentThread() != application.thread():
        raise RuntimeError("Playback observation must start on Cascadeur's Qt application thread.")
    if not _shutdown_connected:
        application.aboutToQuit.connect(release_native_references)
        _shutdown_connected = True
    if _session is not None:
        _dispose_timer(_session)
    session = PlaybackSession(
        uuid.uuid4().hex,
        view,
        scene,
        first,
        last,
        float(timeout),
        time.monotonic(),
        int(scene.get_current_frame(False)),
    )
    session.last_frame = session.initial_frame
    session.last_change = session.started
    _session = session
    _start_observer(session, application)
    return _public(session)


def playback_status():
    return {"phase": "idle", "owned_playback": False} if _session is None else _public(_session)


def playback_stop(playback_id=None):
    if _session is None:
        raise ValueError("There is no playback owned by this adapter; external playback is not toggled.")
    if playback_id is not None and playback_id != _session.id:
        raise ValueError("playback_id does not match the owned playback session.")
    if _session.phase in ("stopped", "cancelled", "closed"):
        return _public(_session)
    _, app, view, scene = common.current_context()
    if view != _session.view or scene != _session.scene:
        raise ValueError("Activate the playback-owned scene before requesting its native stop.")
    if _session.phase == "checking":
        _session.phase, _session.stop_verified = "cancelled", True
        _dispose_timer(_session)
        _release_scene_references(_session)
    elif _session.phase == "playing":
        _toggle_stop(_session, app, time.monotonic())
    elif _session.phase == "failed" and _session.play_commands > _session.stop_commands:
        from PySide6.QtCore import QCoreApplication

        _toggle_stop(_session, app, time.monotonic())
        _start_observer(_session, QCoreApplication.instance())
    return _public(_session)


TOOLS = [
    Tool(
        "playback_start",
        "Start bounded native playback after checking a stationary playhead; observe real frame changes and auto-stop. Never scripts per-frame movement.",
        playback_start,
        {
            "first": {"type": "integer", "minimum": 0},
            "last": {"type": "integer", "minimum": 1},
            "timeout": {"type": "number", "minimum": 0.5, "maximum": 20, "default": 8},
        },
        ("first", "last"),
        False,
    ),
    Tool(
        "playback_status",
        "Read owned native playback observations, frame coverage and stop verification.",
        playback_status,
    ),
    Tool(
        "playback_stop",
        "Stop only adapter-owned native playback in its original scene; poll status until the measured stop is verified.",
        playback_stop,
        {"playback_id": {"type": "string", "minLength": 1}},
        (),
        False,
    ),
]
