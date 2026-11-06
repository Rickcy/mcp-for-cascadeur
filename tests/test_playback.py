from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_for_cascadeur.host import common, playback


@pytest.fixture
def observed(monkeypatch):
    action = Mock()
    boundary = SimpleNamespace(first_frame=0, last_frame=20, first_visible_frame=0, last_visible_frame=20)
    frame = {"value": 0}
    scene = SimpleNamespace(
        get_current_frame=lambda *_: frame["value"], set_current_frame=lambda value: frame.update(value=value)
    )
    view = SimpleNamespace(animation_boundary=lambda: boundary)
    app = SimpleNamespace(get_action_manager=lambda: SimpleNamespace(call_action=action))
    monkeypatch.setattr(common, "current_context", lambda: (None, app, view, scene))
    session = playback.PlaybackSession("owned", view, scene, 0, 10, 8, 100, 0)
    session.last_frame = 0
    session.last_change = 100
    session.timer = Mock()
    monkeypatch.setattr(playback, "_session", session)
    return session, frame, action, boundary


def test_idle_check_does_not_start_until_playhead_stationary(observed):
    session, _, action, boundary = observed
    playback._observe(session, now=100.2)
    action.assert_not_called()
    playback._observe(session, now=100.31)
    action.assert_called_once_with("Timeline.Play")
    assert session.phase == "playing"
    assert session.frame_set_calls == 1
    assert boundary.last_frame == boundary.last_visible_frame == 10


def test_external_playhead_movement_never_toggles_native_play(observed):
    session, frame, action, _ = observed
    frame["value"] = 1
    playback._observe(session, now=100.1)
    assert session.phase == "failed"
    assert "external" in session.error
    action.assert_not_called()


def test_observer_stops_at_boundary_and_measures_stability(observed):
    session, frame, action, _ = observed
    playback._observe(session, now=100.31)
    for index, time in ((2, 100.4), (5, 100.5), (10, 100.6)):
        frame["value"] = index
        playback._observe(session, now=time)
    assert session.phase == "stopping"
    assert action.call_count == 2
    playback._observe(session, now=101.1)
    result = playback.playback_status()
    assert result["stop_verified"] is True
    assert result["continuous_playback_observed"] is True
    assert result["full_range_observed"] is True
    assert result["scripted_frame_set_calls"] == 1
    assert session.view is None
    assert session.scene is None


def test_unverified_failed_stop_blocks_edits(observed):
    session, _, _, _ = observed
    session.phase, session.play_commands, session.stop_commands = "failed", 1, 1
    with pytest.raises(RuntimeError, match="unverified"):
        playback.require_idle()


def test_stop_rejects_wrong_owner_without_native_action(observed):
    _, _, action, _ = observed
    with pytest.raises(ValueError, match="does not match"):
        playback.playback_stop("different")
    action.assert_not_called()


def test_stop_during_check_cancels_without_toggling(observed):
    session, _, action, _ = observed
    result = playback.playback_stop("owned")
    assert result["phase"] == "cancelled"
    assert session.stop_verified is True
    assert session.view is None
    assert session.scene is None
    action.assert_not_called()


def test_scene_change_never_toggles_different_scenes_playback(observed, monkeypatch):
    session, _, action, _ = observed
    session.phase, session.play_commands = "playing", 1
    monkeypatch.setattr(common, "current_context", lambda: (None, None, object(), object()))
    playback._observe(session, now=100.5)
    assert session.phase == "failed"
    action.assert_not_called()
    with pytest.raises(RuntimeError, match="unverified"):
        playback.require_idle()


def test_no_owned_playback_cannot_stop_external_session(monkeypatch):
    monkeypatch.setattr(playback, "_session", None)
    with pytest.raises(ValueError, match="external playback"):
        playback.playback_stop()


def test_stopped_session_stop_is_idempotent_after_native_wrappers_released(observed, monkeypatch):
    session, _, action, _ = observed
    session.phase, session.stop_verified = "stopped", True
    playback._release_scene_references(session)
    monkeypatch.setattr(
        common, "current_context", Mock(side_effect=AssertionError("No native wrappers needed"))
    )
    result = playback.playback_stop("owned")
    assert result["phase"] == "stopped"
    assert result["stop_verified"] is True
    action.assert_not_called()


def test_shutdown_releases_native_wrappers_without_toggling_playback(observed):
    session, _, action, _ = observed
    session.phase, session.play_commands = "playing", 1
    timer = session.timer
    playback.release_native_references()
    assert session.phase == "closed"
    assert session.view is session.scene is session.timer is None
    assert session.stop_verified is False
    timer.stop.assert_called_once()
    timer.deleteLater.assert_called_once()
    action.assert_not_called()
