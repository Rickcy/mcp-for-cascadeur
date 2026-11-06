from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import profiles
from mcp_for_cascadeur.host.common import configure


@pytest.fixture
def rig(tmp_path, monkeypatch):
    configure(tmp_path)
    state = {"signature": "topology-one", "writes": []}
    view = SimpleNamespace(get_path_name=lambda: "example.casc")
    monkeypatch.setattr(profiles, "current_context", lambda: (None, None, view, "native-scene"))
    monkeypatch.setattr(
        profiles,
        "resolve_object",
        lambda scene, identifier: {"hand": "uuid-hand", "finger": "uuid-finger"}.get(identifier, identifier),
    )
    monkeypatch.setattr(profiles.animation, "_signature", lambda scene: state["signature"])
    monkeypatch.setattr(profiles.animation, "pose_apply", lambda *args: state["writes"].append(args))
    return state


def test_profile_checks_topology_on_every_use(rig):
    profiles.rig_profile_create("character", {"right_hand": ["hand"]})
    assert profiles.rig_profile_read("character")["roles"] == {"right_hand": ["uuid-hand"]}
    rig["signature"] = "topology-changed"
    with pytest.raises(ValueError, match="different scene or topology"):
        profiles.rig_profile_read("character")


def test_overlapping_roles_are_rejected_before_native_write(rig):
    profiles.rig_profile_create("character", {"hand": ["hand"], "arm": ["hand"]})
    with pytest.raises(ValueError, match="overlap"):
        profiles.rig_pose_apply(
            "character", {"hand": [{"position": [1, 2, 3]}], "arm": [{"position": [1, 2, 3]}]}
        )
    assert not rig["writes"]


def test_profile_slot_order_is_preserved(rig):
    profiles.rig_profile_create("character", {"hand": ["hand", "finger"]})
    profiles.rig_pose_apply("character", {"hand": [{"position": [1, 2, 3]}, {"rotation_wxyz": [1, 0, 0, 0]}]})
    assert [entry["object"] for entry in rig["writes"][0][0]] == ["uuid-hand", "uuid-finger"]


@pytest.mark.parametrize("name", ["../outside", "two/slashes", "", "name with spaces"])
def test_profile_names_cannot_escape_workspace(rig, name):
    with pytest.raises(ValueError):
        profiles.rig_profile_create(name, {"hand": ["hand"]})
