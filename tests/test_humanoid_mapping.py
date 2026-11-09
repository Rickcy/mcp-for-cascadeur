"""Anatomical mapping uses native ownership and permits explicitly renamed rigs."""

from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import characters, humanoid


def rig_fixture(monkeypatch, *, unreal=False, renamed=False):
    names = {role: role for role in humanoid.ROLES}
    if unreal:
        names.update(stomach="spine_02", chest="spine_04", neck="neck_01")
        for side in ("l", "r"):
            names.update({"arm_" + side: "upperarm_" + side,
                          "forearm_" + side: "lowerarm_" + side, "toe_" + side: "ball_" + side})
    if renamed:
        names = {role: "custom_bone_" + str(index) for index, role in enumerate(names)}
    joints = {"id-" + role: name for role, name in names.items()}
    controls = {"point-" + role: "point-" + role for role in names}
    links = [{"joint": "id-" + role, "main_point": "point-" + role} for role in names]
    rig = SimpleNamespace(id="rig-id", name="Selected skeleton", joints=set(joints), controls=controls,
                          native_links=links, fingerprint="topology")
    model = SimpleNamespace(get_object_name=lambda identifier: joints[identifier])
    scene = SimpleNamespace(model_viewer=lambda: model)
    monkeypatch.setattr(characters, "resolve_character", lambda *_args, **_kwargs: rig)
    return scene, rig, joints


@pytest.mark.parametrize("unreal", [False, True])
def test_cascadeur_and_unreal_skeletons_resolve_same_roles_without_renaming(monkeypatch, unreal):
    scene, rig, names = rig_fixture(monkeypatch, unreal=unreal)
    before = dict(names)
    result = humanoid.discover_mapping(scene, rig.id)
    assert result["ready"] and not result["missing"]
    assert result["mapping"] == {role: "id-" + role for role in humanoid.ROLES}
    assert names == before


def test_unknown_custom_names_require_explicit_mapping(monkeypatch):
    scene, rig, _ = rig_fixture(monkeypatch, renamed=True)
    automatic = humanoid.discover_mapping(scene, rig.id)
    assert not automatic["ready"] and set(automatic["missing"]) == set(humanoid.ROLES)
    explicit = {role: "id-" + role for role in humanoid.ROLES}
    mapped = humanoid.discover_mapping(scene, rig.id, explicit)
    assert mapped["ready"] and mapped["mapping"] == explicit


def test_one_native_joint_cannot_supply_two_roles(monkeypatch):
    scene, rig, _ = rig_fixture(monkeypatch)
    result = humanoid.discover_mapping(scene, rig.id, {"arm_l": "id-arm_r"})
    assert not result["ready"] and {"arm_l", "arm_r"} <= set(result["missing"])


def test_uncontrolled_joint_cannot_be_forced_into_mapping(monkeypatch):
    scene, rig, names = rig_fixture(monkeypatch)
    names["extra"] = "special_joint"
    rig.joints.add("extra")
    result = humanoid.discover_mapping(scene, rig.id, {"head": "extra"})
    assert not result["ready"] and "head" in result["missing"]


def test_unknown_role_is_rejected(monkeypatch):
    scene, _, _ = rig_fixture(monkeypatch)
    with pytest.raises(ValueError, match="unknown anatomical"):
        humanoid.discover_mapping(scene, mapping={"tail": "extra"})
