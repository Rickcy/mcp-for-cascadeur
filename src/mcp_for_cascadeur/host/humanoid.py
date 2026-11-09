"""Map anatomical roles to actual native rig joints without renaming the scene."""

from __future__ import annotations

import re

ROLES = {
    "pelvis": "Таз", "stomach": "Поясница", "chest": "Грудь", "neck": "Шея", "head": "Голова",
}
ALIASES = {
    "pelvis": ("pelvis", "hips"),
    "stomach": ("stomach", "spine_02", "spine1", "spine"),
    "chest": ("chest", "spine_04", "spine_03", "spine3", "spine2"),
    "neck": ("neck", "neck_01", "neck1"), "head": ("head",),
}
for side, label, prefix in (("l", "Л", "left"), ("r", "П", "right")):
    for role, title, alternatives in (
        ("clavicle", "Ключица", ("clavicle", "shoulder")),
        ("arm", "Плечо", ("arm", "upperarm")),
        ("forearm", "Предплечье", ("forearm", "lowerarm")),
        ("hand", "Кисть", ("hand",)), ("thigh", "Бедро", ("thigh", "upleg", "leg")),
        ("calf", "Голень", ("calf", "shin", "leg")), ("foot", "Стопа", ("foot",)),
        ("toe", "Носок", ("toe", "ball", "toebase")),
    ):
        key = role + "_" + side
        ROLES[key] = title + " · " + label
        ALIASES[key] = tuple(value + "_" + side for value in alternatives) + tuple(
            prefix + value for value in alternatives)


def _normalized(name):
    return re.sub(r"[^a-z0-9]", "", name.split(":")[-1].casefold())


def discover_mapping(scene, character_id=None, mapping=None):
    """Expose unambiguous suggestions and accept explicit native joint IDs/names.

    Only joints with real Point-bearing RigAdditionalInfo links are body roles.
    Names suggest a profile; native relationships, not names, establish controls.
    """
    from . import characters

    rig = characters.resolve_character(scene, character_id)
    model = scene.model_viewer()
    joints = {str(obj): model.get_object_name(obj) for obj in rig.joints}
    controlled = {link["joint"] for link in rig.native_links if link.get("main_point") in rig.controls}
    overrides = {} if mapping is None else mapping
    if not isinstance(overrides, dict) or set(overrides) - ROLES.keys():
        raise ValueError("Skeleton mapping contains unknown anatomical roles")
    result, issues = {}, []
    for role, aliases in ALIASES.items():
        explicit = overrides.get(role)
        if explicit:
            matches = [key for key, name in joints.items() if explicit in (key, name)]
        else:
            matches = []
            for alias in aliases:
                matches = [key for key, name in joints.items()
                           if key in controlled and _normalized(name) == _normalized(alias)]
                if matches:
                    break
        if len(matches) == 1 and matches[0] in controlled:
            result[role] = matches[0]
        elif explicit:
            issues.append("Выберите одну кость с контроллерами для роли: " + ROLES[role])
    duplicates = {key for key in result.values() if list(result.values()).count(key) > 1}
    for role in [role for role, key in result.items() if key in duplicates]:
        del result[role]
        issues.append("Одна кость назначена нескольким ролям: " + ROLES[role])
    missing = [role for role in ROLES if role not in result]
    return {"character_id": rig.id, "name": rig.name, "fingerprint": rig.fingerprint,
            "profile": "explicit" if overrides else "native-humanoid",
            "mapping": result, "missing": missing, "ready": not missing and not issues, "issues": issues,
            "joints": [{"id": key, "name": name} for key, name in sorted(joints.items(), key=lambda item: item[1])],
            "roles": [{"id": key, "label": label} for key, label in ROLES.items()]}


def point_aliases(scene, rig, mapping):
    """Bind the solver's anatomical Point vocabulary through native links."""

    links = {link["joint"]: link for link in rig.native_links if link.get("main_point") in rig.controls}
    result = {}

    def assign(alias, identifier):
        if identifier not in rig.controls:
            raise ValueError("Rig has no native controller for " + alias)
        if alias in result and result[alias] != identifier:
            raise ValueError("Ambiguous native controller for " + alias)
        result[alias] = identifier

    def point(role, token):
        side = "_" + role[-1] if role.endswith(("_l", "_r")) else ""
        return role.removesuffix(side) + "_" + token + side if side else role + "_" + token

    for role, identifier in mapping.items():
        link = links[identifier]
        assign(point(role, "MainPoint"), link["main_point"])
        if role in {"pelvis", "stomach", "chest", "neck", "head"} or role.startswith(
                ("clavicle_", "forearm_", "hand_", "calf_", "toe_")):
            assign(point(role, "AdditionalPoint"), link["additional_point"])
        if role == "head" or role.startswith(("hand_", "toe_")):
            assign(point(role, "DirectionPoint"), link["direction_point"])
        if role.startswith("foot_"):
            viewer = scene.behaviour_viewer()
            box = next(obj for obj in rig.owned if str(obj) == link["box_id"])
            information = viewer.get_behaviour_by_name(box, "RigAdditionalInfo")
            extra = [str(viewer.get_behaviour_owner(ref)) for ref in
                     viewer.get_behaviour_reference_range(information, "additional_points")]
            if len(extra) != 1:
                raise ValueError("A humanoid foot requires one native heel controller: " + role)
            assign(point(role, "Self0Point"), extra[0])
    if len(set(result.values())) != len(result) or set(result.values()) != set(rig.controls):
        raise ValueError("Mapped humanoid body must cover every native Point exactly once")
    return result
