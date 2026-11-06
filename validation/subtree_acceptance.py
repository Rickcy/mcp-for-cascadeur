"""Native FK recovery acceptance in a new disposable scene, executed explicitly.

Uses only the installed vendor's Joint construction/hierarchy helpers to build
the fixture. Production subtree tools perform the edits and scoped restoration.
The script never closes or replaces an existing scene tab.
"""

from __future__ import annotations

import json
import uuid

from native_advanced import CLIENT, CONFIG, probe

CHECKED_MODIFY = """
def fixture_modify(label, callback):
    failures = []
    def checked(model, update, updater):
        try:
            callback(model, update, updater)
        except BaseException as error:
            failures.append(error)
            raise
    accepted = scene.modify_update(label, checked)
    if failures:
        raise failures[0]
    if not accepted:
        raise RuntimeError('Native fixture modification declined: ' + label)
"""


def main(full_spin=False):
    records = []
    token = uuid.uuid4().hex[:10]

    def run(name, **arguments):
        receipt = CLIENT.call(name, arguments)
        records.append({"tool": name, "arguments": arguments, **receipt})
        print(name, json.dumps(receipt)[-5000:], flush=True)
        if receipt.get("ok") is False:
            raise RuntimeError(receipt)
        return receipt["result"]

    run("scene_new")
    facts = probe(
        CHECKED_MODIFY
        + """
import importlib
import numpy as np
import pycsc
from mcp_for_cascadeur.host import animation
topology = importlib.import_module('add_function.topology_add')
hierarchy = importlib.import_module('common.hierarchy')
layers_operation = importlib.import_module('common.layers_operation')
created = {}
def build(model, update, updater):
    domain = pycsc.wrap(scene)
    domain.set_modifiers(model, update, updater)
    for name in ('MCP_FK_Root', 'MCP_FK_Child', 'MCP_Unrelated_Joint'):
        created[name] = topology.add_joint(domain, name).object_id()
        layers_operation.move_to_new_layer(model, scene, [created[name]], name)
    updater.generate_update()
    hierarchy.set_parent_by_id(domain, [created['MCP_FK_Child']], created['MCP_FK_Root'])
    updater.generate_update()
    editor = model.layers_editor()
    for layer in scene.layers_viewer().all_layer_ids():
        editor.set_section(csc.layers.layer.Section(), 5, layer)
    model.fit_animation_size_by_layers()
    updater.generate_update()
    for layer in scene.layers_viewer().all_layer_ids():
        for frame in list(scene.layers_viewer().layer(layer).sections):
            if frame not in (0, 5):
                editor.unset_section(frame, layer)
        for frame in (0, 5):
            def flags(section):
                section.interval.interpolation = csc.layers.layer.Interpolation.LINEAR
                section.interval.common.ik_fk = csc.layers.layer.IkFk.FK
                section.key.common.ik_fk = csc.layers.layer.IkFk.FK
            editor.change_section(frame, layer, flags)
    updater.generate_update()
    # Materialize the new FK section flags before writing authored local data.
    interpolator = updater.get_interpolator()
    interpolator.reload()
    interpolator.interpolate()
    data = model.data_editor()
    for frame in range(scene.data_viewer().get_animation_size()):
        changed = set()
        for index, (name, object_id) in enumerate(created.items()):
            reference = animation._channel(scene, object_id, 'local_position')
            template = scene.data_viewer().get_data_value(reference, frame)
            value = np.asarray([frame * (index + 1), index * 10, 0], dtype=np.float32).reshape(template.shape)
            data.set_data_value(reference, frame, value)
            changed.add(reference)
        updater.run_update(changed, frame)
fixture_modify('Build disposable independent FK acceptance scene', build)
settings = {}
for name, object_id in created.items():
    data_viewer = scene.data_viewer()
    settings[name] = [{'name': data_viewer.get_setting_name(identifier),
                      'mode': data_viewer.get_setting(identifier).mode.name,
                      'value': data_viewer.get_setting_value(identifier)}
                     for identifier in data_viewer.get_all_settings_id(object_id)
                     if data_viewer.get_setting(identifier).mode.name == 'Static']
result = {'objects': {name: str(value) for name, value in created.items()},
          'frames': scene.data_viewer().get_animation_size(), 'native_settings': settings,
          'track_modes': {str(identifier): {str(frame): {'interval': section.interval.common.ik_fk.name,
                                                       'key': section.key.common.ik_fk.name}
                                           for frame, section in scene.layers_viewer().layer(identifier).sections.items()}
                          for identifier in scene.layers_viewer().all_layer_ids()}}
"""
    )
    root = facts["objects"]["MCP_FK_Root"]
    unrelated = facts["objects"]["MCP_Unrelated_Joint"]
    assert facts["frames"] == 6, facts
    assert all(
        mode["interval"] == mode["key"] == "FK"
        for track in facts["track_modes"].values()
        for mode in track.values()
    ), facts
    run("timeline_set_range", first=0, last=5)
    if full_spin:
        spin = probe(
            CHECKED_MODIFY
            + f"""
import math
from mcp_for_cascadeur.host import animation, common, state
object_id = common.resolve_object(scene, {root!r})
reference = animation._channel(scene, object_id, 'local_rotation')
stages = []
def author_spin(model, update, updater):
    updater.generate_update()
    interpolator = updater.get_interpolator()
    interpolator.reload()
    interpolator.interpolate()
    node = update.get_node_by_id(reference)
    # The vendor Transform prototype enables the Euler filter. Disable it for
    # this explicit full-turn fixture, and require the actual stored readback.
    node.set_apply_euler_filter(False)
    for frame in range(6):
        requested = [0.0, 0.0, 2 * math.pi * frame / 5]
        node.set_value(csc.math.Rotation.from_euler(*requested), frame)
        after_set = state.encode(scene.data_viewer().get_data_value(reference, frame))['euler_radians']
        updater.run_update({{reference}}, frame)
        after_update = state.encode(scene.data_viewer().get_data_value(reference, frame))['euler_radians']
        stages.append({{'frame': frame, 'requested': requested, 'after_set': after_set,
                       'after_update': after_update, 'apply_euler_filter': node.get_apply_euler_filter()}})
fixture_modify('Author disposable six-frame FK full turn', author_spin)
result = {{'stages': stages, 'local_euler_radians': [state.encode(scene.data_viewer().get_data_value(reference, frame))['euler_radians'] for frame in range(6)]}}
"""
        )
        import math

        diagnostic = CONFIG.workspace / "validation" / f"fk-spin-fixture-{token}.json"
        diagnostic.write_text(json.dumps(spin, indent=2), encoding="utf-8")
        assert abs(spin["local_euler_radians"][-1][2] - 2 * math.pi) <= 1e-4, spin
        records.append({"fixture": "native-six-frame-full-turn", "result": spin})
    target = str(CONFIG.workspace / "validation" / f"fk-subtree-{token}.casc")
    run("scene_save", path=target)
    original = run("subtree_snapshot", root=root)
    if full_spin:
        changed_spin = probe(
            CHECKED_MODIFY
            + f"""
from mcp_for_cascadeur.host import animation, common, state
object_id = common.resolve_object(scene, {root!r})
reference = animation._channel(scene, object_id, 'local_rotation')
def perturb_spin(model, update, updater):
    node = update.get_node_by_id(reference)
    for frame in range(6):
        node.set_value(csc.math.Rotation.from_euler(0.0, 0.0, 0.3 * frame / 5), frame)
        updater.run_update({{reference}}, frame)
fixture_modify('Perturb the disposable FK turn after its snapshot', perturb_spin)
result = {{'local_euler_radians': [state.encode(scene.data_viewer().get_data_value(reference, frame))['euler_radians'] for frame in range(6)]}}
"""
        )
        assert abs(changed_spin["local_euler_radians"][-1][2] - 0.3) <= 1e-4, changed_spin
        records.append({"fixture": "perturbed-six-frame-Euler-after-snapshot", "result": changed_spin})
    run(
        "subtree_pose_apply",
        root=root,
        frame=0,
        space="local",
        poses=[{"object": root, "position": [30, 40, 50]}],
    )
    # This independent edit occurs AFTER the snapshot. Scoped restore must keep it.
    run(
        "subtree_pose_apply",
        root=unrelated,
        frame=0,
        space="local",
        poses=[{"object": unrelated, "position": [70, 80, 90]}],
    )
    protected = run("subtree_pose_read", root=unrelated, frame=0)
    restored = run("subtree_restore", snapshot_id=original["snapshot_id"])
    observed = run("subtree_pose_read", root=unrelated, frame=0)
    assert observed["joints"] == protected["joints"]
    assert restored["frames_verified"] == facts["frames"]
    assert restored["unrelated_scene_edits_preserved"] is True
    if full_spin:
        assert restored["local_euler_fidelity"] is True
        assert restored["max_local_euler_error_radians"] <= 1e-4
    destination = CONFIG.workspace / "validation" / f"subtree-native-{token}.json"
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(str(destination), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-spin",
        action="store_true",
        help="Author and verify one complete Euler turn across six native frames.",
    )
    main(full_spin=parser.parse_args().full_spin)
