"""Exercise authored curve metadata, baked protection and stationary contacts.

Run against the already connected disposable Cascy scene. Every fixture and
receipt is written to the configured workspace; the repository holds no assets.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid

from native_advanced import CLIENT, CONFIG, probe

from mcp_for_cascadeur.runner import RunnerError

SOURCE_KEYS = [0, 4, 8, 12, 16, 20, 24, 25]
TARGET_KEYS = [0, 3, 6, 10, 14, 19, 24, 25]
CACHE = "_mcp_curve_acceptance_" + uuid.uuid4().hex


def main():
    directory = CONFIG.workspace / "validation" / ("curve-metadata-" + str(time.time_ns()))
    directory.mkdir(parents=True)
    records = []

    def record(kind, **value):
        records.append({"case": kind, **value})
        (directory / "receipts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    def run(name, **arguments):
        started = time.monotonic()
        try:
            receipt = CLIENT.call(name, arguments)
        except RunnerError as error:
            record(name, arguments=arguments, error=error.as_dict(), seconds=time.monotonic() - started)
            raise
        record(name, arguments=arguments, receipt=receipt, seconds=time.monotonic() - started)
        print(name, round(time.monotonic() - started, 2), flush=True)
        return receipt["result"]

    def diagnostic(name, code):
        value = probe(code)
        record(name, evidence=value)
        print(name, json.dumps(value), flush=True)
        return value

    # A native save captures current unsaved poses before creating independent cases.
    source = directory / "source.casc"
    run("scene_save", path=str(source))
    deadline = time.monotonic() + 10
    while not source.is_file() or not source.stat().st_size:
        if time.monotonic() >= deadline:
            raise RuntimeError("The native source fixture was not saved")
        time.sleep(0.05)
    curve_fixture, contact_fixture = directory / "curves.casc", directory / "contacts.casc"
    shutil.copy2(source, curve_fixture)
    shutil.copy2(source, contact_fixture)

    run("scene_open", path=str(curve_fixture))
    setup = diagnostic(
        "configure_nondefault_metadata",
        f"""
import builtins, math
from mcp_for_cascadeur.host import characters, common, state
rig = characters.resolve_character(scene, None, require_semantics=True, require_hands=True)
hand = rig.controls[rig.roles['left_hand']['center']]
hand_track = scene.layers_viewer().layer_id_by_obj_id_or_null(hand)
fixed_track = rig.fingers['right']['index1'].track_id
fixed_layer = scene.layers_viewer().layer(fixed_track)
if {{str(obj) for obj in fixed_layer.obj_ids}} & set(rig.controls):
    raise ValueError('The acceptance fixture needs an independent right-finger track')
if not scene.layers_viewer().layer(hand_track).is_key(12):
    raise ValueError('Frame 12 must be an existing hand key')
def configure(model, updater):
    editor = model.layers_editor()
    def authored(section):
        section.interval.interpolation = csc.layers.layer.Interpolation.BEZIER
        section.key.left_weight_velocity = .31
        section.key.right_weight_velocity = .67
        section.key.label = 1
        section.interval.description_for_ai_interpolation = 'Acceptance: retain the left palm curve'
    editor.change_section(12, hand_track, authored)
    for frame in list(fixed_layer.sections):
        if frame < scene.data_viewer().get_animation_size() - 1:
            def baked(section):
                section.interval.interpolation = csc.layers.layer.Interpolation.FIXED
            editor.change_section(frame, fixed_track, baked)
    interpolator = updater.get_interpolator()
    interpolator.reload()
    interpolator.interpolate()
common.modify(scene, 'Prepare authored metadata acceptance fixture', configure)
snapshot = state.capture(scene)
fixed_objects = set(snapshot['tracks'][str(fixed_track)]['objects'])
protected = {{key: value for key, value in snapshot['channels'].items() if value['owner'] in fixed_objects}}
setattr(builtins, {CACHE!r}, {{'character': rig.id, 'snapshot': snapshot,
    'hand_track': str(hand_track), 'fixed_track': str(fixed_track), 'protected': protected}})
metadata = snapshot['tracks'][str(hand_track)]['sections']['12']
assert math.isclose(metadata['key']['left_weight_velocity'], .31, rel_tol=1e-6)
assert math.isclose(metadata['key']['right_weight_velocity'], .67, rel_tol=1e-6)
assert metadata['key']['label'] == 1
assert metadata['ai_description'] == 'Acceptance: retain the left palm curve'
result = {{'character': rig.id, 'metadata': metadata, 'fixed_track': str(fixed_track),
          'protected_channels': len(protected), 'all_track_metadata_sha256': state.digest(snapshot['tracks'])}}
""",
    )
    character = setup["character"]
    checkpoint = run("checkpoint_create", label="Authored Bezier metadata and unselected baked fingers")
    edited = run(
        "character_offset_preserving_curves",
        character_id=character,
        frames=[12],
        offsets={"left_hand": [0, 0.05, 0]},
        position_tolerance=0.01,
    )
    assert edited["curve_preservation"]["verified"]
    proof = diagnostic(
        "metadata_and_baked_protection",
        f"""
import builtins
from mcp_for_cascadeur.host import state
baseline = getattr(builtins, {CACHE!r})
after = state.capture(scene)
metadata_exact = baseline['snapshot']['tracks'] == after['tracks']
mismatches = [key for key, value in baseline['protected'].items()
              if not state.channel_equal(value, after['channels'][key])]
result = {{'metadata_exact': metadata_exact, 'protected_channel_mismatches': mismatches,
          'protected_channels_checked': len(baseline['protected']),
          'before_track_sha256': state.digest(baseline['snapshot']['tracks']),
          'after_track_sha256': state.digest(after['tracks'])}}
""",
    )
    assert proof["metadata_exact"] and not proof["protected_channel_mismatches"]
    run("checkpoint_restore", checkpoint_id=checkpoint["checkpoint_id"])
    assert run("checkpoint_status", checkpoint_id=checkpoint["checkpoint_id"])["comparison"]["equivalent"]
    run("checkpoint_release", checkpoint_id=checkpoint["checkpoint_id"])

    diagnostic(
        "install_unsupported_clamped_fixture",
        f"""
import builtins
from mcp_for_cascadeur.host import common, state
baseline = getattr(builtins, {CACHE!r})
baseline['before_clamped'] = state.capture(scene)
track = next(item for item in scene.layers_viewer().all_layer_ids() if str(item) == baseline['hand_track'])
def configure(model, updater):
    def unsupported(section):
        section.interval.interpolation = csc.layers.layer.Interpolation.CLAMPED_BEZIER
    model.layers_editor().change_section(12, track, unsupported)
common.modify(scene, 'Prepare unsupported interpolation rejection fixture', configure)
baseline['clamped_model_sha256'] = state.digest(state.capture(scene))
result = {{'model_sha256': baseline['clamped_model_sha256']}}
""",
    )
    try:
        CLIENT.call(
            "character_offset_preserving_curves",
            {"character_id": character, "frames": [12], "offsets": {"left_hand": [0, 0.05, 0]}},
        )
    except RunnerError as error:
        record("clamped_preflight_rejection", error=error.as_dict())
        assert "CLAMPED" in str(error).upper(), error.as_dict()
    else:
        raise AssertionError("A CLAMPED_BEZIER curve edit bypassed native preflight")
    rejection = diagnostic(
        "clamped_rejection_identity",
        f"""
import builtins
from mcp_for_cascadeur.host import state
baseline = getattr(builtins, {CACHE!r})
observed = state.digest(state.capture(scene))
result = {{'model_unchanged': observed == baseline['clamped_model_sha256'], 'model_sha256': observed}}
""",
    )
    assert rejection["model_unchanged"]
    cleaned = diagnostic(
        "reset_clamped_fixture_metadata",
        f"""
import builtins
from mcp_for_cascadeur.host import common, state
baseline = getattr(builtins, {CACHE!r})
track = next(item for item in scene.layers_viewer().all_layer_ids() if str(item) == baseline['hand_track'])
original = baseline['before_clamped']['tracks'][str(track)]['sections']['12']
def configure(model, updater):
    model.layers_editor().change_section(12, track, lambda section: state.apply_section_metadata(section, original))
common.modify(scene, 'Reset interpolation rejection fixture metadata', configure)
result = state.compare(baseline['before_clamped'], state.capture(scene))
""",
    )
    assert cleaned["equivalent"]
    run("scene_save", path=str(curve_fixture), overwrite=True)

    run("scene_open", path=str(contact_fixture))
    facts = diagnostic(
        "stationary_contact_fixture_layout",
        """
from mcp_for_cascadeur.host import characters, curves
rig = characters.resolve_character(scene, None, require_semantics=True, require_hands=True)
result = {'character': rig.id, 'stored_frames': scene.data_viewer().get_animation_size(),
          'tracks': {str(item): list(layer.key_frame_indices())
                     for item, layer in curves.character_tracks(scene, rig).items()}}
""",
    )
    assert facts["stored_frames"] == 26, facts
    unsynchronized = [key for key, frames in facts["tracks"].items() if frames != SOURCE_KEYS]
    if unsynchronized:
        run("tracks_set_keys", tracks=unsynchronized, frames=SOURCE_KEYS)
    reference = run("semantic_pose_read", character=facts["character"], frame=25, include_joint_state=False)
    run(
        "semantic_pose_apply_sequence",
        character=facts["character"],
        keyframes=[{"frame": frame, "pose": reference["pose"]} for frame in SOURCE_KEYS[:-1]],
    )
    stationary = diagnostic(
        "stationary_contact_preflight",
        """
from mcp_for_cascadeur.host import characters, character_edit
rig = characters.resolve_character(scene, None, require_semantics=True, require_hands=True)
source = character_edit._positions(scene, rig, [0,4,8,12,16,20,24,25])
anchors = character_edit._contact_targets(rig, source)
result = {'stationary_support_targets': len(anchors), 'target_ids': sorted(anchors)}
""",
    )
    assert stationary["stationary_support_targets"] == 8
    contact_checkpoint = run("checkpoint_create", label="Stationary support targets before contact retiming")
    contacts = run(
        "character_retime",
        character_id=facts["character"],
        stabilize_contacts=True,
        keys=[{"source": source, "target": target} for source, target in zip(SOURCE_KEYS, TARGET_KEYS)],
    )
    assert contacts["contacts"]["targets_pinned"] == 8
    assert contacts["contacts"]["baked_tracks"] == 2
    assert contacts["contacts"]["max_error"] <= 0.03
    assert contacts["rig_geometry"]["max_connection_anchor_error"] <= 0.1
    run("checkpoint_restore", checkpoint_id=contact_checkpoint["checkpoint_id"])
    assert run("checkpoint_status", checkpoint_id=contact_checkpoint["checkpoint_id"])["comparison"][
        "equivalent"
    ]
    run("checkpoint_release", checkpoint_id=contact_checkpoint["checkpoint_id"])
    probe(f"import builtins\ndelattr(builtins, {CACHE!r})\nresult = True")
    record("acceptance_complete", native_verified=True, fixture_directory=str(directory))
    print("ARTIFACT", directory, flush=True)


if __name__ == "__main__":
    main()
