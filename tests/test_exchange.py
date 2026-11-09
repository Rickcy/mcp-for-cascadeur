import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_for_cascadeur.host import common, exchange


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    common.configure(tmp_path)
    native = SimpleNamespace(
        entitled=True,
        value=1,
        rigs=["rig"],
        objects=["joint", "mesh", "point"],
        exports=[],
        content=b"; FBX 7.4.0 project file\n" + b"x" * 3000,
        after_export=None,
    )
    boundary = SimpleNamespace(first_frame=0, last_frame=2)
    view = SimpleNamespace(animation_boundary=lambda: boundary, path=str(tmp_path / "original.casc"))

    def save(path):
        Path(path).write_bytes(b"native .casc checkpoint" * 100)
        view.path = path

    view.save = save
    behaviours = SimpleNamespace(
        get_behaviours=lambda kind: native.rigs if kind == "RigInfo" else ["mesh-behaviour"],
        get_behaviour_owner=lambda _: "mesh",
    )
    model = SimpleNamespace(
        get_objects=lambda: native.objects,
        get_object_type_name=lambda identifier: "Joint" if "joint" in identifier else "Point",
    )
    scene = SimpleNamespace(
        behaviour_viewer=lambda: behaviours,
        model_viewer=lambda: model,
        data_viewer=lambda: SimpleNamespace(get_animation_size=lambda: 3),
    )
    app = SimpleNamespace(is_export_available=lambda: native.entitled)
    settings_type = type("FbxSettings", (), {})
    csc = SimpleNamespace(
        fbx=SimpleNamespace(
            FbxSettings=settings_type,
            FbxSettingsMode=SimpleNamespace(Ascii="Ascii"),
            FbxSettingsAxis=SimpleNamespace(Y="Y"),
        )
    )
    monkeypatch.setattr(common, "current_context", lambda: (csc, app, view, scene))
    rig = SimpleNamespace(id="rig-id", joints={"joint"}, owned={"joint", "mesh", "point"})
    monkeypatch.setattr(exchange.characters, "resolve_character", lambda *args: rig)
    monkeypatch.setattr(exchange.state, "capture", lambda _: {"model": native.value})

    class Loader:
        def set_settings(self, settings):
            native.settings = settings

        def export_all_objects(self, path):
            native.exports.append(Path(path))
            Path(path).write_bytes(native.content)
            if native.after_export:
                native.after_export(Path(path))

    monkeypatch.setattr(
        exchange.scene_tools, "_native_tool", lambda *args: SimpleNamespace(get_fbx_loader=lambda _: Loader())
    )
    return native, view, boundary, tmp_path


def test_native_scope_entitlement_and_full_range_are_read_only_preflight(exporter):
    native, _, boundary, workspace = exporter
    assert exchange.fbx_export_status()["ready"]
    native.entitled = False
    status = exchange.fbx_export_status()
    assert not status["ready"] and not status["entitled"]
    native.entitled = True
    boundary.first_frame = 1
    assert exchange.fbx_export_status()["blockers"][0]["code"] == "EXPORT_RANGE"
    assert not list(workspace.iterdir())


@pytest.mark.parametrize("blocker", ["second_character", "unrelated_joint"])
def test_whole_scene_export_rejects_ambiguous_character_or_foreign_joints(exporter, blocker):
    native, _, _, workspace = exporter
    if blocker == "second_character":
        native.rigs.append("other")
    else:
        native.objects.append("foreign_joint")
    with pytest.raises(exchange.ExportError) as caught:
        exchange.fbx_export_verified(str(workspace / "walk.fbx"), str(workspace / "copy.casc"))
    assert caught.value.code == "EXPORT_SCOPE"
    assert not native.exports
    assert not list(workspace.iterdir())


def test_verified_export_uses_requested_basename_retains_source_and_preserves_model(exporter):
    native, view, _, workspace = exporter
    destination, source = workspace / "Walk Cycle.fbx", workspace / "source.casc"
    receipt = exchange.fbx_export_verified(str(destination), str(source))
    assert receipt["exported"] and receipt["model_unchanged"]
    assert receipt["start_frame"] == 0 and receipt["end_frame"] == 2
    assert receipt["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert receipt["publication"] == "exclusive_new_file"
    assert native.exports[0].name == destination.name
    assert native.exports[0].parent.name.startswith(".mcp-for-cascadeur-export-")
    assert not native.exports[0].parent.exists()
    assert source.is_file() and view.path == str(source)
    assert vars(native.settings) == exchange.SETTINGS


def test_expected_hash_replaces_only_the_version_the_caller_observed(exporter):
    _, _, _, workspace = exporter
    destination = workspace / "clip.fbx"
    destination.write_bytes(b"previous output")
    expected = hashlib.sha256(destination.read_bytes()).hexdigest()
    receipt = exchange.fbx_export_verified(
        str(destination), str(workspace / "source.casc"), expected_sha256=expected
    )
    assert receipt["publication"] == "atomic_replace_after_expected_sha256_recheck"
    assert destination.read_bytes().startswith(b"; FBX ")


@pytest.mark.parametrize("expected", [None, "a" * 64])
def test_existing_destination_conflicts_before_native_save_or_export(exporter, expected):
    native, _, _, workspace = exporter
    destination = workspace / "clip.fbx"
    destination.write_bytes(b"preserve")
    with pytest.raises(exchange.ExportError) as caught:
        exchange.fbx_export_verified(
            str(destination), str(workspace / "source.casc"), expected_sha256=expected
        )
    assert caught.value.code == "EXPORT_CONFLICT"
    assert destination.read_bytes() == b"preserve"
    assert not native.exports and not (workspace / "source.casc").exists()


@pytest.mark.parametrize("replace_existing", [False, True])
def test_racing_writer_is_preserved_and_our_staging_file_is_cleaned(exporter, replace_existing):
    native, _, _, workspace = exporter
    destination = workspace / "clip.fbx"
    expected = None
    if replace_existing:
        destination.write_bytes(b"observed file")
        expected = hashlib.sha256(destination.read_bytes()).hexdigest()
    native.after_export = lambda _: destination.write_bytes(b"racing writer wins")
    with pytest.raises(exchange.ExportError) as caught:
        exchange.fbx_export_verified(
            str(destination), str(workspace / "source.casc"), expected_sha256=expected
        )
    assert caught.value.code == "EXPORT_CONFLICT"
    assert destination.read_bytes() == b"racing writer wins"
    assert (workspace / "source.casc").is_file()
    assert not native.exports[0].parent.exists()


def test_source_copy_is_never_overwritten(exporter):
    native, _, _, workspace = exporter
    source = workspace / "source.casc"
    source.write_bytes(b"keep original")
    with pytest.raises(exchange.ExportError) as caught:
        exchange.fbx_export_verified(str(workspace / "clip.fbx"), str(source))
    assert caught.value.code == "EXPORT_SOURCE_CONFLICT"
    assert source.read_bytes() == b"keep original" and not native.exports


@pytest.mark.parametrize("invalid", ["header", "size", "stale", "state"])
def test_invalid_output_or_model_changes_prevent_publication_but_keep_source(exporter, invalid):
    native, _, _, workspace = exporter
    if invalid == "header":
        native.content = b"not an fbx" * 500
    elif invalid == "size":
        native.content = b"; FBX short"
    elif invalid == "stale":
        native.after_export = lambda path: os.utime(path, (0, 0))
    else:
        native.after_export = lambda _: setattr(native, "value", 2)
    with pytest.raises(exchange.ExportError):
        exchange.fbx_export_verified(str(workspace / "clip.fbx"), str(workspace / "source.casc"))
    assert not (workspace / "clip.fbx").exists()
    assert (workspace / "source.casc").is_file()
    assert not native.exports[0].parent.exists()


def test_workspace_escape_is_rejected_before_output(exporter):
    native, _, _, workspace = exporter
    with pytest.raises(ValueError, match="inside"):
        exchange.fbx_export_verified(str(workspace / ".." / "escape.fbx"), str(workspace / "source.casc"))
    assert not native.exports
