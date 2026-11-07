from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_for_cascadeur.host import common, inspection, scene


def context(monkeypatch, *, view=None, app=None, native_scene=None):
    view = view or SimpleNamespace(name=lambda: "Test", get_path_name=lambda: "test.casc")
    app = app or SimpleNamespace()
    native_scene = native_scene or SimpleNamespace()
    monkeypatch.setattr(common, "current_context", lambda require_scene=True: (None, app, view, native_scene))
    return app, view, native_scene


def test_selection_preflights_every_identifier_before_mutating(monkeypatch):
    selector = Mock()
    context(monkeypatch, native_scene=SimpleNamespace(selector=lambda: selector))

    def resolve(_, identifier):
        if identifier == "ambiguous":
            raise ValueError("multiple matching objects")
        return identifier

    monkeypatch.setattr(common, "resolve_object", resolve)
    with pytest.raises(ValueError, match="multiple"):
        scene.selection_set(["one", "ambiguous"])
    selector.select.assert_not_called()


@pytest.mark.parametrize("index", [-1, 1, True, "0"])
def test_invalid_scene_indices_never_activate(monkeypatch, index):
    manager = Mock()
    manager.scenes.return_value = [object()]
    context(monkeypatch, app=SimpleNamespace(get_scene_manager=lambda: manager))
    with pytest.raises(ValueError, match="index"):
        scene.scene_activate(index)
    manager.set_current_scene.assert_not_called()


def test_scene_open_reports_native_refusal(monkeypatch, tmp_path):
    source = tmp_path / "sample.casc"
    source.write_bytes(b"fixture")
    common.configure(tmp_path)
    manager = Mock()
    manager.load_scene.return_value = False
    context(monkeypatch, app=SimpleNamespace(get_data_source_manager=lambda: manager))
    with pytest.raises(RuntimeError, match="declined"):
        scene.scene_open(str(source))


def test_save_does_not_claim_old_file_as_new_output(monkeypatch, tmp_path):
    target = tmp_path / "saved.casc"
    target.write_bytes(b"unchanged")
    common.configure(tmp_path)
    view = SimpleNamespace(save=Mock())
    context(monkeypatch, view=view)
    result = scene.scene_save(str(target), overwrite=True)
    assert result["verification"] == "pending"
    assert target.read_bytes() == b"unchanged"
    assert "bytes" not in result


def test_save_requires_explicit_overwrite(monkeypatch, tmp_path):
    target = tmp_path / "saved.casc"
    target.write_bytes(b"original")
    common.configure(tmp_path)
    view = SimpleNamespace(save=Mock())
    context(monkeypatch, view=view)
    with pytest.raises(ValueError, match="overwrite"):
        scene.scene_save(str(target))
    view.save.assert_not_called()


def test_save_recognizes_fresh_nonempty_output(monkeypatch, tmp_path):
    import hashlib

    target = tmp_path / "saved.casc"
    common.configure(tmp_path)
    context(monkeypatch, view=SimpleNamespace(save=lambda path: target.write_bytes(b"new scene")))
    result = scene.scene_save(str(target))
    assert result["verification"] == "fresh_file"
    assert result["bytes"] == 9
    assert result["sha256"] == hashlib.sha256(b"new scene").hexdigest()


@pytest.mark.parametrize("root", ["model", "layers"])
def test_api_inspection_resolves_actual_native_scene_viewers(monkeypatch, root):
    getter = Mock(return_value=SimpleNamespace(native_member=17))
    native_scene = SimpleNamespace(**{root + "_viewer": getter})
    context(monkeypatch, native_scene=native_scene)
    result = inspection.api_inspect(root=root, path="native_member")
    assert result["value"] == 17
    getter.assert_called_once_with()


def test_fbx_export_checks_native_license_before_tool_access(monkeypatch, tmp_path):
    common.configure(tmp_path)
    app = SimpleNamespace(is_export_available=lambda: False, get_tools_manager=Mock())
    context(monkeypatch, app=app)
    with pytest.raises(PermissionError, match="entitlement"):
        scene.fbx_export(str(tmp_path / "clip.fbx"))
    app.get_tools_manager.assert_not_called()


def test_unknown_or_wrong_scene_action_never_dispatches(monkeypatch):
    manager = Mock()
    app = SimpleNamespace(get_action_manager=lambda: manager)
    context(monkeypatch, app=app)
    with pytest.raises(ValueError, match="exact"):
        scene.action_run("Application.Exit", "Test")
    with pytest.raises(ValueError, match="differs"):
        scene.action_run("Scene.Undo", "Different")
    manager.call_action.assert_not_called()


def test_action_response_distinguishes_dispatch_from_completion(monkeypatch):
    manager = Mock()
    app = SimpleNamespace(get_action_manager=lambda: manager)
    context(monkeypatch, app=app, native_scene=SimpleNamespace(get_current_frame=lambda: 12))
    result = scene.action_run("Scene.Undo", "Test")
    manager.call_action.assert_called_once_with("Scene.Undo")
    assert result["dispatched"] is True
    assert result["completed"] is None


@pytest.mark.parametrize("position", [[float("nan"), 0, 1], [0, float("inf"), 1], [True, 0, 1]])
def test_invalid_camera_vectors_fail_before_native_access(monkeypatch, position):
    native = Mock(side_effect=AssertionError("native access is too early"))
    monkeypatch.setattr(common, "current_context", native)
    with pytest.raises(ValueError, match="finite"):
        scene.camera_set(position=position)
    native.assert_not_called()


@pytest.mark.parametrize("path", ["__class__", "app.__dict__", "name()", "a[0]", "a..b"])
def test_api_inspection_rejects_expressions_and_private_paths(path):
    with pytest.raises(ValueError, match="public"):
        inspection._public_path(object(), path)


def test_api_inspection_does_not_invoke_discovered_methods():
    method = Mock()
    api = SimpleNamespace(edit=method, value=7)
    result = inspection._describe(api, 10)
    assert {item["name"] for item in result["members"]} == {"edit", "value"}
    method.assert_not_called()


def test_native_tool_inspection_requires_discovered_name(monkeypatch):
    manager = Mock()
    manager.tools.return_value = [SimpleNamespace(name=lambda: "Available")]
    context(monkeypatch, app=SimpleNamespace(get_tools_manager=lambda: manager))
    with pytest.raises(ValueError, match="exact"):
        inspection.native_tool_inspect("Invented")
    manager.get_tool.assert_not_called()


def test_discovery_supports_native_tools_without_name_method(monkeypatch):
    class RenderToFile:
        pass

    tool = RenderToFile()
    manager = SimpleNamespace(tools=lambda: [tool], get_tool=Mock())
    app = SimpleNamespace(get_tools_manager=lambda: manager)
    context(monkeypatch, app=app)
    result = inspection.native_tools_list()
    assert result["tools"] == [{"name": "RenderToFile", "type": "RenderToFile"}]
    assert scene._native_tool(app, "RenderToFile") is tool
    manager.get_tool.assert_not_called()


class NativeId(str):
    def is_null(self):
        return not self


def test_objects_discovery_preserves_ids_and_exact_type_filter(monkeypatch):
    names = {NativeId("b"): "Hand", NativeId("a"): "Hand", NativeId("c"): "Mesh"}
    behaviours = SimpleNamespace(
        get_behaviour_by_name=lambda object_id, _: object_id,
        get_behaviour_data=lambda object_id, _: object_id,
    )
    data = SimpleNamespace(get_data_value=lambda object_id: "Point" if object_id != "c" else "Mesh")
    model = SimpleNamespace(
        get_objects=lambda: list(names),
        get_object_name=names.__getitem__,
        behaviour_viewer=lambda: behaviours,
        data_viewer=lambda: data,
    )
    context(monkeypatch, native_scene=SimpleNamespace(model_viewer=lambda: model))
    first = inspection.objects_list(query="hand", object_type="Point", limit=1)
    assert first["objects"] == [{"id": "a", "name": "Hand", "type": "Point"}]
    assert first["next_offset"] == 1
    second = inspection.objects_list(query="hand", object_type="Point", offset=1, limit=1)
    assert second["objects"][0]["id"] == "b"
    assert second["next_offset"] is None


def test_object_inspection_returns_channels_and_references_with_bound(monkeypatch):
    behaviours = SimpleNamespace(
        get_behaviour_by_name=lambda *_: NativeId(""),
        get_behaviours=lambda _: [NativeId("transform")],
        get_behaviour_name=lambda _: "Transform",
        get_behaviour_property_names=lambda _: ["global_position", "parent", "omitted"],
        get_property_type=lambda _, name: SimpleNamespace(
            name="DataType" if name == "global_position" else "ObjectType"
        ),
        get_behaviour_data=lambda *_: NativeId("position-data"),
        get_behaviour_data_subtype=lambda *_: "Vector3f",
        get_behaviour_object=lambda *_: NativeId("parent-id"),
    )
    reads = Mock(return_value=[1, 2, 3])
    data = SimpleNamespace(
        get_data_value=reads, get_data=lambda _: SimpleNamespace(mode=SimpleNamespace(name="Animation"))
    )
    model = SimpleNamespace(
        get_object_name=lambda _: "Hand", behaviour_viewer=lambda: behaviours, data_viewer=lambda: data
    )
    native_scene = SimpleNamespace(
        model_viewer=lambda: model,
        get_current_frame=lambda: 3,
        layers_viewer=lambda: SimpleNamespace(frames_count=lambda: 10),
    )
    context(monkeypatch, native_scene=native_scene)
    monkeypatch.setattr(common, "resolve_object", lambda *_: NativeId("hand-id"))
    result = inspection.object_inspect("Hand", limit=2)
    properties = result["behaviors"][0]["properties"]
    assert properties[0]["data_id"] == "position-data"
    assert properties[0]["value"] == [1, 2, 3]
    assert properties[1]["object_id"] == "parent-id"
    assert result["truncated"] is True
    reads.assert_called_once_with("position-data", 3)


def test_hierarchy_reports_native_parent_ids_without_recursing(monkeypatch):
    behaviours = SimpleNamespace(
        get_behaviour_by_name=lambda object_id, _: object_id,
        get_behaviour_object=lambda object_id, _: NativeId("b" if object_id == "a" else "a"),
    )
    model = SimpleNamespace(
        get_objects=lambda: [NativeId("a"), NativeId("b")],
        get_object_name=lambda object_id: object_id,
        behaviour_viewer=lambda: behaviours,
    )
    context(monkeypatch, native_scene=SimpleNamespace(model_viewer=lambda: model))
    result = inspection.object_hierarchy(limit=1)
    assert result["objects"] == [{"id": "a", "name": "a", "parent_id": "b"}]
    assert result["total"] == 2
    assert result["truncated"] is True


@pytest.mark.parametrize("kind", ["DataType", "DataRangeType", "SettingType", "SettingRangeType"])
@pytest.mark.parametrize("mode", ["Static", "Animation"])
def test_property_inspection_uses_strict_native_mode_overloads(kind, mode):
    calls = []

    def read(identifier, *frames):
        assert frames == (() if mode == "Static" else (5,))
        calls.append((identifier, frames))
        return 17

    def descriptor(reference):
        return SimpleNamespace(mode=SimpleNamespace(name=mode))

    data = SimpleNamespace(
        get_data=descriptor, get_setting=descriptor, get_data_value=read, get_setting_value=read
    )
    behaviors = SimpleNamespace(
        get_property_type=lambda *_: SimpleNamespace(name=kind),
        get_behaviour_data=lambda *_: "data",
        get_behaviour_data_range=lambda *_: ["data"],
        get_behaviour_data_subtype=lambda *_: "Int",
        get_behaviour_setting=lambda *_: "setting",
        get_behaviour_settings_range=lambda *_: ["setting"],
        get_behaviour_setting_subtype=lambda *_: "Int",
    )
    result = inspection._property_record(behaviors, data, "behavior", "property", 5)
    assert "unavailable" not in result
    assert len(calls) == 1


def test_fbx_scene_import_can_preserve_source_coordinates(monkeypatch, tmp_path):
    from mcp_for_cascadeur.host import recovery

    path = tmp_path / "source.fbx"
    path.write_bytes(b"fixture")
    common.configure(tmp_path)
    loader = SimpleNamespace(import_scene=Mock())
    _, view, native_scene = context(monkeypatch)
    monkeypatch.setattr(scene, "_native_tool", lambda *_: SimpleNamespace(get_fbx_loader=lambda _: loader))
    monkeypatch.setattr(scene, "scene_info", lambda: {})
    monkeypatch.setattr(recovery, "note_native_attempt", Mock())
    committed = Mock()
    monkeypatch.setattr(recovery, "note_native_commit", committed)
    scene.fbx_import(str(path), mode="scene", set_offset=False)
    loader.import_scene.assert_called_once_with(str(path), -1, False)
    committed.assert_called_once_with(native_scene)


@pytest.mark.parametrize("mode,offset", [("animation", False), ("model", False), ("scene", 0)])
def test_fbx_import_rejects_unsupported_offset_before_native_access(monkeypatch, mode, offset):
    native = Mock(side_effect=AssertionError("Native context must not be accessed"))
    monkeypatch.setattr(common, "current_context", native)
    with pytest.raises(ValueError, match="set_offset"):
        scene.fbx_import("unused.fbx", mode=mode, set_offset=offset)
    native.assert_not_called()
