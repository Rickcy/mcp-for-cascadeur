from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mcp_for_cascadeur.host import common, data_channels


class Identifier(str):
    def is_null(self):
        return not self


class Mode(Enum):
    Static = 0
    Animation = 1
    Unsupported = 2


@pytest.fixture
def native(monkeypatch):
    values = {"unbound": [1.0, 2.0, 3.0, 4.0], "visible": True, "count": 7, "text": "old"}
    modes = {key: Mode.Static if key != "unbound" else Mode.Animation for key in values}
    names = {"unbound": "Unbound Value", "visible": "Visible", "count": "Count", "text": "Text"}
    owners = {key: Identifier("object") for key in values}
    keys = {0, 3}
    track = SimpleNamespace(is_locked=False, is_key=keys.__contains__)

    def read(reference, *frames):
        if modes[reference] == Mode.Animation:
            assert len(frames) == 1, "Animation reads require the native frame overload"
            return values[reference][frames[0]]
        assert not frames, "Static reads must never pass a frame to the native API"
        return values[reference]

    def write(reference, *arguments):
        if len(arguments) == 1:
            values[reference] = arguments[0]
        else:
            values[reference][arguments[0]] = arguments[1]

    data = SimpleNamespace(
        get_all_data_id=lambda object_id: list(map(Identifier, values)),
        get_data=lambda reference: SimpleNamespace(mode=modes[reference], object_id=owners[reference]),
        get_data_name=names.__getitem__,
        get_data_value=read,
        get_animation_size=lambda: 4,
    )
    layers = SimpleNamespace(
        layer_id_by_obj_id_or_null=lambda object_id: Identifier("track"), layer=lambda reference: track
    )
    layer_editor = SimpleNamespace(
        set_fixed_interpolation_or_key_if_need=Mock(side_effect=lambda layer, frame, value: keys.add(frame)),
        normalize_sections=Mock(),
    )
    model = SimpleNamespace(
        data_editor=lambda: SimpleNamespace(set_data_value=write), layers_editor=lambda: layer_editor
    )
    updater = SimpleNamespace(run_update=Mock())
    scene = SimpleNamespace(
        data_viewer=lambda: data, layers_viewer=lambda: layers, get_current_frame=lambda: 0
    )
    csc = SimpleNamespace(model=SimpleNamespace(DataMode=Mode))
    mutation = Mock(side_effect=lambda scene, label, callback: callback(model, updater))
    monkeypatch.setattr(common, "current_context", lambda require_scene=True: (csc, None, None, scene))
    monkeypatch.setattr(common, "resolve_object", lambda scene, object: Identifier(object))
    monkeypatch.setattr(common, "modify", mutation)
    return SimpleNamespace(
        values=values,
        modes=modes,
        names=names,
        owners=owners,
        keys=keys,
        track=track,
        layer_editor=layer_editor,
        mutation=mutation,
        updater=updater,
        data=data,
    )


def test_reads_data_without_behavior_bindings_and_filters_pages(native):
    result = data_channels.object_data_read("object", frame=2, query="value", limit=1)
    assert result["channels"] == [
        {
            "id": "unbound",
            "name": "Unbound Value",
            "owner": "object",
            "mode": "Animation",
            "native_value_type": "float",
            "value": 3.0,
        }
    ]
    assert result["total"] == 1
    native.mutation.assert_not_called()


def test_duplicate_display_names_require_id_and_never_cross_owner(native):
    native.names["count"] = native.names["visible"] = "Duplicate"
    with pytest.raises(ValueError, match="matched 2"):
        data_channels.object_data_read("object", data="Duplicate")
    assert data_channels.object_data_read("object", data="count")["channels"][0]["value"] == 7
    native.owners["count"] = Identifier("foreign")
    with pytest.raises(ValueError, match="owner differs"):
        data_channels.object_data_set([{"object": "object", "data": "count", "value": 9}])
    native.mutation.assert_not_called()


def test_batch_can_mix_static_and_unbound_animated_channels_with_one_transaction(native):
    result = data_channels.object_data_set(
        [
            {"object": "object", "data": "Visible", "value": False},
            {"object": "object", "data": "Unbound Value", "value": 22},
            {"object": "object", "data": "Text", "value": "changed"},
        ],
        frame=0,
    )
    assert result["verification"]["matches"]
    assert native.values["visible"] is False
    assert native.values["unbound"] == [22.0, 2.0, 3.0, 4.0]
    assert result["channels"][0]["scope"] == "all frames"
    assert result["channels"][1]["scope"] == "frame"
    native.mutation.assert_called_once()
    native.layer_editor.set_fixed_interpolation_or_key_if_need.assert_not_called()


@pytest.mark.parametrize(
    "defect,match",
    [
        ("locked", "unlocked"),
        ("nokey", "existing key"),
        ("type", "requires int"),
        ("nonfinite", "finite scalar"),
        ("duplicate", "only once"),
        ("missing", "matched 0"),
        ("mode", "Static and Animation"),
    ],
)
def test_complete_prevalidation_precedes_any_native_mutation(native, defect, match):
    changes = [{"object": "object", "data": "visible", "value": False}]
    frame = 0
    if defect == "locked":
        native.track.is_locked = True
    elif defect == "nokey":
        frame = 1
        changes.append({"object": "object", "data": "unbound", "value": 20.0})
    elif defect == "type":
        changes.append({"object": "object", "data": "count", "value": True})
    elif defect == "nonfinite":
        changes.append({"object": "object", "data": "unbound", "value": float("nan")})
    elif defect == "duplicate":
        changes.append({"object": "object", "data": "Visible", "value": True})
    elif defect == "missing":
        changes.append({"object": "object", "data": "unknown", "value": 1})
    else:
        native.modes["visible"] = Mode.Unsupported
    with pytest.raises(ValueError, match=match):
        data_channels.object_data_set(changes, frame=frame)
    assert native.values["visible"] is True
    native.mutation.assert_not_called()


def test_explicit_key_creation_occurs_once_and_preserves_existing_keys(native):
    result = data_channels.object_data_set(
        [{"object": "object", "data": "unbound", "value": 12.0}], frame=1, create_keys=True
    )
    assert result["created_keys"] == ["track"]
    assert native.keys == {0, 1, 3}
    native.layer_editor.set_fixed_interpolation_or_key_if_need.assert_called_once_with(
        Identifier("track"), 1, True
    )


def test_native_solver_overwrite_cannot_be_reported_as_success(native):
    native.updater.run_update.side_effect = lambda *_: native.values["unbound"].__setitem__(0, -10.0)
    with pytest.raises(RuntimeError, match="solver-owned"):
        data_channels.object_data_set([{"object": "object", "data": "unbound", "value": 10.0}])


def test_static_write_does_not_require_creating_an_animation_key(native):
    result = data_channels.object_data_set([{"object": "object", "data": "visible", "value": False}], frame=2)
    assert result["channels"][0]["scope"] == "all frames"
    assert native.keys == {0, 3}


def test_terminal_missing_key_rejected_before_native_editor(native):
    native.keys.remove(3)
    with pytest.raises(ValueError, match="allocated successor"):
        data_channels.object_data_set(
            [{"object": "object", "data": "unbound", "value": 10.0}], frame=3, create_keys=True
        )
    native.mutation.assert_not_called()
