import math

import pytest

from mcp_for_cascadeur.contracts import Tool, validate_value
from mcp_for_cascadeur.host.common import NativeEditError, configure, json_value, modify, workspace_path


def test_nested_batch_rejects_unknown_fields_and_nonfinite_values():
    schema = {
        "type": "object",
        "required": ["poses"],
        "additionalProperties": False,
        "properties": {
            "poses": {
                "type": "array",
                "minItems": 1,
                "maxItems": 64,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["frame", "value"],
                    "properties": {"frame": {"type": "integer", "minimum": 0}, "value": {"type": "number"}},
                },
            }
        },
    }
    validate_value({"poses": [{"frame": 0, "value": 1.0}]}, schema)
    for data in (
        {"poses": [{"frame": True, "value": 1}]},
        {"poses": [{"frame": 1, "value": math.nan}]},
        {"poses": [{"frame": 1, "value": 1, "surprise": 0}]},
    ):
        with pytest.raises(ValueError):
            validate_value(data, schema)


def test_alternatives_and_tool_contract():
    definition = Tool(
        "example",
        "Example",
        lambda **kwargs: kwargs,
        {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        ("value",),
    )
    assert definition.validate({"value": "x"}) == {"value": "x"}
    with pytest.raises(ValueError):
        definition.validate({"value": False})


def test_swallowed_native_callback_failure_is_not_success():
    class Scene:
        def modify_update(self, label, callback):
            try:
                callback(None, None, None)
            except ValueError:
                return True
            return True

    with pytest.raises(NativeEditError, match="inspect"):
        modify(Scene(), "Move controls", lambda model, updater: (_ for _ in ()).throw(ValueError("failed")))


def test_workspace_rejects_traversal_and_symlink_escape(tmp_path):
    configure(tmp_path)
    with pytest.raises(ValueError, match="inside"):
        workspace_path(tmp_path / ".." / "escape.casc")
    assert workspace_path(tmp_path / "new.casc") == tmp_path / "new.casc"


def test_huge_integer_is_rejected_by_bounds_without_overflow():
    with pytest.raises(ValueError, match="outside"):
        validate_value(10**500, {"type": "integer", "minimum": 0, "maximum": 100000})


@pytest.mark.parametrize("matrix", [False, True])
def test_native_quaternion_conversion_rejects_nonfinite_channels(matrix):
    class Quaternion:
        def w(self):
            return math.nan

        def x(self):
            return 0.0

        y = z = x

    class Rotation:
        def to_quaternion(self):
            return Quaternion()

    with pytest.raises(ValueError, match="nonfinite"):
        json_value(Rotation() if matrix else Quaternion())


def test_output_directory_is_rejected_even_when_overwrite_is_enabled(tmp_path):
    configure(tmp_path)
    target = tmp_path / "directory.casc"
    target.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        workspace_path(target, overwrite=True, suffixes=(".casc",))
