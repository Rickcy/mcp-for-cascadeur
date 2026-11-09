"""Tool contracts shared by the external server and Cascadeur's embedded Python."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    properties: dict = field(default_factory=dict)
    required: tuple[str, ...] = ()
    read_only: bool = True

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": self.properties,
            "required": list(self.required),
            "additionalProperties": False,
        }

    def validate(self, arguments: dict) -> dict:
        validate_value(arguments, self.input_schema)
        return arguments


def validate_value(value: Any, schema: dict, path: str = "arguments", depth: int = 0) -> None:
    """Validate the bounded JSON Schema subset used by our tools, without SDK imports."""
    if depth > 32:
        raise ValueError(f"{path}: nesting exceeds 32 levels")
    alternatives = schema.get("anyOf", schema.get("oneOf"))
    if alternatives:
        matches = 0
        for option in alternatives:
            try:
                validate_value(value, option, path, depth + 1)
                matches += 1
            except ValueError:
                pass
        if not matches or ("oneOf" in schema and matches != 1):
            raise ValueError(f"{path}: value does not match an allowed shape")
    expected = schema.get("type")
    if isinstance(expected, list):
        validate_value(value, {"anyOf": [{**schema, "type": kind} for kind in expected]}, path, depth + 1)
        return
    valid = {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if expected and not valid.get(expected, False):
        raise ValueError(f"{path}: expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: expected one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: unexpected constant")
    if type(value) in (int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{path}: number must be finite")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{path}: number is outside the permitted range")
        if value <= schema.get("exclusiveMinimum", -math.inf):
            raise ValueError(f"{path}: number is below the permitted range")
    elif type(value) is str:
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 1_000_000):
            raise ValueError(f"{path}: string length is outside the permitted range")
    elif type(value) is list:
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100_000):
            raise ValueError(f"{path}: array length is outside the permitted range")
        if schema.get("uniqueItems"):
            import json

            encoded = [json.dumps(item, sort_keys=True, allow_nan=False) for item in value]
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"{path}: entries must be unique")
        for index, item in enumerate(value):
            item_schema = schema.get("items", {})
            if "prefixItems" in schema and index < len(schema["prefixItems"]):
                item_schema = schema["prefixItems"][index]
            validate_value(item, item_schema, f"{path}[{index}]", depth + 1)
    elif type(value) is dict:
        properties = schema.get("properties", {})
        missing = set(schema.get("required", ())) - value.keys()
        unknown = value.keys() - properties.keys()
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
        if len(value) > schema.get("maxProperties", 10_000):
            raise ValueError(f"{path}: too many fields")
        for name, item in value.items():
            if type(name) is not str:
                raise ValueError(f"{path}: keys must be strings")
            child = properties.get(name, schema.get("additionalProperties", {}))
            validate_value(item, child if isinstance(child, dict) else {}, f"{path}.{name}", depth + 1)
    elif value is not None and type(value) not in (bool, int, float, str):
        raise ValueError(f"{path}: expected JSON-compatible data")
