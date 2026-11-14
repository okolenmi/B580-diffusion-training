"""Primitive constant nodes -- output a literal value as a real port, so
the same number/string/flag can be wired into several inputs instead of
retyped in each one. No shared ABC: each is a trivial, independent
Node -- one input, matching output, build() returns it unchanged. A
shared base class here would be structure without behavior to justify it.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Node, Port


class FloatConstantNode(Node):
    INPUTS: ClassVar[dict[str, Port]] = {
        "value": Port(name="value", type=float, required=False, default=0.0),
    }
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "value": Port(name="value", type=float, required=True),
    }

    def build(self, **inputs) -> dict[str, float]:
        self.validate_inputs(inputs)
        result = {"value": float(inputs.get("value", self.INPUTS["value"].default))}
        self.validate_outputs(result)
        return result


class IntConstantNode(Node):
    INPUTS: ClassVar[dict[str, Port]] = {
        "value": Port(name="value", type=int, required=False, default=0),
    }
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "value": Port(name="value", type=int, required=True),
    }

    def build(self, **inputs) -> dict[str, int]:
        self.validate_inputs(inputs)
        result = {"value": int(inputs.get("value", self.INPUTS["value"].default))}
        self.validate_outputs(result)
        return result


class StringConstantNode(Node):
    INPUTS: ClassVar[dict[str, Port]] = {
        "value": Port(name="value", type=str, required=False, default=""),
    }
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "value": Port(name="value", type=str, required=True),
    }

    def build(self, **inputs) -> dict[str, str]:
        self.validate_inputs(inputs)
        result = {"value": str(inputs.get("value", self.INPUTS["value"].default))}
        self.validate_outputs(result)
        return result


class BoolConstantNode(Node):
    INPUTS: ClassVar[dict[str, Port]] = {
        "value": Port(name="value", type=bool, required=False, default=False),
    }
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "value": Port(name="value", type=bool, required=True),
    }

    def build(self, **inputs) -> dict[str, bool]:
        self.validate_inputs(inputs)
        result = {"value": bool(inputs.get("value", self.INPUTS["value"].default))}
        self.validate_outputs(result)
        return result
