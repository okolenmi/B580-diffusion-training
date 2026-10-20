"""Constant nodes: a literal as a real port, so one number, text or switch is set once and wired into several inputs."""

from __future__ import annotations

from typing import Any

from ...graph import Node, Port


class IntNode(Node):
    DISPLAY_NAME = "Integer"
    CATEGORY = "Primitive"
    INPUTS = {"value": Port("value", int, required=False, default=0)}
    OUTPUTS = {"value": Port("value", int)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"value": 0 if inputs.get("value") is None else inputs["value"]}


class FloatNode(Node):
    DISPLAY_NAME = "Number"
    CATEGORY = "Primitive"
    INPUTS = {"value": Port("value", float, required=False, default=0.0)}
    OUTPUTS = {"value": Port("value", float)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"value": 0.0 if inputs.get("value") is None else inputs["value"]}


class TextNode(Node):
    DISPLAY_NAME = "Text"
    CATEGORY = "Primitive"
    INPUTS = {"value": Port("value", str, required=False, default="")}
    OUTPUTS = {"value": Port("value", str)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"value": "" if inputs.get("value") is None else inputs["value"]}


class SwitchNode(Node):
    DISPLAY_NAME = "Switch"
    CATEGORY = "Primitive"
    INPUTS = {"value": Port("value", bool, required=False, default=False)}
    OUTPUTS = {"value": Port("value", bool)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        return {"value": False if inputs.get("value") is None else inputs["value"]}
