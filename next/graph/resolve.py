"""Editing-time shape resolution (rework/contracts/resolve.md)."""

from __future__ import annotations

from .catalog import port_info
from .node import Environment, Resolution
from .registry import NodeRegistry


def resolution_dict(res: Resolution) -> dict:
    return {
        "state": res.state,
        "inputs": [port_info(p) for p in res.inputs.values()],
        "outputs": [port_info(p) for p in res.outputs.values()],
        "choices": {k: list(v) for k, v in res.choices.items()},
        "info": res.info,
        "message": res.message,
        "focus": res.focus,
        "detail": res.detail,
    }


def resolve_node(registry: NodeRegistry, class_name: str, params: dict, env: Environment | None = None) -> dict:
    """The JSON answer for one node. Raises KeyError for an unknown class; never for bad params."""
    classes, _ = registry.load()
    cls = classes[class_name]
    try:
        res = cls.resolve(dict(params), env)
    except Exception as exc:  # noqa: BLE001 -- editing in progress must not crash the server
        res = Resolution("error", dict(cls.INPUTS), dict(cls.OUTPUTS),
                         message=f"{type(exc).__name__}: {exc}")
    return resolution_dict(res)
