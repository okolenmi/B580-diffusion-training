"""Reflection: a real ``nodes.core.Node`` subclass -> port ``NodeInfo``.

Ported from ``server/nodegraph_introspect.py`` with three deliberate
changes (see ``docs/design/backend/05-graph-runtime.md`` section 7):

* legacy-class guessing (``introspect_legacy_class``) is dropped
  entirely -- everything discoverable *is* a real Node;
* defaults are emitted JSON-natively in ``default`` **and** as a repr
  string in ``default_repr``, instead of forcing consumers to parse
  reprs;
* ``domain`` comes in as an argument (the caller owns grouping).

Read-only: no instantiation, no side effects -- ``has_diagnostics`` is
an identity check on the method, and presets are only resolved for
``NODE_KIND == "dynamic"`` classes (the base implementation raises by
design, same as legacy).
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from nodes.core import Node

from ...application.ports.graph_catalog import NodeInfo, PortInfo, PresetInfo

# Domain vocabulary this project's own class names use that a generic
# capital-letter split gets wrong -- "ComfyUNetLoRA" would split as
# "Comfy U Net Lo R A". A closed list grounded in real class names
# (same list legacy derived); an unknown token still gets *a* label
# (one word per capital) rather than a crash, and DISPLAY_NAME is the
# per-class override.
_KNOWN_DISPLAY_TOKENS = sorted(
    ["UNet", "LoRA", "DoRA", "QDoRA", "CAME", "AdamW", "SDXL", "SNR", "NF4",
     "BF16", "XPU", "VRAM", "LR", "P2"],
    key=len,
    reverse=True,
)

_WORD_RE = re.compile(r"[A-Z][a-z0-9]*")


def auto_display_name(class_name: str) -> str:
    """"ComfyUNetLoRANode" -> "Comfy UNet LoRA" (strips the trailing
    "Node" suffix every real node carries). Never raises."""
    stem = class_name[:-4] if class_name.endswith("Node") and len(class_name) > 4 else class_name
    words: list[str] = []
    i = 0
    while i < len(stem):
        for token in _KNOWN_DISPLAY_TOKENS:
            if stem.startswith(token, i):
                words.append(token)
                i += len(token)
                break
        else:
            match = _WORD_RE.match(stem, i)
            if match:
                words.append(match.group())
                i = match.end()
            else:
                words.append(stem[i])  # not a capital-start position; consume
                i += 1
    return " ".join(words) if words else class_name


def _display_name_for(cls: type) -> str:
    override = getattr(cls, "DISPLAY_NAME", None)
    return override if override else auto_display_name(cls.__name__)


def _type_str(annotation: Any) -> str:
    if annotation is Any:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _type_mro(annotation: Any) -> tuple[str, ...]:
    if annotation is Any:
        return ("Any",)
    if isinstance(annotation, type):
        return tuple(
            c.__name__ for c in annotation.__mro__ if c.__name__ not in ("object", "ABC")
        )
    return (str(annotation),)


def _is_json_native(value: Any) -> bool:
    """True when the declared default survives a JSON round-trip."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _port_info(port, *, is_output: bool = False) -> PortInfo:
    """One nodes.core.Port -> JSON-ready PortInfo.

    An output has no default to report; a required input likewise
    ("no default" is what *required* means at the wire level, even if
    the Port object carries one). An optional input reports the JSON
    value when representable (else None) plus the repr string always --
    consumers pick whichever they can use instead of parsing reprs.
    """
    if is_output or port.required:
        default = None
        default_repr = None
    else:
        default = port.default if _is_json_native(port.default) else None
        default_repr = repr(port.default)
    return PortInfo(
        name=port.name,
        type=_type_str(port.type),
        required=port.required,
        doc=port.doc,
        type_mro=_type_mro(port.type),
        default=default,
        default_repr=default_repr,
        path_kind=None if is_output else port.path_kind,
        choices=None if is_output else (
            tuple(port.choices) if port.choices is not None else None
        ),
        visible_when=None if is_output else port.visible_when,
        widget_only=False if is_output else port.widget_only,
    )


def _presets(cls: type) -> tuple[PresetInfo, ...] | None:
    """Presets only for dynamic nodes -- the base list_presets() raises
    by design (nodes.core), exactly as legacy guarded it."""
    if cls.NODE_KIND != "dynamic":
        return None
    return tuple(
        PresetInfo(
            name=preset.name,
            required_inputs=tuple(
                _port_info(p) for p in preset.required_inputs.values()
            ),
            required_outputs=tuple(
                _port_info(p, is_output=True) for p in preset.required_outputs.values()
            ),
        )
        for preset in cls.list_presets()
    )


def introspect_node_class(cls: type, *, domain: str) -> NodeInfo:
    """Read DECLARED metadata off a real Node subclass (INPUTS/OUTPUTS
    are real Port objects the author wrote down -- never guessed from a
    constructor signature)."""
    doc = (inspect.getdoc(cls) or "").strip().split("\n")[0]
    bases = tuple(
        b.__name__ for b in cls.__mro__[1:] if b.__name__ not in ("object", "ABC")
    )
    return NodeInfo(
        class_name=cls.__name__,
        display_name=_display_name_for(cls),
        domain=domain,
        module=cls.__module__,
        doc=doc,
        bases=bases,
        inputs=tuple(_port_info(p) for p in cls.INPUTS.values()),
        outputs=tuple(_port_info(p, is_output=True) for p in cls.OUTPUTS.values()),
        node_kind=cls.NODE_KIND,
        presets=_presets(cls),
        # Plain function identity: diagnostics() is a regular instance
        # method, so ClassName.method is already the bare function.
        has_diagnostics=cls.diagnostics is not Node.diagnostics,
    )
