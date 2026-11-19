"""Adapter Stack: several saved adapters, each with its own strength, baked into a denoiser when it is loaded."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from families.api.parts import AdapterMerge, DenoiserPart
from graph import Node, Port


def parse_stack(text: str) -> list[tuple[str, float]]:
    """One adapter per line as ``file`` or ``file=strength``; blank lines and ``#`` comments are skipped, a name ending in ``=off`` is left out."""
    rows: list[tuple[str, float]] = []
    for number, line in enumerate((text or "").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.rpartition("=")
        if not sep:
            name, value = line, "1"
        name, value = name.strip(), value.strip().lower()
        if not name:
            raise ValueError(f"adapters line {number}: no file name before '='")
        if value == "off":
            continue
        try:
            rows.append((name, float(value)))
        except ValueError:
            raise ValueError(f"adapters line {number}: {value!r} is not a strength (a number, or off)") from None
    return rows


class AdapterStackNode(Node):
    DISPLAY_NAME = "Adapter Stack"
    CATEGORY = "Model"
    INPUTS = {
        "denoiser": Port("denoiser", DenoiserPart),
        "adapters": Port("adapters", str, required=False, default="", ui={"widget": "textarea", "rows": 4},
                         doc="One adapter per line under the loras folder: `file.safetensors=0.8`. Without `=` the strength is 1, "
                             "`=off` keeps the line but leaves the adapter out, a line starting with # is a comment. Applied in order."),
        "format": Port("format", str, required=False, default="kohya", advanced=True, doc="The file format of every adapter listed."),
    }
    OUTPUTS = {"denoiser": Port("denoiser", DenoiserPart, doc="The same denoiser with every listed adapter baked in when a generation node loads it "
                                                              "(Sampler), or when Inject Adapter loads it for training.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        part: DenoiserPart = inputs["denoiser"]
        fmt = inputs.get("format") or "kohya"
        merges = tuple(AdapterMerge(name, strength, fmt) for name, strength in parse_stack(inputs.get("adapters") or ""))
        return {"denoiser": replace(part, merges=(*part.merges, *merges))}
