"""apply_lora_gate(): the LoRA timestep gate (core.lora.py's
set_lora_gate()/compute_lora_gate()) applied to an already-computed
delta tensor.

Originally written inside dora_layer.py as a private `_apply_gate()`;
moved here once nodes/model/nf4_lora_layer.py needed the identical
logic too, rather than a second copy of it -- same reasoning as every
other extraction this session (nodes/optimizer/strategies/shape_grouping.py,
nodes/optimizer/strategies/base.py's apply_updates_batched()): a real
duplication bug (nodes/optimizer/strategy_registry.py's own history) is
exactly what this project decided extractions like this exist to
prevent, not a style preference.

core.lora.LoRALinear.forward() reads the same module-level
`core.lora._current_gate` directly; both DoRALinear and NF4LoRALinear/
NF4LoRAConv2d apply the gate to their *entire* computed delta (not just
a raw pre-scaling LoRA term), matching LoRALinear's own semantics
exactly: gate=0 must produce exactly the frozen base output, nothing
else.
"""

from __future__ import annotations

import torch


def apply_lora_gate(delta: torch.Tensor) -> torch.Tensor:
    """delta * gate, broadcasting gate (one scalar per batch element)
    against delta's leading dimension. gate is None (no-op, returns
    delta unchanged) whenever core.lora.set_lora_gate() hasn't been
    called this step -- the default, and the common case for anyone not
    using a restricted t_low/t_high dataset."""
    from core.lora import _current_gate
    gate = _current_gate
    if gate is None:
        return delta
    g = gate.to(device=delta.device, dtype=delta.dtype)
    g = g.view(-1, *([1] * (delta.dim() - 1)))
    return delta * g
