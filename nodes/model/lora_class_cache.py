"""_real_lora_classes()/_real_lora_classes_cache: the real
core.lora.LoRALinear/LoRAConv2d, guaranteed real even while
nodes/model/adapter_injection.py's adapter_strategy_scope has
core.lora.LoRALinear/LoRAConv2d patched to something else.

See adapter_injection.py's module docstring for the full recursion
hazard this exists to prevent, and docs/training_pipeline_design.md
section 3.1 for the design rationale. Originally lived in
adapter_strategy.py; moved here once nodes/model/dora_layer.py needed
it too and importing from adapter_strategy.py directly would have
cycled back through it (adapter_strategy.py imports dora_layer.py for
DoRAAdapter). Same situation, same fix, as lora_scaling.py's own
extraction -- see that module's docstring.
"""

from __future__ import annotations

# Populated by nodes/model/adapter_injection.py's adapter_strategy_scope,
# at the one moment core.lora.LoRALinear/LoRAConv2d are guaranteed to
# still be the real classes -- right before it patches them. Never
# written to from this module, or from adapter_strategy.py/dora_layer.py.
_real_lora_classes_cache: dict = {}


def _real_lora_classes():
    """Why this exists: any AdapterStrategy.wrap() that constructs real
    LoRALinear/LoRAConv2d (PlainLoRAAdapter does; DoRAAdapter does too,
    via composition -- see dora_layer.py) has to get the real classes
    regardless of what's calling it -- including when it's used as a
    delegate or base inside some *other* AdapterStrategy while a patch
    is active. Re-importing `core.lora.LoRALinear` live at that point
    would resolve to whatever adapter_strategy_scope currently has
    installed, recursing forever. Confirmed by hitting exactly that
    RecursionError while building smoke_test_adapter_injection.py, not
    theorized in advance.

    Falls back to a live import when the cache is empty, which is
    correct precisely because adapter_strategy_scope always populates
    the cache itself, at the one point core.lora's classes are still
    guaranteed real, before ever patching them -- so an empty cache
    means core.lora has never been patched at all yet, and its current
    classes are simply the real ones."""
    if _real_lora_classes_cache:
        return _real_lora_classes_cache["LoRALinear"], _real_lora_classes_cache["LoRAConv2d"]
    from core.lora import LoRAConv2d, LoRALinear
    return LoRALinear, LoRAConv2d
