"""Compatibility shim: everything here now lives in `nodes/model/`.

Moved on 2026-10-02:

* `LoRALinear`, `LoRAConv2d`, `LoRAConfig`, the `_inject_lora` walk,
  `inject_lora_into_unet`, `extract_lora_weights`, `load_lora_into_model`,
  `merge_lora_into_unet`, `lora_param_count` -> `nodes/model/lora.py`
* the timestep gate (`set_lora_gate`, `lora_gate_override`,
  `compute_lora_gate`, `_current_gate`) -> `nodes/model/lora.py` too

`nodes/model/` had been patching this module's `LoRALinear`/`LoRAConv2d`
names in place to substitute its own DoRA and NF4 layers, because
`_inject_lora` resolved them as bare module-level names. Owning the file
is what made that patch replaceable by an explicit argument; see
`inject_lora_into_unet`'s `layer_classes` parameter.

`core/`'s trainer and both of its cache builders construct
`LoRALinear`/`LoRAConv2d` and set the gate directly, so this shim stays
until `core/` is retired. It re-exports the same objects rather than
wrapping them, so a mutation through either path is visible through both
— which matters for the gate, a module-level global by design.
"""

# Deliberately NOT re-exporting `_current_gate`. It is a module-level
# global by design, so `from nodes.model.lora import _current_gate`
# captures its value at import time and never tracks later changes --
# re-exporting it here would hand every caller a stale snapshot that reads
# as a live one. (Writing it is still shared, via set_lora_gate/lora_gate_override,
# which are the same function objects on both sides.) Nothing outside
# nodes/model/ reads it: core/ only ever sets the gate.
from nodes.model.lora import (
    GroupedLoRALinear,
    LoRAConfig,
    LoRAConv2d,
    LoRALinear,
    _inject_lora,
    _key_to_lora_key,
    _segment_match,
    compute_lora_gate,
    extract_lora_weights,
    inject_lora_into_unet,
    load_lora_into_model,
    lora_gate_override,
    lora_param_count,
    merge_lora_into_unet,
    set_lora_gate,
)

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LoRAConv2d",
    "GroupedLoRALinear",
    "inject_lora_into_unet",
    "extract_lora_weights",
    "load_lora_into_model",
    "merge_lora_into_unet",
    "lora_param_count",
    "compute_lora_gate",
    "set_lora_gate",
    "lora_gate_override",
]