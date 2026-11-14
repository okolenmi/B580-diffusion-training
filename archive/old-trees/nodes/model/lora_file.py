"""The LoRA *file* format: what goes in a .safetensors, and what stays out.

A trained LoRA's strength is `(alpha / rank) * block_weight`. The in-memory
state dict carries the block weight as a per-module `<module>.block_weight`
tensor (`lora.extract_lora_weights`, `lora_phases.extract_*`), so a resumed
run restores exactly the strength it trained at.

**That key must not reach the file.** Kohya-format LoRA readers take the
tensors they know (`.lora_down.weight`, `.lora_up.weight`, `.alpha`,
`.dora_scale`) and treat anything left over as an error. Measured with
diffusers 0.41.0's own `_convert_non_diffusers_lora_to_diffusers`, executed
on a toy state dict: the same three tensors convert, and adding one
`.block_weight` tensor raises

    ValueError: The following keys have not been correctly renamed: ...

so every LoRA saved with the extra keys would have been unloadable there.
(The comment that introduced the keys said readers "ignore" unknown keys;
that is true of some, not of that one.)

So on the way out the keys are lifted into the safetensors *metadata*, which
every reader ignores, and only when they say something: a block weight of 1.0
is the default and is not listed. The field is still written, as `{}`, when
every weight is the default: its presence is what tells a loader "this file
was trained at these weights", as opposed to a file from before block weights
were recorded, where the configured weight keeps applying. On the way in,
`load_lora_state_dict` puts the keys back, so the state dict a loader sees is
the same as it was before the save.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Per-module key in the in-memory state dict.
BLOCK_WEIGHT_SUFFIX = ".block_weight"

#: Metadata field in the file: JSON {module_key: weight}, non-default only.
BLOCK_WEIGHTS_METADATA_KEY = "block_weights"

_DEFAULT_WEIGHT = 1.0


def split_block_weights(state_dict: dict) -> tuple[dict, dict[str, str]]:
    """`(tensors for the file, metadata entries for the file)`.

    The input is not modified. Only weights that differ from the default are
    listed; the field exists whenever the state dict carried any weights."""
    tensors = {}
    recorded: dict[str, float] = {}
    seen = False
    for key, value in state_dict.items():
        if not key.endswith(BLOCK_WEIGHT_SUFFIX):
            tensors[key] = value
            continue
        seen = True
        weight = float(value.reshape(-1)[0])
        if abs(weight - _DEFAULT_WEIGHT) > 1e-6:
            recorded[key[: -len(BLOCK_WEIGHT_SUFFIX)]] = weight
    if not seen:
        return tensors, {}
    return tensors, {BLOCK_WEIGHTS_METADATA_KEY: json.dumps(recorded, sort_keys=True)}


def restore_block_weights(state_dict: dict, metadata: dict | None) -> dict:
    """The inverse of `split_block_weights`, in place; returns `state_dict`.

    Every module with a down-projection gets its `.block_weight` back: the
    recorded value, or 1.0 when the file listed none for it. A file without
    the metadata field (written before block weights were recorded) is left
    alone, so a loader keeps applying the configured weight to it."""
    import torch

    raw = (metadata or {}).get(BLOCK_WEIGHTS_METADATA_KEY)
    if raw is None:
        return state_dict
    try:
        recorded = json.loads(raw)
    except ValueError:
        print(f"    [LoRA] ignoring unreadable {BLOCK_WEIGHTS_METADATA_KEY!r} metadata")
        return state_dict
    if not isinstance(recorded, dict):
        return state_dict
    down = ".lora_down.weight"
    for key in [k for k in state_dict if k.endswith(down)]:
        module = key[: -len(down)]
        weight = recorded.get(module, _DEFAULT_WEIGHT)
        state_dict.setdefault(
            module + BLOCK_WEIGHT_SUFFIX, torch.tensor([float(weight)], dtype=torch.float32))
    return state_dict


def load_lora_state_dict(path: str | Path) -> dict:
    """`safetensors.torch.load_file` plus the block weights from the metadata."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    state_dict = load_file(str(path))
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
    return restore_block_weights(state_dict, metadata)
