"""Compatibility shim: `ComfyUNetWrapper`, `make_rand_cond` and
`clear_embedder_cache` now live in `nodes/model/unet_wrapper.py`.

Moved on 2026-10-02. `nodes/model/lora_injector.py` constructs this
class for every LoRA path in the node graph, so the model the rewrite
trains on was living in `core/`; owning it is also what let the
`layer_classes` parameter on `inject_lora_into_unet` replace the
module-global patching `adapter_injection.py` relied on.

`manager/builder.py` still builds this wrapper directly, and `core/`'s
own trainer and cache builders use it, so this stays until `core/` is
retired.
"""

from nodes.model.unet_wrapper import (
    ComfyUNetWrapper,
    clear_embedder_cache,
    make_rand_cond,
)

__all__ = ["ComfyUNetWrapper", "make_rand_cond", "clear_embedder_cache"]