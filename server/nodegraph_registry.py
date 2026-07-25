"""Central registry of every nodes/ Node subclass available to the graph
editor. Single source of truth for both what the palette can spawn and
what the executor can resolve a spawned node's class_name back to -- one
list, not two, so those two things can never silently disagree.

Zero import-time side effects: nothing here is imported until
get_registry() is actually called, matching nodegraph_introspect.py's
same rule (see that module's docstring for why).
"""

from __future__ import annotations

_CACHE: dict[str, type] | None = None


def _load() -> dict[str, type]:
    from nodes.dataset.managed import ManagedDatasetSourceNode
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.lora_injector import ComfyUNetLoRANode
    from nodes.model.lora_saver import LoRACheckpointSaverNode
    from nodes.model.text_encoder import SDXLTextEncoderNode
    from nodes.optimizer.adafactor import AdafactorOptimizerNode
    from nodes.optimizer.adamw import AdamWOptimizerNode
    from nodes.optimizer.came import CAMEOptimizerNode
    from nodes.optimizer.foreach_adafactor import ForeachAdafactorOptimizerNode
    from nodes.optimizer.fused_adafactor import FusedAdafactorOptimizerNode
    from nodes.train.loss import MinSNRLossWeightingNode, UniformLossWeightingNode
    from nodes.train.schedule import ConstantLRScheduleNode, CosineLRScheduleNode
    from nodes.train.supervised import SupervisedLoRATrainerNode

    classes = [
        ManagedDatasetSourceNode,
        SafetensorsCheckpointNode, ComfyUNetLoRANode, SDXLTextEncoderNode, LoRACheckpointSaverNode,
        AdamWOptimizerNode, AdafactorOptimizerNode, CAMEOptimizerNode,
        ForeachAdafactorOptimizerNode, FusedAdafactorOptimizerNode,
        ConstantLRScheduleNode, CosineLRScheduleNode,
        UniformLossWeightingNode, MinSNRLossWeightingNode,
        SupervisedLoRATrainerNode,
    ]
    return {cls.__name__: cls for cls in classes}


def get_registry() -> dict[str, type]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load()
    return _CACHE


def domain_of(cls: type) -> str:
    """nodes.dataset.managed -> 'dataset'. Derived from the module path,
    not hand-labeled, so a node can't end up in the wrong palette group by
    someone forgetting to update a second list."""
    parts = cls.__module__.split(".")
    return parts[1] if len(parts) > 1 else "other"
