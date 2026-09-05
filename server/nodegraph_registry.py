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
    from nodes.dataset.prefetch import PrefetchingBatchSourceNode
    from nodes.dataset.renoise import RenoiseBatchSourceNode
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.lora_checkpoint_loader import LoRACheckpointLoaderNode
    from nodes.model.lora_injector import ComfyUNetLoRANode
    from nodes.model.lora_phases import LoRAPhaseSplitNode
    from nodes.model.lora_saver import LoRACheckpointSaverNode
    from nodes.model.parameters import ModelParametersNode
    from nodes.model.resources_controller import ResourcesControllerNode
    from nodes.model.lora_training_config import LoRATrainingConfigNode
    from nodes.model.text_encoder import SDXLTextEncoderNode
    from nodes.model.text_encoder_cache import CachingTextEncoderNode
    from nodes.model.text_encoder_prewarm import PrewarmedTextEncoderNode
    from nodes.monitor.training_progress import TrainingProgressMonitorNode
    from nodes.optimizer.adafactor import AdafactorOptimizerNode
    from nodes.optimizer.adamw import AdamWOptimizerNode, SimpleAdamWOptimizerNode
    from nodes.optimizer.came import CAMEOptimizerNode
    from nodes.optimizer.composed_adafactor import ComposedAdafactorOptimizerNode
    from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
    from nodes.optimizer.composed_came import ComposedCAMEOptimizerNode
    from nodes.optimizer.composed_fused_adafactor import ComposedFusedAdafactorOptimizerNode
    from nodes.optimizer.composed_fused_adamw import ComposedFusedAdamWOptimizerNode
    from nodes.optimizer.composed_fused_came import ComposedFusedCAMEOptimizerNode
    from nodes.optimizer.foreach_came import ForeachCAMEOptimizerNode
    from nodes.optimizer.foreach_adafactor import ForeachAdafactorOptimizerNode
    from nodes.optimizer.fused_adafactor import FusedAdafactorOptimizerNode
    from nodes.primitive.values import (BoolConstantNode, FloatConstantNode,
                                         IntConstantNode, StringConstantNode)
    from nodes.train.loss import (MinSNRLossWeightingNode, P2LossWeightingNode,
                                   UniformLossWeightingNode)
    from nodes.train.schedule import ConstantLRScheduleNode, CosineLRScheduleNode
    from nodes.train.supervised import SupervisedLoRATrainerNode

    classes = [
        ManagedDatasetSourceNode, PrefetchingBatchSourceNode, RenoiseBatchSourceNode,
        SafetensorsCheckpointNode, ComfyUNetLoRANode, LoRACheckpointLoaderNode, SDXLTextEncoderNode,
        CachingTextEncoderNode, PrewarmedTextEncoderNode,
        ModelParametersNode, LoRACheckpointSaverNode, LoRAPhaseSplitNode,
        ResourcesControllerNode, LoRATrainingConfigNode,
        TrainingProgressMonitorNode,
        AdamWOptimizerNode, SimpleAdamWOptimizerNode, AdafactorOptimizerNode, CAMEOptimizerNode,
        ComposedAdamWOptimizerNode, ComposedAdafactorOptimizerNode, ComposedCAMEOptimizerNode,
        ComposedFusedAdamWOptimizerNode, ComposedFusedAdafactorOptimizerNode, ComposedFusedCAMEOptimizerNode,
        ForeachCAMEOptimizerNode,
        ForeachAdafactorOptimizerNode, FusedAdafactorOptimizerNode,
        ConstantLRScheduleNode, CosineLRScheduleNode,
        UniformLossWeightingNode, MinSNRLossWeightingNode, P2LossWeightingNode,
        SupervisedLoRATrainerNode,
        FloatConstantNode, IntConstantNode, StringConstantNode, BoolConstantNode,
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
