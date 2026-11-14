"""Contract-level check for the dataset/model node domains. Pure Python,
no torch/hardware/dataset needed -- verifies INPUTS/OUTPUTS declarations
and missing-input validation, not the wrapped legacy code's behavior."""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.dataset.node import DataSourceNode
from nodes.dataset.managed import ManagedDatasetSourceNode
from nodes.model.node import CheckpointSaverNode, LoRAInjectorNode, ModelProviderNode
from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
from nodes.model.lora_injector import ComfyUNetLoRANode
from nodes.model.lora_saver import LoRACheckpointSaverNode

ABSTRACT = [DataSourceNode, ModelProviderNode, LoRAInjectorNode, CheckpointSaverNode]
CONCRETE = [ManagedDatasetSourceNode, SafetensorsCheckpointNode, ComfyUNetLoRANode,
            LoRACheckpointSaverNode]


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    for cls in ABSTRACT:
        check(inspect.isabstract(cls), f"{cls.__name__} should still be abstract")

    for cls in CONCRETE:
        check(not inspect.isabstract(cls), f"{cls.__name__} should be concrete")
        cls()  # __init_subclass__ OUTPUTS check already ran at class-definition time

    node = SafetensorsCheckpointNode()
    try:
        node.build()
        raise AssertionError("build() with no inputs should have raised")
    except ValueError:
        pass

    print("All dataset/model node contract checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
