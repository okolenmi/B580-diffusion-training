"""Contract-level check for the train domain (LRSchedule, LossWeighting,
TrainerNode) plus the model domain's TextEncoderNode. Pure Python, no
torch/hardware needed -- verifies declarations and the pure-math strategy
classes, not the actual step loop (SupervisedLoRATrainerNode.build()
needs real torch tensors and a real model; see the "Not yet verified"
note in docs/nodes_package_design.md's TrainerNode section)."""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.model.text_encoder import SDXLTextEncoderNode, TextEncoderNode
from nodes.train.loss import (LossWeightingNode, MinSNRLossWeightingNode,
                               UniformLossWeightingNode)
from nodes.train.node import TrainerNode
from nodes.train.schedule import (ConstantLRScheduleNode, CosineLRScheduleNode,
                                   LRScheduleNode)
from nodes.train.supervised import SupervisedLoRATrainerNode

ABSTRACT = [TextEncoderNode, LRScheduleNode, LossWeightingNode, TrainerNode]
CONCRETE = [SDXLTextEncoderNode, ConstantLRScheduleNode, CosineLRScheduleNode,
            UniformLossWeightingNode, MinSNRLossWeightingNode, SupervisedLoRATrainerNode]


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    for cls in ABSTRACT:
        check(inspect.isabstract(cls), f"{cls.__name__} should still be abstract")
    for cls in CONCRETE:
        check(not inspect.isabstract(cls), f"{cls.__name__} should be concrete")
        cls()

    const = ConstantLRScheduleNode().build(lr=1e-4)["schedule"]
    check(const.value(0) == 1e-4 and const.value(9999) == 1e-4, "constant schedule drifted")

    cosine = CosineLRScheduleNode().build(lr=1e-4, total_steps=100)["schedule"]
    check(abs(cosine.value(0) - 1e-4) < 1e-12, "cosine schedule should start at lr")
    check(cosine.value(99) < cosine.value(0), "cosine schedule should decay")

    check(UniformLossWeightingNode().build()["weighting"].weight(0.5) == 1.0,
          "uniform weighting should always be 1.0")
    snr = MinSNRLossWeightingNode().build(gamma=5.0)["weighting"]
    check(snr.weight(0.1) < snr.weight(10.0), "min-SNR should downweight low-noise (small sigma) steps")

    node = SupervisedLoRATrainerNode()
    try:
        node.build(model=None, batches=None)
        raise AssertionError("build() with missing required inputs should have raised")
    except ValueError:
        pass

    print("All train/text-encoder node contract checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
