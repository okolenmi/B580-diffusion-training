"""Contract-level check for the train domain (LRSchedule, LossWeighting,
TrainerNode) plus the model domain's TextEncoderNode. Pure Python, no
torch/hardware needed -- verifies declarations and the pure-math strategy
classes, not the actual step loop (SupervisedLoRATrainerNode.build()
needs real torch tensors and a real model)."""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.model.text_encoder import SDXLTextEncoderNode, TextEncoderNode
from nodes.train.loss import (LossWeightingNode, MinSNRLossWeightingNode,
                               UniformLossWeightingNode)
from nodes.train.node import TrainerNode
from nodes.train.schedule import (ConstantLRScheduleNode, CosineLRScheduleNode,
                                   LRScheduleNode, WarmupLRScheduleNode)
from nodes.train.supervised import SupervisedLoRATrainerNode

ABSTRACT = [TextEncoderNode, LRScheduleNode, LossWeightingNode, TrainerNode]
CONCRETE = [SDXLTextEncoderNode, ConstantLRScheduleNode, CosineLRScheduleNode,
            WarmupLRScheduleNode,
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

    # WarmupLRSchedule: linear ramp toward the *wrapped* schedule's own value
    # at each step, exact join at the boundary, pure passthrough at 0.
    const2 = ConstantLRScheduleNode().build(lr=1e-4)["schedule"]
    warm = WarmupLRScheduleNode().build(
        schedule=const2, warmup_steps=100, warmup_start=1e-8)["schedule"]
    check(abs(warm.value(0) - (1e-8 + (1e-4 - 1e-8) * 0.01)) < 1e-15,
          "warmup step 0 must sit one ramp-step above warmup_start")
    check(warm.value(50) > warm.value(10) > warm.value(0),
          "warmup must ramp up across its window")
    check(abs(warm.value(99) - const2.value(99)) < 1e-15,
          "last warmup step must join the wrapped schedule exactly (frac reaches 1.0)")
    check(warm.value(100) == const2.value(100) and warm.value(9999) == 1e-4,
          "after warmup the wrapped schedule passes through unchanged")
    wc = WarmupLRScheduleNode().build(schedule=cosine, warmup_steps=100)["schedule"]
    check(abs(wc.value(50) - cosine.value(50) * 51 / 100) < 1e-15,
          "wrapping a cosine must track the cosine's own value per step during "
          "warmup (lerp toward inner.value(step)), not toward a frozen target")
    w0 = WarmupLRScheduleNode().build(schedule=const2, warmup_steps=0)["schedule"]
    check(w0.value(0) == 1e-4 and w0.value(500) == 1e-4,
          "warmup_steps=0 must be a pure passthrough")
    try:
        WarmupLRScheduleNode().build(schedule=const2, warmup_steps=-1)
        raise AssertionError("negative warmup_steps should have raised")
    except ValueError:
        pass

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
