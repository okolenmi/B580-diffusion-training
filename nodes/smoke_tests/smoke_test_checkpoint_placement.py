"""Correctness check for nodes/model/checkpoint_placement.py.

Run this directly: `python nodes/smoke_tests/smoke_test_checkpoint_placement.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.model.checkpoint_placement import (
    BlockCost, CheckpointPlacementPolicy, EveryBlockPlacement, GreedyRatioPlacement,
)
from nodes.resource_policy import ResourceBudget

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check_every_block_placement():
    print("\n=== EveryBlockPlacement: today's actual, unconditional behavior ===")
    blocks = [BlockCost(activation_bytes=1, recompute_ms=1.0),
              BlockCost(activation_bytes=10**9, recompute_ms=0.001)]
    policy = EveryBlockPlacement()
    tiny_budget = ResourceBudget(vram_budget_mb=0.001, vram_reserve_mb=0.0)
    huge_budget = ResourceBudget(vram_budget_mb=10**9, vram_reserve_mb=0.0)
    record(policy.select(blocks, tiny_budget) == [True, True],
           "checkpoints everything even under a budget that wouldn't require it")
    record(policy.select(blocks, huge_budget) == [True, True],
           "checkpoints everything even when nothing would need to be checkpointed")
    record(policy.select([], tiny_budget) == [],
           "empty block list -> empty result, not an error")
    record(isinstance(policy, CheckpointPlacementPolicy),
           "EveryBlockPlacement conforms to CheckpointPlacementPolicy")


def check_greedy_ratio_fits_under_budget_without_checkpointing_anything():
    print("\n=== GreedyRatioPlacement: budget already satisfied -> checkpoints nothing ===")
    blocks = [BlockCost(activation_bytes=10 * 2**20, recompute_ms=1.0),
              BlockCost(activation_bytes=10 * 2**20, recompute_ms=1.0)]
    budget = ResourceBudget(vram_budget_mb=100.0, vram_reserve_mb=0.0)  # 100MB >> 20MB total
    result = GreedyRatioPlacement().select(blocks, budget)
    record(result == [False, False],
           "20MB total activations fits a 100MB budget -- nothing needs checkpointing",
           detail=str(result))


def check_greedy_ratio_picks_best_ratio_blocks_first():
    print("\n=== GreedyRatioPlacement: checkpoints highest activation_bytes/recompute_ms first ===")
    # Block 0: 100MB / 1ms = ratio 100 (best -- cheap to recompute, big VRAM win)
    # Block 1: 100MB / 100ms = ratio 1 (worst -- expensive to recompute for the same win)
    # Block 2: 50MB / 1ms = ratio 50 (middle)
    blocks = [
        BlockCost(activation_bytes=100 * 2**20, recompute_ms=1.0),
        BlockCost(activation_bytes=100 * 2**20, recompute_ms=100.0),
        BlockCost(activation_bytes=50 * 2**20, recompute_ms=1.0),
    ]
    # Total = 250MB. Budget usable = 200MB (vram_budget - vram_reserve). Need to
    # shed >= 50MB. Best-ratio-first order is [0, 2, 1]. Checkpointing block 0
    # alone sheds 100MB, remaining = 150MB <= 200MB -- should stop right there,
    # leaving 1 and 2 resident (both False).
    budget = ResourceBudget(vram_budget_mb=220.0, vram_reserve_mb=20.0)
    result = GreedyRatioPlacement().select(blocks, budget)
    record(result == [True, False, False],
           "checkpoints only the single best-ratio block once that's enough to fit",
           detail=str(result))


def check_greedy_ratio_subtracts_the_reserve_margin():
    print("\n=== GreedyRatioPlacement: fits against (vram_budget_mb - vram_reserve_mb), "
          "not the raw ceiling ===")
    blocks = [BlockCost(activation_bytes=180 * 2**20, recompute_ms=1.0)]
    # Fits the raw 200MB ceiling with room to spare, but NOT 200MB - 50MB = 150MB
    # usable -- should checkpoint it.
    budget_with_reserve = ResourceBudget(vram_budget_mb=200.0, vram_reserve_mb=50.0)
    result = GreedyRatioPlacement().select(blocks, budget_with_reserve)
    record(result == [True],
           "180MB doesn't fit a 150MB usable budget (200 - 50 reserve) -- checkpointed",
           detail=str(result))

    budget_no_reserve = ResourceBudget(vram_budget_mb=200.0, vram_reserve_mb=0.0)
    result_no_reserve = GreedyRatioPlacement().select(blocks, budget_no_reserve)
    record(result_no_reserve == [False],
           "same 180MB fits the full 200MB when there's no reserve margin to protect",
           detail=str(result_no_reserve))


def check_greedy_ratio_checkpoints_everything_if_still_over_budget():
    print("\n=== GreedyRatioPlacement: checkpoints every block if even that isn't enough ===")
    blocks = [BlockCost(activation_bytes=100 * 2**20, recompute_ms=1.0),
              BlockCost(activation_bytes=100 * 2**20, recompute_ms=1.0)]
    budget = ResourceBudget(vram_budget_mb=10.0, vram_reserve_mb=0.0)  # far too small
    result = GreedyRatioPlacement().select(blocks, budget)
    record(result == [True, True],
           "checkpoints every block when the budget can't be met otherwise",
           detail=str(result))
    record(isinstance(GreedyRatioPlacement(), CheckpointPlacementPolicy),
           "GreedyRatioPlacement conforms to CheckpointPlacementPolicy")


def check_block_cost_is_frozen():
    print("\n=== BlockCost is frozen -- plain immutable data ===")
    cost = BlockCost(activation_bytes=100, recompute_ms=1.0)
    try:
        cost.activation_bytes = 200
        record(False, "assigning to a field should raise (frozen dataclass)")
    except Exception:
        record(True, "assigning to a field raises, as a frozen dataclass should")


def main():
    check_every_block_placement()
    check_greedy_ratio_fits_under_budget_without_checkpointing_anything()
    check_greedy_ratio_picks_best_ratio_blocks_first()
    check_greedy_ratio_subtracts_the_reserve_margin()
    check_greedy_ratio_checkpoints_everything_if_still_over_budget()
    check_block_cost_is_frozen()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
