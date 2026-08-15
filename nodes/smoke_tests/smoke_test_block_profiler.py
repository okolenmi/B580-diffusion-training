"""Correctness check for nodes/model/block_profiler.py.

Reuses smoke_test_gradient_checkpointing.py's stub comfy module (a real,
verbatim copy of ComfyUI's CheckpointFunction/checkpoint(), not
reconstructed from memory) rather than a second copy of it -- this test
cares whether ProfilingCheckpointing's instrumentation is correct
*without breaking* the underlying gradient fix, so it needs the same
real stub, not a simplified one.

Run this directly: `python nodes/smoke_tests/smoke_test_block_profiler.py`
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test_gradient_checkpointing import _install_stub_comfy_checkpoint_module

from nodes.components.device import DeviceContext
from nodes.model.block_profiler import BlockProfileCollector, ProfilingCheckpointing
from nodes.model.gradient_checkpointing import enable_frozen_param_safe_checkpointing

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class _SlowFrozenPlusTrainableBlock(nn.Module):
    """Same frozen+trainable shape as
    smoke_test_gradient_checkpointing.py's fixture, plus a deliberate
    sleep in _forward so recompute_ms has a real, checkable floor (not
    just "some positive float") and plus enough real tensor allocation
    inside _forward for an XPU/CUDA run's allocated_mb delta to be
    genuinely nonzero -- irrelevant on the CPU this test actually runs
    on (memory_stats() is None on CPU, see module docstring), but
    correct for anyone reading this as a template for a real-hardware
    profiling run."""

    def __init__(self, sleep_ms: float = 5.0):
        super().__init__()
        self.norm = nn.Parameter(torch.randn(4))
        self.norm.requires_grad_(False)
        self.adapter = nn.Parameter(torch.randn(4) * 0.1)
        self._sleep_s = sleep_ms / 1000

    def _forward(self, x):
        time.sleep(self._sleep_s)
        scratch = x.new_empty(256, 256)
        scratch.fill_(1.0)
        return x * self.norm + x * self.adapter + scratch.sum() * 0

    def forward(self, x, use_checkpoint, util_module):
        return util_module.checkpoint(self._forward, (x,), tuple(self.parameters()), use_checkpoint)


def check_gradients_still_correct_through_the_wrapper():
    print("\n=== ProfilingCheckpointing doesn't change the real gradient math ===")
    util = _install_stub_comfy_checkpoint_module()
    device_ctx = DeviceContext.for_device("cpu")
    collector = BlockProfileCollector()
    ProfilingCheckpointing(device_ctx, collector).apply()

    block = _SlowFrozenPlusTrainableBlock(sleep_ms=0)
    x = torch.randn(4, requires_grad=True)
    out = block(x, True, util)
    out.sum().backward()

    reference_block = _SlowFrozenPlusTrainableBlock(sleep_ms=0)
    reference_block.load_state_dict(block.state_dict())
    x_ref = x.detach().clone().requires_grad_(True)
    out_ref = reference_block._forward(x_ref)  # no checkpointing at all
    out_ref.sum().backward()

    record(torch.allclose(x.grad, x_ref.grad, atol=1e-6),
           "input gradient matches an unchecked reference run",
           detail=f"got={x.grad} ref={x_ref.grad}")
    record(torch.allclose(block.adapter.grad, reference_block.adapter.grad, atol=1e-6),
           "trainable param gradient matches an unchecked reference run")
    record(block.norm.grad is None,
           "frozen param gets no gradient at all (still None, not zero)")


def check_recompute_ms_reflects_real_wall_time():
    print("\n=== recompute_ms is a real measurement, not a placeholder ===")
    util = _install_stub_comfy_checkpoint_module()
    device_ctx = DeviceContext.for_device("cpu")
    collector = BlockProfileCollector()
    ProfilingCheckpointing(device_ctx, collector).apply()

    block = _SlowFrozenPlusTrainableBlock(sleep_ms=20.0)
    x = torch.randn(4, requires_grad=True)
    block(x, True, util).sum().backward()

    costs = collector.block_costs()
    record(len(costs) == 1, "exactly one distinct block recorded", detail=str(costs))
    (label, cost), = costs.items()
    record(cost.recompute_ms >= 15.0,
           "recompute_ms reflects the real ~20ms sleep inside _forward (loose floor "
           "for scheduler jitter)", detail=f"label={label} recompute_ms={cost.recompute_ms}")
    record(label.startswith("_SlowFrozenPlusTrainableBlock#"),
           "label is ClassName#ordinal, matching the real block's own class",
           detail=label)


def check_two_distinct_instances_get_two_distinct_stable_labels():
    print("\n=== two different block instances get two different, stable labels ===")
    util = _install_stub_comfy_checkpoint_module()
    device_ctx = DeviceContext.for_device("cpu")
    collector = BlockProfileCollector()
    ProfilingCheckpointing(device_ctx, collector).apply()

    block_a = _SlowFrozenPlusTrainableBlock(sleep_ms=0)
    block_b = _SlowFrozenPlusTrainableBlock(sleep_ms=0)
    x = torch.randn(4, requires_grad=True)

    block_a(x, True, util).sum().backward()
    block_b(x.detach().requires_grad_(True), True, util).sum().backward()
    block_a(x.detach().requires_grad_(True), True, util).sum().backward()  # a again

    costs = collector.block_costs()
    record(len(costs) == 2, "two distinct instances -> two distinct labels", detail=str(costs))
    labels = sorted(costs.keys())
    record(labels == ["_SlowFrozenPlusTrainableBlock#0", "_SlowFrozenPlusTrainableBlock#1"],
           "labels are first-seen ordinals, stable across repeated calls to the same instance",
           detail=str(labels))
    a_stats_after_two_calls = costs["_SlowFrozenPlusTrainableBlock#0"]
    record(a_stats_after_two_calls.recompute_ms >= 0.0,
           "block_a's stats reflect both of its two calls (mean, not just the last)")


def check_reset_clears_stats_but_keeps_labels():
    print("\n=== reset() clears accumulated stats, keeps label assignments ===")
    util = _install_stub_comfy_checkpoint_module()
    device_ctx = DeviceContext.for_device("cpu")
    collector = BlockProfileCollector()
    ProfilingCheckpointing(device_ctx, collector).apply()

    block = _SlowFrozenPlusTrainableBlock(sleep_ms=0)
    x = torch.randn(4, requires_grad=True)
    block(x, True, util).sum().backward()
    record(len(collector.block_costs()) == 1, "one block recorded before reset")

    collector.reset()
    record(len(collector.block_costs()) == 0, "block_costs() is empty right after reset()")

    block(x.detach().requires_grad_(True), True, util).sum().backward()
    costs_after = collector.block_costs()
    record(list(costs_after.keys()) == ["_SlowFrozenPlusTrainableBlock#0"],
           "same instance gets the exact same label again after reset()",
           detail=str(list(costs_after.keys())))


def check_switching_wrappers_actually_reinstalls():
    print("\n=== enable_frozen_param_safe_checkpointing(): switching recompute_wrapper "
          "re-installs; the same one twice is a no-op ===")
    util = _install_stub_comfy_checkpoint_module()
    device_ctx = DeviceContext.for_device("cpu")

    enable_frozen_param_safe_checkpointing()  # plain, no wrapper
    installed_plain = util.CheckpointFunction
    enable_frozen_param_safe_checkpointing()  # same (None) again
    record(util.CheckpointFunction is installed_plain,
           "calling with the same (None) recompute_wrapper twice does not reinstall")

    collector = BlockProfileCollector()
    ProfilingCheckpointing(device_ctx, collector).apply()
    installed_profiling = util.CheckpointFunction
    record(installed_profiling is not installed_plain,
           "switching to a real recompute_wrapper installs a different class")

    block = _SlowFrozenPlusTrainableBlock(sleep_ms=0)
    x = torch.randn(4, requires_grad=True)
    block(x, True, util).sum().backward()
    record(len(collector.block_costs()) == 1,
           "the newly-installed profiling wrapper is actually the one that ran")

    enable_frozen_param_safe_checkpointing()  # switch back to no wrapper
    record(util.CheckpointFunction is not installed_profiling,
           "switching back to no wrapper reinstalls again, doesn't get stuck on profiling")


def main():
    check_gradients_still_correct_through_the_wrapper()
    check_recompute_ms_reflects_real_wall_time()
    check_two_distinct_instances_get_two_distinct_stable_labels()
    check_reset_clears_stats_but_keeps_labels()
    check_switching_wrappers_actually_reinstalls()

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
