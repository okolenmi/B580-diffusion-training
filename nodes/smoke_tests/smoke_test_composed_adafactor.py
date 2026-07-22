"""Real-hardware smoke test for ComposedAdafactorOptimizerNode.

Run this directly: `python nodes/smoke_tests/smoke_test_composed_adafactor.py`
Or for just one strategy: `python ... --strategy chunked`

Mirrors smoke_test_composed_came.py's structure and purpose exactly --
see that file's module docstring for why real device/torch behavior
(dtype casting, actual tensor placement, offload/reload round trips)
needs its own check beyond pure numerical equivalence. The numerical
correctness of AdafactorAlgorithm's actual formula is checked separately
in smoke_test_adafactor_equivalence.py, against the legacy reference
directly -- this file doesn't re-derive that, it only exercises the real
device/lifecycle plumbing around it.

What it checks per strategy, in order: same four checks as
smoke_test_composed_came.py (toy regression via real autograd, every
lifecycle method, an offload -> reload round trip with training resumed
after, and -- chunked only -- MemoryManager caching/cleanup). See that
file for the detailed reasoning behind each; not repeated here.

Prints a clear PASS/FAIL summary per strategy, plus an overall summary.
Does not touch core/, manager/, server/, or the training pipeline in any
way -- fully standalone.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.composed_adafactor import ComposedAdafactorOptimizerNode, _STRATEGIES


def pick_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_for_strategy(strategy_name: str, device: str) -> list:
    """Returns a list of failure descriptions (empty = all passed)."""
    print(f"\n{'#'*60}\n# strategy = {strategy_name!r}\n{'#'*60}")
    torch.manual_seed(0)
    failures = []

    true_W = torch.randn(4, 6, device=device) * 0.5
    W = (torch.randn(4, 6, device=device) * 0.1).requires_grad_(True)

    node = ComposedAdafactorOptimizerNode()
    handle = node.build(params=[W], lr=0.05, device=device, strategy=strategy_name)["optimizer"]

    losses = []
    for step in range(150):
        x = torch.randn(6, 10, device=device)
        y_true = true_W @ x
        y_pred = W @ x
        loss = ((y_pred - y_true) ** 2).mean()
        losses.append(loss.item())

        loss.backward()
        handle.step()
        handle.zero_grad()

    print(f"\n[1] Toy regression: loss {losses[0]:.6f} -> {losses[-1]:.6f} "
          f"({(1 - losses[-1]/losses[0])*100:.1f}% reduction)")
    if losses[-1] >= losses[0] * 0.5:
        failures.append(f"[{strategy_name}] Loss did not decrease meaningfully")
        print("    FAIL: expected substantial loss reduction")
    else:
        print("    PASS")

    print("\n[2] Lifecycle methods (real device tensors):")
    handle.decay_states(0.5)
    print("    decay_states(0.5): ran without error")

    handle.update_lr(0.02)
    if handle.lr != 0.02:
        failures.append(f"[{strategy_name}] update_lr did not update handle.lr")
        print(f"    FAIL: update_lr -- handle.lr={handle.lr}, expected 0.02")
    else:
        print(f"    update_lr(0.02): handle.lr correctly = {handle.lr}")

    print("\n[3] Offload -> reload round trip, then continue training:")
    handle.offload_states_to_cpu()
    post_offload_devices = {name: t.device.type for name, t in handle.states[0].items()}
    if not all(d == "cpu" for d in post_offload_devices.values()):
        failures.append(f"[{strategy_name}] offload_states_to_cpu did not move all state to CPU: {post_offload_devices}")
        print(f"    FAIL: state devices after offload = {post_offload_devices}")
    else:
        print(f"    offload_states_to_cpu: all state moved to CPU correctly")

    handle.reload_states_to_device(device)
    post_reload_devices = {name: t.device.type for name, t in handle.states[0].items()}
    expected_type = "xpu" if device == "xpu" else ("cuda" if device == "cuda" else "cpu")
    if not all(d == expected_type for d in post_reload_devices.values()):
        failures.append(f"[{strategy_name}] reload_states_to_device did not restore device correctly: {post_reload_devices}")
        print(f"    FAIL: state devices after reload = {post_reload_devices} (expected {expected_type})")
    else:
        print(f"    reload_states_to_device: all state correctly back on {expected_type}")

    loss_before_resume = losses[-1]
    resumed_losses = []
    for step in range(50):
        x = torch.randn(6, 10, device=device)
        y_true = true_W @ x
        y_pred = W @ x
        loss = ((y_pred - y_true) ** 2).mean()
        resumed_losses.append(loss.item())
        loss.backward()
        handle.step()
        handle.zero_grad()

    if any(torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l)) for l in resumed_losses):
        failures.append(f"[{strategy_name}] NaN/Inf loss after offload/reload round trip")
        print(f"    FAIL: NaN/Inf appeared in post-reload training")
    elif resumed_losses[-1] > loss_before_resume * 2:
        failures.append(f"[{strategy_name}] Loss got substantially worse after offload/reload round trip: "
                         f"{loss_before_resume:.6f} -> {resumed_losses[-1]:.6f}")
        print(f"    FAIL: loss degraded after round trip: "
              f"{loss_before_resume:.6f} -> {resumed_losses[-1]:.6f}")
    else:
        print(f"    PASS: training continues correctly after round trip "
              f"(loss {loss_before_resume:.6f} -> {resumed_losses[-1]:.6f})")

    if strategy_name == "chunked":
        print("\n[4] MemoryManager caching and cleanup (chunked strategy only):")
        mem = handle.strategy.memory
        stats = mem.stats()
        if stats["total_bytes"] <= 0:
            failures.append(f"[{strategy_name}] MemoryManager holds no cached buffer "
                             f"after training -- cross-step caching not exercised")
            print("    FAIL: no cached scratch buffer found after training")
        else:
            print(f"    Cached scratch buffer present after training: "
                  f"{stats['total_bytes']} bytes")

        ptr_before = mem.get_buffer("grad_cast", W.numel(), torch.float32, device).data_ptr()
        mem.release("grad_cast")
        x = torch.randn(6, 10, device=device)
        loss = ((W @ x - true_W @ x) ** 2).mean()
        loss.backward()
        handle.step()
        handle.zero_grad()
        ptr_after = mem.get_buffer("grad_cast", W.numel(), torch.float32, device).data_ptr()
        mem.release("grad_cast")
        if ptr_before != ptr_after:
            failures.append(f"[{strategy_name}] scratch buffer reallocated across steps "
                             f"instead of reused (cross-step caching broken)")
            print(f"    FAIL: buffer identity changed across steps "
                  f"({ptr_before} -> {ptr_after})")
        else:
            print(f"    PASS: same underlying buffer reused across steps "
                  f"(data_ptr={ptr_before})")

        handle.offload_states_to_cpu()
        stats_after_offload = mem.stats()
        if stats_after_offload["total_bytes"] != 0:
            failures.append(f"[{strategy_name}] MemoryManager still holds "
                             f"{stats_after_offload['total_bytes']} bytes after "
                             f"offload_states_to_cpu (offload/free asymmetry)")
            print(f"    FAIL: {stats_after_offload['total_bytes']} bytes still held "
                  f"after offload")
        else:
            print("    PASS: offload_states_to_cpu freed the cached scratch buffer")
        handle.reload_states_to_device(device)

    print("\n[5] reset_states / free_states:")
    handle.reset_states()
    all_zero = all(torch.count_nonzero(t) == 0 for t in handle.states[0].values())
    if not all_zero:
        failures.append(f"[{strategy_name}] reset_states did not zero all state")
        print("    FAIL: state not fully zeroed after reset_states()")
    else:
        print("    reset_states(): all state correctly zeroed")

    handle.free_states()
    if handle.states != []:
        failures.append(f"[{strategy_name}] free_states did not clear the states list")
        print("    FAIL: states list not cleared after free_states()")
    else:
        print("    free_states(): states list correctly cleared")

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(_STRATEGIES) + ["all"], default="all",
                         help="Which strategy to test (default: all of them)")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    if device == "cpu":
        print("  (no XPU/CUDA detected -- running on CPU. Still a real, "
              "meaningful check of the code path, just not the actual "
              "target hardware.)")

    strategy_names = list(_STRATEGIES) if args.strategy == "all" else [args.strategy]

    all_failures = []
    for name in strategy_names:
        all_failures.extend(run_for_strategy(name, device))

    print("\n" + "=" * 60)
    if all_failures:
        print(f"SMOKE TEST: {len(all_failures)} FAILURE(S):")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"SMOKE TEST: ALL CHECKS PASSED (strategies tested: {strategy_names}, device={device})")


if __name__ == "__main__":
    main()
