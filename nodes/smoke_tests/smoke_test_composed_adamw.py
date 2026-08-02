"""Real-hardware smoke test for ComposedAdamWOptimizerNode.

Run this directly: `python nodes/smoke_tests/smoke_test_composed_adamw.py`
Or for just one strategy: `python ... --strategy foreach`

Mirrors smoke_test_composed_came.py/smoke_test_composed_adafactor.py's
structure exactly -- see those files for the detailed reasoning behind
each check. AdamWAlgorithm's own formula correctness is checked
separately, against CPUAdamW directly, in
smoke_test_adamw_equivalence.py; this file only exercises the real
device/lifecycle plumbing around it (offload/reload, decay, reset/free).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode, _STRATEGIES


def pick_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_for_strategy(strategy_name: str, device: str) -> list:
    print(f"\n{'#'*60}\n# strategy = {strategy_name!r}\n{'#'*60}")
    torch.manual_seed(0)
    failures = []

    true_W = torch.randn(4, 6, device=device) * 0.5
    W = (torch.randn(4, 6, device=device) * 0.1).requires_grad_(True)

    node = ComposedAdamWOptimizerNode()
    handle = node.build(params=[W], lr=0.05, weight_decay=0.01,
                         device=device, strategy=strategy_name)["optimizer"]

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

    print("\n[2] Lifecycle methods (real device tensors, weight_decay > 0 -- "
          "exercises the decay path):")
    handle.decay_states(0.5)
    print("    decay_states(0.5): ran without error")

    handle.update_lr(0.02)
    if handle.lr != 0.02:
        failures.append(f"[{strategy_name}] update_lr did not update handle.lr")
        print(f"    FAIL: update_lr -- handle.lr={handle.lr}, expected 0.02")
    else:
        print(f"    update_lr(0.02): handle.lr correctly = {handle.lr}")

    print("\n[3] Offload -> reload round trip, then continue training:")
    pre_offload_snapshot = {name: t.clone() for name, t in handle.states[0].items()}
    handle.offload_states_to_cpu()
    post_offload_devices = {name: t.device.type for name, t in handle.states[0].items()}
    if not all(d == "cpu" for d in post_offload_devices.values()):
        failures.append(f"[{strategy_name}] offload_states_to_cpu did not move all state to CPU: {post_offload_devices}")
        print(f"    FAIL: state devices after offload = {post_offload_devices}")
    else:
        print("    offload_states_to_cpu: all state moved to CPU correctly")

    handle.reload_states_to_device(device)
    post_reload_devices = {name: t.device.type for name, t in handle.states[0].items()}
    expected_type = "xpu" if device == "xpu" else ("cuda" if device == "cuda" else "cpu")
    if not all(d == expected_type for d in post_reload_devices.values()):
        failures.append(f"[{strategy_name}] reload_states_to_device did not restore device correctly: {post_reload_devices}")
        print(f"    FAIL: state devices after reload = {post_reload_devices} (expected {expected_type})")
    else:
        print(f"    reload_states_to_device: all state correctly back on {expected_type}")

    values_match = all(torch.equal(pre_offload_snapshot[name], t)
                        for name, t in handle.states[0].items())
    if not values_match:
        failures.append(f"[{strategy_name}] offload/reload round trip did not "
                         f"preserve state values exactly")
        print("    FAIL: state values changed across the offload/reload round trip")
    else:
        print("    PASS: state values preserved exactly across the round trip")

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
        print("    FAIL: NaN/Inf appeared in post-reload training")
    elif resumed_losses[-1] > losses[0] * 0.5:
        failures.append(f"[{strategy_name}] Loss did not stay well below its original "
                         f"starting point after offload/reload round trip: "
                         f"{losses[0]:.6f} -> {resumed_losses[-1]:.6f}")
        print(f"    FAIL: loss no longer well below its starting point: "
              f"{losses[0]:.6f} -> {resumed_losses[-1]:.6f}")
    else:
        print(f"    PASS: training continues correctly after round trip "
              f"(started at {losses[0]:.6f}, now at {resumed_losses[-1]:.6f})")

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
