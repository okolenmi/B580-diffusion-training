"""Real-hardware lifecycle smoke test for ComposedFusedOptimizerHandle,
run against all three Algorithms it drives (proving it's genuinely
algorithm-agnostic, not just for the one Adafactor formally verified
against a legacy reference in smoke_test_fused_adafactor_equivalence.py).

Run this directly: `python nodes/smoke_tests/smoke_test_composed_fused.py`

Two things checked per algorithm:
1. Toy regression + full OptimizerHandle/FusedOptimizerHandle lifecycle
   (offload/reload with exact state preservation, decay/reset/free,
   hook teardown on free_states()) -- same structure as
   smoke_test_composed_adafactor.py/smoke_test_composed_adamw.py, adapted
   for real backward()-driven execution instead of step()/zero_grad().
2. Each of the three *Node* classes' build() -- confirming the Node/Port
   plumbing (not just the Handle underneath it) works for each algorithm,
   since that layer has its own, separately-checkable logic (defaults,
   validate_inputs/validate_outputs).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.algorithms.adamw import AdamWAlgorithm
from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.algorithms.came import CAMEAlgorithm
from nodes.optimizer.composed_fused import ComposedFusedOptimizerHandle
from nodes.optimizer.composed_fused_adafactor import ComposedFusedAdafactorOptimizerNode
from nodes.optimizer.composed_fused_came import ComposedFusedCAMEOptimizerNode
from nodes.optimizer.composed_fused_adamw import ComposedFusedAdamWOptimizerNode
from nodes.optimizer.handle import FusedOptimizerHandle
from nodes.model.handle import ParameterList

DEVICE = "cpu"
_ALGORITHM_FACTORIES = {
    "adamw": lambda: AdamWAlgorithm(weight_decay=1e-2),
    "adafactor": lambda: AdafactorAlgorithm(weight_decay=0.5),
    "came": lambda: CAMEAlgorithm(weight_decay=0.01),
}


def run_lifecycle_for(algorithm_name: str) -> list:
    print(f"\n{'#'*60}\n# algorithm = {algorithm_name!r}\n{'#'*60}")
    torch.manual_seed(0)
    failures = []

    true_W = torch.randn(4, 6) * 0.5
    W = (torch.randn(4, 6) * 0.1).requires_grad_(True)

    handle = ComposedFusedOptimizerHandle(_ALGORITHM_FACTORIES[algorithm_name](),
                                           [W], lr=0.05, device=DEVICE)

    losses = []
    for step in range(150):
        x = torch.randn(6, 10)
        handle.begin_step(1)
        loss = ((W @ x - true_W @ x) ** 2).mean()
        losses.append(loss.item())
        loss.backward()

    print(f"[1] Toy regression (real backward(), hook-driven): "
          f"loss {losses[0]:.6f} -> {losses[-1]:.6f}")
    if losses[-1] >= losses[0] * 0.5:
        failures.append(f"[{algorithm_name}] Loss did not decrease meaningfully")
        print("    FAIL")
    else:
        print("    PASS")

    print("[2] Lifecycle methods:")
    handle.decay_states(0.5)
    print("    decay_states(0.5): ran without error")

    handle.update_lr(0.02)
    if handle.lr != 0.02:
        failures.append(f"[{algorithm_name}] update_lr did not update handle.lr")
        print(f"    FAIL: handle.lr={handle.lr}")
    else:
        print(f"    update_lr(0.02): handle.lr correctly = {handle.lr}")

    pre_offload = {name: t.clone() for name, t in handle.states[0].items()}
    handle.offload_states_to_cpu()
    post_offload = {name: t.device.type for name, t in handle.states[0].items()}
    if not all(d == "cpu" for d in post_offload.values()):
        failures.append(f"[{algorithm_name}] offload_states_to_cpu left state off-CPU: {post_offload}")
        print(f"    FAIL: {post_offload}")
    else:
        print("    offload_states_to_cpu: all state moved to CPU correctly")

    handle.reload_states_to_device(DEVICE)
    values_match = all(torch.equal(pre_offload[name], t) for name, t in handle.states[0].items())
    if not values_match:
        failures.append(f"[{algorithm_name}] offload/reload round trip changed state values")
        print("    FAIL: state values changed across the round trip")
    else:
        print("    PASS: offload/reload round trip preserves state values exactly")

    for step in range(30):
        x = torch.randn(6, 10)
        handle.begin_step(1)
        loss = ((W @ x - true_W @ x) ** 2).mean()
        loss.backward()
    if torch.isnan(loss) or torch.isinf(loss):
        failures.append(f"[{algorithm_name}] NaN/Inf after offload/reload round trip")
        print("    FAIL: NaN/Inf after round trip")
    else:
        print(f"    PASS: training continues correctly after round trip (loss now {loss.item():.6f})")

    print("[3] reset_states / free_states (must also remove hooks):")
    handle.reset_states()
    all_zero = all(torch.count_nonzero(t) == 0 for t in handle.states[0].values())
    if not all_zero:
        failures.append(f"[{algorithm_name}] reset_states did not zero all state")
        print("    FAIL: state not fully zeroed")
    else:
        print("    reset_states(): all state correctly zeroed")

    n_hooks_before = len(handle._hooks)
    handle.free_states()
    if handle.states != [] or handle._hooks != []:
        failures.append(f"[{algorithm_name}] free_states did not clear states and/or hooks "
                         f"(states={handle.states}, hooks left={len(handle._hooks)})")
        print(f"    FAIL: states={handle.states}, hooks left={len(handle._hooks)}")
    else:
        print(f"    free_states(): states and all {n_hooks_before} hook(s) correctly cleared")

    # A backward() after free_states() must not touch W -- the hook should
    # be gone, not merely inert.
    W_snapshot = W.detach().clone()
    x = torch.randn(6, 10)
    loss = ((W @ x - true_W @ x) ** 2).mean()
    loss.backward()
    if not torch.equal(W.detach(), W_snapshot):
        failures.append(f"[{algorithm_name}] parameter still mutated by a backward() call "
                         f"after free_states() -- hook wasn't actually removed")
        print("    FAIL: W changed after free_states(), hook still active")
    else:
        print("    PASS: backward() after free_states() no longer touches W")

    return failures


def run_node_build_check(node_cls, name: str) -> list:
    W = (torch.randn(4, 6) * 0.1).requires_grad_(True)
    node = node_cls()
    result = node.build(params=ParameterList([W]), lr=0.01, device=DEVICE)
    handle = result["optimizer"]
    ok = isinstance(handle, FusedOptimizerHandle) and len(handle._hooks) == 1
    print(f"  {'PASS' if ok else 'FAIL'}: {name}.build() -> FusedOptimizerHandle with hooks registered")
    handle.free_states()
    if ok:
        return []
    return [f"{name}.build() did not produce a correctly hooked-up FusedOptimizerHandle"]


def main():
    print(f"Device: {DEVICE}")
    all_failures = []
    for algorithm_name in _ALGORITHM_FACTORIES:
        all_failures.extend(run_lifecycle_for(algorithm_name))

    print(f"\n{'#'*60}\n# Node build() checks\n{'#'*60}")
    all_failures.extend(run_node_build_check(ComposedFusedAdafactorOptimizerNode, "ComposedFusedAdafactorOptimizerNode"))
    all_failures.extend(run_node_build_check(ComposedFusedCAMEOptimizerNode, "ComposedFusedCAMEOptimizerNode"))
    all_failures.extend(run_node_build_check(ComposedFusedAdamWOptimizerNode, "ComposedFusedAdamWOptimizerNode"))

    print("\n" + "=" * 60)
    if all_failures:
        print(f"SMOKE TEST: {len(all_failures)} FAILURE(S):")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"SMOKE TEST: ALL CHECKS PASSED (algorithms: {list(_ALGORITHM_FACTORIES)}, device={DEVICE})")


if __name__ == "__main__":
    main()
