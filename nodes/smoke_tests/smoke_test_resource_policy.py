"""Correctness check for ResourceBudget/ResourcePolicy/ManualResourcePolicy
(nodes/resource_policy.py) and its wiring into ComfyUNetLoRANode and the
three Composed*OptimizerNode classes.

Three things checked, in order:
  1. resource_policy.py's own contract, standalone (no torch needed).
  2. ComfyUNetLoRANode declares a resource_policy port, contract-level
     only (building a real ComfyUNetTrainableModel needs a real UNet
     state dict, out of scope for a smoke test -- see
     smoke_test_train_contracts.py for the same declaration-only pattern
     applied to SupervisedLoRATrainerNode).
  3. group_policy, newly exposed on ComposedCAMEOptimizerNode/
     ComposedAdamWOptimizerNode/ComposedAdafactorOptimizerNode: None
     reproduces the pre-existing UniformGroups default exactly; an
     explicit LoRAPlusGroups is actually threaded through to the
     resulting handle's param_lr -- built and exercised end to end with
     real torch tensors, the same discipline
     smoke_test_parameter_group_policy.py uses for ComposedOptimizerHandle
     directly.

Run this directly: `python nodes/smoke_tests/smoke_test_resource_policy.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from nodes.optimizer.composed import LoRAPlusGroups, ParameterGroupPolicy, UniformGroups
from nodes.optimizer.composed_adafactor import ComposedAdafactorOptimizerNode
from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
from nodes.optimizer.composed_came import ComposedCAMEOptimizerNode
from nodes.resource_policy import ManualResourcePolicy, ResourceBudget, ResourcePolicy

DEVICE = "cpu"
failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def _params():
    torch.manual_seed(0)
    return [torch.randn(4, 4, requires_grad=True) for _ in range(3)]


# --- 1. resource_policy.py's own contract -----------------------------

def check_resource_budget_is_a_frozen_value_object():
    print("\n=== ResourceBudget ===")
    budget = ResourceBudget(vram_budget_mb=8000.0)
    record(budget.vram_budget_mb == 8000.0, "vram_budget_mb stored")
    record(budget.vram_reserve_mb == 512.0, "vram_reserve_mb defaults to 512.0")
    try:
        budget.vram_budget_mb = 1.0
        ok = False
    except Exception:
        ok = True
    record(ok, "ResourceBudget is frozen (assignment raises)")


def check_resource_policy_is_a_real_abc():
    print("\n=== ResourcePolicy ABC is enforced ===")

    class Incomplete(ResourcePolicy):
        def checkpointing_strategy(self):
            return None

    try:
        Incomplete()
        ok = False
    except TypeError:
        ok = True
    record(ok, "can't instantiate a ResourcePolicy missing lora_scaling_policy/"
               "parameter_group_policy")


def check_manual_resource_policy_is_a_pure_carrier():
    print("\n=== ManualResourcePolicy stores and returns exactly what it's given ===")
    checkpointing = object()
    scaling = object()
    group_policy = object()
    policy = ManualResourcePolicy(
        checkpointing=checkpointing,
        lora_scaling_policy=scaling,
        parameter_group_policy=group_policy,
    )
    record(policy.checkpointing_strategy() is checkpointing,
           "checkpointing_strategy() returns the exact object given")
    record(policy.lora_scaling_policy() is scaling,
           "lora_scaling_policy() returns the exact object given")
    record(policy.parameter_group_policy() is group_policy,
           "parameter_group_policy() returns the exact object given")


# --- 2. ComfyUNetLoRANode contract -------------------------------------

def check_comfy_unet_lora_node_declares_resource_policy_port():
    print("\n=== ComfyUNetLoRANode.INPUTS declares resource_policy ===")
    from nodes.model.lora_injector import ComfyUNetLoRANode

    port = ComfyUNetLoRANode.INPUTS.get("resource_policy")
    record(port is not None, "resource_policy port exists")
    if port is not None:
        record(port.required is False, "resource_policy is optional")
        record(port.default is None, "resource_policy defaults to None")
        record(port.type is ResourcePolicy, "resource_policy is typed as ResourcePolicy",
               detail=str(port.type))
    # use_checkpoint/scaling_policy still exist unchanged alongside it --
    # the new port is additive, not a replacement.
    record("use_checkpoint" in ComfyUNetLoRANode.INPUTS,
           "use_checkpoint port still exists")
    record("scaling_policy" in ComfyUNetLoRANode.INPUTS,
           "scaling_policy port still exists")


# --- 3. group_policy on the three Composed*OptimizerNode classes ------

def check_group_policy_port_declared(node_cls, name: str):
    port = node_cls.INPUTS.get("group_policy")
    record(port is not None, f"{name}.INPUTS declares group_policy")
    if port is not None:
        record(port.required is False, f"{name}: group_policy is optional")
        record(port.default is None, f"{name}: group_policy defaults to None")
        record(port.type is ParameterGroupPolicy,
               f"{name}: group_policy is typed as ParameterGroupPolicy")


def check_group_policy_default_reproduces_uniform_groups(node_cls, name: str,
                                                           **extra_inputs):
    """None (the default) must still produce exactly [lr]*len(params) --
    the exact pre-existing behavior, unchanged by this port's addition."""
    params = _params()
    node = node_cls()
    result = node.build(params=params, lr=1e-3, device=DEVICE, **extra_inputs)
    handle = result["optimizer"]
    record(handle.param_lr == [1e-3] * len(params),
           f"{name}: group_policy=None reproduces UniformGroups exactly",
           detail=str(handle.param_lr))


def check_group_policy_explicit_lora_plus_threads_through(node_cls, name: str,
                                                            **extra_inputs):
    params = _params()
    is_b_matrix = lambda p: p is params[1]  # noqa: E731
    node = node_cls()
    result = node.build(params=params, lr=1e-3, device=DEVICE,
                         group_policy=LoRAPlusGroups(is_b_matrix, ratio=16.0),
                         **extra_inputs)
    handle = result["optimizer"]
    expected = [1e-3, 1e-3 * 16.0, 1e-3]
    record(handle.param_lr == expected,
           f"{name}: explicit LoRAPlusGroups threads through to param_lr",
           detail=str(handle.param_lr))


def main():
    print(f"Device: {DEVICE}")

    check_resource_budget_is_a_frozen_value_object()
    check_resource_policy_is_a_real_abc()
    check_manual_resource_policy_is_a_pure_carrier()

    check_comfy_unet_lora_node_declares_resource_policy_port()

    print("\n=== group_policy: declared on all three Composed*OptimizerNode classes ===")
    check_group_policy_port_declared(ComposedCAMEOptimizerNode, "ComposedCAMEOptimizerNode")
    check_group_policy_port_declared(ComposedAdamWOptimizerNode, "ComposedAdamWOptimizerNode")
    check_group_policy_port_declared(ComposedAdafactorOptimizerNode,
                                      "ComposedAdafactorOptimizerNode")

    print("\n=== group_policy=None: exact pre-existing UniformGroups behavior ===")
    check_group_policy_default_reproduces_uniform_groups(
        ComposedCAMEOptimizerNode, "ComposedCAMEOptimizerNode")
    check_group_policy_default_reproduces_uniform_groups(
        ComposedAdamWOptimizerNode, "ComposedAdamWOptimizerNode")
    check_group_policy_default_reproduces_uniform_groups(
        ComposedAdafactorOptimizerNode, "ComposedAdafactorOptimizerNode")

    print("\n=== group_policy=LoRAPlusGroups(...): actually applied ===")
    check_group_policy_explicit_lora_plus_threads_through(
        ComposedCAMEOptimizerNode, "ComposedCAMEOptimizerNode")
    check_group_policy_explicit_lora_plus_threads_through(
        ComposedAdamWOptimizerNode, "ComposedAdamWOptimizerNode")
    check_group_policy_explicit_lora_plus_threads_through(
        ComposedAdafactorOptimizerNode, "ComposedAdafactorOptimizerNode")

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
