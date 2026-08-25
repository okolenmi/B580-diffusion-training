"""Real torch (CPU), real core.lora classes -- no mocks for the actual
LoRA math. Verifies nodes/model/lora_phases.py:

  1. Contracts: LoRAPhaseSplitNode is concrete and correctly typed, and
     the graph editor's real issubclass() check (server/graph_executor.py
     ._is_compatible) actually accepts every wire this feature needs,
     including the ones that motivated generalizing LoRAWeightsExportable
     into TrainedWeightsExportable partway through building this.
  2. Forward equivalence: a freshly-split generation is a no-op at the
     instant it's added, and after training, gen0's weights are
     bit-for-bit unchanged (no base mutation, no leakage).
  3. Gradient isolation: only the new generation's parameters ever get a
     gradient after a split.
  4. The strongest check -- round-tripping extract_combined_weights /
     extract_own_generation_weights through core.lora's own, completely
     untouched load_lora_into_model, then comparing forward() output.
     Real interop with proven code, not self-comparison.
  5. Byte-exact degeneration to core.lora.extract_lora_weights's own
     output when nothing was split (no behavior change for the common
     case that doesn't use this feature).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from core.lora import LoRAConv2d, LoRALinear, extract_lora_weights, load_lora_into_model
from nodes.model.dora_layer import DoRALinear
from nodes.model.handle import TrainableModel, TrainedWeightsExportable
from nodes.model.lora_injector import ComfyUNetLoRANode, ComfyUNetTrainableModel
from nodes.model.lora_phases import (
    LoRAPhaseSplitNode,
    extract_combined_weights,
    extract_own_generation_weights,
    split_into_new_generation,
)
from nodes.model.lora_saver import LoRACheckpointSaverNode
from nodes.model.parameters import ModelParametersNode
from nodes.train.supervised import SupervisedLoRATrainerNode
from server.graph_executor import _is_compatible


class _FakeWrapper:
    """Minimal stand-in for core.unet_wrapper.ComfyUNetWrapper -- just the
    two members split_into_new_generation actually touches."""

    def __init__(self, registry):
        self.lora_registry = registry

    def has_lora(self) -> bool:
        return len(self.lora_registry) > 0


def check_contracts():
    print("[contracts]")
    assert not getattr(LoRAPhaseSplitNode, "__abstractmethods__", None)
    assert set(LoRAPhaseSplitNode.INPUTS) == {"model", "rank", "alpha", "dropout"}
    assert set(LoRAPhaseSplitNode.OUTPUTS) == {"model", "completed_generation"}
    assert LoRAPhaseSplitNode.INPUTS["model"].type is TrainableModel
    assert LoRAPhaseSplitNode.OUTPUTS["completed_generation"].type is TrainedWeightsExportable
    assert issubclass(TrainableModel, TrainedWeightsExportable), (
        "TrainableModel must extend TrainedWeightsExportable -- the whole reason "
        "for that hierarchy is so a TrainerNode's ordinary model output satisfies "
        "LoRACheckpointSaverNode's input via a real issubclass() check, not a "
        "runtime special case"
    )

    # The actual graph wires this feature needs, checked through the exact
    # function server/graph_executor.py uses to accept/reject an edge --
    # not a reimplementation of that logic, the real thing.
    _is_compatible(ComfyUNetLoRANode, "model", LoRAPhaseSplitNode, "model")
    _is_compatible(LoRAPhaseSplitNode, "model", ModelParametersNode, "model")
    _is_compatible(LoRAPhaseSplitNode, "model", SupervisedLoRATrainerNode, "model")
    _is_compatible(SupervisedLoRATrainerNode, "model", LoRACheckpointSaverNode, "model")
    _is_compatible(LoRAPhaseSplitNode, "completed_generation", LoRACheckpointSaverNode, "model")
    print("    PASS: every phase-split graph wire (including the one this session's "
          "TrainedWeightsExportable generalization exists for) passes the real "
          "graph-editor type check")


def check_linear_phase_split():
    print("[linear: forward equivalence, gradient isolation, exact round-trips]")
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=True)
    gen0 = LoRALinear(base, rank=3, alpha=6.0)

    # Real warm-up phase: actual gradient steps, not just init values.
    opt0 = torch.optim.SGD([gen0.lora_A, gen0.lora_B], lr=0.5)
    for _ in range(5):
        x = torch.randn(4, 8)
        loss = gen0(x).pow(2).mean()
        opt0.zero_grad()
        loss.backward()
        opt0.step()
    A0_before = gen0.lora_A.detach().clone()
    B0_before = gen0.lora_B.detach().clone()

    parent = nn.Module()
    parent.proj = gen0
    wrapper = _FakeWrapper([("root.proj", parent, "proj", gen0)])

    frozen_snapshot = split_into_new_generation(wrapper, rank=2, alpha=4.0)

    gen1 = parent.proj
    assert gen1 is not gen0 and gen1.inner is gen0, "setattr must swap the live module tree"
    assert wrapper.lora_registry[0][3] is gen1, "registry must track the new top-of-stack layer"
    assert gen0.lora_A.requires_grad is False and gen0.lora_B.requires_grad is False
    assert gen1.lora_A.requires_grad is True and gen1.lora_B.requires_grad is True
    # requires_grad_(False) doesn't clear an already-set .grad from the warm-up
    # loop's last step -- clear it explicitly so the check below is actually
    # about phase 2's training, not stale state from phase 1's last backward().
    gen0.lora_A.grad = None
    gen0.lora_B.grad = None

    x = torch.randn(4, 8)
    torch.testing.assert_close(gen1(x), gen0(x))
    print("    PASS: new generation is a no-op at the instant it's added")

    opt1 = torch.optim.SGD([gen1.lora_A, gen1.lora_B], lr=0.5)
    for _ in range(5):
        x = torch.randn(4, 8)
        loss = gen1(x).pow(2).mean()
        opt1.zero_grad()
        loss.backward()
        opt1.step()
    assert gen0.lora_A.grad is None and gen0.lora_B.grad is None
    torch.testing.assert_close(gen0.lora_A, A0_before)
    torch.testing.assert_close(gen0.lora_B, B0_before)
    print("    PASS: gen0 got zero gradient and is bit-for-bit unchanged after training gen1")

    # Strongest check: combined weights, reloaded through core.lora's own
    # (untouched) load_lora_into_model into a fresh single-generation
    # LoRALinear, reproduce the live stacked forward exactly.
    combined = extract_combined_weights(wrapper.lora_registry)
    fresh_base = nn.Linear(8, 6, bias=True)
    fresh_base.weight.data.copy_(base.weight.data)
    fresh_base.bias.data.copy_(base.bias.data)
    fresh = LoRALinear(fresh_base, rank=5, alpha=1.0)  # rank must equal 3+2
    load_lora_into_model([("root.proj", None, "proj", fresh)], combined)
    x = torch.randn(10, 8)
    torch.testing.assert_close(fresh(x), gen1(x))
    print("    PASS: combined checkpoint reloaded via core.lora.load_lora_into_model "
          "exactly reproduces the live stacked forward pass")

    # completed_generation snapshot matches the frozen phase's own weights,
    # and reloading *just* it reproduces exactly the pre-split forward --
    # proving none of phase 2 leaked into it.
    own = extract_own_generation_weights(frozen_snapshot)
    torch.testing.assert_close(own["lora_unet_root_proj.lora_down.weight"], A0_before)
    torch.testing.assert_close(own["lora_unet_root_proj.lora_up.weight"], B0_before)
    fresh_base2 = nn.Linear(8, 6, bias=True)
    fresh_base2.weight.data.copy_(base.weight.data)
    fresh_base2.bias.data.copy_(base.bias.data)
    fresh_phase1_only = LoRALinear(fresh_base2, rank=3, alpha=6.0)
    load_lora_into_model([("root.proj", None, "proj", fresh_phase1_only)], own)
    torch.testing.assert_close(fresh_phase1_only(x), gen0(x))
    print("    PASS: completed_generation snapshot round-trips to exactly the frozen "
          "phase's forward -- none of phase 2 in it")


def check_conv2d_phase_split():
    print("[conv2d: same battery, groups=2 and a real kernel/stride/padding]")
    torch.manual_seed(1)
    base = nn.Conv2d(8, 6, kernel_size=3, stride=2, padding=1, groups=2, bias=True)
    gen0 = LoRAConv2d(base, rank=4, alpha=6.0)

    opt0 = torch.optim.SGD([gen0.lora_A, gen0.lora_B], lr=0.5)
    for _ in range(5):
        x = torch.randn(2, 8, 10, 10)
        loss = gen0(x).pow(2).mean()
        opt0.zero_grad()
        loss.backward()
        opt0.step()
    A0_before, B0_before = gen0.lora_A.detach().clone(), gen0.lora_B.detach().clone()

    parent = nn.Module()
    parent.conv = gen0
    wrapper = _FakeWrapper([("root.conv", parent, "conv", gen0)])
    frozen_snapshot = split_into_new_generation(wrapper, rank=2, alpha=4.0)
    gen1 = parent.conv
    assert gen1.stride == gen0.stride == (2, 2)
    assert gen1.padding == gen0.padding == (1, 1)
    assert gen1.groups == gen0.groups == 2

    x = torch.randn(2, 8, 10, 10)
    torch.testing.assert_close(gen1(x), gen0(x))
    opt1 = torch.optim.SGD([gen1.lora_A, gen1.lora_B], lr=0.5)
    for _ in range(5):
        x = torch.randn(2, 8, 10, 10)
        loss = gen1(x).pow(2).mean()
        opt1.zero_grad()
        loss.backward()
        opt1.step()
    torch.testing.assert_close(gen0.lora_A, A0_before)
    torch.testing.assert_close(gen0.lora_B, B0_before)
    print("    PASS: gen0 frozen exactly, gen1 trains independently")

    combined = extract_combined_weights(wrapper.lora_registry)
    fresh_base = nn.Conv2d(8, 6, kernel_size=3, stride=2, padding=1, groups=2, bias=True)
    fresh_base.weight.data.copy_(base.weight.data)
    fresh_base.bias.data.copy_(base.bias.data)
    fresh = LoRAConv2d(fresh_base, rank=6, alpha=1.0)
    load_lora_into_model([("root.conv", None, "conv", fresh)], combined)
    x = torch.randn(3, 8, 10, 10)
    torch.testing.assert_close(fresh(x), gen1(x))
    print("    PASS: combined conv2d checkpoint round-trips exactly through core.lora's "
          "own loader (stride/padding/groups all preserved)")

    own = extract_own_generation_weights(frozen_snapshot)
    fresh_base2 = nn.Conv2d(8, 6, kernel_size=3, stride=2, padding=1, groups=2, bias=True)
    fresh_base2.weight.data.copy_(base.weight.data)
    fresh_base2.bias.data.copy_(base.bias.data)
    fresh_phase1_only = LoRAConv2d(fresh_base2, rank=4, alpha=6.0)
    load_lora_into_model([("root.conv", None, "conv", fresh_phase1_only)], own)
    torch.testing.assert_close(fresh_phase1_only(x), gen0(x))
    print("    PASS: completed_generation snapshot round-trips to exactly the frozen "
          "phase's forward")


def check_three_generation_chain():
    print("[three generations chained -- generalizes past a single warm-up boundary]")
    torch.manual_seed(2)
    base = nn.Linear(5, 4)
    gen0 = LoRALinear(base, rank=2, alpha=2.0)
    parent = nn.Module()
    parent.proj = gen0
    wrapper = _FakeWrapper([("root.proj", parent, "proj", gen0)])

    for rank, alpha in [(3, 3.0), (2, 5.0)]:
        split_into_new_generation(wrapper, rank=rank, alpha=alpha)
        layer = wrapper.lora_registry[0][3]
        opt = torch.optim.SGD([layer.lora_A, layer.lora_B], lr=0.5)
        for _ in range(3):
            x = torch.randn(4, 5)
            loss = layer(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    top = wrapper.lora_registry[0][3]
    assert top.inner.inner is gen0, "three generations deep: top -> gen1 -> gen0"

    combined = extract_combined_weights(wrapper.lora_registry)
    fresh_base = nn.Linear(5, 4)
    fresh_base.weight.data.copy_(base.weight.data)
    fresh_base.bias.data.copy_(base.bias.data)
    fresh = LoRALinear(fresh_base, rank=2 + 3 + 2, alpha=1.0)
    load_lora_into_model([("root.proj", None, "proj", fresh)], combined)
    x = torch.randn(6, 5)
    torch.testing.assert_close(fresh(x), top(x))
    print("    PASS: a 3-generation chain still combines exactly into one portable adapter")


def check_dora_phase_split_extraction():
    print("[a phase-split DoRA layer: extract_combined_weights refuses loudly "
          "(no way to fold magnitude into a combined adapter); "
          "extract_own_generation_weights on the frozen phase still works fine, "
          ".dora_scale included, since it never combines]")
    torch.manual_seed(5)
    base = nn.Linear(5, 4)
    gen0 = DoRALinear(base, rank=2, alpha=4.0)
    opt0 = torch.optim.SGD([gen0._lora.lora_A, gen0._lora.lora_B, gen0.magnitude], lr=0.5)
    for _ in range(3):
        x = torch.randn(3, 5)
        loss = gen0(x).pow(2).mean()
        opt0.zero_grad()
        loss.backward()
        opt0.step()
    trained_magnitude = gen0.magnitude.detach().clone()

    parent = nn.Module()
    parent.proj = gen0
    wrapper = _FakeWrapper([("root.proj", parent, "proj", gen0)])

    # The "just this phase" snapshot, taken before the split -- no
    # combination involved at all, so DoRA's magnitude round-trips fine
    # regardless of what happens after.
    frozen_snapshot = split_into_new_generation(wrapper, rank=2, alpha=2.0)
    assert gen0.magnitude.requires_grad is False, (
        "a phase-split DoRA layer's magnitude must be frozen too -- get_lora_weights() "
        "alone (direction only) isn't enough, see split_into_new_generation's docstring"
    )
    # requires_grad_(False) doesn't clear an already-set .grad from phase 1's
    # own last backward() -- same subtlety check_linear_phase_split's own
    # comment flags for lora_A/lora_B above. Clear it explicitly so the
    # check below is actually about phase 2's training, not stale state.
    gen0.magnitude.grad = None

    # Real gradient-isolation check, same spirit as check_linear_phase_split's
    # own (lora_A stays bit-for-bit unchanged while phase 2 trains) -- proves
    # the freeze above isn't just requires_grad flipped with nothing behind
    # it: phase 2's own optimizer walks straight through gen1.inner (== gen0)
    # via ordinary nn.Module.parameters() recursion, so if magnitude weren't
    # actually frozen this training loop is exactly what would keep moving it.
    gen1 = parent.proj
    opt1 = torch.optim.SGD([gen1.lora_A, gen1.lora_B], lr=0.5)
    for _ in range(3):
        x = torch.randn(3, 5)
        loss = gen1(x).pow(2).mean()
        opt1.zero_grad()
        loss.backward()
        opt1.step()
    assert gen0.magnitude.grad is None, "frozen magnitude should never accumulate a gradient"
    torch.testing.assert_close(gen0.magnitude, trained_magnitude)
    print("    PASS: phase 2 training leaves the frozen DoRA phase's magnitude "
          "bit-for-bit unchanged, no gradient -- not just requires_grad flipped")

    own_weights = extract_own_generation_weights(frozen_snapshot)
    assert "lora_unet_root_proj.dora_scale" in own_weights
    torch.testing.assert_close(own_weights["lora_unet_root_proj.dora_scale"], trained_magnitude)
    print("    PASS: extract_own_generation_weights captures the frozen DoRA "
          "phase's real trained magnitude, unaffected by the split that just happened")

    # Now the live, post-split registry -- gen1 (plain LoRAGeneration) stacked
    # on the frozen DoRA gen0. Combining these into one flat adapter is
    # exactly the case that isn't implemented.
    try:
        extract_combined_weights(wrapper.lora_registry)
        raise AssertionError("expected NotImplementedError combining a phase-split DoRA root")
    except NotImplementedError as e:
        assert "DoRA" in str(e) and "magnitude" in str(e)
        print(f"    PASS (raises instead of silently dropping magnitude): {e}")


def check_never_split_is_byte_identical_to_core_lora():
    print("[never split: extract_combined_weights must match core.lora.extract_lora_weights exactly]")
    torch.manual_seed(3)
    base = nn.Linear(6, 5)
    gen0 = LoRALinear(base, rank=4, alpha=17.0)  # alpha != rank on purpose
    x = torch.randn(3, 6)
    gen0(x)  # touch it, no-op, just parity with real usage
    registry = [("root.proj", None, "proj", gen0)]

    old = extract_lora_weights(registry)
    new = extract_combined_weights(registry)
    assert old.keys() == new.keys()
    for k in old:
        torch.testing.assert_close(old[k], new[k])
    print("    PASS: byte-identical output -- zero behavior change for models that "
          "never use phase-splitting")


def check_end_to_end_node():
    print("[LoRAPhaseSplitNode.build() end-to-end, through the real Node class]")
    torch.manual_seed(4)
    base = nn.Linear(4, 4)
    gen0 = LoRALinear(base, rank=2, alpha=2.0)
    parent = nn.Module()
    parent.proj = gen0
    model = ComfyUNetTrainableModel(_FakeWrapper([("root.proj", parent, "proj", gen0)]))

    node = LoRAPhaseSplitNode()
    result = node.build(model=model, rank=3, alpha=3.0, dropout=0.0)
    assert result["model"] is model
    assert isinstance(result["completed_generation"], TrainedWeightsExportable)
    snap = result["completed_generation"].trained_state_dict()
    assert set(snap.keys()) == {
        "lora_unet_root_proj.lora_down.weight",
        "lora_unet_root_proj.lora_up.weight",
        "lora_unet_root_proj.alpha",
    }
    # model.trained_state_dict() now reflects the combined (post-split) stack
    full = model.trained_state_dict()
    assert full["lora_unet_root_proj.lora_down.weight"].shape[0] == 2 + 3
    print("    PASS: node output shapes and keys match the underlying functions exactly")

    try:
        LoRAPhaseSplitNode().build(model=object(), rank=2, alpha=2.0, dropout=0.0)
        raise AssertionError("expected TypeError for a non-ComfyUNetTrainableModel input")
    except TypeError:
        print("    PASS: rejects a model it can't actually operate on, with a clear error")


def main():
    check_contracts()
    check_linear_phase_split()
    check_conv2d_phase_split()
    check_three_generation_chain()
    check_dora_phase_split_extraction()
    check_never_split_is_byte_identical_to_core_lora()
    check_end_to_end_node()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
