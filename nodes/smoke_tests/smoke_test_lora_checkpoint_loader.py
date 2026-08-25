"""Real torch, real core.lora classes, real safetensors I/O against a
temp directory (paths.set_loras_dir) -- verifies nodes/model/
lora_checkpoint_loader.py end to end: save a trained LoRA with the real
extraction path, load it into a fresh injection, forward output matches
the original exactly.

Also covers DoRALinear/DoRAConv2d specifically (nodes/model/dora_layer.py,
DoRAAdapter) -- added alongside the existing plain-LoRA checks below,
not a separate file, since this is still exactly the same node under
test. Real, closed-here gap: before this, this file (and every other
smoke test) exercised LoRACheckpointLoaderNode against plain
LoRALinear/LoRAConv2d layers only -- a DoRA-adapted layer was silently
skipped by core.lora.load_lora_into_model's isinstance gate this whole
time (LoRACheckpointLoaderNode's own missing-key/rank-mismatch
validation loop had the same gate, so it never even got a chance to
catch a DoRA-layer problem), and nothing here would have noticed. See
lora_checkpoint_loader.py's own module docstring for the fix.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

import paths
from core.lora import LoRALinear, extract_lora_weights
from nodes.model.dora_layer import DoRALinear
from nodes.model.lora_checkpoint_loader import LoRACheckpointLoaderNode
from nodes.model.lora_injector import ComfyUNetTrainableModel
from nodes.model.lora_phases import extract_own_generation_weights


class _FakeWrapper:
    def __init__(self, registry):
        self.lora_registry = registry

    def has_lora(self) -> bool:
        return len(self.lora_registry) > 0


def check_contracts():
    print("[contracts]")
    assert not getattr(LoRACheckpointLoaderNode, "__abstractmethods__", None)
    assert set(LoRACheckpointLoaderNode.INPUTS) == {"model", "relative_path", "project_layout"}
    assert set(LoRACheckpointLoaderNode.OUTPUTS) == {"model"}
    print("    PASS")


def _make_model(seed, rank=4, alpha=6.0, in_f=8, out_f=6):
    torch.manual_seed(seed)
    base = nn.Linear(in_f, out_f)
    layer = LoRALinear(base, rank=rank, alpha=alpha)
    parent = nn.Module()
    parent.proj = layer
    model = ComfyUNetTrainableModel(_FakeWrapper([("root.proj", parent, "proj", layer)]))
    return model, base, layer


def _make_dora_model(seed, rank=4, alpha=6.0, in_f=8, out_f=6):
    torch.manual_seed(seed)
    base = nn.Linear(in_f, out_f)
    layer = DoRALinear(base, rank=rank, alpha=alpha)
    parent = nn.Module()
    parent.proj = layer
    model = ComfyUNetTrainableModel(_FakeWrapper([("root.proj", parent, "proj", layer)]))
    return model, base, layer


def _train_a_few_steps(layer, params, in_f=8):
    opt = torch.optim.SGD(params, lr=0.5)
    for _ in range(5):
        x = torch.randn(4, in_f)
        loss = layer(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def check_load_reproduces_the_saved_forward_exactly(tmpdir):
    print("[loading reproduces the saved model's forward exactly]")
    trained_model, base, trained_layer = _make_model(seed=0)
    opt = torch.optim.SGD([trained_layer.lora_A, trained_layer.lora_B], lr=0.5)
    for _ in range(5):
        x = torch.randn(4, 8)
        loss = trained_layer(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    from safetensors.torch import save_file
    state_dict = extract_lora_weights(trained_model.raw.lora_registry)
    save_path = tmpdir / "test_lora.safetensors"
    save_file(state_dict, str(save_path))

    fresh_model, fresh_base, fresh_layer = _make_model(seed=999)  # different init on purpose
    fresh_base.weight.data.copy_(base.weight.data)
    fresh_base.bias.data.copy_(base.bias.data)

    node = LoRACheckpointLoaderNode()
    result = node.build(model=fresh_model, relative_path="test_lora.safetensors")
    assert result["model"] is fresh_model

    x = torch.randn(10, 8)
    torch.testing.assert_close(fresh_layer(x), trained_layer(x))
    print("    PASS: forward output matches the originally-trained model exactly")


def check_dora_round_trip_with_magnitude_reproduces_exactly(tmpdir):
    print("[DoRA: save+load restores magnitude exactly (not recomputed), "
          "forward matches the trained model bit-for-bit]")
    trained_model, base, trained_layer = _make_dora_model(seed=0)
    _train_a_few_steps(trained_layer, [trained_layer._lora.lora_A,
                                        trained_layer._lora.lora_B,
                                        trained_layer.magnitude])

    from safetensors.torch import save_file
    state_dict = extract_own_generation_weights(trained_model.raw.lora_registry)
    assert any(k.endswith(".dora_scale") for k in state_dict), (
        "extract_own_generation_weights must include a .dora_scale key for a "
        "DoRALinear layer -- see nodes/model/lora_phases.py"
    )
    save_path = tmpdir / "test_dora.safetensors"
    save_file(state_dict, str(save_path))

    fresh_model, fresh_base, fresh_layer = _make_dora_model(seed=999)  # different init
    fresh_base.weight.data.copy_(base.weight.data)
    fresh_base.bias.data.copy_(base.bias.data)
    assert not torch.allclose(fresh_layer.magnitude, trained_layer.magnitude), (
        "test fixture problem -- fresh and trained magnitude already coincide, "
        "so a passing round-trip check below wouldn't actually prove anything"
    )

    node = LoRACheckpointLoaderNode()
    result = node.build(model=fresh_model, relative_path="test_dora.safetensors")
    assert result["model"] is fresh_model

    torch.testing.assert_close(fresh_layer.magnitude, trained_layer.magnitude)
    x = torch.randn(10, 8)
    torch.testing.assert_close(fresh_layer(x), trained_layer(x))
    print("    PASS: magnitude restored exactly, forward output matches the "
          "originally-trained model exactly")


def check_dora_without_magnitude_key_recomputes_it(tmpdir):
    print("[DoRA: a checkpoint with no .dora_scale key still loads -- direction "
          "only, magnitude recomputed fresh, same as load_lora_weights() directly]")
    trained_model, base, trained_layer = _make_dora_model(seed=1)
    _train_a_few_steps(trained_layer, [trained_layer._lora.lora_A, trained_layer._lora.lora_B,
                                        trained_layer.magnitude])

    from safetensors.torch import save_file
    state_dict = extract_own_generation_weights(trained_model.raw.lora_registry)
    direction_only = {k: v for k, v in state_dict.items() if not k.endswith(".dora_scale")}
    assert len(direction_only) == len(state_dict) - 1
    save_path = tmpdir / "test_dora_direction_only.safetensors"
    save_file(direction_only, str(save_path))

    fresh_model, fresh_base, fresh_layer = _make_dora_model(seed=999)
    fresh_base.weight.data.copy_(base.weight.data)
    fresh_base.bias.data.copy_(base.bias.data)

    LoRACheckpointLoaderNode().build(model=fresh_model, relative_path=str(save_path.name))

    # Independent reference: construct a second fresh layer and call
    # load_lora_weights() directly -- both paths are the same deterministic
    # function of (base_weight, loaded A, loaded B, scaling), so they must
    # land on the same recomputed magnitude if the loader's fallback really
    # is calling that path and not, say, leaving magnitude untouched at its
    # own fresh init value.
    reference_model, reference_base, reference_layer = _make_dora_model(seed=999)
    reference_base.weight.data.copy_(base.weight.data)
    reference_base.bias.data.copy_(base.bias.data)
    reference_layer.load_lora_weights(trained_layer._lora.lora_A.detach(),
                                       trained_layer._lora.lora_B.detach())
    torch.testing.assert_close(fresh_layer.magnitude, reference_layer.magnitude)
    print("    PASS: recomputed magnitude matches calling load_lora_weights() directly")


def check_dora_alpha_restore_affects_forward(tmpdir):
    print("[DoRA: alpha/scaling restored from the checkpoint, not left at the "
          "fresh injection's own config -- verified by actual forward output, "
          "not just reading .alpha back]")
    trained_model, base, trained_layer = _make_dora_model(seed=2, alpha=6.0)
    _train_a_few_steps(trained_layer, [trained_layer._lora.lora_A, trained_layer._lora.lora_B,
                                        trained_layer.magnitude])

    from safetensors.torch import save_file
    state_dict = extract_own_generation_weights(trained_model.raw.lora_registry)
    save_path = tmpdir / "test_dora_alpha.safetensors"
    save_file(state_dict, str(save_path))

    # Same rank (a rank mismatch is caught earlier and would mask this check),
    # deliberately different alpha -- if restore_alpha() weren't called, this
    # layer's own scaling (from alpha=2.0) would stay in forward()'s math and
    # the output below would NOT match the originally-trained model's.
    fresh_model, fresh_base, fresh_layer = _make_dora_model(seed=999, alpha=2.0)
    fresh_base.weight.data.copy_(base.weight.data)
    fresh_base.bias.data.copy_(base.bias.data)
    assert fresh_layer.alpha != trained_layer.alpha

    LoRACheckpointLoaderNode().build(model=fresh_model, relative_path=str(save_path.name))

    assert fresh_layer.alpha == trained_layer.alpha
    x = torch.randn(10, 8)
    torch.testing.assert_close(fresh_layer(x), trained_layer(x))
    print("    PASS: alpha restored, and forward output matches exactly -- "
          "proves scaling was actually recomputed, not just .alpha updated")


def check_dora_missing_keys_and_rank_mismatch_are_caught(tmpdir):
    print("[DoRA: missing-key / rank-mismatch validation covers DoRA layers too "
          "-- previously silently skipped by the same isinstance gate that used "
          "to skip loading them at all]")
    from safetensors.torch import save_file
    save_file({"lora_unet_some_other_layer.lora_down.weight": torch.zeros(2, 8),
                "lora_unet_some_other_layer.lora_up.weight": torch.zeros(6, 2),
                "lora_unet_some_other_layer.alpha": torch.tensor([2.0])},
               str(tmpdir / "dora_wrong_layers.safetensors"))
    model, _, _ = _make_dora_model(seed=3)
    try:
        LoRACheckpointLoaderNode().build(model=model, relative_path="dora_wrong_layers.safetensors")
        raise AssertionError("expected ValueError for missing keys")
    except ValueError as e:
        assert "missing" in str(e).lower()
        print(f"    PASS (missing keys): {e}")

    trained_model, _, trained_layer = _make_dora_model(seed=4, rank=4)
    state_dict = extract_own_generation_weights(trained_model.raw.lora_registry)
    save_file(state_dict, str(tmpdir / "dora_rank4.safetensors"))
    mismatched_model, _, _ = _make_dora_model(seed=5, rank=7)
    try:
        LoRACheckpointLoaderNode().build(model=mismatched_model, relative_path="dora_rank4.safetensors")
        raise AssertionError("expected ValueError for rank mismatch")
    except ValueError as e:
        assert "rank" in str(e).lower()
        print(f"    PASS (rank mismatch): {e}")


def check_missing_keys_raises_clearly(tmpdir):
    print("[a checkpoint missing an expected layer raises, doesn't silently partial-load]")
    from safetensors.torch import save_file
    save_file({"lora_unet_some_other_layer.lora_down.weight": torch.zeros(2, 8),
                "lora_unet_some_other_layer.lora_up.weight": torch.zeros(6, 2),
                "lora_unet_some_other_layer.alpha": torch.tensor([2.0])},
               str(tmpdir / "wrong_layers.safetensors"))
    model, _, _ = _make_model(seed=1)
    try:
        LoRACheckpointLoaderNode().build(model=model, relative_path="wrong_layers.safetensors")
        raise AssertionError("expected ValueError for missing keys")
    except ValueError as e:
        assert "missing" in str(e).lower()
        print(f"    PASS: {e}")


def check_rank_mismatch_raises_clearly(tmpdir):
    print("[a rank mismatch raises with a clear message, not an internal assertion]")
    trained_model, _, trained_layer = _make_model(seed=2, rank=4)
    from safetensors.torch import save_file
    state_dict = extract_lora_weights(trained_model.raw.lora_registry)
    save_file(state_dict, str(tmpdir / "rank4.safetensors"))

    mismatched_model, _, _ = _make_model(seed=3, rank=7)  # different rank
    try:
        LoRACheckpointLoaderNode().build(model=mismatched_model, relative_path="rank4.safetensors")
        raise AssertionError("expected ValueError for rank mismatch")
    except ValueError as e:
        assert "rank" in str(e).lower()
        print(f"    PASS: {e}")


def check_rejects_wrong_model_type():
    print("[rejects a model it can't operate on]")
    try:
        LoRACheckpointLoaderNode().build(model=object(), relative_path="x.safetensors")
        raise AssertionError("expected TypeError")
    except TypeError:
        print("    PASS")


def main():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        paths.set_loras_dir(tmpdir)
        try:
            check_contracts()
            check_load_reproduces_the_saved_forward_exactly(tmpdir)
            check_dora_round_trip_with_magnitude_reproduces_exactly(tmpdir)
            check_dora_without_magnitude_key_recomputes_it(tmpdir)
            check_dora_alpha_restore_affects_forward(tmpdir)
            check_dora_missing_keys_and_rank_mismatch_are_caught(tmpdir)
            check_missing_keys_raises_clearly(tmpdir)
            check_rank_mismatch_raises_clearly(tmpdir)
            check_rejects_wrong_model_type()
        finally:
            paths.set_loras_dir(None)
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
