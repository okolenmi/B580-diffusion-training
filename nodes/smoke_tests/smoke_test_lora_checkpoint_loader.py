"""Real torch, real core.lora classes, real safetensors I/O against a
temp directory (paths.set_loras_dir) -- verifies nodes/model/
lora_checkpoint_loader.py end to end: save a trained LoRA with the real
extraction path, load it into a fresh injection, forward output matches
the original exactly.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

import paths
from core.lora import LoRALinear, extract_lora_weights
from nodes.model.lora_checkpoint_loader import LoRACheckpointLoaderNode
from nodes.model.lora_injector import ComfyUNetTrainableModel


class _FakeWrapper:
    def __init__(self, registry):
        self.lora_registry = registry

    def has_lora(self) -> bool:
        return len(self.lora_registry) > 0


def check_contracts():
    print("[contracts]")
    assert not getattr(LoRACheckpointLoaderNode, "__abstractmethods__", None)
    assert set(LoRACheckpointLoaderNode.INPUTS) == {"model", "relative_path"}
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
