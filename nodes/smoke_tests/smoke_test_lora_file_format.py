"""What a saved LoRA file contains, and that it survives the trip.

Real torch, real safetensors I/O in a temp directory, real layer classes
(LoRALinear, DoRALinear, the phase-split generations) -- tiny CPU shapes.

Why this file exists. The block-weight round trip (a LoRA trained at block
weight w must come back at w) was first built for one of the three extraction
functions and tested only in memory, on plain `LoRALinear`. That left three
real defects, each reproduced here before it was fixed:

* `extract_lora_weights` raised AttributeError on a DoRA layer (which keeps
  the weight on its inner `_lora`) -- a DoRA run could not save;
* the graph route saves through `lora_phases.extract_combined_weights`,
  which never wrote the weight, so the round trip did not exist where runs
  actually save;
* writing the weight as a `.block_weight` *tensor* makes the file
  unloadable by diffusers (its kohya converter raises on any leftover key --
  see nodes/model/lora_file.py), so the weight now travels in the file's
  metadata. `check_a_file_has_only_the_tensors_other_readers_know` pins the
  rule itself: nothing in the file but the four suffixes readers consume.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

import paths
from nodes.components.layout import ProjectLayout
from nodes.model.dora_layer import DoRALinear
from nodes.model.lora import (
    LoRALinear, extract_lora_weights, layer_training_weight, load_lora_into_model,
    lora_health_summary,
)
from nodes.model.lora_file import (
    BLOCK_WEIGHTS_METADATA_KEY, load_lora_state_dict, restore_block_weights,
    split_block_weights,
)
from nodes.model.lora_injector import ComfyUNetTrainableModel
from nodes.model.lora_phases import (
    extract_combined_weights, extract_own_generation_weights, split_into_new_generation,
)
from nodes.model.lora_saver import save_trained_weights

#: The only tensor suffixes a kohya-format reader consumes. diffusers 0.41.0
#: (`_convert_non_diffusers_lora_to_diffusers`) raises on anything else.
KNOWN_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha", ".dora_scale")


class _Wrapper:
    def __init__(self, registry):
        self.lora_registry = registry

    def has_lora(self) -> bool:
        return len(self.lora_registry) > 0


def _model(layer_cls, weight, seed=1):
    torch.manual_seed(seed)
    base = nn.Linear(8, 6)
    if layer_cls is DoRALinear:
        layer = DoRALinear(base, rank=4, alpha=6.0, dropout=0.0, weight=weight)
    else:
        layer = layer_cls(base, rank=4, alpha=6.0, weight=weight)
    parent = nn.Module()
    parent.proj = layer
    registry = [("root.proj", parent, "proj", layer)]
    return ComfyUNetTrainableModel(_Wrapper(registry)), registry, layer


def _train_a_little(layer):
    params = [p for p in layer.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.3)
    for _ in range(4):
        loss = layer(torch.randn(5, 8)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def check_every_layer_kind_reports_its_weight():
    print("[layer_training_weight reads every adapter kind; the DoRA save no longer raises]")
    for cls in (LoRALinear, DoRALinear):
        _, registry, layer = _model(cls, 0.5)
        assert abs(layer_training_weight(layer) - 0.5) < 1e-9, cls.__name__
        sd = extract_lora_weights(registry, layer_classes=(LoRALinear, DoRALinear))
        assert abs(float(sd["lora_unet_root_proj.block_weight"]) - 0.5) < 1e-9, cls.__name__

    # NF4 layers and phase-split generations store only the product
    # `scaling == (alpha / rank) * weight`; the weight is recovered from it.
    class _OnlyScaling:
        alpha, rank = 6.0, 4
        scaling = (6.0 / 4) * 0.25

    assert abs(layer_training_weight(_OnlyScaling()) - 0.25) < 1e-9
    print("    PASS")


def check_all_three_extractors_agree_for_an_unsplit_layer():
    print("[extract_lora_weights / extract_combined_weights / extract_own_generation_weights]")
    for cls in (LoRALinear, DoRALinear):
        _, registry, layer = _model(cls, 0.5)
        keys = set(extract_lora_weights(registry, layer_classes=(LoRALinear, DoRALinear)))
        # The two phase extractors also write a DoRA layer's `.dora_scale`;
        # everything else must be the same set.
        for fn in (extract_combined_weights, extract_own_generation_weights):
            assert set(fn(registry)) - {"lora_unet_root_proj.dora_scale"} == keys, \
                (fn.__name__, cls.__name__)
        for fn in (extract_combined_weights, extract_own_generation_weights):
            w = float(fn(registry)["lora_unet_root_proj.block_weight"])
            assert abs(w - 0.5) < 1e-9, (fn.__name__, cls.__name__, w)
    print("    PASS")


def check_a_split_chain_is_baked_so_its_weight_is_one():
    print("[phase-split chain: each generation's scaling is baked into B, so the weight is 1.0]")
    _, registry, layer = _model(LoRALinear, 0.5)
    wrapper = _Wrapper(registry)
    split_into_new_generation(wrapper, rank=2, alpha=4.0)
    sd = extract_combined_weights(wrapper.lora_registry)
    assert abs(float(sd["lora_unet_root_proj.block_weight"]) - 1.0) < 1e-9
    assert float(sd["lora_unet_root_proj.alpha"]) == float(sd["lora_unet_root_proj.lora_down.weight"].shape[0])
    print("    PASS")


def check_a_file_has_only_the_tensors_other_readers_know(tmp: Path):
    print("[the saved file carries no tensor beyond what kohya-format readers consume]")
    paths.set_loras_dir(tmp)
    layout = ProjectLayout.from_paths_module()
    for weight in (1.0, 0.5):
        model, _, layer = _model(LoRALinear, weight)
        _train_a_little(layer)
        out = save_trained_weights(model, f"w{weight}.safetensors", layout=layout)
        from safetensors import safe_open
        with safe_open(out, framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata() or {}
        stray = [k for k in keys if not k.endswith(KNOWN_SUFFIXES)]
        assert not stray, f"weight {weight}: keys no kohya reader consumes: {stray}"
        assert BLOCK_WEIGHTS_METADATA_KEY in meta, \
            "the metadata field marks a file as 'trained at these weights'"
        listed = json.loads(meta[BLOCK_WEIGHTS_METADATA_KEY])
        if weight == 1.0:
            assert listed == {}, "the default weight is not listed"
        else:
            assert abs(listed["lora_unet_root_proj"] - 0.5) < 1e-9
    print("    PASS")


def check_a_trained_weight_survives_the_file(tmp: Path):
    print("[a block weight survives save -> file -> load, on the graph save path, for LoRA and DoRA]")
    paths.set_loras_dir(tmp)
    layout = ProjectLayout.from_paths_module()
    from nodes.model.lora_checkpoint_loader import load_lora_into_registry
    for cls in (LoRALinear, DoRALinear):
        model, registry, trained = _model(cls, 2.0, seed=4)
        _train_a_little(trained)
        out = save_trained_weights(model, f"rt_{cls.__name__}.safetensors", layout=layout)

        # A fresh layer configured at the default weight 1.0 (different seed),
        # given the same frozen base so the two forwards are comparable.
        _, fresh_registry, fresh = _model(cls, 1.0, seed=99)
        fresh_core = fresh._lora if cls is DoRALinear else fresh
        trained_core = trained._lora if cls is DoRALinear else trained
        fresh_core.base_weight.data.copy_(trained_core.base_weight.data)
        if trained_core.base_bias is not None:
            fresh_core.base_bias.data.copy_(trained_core.base_bias.data)

        state = load_lora_state_dict(out)
        assert "lora_unet_root_proj.block_weight" in state, "the reader restores the key"
        load_lora_into_registry(fresh_registry, state, source_description=out)

        x = torch.randn(7, 8)
        if cls is LoRALinear:
            assert abs(fresh.training_weight - 2.0) < 1e-9, fresh.training_weight
        assert abs(fresh.scaling - (6.0 / 4) * 2.0) < 1e-6, (cls.__name__, fresh.scaling)
        torch.testing.assert_close(fresh(x), trained(x), rtol=1e-4, atol=1e-5)
    print("    PASS")


def check_a_file_from_before_the_field_keeps_the_configured_weight():
    print("[no metadata field -> the configured weight keeps applying (old files)]")
    _, registry, layer = _model(LoRALinear, 1.0)
    old_file = {k: v for k, v in extract_lora_weights(registry).items()
                if not k.endswith(".block_weight")}
    assert restore_block_weights(dict(old_file), None) == old_file or \
        set(restore_block_weights(dict(old_file), {}).keys()) == set(old_file.keys())
    _, reg2, configured = _model(LoRALinear, 0.5, seed=7)
    load_lora_into_model(reg2, old_file)
    assert abs(configured.training_weight - 0.5) < 1e-9
    print("    PASS")


def check_split_leaves_the_input_alone():
    print("[split_block_weights does not mutate its input; the mid-run state dict still has the keys]")
    _, registry, _ = _model(LoRALinear, 0.5)
    sd = extract_lora_weights(registry)
    before = dict(sd)
    tensors, meta = split_block_weights(sd)
    assert sd.keys() == before.keys()
    assert not any(k.endswith(".block_weight") for k in tensors)
    assert BLOCK_WEIGHTS_METADATA_KEY in meta
    print("    PASS")


def check_health_summary_handles_conv_adapters():
    print("[lora_health_summary on a Conv2d-shaped adapter (it used to raise and skip the summary)]")
    rank, cin, cout, k = 4, 3, 5, 3
    sd = {
        "lora_unet_input_blocks_1_0_conv.lora_down.weight": torch.randn(rank, cin, k, k),
        "lora_unet_input_blocks_1_0_conv.lora_up.weight": torch.randn(cout, rank, 1, 1),
        "lora_unet_input_blocks_1_0_conv.alpha": torch.tensor([float(rank)]),
    }
    summary = lora_health_summary(sd)
    group = summary["groups"]["other_unet"]
    # Reference: the true delta of the conv, built the long way.
    down, up = sd["lora_unet_input_blocks_1_0_conv.lora_down.weight"], \
        sd["lora_unet_input_blocks_1_0_conv.lora_up.weight"]
    delta = torch.einsum("or,rikl->oikl", up.reshape(cout, rank), down)
    assert abs(group["median_dW"] - delta.norm().item()) < 1e-4, (group, delta.norm().item())
    print("    PASS")


def check_a_zero_lr_schedule_still_builds_its_summary_line():
    print("[schedule_summary/format_schedule_line at lr == 0: no ZeroDivisionError]")
    from nodes.train.schedule import ConstantLRSchedule, format_schedule_line, schedule_summary
    line = format_schedule_line(schedule_summary(ConstantLRSchedule(lr=0.0), 5), "T")
    assert "peak 0.000e+00" in line and "WARNING" not in line, line
    print("    PASS")


def main():
    check_every_layer_kind_reports_its_weight()
    check_all_three_extractors_agree_for_an_unsplit_layer()
    check_a_split_chain_is_baked_so_its_weight_is_one()
    check_split_leaves_the_input_alone()
    check_a_file_from_before_the_field_keeps_the_configured_weight()
    check_health_summary_handles_conv_adapters()
    check_a_zero_lr_schedule_still_builds_its_summary_line()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            check_a_file_has_only_the_tensors_other_readers_know(tmp)
            check_a_trained_weight_survives_the_file(tmp)
        finally:
            paths.set_loras_dir(None)
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
