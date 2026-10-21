"""Self-checks for adapters: layers (temporary) and the plan, injection and file round trips (signal).

    python -m next.tests.check_adapters
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..families.general.adapters.gate import TimestepGate, compute_gate
from ..families.general.adapters.layers import AdaptedConv2d, AdaptedLinear
from ..families.general.adapters.scaling import scaling_factor
import tempfile

from ..families.api.adapters import GroupOverride, TrainableSelection, plan_adapter
from ..families.general.adapters.inject import descriptors_from_model
from ..families.general.adapters.schemes import load_checked, read_adapter, save_adapter
from ..families.general.ldm_unet.unet import UNetModel
from ..families.sdxl.adapters import LAYOUT, SCHEMES, inject_denoiser
from ..families.sdxl.denoiser import SDXLDenoiser, build_unet
from .check_sdxl import TINY_UNET
from .support import check, raises, run


def make(kind, module, **kw):
    layer = (AdaptedLinear if isinstance(module, nn.Linear) else AdaptedConv2d)(module, kind=kind, rank=4, alpha=2.0, **kw)
    with torch.no_grad():
        layer.lora_B.normal_(std=0.3)
        if kind == "dora":
            layer.magnitude.mul_(1.2)
    return layer


def scaling_and_gate():
    check(scaling_factor("classic", 8, 16) == 0.5 and abs(scaling_factor("rank_stabilized", 8, 16) - 2.0) < 1e-9, "scaling policies")
    raises(ValueError, lambda: scaling_factor("nope", 1, 1), "unknown policy", "nope")
    raises(ValueError, lambda: scaling_factor("classic", 1, 0), "rank must be positive", "positive")
    t = torch.tensor([0.0, 200.0, 500.0, 800.0, 999.0])
    g = compute_gate(t, 200, 800, 50)
    check(g[1] > 0.4 and g[1] < 0.6 and g[3] > 0.4 and g[3] < 0.6, "the gate is one half on the range edge")
    check(g[2] > 0.99 and g[0] < 0.02 and g[4] < 0.02, "the gate is open inside and closed far outside")
    gate = TimestepGate()
    out = torch.ones(2, 3)
    check(gate.apply(out) is out, "no gate, no change")
    with gate.override(torch.tensor([0.0, 1.0])):
        check(gate.apply(out)[0].sum() == 0 and gate.apply(out)[1].sum() == 3, "gate scales per sample")
    check(gate.value is None, "override restores the previous gate")


def layer_cases():
    torch.manual_seed(0)
    lin, conv = nn.Linear(24, 16), nn.Conv2d(8, 16, 3, padding=1)
    x, xc = torch.randn(3, 5, 24), torch.randn(3, 8, 6, 6)
    for kind in ("lora", "dora"):
        for module, inp in ((lin, x), (conv, xc)):
            name = f"{kind} {type(module).__name__}"
            fresh = (AdaptedLinear if isinstance(module, nn.Linear) else AdaptedConv2d)(module, kind=kind, rank=4, alpha=2.0)
            with torch.no_grad():
                check(torch.allclose(fresh(inp), module(inp), atol=1e-5), f"{name}: a fresh adapter changes nothing")
            layer = make(kind, module)
            with torch.no_grad():
                base, adapted = module(inp), layer(inp)
            check(not torch.allclose(base, adapted, atol=1e-4), f"{name}: trained factors change the output")
            check(layer.base_weight.requires_grad is False and all(p.requires_grad for p in layer.trainable_parameters()),
                  f"{name}: base is a frozen buffer, factors train")
            check(layer.lora_A.dtype == torch.float32, f"{name}: factors are float32")
            gate = TimestepGate()
            layer.gate = gate
            with torch.no_grad(), gate.override(torch.tensor([0.0, 0.0, 0.0])):
                check(torch.allclose(layer(inp), base, atol=1e-5), f"{name}: a closed gate removes the adapter")
            with torch.no_grad(), gate.override(torch.tensor([1.0, 1.0, 1.0])):
                check(torch.allclose(layer(inp), adapted, atol=1e-5), f"{name}: an open gate is no gate")
            state = layer.adapter_state()
            other = make(kind, module)
            other.load_adapter_state(state)
            with torch.no_grad():
                check(torch.allclose(other(inp), adapted, atol=1e-5), f"{name}: state round-trips")
            check(set(state) == ({"lora_A", "lora_B"} | ({"magnitude"} if kind == "dora" else set())), f"{name}: state keys")
            loss = layer(inp).square().mean()
            loss.backward()
            check(all(p.grad is not None for p in layer.trainable_parameters()), f"{name}: gradients reach the factors")
    lora = make("lora", lin)
    with torch.no_grad():
        before = lora(x)
        lora.merge()
        after = F.linear(x, lora.base_weight, lora.base_bias)
    check(torch.allclose(before, after, atol=1e-4) and lora.lora_B.abs().max() == 0, "merging folds the update into the weight")
    raises(NotImplementedError, lambda: make("dora", lin).merge(), "DoRA merge is refused", "DoRA")
    half = make("lora", lin, weight=0.5)
    full = make("lora", lin, weight=1.0)
    full.load_adapter_state(half.adapter_state())
    check(abs(half.scaling * 2 - full.scaling) < 1e-9, "region weight multiplies the scaling")
    raises(ValueError, lambda: lora.load_adapter_state({"lora_A": torch.zeros(3, 3), "lora_B": lora.lora_B}), "shape mismatch", "does not fit")
    raises(ValueError, lambda: AdaptedLinear(lin, kind="nope"), "unknown kind", "nope")
    drop = AdaptedLinear(lin, kind="lora", rank=4, alpha=2.0, dropout=0.5)
    check(isinstance(drop.dropout, nn.Dropout), "dropout is used when asked for")


def merging_into_weights():
    import copy
    from ..families.general.adapters.merge import merge_adapter

    torch.manual_seed(0)
    original = UNetModel(**TINY_UNET)
    merged_net, pristine = copy.deepcopy(original), copy.deepcopy(original)
    den = SDXLDenoiser(original, dtype=torch.float32)
    part = inject_denoiser(den, TrainableSelection(), rank=4, alpha=2.0)
    for layer in part.layers.values():
        torch.nn.init.normal_(layer.lora_B, std=0.05)
    state = SCHEMES["kohya"].export(part)
    x, t, c, y = torch.randn(2, 4, 16, 16), torch.tensor([10.0, 500.0]), torch.randn(2, 5, 48), torch.randn(2, 40)
    with torch.no_grad():
        trained = den(x, t, c, y)
        report = merge_adapter(merged_net, SCHEMES["kohya"], state)
        plain = SDXLDenoiser(merged_net, dtype=torch.float32)
        check(report.merged == len(part.plan.modules) and not report.unused, f"every adapted module is merged ({report.merged})")
        check(torch.allclose(plain(x, t, c, y), trained, atol=1e-4), "the merged network gives what the injected adapter gave")
        before = SDXLDenoiser(pristine, dtype=torch.float32)(x, t, c, y)
        check(not torch.allclose(before, trained, atol=1e-4), "the adapter changes the output, so the comparison above means something")
        merge_adapter(pristine, SCHEMES["kohya"], state, 0.0)
        check(torch.allclose(SDXLDenoiser(pristine, dtype=torch.float32)(x, t, c, y), before, atol=1e-6), "strength 0 leaves the weights as they were")
        check(merge_adapter(pristine, SCHEMES["kohya"], {}).merged == 0, "a file that matches nothing merges nothing")


def sdxl_plan_and_injection():
    meta = build_unet()
    descriptors = descriptors_from_model(meta)
    count = lambda sel: len(plan_adapter(LAYOUT, descriptors, sel, 8, 8).modules)  # noqa: E731
    check(count(TrainableSelection()) == 722, f"default selection is the 722 modules of the old kohya default ({count(TrainableSelection())})")
    check(count(TrainableSelection(groups=("attn_self", "attn_cross", "ff", "transformer_proj", "label_emb"))) == 724, "plus label_emb is 724")
    check(count(TrainableSelection(groups=tuple(g.name for g in LAYOUT.groups()))) == 726, "plus time_embed is 726")
    shapes = {"model.diffusion_model." + k: tuple(v.shape) for k, v in meta.state_dict().items()}
    from ..families.api.adapters import descriptors_from_shapes
    from_header = plan_adapter(LAYOUT, descriptors_from_shapes(shapes, "model.diffusion_model."), TrainableSelection(), 8, 8)
    check(from_header == plan_adapter(LAYOUT, descriptors, TrainableSelection(), 8, 8), "a plan from key names equals a plan from the model")
    sel = TrainableSelection(groups=("attn_cross",), exclude=(r".*\.attn2\.to_k",), regions={"input_blocks": None})
    chosen = plan_adapter(LAYOUT, descriptors, sel, 8, 8).modules
    check(chosen and all(m.group == "attn_cross" and "to_k" not in m.path and not m.path.startswith("input_blocks") for m in chosen),
          "exclude wins, a region set off removes its modules")
    only_conv = plan_adapter(LAYOUT, descriptors, TrainableSelection(groups=(), include=(r"input_blocks\.0\.0",)), 8, 8).modules
    check([m.group for m in only_conv] == ["custom"], "a pattern can add a module no group covers")
    raises(ValueError, lambda: plan_adapter(LAYOUT, descriptors, TrainableSelection(groups=("nope",)), 8, 8), "unknown group lists valid ones", "attn_self")
    tuned = plan_adapter(LAYOUT, descriptors, TrainableSelection(per_group={"ff": GroupOverride(rank=2, lr_mult=0.5)}), 8, 8)
    check({m.rank for m in tuned.modules if m.group == "ff"} == {2} and {m.rank for m in tuned.modules if m.group == "attn_self"} == {8},
          "per-group rank override")

    torch.manual_seed(0)
    den = SDXLDenoiser(UNetModel(**TINY_UNET), dtype=torch.float32)
    x, t, c, y = torch.randn(2, 4, 16, 16), torch.tensor([10.0, 500.0]), torch.randn(2, 5, 48), torch.randn(2, 40)
    with torch.no_grad():
        before = den(x, t, c, y)
    part = inject_denoiser(den, TrainableSelection(per_group={"ff": GroupOverride(lr_mult=0.1)}), rank=4, alpha=4.0)
    with torch.no_grad():
        check(torch.allclose(den(x, t, c, y), before, atol=1e-5), "a fresh adapter leaves the model's output unchanged")
    trainable = {id(p) for p in part.trainable_parameters()}
    check(trainable == {id(p) for p in den.parameters() if p.requires_grad} and len(trainable) == 2 * len(part.plan.modules),
          "only adapter factors train")
    groups = part.param_groups(lr=1e-4)
    check({g["name"] for g in groups} == {"attn_self", "attn_cross", "ff@0.1", "transformer_proj"}
          and next(g for g in groups if g["group"] == "ff")["lr"] == 1e-5, "optimizer groups follow group and lr multiplier")
    with torch.no_grad():
        for layer in part.layers.values():
            layer.lora_B.normal_(std=0.1)
        trained = den(x, t, c, y)
    for name, scheme in SCHEMES.items():
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter(f"{tmp}/a.safetensors", scheme.export(part), {"family": "sdxl", "scheme": name})
            tensors, meta_out = read_adapter(f"{tmp}/a.safetensors")
        report = load_checked(scheme, part, tensors)
        check(report.matched == len(part.layers) and not report.missing and not report.unexpected and meta_out["scheme"] == name,
              f"{name} file round-trips onto the same layers")
    with torch.no_grad():
        check(torch.allclose(den(x, t, c, y), trained, atol=1e-5), "loading an exported file restores the trained output")
    dora = inject_denoiser(SDXLDenoiser(UNetModel(**TINY_UNET), dtype=torch.float32), TrainableSelection(), rank=2, alpha=2.0, kind="dora")
    raises(ValueError, lambda: SCHEMES["kohya"].export(dora), "kohya refuses DoRA instead of writing a wrong file", "native")
    check(any(k.endswith(".magnitude") for k in SCHEMES["native"].export(dora)), "native holds DoRA magnitudes")
    raises(ValueError, lambda: load_checked(SCHEMES["kohya"], part, {"other.weight": torch.zeros(1)}), "a file matching nothing is refused", "no layer matched")


if __name__ == "__main__":
    run(scaling_and_gate, layer_cases, merging_into_weights, sdxl_plan_and_injection)
