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


def int8_base_weights():
    import warnings
    from types import SimpleNamespace

    from ..families.general.quantize.backends import TorchBackend, XpuKernelsBackend, make_backend, shape_problem
    from ..families.general.quantize.int8 import Int8Linear, Int8Quantizer, quantize_frozen

    torch.manual_seed(0)
    backend = TorchBackend()
    weight = torch.randn(64, 512) * 0.05
    w8, scale = backend.prepare(weight)
    dense = backend.dequant(w8, scale).float()
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
    check(w8.dtype == torch.int8 and scale.shape == (64,) and rel(dense, weight) < 0.02, "an int8 weight comes back within quantization error")
    x = torch.randn(40, 512)
    check(rel(backend.linear(x, w8, scale, None), x @ dense.T) < 0.02 and rel(backend.linear(x, w8, scale, None), x @ weight.T) < 0.03,
          "the int8 linear matches the dense product of the dequantized weight")
    check(shape_problem(64, 512) is None and "256" in shape_problem(64, 300) and "16" in shape_problem(70, 512), "layer shapes the rotation cannot take are named")

    net = nn.Sequential(nn.Linear(512, 64), nn.Linear(100, 64), nn.Linear(512, 64))
    wrapped = make("lora", net[0])
    net[0] = wrapped
    net[2] = make("dora", net[2])
    report = Int8Quantizer(backend, min_params=0).run(net)
    check(wrapped.is_int8 and wrapped.base_weight.dtype == torch.int8 and report.layers == 1 and report.saved_mb > 0
          and len(report.skipped) == 2 and not net[2].is_int8,
          "a LoRA layer converts; DoRA and a shape the rotation cannot take stay as they were")
    plain = nn.Sequential(nn.Linear(512, 64))
    quantize_frozen(plain, backend="torch", min_params=0)
    check(isinstance(plain[0], Int8Linear), "an unadapted linear becomes an int8 linear")
    check(quantize_frozen(nn.Sequential(nn.Linear(512, 64)), backend="torch").layers == 0, "small layers are left alone by default")

    x = torch.randn(40, 512, requires_grad=True)
    out = wrapped(x)
    out.float().square().sum().backward()
    reference = torch.randn(64, 512)
    check(x.grad is not None and wrapped.lora_B.grad is not None and wrapped.lora_A.grad is not None and wrapped.base_weight.grad is None,
          "gradients reach the input and the adapter factors but not the frozen weight")
    x2 = x.detach().clone().requires_grad_(True)
    dy = torch.randn(40, 64)
    base = Int8Linear(nn.Linear(512, 64), backend)
    base(x2).backward(dy)
    expected = dy @ backend.dequant(base.weight_int8, base.weight_scale).float()
    check(rel(x2.grad, expected) < 1e-2, "the input gradient is dy times the dequantized weight")
    x3 = x.detach().clone().requires_grad_(True)
    base.quantized.grad = "int8"
    base(x3).backward(dy)
    check(rel(x3.grad, expected) < 3e-2 and x3.grad is not None, "the int8 gradient path stays close to the exact one without a dense weight")
    raises(ValueError, lambda: Int8Linear(nn.Linear(512, 64), backend, "nope"), "an unknown gradient path is refused", "nope")
    raises(NotImplementedError, wrapped.merge, "merging into int8 weights is refused", "quantized")

    calls = []
    fake = SimpleNamespace(available=lambda: True, prepare_weight=lambda w: backend.prepare(w),
                           linear=lambda x, w8, s, bias: calls.append(1) or backend.linear(x, w8, s, bias),
                           dequant_weight=lambda w8, s: backend.dequant(w8, s), linear_dx_int8=backend.dx_int8, __version__="0.3.0")
    kernels = XpuKernelsBackend(fake)
    check(XpuKernelsBackend(SimpleNamespace(available=lambda: True, __version__="0.2.0")).problem() is not None,
          "an xpu-kernels older than the fused weight dequantizer is refused")
    check(kernels.problem() is None and kernels.call_problem(torch.zeros(40, 512), w8) is not None,
          "the kernels backend runs only on an XPU tensor and says why otherwise")
    layer = Int8Linear(nn.Linear(512, 64), kernels)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = layer(torch.randn(40, 512))
    check(result.shape == (40, 64) and not calls and any("dense weight" in str(w.message) for w in caught),
          "a call the kernels cannot take falls back to the dense weight and warns once")
    check(isinstance(make_backend("auto", "cpu"), TorchBackend) and isinstance(make_backend("torch", "xpu"), TorchBackend),
          "auto on a CPU uses the torch version")
    raises(ValueError, lambda: make_backend("nope", "cpu"), "an unknown backend is refused", "nope")


def nf4_base_weights():
    from types import SimpleNamespace

    from ..families.general.quantize.nf4 import Nf4Linear, Nf4Quantizer, quantize_frozen
    from ..families.general.quantize.nf4_backends import (BLOCK, DQ_BLOCK, TorchNf4Backend, XpuKernelsNf4Backend, codebook, make_nf4_backend,
                                                          shape_problem)

    torch.manual_seed(0)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
    bnb = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
           0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
           0.7229568362236023, 1.0]
    check(torch.allclose(codebook(), torch.tensor(bnb), atol=1e-6), "the codebook is the published NF4 table")
    backend = TorchNf4Backend()
    weight = torch.randn(64, 512) * 0.02
    packed, absmax, scale, offset = backend.quantize(weight, False)
    dense = backend.dequant(packed, absmax, scale, offset, weight.shape, torch.float32)
    check(packed.dtype == torch.uint8 and packed.numel() == weight.numel() // 2 and absmax.shape == (weight.numel() // BLOCK,) and scale is None
          and 0.05 < rel(dense, weight) < 0.15, f"NF4 packs two weights per byte and comes back within quantization error ({rel(dense, weight):.3f})")
    levels = dense.view(-1, BLOCK) / absmax.unsqueeze(1)
    check((levels.unsqueeze(-1) - codebook()).abs().amin(-1).max() < 1e-5, "every dequantized value is a codebook level times its block's absmax")
    zeros = torch.zeros(1, BLOCK)
    z = backend.quantize(zeros, False)
    check(not backend.dequant(*z, zeros.shape, torch.float32).any(), "an all-zero block stays zero")
    p2, a2, s2, o2 = backend.quantize(weight, True)
    dense2 = backend.dequant(p2, a2, s2, o2, weight.shape, torch.float32)
    check(a2.dtype == torch.int8 and s2.shape == (-(-absmax.numel() // DQ_BLOCK),) and torch.equal(p2, packed) and abs(rel(dense2, weight) - rel(dense, weight)) < 0.01,
          "double quantization keeps the same nibbles, stores absmax in int8 and costs almost no accuracy")
    check(stored_bytes(packed, absmax) > stored_bytes(p2, a2, s2, o2), "double quantization stores fewer bytes")
    check(shape_problem((64, 512)) is None and "64" in shape_problem((3, 5)), "a weight that is not a whole number of blocks is named")

    net = nn.Sequential(nn.Linear(512, 64), nn.Linear(10, 7), nn.Linear(512, 64))
    wrapped = make("lora", net[0])
    net[0] = wrapped
    net[2] = make("dora", net[2])
    report = Nf4Quantizer(backend, min_params=0).run(net)
    check(wrapped.is_nf4 and wrapped.is_quantized and not wrapped.is_int8 and wrapped.base_weight.dtype == torch.uint8 and report.layers == 1
          and len(report.skipped) == 2 and not net[2].is_quantized and report.summary().startswith("nf4 base weights"),
          "a LoRA layer converts; DoRA and a shape that is not a whole number of blocks stay as they were")
    plain = nn.Sequential(nn.Linear(512, 64))
    quantize_frozen(plain, backend="torch", min_params=0)
    check(isinstance(plain[0], Nf4Linear), "an unadapted linear becomes an NF4 linear")
    check(quantize_frozen(nn.Sequential(nn.Linear(512, 64)), backend="torch").layers == 0, "small layers are left alone by default")
    raises(ValueError, lambda: quantize_frozen(nn.Sequential(nn.Linear(512, 64)), backend="nope"), "an unknown backend is refused", "nope")

    x = torch.randn(40, 512, requires_grad=True)
    wrapped(x).float().square().sum().backward()
    check(x.grad is not None and wrapped.lora_B.grad is not None and wrapped.base_weight.grad is None, "gradients reach the input and the factors, not the frozen weight")
    layer = Nf4Linear(nn.Linear(512, 64), backend)
    x2, dy = torch.randn(40, 512, requires_grad=True), torch.randn(40, 64)
    layer(x2).backward(dy)
    stored = backend.dequant(layer.weight_nf4, layer.weight_absmax, None, None, (64, 512), torch.float32)
    check(rel(x2.grad, dy @ stored) < 1e-5, "the input gradient is dy times the dequantized weight")
    check(rel(layer(x2.detach()), x2.detach() @ stored.T + layer.bias) < 1e-5, "the forward is the product with the dequantized weight")
    raises(NotImplementedError, wrapped.merge, "merging into NF4 weights is refused", "quantized")
    check(torch.allclose(wrapped.frozen_weight.float(), backend.dequant(wrapped.base_weight, wrapped.base_scale, None, None, (64, 512), torch.bfloat16).float()),
          "frozen_weight rebuilds the dense weight")
    saved_buffers = {k for k, _ in wrapped.named_buffers() if k.startswith("base_")}
    check({"base_weight", "base_scale"} <= saved_buffers, "the quantized tensors are buffers of the layer (they move with it)")

    calls = []
    fused = XpuKernelsNf4Backend()
    fused._module = SimpleNamespace(usable=lambda packed: True,
                                    dequant_nf4=lambda packed, absmax, book, dtype: calls.append(1) or backend.codec.dequant(packed, absmax, (-1, 1), dtype).flatten())
    out = fused.dequant(packed, absmax, None, None, weight.shape, torch.bfloat16)
    check(calls and out.shape == weight.shape and out.dtype == torch.bfloat16 and rel(out, dense) < 0.01, "the fused kernel is used when it can run")
    fused._module = SimpleNamespace(usable=lambda packed: False, dequant_nf4=lambda *a, **k: calls.append(2))
    again = fused.dequant(packed, absmax, None, None, weight.shape, torch.float32)
    check(calls == [1] and torch.equal(again, dense), "when the kernel refuses a tensor the torch chain runs instead")
    check(isinstance(make_nf4_backend("auto", "cpu"), TorchNf4Backend), "auto on a CPU uses the torch version")
    raises(ValueError, lambda: make_nf4_backend("nope", "cpu"), "an unknown NF4 backend is refused", "nope")


def stored_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


if __name__ == "__main__":
    run(scaling_and_gate, layer_cases, merging_into_weights, int8_base_weights, nf4_base_weights, sdxl_plan_and_injection)
