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
    part.up_ratio = 4.0
    plus = part.param_groups(lr=1e-4)
    up = [g for g in plus if g["name"].endswith("@4") or ".up@" in g["name"]]
    check(all(abs(g["lr"] - 1e-4 * g["lr_mult"]) < 1e-12 for g in plus) and {g["name"] for g in up} >= {"attn_self.up@4", "ff.up@0.4"}
          and all(id(p) == id(layer.lora_B) for g in plus if ".up" in g["name"] for p in g["params"][:1] for layer in [next(l for l in part.layers.values() if l.lora_B is p)])
          and sum(len(g["params"]) for g in plus) == 2 * len(part.plan.modules), "LoRA+ puts the B factors in groups with the rate multiplied")
    part.up_ratio = 1.0
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


def aligned_start():
    """The gradient-aligned initialisation finds the gradient's main directions and leaves the function unchanged."""
    from ..engine.step.aligned_init import AlignedInit
    torch.manual_seed(0)
    out, inn, rank = 20, 30, 4
    layer = AdaptedLinear(nn.Linear(inn, out), kind="lora", rank=rank, alpha=float(rank))
    left, right = torch.linalg.qr(torch.randn(out, rank)).Q, torch.linalg.qr(torch.randn(inn, rank)).Q
    target = (left * torch.tensor([5.0, 4.0, 3.0, 2.0])) @ right.T + 0.05 * torch.randn(out, inn)  # dL/dW: rank 4 plus noise
    eye = torch.eye(inn)

    def backward(_batch):
        (layer(eye) * target.T).sum().backward()  # y = x W^T, so dL/dW = target

    def zero():
        for p in (layer.lora_A, layer.lora_B):
            p.grad = None

    before = layer(eye).detach()
    old_norm = layer.lora_A.detach().norm(dim=1).mean()
    summary = AlignedInit({"layer": layer}, batches=2).run(lambda: {}, backward, zero)
    rows = layer.lora_A.detach()
    kept = ((rows @ right).pow(2).sum() / rows.pow(2).sum()).item()
    check(summary["layers"] == 1 and kept > 0.97 and float(layer.lora_B.abs().max()) == 0.0 and torch.equal(layer(eye).detach(), before),
          f"A's rows lie in the gradient's main subspace ({kept:.3f} of their energy), B is zero, the function is unchanged")
    check(abs(float(rows.norm(dim=1).mean()) - float(old_norm)) < 1e-5, "the new rows keep the old rows' average length")
    gain = lambda a: float((target @ a.T).norm())  # how strongly the first update of B (target A^T) moves
    fresh = AdaptedLinear(nn.Linear(inn, out), kind="lora", rank=rank, alpha=float(rank))
    check(gain(rows) > 1.5 * gain(fresh.lora_A.detach()), "the first update of B is larger than with a random A")
    raises(ValueError, lambda: AlignedInit({"layer": layer}, batches=0), "at least one batch is needed", "at least one")
    with torch.no_grad():
        layer.lora_B.fill_(0.1)
    raises(ValueError, lambda: AlignedInit({"layer": layer}, batches=1).run(lambda: {}, backward, zero), "a loaded adapter is refused", "fresh adapter")
    dora = make("dora", nn.Linear(inn, out))
    raises(ValueError, lambda: AlignedInit({"layer": dora}, batches=1), "DoRA is refused", "LoRA")


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


def activation_compute():
    """The low-rank path in the activation dtype matches the float32 path within bf16 rounding, forward and gradient, and makes no float32 full-size tensor."""
    torch.manual_seed(3)
    for module, shape in ((nn.Linear(24, 40).to(torch.bfloat16), (2, 5, 24)), (nn.Conv2d(8, 12, 3, padding=1).to(torch.bfloat16), (2, 8, 6, 6))):
        reference, fast = make("lora", module), make("lora", module)
        fast.load_state_dict(reference.state_dict(), strict=False)
        fast.compute_in_activation()
        x = torch.randn(*shape).to(torch.bfloat16)
        x_ref, x_fast = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        out_ref, out_fast = reference(x_ref), fast(x_fast)
        out_ref.float().square().mean().backward()
        out_fast.float().square().mean().backward()
        name = type(module).__name__
        check(out_fast.dtype == torch.bfloat16 and torch.allclose(out_ref.float(), out_fast.float(), atol=0.05, rtol=0.05), f"{name}: forward agrees within bf16 rounding")
        grad_error = (reference.lora_B.grad - fast.lora_B.grad).abs().max() / reference.lora_B.grad.abs().max()
        check(fast.lora_A.grad.dtype == torch.float32 and grad_error < 0.05 and torch.allclose(x_ref.grad.float(), x_fast.grad.float(), atol=0.05, rtol=0.1),
              f"{name}: factor gradients stay float32 parameters and agree ({grad_error:.3f})")
    lin = make("lora", nn.Linear(24, 40).to(torch.bfloat16))
    lin.compute_in_activation()
    gate = TimestepGate()
    lin.gate = gate
    x = torch.randn(3, 24).to(torch.bfloat16)
    plain = lin(x)
    gate.set(torch.tensor([1.0, 0.5, 0.0]))
    gated = lin(x)
    base = F.linear(x, lin.base_weight, lin.base_bias)
    check(torch.allclose(gated[2].float(), base[2].float(), atol=1e-2) and not torch.allclose(gated[0].float(), base[0].float(), atol=1e-3) and plain.shape == gated.shape,
          "an active timestep gate scales the update per sample on this path too")
    dora = make("dora", nn.Linear(24, 40))
    raises(ValueError, lambda: dora.compute_in_activation(), "DoRA cannot compute in the activation dtype", "LoRA")
    fp32 = make("lora", nn.Linear(24, 40))
    fp32.compute_in_activation()
    x32 = torch.randn(2, 24)
    check(torch.equal(fp32(x32), make("lora", nn.Linear(24, 40)).__class__.forward(fp32, x32)), "float32 activations take the ordinary path")


def stored_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def continue_from_file():
    """Inject Adapter's `init_adapter` loads a saved adapter into a fresh one (same output as the saved one) and refuses files that do not fit or do not exist."""
    from types import SimpleNamespace

    from ..nodes.model.inject_adapter import InjectAdapterNode

    torch.manual_seed(0)
    x, t, c, y = torch.randn(2, 4, 16, 16), torch.tensor([10.0, 500.0]), torch.randn(2, 5, 48), torch.randn(2, 40)
    family = SimpleNamespace(id="sdxl", read_adapter=read_adapter, schemes=SCHEMES, load_checked=load_checked)
    node = object.__new__(InjectAdapterNode)
    import copy

    base = UNetModel(**TINY_UNET)
    first = SDXLDenoiser(copy.deepcopy(base), dtype=torch.float32)
    part = inject_denoiser(first, TrainableSelection(), rank=4, alpha=4.0)
    with torch.no_grad():
        for layer in part.layers.values():
            layer.lora_B.normal_(std=0.1)
        trained = first(x, t, c, y)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/a.safetensors"
        save_adapter(path, SCHEMES["kohya"].export(part), {"family": "sdxl", "scheme": "kohya", "step": "600"})
        second = SDXLDenoiser(copy.deepcopy(base), dtype=torch.float32)
        fresh = inject_denoiser(second, TrainableSelection(), rank=4, alpha=4.0)
        node._start_from(family, fresh, "") is None or check(False, "no file, no loading")
        info = node._start_from(family, fresh, path)
        with torch.no_grad():
            check(info["step"] == "600" and info["matched"] == len(fresh.layers) and torch.allclose(second(x, t, c, y), trained, atol=1e-5),
                  "a fresh adapter that starts from a file gives the saved output")
        other = inject_denoiser(SDXLDenoiser(UNetModel(**TINY_UNET), dtype=torch.float32), TrainableSelection(), rank=2, alpha=2.0)
        raises(ValueError, lambda: node._start_from(family, other, path), "a file of another rank is refused with a hint", "same rank")
        raises(ValueError, lambda: node._start_from(family, fresh, f"{tmp}/gone.safetensors"), "a missing file is refused", "does not exist")


def lora_cleaner():
    """The cleaner: the singular rewrite keeps the change, gates reach the logits through the layers' own forward and give the parameters back, the kept share is exact, and the frozen part survives an update."""
    from ..engine.step.cleaner import Gates, LoraCleaner, to_singular_directions

    torch.manual_seed(0)
    x, t, c, y = torch.randn(2, 4, 16, 16), torch.tensor([10.0, 500.0]), torch.randn(2, 5, 48), torch.randn(2, 40)
    den = SDXLDenoiser(UNetModel(**TINY_UNET), dtype=torch.float32)
    part = inject_denoiser(den, TrainableSelection(), rank=4, alpha=4.0)
    with torch.no_grad():
        for layer in part.layers.values():
            layer.lora_B.normal_(std=0.1)
        trained = den(x, t, c, y)
    layers = part.layers
    first = next(iter(layers.values()))
    delta = (first.lora_B.detach().reshape(first.out_features, -1) @ first.lora_A.detach().reshape(first.rank, -1)).clone() if hasattr(first, "out_features") else None
    to_singular_directions(layers)
    with torch.no_grad():
        check(torch.allclose(den(x, t, c, y), trained, atol=1e-5), "the singular rewrite leaves the network's output unchanged")
        if delta is not None:
            check(torch.allclose(first.lora_B.reshape(first.out_features, -1) @ first.lora_A.reshape(first.rank, -1), delta, atol=1e-5), "and each layer's change B A")
    for mode in ("element", "component"):
        gates = Gates(layers, mode)
        before = {p: (l.lora_A.detach().clone(), l.lora_B.detach().clone()) for p, l in layers.items()}
        gates.engage()
        out = den(x, t, c, y)
        out.pow(2).mean().backward()
        gates.collect()
        got = [g.grad is not None and float(g.grad.abs().sum()) > 0 for g in gates.leaves()]
        check(sum(got) > len(got) // 2 and not any(p.requires_grad for per in gates.params.values() for p in per.values()),
              f"{mode}: the loss reaches the gates and the adapter itself stays frozen ({sum(got)}/{len(got)})")
        gates.release()
        check(all(torch.equal(l.lora_A, before[p][0]) and torch.equal(l.lora_B, before[p][1]) and l.lora_A.requires_grad and isinstance(l.lora_A, torch.nn.Parameter)
                  for p, l in layers.items()), f"{mode}: the parameters come back unchanged")

    for mode in ("element", "component"):
        gates = Gates(layers, mode)
        gates.engage()
        den(x, t, c, y).pow(2).mean().backward()
        gates.collect()
        mine = {p: {k: v.grad.clone() for k, v in per.items()} for p, per in gates.logits.items()}
        for per in gates.logits.values():  # the same gradient through an ordinary autograd graph
            for v in per.values():
                v.grad = None
        for p, layer in layers.items():
            a, b = gates.params[p]["lora_A"], gates.params[p]["lora_B"]
            g = {k: torch.sigmoid(v) for k, v in gates.logits[p].items()}
            layer.lora_A = a * (g["a"].view(-1, *([1] * (a.ndim - 1))) if mode == "component" else g["a"])
            layer.lora_B = b if mode == "component" else b * g["b"]
        den(x, t, c, y).pow(2).mean().backward()
        gates.release()
        scale = max(float(v.abs().max()) for per in mine.values() for v in per.values())
        check(all(torch.allclose(mine[p][k], gates.logits[p][k].grad, atol=1e-4 * scale) for p in mine for k in mine[p]) and scale > 0,
              f"{mode}: the hand-applied chain rule equals the gradient through an autograd graph")
    for mode, expected in (("element", 0.2), ("component", 0.25)):
        for layer in layers.values():
            layer.lora_A.grad = None
        cleaner = LoraCleaner(layers, keep=expected, steps=3, mode=mode, lr=0.5, freeze=True)

        def backward(batch):
            loss = den(x, t, c, y).pow(2).mean()
            loss.backward()
            return float(loss.detach())

        summary = cleaner.run(lambda: {}, backward, lambda: None)
        kept = {k: sum(int(cleaner._kept[p][k].sum()) for p in layers) / sum(cleaner._kept[p][k].numel() for p in layers) for k in cleaner._kept[next(iter(layers))]}
        check(all(abs(v - expected) < 0.02 for v in kept.values()) and summary["steps"] == 3, f"{mode}: about the asked share is kept ({kept})")
        before_update = {p: (l.lora_A.detach().clone(), l.lora_B.detach().clone()) for p, l in layers.items()}
        with torch.no_grad():
            for layer in layers.values():
                layer.lora_A.add_(1.0)
                layer.lora_B.add_(1.0)
        cleaner.after_update()
        same = [bool((mask_equal := torch.equal(torch.where(m, l.lora_A, before_update[p][0]), before_update[p][0])))
                for p, l in layers.items() for m in [cleaner._frozen[p]["lora_A"][0]]]
        check(all(same) and any(not torch.equal(l.lora_A, before_update[p][0]) for p, l in layers.items()), f"{mode}: after an update the kept weights are put back and the others stay free")
    with tempfile.TemporaryDirectory() as tmp:
        saved = []
        cleaner = LoraCleaner(layers, keep=0.3, steps=2, mode="element", freeze=True, directory=tmp, save=saved.append)
        cleaner.run(lambda: {}, lambda batch: float((den(x, t, c, y).pow(2).mean().backward() or 0.0)), lambda: None)
        again = LoraCleaner(layers, keep=0.3, steps=2, mode="element", freeze=True, directory=tmp)
        again.restore()
        check(saved and saved[0].endswith("lora_cleaned.safetensors") and all(torch.equal(again._frozen[p][n][0], cleaner._frozen[p][n][0]) for p in layers for n in ("lora_A", "lora_B")),
              "the kept mask is written and read back for a resumed run, and the cleaned adapter is saved")
        raises(ValueError, lambda: LoraCleaner(layers, keep=0.3, freeze=True, directory=tmp + "/missing").restore(), "a resume without the mask is refused", "missing")
    raises(ValueError, lambda: LoraCleaner(layers, keep=1.5), "a keep share outside 0..1 is refused", "keep fraction")


if __name__ == "__main__":
    run(scaling_and_gate, layer_cases, merging_into_weights, aligned_start, int8_base_weights, nf4_base_weights, activation_compute, continue_from_file, lora_cleaner, sdxl_plan_and_injection)
