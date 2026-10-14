"""Self-checks for next/families/general/adapters (CPU, tiny layers).

    python -m next.tests.check_adapters
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..families.general.adapters.gate import TimestepGate, compute_gate
from ..families.general.adapters.layers import AdaptedConv2d, AdaptedLinear
from ..families.general.adapters.scaling import scaling_factor
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


if __name__ == "__main__":
    run(scaling_and_gate, layer_cases)
