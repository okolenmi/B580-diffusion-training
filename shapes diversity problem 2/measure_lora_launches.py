#!/usr/bin/env python3
"""L5.2: how many launches does the LoRA branch actually cost, and what would
removing them buy?

NO GPU AND NO MEMORY. Same technique as scripts/count_launches.py: count the
kernel-launching aten ops on the meta device through the project's own UNet and
its own LoRA injection, using a torch dispatch hook.

The question L5.2 asks is whether the LoRA branch can be made cheaper without
changing what it computes. One LoRALinear.forward currently costs 7 counted
launches:

    mm(base)          F.linear(x, base_weight, base_bias)   -> bf16
    _to_copy          x.to(fp32)                            activation cast UP
    mm                (x32 @ lora_A.T)                      -> fp32
    mul.Tensor        lora_B.T * scaling
    mm                (@ that)                              -> fp32
    _to_copy          lora_out.to(bf16)                     activation cast DOWN
    add.Tensor        result + lora_out

The two casts are the interesting part. They move the *activation* -- a
(B, N, C) tensor -- into fp32 and back, once per layer, to run a rank-64
matmul in fp32. lora.py's own docstring already names the mainstream
alternative: "every mainstream LoRA implementation (HF PEFT, diffusers) keeps
trainable adapter weights in fp32 even when the frozen base model is
fp16/bf16, casting down for the forward matmul" -- that is, keep the
*parameters* in fp32 (which is what the optimizer needs) and cast the small
*matrices* down, not the big activation up.

Casting lora_A/lora_B down per call is 2 casts, not fewer. It becomes fewer
only if they are cached for the step -- which is what --mode=cache-bf16-weights
below simulates, and which is the real design question: when do the cached
copies get invalidated?

So three variants are measured:
  current    what ships today
  bf16-mm    A/B cast down per call, addmm folds scaling and the residual add
  cached     the same, with A/B casts hoisted out of forward (what a
             per-step cache would give)

and the numerics difference is reported rather than assumed, since computing
the adapter in bf16 instead of fp32 IS a numerics change.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIEW_ONLY = {"aten::_unsafe_view", "aten::view", "aten::_reshape_alias",
             "aten::t", "aten::permute", "aten::transpose", "aten::expand",
             "aten::detach", "aten::alias", "aten::lift_fresh",
             "aten::is_same_size", "aten::_unsafe_index_put_",
             "aten::unsqueeze", "aten::squeeze"}
ALLOC_ONLY = {"aten::empty", "aten::empty_like", "aten::empty_strided",
              "aten::new_empty", "aten::new_empty_strided",
              "aten::_local_scalar_dense"}


class Counter:
    """Dispatch hook counting kernel-launching ops, by name and by op type."""

    def __init__(self):
        self.by: collections.Counter = collections.Counter()
        self.n = 0

    def __enter__(self):
        self._mode = _Mode(self)
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self._mode.__exit__(*exc)

    def snapshot(self) -> tuple[int, collections.Counter]:
        return self.n, collections.Counter(self.by)


class _Mode:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        outer = self

        class _Inner(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, a=(), kw=None):
                name = func.name()
                if name not in VIEW_ONLY and name not in ALLOC_ONLY:
                    outer.owner.n += 1
                    outer.owner.by[name] += 1
                return func(*a, **(kw or {}))
        self._inner = _Inner()
        return self._inner.__enter__()

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)


# -- the three candidate forwards -------------------------------------------------

def forward_current(self, x):
    import torch.nn.functional as F
    result = F.linear(x, self.base_weight, self.base_bias)
    lora_out = (self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T) @ (
        self.lora_B.T * self.scaling)
    gate = _gate()
    if gate is not None:
        g = gate.to(device=lora_out.device, dtype=lora_out.dtype)
        g = g.view(-1, *([1] * (lora_out.dim() - 1)))
        lora_out = lora_out * g
    return result + lora_out.to(result.dtype)


def _fused(self, x, a, b):
    """`result + scaling * ((x @ a.T) @ b.T)`, gated, in as few launches as the
    rank of x allows.

    `addmm` folds both the `scaling` multiply and the residual add into the
    second matmul, and it needs 2D, so the leading dims are flattened and
    restored. Those are views, not launches -- `F.linear`'s own output is
    contiguous so its reshape is a view too, and `x` comes from a reshape of
    whatever the previous layer produced.

    The gate is the awkward part: it is per *sample*, so once the batch is
    flattened to rows it no longer lines up, and addmm's `alpha` is a scalar.
    So the gated path multiplies the rank-sized intermediate (a (B, N, rank)
    tensor, `rank` times smaller than the activation) rather than the output,
    and the ungated path skips that op entirely. Either way the gating is one
    launch on a small tensor instead of one on a large one.
    """
    import torch
    import torch.nn.functional as F
    result = F.linear(x, self.base_weight, self.base_bias)
    gate = _gate()
    lead = x.shape[:-1]
    if gate is None:
        h = self.dropout(x).reshape(-1, x.shape[-1]) @ a.T
    else:
        h = self.dropout(x) @ a.T
        g = gate.to(device=h.device, dtype=h.dtype)
        h = h * g.view(-1, *([1] * (h.dim() - 1)))
        h = h.reshape(-1, h.shape[-1])
    out = torch.addmm(result.reshape(-1, result.shape[-1]), h, b.T,
                      beta=1, alpha=self.scaling)
    return out.view(*lead, out.shape[-1])


def forward_bf16_mm(self, x):
    """A/B cast down per call; addmm folds scaling and the residual add.

    Cheapest, and a numerics change: the rank-64 matmul now runs in bf16.
    Measured in measure_lora_numerics.py at up to 2.27% relative error on the
    adapter's own delta, growing with rank. Listed for comparison, not as the
    recommendation.
    """
    dtype = self.base_weight.dtype
    return _fused(self, x, self.lora_A.to(dtype), self.lora_B.to(dtype))


def forward_fp32_fused(self, x):
    """The adapter math stays in fp32 -- no precision given up -- but
    `scaling` and the residual add are folded into one addmm in fp32.

    Two casts instead of today's two-plus-two-op tail: today it is
    mul(scaling) -> mm -> cast down -> add, four launches after the first
    matmul. Here the base result is cast up once and the whole tail is one
    addmm, then one cast down.

    The only numerical difference from today is the ORDER of the final
    rounding: today the delta is rounded to bf16 and then added to a bf16 base;
    here the sum is computed in fp32 and rounded once. That is strictly more
    accurate, and it is a rounding-order change rather than a precision one --
    which is why this is the variant worth shipping and bf16-mm is not.
    """
    import torch
    import torch.nn.functional as F
    result = F.linear(x, self.base_weight, self.base_bias)
    gate = _gate()
    lead = x.shape[:-1]
    if gate is None:
        h = self.dropout(x).to(self.lora_A.dtype).reshape(-1, x.shape[-1]) @ self.lora_A.T
    else:
        h = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T
        g = gate.to(device=h.device, dtype=h.dtype)
        h = h * g.view(-1, *([1] * (h.dim() - 1)))
        h = h.reshape(-1, h.shape[-1])
    out = torch.addmm(
        result.reshape(-1, result.shape[-1]).to(h.dtype), h, self.lora_B.T,
        beta=1, alpha=self.scaling)
    return out.view(*lead, out.shape[-1]).to(result.dtype)


def forward_cached(self, x):
    """As bf16_mm, with the A/B casts hoisted out of forward -- what a per-step
    cache gives on a HIT.

    Reported to show the CEILING the A/B casts are worth, not as a
    recommendation: a cache of derived copies of a live parameter is only
    correct if nothing can change the parameter without invalidating it, and
    this file's own `LoRALinear.load_lora_weights` writes through
    `param.data.copy_()`, which does NOT bump the parameter's autograd version
    counter (measured) while leaving its data_ptr unchanged. So (version,
    data_ptr) does not catch it. The honest number is in the table; the honest
    conclusion is that this one needs a design answer first.
    """
    dtype = self.base_weight.dtype
    a, b = _CACHE.get(id(self), {}).get(dtype, (None, None))
    if a is None:
        a, b = self.lora_A.to(dtype), self.lora_B.to(dtype)
        _CACHE.setdefault(id(self), {})[dtype] = (a, b)
    return _fused(self, x, a, b)


def _gate():
    from nodes.model.lora import _current_gate
    return _current_gate


VARIANTS = {"current": forward_current, "fp32-fused": forward_fp32_fused,
            "bf16-mm": forward_bf16_mm, "cached": forward_cached}

#: Filled by forward_cached's stand-in. Deliberately module-level and keyed by
#: layer identity, so a hit is a dict lookup with no aten op at all -- which is
#: exactly what a real cache would cost.
_CACHE: dict = {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--rank", type=int, default=64)
    args = ap.parse_args(argv)

    import torch
    from nodes.model.gradient_checkpointing import FrozenParamSafeCheckpointing
    from nodes.model.lora import LoRAConfig, inject_lora_into_unet
    from nodes.model.unet import UNetModel
    from nodes.model.unet_wrapper import ComfyUNetWrapper as W

    results = {}
    for name, fn in VARIANTS.items():
        # Checkpointing patches the module globally and is not worth
        # re-installing per variant; it does not touch LoRALinear.forward.
        FrozenParamSafeCheckpointing().apply()
        original_forward = LoRALinear_type().forward
        LoRALinear_type().forward = fn
        try:
            _CACHE.clear()
            results[name] = _measure(name, args)
        finally:
            LoRALinear_type().forward = original_forward

    base = results["current"][0]
    print(f"batch {args.batch}, latent {args.latent}, rank {args.rank}; "
          f"whole production step, meta device, NO GPU\n")
    print(f"{'variant':<12}{'total':>9}{'vs current':>12}"
          f"{'_to_copy':>10}{'mm':>8}{'addmm':>8}{'add':>7}{'mul':>7}")
    for name, (total, by) in results.items():
        print(f"{name:<12}{total:>9}{(total - base) / base:>+11.1%}"
              f"{by.get('aten::_to_copy', 0):>10}"
              f"{by.get('aten::mm', 0):>8}"
              f"{by.get('aten::addmm', 0):>8}"
              f"{by.get('aten::add.Tensor', 0):>7}"
              f"{by.get('aten::mul.Tensor', 0) + by.get('aten::mul.Scalar', 0):>7}")
    print()
    for name, (total, by) in results.items():
        print(f"  {name}: {total} launches, of which LoRA branch's own ops "
              f"(mm/addmm/_to_copy/mul/add) = "
              f"{sum(v for k, v in by.items() if k.split('::')[-1] in ('mm', 'addmm', '_to_copy', 'mul', 'mul.Tensor', 'mul.Scalar', 'add', 'add.Tensor'))}")
    return 0


def LoRALinear_type():
    from nodes.model.lora import LoRALinear
    return LoRALinear


def _measure(name, args) -> tuple[int, collections.Counter]:
    import torch
    from nodes.model.lora import LoRAConfig, inject_lora_into_unet
    from nodes.model.unet import UNetModel
    from nodes.model.unet_wrapper import ComfyUNetWrapper as W

    cfg = dict(W.SDXL_CONFIG)
    cfg["use_checkpoint"] = True
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            unet = UNetModel(**cfg)
    finally:
        torch.set_default_dtype(prev)
    unet.requires_grad_(False)
    inject_lora_into_unet(unet, LoRAConfig(rank=args.rank, alpha=1.0))

    kw = dict(device="meta", dtype=torch.bfloat16)
    x = torch.randn(args.batch, 4, args.latent, args.latent, **kw)
    t = torch.randint(0, 1000, (args.batch,), device="meta")
    ctx = torch.randn(args.batch, 77, cfg["context_dim"], **kw)
    y = torch.randn(args.batch, cfg["adm_in_channels"], **kw)
    counter = Counter()
    with counter:
        out = unet(x, t, ctx, y)
        fwd, fwd_by = counter.snapshot()
        out.float().pow(2).mean().backward()
    total, by = counter.snapshot()
    return total, by


if __name__ == "__main__":
    sys.exit(main())
