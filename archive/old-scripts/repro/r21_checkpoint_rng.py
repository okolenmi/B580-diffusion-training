"""R5-01: does the checkpoint recompute replay the forward's randomness?

A checkpointed block runs its forward under `no_grad`, throws the
activations away, and re-runs it in backward to rebuild the graph. If
anything inside the block draws random numbers -- `nn.Dropout`, which
`nodes/model/lora.py`'s `LoRALinear` builds when `dropout > 0` -- then
without an RNG replay the recompute draws a *different* mask. The upstream
gradient belongs to mask A and the recomputed Jacobian to mask B, so the
result is a finite, plausible, entirely wrong gradient.

`dropout` is reachable: `tuning.dropout` in `config.toml`, labelled "LoRA
Dropout" in the settings UI, and `use_checkpoint` is on by default in
`SDXL_CONFIG` because it is what makes 1024/batch 2 fit on a 12 GB card.

The control matters as much as the measurement. `STOCK` below is the
checkpoint function as it was *before* the fix, and it is here so this
script can fail: a check that only ever compares the fixed version against
itself would pass whatever the fix did.

Run from the repo root:  python scripts/repro/r21_checkpoint_rng.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.model.checkpoint import active_checkpoint_function  # noqa: E402

FIXED = active_checkpoint_function()


class STOCK(torch.autograd.Function):
    """`nodes/model/checkpoint.py` before the RNG replay, kept as a control.

    One gradient slot per input *parameter*, `None` for the frozen ones, which
    is the calling convention `(run_function, length, *args)` implies.
    """

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            return run_function(*ctx.input_tensors)

    @staticmethod
    def backward(ctx, *output_grads):
        xs = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            out = ctx.run_function(*[x.view_as(x) for x in xs])
        trainable = [p for p in ctx.input_params if p.requires_grad]
        got = torch.autograd.grad(out, xs + trainable, output_grads,
                                  allow_unused=True)
        tensor_grads = got[:len(xs)]
        rest = iter(got[len(xs):])
        param_grads = tuple(next(rest) if p.requires_grad else None
                            for p in ctx.input_params)
        return (None, None) + tuple(tensor_grads) + param_grads


class Block(torch.nn.Module):
    """Frozen base Linear plus a LoRA path with dropout, as `lora.py` builds.

    The base is frozen, so this exercises both fixes at once: the frozen
    parameter slots and the RNG replay.
    """

    def __init__(self, p: float, device: str = "cpu"):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.base = torch.nn.Linear(64, 64, device=device)
        self.base.requires_grad_(False)
        self.A = torch.nn.Parameter(
            (torch.randn(8, 64, generator=g) * 0.1).to(device))
        self.B = torch.nn.Parameter(
            (torch.randn(64, 8, generator=g) * 0.1).to(device))
        self.drop = torch.nn.Dropout(p)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.T) @ self.B.T


def gradient(block, how: str, x0: torch.Tensor, seed: int = 123):
    """Gradients for A and B from one training-mode pass."""
    block.train()
    block.A.grad = block.B.grad = None
    x = x0.clone().requires_grad_(True)
    torch.manual_seed(seed)
    if how == "plain":
        y = block(x)
    elif how == "torch":
        y = torch.utils.checkpoint.checkpoint(
            block.forward, x, use_reentrant=True, preserve_rng_state=True)
    else:
        function = {"fixed": FIXED, "stock": STOCK}[how]
        y = function.apply(block.forward, 1, x, *list(block.parameters()))
    y.pow(2).sum().backward()
    return block.A.grad.clone(), block.B.grad.clone()


def relative_error(got, ref) -> float:
    return max(((got[i] - ref[i]).norm() / ref[i].norm()).item()
               for i in (0, 1))


def seed_both(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "xpu" and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report(device: str) -> int:
    """Measure on one device. Returns the number of failing cases."""
    if device != "cpu" and not getattr(torch, device).is_available():
        print(f"\n== {device} ==\n  SKIP: no {device} device")
        return 0

    print(f"\n== {device} ==")
    x0 = torch.randn(16, 64, generator=torch.Generator().manual_seed(1)).to(device)
    header = (f"  {'dropout':8} {'fixed':>12} {'stock (control)':>17} "
              f"{'torch.utils.ckpt':>17}")
    print(header)
    failures = 0
    for p in (0.0, 0.1, 0.5):
        block = Block(p, device=device)
        ref = gradient(block, "plain", x0)
        fixed = relative_error(gradient(block, "fixed", x0), ref)
        stock = relative_error(gradient(block, "stock", x0), ref)
        theirs = relative_error(gradient(block, "torch", x0), ref)
        print(f"  {p:<8} {fixed:>12.3e} {stock:>17.3e} {theirs:>17.3e}")
        # The fixed version must be exact; the stock one must visibly not be,
        # or this table says nothing.
        if fixed > 1e-6:
            failures += 1
        if p > 0 and stock < 1e-3:
            failures += 1
        if p > 0 and theirs > 1e-6:
            failures += 1
    return failures


def rng_state_is_consumed_the_same_way(device: str) -> bool:
    """A seeded run must consume the same randomness either way.

    The checkpointed forward runs the block exactly once, so it must leave
    the generator in the same place an uncheckpointed forward would. Nothing
    restores state at the end of forward, which is what makes this true --
    but "nothing does the obvious wrong thing" is worth measuring.
    """
    block = Block(0.5, device=device)
    x0 = torch.randn(16, 64, generator=torch.Generator().manual_seed(1)).to(device)

    seed_both(11, device)
    with torch.no_grad():
        block(x0)
    after_plain = (torch.get_rng_state().clone(),
                   torch.xpu.get_rng_state(0).clone()
                   if device == "xpu" and torch.xpu.is_available() else None)

    seed_both(11, device)
    with torch.no_grad():
        FIXED.apply(block.forward, 1, x0, *list(block.parameters()))
    after_ckpt = (torch.get_rng_state().clone(),
                  torch.xpu.get_rng_state(0).clone()
                  if device == "xpu" and torch.xpu.is_available() else None)

    same_cpu = torch.equal(after_plain[0], after_ckpt[0])
    same_dev = (after_plain[1] is None
                or torch.equal(after_plain[1], after_ckpt[1]))
    return same_cpu and same_dev


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print(f"  checkpoint function under test: {FIXED.__name__}")

    failures = report("cpu")
    for device in ("xpu", "cuda"):
        failures += report(device)

    print("\n== the generator ends where an uncheckpointed forward leaves it ==")
    for device in ("cpu", "xpu", "cuda"):
        if device != "cpu" and not getattr(torch, device).is_available():
            print(f"  SKIP: {device}")
            continue
        ok = rng_state_is_consumed_the_same_way(device)
        print(f"  {'PASS' if ok else 'FAIL'}: {device}: a seeded forward's "
              f"end state is the same checkpointed as uncheckpointed")
        if not ok:
            failures += 1

    print("\n" + "=" * 60)
    if failures:
        print(f"R5-01: {failures} FAILING CASE(S)")
        return 1
    print("R5-01: the recompute replays the forward's randomness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
