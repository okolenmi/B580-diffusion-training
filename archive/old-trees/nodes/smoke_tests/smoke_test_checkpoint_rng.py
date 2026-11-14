"""Activation checkpointing must replay the forward's randomness.

Design doc 12, section 7.3-A, and round-5 finding R5-01.

A checkpointed block runs its forward under `no_grad`, throws the
activations away, and re-runs it in backward to rebuild the graph. If
anything inside draws random numbers, the recompute must draw the *same*
ones. If it does not, the upstream gradient belongs to mask A and the
recomputed Jacobian to mask B, and the result is a finite, plausible,
entirely wrong gradient.

**This is live, not latent.** `tuning.dropout` is a user-facing config key,
labelled "LoRA Dropout" in the settings UI and documented in
`config.example.toml`; `nodes/model/lora.py`'s `LoRALinear` builds
`nn.Dropout(dropout)` from it when the value is above zero. And
`use_checkpoint` is on by default in `SDXL_CONFIG`, because it is what
makes 1024 / batch 2 fit on a 12 GB card. So "LoRA dropout > 0 with
checkpointing on" is a supported configuration and it produced wrong
gradients.

Measured against a non-checkpointed reference, same inputs and seed:

    dropout   before the fix   torch.utils.checkpoint
    0.0        0.000e+00        0.000e+00
    0.1        4.343e-01        0.000e+00
    0.5        9.337e-01        0.000e+00

**The control is the point of this file.** `_NoRngCheckpoint` below is the
same function without the replay, and every measurement here is reported
beside it. A test that only compared the fixed version against itself would
pass no matter what the fix did -- and the first version of this file
asserted bitwise equality and failed, because the *uncheckpointed* run is
not bitwise equal to itself on this hardware (the B580's kernels are not
bit-reproducible). Comparing against a control that visibly disagrees is
what makes the measurement mean something.

CPU only, like `smoke_test_checkpoint.py`, and deliberately so: the device
path has its own measurement in `scripts/repro/r21_checkpoint_rng.py`,
which runs on the card and skips when there is none. What is checked here
is the arithmetic, which has no device in it.

Run: `python nodes/smoke_tests/smoke_test_checkpoint_rng.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.checkpoint import checkpoint  # noqa: E402

failures: list[str] = []

#: float32 resolves to about 1.2e-7 relative. A correctly replayed mask gives
#: bitwise equality on CPU, so this is a loose bound rather than a tight one:
#: it is here to catch a wrong mask (4.343e-01), not to police rounding.
TOLERANCE = 1e-6


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class _NoRngCheckpoint(torch.autograd.Function):
    """`nodes/model/checkpoint.py` without the RNG replay, kept as a control.

    Otherwise identical -- same frozen-parameter slots, same recompute, same
    `(run_function, length, *args)` convention -- so the only difference
    between this and the real thing is the replay. One gradient slot per
    input parameter, `None` for the frozen ones.
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


class LoRALike(torch.nn.Module):
    """Frozen base Linear plus a LoRA path with dropout, as lora.py builds.

    The base is frozen, so one block exercises the frozen-parameter fix and
    the RNG replay together -- which is how a real LoRA block is built.
    """

    def __init__(self, p: float):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.base = torch.nn.Linear(64, 64)
        self.base.requires_grad_(False)
        self.A = torch.nn.Parameter(torch.randn(8, 64, generator=g) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(64, 8, generator=g) * 0.1)
        self.drop = torch.nn.Dropout(p)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.T) @ self.B.T


def _gradients(module, x0, how: str, seed: int = 123):
    """Gradients for A and B from one training-mode pass."""
    module.train()
    module.A.grad = module.B.grad = None
    x = x0.clone().requires_grad_(True)
    torch.manual_seed(seed)
    if how == "plain":
        y = module(x)
    elif how == "torch":
        y = torch.utils.checkpoint.checkpoint(
            module.forward, x, use_reentrant=True, preserve_rng_state=True)
    elif how == "ours":
        y = checkpoint(module.forward, (x,), tuple(module.parameters()), True)
    else:
        # An autograd.Function is applied, not called.
        y = _NoRngCheckpoint.apply(module.forward, 1, x,
                                   *tuple(module.parameters()))
    y.pow(2).sum().backward()
    return module.A.grad.clone(), module.B.grad.clone()


def _relative_error(got, ref) -> float:
    return max(((got[i] - ref[i]).norm() / ref[i].norm()).item()
               for i in (0, 1))


def _x0() -> torch.Tensor:
    return torch.randn(16, 64, generator=torch.Generator().manual_seed(1))


def check_dropout_gradients_match():
    """The headline: with dropout on, the replay makes the gradients exact."""
    x0 = _x0()
    print(f"  {'dropout':8} {'ours':>12} {'control':>12} {'torch.utils.ckpt':>18}")
    for p in (0.0, 0.1, 0.5):
        block = LoRALike(p)
        ref = _gradients(block, x0, "plain")
        ours = _relative_error(_gradients(block, x0, "ours"), ref)
        control = _relative_error(_gradients(block, x0, "control"), ref)
        theirs = _relative_error(_gradients(block, x0, "torch"), ref)
        print(f"  {p:<8} {ours:>12.3e} {control:>12.3e} {theirs:>18.3e}")
        record(ours <= TOLERANCE,
               f"dropout {p}: the checkpointed gradient equals the "
               f"uncheckpointed one ({ours:.3e} against {TOLERANCE:.0e})",
               f"{ours:.3e}")
        if p > 0:
            record(control > 1e-3,
                   f"dropout {p}: and the control without the replay visibly "
                   f"disagrees, so this measurement is not vacuous "
                   f"({control:.3e})",
                   f"only {control:.3e} -- the control no longer reproduces "
                   f"the bug, so it cannot prove the fix does anything")


def check_adapters_still_get_gradients():
    """A replayed block must still hand gradients to its adapters.

    Wrapping the recompute in an RNG fork is the kind of change that can
    quietly detach something. A frozen parameter getting no gradient is
    correct; a *trainable* one getting none is the original reason the
    parameters are passed positionally at all.
    """
    block = LoRALike(0.5)
    x0 = _x0()
    _gradients(block, x0, "ours")
    record(block.A.grad is not None and block.B.grad is not None,
           "both adapters have gradients after a checkpointed backward",
           f"A={block.A.grad is None and 'none' or 'present'}")
    record(block.base.weight.grad is None,
           "and the frozen base has none, which is the frozen-parameter "
           "behaviour and not a side effect of the replay")


def check_generator_ends_where_it_would():
    """A seeded forward must consume randomness exactly as before.

    Nothing is restored at the end of forward -- the block runs once, under
    `no_grad`, which consumes no different randomness than an uncheckpointed
    run -- so the generator ends in the same place. That is what keeps a
    seeded training run reproducible across a checkpointing flag change.
    """
    block = LoRALike(0.5)
    x0 = _x0()

    torch.manual_seed(11)
    before_plain = torch.get_rng_state().clone()
    with torch.no_grad():
        block(x0)
    after_plain = torch.get_rng_state().clone()

    torch.manual_seed(11)
    before_ckpt = torch.get_rng_state().clone()
    with torch.no_grad():
        checkpoint(block.forward, (x0,), tuple(block.parameters()), True)
    after_checkpointed = torch.get_rng_state().clone()

    record(torch.equal(after_plain, after_checkpointed),
           "the generator is in the same state after a checkpointed forward "
           "as after an uncheckpointed one",
           "a checkpointed forward consumed a different amount of "
           "randomness, so a seeded run would not be reproducible")

    # And the forward must actually advance it, or the check above compares
    # two untouched states and passes for the wrong reason. Before against
    # after *within* each pass, which is what "consumed" means.
    plain_moved = not torch.equal(before_plain, after_plain)
    ckpt_moved = not torch.equal(before_ckpt, after_checkpointed)
    record(plain_moved and ckpt_moved,
           "and both passes did consume randomness, so that comparison is "
           "between two different states rather than two identical ones",
           f"plain moved={plain_moved}, checkpointed moved={ckpt_moved}")


def check_backward_leaves_the_generator_alone():
    """The fork must wrap only the recompute, and restore on exit.

    If the RNG were pinned across `torch.autograd.grad` as well, the
    caller's generator position after a step would depend on autograd's
    internals. The observable consequence is that two identical uncheckpointed
    steps and two identical checkpointed steps leave the same state.
    """
    block = LoRALike(0.5)
    x0 = _x0()

    def two_steps(use_ckpt: bool) -> torch.Tensor:
        torch.manual_seed(11)
        for _ in range(2):
            _gradients(block, x0, "ours" if use_ckpt else "plain")
        return torch.get_rng_state().clone()

    record(torch.equal(two_steps(False), two_steps(True)),
           "after two full steps the generator is in the same place "
           "checkpointed as uncheckpointed")


def check_deterministic_block_is_unchanged():
    """No randomness inside means no behaviour change and no cost."""
    block = LoRALike(0.0)
    x0 = _x0()
    ref = _gradients(block, x0, "plain")
    got = _gradients(block, x0, "ours")
    record(_relative_error(got, ref) == 0.0,
           "a block with no randomness still gives bitwise identical "
           "gradients",
           f"{_relative_error(got, ref):.3e}")

    record(torch.equal(_gradients(block, x0, "ours")[0], ref[0]),
           "including exactly, which is the pre-existing claim in "
           "smoke_test_checkpoint.py and must not have been weakened")


def check_autocast_and_replay_together():
    """The two fixes must compose: dropout inside autocast, on CPU.

    Both fixes are context managers wrapped around the same recompute, so
    they could nest wrongly. CPU autocast is the honest test of the nesting
    that a CPU-only run can perform; the device path is measured by
    `scripts/repro/r21_checkpoint_rng.py`.
    """
    block = LoRALike(0.5)
    x0 = _x0()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ref = _gradients(block, x0, "plain")
        got = _gradients(block, x0, "ours")
        control = _gradients(block, x0, "control")
    error = _relative_error(got, ref)
    control_error = _relative_error(control, ref)
    record(error <= 1e-2,
           f"with autocast on and dropout at 0.5, the replayed gradient "
           f"matches to {error:.3e} (bfloat16 autocast, so not exact)",
           f"{error:.3e} -- the replay and the autocast re-entry are not "
           f"composing")
    record(control_error > error,
           f"and the control is still worse ({control_error:.3e}), so this "
           f"is not passing because autocast dominates",
           f"control {control_error:.3e} vs ours {error:.3e}")


def main() -> int:
    print("== dropout gradients: the replay against a control that disagrees ==")
    check_dropout_gradients_match()
    print("\n== the adapters still get gradients ==")
    check_adapters_still_get_gradients()
    print("\n== the generator is consumed exactly as before ==")
    check_generator_ends_where_it_would()
    print("\n== and the fork does not leak past the recompute ==")
    check_backward_leaves_the_generator_alone()
    print("\n== a block with no randomness is untouched ==")
    check_deterministic_block_is_unchanged()
    print("\n== autocast and the replay compose ==")
    check_autocast_and_replay_together()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
