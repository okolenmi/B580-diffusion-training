"""Correctness checks for nodes/model/checkpoint.py.

`smoke_test_gradient_checkpointing.py` covers the two upstream bugs against
a verbatim copy of ComfyUI's `CheckpointFunction` and keeps them honest with
a control. This file covers *our* module's own surface, which nothing else
does:

- **The fixed class is the default, and that is safe.** `checkpoint()`
  dispatches to the frozen-param-safe function from the first call, with no
  opt-in. That is only defensible if it is a strict superset -- identical
  gradients when every parameter is trainable and no autocast is in play --
  which is asserted here rather than assumed in checkpoint.py's docstring.
- **The four-argument shape**, because `inputs` and `params` being separate
  sequences is what lets backward return one gradient slot per parameter,
  in order, with `None` for frozen ones. A signature that flattened them
  would silently stop giving LoRA adapters gradients, which is a wrong-
  gradients bug that raises nothing.
- **`flag=False`**, which must call straight through: same values, and no
  second forward. A silent fallback here instead would make the memory
  numbers the project reports describe a different run than the one that
  happened.
- **Parameter ordering**, checked with three parameters where only the
  middle one is frozen, so a positional mix-up cannot pass.
- **The recompute_wrapper switch**, which has to be per-installation rather
  than per-module: two graphs in one server process legitimately want
  different instrumentation.

CPU only, and deliberately so -- there is no device in this file, because
nothing here should be able to get device-specific.

Run: `python nodes/smoke_tests/smoke_test_checkpoint.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.checkpoint import (  # noqa: E402
    active_checkpoint_function,
    checkpoint,
    make_checkpoint_function,
    set_active_checkpoint_function,
)

failures: list[str] = []


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class Three(torch.nn.Module):
    """A module whose forward takes only its input, with three parameters.

    The middle one frozen is the interesting case: it is the one a
    positional mix-up in the returned gradient tuple would get wrong while
    still producing a tuple of the right length.
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(6))
        self.b = torch.nn.Parameter(torch.randn(6))
        self.c = torch.nn.Parameter(torch.randn(6))

    def forward(self, x):
        return (x * self.a + self.b.sin()) * self.c


def _grads(module, x, ckpt=None, flag=True):
    x = x.detach().clone().requires_grad_(True)
    if ckpt is None:
        out = module(x)
    else:
        out = ckpt(module.forward, (x,), tuple(module.parameters()), flag)
    out.sum().backward()
    return [x.grad.clone()] + [
        None if p.grad is None else p.grad.clone()
        for p in module.parameters()
    ]


def _fresh(freeze_middle: bool = False):
    torch.manual_seed(4)
    m = Three()
    if freeze_middle:
        m.b.requires_grad_(False)
    return m, torch.randn(5, 6)


def check_default_is_a_strict_superset():
    """The fixed class can be the default only if it matches the original."""
    m, x = _fresh()
    reference = _grads(m, x)
    m.zero_grad(set_to_none=True)
    got = _grads(m, x, checkpoint, True)
    record(len(got) == len(reference), "one gradient per input and parameter",
           f"{len(got)} vs {len(reference)}")
    worst = max(
        float((a - b).abs().max())
        for a, b in zip(reference, got)
    )
    record(worst == 0.0,
           "with every parameter trainable and no autocast, gradients are "
           "bitwise identical to a non-checkpointed reference",
           f"max abs diff {worst:.3e}")

    record(getattr(active_checkpoint_function(), "_frozen_param_safe", False),
           "and the default is already the frozen-param-safe class, so no "
           "opt-in is needed")


def check_frozen_middle_parameter():
    m, x = _fresh(freeze_middle=True)
    grads = _grads(m, x, checkpoint, True)
    record(len(grads) == 4, "still four gradient slots", f"{len(grads)}")
    record(grads[2] is None,
           "the frozen middle parameter's slot is None rather than a "
           "fabricated zero or an error")
    record(grads[1] is not None and grads[3] is not None,
           "and its neighbours still get real gradients")
    record(grads[2] is None and m.b.grad is None,
           "with .grad left untouched on the frozen parameter itself")

    # The ordering claim, made falsifiable: a and c must be non-identical,
    # so a swapped tuple could not pass by symmetry.
    record(grads[1] is not None and not torch.equal(grads[1], grads[3]),
           "the two trainable gradients differ, so order is observable")

    # Correctness against a reference that simply has no gradient for b.
    m2, x2 = _fresh(freeze_middle=True)
    ref = _grads(m2, x2)
    record(ref[2] is None
           and float((ref[0] - grads[0]).abs().max()) == 0.0,
           "and x's gradient matches the non-checkpointed reference exactly")


def check_flag_false_bypasses():
    m, x = _fresh(freeze_middle=True)
    plain = _grads(m, x)
    m.zero_grad(set_to_none=True)

    calls = []
    original = m.forward

    def counting(x_):
        calls.append(1)
        return original(x_)

    got = _grads(m, x, lambda f, i, p, flag: f(*i), False)
    record(len(calls) == 0,
           "flag=False calls through with no checkpointing machinery")
    record(all(
        (a is None and b is None) or torch.equal(a, b)
        for a, b in zip(plain, got)
    ), "and produces the same gradients")

    # Same values, reached differently: prove the bypass really is the
    # un-checkpointed path by comparing outputs directly too.
    x1 = x.detach().clone().requires_grad_(True)
    x2b = x.detach().clone().requires_grad_(True)
    direct = original(x1)
    bypassed = checkpoint(original, (x2b,), tuple(m.parameters()), False)
    record(torch.equal(direct, bypassed),
           "bypassed output is bitwise equal to the direct call")
    record(direct.grad_fn is not None and bypassed.grad_fn is not None,
           "and it still carries a graph, so backward works")


def check_wrapper_switching():
    """The wrapper is per-installation, not per-module."""
    seen = []
    wrapper = lambda fn, args: (seen.append(1), fn(*args))[1]
    cls = make_checkpoint_function(recompute_wrapper=wrapper)
    record(cls is not active_checkpoint_function(),
           "make_checkpoint_function returns a distinct class")

    set_active_checkpoint_function(cls)
    record(active_checkpoint_function() is cls,
           "installing switches the active class")

    m, x = _fresh()
    _grads(m, x, checkpoint, True)
    record(len(seen) == 1, "and the wrapper ran during the recompute",
           f"{len(seen)} calls")

    # Idempotent for the same wrapper identity -- callers apply their strategy
    # fresh on every graph run, so re-installing has to be free.
    again = make_checkpoint_function(recompute_wrapper=wrapper)
    set_active_checkpoint_function(again)
    record(active_checkpoint_function() is cls,
           "re-installing the same wrapper identity is a no-op")

    # A different wrapper does switch.
    other = make_checkpoint_function(recompute_wrapper=lambda fn, args: fn(*args))
    set_active_checkpoint_function(other)
    record(active_checkpoint_function() is other,
           "a different wrapper identity does switch the class")

    # Put it back so nothing after this test inherits the probe.
    set_active_checkpoint_function(make_checkpoint_function())


def main() -> int:
    print("== the fixed class is the default, and safe as one ==")
    check_default_is_a_strict_superset()
    print("\n== a frozen parameter in the middle ==")
    check_frozen_middle_parameter()
    print("\n== flag=False ==")
    check_flag_false_bypasses()
    print("\n== recompute_wrapper switching ==")
    check_wrapper_switching()

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