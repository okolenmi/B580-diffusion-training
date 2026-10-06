"""Notify a step callback, tolerating a callback that takes no shape.

`on_step` is public: `SupervisedLoRATrainerNode` and
`ManagedLoRATrainerNode` both pass it straight through to whoever built them,
and this repo's own callers include two-argument lambdas
(`nodes/smoke_tests/smoke_test_managed_trainer.py`'s
`lambda s, l: on_steps.append(s)`), `None`, and functions of unknown arity.

So both MonitoringPhase classes now pass a third argument -- the step's latent
shape -- which is what makes a shape-dependent cost measurable at all
(`docs/known-issues/pending-testing.md`'s non-shape-throughput entry could
only be narrowed by reconstructing the shape order offline, and got it wrong).
Adding an argument to a public callback breaks every existing caller, and the
alternatives are both worse: silently dropping the shape for callbacks that
would have accepted it, or editing each caller's signature and so making a
measurement change look like a refactor.

This is the "call with what it accepts" rule: inspect the signature once, cache
the arity, and pass the shape only to callbacks that can receive it. A
`TypeError` raised *inside* a callback must not be swallowed and retried as an
arity problem -- that would re-run the callback and hide a real bug, so only
the argument-binding failure is treated as an arity signal, and only ever on
the first step for a given callback.

Anything unusual in the signature (``*args``, keyword-only, a builtin without an
introspectable signature) gets the full three-argument call, which is the
historical-plus-shape contract this project's own callers are written against.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

#: step callback -> True when it can receive the latent shape as a third
#: positional argument. Cached per callback so the signature is inspected
#: once per run rather than once per step; a bound method is re-created per
#: access, so the cache is keyed on the underlying function where possible.
_accepts_shape: dict[Any, bool] = {}


def _can_take_shape(callback: Callable[..., Any]) -> bool:
    """Whether ``callback`` accepts a third positional argument."""
    key = getattr(callback, "__func__", callback)
    cached = _accepts_shape.get(key)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):
        # Not introspectable (a C builtin, some proxies). Assume the full
        # call: the alternative is silently withholding a measurement.
        verdict = True
    else:
        params = list(sig.parameters.values())
        positional = [
            p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind is p.VAR_POSITIONAL for p in params)
        verdict = has_varargs or len(positional) >= 3
    _accepts_shape[key] = verdict
    return verdict


def notify_step(callback: Callable[..., Any] | None, step: int, loss: float,
                shape: str | None) -> None:
    """Call ``callback(step, loss, shape)``, or ``callback(step, loss)``.

    ``callback`` may be None (no monitoring wanted), which is the common
    unmonitored case and stays a no-op.
    """
    if callback is None:
        return
    if _can_take_shape(callback):
        callback(step, loss, shape)
    else:
        callback(step, loss)
