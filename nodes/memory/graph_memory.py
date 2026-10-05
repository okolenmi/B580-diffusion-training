"""GraphMemory -- what one graph run may use on the device (MEM-05).

Built by the worker (and the in-process gateway) from the spawn
arguments *before anything loads* -- before the graph JSON is read,
before discovery, before torch -- so a bad number is refused while
there is nothing loaded yet, and every later consumer asks this object
instead of re-deriving the answer. Nodes reach it through
``ExecutionContext.memory``.

Two numbers, each optional but never silently:

* ``grant_mb`` -- the device-MB claim admission held for this run. The
  ledger reserved it, but the ledger does not know foreign users
  (ComfyUI, the desktop), which is what the physical check compares it
  against: the card's real free memory.
* ``budget_mb`` -- the allocator MB this process may use: the run's
  stated/observed demand, the input to the allocator's per-process
  fraction backstop.

``None`` on either means *never supplied* -- an explicit UNKNOWN the
consumer names, never a zero to guess with (a zero claim would read as
"fits everything", which is the opposite of "nobody told me", task
rule 2).

The number checks mirror ``backend/domain/memory_settings.py``'s
``_budget`` on purpose: nodes/ never imports backend/ (task rule 9), so
the domain's validation is reimplemented at this layer instead of
shared. It matters here more than at the edge: a NaN grant would make
every ``free < grant`` comparison False, so the physical check would
pass on garbage -- the check itself is only as honest as this
constructor.
"""

from __future__ import annotations

import math


def _size(name: str, value) -> float | None:
    """Validate one memory size; ``None`` passes through as unknown.

    Rejects what a device size cannot be: a boolean (``True`` is an
    ``int`` in Python), a non-number, a non-finite number (NaN and
    infinity poison every comparison they join), a number past the
    float range (``10**400`` overflows into an ``OverflowError`` that
    would escape as a crash instead of a refusal), and zero or below --
    no claim is ``None``, not ``0``.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: {value!r} is not a number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(
            f"{name}: {value!r} is too large to be a device size"
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"{name}: {value!r} is not a finite number")
    if number <= 0:
        raise ValueError(f"{name}: {number} MB is not above zero")
    return number


class GraphMemory:
    """The two numbers one run is allowed, with their unknowns named.

    See the module docstring. Deliberately small: the resident registry
    and the lease policy are added here as the rework lands, and the
    physical check and allocator backstop hang off these two properties.
    """

    def __init__(
        self, *, grant_mb: float | None = None, budget_mb: float | None = None
    ) -> None:
        self._grant_mb = _size("grant_mb", grant_mb)
        self._budget_mb = _size("budget_mb", budget_mb)

    @property
    def grant_mb(self) -> float | None:
        """The device-MB claim admission held, or ``None`` (unknown)."""
        return self._grant_mb

    @property
    def budget_mb(self) -> float | None:
        """The allocator MB this process may use, or ``None`` (unknown)."""
        return self._budget_mb
