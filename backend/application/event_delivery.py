"""Delivery classes -- how a client may treat a kind of event.

Three classes, and the distinction is the whole contract:

``lifecycle``
    Something happened that changes what a client should believe.
    ``run_completed``, ``run_failed``, ``run_cancelled``. Never coalesced
    and never dropped for a slow client: a missed ``run_completed`` is a
    row that stays ``running`` forever, in the database and on every
    screen watching it.

``state``
    A report of where something stands. ``run_progressed``. An older
    sample for the same run is *superseded* by a newer one, so a client
    behind may lose one safely.

``delta``
    A partial update that only makes sense in order. ``graph_execution_
    progressed``. Never coalesced, because dropping one corrupts the
    sequence it belongs to.

Anything not listed is **lifecycle**, which is the safe direction to
default: "never drop it" rather than "probably redundant".

This lives in ``application/`` rather than ``presentation/`` because it
is policy about event *kinds*, and the infrastructure bus needs it too --
to decide what is worth ring-buffering for replay
(``infrastructure/events/callback_event_bus.py``). A bus that had to
import presentation to answer that question would be told, by its own
import graph, that the question is not really presentation's.
"""

from __future__ import annotations

STATE = "state"
DELTA = "delta"
LIFECYCLE = "lifecycle"

#: Kinds whose newest value supersedes an older one.
#:
#: **Empty.** The only member was `run_progressed`, which went with the
#: supervised-subprocess route. No remaining event is a periodic sample of
#: something's current position -- `graph_execution_progressed` reports a
#: node *completing*, which is a fact that happened and must arrive, so it
#: is a delta and never coalesced.
#:
#: The class is kept because the rest of the mechanism depends on it:
#: `ClientBuffer` orders its overflow sacrifices by class, and a delta must
#: be distinguishable from a lifecycle event to be protected from both
#: coalescing and eviction. Adding a state event is a one-line entry here,
#: and `coalesce_key` already gives it a key.
STATE_EVENT_TYPES: frozenset[str] = frozenset()

#: Kinds that are only meaningful in order, so never coalesced.
DELTA_EVENT_TYPES = frozenset({"graph_execution_progressed"})


def delivery_class(event_type: str) -> str:
    """``"state"``, ``"delta"`` or ``"lifecycle"`` for one event type.

    Unlisted kinds are lifecycle, deliberately: the cost of wrongly
    treating something as coalescible is a lost terminal event, and the
    cost of wrongly treating it as lifecycle is one extra frame.
    """
    if event_type in DELTA_EVENT_TYPES:
        return DELTA
    if event_type in STATE_EVENT_TYPES:
        return STATE
    return LIFECYCLE


def is_lifecycle(event_type: str) -> bool:
    return delivery_class(event_type) == LIFECYCLE


def is_coalescible(event_type: str) -> bool:
    """Only state events. Deltas are not: see the module docstring."""
    return delivery_class(event_type) == STATE
