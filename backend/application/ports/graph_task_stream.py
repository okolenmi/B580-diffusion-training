"""The parent/child event protocol, as a port.

The *shape* of a record lives here; how it is stored does not. The
supervisor needs to know that a ``node`` record means "a node finished"
and an ``outcome`` record means "the run ended and here is how", and it
must be able to say so without importing a file reader.

`infrastructure/graph_event_stream.py` implements this over an append-only
JSONL file and is what the composition root hands in. Anything else that
can carry the same records -- a socket, a pipe, a database table -- is a
different implementation of this and nothing above it would change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """What a record is about.

    ``NODE``   a node finished; the payload is a described ``NodeResult``
    ``MONITOR``a live monitor report; payload has ``monitor_id``/``data``
    ``MEMORY``a memory telemetry frame; payload has ``reserved_mb``,
               ``allocated_mb``, ``peak_mb`` and ``budget_mb`` (null when
               no budget was stated -- never 0.0, which would claim the
               run was given nothing on purpose)
    ``OUTCOME``the run ended; payload has ``error``/``results_count``

    ``OUTCOME`` is the one that earns its keep: without it, "the graph
    failed" and "the process died" are the same absence, and a run killed
    by a device fault would be reported as a success because nobody wrote
    down a problem.
    """

    NODE = "node"
    MONITOR = "monitor"
    MEMORY = "memory"
    OUTCOME = "outcome"


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """One record read back off the channel.

    ``payload`` is whatever the kind implies and is only interpreted by
    the consumer. The writer does not validate it: the values come from
    the same runtime that produced them in-process, and a second
    validation layer would only be a second thing to keep in step.
    """

    kind: EventKind
    payload: dict[str, Any]