"""Domain events -- immutable facts about things that happened.

Events are plain frozen dataclasses with no knowledge of JSON, SSE,
or any transport. Entities append them to an internal buffer as they
mutate; the application layer drains the buffer (``collect_events``)
after persisting and publishes each event through the ``EventBus``
port. Presentation decides how events reach clients (SSE today).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC



def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, kw_only=True)
class DomainEvent:
    """Base event: every fact knows when it occurred."""

    occurred_at: datetime = field(default_factory=_utcnow)

    @classmethod
    def wire_name(cls) -> str:
        """Stable wire name, e.g. ``RunCompleted`` -> ``run_completed``.

        Derived from the class name so the naming lives with the event
        itself; presentation only embeds it in the payload.

        A classmethod because generating the event schema
        (``presentation/event_schema.py``) needs every event's name
        *without* constructing one -- events have required fields.
        Deriving it in a single place is the point: a second copy of this
        algorithm inside the generator is a second thing to keep correct.
        """
        out = ""
        for char in cls.__name__:
            if char.isupper():
                out += "_" + char.lower()
            else:
                out += char
        return out.strip("_")

    @property
    def event_type(self) -> str:
        """This event's wire name. See ``wire_name``."""
        return type(self).wire_name()


# --------------------------------------------------------------------------
# Graph execution lifecycle (M4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphExecutionQueued(DomainEvent):
    """A graph passed validation and received its execution id."""

    execution_id: int
    node_count: int


@dataclass(frozen=True)
class GraphExecutionStarted(DomainEvent):
    """The worker thread claimed the queued row and began building nodes."""

    execution_id: int


@dataclass(frozen=True)
class GraphExecutionProgressed(DomainEvent):
    """Telemetry: one node finished building (published by the
    supervisor, never buffered by the entity -- same posture as
    ``RunProgressed``: lifecycle events mark state changes, this one
    streams per-node timing while ``running``."""

    execution_id: int
    node_id: str
    ok: bool
    duration_ms: float


@dataclass(frozen=True)
class GraphExecutionFinished(DomainEvent):
    """Every node in the graph built successfully."""

    execution_id: int
    nodes: int


@dataclass(frozen=True)
class GraphExecutionFailed(DomainEvent):
    """Graph-level failure, a node's build() raised, or startup
    reconciliation swept a row left behind by a dead process."""

    execution_id: int
    error: str | None


@dataclass(frozen=True)
class GraphExecutionStopped(DomainEvent):
    """Cancel requested (stop endpoint, or a queued row stopped before
    its thread started)."""

    execution_id: int
    reason: str | None = None


@dataclass(frozen=True)
class GraphExecutionsDeleted(DomainEvent):
    """Execution history was wiped (a batch deletion)."""

    deleted: int
