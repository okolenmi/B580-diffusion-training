"""Domain events -- immutable facts about things that happened.

Events are plain frozen dataclasses with no knowledge of JSON, SSE,
or any transport. Entities append them to an internal buffer as they
mutate; the application layer drains the buffer (``collect_events``)
after persisting and publishes each event through the ``EventBus``
port. Presentation decides how events reach clients (SSE today).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .value_objects import RunId


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class DomainEvent:
    """Base event: every fact knows when it occurred."""

    occurred_at: datetime = field(default_factory=_utcnow)

    @property
    def event_type(self) -> str:
        """Stable wire name, e.g. ``RunCompleted`` -> ``run_completed``.

        Derived from the class name so the naming lives with the event
        itself; presentation only embeds it in the payload.
        """
        name = type(self).__name__
        out = ""
        for char in name:
            if char.isupper():
                out += "_" + char.lower()
            else:
                out += char
        return out.strip("_")


# --------------------------------------------------------------------------
# Run lifecycle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunCreated(DomainEvent):
    """A run was registered and received its id."""

    run_id: RunId
    config_path: str
    mode: str
    total_steps: int


@dataclass(frozen=True)
class RunStarted(DomainEvent):
    """The training process is alive."""

    run_id: RunId
    pid: int | None


@dataclass(frozen=True)
class RunCompleted(DomainEvent):
    """The training process exited successfully."""

    run_id: RunId
    done_steps: int


@dataclass(frozen=True)
class RunFailed(DomainEvent):
    """The training process died with an error."""

    run_id: RunId
    error: str | None
    exit_code: int | None


@dataclass(frozen=True)
class RunCancelled(DomainEvent):
    """The run was stopped on request."""

    run_id: RunId


@dataclass(frozen=True)
class RunsDeleted(DomainEvent):
    """Run history was wiped (a batch deletion, not tied to one run)."""

    deleted: int
