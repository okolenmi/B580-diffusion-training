"""Domain errors -- raised when an entity invariant would be violated.

Only *rule* violations live here (illegal transition, negative step
count, missing id). "Resource does not exist" is an application-level
concern and lives in ``backend.application.errors``.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every domain-rule violation."""


class InvalidTransitionError(DomainError):
    """A state transition the run's state machine does not allow."""
