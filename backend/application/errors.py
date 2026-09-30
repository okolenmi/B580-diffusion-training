"""Application errors -- "the use case could not fulfil the request".

Distinct from domain errors (which mean a rule was violated while
mutating an entity): these describe outcomes the caller of a use case
must be able to react to. Each carries a stable machine-readable
``code`` that presentation maps onto an HTTP status.
"""

from __future__ import annotations


class ApplicationError(Exception):
    """Base class; ``code`` becomes the API error envelope's code.

    ``details`` is optional machine-readable context (a per-field map,
    a conflict payload) that presentation merges into the envelope.
    """

    code = "application_error"

    def __init__(self, message: str = "", *, details: object = None) -> None:
        super().__init__(message)
        self.details = details


class RunNotFoundError(ApplicationError):
    """No run exists under the requested id."""

    code = "run_not_found"


class InvalidQueryError(ApplicationError):
    """The caller passed parameters no use case can honour."""

    code = "invalid_query"


class ConfigNotFoundError(ApplicationError):
    """The training config file does not exist."""

    code = "config_not_found"


class ConfigInvalidError(ApplicationError):
    """The config exists but cannot be parsed/validated."""

    code = "config_invalid"


class RunAlreadyActiveError(ApplicationError):
    """Another run is created or running; single-run invariant holds."""

    code = "run_already_active"


class RunNotRunningError(ApplicationError):
    """The action needs a running run, but this one is not running."""

    code = "run_not_running"


class NoActiveRunError(ApplicationError):
    """No run is currently created or running."""

    code = "no_active_run"


class TrainingLaunchError(ApplicationError):
    """The trainer subprocess could not be launched.

    Raised through the ``TrainingGateway.spawn`` contract (the adapter
    wraps all OS-level failures into this).
    """

    code = "training_launch_failed"


class SettingsInvalidError(ApplicationError):
    """One or more setting values are invalid for this filesystem.

    ``details`` is a ``{key: message}`` map of every rejected value;
    the whole update was rejected (nothing persisted).
    """

    code = "settings_invalid"
