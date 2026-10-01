"""Application errors -- "the use case could not fulfil the request".

Distinct from domain errors (which mean a rule was violated while
mutating an entity): these describe outcomes the caller of a use case
must be able to react to. Each carries a stable machine-readable
``code`` -- the API error envelope's code -- and the HTTP ``status`` the
presentation layer answers with.

The status lives *here*, beside the code it belongs to. It used to be a
26-entry table in ``presentation/errors.py`` keyed by the code string,
which meant adding an error took two edits and a typo in the table
silently produced a 400 for something documented as a 404
(docs 08 S-10). The code table in ``02-api-reference.md`` is still the
contract; this module is now its implementation.
"""

from __future__ import annotations


class ApplicationError(Exception):
    """Base class; ``code`` becomes the API error envelope's code.

    ``status_code`` is the HTTP status the handler replies with. It is a
    class attribute so every subclass declares it once and the handler
    has nothing left to look up.

    ``details`` is optional machine-readable context (a per-field map,
    a conflict payload) that presentation merges into the envelope.
    """

    code = "application_error"
    status_code = 500

    def __init__(self, message: str = "", *, details: object = None) -> None:
        super().__init__(message)
        self.details = details


class RunNotFoundError(ApplicationError):
    """No run exists under the requested id."""

    code = "run_not_found"
    status_code = 404


class InvalidQueryError(ApplicationError):
    """The caller passed parameters no use case can honour."""

    code = "invalid_query"
    status_code = 422


class AssetTooLargeError(ApplicationError):
    """An upload body exceeds the contract's MAX_UPLOAD_BYTES cap."""

    code = "asset_too_large"
    status_code = 413


class AssetExistsError(ApplicationError):
    """An upload target already exists and overwriting was not asked for.

    A ``PUT`` that silently replaces a real checkpoint is data loss with
    no signal to the user; overwriting must be explicit (docs 08 N-14).
    """

    code = "asset_exists"
    status_code = 409


class ConfigNotFoundError(ApplicationError):
    """The training config file does not exist."""

    code = "config_not_found"
    status_code = 404


class ConfigInvalidError(ApplicationError):
    """The config exists but cannot be parsed/validated."""

    code = "config_invalid"
    status_code = 422


class RunAlreadyActiveError(ApplicationError):
    """Another run is created or running; single-run invariant holds."""

    code = "run_already_active"
    status_code = 409


class RunNotRunningError(ApplicationError):
    """The action needs a running run, but this one is not running."""

    code = "run_not_running"
    status_code = 409


class NoActiveRunError(ApplicationError):
    """No run is currently created or running."""

    code = "no_active_run"
    status_code = 404


class TrainingLaunchError(ApplicationError):
    """The trainer subprocess could not be launched.

    Raised through the ``TrainingGateway.spawn`` contract (the adapter
    wraps all OS-level failures into this).
    """

    code = "training_launch_failed"
    status_code = 500


class SettingsInvalidError(ApplicationError):
    """One or more setting values are invalid for this filesystem.

    ``details`` is a ``{key: message}`` map of every rejected value;
    the whole update was rejected (nothing persisted).
    """

    code = "settings_invalid"
    status_code = 400


class DatasetNotFoundError(ApplicationError):
    """No dataset directory exists under the requested name."""

    code = "dataset_not_found"
    status_code = 404


class DatasetItemNotFoundError(ApplicationError):
    """No trajectory row exists under the requested id in this dataset."""

    code = "dataset_item_not_found"
    status_code = 404


class DatasetFileNotFoundError(ApplicationError):
    """No readable file at the requested path inside the dataset
    directory: missing, not a regular file, or escaping the dataset
    root (the escape is reported as not-found, never resolved)."""

    code = "dataset_file_not_found"
    status_code = 404


class DatasetAlreadyExistsError(ApplicationError):
    """A dataset already occupies the requested name."""

    code = "dataset_exists"
    status_code = 409


class DatasetDirectoryConflictError(ApplicationError):
    """A non-dataset directory occupies the requested dataset name.

    ``datasets/<name>`` exists but holds no ``metadata.db`` *and* is not
    an empty skeleton -- i.e. it holds files this code did not create. The
    legacy server deleted such a directory ("nothing in it can be
    loadable"); doing that to a user's own image folder is silent data
    loss, so the name is refused instead (docs 07 F-10). The user renames
    or removes the directory, then creates the dataset.
    """

    code = "dataset_directory_conflict"
    status_code = 409


class RunDirectoryCollisionError(ApplicationError):
    """``runs/run_<id>/`` already holds another run's files.

    The database handed out an id whose directory is occupied -- e.g. a
    legacy run that predates this backend's table. Writing there would
    truncate the old log (``spawn`` opens it with ``"w"``), so the start
    is refused (docs 07 F-04). Startup seeds the id sequence above the
    highest existing ``run_*`` directory, so this is the safety net for
    anything that appeared after boot.
    """

    code = "run_directory_conflict"
    status_code = 409


class DatasetNotMigratedError(ApplicationError):
    """The dataset is still in legacy format v1.

    ``details`` carries the migration hint; every v2-only operation
    refuses loudly rather than failing on a missing column.
    """

    code = "dataset_not_migrated"
    status_code = 409


class DatasetTaskActiveError(ApplicationError):
    """A pending/running task already owns the dataset (one at a time)."""

    code = "dataset_task_active"
    status_code = 409


class DatasetTaskNotFoundError(ApplicationError):
    """No dataset task exists under the requested id."""

    code = "dataset_task_not_found"
    status_code = 404


class DatasetTaskNotActiveError(ApplicationError):
    """The action needs a pending/running task; this one already ended."""

    code = "dataset_task_not_active"
    status_code = 409


class DatasetTaskLaunchError(ApplicationError):
    """The ingestion child could not be started (row already failed)."""

    code = "dataset_task_launch_failed"
    status_code = 500


# Graphs (M4)


class GraphInvalidError(ApplicationError):
    """The submitted graph failed validation.

    ``details`` is the issue list (``issue_to_dict`` per entry) -- every
    error-severity finding, not just the first.
    """

    code = "graph_invalid"
    status_code = 422


class GraphExecutionNotFoundError(ApplicationError):
    """No graph execution exists under the requested id."""

    code = "graph_execution_not_found"
    status_code = 404


class GraphExecutionActiveError(ApplicationError):
    """Another execution is queued or running; single-active holds.

    ``details`` carries the blocking execution's id/status.
    """

    code = "graph_execution_active"
    status_code = 409


class GraphExecutionNotActiveError(ApplicationError):
    """The action needs a live execution; this one already ended.

    ``details`` carries the terminal status that won.
    """

    code = "graph_execution_not_active"
    status_code = 409


class NodeClassNotFoundError(ApplicationError):
    """No discovered node class answers to the requested name."""

    code = "node_class_not_found"
    status_code = 404


class NodeDiagnosticsError(ApplicationError):
    """The node's own diagnostics() raised for these params.

    An ordinary outcome mid-edit (bad path, missing file) -- reported
    as a normal 400, never a 500.
    """

    code = "node_diagnostics_failed"
    status_code = 400


class GraphNotFoundError(ApplicationError):
    """No saved graph exists under the requested library name."""

    code = "graph_not_found"
    status_code = 404