"""Application errors -- "the use case could not fulfil the request".

Distinct from domain errors (which mean a rule was violated while
mutating an entity): these describe outcomes the caller of a use case
must be able to react to. Each carries a stable machine-readable
``code`` that presentation maps onto an HTTP status.
"""

from __future__ import annotations


class ApplicationError(Exception):
    """Base class; ``code`` becomes the API error envelope's code."""

    code = "application_error"


class RunNotFoundError(ApplicationError):
    """No run exists under the requested id."""

    code = "run_not_found"


class InvalidQueryError(ApplicationError):
    """The caller passed parameters no use case can honour."""

    code = "invalid_query"
