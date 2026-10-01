"""The error contract: codes, statuses, and the envelope around them.

Two things are pinned here, and both used to be able to drift silently
(docs 08 S-10):

* every ``ApplicationError`` subclass declares the status that
  ``docs/design/backend/02-api-reference.md`` documents. The test
  *parses that table* rather than restating it, so there is one source:
  the doc. A new error that forgets its status, or a status that
  changed in one place only, fails at the gate instead of in a client;
* an error raised through the real handler leaves as the documented
  envelope with that status -- the code -> status lookup that used to
  live in ``presentation/errors.py`` is gone, so the mapping is
  exercised end to end.

Run:  python backend/tests/test_error_contract.py
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.tests.support import check, finish  # noqa: E402
# Imported so its two refusal errors exist: the class walk below uses
# ``__subclasses__``, which only sees modules that are loaded. Both are
# documented in the table this test parses.
from backend.presentation import security as _security  # noqa: E402,F401
from backend.application.errors import (  # noqa: E402
    ApplicationError,
    ConfigNotFoundError,
    DatasetNotMigratedError,
    DatasetTaskActiveError,
    GraphExecutionNotActiveError,
    InvalidQueryError,
    NodeDiagnosticsError,
    RunDirectoryCollisionError,
    RunNotFoundError,
    SettingsInvalidError,
    TrainingLaunchError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DOC = REPO_ROOT / "docs/design/backend/02-api-reference.md"

_TABLE_ROW = re.compile(r"^\s*\|\s*(.+?)\s*\|\s*(\d{3})\s*\|")


def documented_codes() -> dict[str, int]:
    """The code -> status table as written in the API reference.

    Rows look like::

        | `dataset_not_found`, `dataset_item_not_found` | 404 | datasets |

    so a row names one or more backticked codes and one status. Anything
    that is not of that shape (the header row, the prose around the
    table) is not a row, which is why the section is bounded by its
    heading.
    """
    text = API_DOC.read_text(encoding="utf-8")
    start = text.index("**Error codes**")
    end = text.index("## 2.", start)
    table: dict[str, int] = {}
    for line in text[start:end].splitlines():
        match = _TABLE_ROW.match(line)
        if not match:
            continue
        codes = re.findall(r"`(\w+)`", match.group(1))
        if not codes:
            continue  # the header row: | Code | Status | Domain |
        status = int(match.group(2))
        for code in codes:
            table[code] = status
    return table


def _subclasses(cls: type) -> list[type]:
    return [
        sub
        for sub in cls.__subclasses__()
        if not inspect.isabstract(sub) and sub is not ApplicationError
    ]


def test_every_error_declares_its_status() -> None:
    print("\n== every ApplicationError declares the documented status ==")
    documented = documented_codes()
    check(len(documented) >= 25, f"the API reference table was parsed ({len(documented)} codes)")

    subclasses = _subclasses(ApplicationError)
    check(
        len(subclasses) == len(documented),
        f"one error class per documented code "
        f"({len(subclasses)} classes, {len(documented)} codes)",
    )

    undocumented, mistyped, duplicated = [], [], []
    seen: dict[str, type] = {}
    for sub in subclasses:
        if not isinstance(sub.status_code, int) or not 100 <= sub.status_code < 600:
            mistyped.append(f"{sub.__name__}={sub.status_code!r}")
        elif sub.code not in documented:
            undocumented.append(f"{sub.__name__}:{sub.code}")
        elif documented[sub.code] != sub.status_code:
            mistyped.append(
                f"{sub.__name__}: {sub.code} is {sub.status_code}, "
                f"doc says {documented[sub.code]}"
            )
        if sub.code in seen:
            duplicated.append(f"{sub.code}: {seen[sub.code].__name__}, {sub.__name__}")
        seen[sub.code] = sub

    check(not undocumented, f"every code is documented ({undocumented})")
    check(not mistyped, f"every status matches the documented table ({mistyped})")
    check(not duplicated, f"no two errors share a code ({duplicated})")

    # The reverse direction: a documented code nobody raises is a doc
    # promise the API cannot keep.
    orphans = sorted(set(documented) - set(seen))
    check(not orphans, f"every documented code has a class ({orphans})")

    # A status is a *class* fact, so an instance inherits it instead of
    # carrying its own copy -- which is what let the old lookup table
    # disagree with the error it was describing.
    check(
        RunNotFoundError("x").status_code == 404,
        "an instance reads its status off the class",
    )
    check(ApplicationError.status_code == 500, "the base default is a 500")


def test_envelope_shape() -> None:
    print("\n== the envelope carries code, message and optional details ==")
    from backend.presentation.errors import error_body

    bare = error_body("run_not_found", "run 7 not found")
    check(
        bare == {"error": {"code": "run_not_found", "message": "run 7 not found"}},
        "no details key when there are none",
    )
    check(
        error_body("graph_invalid", "bad graph", [{"loc": "a"}])["error"]["details"]
        == [{"loc": "a"}],
        "an issue list is passed through verbatim",
    )
    check(
        error_body("x", "y", {"run_id": 1})["error"]["details"] == {"run_id": 1},
        "a detail map works as well as a list",
    )


def test_handler_replies_with_the_declared_status() -> None:
    print("\n== the handler reads the status off the error ==")
    import asyncio

    from backend.presentation.errors import register_error_handlers

    class _App:
        """Just enough of a FastAPI to capture the registered handler."""

        def __init__(self) -> None:
            self.handlers: dict[type, object] = {}

        def exception_handler(self, exc_type):
            def register(fn):
                self.handlers[exc_type] = fn
                return fn

            return register

    app = _App()
    register_error_handlers(app)
    check(ApplicationError in app.handlers, "the ApplicationError handler is registered")
    handler = app.handlers[ApplicationError]

    async def call(exc: ApplicationError) -> tuple[int, dict]:
        response = await handler(object(), exc)
        return response.status_code, json.loads(response.body)

    documented = documented_codes()
    for error in (
        RunNotFoundError("run 7 not found"),
        InvalidQueryError("limit must be 1..500"),
        RunDirectoryCollisionError("runs/run_1 has files"),
        SettingsInvalidError("venv_python is not executable"),
        TrainingLaunchError("exec failed"),
        DatasetTaskActiveError("already running"),
        DatasetNotMigratedError("format v1"),
        GraphExecutionNotActiveError("already finished"),
        NodeDiagnosticsError("bad params"),
        ConfigNotFoundError("no such config"),
    ):
        status, body = asyncio.run(call(error))
        check(
            status == documented[error.code] and status == error.status_code,
            f"{type(error).__name__} -> {documented[error.code]} (got {status})",
        )
        check(
            body["error"]["code"] == error.code and body["error"]["message"],
            f"{type(error).__name__} keeps its code in the envelope",
        )

    status, body = asyncio.run(
        call(InvalidQueryError("bad", details={"field": "limit"}))
    )
    check(status == 422, "invalid_query -> 422")
    check(
        body["error"]["details"] == {"field": "limit"},
        "details reach the envelope",
    )


def main() -> None:
    test_every_error_declares_its_status()
    test_envelope_shape()
    test_handler_replies_with_the_declared_status()
    finish()


if __name__ == "__main__":
    main()