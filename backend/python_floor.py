"""The Python this code is written for, checked before it runs.

Both quality tools are already configured for 3.14 -- `pyproject.toml`
sets `target-version = "py314"` for ruff and `python_version = "3.14"` for
mypy -- and neither of those says anything to the person running the code.
So a checkout on 3.12 passed both gates and then failed somewhere else,
with a message that pointed at the schema layer rather than at the
interpreter.

3.14 is not a preference. One place in the code depends on it at runtime:
`presentation/event_schema.py` decides "is this annotation `X | None`" with
`get_origin(annotation) is Union`, and that identity holds from 3.14. On an
earlier interpreter the check is silently false for every union, so the
schema comes out wrong rather than erroring -- which is the worst way for
it to fail. (That dependency is the review's claim about older versions,
not something measured here; this box runs 3.14.7 and only 3.14 was
available to check against. What *is* measured is that the check passes on
3.14, and that nothing told a developer which interpreter was expected.)

So the floor is stated in one place and enforced at both process entry
points, where the message can still be useful.
"""

from __future__ import annotations

import sys

#: The oldest interpreter this code runs on. Kept as a tuple so the
#: comparison is a version comparison and not a string one.
MIN_PYTHON: tuple[int, int] = (3, 14)

#: One line, so the two entry points cannot drift apart in what they say.
#: Note the version in the explanation is a literal, not the floor being
#: asked about. Deriving it from `{major}.{minor}` produced a message that
#: said "which only holds from 99.0" when handed an arbitrary floor, which
#: is worse than no explanation.
MESSAGE: str = (
    "This project needs Python {major}.{minor} or newer; this is "
    "Python {actual}. The floor is not arbitrary: "
    "backend/presentation/event_schema.py decides whether an annotation is "
    "`X | None` with `get_origin(annotation) is Union`, which only holds "
    "from 3.14, and both quality tools are configured for 3.14 "
    "(pyproject.toml: ruff target-version, mypy python_version)."
)


def require_python() -> None:
    """Exit with a readable message if the interpreter is too old.

    Raises `SystemExit` rather than returning a bool, because every caller
    is a `main()` whose only sensible response is to stop: there is no
    reduced-functionality mode to fall back to, and continuing would fail
    later and further away.
    """
    require_python_at_least(MIN_PYTHON)


def require_python_at_least(floor: tuple[int, int]) -> None:
    """The check itself, against a stated floor.

    Split out so the refusal path can be tested on a machine that satisfies
    the real one. A guard whose every branch returns on the interpreter it
    runs on has never been observed to refuse anything, which is not the
    same as knowing it would.
    """
    if sys.version_info >= floor:
        return
    actual = ".".join(str(part) for part in sys.version_info[:3])
    major, minor = floor
    sys.exit(MESSAGE.format(major=major, minor=minor, actual=actual))