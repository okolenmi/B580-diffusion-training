"""Tiny assertion helpers in the repo's smoke-test style (no pytest).

Each check file runs standalone: ``check()`` prints PASS/FAIL, ``finish()`` exits non-zero
if anything failed.
"""

from __future__ import annotations

import sys
import traceback

_failures: list[str] = []
_count = 0


def check(condition: bool, label: str) -> None:
    global _count
    _count += 1
    if condition:
        print(f"  PASS {label}")
    else:
        _failures.append(label)
        print(f"  FAIL {label}")


def raises(exc_type: type, fn, label: str, contains: str | None = None) -> None:
    try:
        fn()
    except exc_type as exc:
        check(contains is None or contains in str(exc), f"{label} (message mentions {contains!r}: {exc})")
    except Exception as exc:  # noqa: BLE001
        check(False, f"{label}: wrong exception {type(exc).__name__}: {exc}")
    else:
        check(False, f"{label}: nothing raised")


def section(title: str) -> None:
    print(f"\n{title}")


def run(*functions) -> None:
    for fn in functions:
        section(fn.__name__)
        try:
            fn()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            _failures.append(f"{fn.__name__} crashed")
    finish()


def finish() -> None:
    print(f"\n{_count - len(_failures)}/{_count} checks passed")
    if _failures:
        print("FAILED:", *_failures, sep="\n  ")
        sys.exit(1)
