"""Retired code, kept runnable.

Not on any import path a caller is expected to use, and not maintained.
Two things live here:

``server/``
    The legacy web server, retired in M9. Its entry point is
    ``archive/server_cli.py``.

``core/``
    The original trainer and its helpers, superseded by ``nodes/``
    (``docs/design/11-core-removal.md``). Its entry point is
    ``python -m archive.core.cli``, which the backend still launches --
    see ``backend/infrastructure/subprocess_gateway.py`` for why that is
    an interim state and what replaces it.

Kept rather than deleted because ``nodes/smoke_tests/``'s equivalence
tests use ``core`` as the *reference* implementation, and because three
of the four training modes exist only there. Deleting it would delete the
evidence that the rewrite computes the same numbers, and the only
working implementation of those modes.
"""
