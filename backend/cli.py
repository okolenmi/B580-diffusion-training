"""CLI entry point.

Run from the repository root:

    python -m backend.cli [--host H] [--port P] [--db PATH]

All environment/argument side effects are confined to ``main()``:
importing any other backend module does nothing on its own.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

from .bootstrap import build_container
from .config import Settings
from .presentation.app import create_app
from .python_floor import require_python


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backend.cli",
        description="Run the training backend (clean-room replacement for server/).",
    )
    parser.add_argument("--host", default=None, help="bind address (default: env/127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port (default: env/8766)")
    parser.add_argument("--db", default=None, help="SQLite file (default: env/backend/data/backend.db)")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Before argument parsing and before anything imports torch: an
    # interpreter that is too old should say so in one line, rather than
    # reaching the schema layer and producing a wrong answer there.
    require_python()

    args = build_parser().parse_args(argv)

    # Logging first, before anything that logs during start-up. Placed
    # here rather than just before uvicorn.run because the interesting
    # start-up lines come from build_container's startup reconcile --
    # including "adopted N still-running graph execution(s)", which is
    # the one line that tells an operator who just restarted the server
    # that a run was re-attached to rather than thrown away. Configuring
    # logging after it means exactly those lines are the ones lost, since
    # without a handler Python's last resort shows WARNING and nothing
    # else.
    #
    # uvicorn's ``log_level`` configures uvicorn's own loggers only, so
    # it does not cover the backend's twelve logger.info calls.
    logging.basicConfig(
        level=os.environ.get("BACKEND_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # Entry-point parity with the retired server_cli (archived with M9):
    # both gateways spawn children via os.environ.copy(), so the XPU
    # perf env must be set in THIS process before anything spawns. Pure
    # os.environ writes, no torch import -- safe before any child.
    from nodes.xpu_env import set_xpu_perf_env_vars  # noqa: PLC0415

    set_xpu_perf_env_vars()

    settings = Settings.load(os.environ)
    overrides = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.db is not None:
        overrides["db_path"] = Path(args.db)
    if overrides:
        settings = replace(settings, **overrides)

    container = build_container(settings)
    # The frontend ships with the repo (not with the user's project
    # root): one directory above this package.
    static_dir = Path(__file__).resolve().parent.parent / "frontend"
    app = create_app(container.services, static_dir=static_dir)

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
