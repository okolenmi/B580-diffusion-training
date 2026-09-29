"""CLI entry point.

Run from the repository root:

    python -m backend.cli [--host H] [--port P] [--db PATH]

All environment/argument side effects are confined to ``main()``:
importing any other backend module does nothing on its own.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

from .bootstrap import build_container
from .config import Settings
from .presentation.app import create_app


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
    args = build_parser().parse_args(argv)

    settings = Settings.load(os.environ)
    overrides = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.db is not None:
        from pathlib import Path

        overrides["db_path"] = Path(args.db)
    if overrides:
        settings = replace(settings, **overrides)

    container = build_container(settings)
    app = create_app(container.services)

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
