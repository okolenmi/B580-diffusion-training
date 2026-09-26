"""CLI entry point for the web server."""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Before anything else touches torch/XPU -- see core/xpu_env.py's own
# docstring. This is the fix half of a real gap found investigating a
# reported speed regression: unlike core/cli.py (the older, subprocess-
# per-run route, which already set these), nothing in this process ever
# did, and graph execution runs as a background thread inside *this*
# same long-lived process (server/routes_nodegraph.py's _worker()), not
# as its own core.cli subprocess -- so core/cli.py's own env vars,
# module-level as they are, never had a chance to apply here.
from core.xpu_env import set_xpu_perf_env_vars
set_xpu_perf_env_vars()


def main():
    p = argparse.ArgumentParser(
        description="Web UI for the training pipeline",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")

    args = p.parse_args()

    # ComfyUI is resolved here, after argument parsing, rather than
    # unconditionally at module import time -- previously `--help` (or just
    # importing this module for any other reason, e.g. tests) would crash
    # with paths.get_comfy_dir()'s RuntimeError if ComfyUI wasn't
    # auto-detectable yet, before argparse even got a chance to run.
    from paths import get_comfy_dir
    comfy_dir = get_comfy_dir()
    if str(comfy_dir) not in sys.path:
        sys.path.append(str(comfy_dir))

    from server.main import run
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
