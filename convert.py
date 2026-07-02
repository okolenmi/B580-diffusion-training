#!/usr/bin/env python3
"""Training entry point.

Run from the ComfyUI root directory:
    cd /path/to/ComfyUI
    python ../<this-project>/convert.py --config my_run.toml
"""

import sys
from pathlib import Path

# Ensure the package directory is importable
_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

# NOTE: ComfyUI's own directory is intentionally *not* resolved here.
# paths.get_comfy_dir() can raise (no COMFY_DIR env var, not auto-detectable
# from cwd) and previously did so unconditionally at import time, which
# broke `--help`, `--reset-config`, and first-run config scaffolding even
# though none of those touch ComfyUI. It's now resolved lazily inside
# core.cli.main(), right before Trainer actually needs it.
from core.cli import main

if __name__ == "__main__":
    main()
