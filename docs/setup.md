# Setup and running

This doc collects the setup/run information that used to live only as
scattered comments in `run_server.sh`, `.env.example`, and `paths.py`.
If something here disagrees with those files, the files win -- this is
a summary, not a second source of truth (see
[`review_notes.md`](review_notes.md) for the general caveat about
docs drifting from code).

## Expected folder layout

The default, zero-config assumption is that this project, a ComfyUI
checkout, and a Python virtual environment are three **sibling**
directories:

```
some-parent-dir/
├── ComfyUI/
├── venv/                 # has torch, ComfyUI's own deps, etc. installed
└── B580-diffusion-training/   # this repo
```

If your layout matches this, nothing below needs configuring --
`paths.py` auto-detects both `ComfyUI/` and `venv/` from this repo's
own location.

## Configuring a different layout

If your layout is different, set `COMFY_DIR` and `VENV_PYTHON` one of
two ways (env vars always take precedence over `.env`):

- **`.env` file** (recommended for a persistent setup): copy
  `.env.example` to `.env` in this repo's root and fill in the two
  paths. `.env` is gitignored -- it's meant to hold machine-specific
  paths, not something to commit.
- **Real environment variables**, e.g.:
  ```bash
  COMFY_DIR=/path/to/ComfyUI VENV_PYTHON=/path/to/venv/bin/python ./run_server.sh
  ```

`VENV_PYTHON` should point at the interpreter *inside* your venv (the
one with `torch` and ComfyUI's own dependencies already installed) --
it's used both to launch the server itself and, internally, to launch
each training subprocess it spawns.

## Installing this project's own dependencies

This project's own direct requirements are minimal (the training math
itself depends on whatever's already in your ComfyUI venv -- `torch`,
etc.):

```bash
pip install -r requirements.txt
```

## Running: two separate entry points

This repo has two independent ways to actually train something -- see
[`architecture.md`](architecture.md) for how they relate to each
other's code.

### 1. Legacy CLI trainer (`core/`/`manager/`, production path)

Config-driven, TOML-based. Must be run **from ComfyUI's own root
directory** (see `convert.py`'s own docstring for why):

```bash
cd /path/to/ComfyUI
python /path/to/B580-diffusion-training/convert.py --config my_run.toml
```

`convert-cfg.example.toml` in this repo's root is the committed
template config to copy and edit; copy it to `convert-cfg.toml` (also
in the repo root) for your own working config:

```bash
cp convert-cfg.example.toml convert-cfg.toml
```

`convert-cfg.toml` itself is gitignored -- it holds machine-specific
values (real checkpoint/dataset names, real paths, real preview
prompts) and isn't committed; the example carries the same structure
with placeholders. `convert.py --config` will also create a
defaults-filled config for any filename you point it at, so the copy
step is a convenience rather than a requirement.

### 2. Node-graph web UI (`nodes/`/`backend/`, the active rewrite)

```bash
./run_server.sh                  # binds 0.0.0.0:8766 by default
./run_server.sh --host 127.0.0.1 --port 8080   # override either
```

This starts a browser-based visual node editor for building and
running training graphs out of the `nodes/` package's Node classes.
Open the printed URL in a browser once the server starts.
`run_server.sh` launches `python -m backend.cli` -- the `backend/`
package (REST API under `/api/v1` + the frontend it serves). The
first-cut `server/` web layer is retired under `archive/` as of M9;
`archive/server_cli.py` still launches it for reference.

## Running the test suite

Every test here is a plain, independently-runnable `smoke_test_*.py`
script -- CPU-only, no ComfyUI/XPU hardware required, no test
framework dependency beyond what's already installed:

```bash
# Everything, both suites (nodes/ + manager/), one command:
python run_tests.py

# Filter by filename substring, e.g. only memory-related tests
python run_tests.py memory

# Per-suite runners still work on their own:
python nodes/smoke_tests/run_all.py          # nodes/ only
python manager/smoke_tests/smoke_test_lora_raw_dataset.py
```

(`server/`'s six smoke tests retired with `archive/` at M9.) The web
backend has its own suite, run under the torch venv interpreter, and
the full gate ties every suite together:

```bash
$VENV_PYTHON backend/tests/run_all.py   # backend suite (API, pages, use cases)
scripts/full_gate.sh                    # legacy suites + backend suite + frontend node --check
```

`run_tests.py` picks the interpreter itself: the tests import torch,
which lives in your ComfyUI venv, not in whatever system `python`
happens to be first on PATH. If the running interpreter has no torch,
it resolves `VENV_PYTHON` (environment variable, then `.env`, same
precedence as `paths.py`) and runs every test under that -- rather
than emitting ~68 identical `ModuleNotFoundError: No module named
'torch'` tracebacks, which is exactly what running the suite with the
wrong python looks like.

## Hardware notes

Development and tuning happened against Intel Arc B580 (XPU) hardware
specifically -- several fixes in `docs/known-issues/` are
XPU-specific (device-lost/hang reports, CPU-side kernel-dispatch
overhead on optimizer math). Nothing here is known to *require*
Intel/XPU hardware, but CUDA-specific behavior hasn't been the focus of
testing.
