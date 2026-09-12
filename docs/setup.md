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

`convert-cfg.toml` in this repo's root is a real example config to
copy and edit, not a template with placeholder syntax -- check every
path in it (`base_model`, `dataset_name`, `comfy_dir`) against your own
setup before using it as-is; those are one specific person's real
paths, not portable defaults (flagged in
[`review_notes.md`](review_notes.md)).

### 2. Node-graph web UI (`nodes/`/`server/`, the active rewrite)

```bash
./run_server.sh                  # binds 0.0.0.0:8765 by default
./run_server.sh --host 127.0.0.1 --port 8080   # override either
```

This starts a browser-based visual node editor for building and
running training graphs out of the `nodes/` package's Node classes.
Open the printed URL in a browser once the server starts.

## Running the test suite

Every test here is a plain, independently-runnable `smoke_test_*.py`
script -- CPU-only, no ComfyUI/XPU hardware required, no test
framework dependency beyond what's already installed:

```bash
# Everything under nodes/ (the bulk of the suite)
python nodes/smoke_tests/run_all.py

# Filter by filename substring, e.g. only memory-related tests
python nodes/smoke_tests/run_all.py memory

# server/ and manager/ each have their own smaller suites, run individually:
python server/smoke_tests/smoke_test_graph_executor.py
python manager/smoke_tests/smoke_test_lora_raw_dataset.py
```

There's no single script that runs `nodes/`, `server/`, and `manager/`
tests together in one command as of this writing -- worth adding if
that becomes annoying (noted in [`review_notes.md`](review_notes.md)).

## Hardware notes

Development and tuning happened against Intel Arc B580 (XPU) hardware
specifically -- several fixes in `docs/known-issues/` are
XPU-specific (device-lost/hang reports, CPU-side kernel-dispatch
overhead on optimizer math). Nothing here is known to *require*
Intel/XPU hardware, but CUDA-specific behavior hasn't been the focus of
testing.
