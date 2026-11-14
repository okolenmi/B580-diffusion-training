# Setup and running

## Python 3.14 or newer

`pyproject.toml` sets ruff and mypy to 3.14, and the backend depends on it at runtime
(`backend/presentation/event_schema.py` tells `X | None` annotations apart with an identity that only holds from 3.14).
The floor is stated once, in `backend/python_floor.py`; `python -m backend.cli` and the graph child check it first.

## Folder layout

Zero-config layout: this project, a ComfyUI checkout and a virtual environment are siblings.

```
some-parent-dir/
├── ComfyUI/
├── venv/                      # torch (XPU or CUDA) and ComfyUI's own packages
└── B580-diffusion-training/   # this repo
```

Another layout: copy `.env.example` to `.env` (gitignored) and set `COMFY_DIR` and `VENV_PYTHON`, or export them for one
run (environment variables win over `.env`). `VENV_PYTHON` is the interpreter that has torch; the server and every run it
starts use it. The browser wizard at `/setup` can also create the environment on a first run.

## This project's own packages

```bash
pip install -r requirements.txt        # the server only (five lines); torch comes from the venv
```

The trainer itself needs `torch`, `numpy`, `pillow`, `safetensors`, `tqdm`, `regex` and `tokenizers` (Anima) in the venv;
the manifest the wizard checks is `backend/application/ports/requirements_manifest.py`.

## Running

| What | Command |
|---|---|
| Server and UI (editor at `/graph`, Templates, Datasets, Settings) | `./run_server.sh` (port 8765 by default) |
| One run from the command line | `python -m next.run sdxl-lora --help`, then see `MANUAL.md` |
| Everything the sandbox can check | `python run_tests.py` |

The command-line trainer and the editor run the same graphs: `python -m next.run ... --dump graph.json` writes the graph a
command would run, and the editor opens it with *Import file*.

## Tests

`python run_tests.py` runs the Python checks (`next/tests`), the backend tests and the frontend tests with an interpreter
that has torch. It filters by name part (`python run_tests.py memory`). Checks that need the card are listed in
`TESTING.md`, not run here.

## Backend environment variables

| Variable | Effect |
|---|---|
| `BACKEND_HOST`, `BACKEND_PORT` | defaults for the CLI; command-line flags win |
| `BACKEND_DB_PATH` | database location (a scratch path for tests) |
| `BACKEND_ALLOWED_HOSTS` | extra `Host` names the server answers to (comma-separated) |
| `BACKEND_ALLOWED_ORIGINS` | extra origins allowed to make state changes |
| `BACKEND_LOG_LEVEL` | log level of the backend's own loggers (default `INFO`) |
| `BACKEND_ACCESS_LOG` | `1` prints one uvicorn line per request (off by default) |

A `comfy_dir` / `venv_python` set in Settings is stored in the database; it sits below the environment for those two keys
and above it for `checkpoints_dir` and `loras_dir` (`backend/infrastructure/path_tiers.py` owns that order).

## Where model files live

The app reads and writes inside ComfyUI's layout (`models/checkpoints`, `models/loras`) and the repository's `datasets/`.
`POST /api/v1/settings {"models_dir": "/srv/models"}` moves checkpoints and loras to `<dir>/checkpoints` and
`<dir>/loras`; an explicit `checkpoints_dir` or `loras_dir` still wins, and `""` clears it.

## ComfyUI is not imported

Production code never imports ComfyUI modules. Only the acceptance scripts `accept/comfy_forward.py` and
`accept/comfy_render.py` do, as separate processes that put ComfyUI first on `sys.path` (and take our own folders off it);
`scripts/check_imports.py` fails if one of them forgets.
