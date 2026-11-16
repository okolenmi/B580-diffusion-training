# Setup and running

## Python 3.14 or newer

`pyproject.toml` sets ruff and mypy to 3.14, and the backend depends on it at runtime
(`backend/presentation/event_schema.py` tells `X | None` annotations apart with an identity that only holds from 3.14).
The floor is stated once, in `backend/python_floor.py`; `python -m backend.cli` and the graph child check it first.

## First run: the setup page

`./run_server_linux.sh` (or `run_server_windows.bat`) installs the five server packages into a temporary environment, starts the
server and opens `/setup`. Setup has five steps; nothing is installed until you press *Install*.

1. **Graphics card**: the Intel GPU the system lists (PyTorch may not exist yet).
2. **Environment**: *a new virtualenv* (created at `./venv`) or *an existing virtualenv* (its folder, the one that holds `bin/python`;
   ComfyUI's works). For an existing one the packages found there are listed first and everything already installed is pinned, so
   pip adds what is missing and cannot move what is there. An existing environment whose torch is not the Intel XPU build is refused. The new virtualenv is the project's own: if `./venv` is already there it is completed, and a torch in it that is not the XPU build is replaced.
3. **Install**: pip output streams into a scrollable log. PyTorch comes from the Intel XPU index (`download.pytorch.org/whl/xpu`),
   everything else from PyPI. A new environment offers the tested PyTorch release (2.12.1, `TORCH_TESTED_VERSION` in the manifest),
   the newest on that index, or an exact version. Afterwards the card is checked through that environment's own PyTorch.
4. **Prompt files**: the CLIP, Qwen and T5 vocabulary files, downloaded into `./assets/<set>_tokenizer`. Optional; the buttons stay
   available if you skip it. Where the network is blocked, put the files into those folders by hand (`TESTING.md` lists the names).
5. **Models folder**: default `./models`. A ComfyUI `models` folder works as it is: checkpoints are read from `<folder>/checkpoints`
   and LoRAs are saved to `<folder>/loras`. Missing folders are created.

Finishing stores `models_dir` and the environment's interpreter (`venv_python`) in the settings, restarts the server inside that
environment (the node library imports torch, which the temporary environment does not have) and opens Help. Every later start of
the launcher runs in the stored environment unless `VENV_PYTHON` says otherwise. ComfyUI is not needed anywhere. After that
`/setup` redirects to Help and the installer routes are refused (`installer_not_allowed`); change both from Settings.

While setup is unfinished the sidebar has a *Setup* entry at the top; during an install every other entry except Help is locked.
A reloaded setup page resumes the running install (or the card check after a finished one).

Without the page: set `VENV_PYTHON` (in `.env` or the environment) to an interpreter that has the packages in
`backend/application/ports/requirements_manifest.py`, and `models_dir` in Settings.

## Folder layout

Nothing is required next to the project. The interpreter used for runs is, in order: `VENV_PYTHON`, the `venv_python` setting,
`./venv`, a `venv` folder beside the project. Copy `.env.example` to `.env` (gitignored) to set variables for every start.

```
some-parent-dir/
└── B580-diffusion-training/   # this repo: ./venv (optional), ./models (default), ./assets, ./datasets
```

## This project's own packages

```bash
pip install -r requirements.txt        # the server only (five lines)
```

The trainer itself needs `torch` (Intel XPU build), `numpy`, `pillow`, `safetensors`, `tqdm`, `regex` and `tokenizers` (Anima) in the
environment; `transformers` is optional (auto-captioning). The manifest the setup page checks is
`backend/application/ports/requirements_manifest.py`.

## Running

| What | Command |
|---|---|
| Server and UI (editor at `/graph`, Templates, Datasets, Settings) | `./run_server_linux.sh`, on Windows `run_server_windows.bat` (port 8766 by default) |
| One run from the command line | `python -m run sdxl-lora --help`, then see `MANUAL.md` |
| Everything the sandbox can check | `python run_tests.py` |

The command-line trainer and the editor run the same graphs: `python -m run ... --dump graph.json` writes the graph a
command would run, and the editor opens it with *Import file*.

## Tests

`python run_tests.py` runs the Python checks (`tests`), the backend tests and the frontend tests with an interpreter
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

`venv_python` and the model folders set in Settings are stored in the database (`backend/infrastructure/path_tiers.py` owns the order).

## Where model files live

The app reads and writes `<models folder>/checkpoints` and `<models folder>/loras` (the layout ComfyUI uses) and the repository's `datasets/`.
`POST /api/v1/settings {"models_dir": "/srv/models"}` moves checkpoints and loras to `<dir>/checkpoints` and
`<dir>/loras`; an explicit `checkpoints_dir` or `loras_dir` still wins, and `""` clears it.

## What is shared with ComfyUI

Only things on disk: the model folders (`models/checkpoints`, `models/loras`) and, if you want, the virtual environment.
No ComfyUI module is imported by the trainer or the server. The one exception is optional and separate: the parity scripts
`accept/comfy_forward.py` and `accept/comfy_render.py` run ComfyUI's own code in their own process to compare results with
ours. Because our top-level folders (`nodes`, `graph`, ...) share names with ComfyUI's modules (`nodes.py`, `utils`, ...),
those two scripts take the repository root off `sys.path` before putting ComfyUI first; `scripts/check_imports.py` fails if
one of them forgets, and refuses a ComfyUI import anywhere else.
