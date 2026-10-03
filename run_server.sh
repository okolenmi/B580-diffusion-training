#!/bin/bash
# Launch the training web server.
#
# ComfyUI directory and the Python interpreter to use are configurable two
# ways (not hardcoded to any particular folder layout):
#
#   1. Recommended: copy .env.example to .env (right next to this script)
#      and fill in COMFY_DIR / VENV_PYTHON there. One file, used by this
#      script, the server, and every training subprocess it launches.
#
#   2. Environment variables, which always take precedence over .env:
#        COMFY_DIR=/path/to/ComfyUI VENV_PYTHON=/path/to/venv/bin/python ./run_server.sh
#
# If neither is set, both fall back to auto-detection: works out of the box
# if ComfyUI and a venv/ folder are sibling directories of this project.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env (same file paths.py's _load_dotenv() reads) so it configures
# both this script's own interpreter selection *and* everything downstream
# -- one file, not two separate places to edit. Real env vars already set
# still win (only fills in what isn't already set), matching the Python
# side's behavior.
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        case "$line" in
            ''|'#'*) continue ;;
        esac
        key="${line%%=*}"
        value="${line#*=}"
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        if [ -z "${!key:-}" ]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
fi

# Resolve the python interpreter with the same precedence as paths.py's
# VENV_PYTHON resolution (the backend launches training and dataset
# tasks through the same rules), so the server itself and the trainer
# it spawns are guaranteed to agree on which interpreter/venv to use:
#   1. VENV_PYTHON env var (or .env), if set and it actually exists
#   2. <this-project>/venv/bin/python, if it exists -- this is where the
#      first-run wizard creates one, so a machine set up through the
#      browser needs no .env edit to find its own environment next start
#   3. <parent-of-this-project>/venv/bin/python, if it exists (the
#      project / ComfyUI / venv sibling-folder layout)
#   4. Whatever "python" resolves to on PATH
#
# Step 2 before step 3 on purpose: a wizard-created venv is a deliberate
# choice by this install, so it outranks a folder that happens to be
# sitting next to the project. Both still lose to an explicit VENV_PYTHON,
# which is the documented override.
PROJECT_VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
SIBLING_VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"
if [ -n "${VENV_PYTHON:-}" ] && [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
elif [ -x "$PROJECT_VENV_PYTHON" ]; then
    PYTHON="$PROJECT_VENV_PYTHON"
elif [ -x "$SIBLING_VENV_PYTHON" ]; then
    PYTHON="$SIBLING_VENV_PYTHON"
else
    PYTHON="python"
fi

# M9: the backend replaces the old server/ (still launchable from
# archive/server_cli.py for reference). Run from the project root --
# `-m backend.cli` needs it importable -- and resolve ComfyUI/venv
# through paths.py (.env + env) instead of cd'ing there: the backend
# spawns training with cwd=COMFY_DIR itself. Bind all interfaces like
# the old server did; --host/--port passed after this win (argparse
# keeps the last value).
cd "$SCRIPT_DIR"

# Bootstrap: if the server's own packages are missing, install them into a
# temporary environment and come back through here with VENV_PYTHON set.
#
# Asked for before launching rather than inside the server, because the
# server is what those packages are for -- asking it to report its own
# missing dependencies is asking the thing that cannot start. This is a
# no-op (exit 0, no output) whenever they are present, which is every
# start after the first, so the cost on a working install is one subprocess.
#
# Two things about it are deliberate:
#   - It installs ONLY the four packages in requirements.txt. The training
#     stack and the GPU choice are the wizard's, because they are questions
#     and not imports, and this runs before anyone can be asked.
#   - It uses a venv under the system temp directory, named with the pid.
#     Nothing here can break an environment the user already has.
#
# `python -m backend.first_run --check` reports what is missing and exits
# without installing. Set DISTILLATION_NO_BROWSER=1 to stop it opening a
# browser; the install link is printed either way.
"$PYTHON" -m backend.first_run "$@" || exit $?

exec "$PYTHON" -m backend.cli --host 0.0.0.0 "$@"
