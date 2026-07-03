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

# Resolve the python interpreter with the same precedence as
# server.config.Settings.venv_python (used internally to launch training
# subprocesses), so the server itself and the trainer it spawns are
# guaranteed to agree on which interpreter/venv to use:
#   1. VENV_PYTHON env var (or .env), if set and it actually exists
#   2. <parent-of-this-project>/venv/bin/python, if it exists (the
#      project / ComfyUI / venv sibling-folder layout)
#   3. Whatever "python" resolves to on PATH
SIBLING_VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"
if [ -n "${VENV_PYTHON:-}" ] && [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
elif [ -x "$SIBLING_VENV_PYTHON" ]; then
    PYTHON="$SIBLING_VENV_PYTHON"
else
    PYTHON="python"
fi

if [ -n "${COMFY_DIR:-}" ]; then
    cd "$COMFY_DIR"
fi

exec "$PYTHON" "$SCRIPT_DIR/server_cli.py" "$@"
