#!/bin/bash
# Launch the training web server.
#
# ComfyUI directory and the Python interpreter to use are no longer
# hardcoded here -- both can be customized via environment variables
# (same convention used internally by paths.get_comfy_dir() /
# server.config.Settings.venv_python):
#
#   COMFY_DIR    Path to your ComfyUI installation.
#                Default: auto-detected (see paths.get_comfy_dir()) --
#                works out of the box if ComfyUI is a sibling folder of
#                this project, or if you run this script from inside
#                ComfyUI's own directory.
#
#   VENV_PYTHON  Path to the python interpreter to launch the server (and,
#                internally, training subprocesses) with.
#                Default: <parent-of-this-project>/venv/bin/python if it
#                exists, otherwise whatever "python" resolves to on PATH.
#
# Example (fully explicit, no directory-layout assumptions at all):
#   COMFY_DIR=/path/to/ComfyUI VENV_PYTHON=/path/to/venv/bin/python ./run_server.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the python interpreter with the same precedence as
# server.config.Settings.venv_python (used internally to launch training
# subprocesses), so the server itself and the trainer it spawns are
# guaranteed to agree on which interpreter/venv to use:
#   1. VENV_PYTHON env var, if set and it actually exists
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
