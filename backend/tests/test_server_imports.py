"""The server must start on the install the setup page runs on: no torch, numpy, PIL or safetensors yet.

Importing the server's modules in a process where those packages are blocked catches an import chain that reaches the ML stack
(the sandbox has torch, so only this test can see it). Packages the server may use are in ``requirements.txt``.

Run directly: python backend/tests/test_server_imports.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.tests.support import check, finish

BLOCKED = ("torch", "numpy", "safetensors", "PIL")

PROBE = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {BLOCKED!r}:
            raise ModuleNotFoundError("No module named " + repr(name))
sys.meta_path.insert(0, Block())
import backend.cli, backend.bootstrap
from backend.presentation.app import create_app
from engine.xpu_env import set_perf_env
print("imported")
"""


def test_server_imports_without_the_ml_stack() -> None:
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=120)
    tail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else ""
    check(done.returncode == 0 and "imported" in done.stdout, f"server modules import with {', '.join(BLOCKED)} blocked ({tail})")


if __name__ == "__main__":
    test_server_imports_without_the_ml_stack()
    finish()
