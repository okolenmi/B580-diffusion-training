"""Looking inside a virtualenv folder without importing anything from it.

The setup page asks which environment the trainer should use. For an existing one the answer is read here: the interpreter inside the
folder lists its installed distributions (metadata only), and the caller decides what is missing. Never writes.
"""

from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

_INVENTORY_CHILD = """
import json, sys
import importlib.metadata as md
seen, out = set(), []
for dist in md.distributions():
    try:
        name, version = dist.metadata["Name"], dist.version
    except Exception:
        continue
    if name and name.lower() not in seen:
        seen.add(name.lower())
        out.append([name, version or ""])
print(json.dumps({"ok": True, "python": "%d.%d.%d" % sys.version_info[:3], "packages": out}))
"""


_DEVICE_CHILD = """
import json
try:
    import torch
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print(json.dumps({"ok": True, "present": False, "reason": "torch %s is installed but sees no Intel GPU" % torch.__version__}))
    else:
        props = torch.xpu.get_device_properties(0)
        print(json.dumps({"ok": True, "present": True, "name": props.name, "total_memory_mb": props.total_memory // (1024 * 1024),
                          "torch": torch.__version__}))
except Exception as exc:
    print(json.dumps({"ok": True, "present": False, "reason": "%s: %s" % (type(exc).__name__, exc)}))
"""


@dataclass(frozen=True, slots=True)
class VenvDevice:
    present: bool
    name: str | None = None
    total_memory_mb: int | None = None
    torch_version: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class VenvInventory:
    interpreter: str | None
    python_version: str | None = None
    packages: tuple[tuple[str, str], ...] = ()
    error: str | None = None


def find_interpreter(venv_dir: str) -> str | None:
    """The interpreter of a virtualenv folder, given the folder or the interpreter file itself."""
    try:
        path = Path(venv_dir).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if path.is_file():
        return str(path) if path.name.lower().startswith("python") else None
    relative = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    candidate = path.joinpath(*relative)
    return str(candidate) if candidate.exists() else None


class VenvInspector(ABC):
    @abstractmethod
    def inspect(self, venv_dir: str) -> VenvInventory:
        """What the environment in ``venv_dir`` holds; an unreadable one is an ``error``, never an exception."""


    @abstractmethod
    def device(self, venv_dir: str) -> VenvDevice:
        """Whether torch in that environment can see the Intel GPU. Imports torch there, so it takes seconds."""


class LocalVenvInspector(VenvInspector):
    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout

    def inspect(self, venv_dir: str) -> VenvInventory:
        interpreter = find_interpreter(venv_dir) if venv_dir.strip() else None
        if interpreter is None:
            where = "bin/python" if os.name != "nt" else "Scripts\\python.exe"
            return VenvInventory(None, error=f"no Python found in {venv_dir or 'that folder'}: expected {where} in a virtualenv folder")
        try:
            done = subprocess.run([interpreter, "-I", "-c", _INVENTORY_CHILD], capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return VenvInventory(interpreter, error=f"{interpreter} did not answer within {self.timeout:.0f}s")
        except OSError as exc:
            return VenvInventory(interpreter, error=f"{interpreter} could not be run: {exc}")
        for line in reversed(done.stdout.splitlines()):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and payload.get("ok"):
                packages = tuple((str(n), str(v)) for n, v in payload.get("packages", ()))
                return VenvInventory(interpreter, payload.get("python"), packages)
        detail = (done.stderr or "").strip().splitlines()
        return VenvInventory(interpreter, error=f"{interpreter} gave no package list" + (f": {detail[-1]}" if detail else ""))

    def device(self, venv_dir: str) -> VenvDevice:
        interpreter = find_interpreter(venv_dir) if venv_dir.strip() else None
        if interpreter is None:
            return VenvDevice(False, reason="no Python found in that folder")
        try:
            done = subprocess.run([interpreter, "-I", "-c", _DEVICE_CHILD], capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return VenvDevice(False, reason=f"the check could not run: {exc}")
        for line in reversed(done.stdout.splitlines()):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and payload.get("ok"):
                return VenvDevice(bool(payload.get("present")), payload.get("name"), payload.get("total_memory_mb"),
                                  payload.get("torch"), payload.get("reason"))
        return VenvDevice(False, reason="the check printed nothing; " + ((done.stderr or "").strip().splitlines() or ["no error text"])[-1])
