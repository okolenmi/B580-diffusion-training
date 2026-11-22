"""Running pip, as a port -- the only thing in the installer that writes.

Everything else in the installer *looks*. This port is where the project's safety argument stops being free, so it keeps two
properties:

* **`--constraint`, not resolution.** For an environment that already exists the caller passes every installed package as an exact pin.
  pip can then add what is missing but cannot move anything already present; "this cannot break that environment" is a property of
  the command line rather than a hope about pip's behaviour.
* **No resolver of our own, no ``--user``, no implicit global.** Every install names an interpreter explicitly. If pip refuses, that
  is the answer shown.

The output is streamed line by line: a multi-gigabyte download that shows nothing is indistinguishable from a hung one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..errors import ApplicationError


class InstallError(ApplicationError):
    """pip failed, or the request to run it was refused. Reaches the API as a 400 with a code, never a 500."""

    code = "install_refused"
    status_code = 400


@dataclass(frozen=True, slots=True)
class InstallRequest:
    """One pip invocation, fully determined before anything runs.

    ``target_python`` is an absolute interpreter path so that the caller and pip cannot disagree about which environment is written.
    """

    target_python: str
    packages: tuple[str, ...]
    #: Exact ``name==version`` pins protecting an environment that already exists.
    constraints: tuple[str, ...] = ()
    #: Replaces pip's default index. Used for the accelerator wheels, which PyPI does not carry for Intel GPUs.
    index_url: str | None = None
    target_label: str = "this project"


class PackageInstaller(ABC):
    @abstractmethod
    def install(self, request: InstallRequest, on_line=None) -> None:
        """Install, streaming each output line to ``on_line`` (called from a worker thread).

        Raises ``InstallError`` if the installer exits non-zero and nothing else.
        """






def remove_virtualenv(target: Path) -> None:
    """Delete a virtualenv folder. Refuses anything that is not one (no ``pyvenv.cfg``) or is a link: it is something the user put there."""
    if target.is_symlink() or not (target / "pyvenv.cfg").is_file():
        raise InstallError(f"{target} is not a virtualenv, so it was not deleted. Nothing was changed.")
    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise InstallError(f"could not delete {target}: {exc}. Close whatever is using it, or delete the folder yourself.") from exc


def create_virtualenv(python: str, target: Path) -> str:
    """Create a virtualenv at ``target`` and return its interpreter; an existing virtualenv there is reused.

    Refuses rather than repairs a directory that is not a virtualenv: it is something the user put there.
    """
    relative = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    interpreter = target.joinpath(*relative)
    if interpreter.exists():
        return str(interpreter)
    if target.exists() and any(target.iterdir()):
        raise InstallError(f"{target} already exists and is not a virtualenv. Move it aside, or choose an existing environment instead. "
                           "Nothing was changed.")
    try:
        subprocess.run([python, "-m", "venv", str(target)], check=True, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise InstallError(f"creating {target} timed out after 10 minutes. Is the disk full or not writable?") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise InstallError(f"could not create a virtualenv at {target}" + (f":\n    {detail[-1]}" if detail else "")) from exc
    if not interpreter.exists():
        raise InstallError(f"the environment was created but has no interpreter at {interpreter}")
    return str(interpreter)
