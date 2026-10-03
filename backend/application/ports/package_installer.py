"""Running pip, as a port -- the only thing in the installer that writes.

Everything else in the installer *looks*. `CheckComfyConflicts` reads two
files in another interpreter, `CheckRequirements` reads dist-info, both are
pure, and neither can leave the disk changed. This port is where the
project's safety argument stops being free: `pip install` into a virtualenv
this project does not own is the one operation that can break something the
user already had.

Three properties, and each one is a refusal rather than a mechanism:

* **`--constraint`, not resolution.** The caller supplies an exact-pinned
  constraints file (built by `CheckComfyConflicts`, which pins *every*
  pre-existing package to its installed version). pip is then unable to
  move anything already present, which makes "this cannot break their
  ComfyUI" a property of the command line rather than a hope about pip's
  behaviour.
* **No resolver of our own.** We do not pre-compute a solution, do not
  offer "try again with --upgrade", and do not fall back to another index.
  If pip refuses, that is the answer, and it is a correct one: a conflict
  this project cannot resolve without changing something ComfyUI declares
  is the user's to resolve. An installer that negotiates is an installer
  that eventually negotiates the wrong thing.
* **No ``--user``, no ``--break-system-packages``, no implicit global.**
  Every install targets a named interpreter explicitly, so there is no
  version of this that writes outside a virtualenv the user chose.

The command is streamed, not captured: a 2.5 GB download that appears to
have hung is indistinguishable from one that is about to fail, and a user
watching a frozen terminal for ten minutes learns nothing. The lines go
into the job's log where the UI can show them.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..errors import ApplicationError


class InstallError(ApplicationError):
    """pip failed, or the request to run it was refused.

    An `ApplicationError` rather than a bare Exception, so it reaches the
    API error envelope with a code and a 400 instead of escaping as a 500.
    That distinction is the whole point of a refusal: "we will not do this
    and nothing changed" is not the same report as "something broke", and
    a wizard that shows a 500 for a deliberate refusal cannot tell them
    apart either.

    Not carrying the whole log: the log goes to the job, and a job that
    must be read in two places to understand a failure is a job nobody
    reads.
    """

    code = "install_refused"
    status_code = 400


@dataclass(frozen=True, slots=True)
class InstallRequest:
    """One pip invocation, fully determined before anything runs.

    `target_python` is an absolute path to an interpreter, never a bare
    ``python`` -- resolving it here would mean the caller and pip could
    disagree about which environment is being written to, and the whole
    point of naming it is that there is no ambiguity about it.
    """

    target_python: str
    packages: tuple[str, ...]
    #: Exact pins from the conflict check, or None for a fresh venv where
    #: there is nothing to protect.
    constraints: tuple[str, ...] = ()
    #: Human label for the job list, e.g. "ComfyUI's virtualenv".
    target_label: str = "this project"


class PackageInstaller(ABC):
    """Runs pip. One method, and it is the only method."""

    @abstractmethod
    def install(self, request: InstallRequest, on_line=None) -> None:
        """Install, streaming each output line to `on_line`.

        `on_line` is called with one decoded line at a time. It is called
        from a worker thread, so anything it touches must be safe to touch
        from there -- which is why it is a callback and not a queue the
        caller has to remember to drain.

        Raises `InstallError` if pip exits non-zero. Returns None on
        success. Never raises anything else: a port whose failure modes
        are only some of the failure modes is one the caller cannot handle
        exhaustively.
        """


@dataclass(slots=True)
class PipInstaller(PackageInstaller):
    """The real one: `python -m pip`, in the target interpreter.

    A subprocess rather than an in-process pip API, for the same reason
    the device probe is one: importing torch to install it would cost
    gigabytes in the server process, and a torch build that crashes on
    import could take the server with it.
    """

    timeout: float = 3600.0
    #: Kept, and the tail is what a failure message shows when the error
    #: is pip's last words rather than ours.
    max_log_lines: int = 400

    def build_command(
        self, request: InstallRequest, constraints_path: Path | None = None
    ) -> list[str]:
        """The exact command. **Pure** -- it writes nothing.

        Split from the constraints file because the command is what gets
        shown to the user *before* anything runs, and a function that both
        displays a plan and creates a file cannot be called to look at the
        plan. An earlier version wrote the file here, so inspecting the
        command left a temporary directory behind every time.
        """
        command = [
            request.target_python,
            "-m", "pip", "install",
            "--disable-pip-version-check",
            "--no-input",
        ]
        if constraints_path is not None:
            # A constraints file, not a requirements file: constraints are
            # additive and cannot cause anything to be installed, so a
            # constraint that turns out to be unnecessary is simply ignored.
            # Writing them to a real file is what makes pip honour them;
            # `-c` takes a path, not stdin.
            command += ["--constraint", str(constraints_path)]
        command += list(request.packages)
        return command

    @staticmethod
    def write_constraints(
        request: InstallRequest,
    ) -> tuple[Path, Path] | None:
        """`(file, directory_to_remove)`, or None when there are none.

        Both returned rather than the directory being stashed on the Path:
        an attribute set at runtime on a stdlib type is a trick that reads
        as clever and costs the next reader an afternoon.
        """
        """The pins, as a file, or None when there are none.

        In the system temp directory, not the project: it is a build
        artefact of one install, it can be 185 lines, and it must not turn
        up in a git status. `mkdtemp` rather than a predictable name so two
        installs cannot read each other's pins.
        """
        import tempfile

        if not request.constraints:
            return None
        directory = Path(tempfile.mkdtemp(prefix="distillation-install-"))
        path = directory / "constraints.txt"
        path.write_text(
            "\n".join(request.constraints) + "\n", encoding="utf-8"
        )
        # Returned so the caller can remove it; the directory is small but
        # "how many stale constraint files are in /tmp" is not a question
        # anyone should have to ask.
        return path, directory

    def install(self, request: InstallRequest, on_line=None) -> None:
        if not Path(request.target_python).exists():
            raise InstallError(
                f"{request.target_python} does not exist, so nothing was "
                f"installed into {request.target_label}."
            )

        import shutil

        written = self.write_constraints(request)
        constraints_path = written[0] if written else None
        command = self.build_command(request, constraints_path)
        if on_line:
            on_line(f"$ {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise InstallError(
                f"pip did not finish within {self.timeout / 60:.0f} minutes. "
                f"Nothing was changed. If this machine is behind a proxy, "
                f"pip needs its index configured -- the lines above show "
                f"where it stopped."
            ) from exc
        except OSError as exc:
            raise InstallError(
                f"could not run {request.target_python} -m pip: {exc}"
            ) from exc
        finally:
            if written is not None:
                shutil.rmtree(written[1], ignore_errors=True)

        output = (result.stdout or "") + (result.stderr or "")
        if on_line:
            for line in output.splitlines():
                on_line(line.rstrip())

        if result.returncode != 0:
            tail = [
                line for line in output.splitlines()
                if line.strip() and "WARNING" not in line
            ][-6:]
            raise InstallError(
                f"pip exited {result.returncode}. Nothing was changed. "
                + ("\n".join(tail) if tail else "")
            )


def create_virtualenv(python: str, target: Path) -> str:
    """Create a virtualenv at `target`, returning its interpreter.

    `python -m venv` rather than virtualenv/uv: it is stdlib, so this
    works on a machine with nothing installed, and it is the same call the
    preflight uses -- so "what the bootstrap made" and "what the wizard
    makes" are the same kind of directory.

    Refuses rather than repairing: an existing directory at `target` that
    is not a virtualenv is a thing the user put there, and overwriting it
    is not this code's decision to make.
    """
    if target.exists() and not (target / "bin" / "python").exists() \
            and not (target / "Scripts" / "python.exe").exists():
        raise InstallError(
            f"{target} already exists and is not a virtualenv. Move it "
            f"aside, or choose the other option -- nothing was changed."
        )

    try:
        subprocess.run(
            [python, "-m", "venv", str(target)],
            check=True, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise InstallError(
            f"creating {target} timed out after 10 minutes. Is the disk "
            f"full, or is it not writable?"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise InstallError(
            f"could not create a virtualenv at {target}"
            + (f":\n    {detail[-1]}" if detail else "")
        ) from exc

    interpreter = target / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    if not interpreter.exists():
        raise InstallError(
            f"the environment was created but has no interpreter at "
            f"{interpreter}"
        )
    return str(interpreter)