"""The package installer that runs pip in a subprocess."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path

from ..application.ports.package_installer import InstallError, InstallRequest, PackageInstaller


class PipInstaller(PackageInstaller):
    """``python -m pip`` in the target interpreter, or ``uv pip`` when that interpreter has no pip (environments made by uv)."""

    def __init__(self, timeout: float = 3600.0) -> None:
        self.timeout = timeout

    @staticmethod
    def build_command(request: InstallRequest, constraints_path: Path | None = None, *, uv: str | None = None) -> list[str]:
        """The exact command. Pure: it writes nothing."""
        if uv:
            command = [uv, "pip", "install", "--python", request.target_python, "--no-progress"]
        else:
            command = [request.target_python, "-m", "pip", "install",
                       "--disable-pip-version-check", "--no-input", "--progress-bar", "off"]
        if constraints_path is not None:
            command += ["--constraint", str(constraints_path)]
        if request.index_url:
            command += ["--index-url", request.index_url]
        # ``--`` makes everything after it a requirement even if it starts with ``-``.
        return command + ["--", *request.packages]

    @staticmethod
    def _has_pip(interpreter: str) -> bool:
        try:
            done = subprocess.run([interpreter, "-m", "pip", "--version"], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return done.returncode == 0

    def install(self, request: InstallRequest, on_line=None) -> None:
        if not Path(request.target_python).exists():
            raise InstallError(f"{request.target_python} does not exist, so nothing was installed into {request.target_label}.")
        uv = None
        if not self._has_pip(request.target_python):
            uv = shutil.which("uv")
            if uv is None:
                raise InstallError(
                    f"{request.target_python} has no pip and uv is not installed. Run "
                    f"'{request.target_python} -m ensurepip' once, or choose a new environment. Nothing was changed.")

        directory = None
        constraints_path = None
        if request.constraints:
            directory = Path(tempfile.mkdtemp(prefix="trainer-install-"))
            constraints_path = directory / "constraints.txt"
            constraints_path.write_text("\n".join(request.constraints) + "\n", encoding="utf-8")
        command = self.build_command(request, constraints_path, uv=uv)
        if on_line:
            on_line("$ " + " ".join(command))
        tail: deque[str] = deque(maxlen=8)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as exc:
            raise InstallError(f"could not run the installer for {request.target_python}: {exc}") from exc
        try:
            timer = _Deadline(process, self.timeout)
            for raw in process.stdout:
                line = raw.rstrip()
                if on_line:
                    on_line(line)
                if line.strip() and "WARNING" not in line:
                    tail.append(line)
            code = process.wait()
        finally:
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)
        if timer.expired:
            raise InstallError(f"the install did not finish within {self.timeout / 60:.0f} minutes and was stopped. "
                               "If this machine is behind a proxy, pip needs its index configured.")
        if code != 0:
            raise InstallError(f"the installer exited {code}. " + "\n".join(tail))


class _Deadline:
    """Kills a process after ``seconds``; ``expired`` says whether it did."""

    def __init__(self, process: subprocess.Popen, seconds: float) -> None:
        import threading

        self.expired = False

        def fire() -> None:
            if process.poll() is None:
                self.expired = True
                process.kill()

        self._timer = threading.Timer(seconds, fire)
        self._timer.daemon = True
        self._timer.start()
