"""StartInstall / GetInstall -- the first thing that writes anything.

A full install is a multi-gigabyte download, so the request returns a job id immediately and the page polls. Jobs live in this
process; a job that cannot be found afterwards is reported as ``interrupted`` rather than success -- pip's own writes are durable, so
only the *report* is lost, and re-inspecting the environment answers "is it installed" regardless of any job.

What gets installed is decided here, from the environment itself, never from the request: the request names only the target. The
missing manifest packages are computed by ``InspectEnvironment``; the accelerator build of torch comes from its own index in its own
stage; for an environment that already exists every installed package is pinned so pip can add but not move.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from packaging.utils import canonicalize_name

from ..errors import ApplicationError
from ..ports.package_installer import InstallError, InstallRequest, PackageInstaller, create_virtualenv
from ..ports.requirements_manifest import ACCELERATOR_PACKAGES, TORCH_INDEX_URL
from .inspect_environment import InspectEnvironment

logger = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
INTERRUPTED = "interrupted"
TERMINAL = frozenset({SUCCEEDED, FAILED, INTERRUPTED})


def canonicalize(name: str) -> str:
    return canonicalize_name(name)


@dataclass
class InstallJob:
    """One install and its log. Written by the worker thread and read by request threads, so every field a reader touches goes through ``_lock``."""

    id: str
    target_label: str
    packages: tuple[str, ...]
    state: str = QUEUED
    log: deque = field(default_factory=lambda: deque(maxlen=600))
    error: str | None = None
    #: The interpreter the packages went into, for the apply step to store.
    target_python: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "state": self.state,
                "target_label": self.target_label,
                "packages": list(self.packages),
                "log": list(self.log),
                "error": self.error,
                "target_python": self.target_python,
                "terminal": self.state in TERMINAL,
            }

    def line(self, text: str) -> None:
        with self._lock:
            self.log.append(text)

    def set_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self.state = state
            if error is not None:
                self.error = error


class JobNotFound(Exception):
    """No such job. Not an error the user caused."""


class InstallJobNotFound(ApplicationError):
    """A job id this process does not know (404): never existed, or lost in a restart."""

    code = "install_job_not_found"
    status_code = 404

    def __init__(self, job_id: str) -> None:
        super().__init__(f"install job {job_id} is not known to this server. It either never existed or was lost when the server "
                         "restarted; inspect the environment again to see what is installed.")


class InstallAlreadyRunning(Exception):
    """A second install was asked for while one is in flight."""


@dataclass(slots=True)
class StartInstall:
    """Begin an install and return its job immediately."""

    installer: PackageInstaller
    inspect: InspectEnvironment
    #: The interpreter that *creates* a new venv: the one running the server, the only one known to work.
    base_python: str = ""
    jobs: dict[str, InstallJob] = field(default_factory=dict)
    _active: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def execute(self, target: str, venv_dir: str | None = None) -> InstallJob:
        """Queue an install into ``target`` ("new" or "existing" with ``venv_dir``). Returns without waiting for it."""
        report = self.inspect.for_target(target, venv_dir)
        if report.error:
            raise InstallError(f"{report.error}. Nothing was changed.")
        if report.problems:
            raise InstallError(" ".join(report.problems) + " Nothing was changed.")
        if not report.to_install:
            raise InstallError("every package this project needs is already installed there; there is nothing to install.")

        label = "a new environment in the project folder" if target == "new" else report.venv_dir
        job = InstallJob(id=uuid.uuid4().hex[:12], target_label=label, packages=report.to_install)
        with self._lock:
            if self._active is not None:
                running = self.jobs.get(self._active)
                if running and running.state not in TERMINAL:
                    raise InstallAlreadyRunning(running.id)
            self.jobs[job.id] = job
            self._active = job.id
        threading.Thread(target=self._run, args=(job, report), name=f"install-{job.id}", daemon=True).start()
        return job

    def _run(self, job: InstallJob, report) -> None:
        job.set_state(RUNNING)
        try:
            python = report.interpreter
            if python is None:
                target = Path(report.venv_dir)
                job.line(f"Creating a virtualenv at {target}")
                python = create_virtualenv(self.base_python or "python3", target)
                job.line(f"Created {python}")
            accelerator = tuple(n for n in report.to_install if canonicalize(n) in ACCELERATOR_PACKAGES)
            rest = tuple(n for n in report.to_install if n not in accelerator)
            # The accelerator build first and alone: it is the one package that comes from a different index, and the rest must
            # then see it as already installed instead of replacing it with the CPU/CUDA wheel from PyPI.
            for packages, index in ((accelerator, TORCH_INDEX_URL), (rest, None)):
                if not packages:
                    continue
                job.line("")
                job.line(f"== {', '.join(packages)}" + (f"  (from {index})" if index else ""))
                self.installer.install(
                    InstallRequest(target_python=python, packages=packages, constraints=report.constraints,
                                   index_url=index, target_label=job.target_label),
                    on_line=job.line,
                )
            job.target_python = python
            job.set_state(SUCCEEDED)
        except InstallError as exc:
            job.set_state(FAILED, str(exc))
        except Exception as exc:  # noqa: BLE001 -- a job must always finish, or the page polls a spinner for ever
            logger.exception("install job %s failed unexpectedly", job.id)
            job.set_state(FAILED, f"the install failed unexpectedly: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                if self._active == job.id:
                    self._active = None


@dataclass(slots=True)
class GetInstall:
    jobs: dict[str, InstallJob]

    def execute(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job.snapshot()
