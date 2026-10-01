"""SubprocessDatasetTaskGateway -- fork task gateway for ingestion.

One child per task, spawned as ``<venv python> -m
backend.infrastructure.dataset_task_worker --db ... --task ... --dataset
... --kind ... --params <json>``:

- env: ``PYTHONUNBUFFERED``, ``PYTHONPATH`` rooted at the project (the
  child imports ``backend.*`` and ``manager.*``/``core.*``), plus the
  checkpoint/lora directory overrides for parity with the training
  gateway.
- cwd: the project root (unlike the trainer, which needs the ComfyUI
  dir -- the ingestion child takes absolute paths for everything).
- stdout+stderr append to ``<dataset>/task_<id>.log`` (the builder
  prints VAE/encode progress; a pipe could fill and block it, DEVNULL
  would lose the only post-mortem detail we have).
- ``start_new_session=True``: stop() SIGKILLs the whole group.

PID-reuse posture: ``kill`` requires our cmdline marker before it
signals (refuse to kill a stranger); ``is_alive`` requires the marker
too, which also makes *zombies* -- empty cmdline -- read as dead, and
lets a task that outlived a server restart still be seen as alive.
``/proc`` absent (non-Linux) degrades to ``kill(pid, 0)``, training
parity: a mitigation, not a guarantee.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
from pathlib import Path

from ..application.errors import DatasetTaskLaunchError
from ..application.ports.dataset_task_gateway import (
    DatasetTaskGateway,
    DatasetTaskLaunch,
)
from .process_identity import cmdline_mentions
from .workspace import WorkspaceLayout

logger = logging.getLogger(__name__)

CMDLINE_MARKER = "dataset_task_worker"


class SubprocessDatasetTaskGateway(DatasetTaskGateway):
    def __init__(
        self,
        layout: WorkspaceLayout,
        db_path: Path,
        *,
        cmdline_marker: str = CMDLINE_MARKER,
    ) -> None:
        self._layout = layout
        self._db_path = db_path
        self._marker = cmdline_marker
        self._procs: dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # -- spawn -----------------------------------------------------------

    def spawn(self, launch: DatasetTaskLaunch) -> int:
        try:
            cmd = self._build_command(launch)
            env = self._build_env()
            log_path = launch.dataset_root / f"task_{launch.task_id}.log"
            with open(log_path, "w", encoding="utf-8", buffering=1) as log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self._layout.project_root),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except DatasetTaskLaunchError:
            raise
        except Exception as exc:
            raise DatasetTaskLaunchError(
                f"failed to launch dataset task: {exc}"
            ) from exc
        with self._lock:
            self._procs[proc.pid] = proc
        return proc.pid

    def _build_command(self, launch: DatasetTaskLaunch) -> list[str]:
        return [
            self._layout.venv_python,
            "-m",
            "backend.infrastructure.dataset_task_worker",
            "--db",
            str(self._db_path),
            "--task",
            str(launch.task_id),
            "--dataset",
            str(launch.dataset_root),
            "--kind",
            launch.kind,
            "--params",
            json.dumps(launch.params),
        ]

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        project_root = self._layout.project_root
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["CHECKPOINTS_DIR"] = str(self._layout.checkpoints_dir)
        env["LORAS_DIR"] = str(self._layout.loras_dir)
        return env

    # -- liveness ---------------------------------------------------------

    def is_alive(self, pid: int) -> bool:
        with self._lock:
            proc = self._procs.get(pid)
        if proc is not None and proc.poll() is not None:
            return False  # reaped: exited (no zombie left behind)
        match = self._cmdline_marker_match(pid)
        if match is None:
            # /proc unavailable: fall back to the bare liveness probe.
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        return match

    # -- stop --------------------------------------------------------------

    def kill(self, pid: int) -> None:
        with self._lock:
            proc = self._procs.get(pid)
        if proc is not None and proc.poll() is not None:
            return  # already exited; nothing to signal
        match = self._cmdline_marker_match(pid)
        if match is False:
            logger.warning(
                "refusing to signal pid %s: not a dataset task worker "
                "(pid reuse?)", pid,
            )
            return
        # match True, or None (/proc unavailable -- training parity).
        self._signal(pid, signal.SIGKILL)

    @staticmethod
    def _signal(pid: int, sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    # -- guards -------------------------------------------------------------

    def _cmdline_marker_match(self, pid: int) -> bool | None:
        """True: our worker. False: something else (incl. a zombie --
        its /proc cmdline is empty). None: /proc unavailable.

        Shared with the training gateway so the two cannot drift on what
        "is this pid still ours" means (docs 07 F-12).
        """
        return cmdline_mentions(pid, self._marker)
