"""SubprocessTrainingGateway -- spawn/stop/reap the trainer via subprocess.

Ports the legacy launch pipeline (``control.build_training_command`` +
``process_manager``), keeping its proven behaviour:

- argv: ``<venv python> -m core.cli --config ... [--steps] [--run-id]
  <start-from flags> [--reset-optimizer]`` -- the config is re-read
  here to resolve student/resume paths (adapter-owned integration).
- env: ``PYTHONUNBUFFERED``, ``PYTHONPATH`` rooted at the project,
  ``CHECKPOINTS_DIR``/``LORAS_DIR`` injected so the child resolves the
  exact directories the server UI shows.
- cwd: the ComfyUI dir; stdout+stderr append to the run's log;
  ``start_new_session=True`` so the whole process group can be
  signalled.
- stop: SIGINT to the process group with SIGKILL escalation after a
  grace period (``force`` skips straight to SIGKILL); fallback to a
  direct signal when the group cannot be addressed.
- kill: the ``/proc/<pid>/cmdline`` guard against PID reuse (fail-open,
  as the legacy implementation did).

Addressed by PID with an internal pid -> Popen map: state survives the
supervisor's view of the world, and every pid we spawned in *this*
process gets reaped (no zombies).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from pathlib import Path

from ..application.errors import TrainingLaunchError
from ..application.ports.training_gateway import (
    TrainingGateway,
    TrainingLaunch,
)
from .workspace import WorkspaceLayout

logger = logging.getLogger(__name__)


class SubprocessTrainingGateway(TrainingGateway):
    def __init__(
        self,
        layout: WorkspaceLayout,
        *,
        stop_grace: float = 3.0,
        cmdline_marker: str = "core.cli",
    ) -> None:
        self._layout = layout
        self._stop_grace = stop_grace
        self._marker = cmdline_marker
        self._procs: dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def spawn(self, launch: TrainingLaunch) -> int:
        try:
            cmd = self._build_command(launch)
            env = self._build_env()
            launch.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(launch.log_path, "w", encoding="utf-8", buffering=1) as log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self._layout.comfy_dir),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except TrainingLaunchError:
            raise
        except Exception as exc:
            raise TrainingLaunchError(
                f"failed to launch trainer: {exc}"
            ) from exc
        with self._lock:
            self._procs[proc.pid] = proc
        return proc.pid

    def _build_command(self, launch: TrainingLaunch) -> list[str]:
        """Faithful port of server/control.build_training_command."""
        from core.config_io import read_config  # repo bridge (adapter-owned)

        config = read_config(launch.config_path)
        cmd = [
            self._layout.venv_python,
            "-m",
            "core.cli",
            "--config",
            str(launch.config_path),
        ]
        if launch.total_steps > 0:
            cmd.extend(["--steps", str(launch.total_steps)])
        if launch.run_id > 0:
            cmd.extend(["--run-id", str(launch.run_id)])

        start_from = launch.start_from
        if start_from == "teacher":
            cmd.append("--fresh")
        elif start_from == "student":
            cmd.append("--fresh")
            student_path = config.paths.student or ""
            if student_path:
                cmd.extend(["--student", student_path])
        elif start_from == "lora_checkpoint":
            cmd.append("--fresh")
        elif start_from == "resume":
            cmd.extend(["--start-from", "resume"])
            resume_checkpoint = config.paths.resume_checkpoint or ""
            if resume_checkpoint and Path(resume_checkpoint).exists():
                cmd.extend(["--student", resume_checkpoint])
            if not launch.reset_optimizer:
                resume_optimizer = config.paths.resume_optimizer or ""
                if resume_optimizer and Path(resume_optimizer).exists():
                    cmd.extend(["--resume-optimizer", resume_optimizer])

        if launch.reset_optimizer:
            cmd.append("--reset-optimizer")
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        project_root = self._layout.project_root
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["CHECKPOINTS_DIR"] = str(self._layout.checkpoints_dir)
        env["LORAS_DIR"] = str(self._layout.loras_dir)
        return env

    # ------------------------------------------------------------------
    # Watch / reap
    # ------------------------------------------------------------------

    def is_alive(self, pid: int) -> bool:
        proc = self._procs.get(pid)
        if proc is not None and proc.poll() is not None:
            return False  # reaped: no longer alive (nor a zombie)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False  # ProcessLookup/Permission/OSError: legacy parity

    def wait_exit_code(self, pid: int, timeout: float = 5.0) -> int | None:
        proc = self._procs.get(pid)
        if proc is None:
            return None
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def stop(self, pid: int, *, force: bool = False) -> bool:
        delivered = self._signal(pid, signal.SIGKILL if force else signal.SIGINT)
        if delivered and not force:
            # Escalate: SIGINT may be ignored by a stuck trainer; the
            # legacy service did the same 3s-after check.
            threading.Thread(
                target=self._escalate,
                args=(pid,),
                name=f"backend-stop-escalation-{pid}",
                daemon=True,
            ).start()
        return delivered

    def _escalate(self, pid: int) -> None:
        threading.Event().wait(self._stop_grace)
        if self.is_alive(pid):
            logger.warning("pid %s ignored SIGINT; escalating to SIGKILL", pid)
            self._signal(pid, signal.SIGKILL)

    def kill(self, pid: int) -> bool:
        if not self._looks_like_our_training_process(pid):
            return False
        return self._signal(pid, signal.SIGKILL)

    @staticmethod
    def _signal(pid: int, sig: int) -> bool:
        try:
            os.killpg(os.getpgid(pid), sig)
            return True
        except OSError:
            try:
                os.kill(pid, sig)
                return True
            except OSError:
                return False

    def _looks_like_our_training_process(self, pid: int) -> bool:
        """PID-reuse guard (legacy port): /proc cmdline must mention our
        entry point. Fails open (True) when /proc is unavailable -- a
        mitigation for the common case, not a hard guarantee."""
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if not cmdline_path.exists():
            return True
        try:
            cmdline = cmdline_path.read_bytes().decode(errors="replace")
            return self._marker in cmdline
        except OSError:
            return True
