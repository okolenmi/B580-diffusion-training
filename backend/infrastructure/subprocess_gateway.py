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
  direct signal when the group cannot be addressed. Every signal --
  ``stop``, the escalation and ``kill`` -- is gated on :meth:`owns`, the
  ``/proc`` cmdline guard against PID reuse (fail-open when ``/proc``
  cannot answer), so a stale row cannot point a signal at an unrelated
  process (docs 07 F-12).

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
from ..application.limits import DEFAULT_STOP_GRACE_SECONDS
from .process_identity import cmdline_mentions
from .workspace import WorkspaceLayout

logger = logging.getLogger(__name__)


class SubprocessTrainingGateway(TrainingGateway):
    def __init__(
        self,
        layout: WorkspaceLayout,
        *,
        stop_grace: float = DEFAULT_STOP_GRACE_SECONDS,
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
            self._refuse_existing_history(launch)
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

    @staticmethod
    def _refuse_existing_history(launch: TrainingLaunch) -> None:
        """Never open a run's log/progress with ``"w"`` over real content.

        Opening with ``"w"`` is how a legacy run's history disappears
        without a word when a fresh database hands out an id that is
        already on disk. Two layers sit above this (startup seeds the id
        sequence above the highest existing ``runs/run_*`` directory, and
        ``StartTraining`` refuses an occupied directory, docs 07 F-04);
        this is the last line for a caller that reaches the gateway
        directly. An empty file is fine -- it is what a failed spawn
        leaves behind.
        """
        for path in (launch.log_path, launch.progress_path):
            try:
                occupied = path.is_file() and path.stat().st_size > 0
            except OSError:
                occupied = False  # unreadable: let the open() below decide
            if occupied:
                raise TrainingLaunchError(
                    f"refusing to overwrite {path}: run {launch.run_id} already "
                    f"has output there. Move the old run aside first -- this "
                    f"server never truncates another run's history."
                )

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
        """Is this pid still the process we think it is?

        Three cases, because they need different evidence:

        * **We spawned it.** `Popen.poll()` is authoritative and cannot be
          confused by a recycled number.
        * **We adopted it** (the server restarted). `os.kill(pid, 0)` only
          answers "does *a* process hold this number" -- so once the
          trainer dies and the number is reused, the supervisor sees
          "alive" forever, the row stays `running`, and `stop()` is then
          refused as "not our trainer". Only a backend restart clears it
          (docs 08 N-07). So for an adopted pid, the liveness signal has
          to be the same one `owns()` uses: does its cmdline still look
          like our trainer?
        * **PermissionError.** The process exists and is simply not ours
          to signal. Reading that as "dead" would tell the supervisor a
          live trainer finished (docs 07 F-12).

        "Cannot tell" stays alive. A cmdline we cannot read is not
        evidence of death, and the cost of guessing wrong here is a stuck
        run rather than a spurious completion.
        """
        proc = self._procs.get(pid)
        if proc is not None:
            return proc.poll() is None

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False  # no such process
        except PermissionError:
            # It exists, it just is not ours to signal.
            logger.debug("pid %s exists but is not ours to signal", pid)
            return True
        except OSError as exc:
            logger.warning("cannot probe pid %s: %s", pid, exc)
            return False

        # The number is taken. For a pid this process did not spawn, that
        # only means something if it is still *our* trainer.
        verdict = cmdline_mentions(pid, self._marker)
        if verdict is False:
            logger.info(
                "pid %s is alive but is not this project's trainer "
                "(recycled pid): reporting it as gone", pid,
            )
            return False
        return True

    def owns(self, pid: int) -> bool:
        """PID-reuse guard, asked directly (docs 07 F-12).

        A process we spawned in this process is ours by construction --
        no /proc race, no marker dependency. For anything else the
        cmdline must mention our entry point; an unreadable /proc cannot
        disprove it, which is the legacy fail-open behaviour.
        """
        proc = self._procs.get(pid)
        if proc is not None:
            return proc.poll() is None
        return cmdline_mentions(pid, self._marker) is not False

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
        if not self._refuse_stranger(pid, "stop"):
            return False
        delivered = self._signal(pid, signal.SIGKILL if force else signal.SIGINT)
        if delivered and not force:
            # Escalate: SIGINT asks the trainer to save and exit, which
            # can take a while on a big checkpoint -- so the grace period
            # is generous by default (docs 07 F-12) rather than the
            # legacy's 3s, which turned "saving" into "killed".
            threading.Thread(
                target=self._escalate,
                args=(pid,),
                name=f"backend-stop-escalation-{pid}",
                daemon=True,
            ).start()
        return delivered

    def _escalate(self, pid: int) -> None:
        threading.Event().wait(self._stop_grace)
        if self.is_alive(pid) and not self._refuse_stranger(pid, "escalation"):
            logger.warning("pid %s ignored SIGINT; escalating to SIGKILL", pid)
            self._signal(pid, signal.SIGKILL)

    def kill(self, pid: int) -> bool:
        if not self._refuse_stranger(pid, "kill"):
            return False
        return self._signal(pid, signal.SIGKILL)

    def _refuse_stranger(self, pid: int, action: str) -> bool:
        """False when this pid is provably not our trainer -- refuse it.

        Every signal goes through here, not just ``kill``: a stale row
        whose pid was recycled must not be able to SIGINT an unrelated
        process (docs 07 F-12).
        """
        if self.owns(pid):
            return True
        logger.warning(
            "refusing to %s pid %s: it is not this project's trainer "
            "(pid reused?)", action, pid,
        )
        return False

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
