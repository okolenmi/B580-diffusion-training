"""Training service — orchestrates training runs and monitors progress."""

import os
from pathlib import Path
from threading import Thread
from typing import Optional

from . import db
from .config import Settings, settings
from .monitor import RunMonitor
from .process_manager import (
    is_process_alive,
    launch_training_process,
    send_signal,
)
from .sse import sse


class TrainingService:
    """Manages training subprocesses."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._proc = None
        self._stop_requested = False
        self._run_id: Optional[int] = None
        self._monitor_thread: Optional[Thread] = None
        self._total_steps: int = 0

    @property
    def is_running(self) -> bool:
        """True if a training process is currently active."""
        return is_process_alive(self._proc)

    @property
    def active_run_id(self) -> Optional[int]:
        """ID of the currently active run, if any."""
        return self._run_id if self.is_running else None

    def start_run(self, config_path: str, mode: str, total_steps: int, cmd: list[str] = None, run_id: int = None) -> int:
        """Launch a training run. Returns the run_id."""
        if self.is_running:
            raise RuntimeError("A run is already in progress")

        self._cleanup_stale_db_entry()

        # Use provided run_id or create a new one
        if run_id is None:
            run_id = db.create_run(self.settings.db_path, config_path, mode, total_steps)
        self._run_id = run_id
        self._total_steps = total_steps
        self._stop_requested = False

        run_dir = self.settings.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.settings.run_log_path(run_id)
        progress_path = self.settings.run_progress_path(run_id)

        if cmd is None:
            cmd = [
                self.settings.venv_python, "-m", "core.cli",
                "--config", config_path,
            ]

        # Launch subprocess
        self._proc = launch_training_process(
            cmd=cmd,
            comfy_dir=self.settings.comfy_dir,
            project_root=self.settings.project_root,
            log_path=log_path,
            checkpoints_dir=self.settings.checkpoints_dir,
            loras_dir=self.settings.loras_dir,
        )

        # Update DB and notify
        db.update_run_status(self.settings.db_path, run_id, "running", pid=self._proc.pid)
        db.log_event(
            self.settings.db_path, run_id, "start",
            message=f"PID={self._proc.pid}, config={config_path}, mode={mode}",
        )

        sse.status(run_id, "running")
        sse.progress(run_id, step=0, total=total_steps)

        # Start monitor thread
        self._monitor_thread = Thread(
            target=self._monitor,
            args=(log_path,),
            daemon=True,
        )
        self._monitor_thread.start()

        return run_id

    def stop_run(self, force: bool = False):
        """Stop the current run."""
        if not self.is_running:
            return

        self._stop_requested = True

        if force:
            send_signal(self._proc, force=True)
            return

        send_signal(self._proc, force=False)

        def _escalate_if_still_running():
            import time
            time.sleep(3)  # Give process time to flush output
            if self.is_running:
                send_signal(self._proc, force=True)

        Thread(target=_escalate_if_still_running, daemon=True).start()

    def reset(self):
        """Force-clear internal state."""
        self._proc = None
        self._run_id = None
        self._total_steps = 0
        self._stop_requested = False

    def get_log_tail(self, run_id: int, lines: int = 100) -> str:
        log_file = self.settings.run_log_path(run_id)
        if not log_file.exists():
            return ""
        
        try:
            with open(log_file, "r") as f:
                all_lines = f.readlines()
            return "".join(all_lines[-lines:])
        except Exception:
            return ""

    def _cleanup_stale_db_entry(self):
        """Check if DB thinks a run is active but it's not."""
        active = db.get_active_run(self.settings.db_path)
        if active and active.get("pid"):
            try:
                os.kill(active["pid"], 0)
            except (ProcessLookupError, OSError):
                db.update_run_status(self.settings.db_path, active["id"], "failed",
                                     error_msg="Stale DB entry cleaned up")

    def _monitor(self, log_path: Path):
        """Internal monitoring loop (runs in a separate thread)."""
        if self._proc is None or self._run_id is None:
            return

        run_id = self._run_id
        monitor = RunMonitor(self.settings.db_path, run_id, self._total_steps)

        final_progress = monitor.start(
            proc=self._proc,
            log_path=log_path,
            stop_requested=lambda: self._stop_requested,
        )

        exit_code = self._proc.wait()
        
        # Determine final status
        status = self._determine_status(exit_code)
        
        db.update_run_status(self.settings.db_path, run_id, status,
                             error_msg=f"Exit code {exit_code}" if status == "failed" else None)
        sse.status(run_id, status, error=f"Exit code {exit_code}" if status == "failed" else None)

        db.log_event(
            self.settings.db_path, run_id, "end",
            step=final_progress.get("step", 0),
            message=f"Exited with code {exit_code}",
        )

        monitor.monitor_post_exit(
            proc=self._proc,
            log_path=log_path,
            stop_requested=self._stop_requested,
            status=status,
        )

        self._proc = None
        self._run_id = None

    def _determine_status(self, exit_code: int) -> str:
        if exit_code == -9: return "killed"
        if self._stop_requested: return "stopped"
        if exit_code == 0: return "finished"
        return "failed"


# Dependency Provider
_service: Optional[TrainingService] = None

def get_training_service() -> TrainingService:
    global _service
    if _service is None:
        _service = TrainingService(settings)
    return _service
