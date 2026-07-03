"""Run monitor — reads progress files for training status."""

import sys
import time
from pathlib import Path

# Ensure paths is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from paths import get_progress_path
from . import db
from .progress_file import ProgressFileReader


class RunMonitor:
    def __init__(self, db_path: Path, run_id: int, total_steps: int):
        self._db_path = db_path
        self._run_id = run_id
        self._total_steps = total_steps

    def start(self, proc, log_path: Path, stop_requested: callable) -> dict:
        self._progress_reader = ProgressFileReader(
            self._db_path, self._run_id, self._total_steps)

        last_step = 0
        last_loss = None
        last_avg = None

        progress_path = get_progress_path(self._run_id)

        try:
            # Wait for process to finish
            while proc.poll() is None:
                if progress_path.exists():
                    data = self._progress_reader.read_and_broadcast(progress_path)
                    if data:
                        if data.get("step") is not None:
                            last_step = data.get("step", last_step)
                        if data.get("loss") is not None:
                            last_loss = data.get("loss")
                        if data.get("avg") is not None:
                            last_avg = data.get("avg")

                time.sleep(0.5)

                if stop_requested():
                    break

        except Exception as e:
            db.log_event(
                self._db_path, self._run_id, "error",
                message=f"Monitor error: {e}",
            )

        try:
            proc.wait(timeout=5)
        except Exception:
            pass

        return {
            "step": last_step,
            "loss": last_loss,
            "avg_loss": last_avg,
            "total_steps": self._total_steps,
        }

    def monitor_post_exit(self, proc, log_path: Path,
                         stop_requested: bool, status: str) -> dict:
        exit_code = proc.returncode if proc else -1

        try:
            with open(log_path, "a") as f:
                f.write(f"\n--- RUN ENDED: status={status}, exit_code={exit_code} ---\n")
        except Exception:
            pass
