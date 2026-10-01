"""Server-side progress reader for structured progress files."""

from __future__ import annotations

import json
from pathlib import Path

from . import db
from .sse import sse


class ProgressFileReader:
    def __init__(self, db_path: Path, run_id: int, total_steps: int):
        self._db_path = db_path
        self._run_id = run_id
        self._total_steps = total_steps
        self._offset = 0

    def read_and_broadcast(self, progress_path: Path) -> dict | None:
        try:
            if not progress_path.exists():
                return None
            
            size = progress_path.stat().st_size
            if size <= self._offset:
                return None

            last_progress = None
            with open(progress_path, "r") as f:
                f.seek(self._offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self._broadcast(data)
                        last_progress = data
                    except json.JSONDecodeError:
                        continue
                
                self._offset = f.tell()

            return last_progress

        except Exception:
            return None

    def _broadcast(self, data: dict):
        phase = data.get("phase", "")

        if phase == "cache_start":
            sse.progress(
                self._run_id,
                step=0,
                total=self._total_steps,
                loss=None, avg_loss=None, lr=None,
                status="Building cache (0%)",
                phase="cache",
                cache_done=0,
                cache_total=data.get("est_trajs", 1),
            )
            db.update_run_progress(
                self._db_path, self._run_id,
                done_steps=0,
                current_loss=None,
                avg_loss=None,
            )
            db.update_run_phase(
                self._db_path, self._run_id,
                phase="cache",
                cache_done=0,
                cache_total=data.get("est_trajs", 1),
            )

        elif phase == "cache":
            # During cache, we still report current training step (0 if start)
            sse.progress(
                self._run_id,
                step=0,
                total=self._total_steps,
                loss=None, avg_loss=None, lr=None,
                status=f"Building cache ({data.get('pct', 0):.0f}%)",
                phase="cache",
                cache_done=data.get("done", 0),
                cache_total=data.get("total", 1),
            )
            db.update_run_phase(
                self._db_path, self._run_id,
                phase="cache",
                cache_done=data.get("done", 0),
                cache_total=data.get("total", 1),
            )

        elif phase == "cache_done":
            sse.progress(
                self._run_id,
                step=0,
                total=self._total_steps,
                loss=None, avg_loss=None, lr=None,
                status="Cache complete",
                phase="cache_done",
                cache_done=data.get("total", 1),
                cache_total=data.get("total", 1),
            )
            db.update_run_phase(
                self._db_path, self._run_id,
                phase="cache_done",
                cache_done=data.get("total", 1),
                cache_total=data.get("total", 1),
            )

        elif phase == "training_start":
            new_total = data.get("total_steps")
            if new_total and (not self._total_steps or new_total > self._total_steps):
                self._total_steps = new_total
            
            sse.progress(
                self._run_id,
                step=0,
                total=self._total_steps,
                status="Training starting...",
                phase="training",
            )
            db.update_run_phase(
                self._db_path, self._run_id,
                phase="training",
            )

        elif phase == "step":
            step = data.get("step", 0)
            data_total = data.get("total")
            if data_total and (not self._total_steps or data_total > self._total_steps):
                self._total_steps = data_total
            
            total = self._total_steps
            loss = data.get("loss")
            avg = data.get("avg")
            lr = data.get("lr")

            db.update_run_progress(
                self._db_path, self._run_id,
                done_steps=step,
                current_loss=loss,
                avg_loss=avg,
                total_steps=total,
            )
            db.log_event(
                self._db_path, self._run_id, "step",
                step=step, loss=loss, avg_loss=avg, lr=lr,
            )
            sse.progress(
                self._run_id,
                step=step, total=total,
                loss=loss, avg_loss=avg, lr=lr,
                phase="training",
            )

        elif phase in ("finished", "stopped", "failed", "killed"):
            sse.status(self._run_id, phase)
