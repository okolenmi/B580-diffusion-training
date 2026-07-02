"""Structured progress writer.

Writes JSON progress lines to a .progress.jsonl file alongside the log.
The server monitors this file for reliable progress tracking.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure paths is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from paths import get_progress_path


class ProgressWriter:
    # step() is called on every single training step (potentially hundreds
    # of thousands of times per run). Flushing on every call -- as the code
    # used to do unconditionally -- turns into that many flush syscalls,
    # which is real overhead on a slow/networked filesystem for no benefit
    # a human is going to notice. Phase-transition events (cache_start,
    # training_start, done, ...) still force an immediate flush; step()
    # writes are throttled to at most once per _FLUSH_INTERVAL_SEC so the
    # UI still updates in near-real-time.
    _FLUSH_INTERVAL_SEC = 1.0

    def __init__(self, run_id: int | None = None):
        if run_id is not None:
            self._path = get_progress_path(run_id)
        else:
            self._path = None

        # Skip if no path
        if self._path is None:
            self._disabled = True
            self._handle = None
            self._last_ts = 0.0
            return

        self._disabled = False
        self._handle = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Keep the file handle open to avoid repeated open/close overhead
            self._handle = open(self._path, "w", encoding="utf-8")
        except OSError as e:
            # Progress reporting is a monitoring nice-to-have, not something
            # that should be able to take down an otherwise-healthy training
            # run just because the run directory happened to be unwritable.
            print(f"    [WARN] Could not open progress file {self._path}: {e}. "
                  f"Progress reporting disabled for this run.")
            self._disabled = True

        self._last_ts = 0.0
        self._last_flush_ts = 0.0
        self._step_count = 0

    def _write(self, data: dict, force_flush: bool = False):
        if self._disabled:
            return
        now = time.time()
        data["ts"] = now
        line = json.dumps(data, separators=(",", ":"))
        try:
            self._handle.write(line + "\n")
            self._step_count += 1
            if force_flush or (now - self._last_flush_ts) >= self._FLUSH_INTERVAL_SEC:
                self._handle.flush()
                self._last_flush_ts = now
        except OSError as e:
            print(f"    [WARN] Progress write failed ({e}). Disabling progress reporting.")
            self._disabled = True
            return

        self._last_ts = now

    def cache_start(self, target: int, est_trajs: int):
        self._write({"phase": "cache_start", "target": target, "est_trajs": est_trajs}, force_flush=True)

    def cache_progress(self, done: int, total: int, pct: float):
        self._write({"phase": "cache", "done": done, "total": total, "pct": round(pct, 1)})

    def cache_done(self, n_samples: int, n_trajs: int):
        self._write({"phase": "cache_done", "samples": n_samples, "trajs": n_trajs}, force_flush=True)

    def training_start(self, run_steps: int, total_steps: int, cycles: int):
        self._write({
            "phase": "training_start",
            "run_steps": run_steps,
            "total_steps": total_steps,
            "cycles": cycles,
        }, force_flush=True)

    def step(self, global_step: int, total_steps: int,
             loss: float, avg: float, std: float, lr: float,
             eta_sec: float | None = None, cycle: int | None = None):
        data = {
            "phase": "step",
            "step": global_step,
            "total": total_steps,
            "loss": round(loss, 6),
            "avg": round(avg, 6),
            "std": round(std, 6),
            "lr": round(lr, 10),
        }
        if eta_sec is not None:
            data["eta"] = round(eta_sec, 1)
        if cycle is not None:
            data["cycle"] = cycle
        self._write(data)

    def done(self, status: str = "finished"):
        self._write({"phase": status}, force_flush=True)
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass

    def __del__(self):
        if getattr(self, "_handle", None) is not None:
            try:
                self._handle.close()
            except OSError:
                pass

