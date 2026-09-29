"""TrainingGateway port -- launching, signalling, and reaping the trainer.

The application speaks in terms of a :class:`TrainingLaunch` (pure
intent: which config, which start-from policy, where artifacts go) and
raw PIDs. All knowledge of *how* this repo invokes ``core.cli`` --
interpreter path, argv construction, process groups, the ``/proc``
PID-reuse guard -- lives in the infrastructure adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingLaunch:
    """Everything the application knows about a run to be spawned."""

    run_id: int
    config_path: Path  # absolute
    mode: str
    total_steps: int
    start_from: str  # teacher | student | resume | lora_checkpoint
    reset_optimizer: bool
    log_path: Path  # child's stdout+stderr land here
    progress_path: Path  # child writes JSONL telemetry here (its own
    #                     convention via ``paths.get_progress_path``; the
    #                     adapter must pass through, never re-invent)


class TrainingGateway(ABC):
    """Spawn/stop/watch a training subprocess, addressed by PID.

    Contract: ``spawn`` raises ``TrainingLaunchError`` for *any* launch
    failure (missing interpreter, unreadable config, ``OSError``) --
    the caller finalises the run and surfaces the error, so the adapter
    must never leak raw OS exceptions.
    """

    @abstractmethod
    def spawn(self, launch: TrainingLaunch) -> int:
        """Start the trainer detached into its own session; return PID."""
        raise NotImplementedError

    @abstractmethod
    def is_alive(self, pid: int) -> bool:
        """Process-exists check (reaps our own children as a side effect)."""
        raise NotImplementedError

    @abstractmethod
    def wait_exit_code(self, pid: int, timeout: float = 5.0) -> int | None:
        """Exit code once the process is dead; ``None`` if still alive
        or the pid is not ours to reap."""
        raise NotImplementedError

    @abstractmethod
    def stop(self, pid: int, *, force: bool = False) -> bool:
        """Graceful stop (SIGINT to the process group) with escalation to
        SIGKILL after a grace period; ``force`` skips straight to SIGKILL.
        Returns whether the initial signal was delivered."""
        raise NotImplementedError

    @abstractmethod
    def kill(self, pid: int) -> bool:
        """Unconditional force-kill, guarded against PID reuse (only
        signals a process whose cmdline still looks like our trainer).
        Used by startup orphan cleanup."""
        raise NotImplementedError
