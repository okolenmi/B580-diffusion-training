"""dataset_task_worker -- child entry point for dataset tasks (M3b).

Spawned by ``SubprocessDatasetTaskGateway`` as a fresh interpreter so
the server never imports torch for a dataset operation. Contract:

- the reporter below writes task rows into backend.db through the same
  ``SqliteDatasetTasks`` port the server uses (WAL makes the side-by-
  side writer safe), so progress, completion, and failure are all just
  CAS row updates;
- ``manager.builder.DataTaskRunner`` owns the real work and calls
  ``progress``/``finished``/``failed`` itself -- including the
  ``ensure_v2`` refusal, which lands in the row as a normal failure
  message instead of a stack trace in a log nobody reads;
- any exception escaping the builder is reported here as a backstop
  (the CAS makes a duplicate ``failed`` write a no-op).

Heavy imports (``manager`` -> torch/xpu) happen after argparse so
``--help`` and argument errors stay instant and dependency-free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from ..application.ports.dataset_tasks import TaskStatus


class _Reporter:
    """Duck-typed ``progress``/``finished``/``failed`` for the builder.

    Every write carries this process's pid, so the row's pid is the
    child's real pid even when the server's post-spawn bookkeeping was
    lost to a crash (the row self-identifies on first progress).
    """

    def __init__(self, tasks, task_id: int) -> None:
        self._tasks = tasks
        self._task_id = task_id

    def progress(self, current: int) -> None:
        self._tasks.update_progress(self._task_id, current, pid=os.getpid())

    def finished(self) -> None:
        self._tasks.finalize_if_active(self._task_id, TaskStatus.FINISHED)

    def failed(self, error: str) -> None:
        self._tasks.finalize_if_active(
            self._task_id, TaskStatus.FAILED, error=str(error)
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dataset_task_worker")
    parser.add_argument("--db", required=True, help="backend.db path")
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--dataset", required=True, help="dataset root")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--params", required=True, help="JSON launch payload")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    params = json.loads(args.params)

    # Local imports: the DB layer is cheap, manager/torch is not.
    from .clock import SystemClock
    from .dataset_tasks import SqliteDatasetTasks
    from .persistence.sqlite import SqliteDatabase

    database = SqliteDatabase(Path(args.db))
    database.initialize()  # idempotent; self-sufficient if racing boots
    tasks = SqliteDatasetTasks(database, SystemClock())
    reporter = _Reporter(tasks, args.task)

    try:
        if args.kind == "ingest_lora":
            from manager.builder import DataTaskRunner  # repo bridge: torch/xpu

            runner = DataTaskRunner(device="xpu")  # this project is B580-only
            runner.run_lora_ingestion_task(
                Path(args.dataset),
                Path(params["model"]),
                Path(params["image_dir"]),
                latent_size=int(params["latent_size"]),
                recursive=bool(params["recursive"]),
                resize_mode=str(params["resize_mode"]),
                neg_prompt=str(params["neg_prompt"]),
                model_type=str(params["model_type"]),
                seed=int(params["seed"]),
                max_aspect_ratio=float(params["max_aspect_ratio"]),
                task_id=None,  # legacy DB channel is gone in v2; reporter only
                reporter=reporter,
            )
        elif args.kind == "generate_teacher":
            # Flat params record -> DTO -> the same builders the start
            # use case validated with (both sides import application,
            # so stored payload and child cannot disagree silently).
            from ..application.dto import TeacherTaskParams
            from ..application.teacher_prompts import build_neg_cfg, build_pos_cfg

            keys = TeacherTaskParams.__dataclass_fields__
            teacher = TeacherTaskParams(**{k: params[k] for k in keys})
            pos_cfg = build_pos_cfg(teacher)
            neg_cfg = build_neg_cfg(teacher)

            from manager.builder import DataTaskRunner  # repo bridge: torch/xpu

            runner = DataTaskRunner(device="xpu")  # this project is B580-only
            runner.run_teacher_task(
                Path(args.dataset),
                Path(params["model"]),
                pos_cfg,
                neg_cfg=neg_cfg,
                n_conditions=teacher.n_conditions,
                n_samples_per_cond=teacher.n_samples_per_cond,
                steps_range=(teacher.steps_min, teacher.steps_max),
                cfg_range=(teacher.cfg_min, teacher.cfg_max),
                batch_size=teacher.batch_size,
                latent_size=teacher.latent_size,
                seed=teacher.seed,
                model_type=teacher.model_type,
                t_mode=teacher.t_mode,
                t_low=teacher.t_low,
                t_high=teacher.t_high,
                task_id=None,  # legacy DB channel is gone in v2; reporter only
                reporter=reporter,
            )
        else:
            # The use case refuses unknown kinds before spawning; this
            # is the backstop for a stale/foreign row.
            reporter.failed(f"unknown task kind {args.kind!r}")
            return 2
    except Exception as exc:  # noqa: BLE001 -- process-level backstop:
        # this is main() of a child process whose whole job is to report
        # whatever happened. Naming exception types here would mean
        # re-deciding, in another module, which failures are reportable --
        # and any gap between that list and the builder's becomes an
        # unhandled crash with no report at all.
        # builder usually reports first; CAS dedupes
        reporter.failed(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
