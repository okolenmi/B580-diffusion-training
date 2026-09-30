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
        self._tasks.finish_if_active(self._task_id)

    def failed(self, error: str) -> None:
        self._tasks.fail_if_active(self._task_id, str(error))


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

    if args.kind != "ingest_lora":
        reporter.failed(f"unknown task kind {args.kind!r}")
        return 2

    try:
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
    except Exception as exc:  # builder usually reports first; CAS dedupes
        reporter.failed(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
