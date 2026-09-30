"""StartDatasetTask -- validate, persist, spawn one ingestion child.

Check-then-act under a lock (one process, two concurrent starts must
not both pass the active-task check); the row is added *before* the
spawn so a crash between the two leaves something for startup
reconciliation to sweep. A spawn failure fails the row here and
re-raises as ``DatasetTaskLaunchError`` -- there is nothing to reap.

``model`` is sandboxed against the checkpoints directory (client input
is untrusted, same posture as the assets API); ``image_dir`` must be
an existing absolute directory but is deliberately *not* sandboxed --
source images legitimately live anywhere on the machine, as in the
legacy API. The image count uses the legacy rule exactly so ``total``
matches what the child will iterate.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..dto import StartDatasetTaskCommand
from ..errors import (
    DatasetTaskActiveError,
    DatasetTaskLaunchError,
    InvalidQueryError,
)
from ..ports.dataset_library import DatasetLibrary
from ..ports.dataset_task_gateway import DatasetTaskGateway, DatasetTaskLaunch
from ..ports.dataset_tasks import (
    DatasetTask,
    DatasetTasks,
    TASK_KINDS,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class StartDatasetTask:
    """The only place a dataset task is born (single-active per dataset)."""

    def __init__(
        self, *,
        library: DatasetLibrary,
        tasks: DatasetTasks,
        gateway: DatasetTaskGateway,
        checkpoints_dir: Path,
    ) -> None:
        self._library = library
        self._tasks = tasks
        self._gateway = gateway
        self._checkpoints = checkpoints_dir
        self._lock = threading.Lock()

    def execute(self, command: StartDatasetTaskCommand) -> DatasetTask:
        with self._lock:
            root = self._library.root(command.dataset)  # exists + validated
            if command.kind not in TASK_KINDS:
                raise InvalidQueryError(
                    f"unknown task kind {command.kind!r}; "
                    f"expected one of {list(TASK_KINDS)}"
                )
            image_dir = self._validate_image_dir(command.image_dir)
            model_rel = self._validate_model(command.model)

            total = self._count_images(image_dir, command.recursive)
            if total == 0:
                raise InvalidQueryError(
                    f"no images ({', '.join(sorted(_IMAGE_EXTENSIONS))}) "
                    f"found in '{image_dir}'"
                )

            active = self._tasks.find_active(command.dataset)
            if active is not None:
                raise DatasetTaskActiveError(
                    f"dataset '{command.dataset}' already has task "
                    f"{active.id} {active.status}",
                    details={"task_id": active.id, "status": active.status},
                )

            params: dict = {
                "image_dir": str(image_dir),
                "model": model_rel,
                "recursive": command.recursive,
                "resize_mode": command.resize_mode,
                "latent_size": command.latent_size,
                "neg_prompt": command.neg_prompt,
                "model_type": command.model_type,
                "seed": command.seed,
                "max_aspect_ratio": command.max_aspect_ratio,
            }
            task = self._tasks.add(
                dataset=command.dataset,
                kind=command.kind,
                total=total,
                params=params,
            )
            try:
                pid = self._gateway.spawn(
                    DatasetTaskLaunch(
                        task_id=task.id,
                        dataset_root=root,
                        kind=command.kind,
                        params=dict(params, model=str(self._model_path(command.model))),
                    )
                )
            except DatasetTaskLaunchError as exc:
                self._tasks.fail_if_active(task.id, str(exc))
                raise
            # Record the pid immediately: stop() must be able to kill the
            # child during the seconds it spends importing torch before
            # its first progress tick, and reconciliation must recognise
            # it after a crash in this window. The child's own first
            # progress write repeats the same pid (self-identifying).
            self._tasks.update_progress(task.id, 0, pid)
            return self._tasks.get(task.id) or task

    # -- validation helpers ---------------------------------------------

    def _validate_image_dir(self, raw: str) -> Path:
        if not raw:
            raise InvalidQueryError("image_dir is required")
        path = Path(raw)
        if not path.is_absolute():
            raise InvalidQueryError(
                f"image_dir must be an absolute path, got {raw!r}"
            )
        if not path.is_dir():
            raise InvalidQueryError(f"image_dir is not a directory: {raw!r}")
        return path

    def _validate_model(self, raw: str) -> str:
        if not raw:
            raise InvalidQueryError("model (checkpoint path) is required")
        resolved = self._model_path(raw)
        if not resolved.is_file():
            raise InvalidQueryError(
                f"no such checkpoint: {raw!r} (looked in "
                f"{self._checkpoints})"
            )
        return raw

    def _model_path(self, raw: str) -> Path:
        """Untrusted relative checkpoint path -> path inside the dir."""
        base = self._checkpoints.resolve()
        candidate = Path(raw)
        if candidate.is_absolute():
            raise InvalidQueryError(f"model must be a relative path: {raw!r}")
        resolved = (base / candidate).resolve()
        if resolved != base and not resolved.is_relative_to(base):
            raise InvalidQueryError(
                f"model path escapes the checkpoints directory: {raw!r}"
            )
        return resolved

    @staticmethod
    def _count_images(image_dir: Path, recursive: bool) -> int:
        # Legacy rule (server/routes_datasets.py): same glob + extensions.
        pattern = "**/*" if recursive else "*"
        return sum(
            1
            for p in image_dir.glob(pattern)
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
