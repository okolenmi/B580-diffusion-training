"""StartDatasetTask -- validate, persist, spawn one dataset child.

Check-then-act under a lock (one process, two concurrent starts must
not both pass the active-task check); the row is added *before* the
spawn so a crash between the two leaves something for startup
reconciliation to sweep. A spawn failure fails the row here and
re-raises as ``DatasetTaskLaunchError`` -- there is nothing to reap.

Two kinds, both following the same shape: validate everything first,
compute ``total`` honestly, then insert + spawn:

* ``ingest_lora`` -- VAE-encode an image directory. ``model`` is
  sandboxed against the checkpoints directory (client input is
  untrusted, same posture as the assets API); ``image_dir`` must be an
  existing absolute directory but is deliberately *not* sandboxed --
  source images legitimately live anywhere on the machine, as in the
  legacy API. The image count uses the legacy rule exactly so ``total``
  matches what the child will iterate.
* ``generate_teacher`` (M8e) -- sample new trajectories from a
  checkpoint. All options validate through
  ``application.teacher_prompts`` (modes, ranges, prompt content), so
  an impossible launch never becomes a row; ``total =
  n_conditions * n_samples_per_cond``, the legacy rule.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..dataset_task_sweeper import DatasetTaskSweeper
from ..dto import StartDatasetTaskCommand, TeacherTaskParams
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
    TaskKind,
    TaskStatus,
)
from ..teacher_prompts import MODEL_TYPES, teacher_payload

def _task_kind(raw: str) -> TaskKind:
    """Unknown kind is a 422 naming the vocabulary, not a raw ValueError.

    The enum decides what a kind *is*; this decides how an unknown one is
    reported (docs 08 S-24).
    """
    try:
        return TaskKind(raw)
    except ValueError:
        raise InvalidQueryError(
            f"unknown task kind {raw!r}; expected one of "
            f"{[k.value for k in TaskKind]}"
        ) from None


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_RESIZE_MODES: tuple[str, ...] = ("fit", "center_crop", "pad", "resize")


class StartDatasetTask:
    """The only place a dataset task is born (single-active per dataset)."""

    def __init__(
        self,
        *,
        library: DatasetLibrary,
        tasks: DatasetTasks,
        gateway: DatasetTaskGateway,
        checkpoints_dir: Path,
        sweeper: DatasetTaskSweeper | None = None,
    ) -> None:
        self._library = library
        self._tasks = tasks
        self._gateway = gateway
        self._checkpoints = checkpoints_dir
        # Optional because only the *liveness* judgement needs it, and
        # the composition root always passes it; a container without one
        # simply keeps a dead predecessor's row until startup.
        self._sweeper = sweeper
        self._lock = threading.Lock()

    def execute(self, command: StartDatasetTaskCommand) -> DatasetTask:
        with self._lock:
            root = self._library.root(command.dataset)  # exists + validated
            kind = _task_kind(command.kind)
            model_rel = self._validate_model(command.model)

            # Per-kind validation + honest total, before the active-task
            # check: a bad request stays a bad request even while
            # another task runs (422 beats 409, as before M8e).
            if kind is TaskKind.INGEST_LORA:
                image_dir = self._validate_image_dir(command.image_dir)
                total = self._count_images(image_dir, command.recursive)
                if total == 0:
                    raise InvalidQueryError(
                        f"no images ({', '.join(sorted(_IMAGE_EXTENSIONS))}) "
                        f"found in '{image_dir}'"
                    )
                params = self._ingest_params(command, model_rel)
            else:  # TaskKind.GENERATE_TEACHER
                teacher = self._require_teacher(command)
                total = teacher.n_conditions * teacher.n_samples_per_cond
                params = self._teacher_params(teacher, model_rel)

            # A predecessor whose child died must not answer 409 "a task
            # is already active" -- sweep the debris before asking
            # (docs 08 S-03; the rule itself lives in the sweeper).
            if self._sweeper is not None:
                self._sweeper.sweep()

            active = self._tasks.find_active(command.dataset)
            if active is not None:
                raise DatasetTaskActiveError(
                    f"dataset '{command.dataset}' already has task "
                    f"{active.id} {TaskStatus(active.status).value}",
                    details={
                        "task_id": active.id,
                        "status": TaskStatus(active.status).value,
                    },
                )

            task = self._tasks.add(
                dataset=command.dataset,
                kind=kind,
                total=total,
                params=params,
            )
            try:
                pid = self._gateway.spawn(
                    DatasetTaskLaunch(
                        task_id=task.id,
                        dataset_root=root,
                        kind=kind,
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

    # -- per-kind params --------------------------------------------------

    def _ingest_params(
        self, command: StartDatasetTaskCommand, model_rel: str
    ) -> dict:
        # Enum checks live here rather than in the pydantic schema so
        # unknown values leave as the envelope's invalid_query with an
        # "expected one of" list: the builder silently *defaults* an
        # unknown resize_mode and crashes deep on an unknown
        # model_type -- neither is honest feedback on its own.
        if command.resize_mode not in _RESIZE_MODES:
            raise InvalidQueryError(
                f"unknown resize_mode {command.resize_mode!r}; "
                f"expected one of {list(_RESIZE_MODES)}"
            )
        if command.model_type not in MODEL_TYPES:
            raise InvalidQueryError(
                f"unknown model_type {command.model_type!r}; "
                f"expected one of {list(MODEL_TYPES)}"
            )
        return {
            "image_dir": command.image_dir,
            "model": model_rel,
            "recursive": command.recursive,
            "resize_mode": command.resize_mode,
            "latent_size": command.latent_size,
            "neg_prompt": command.neg_prompt,
            "model_type": command.model_type,
            "seed": command.seed,
            "max_aspect_ratio": command.max_aspect_ratio,
        }

    @staticmethod
    def _require_teacher(
        command: StartDatasetTaskCommand,
    ) -> TeacherTaskParams:
        # The wire schema always builds this for the kind; the guard
        # keeps a hand-constructed command honest.
        if command.teacher is None:
            raise InvalidQueryError(
                f"task kind {command.kind!r} needs teacher parameters"
            )
        return command.teacher

    @staticmethod
    def _teacher_params(teacher: TeacherTaskParams, model_rel: str) -> dict:
        try:
            payload = teacher_payload(teacher)
        except ValueError as exc:
            raise InvalidQueryError(f"teacher task: {exc}") from exc
        return dict(payload, model=model_rel)

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
