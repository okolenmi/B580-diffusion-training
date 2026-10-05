"""StartDatasetTask -- validate, persist, spawn one dataset child.

Check-then-act under a lock (one process, two concurrent starts must
not both pass the active-task check); the admission claim and the row
are both taken inside that lock, the row *before* the spawn, so a crash
between any of the steps leaves something for startup reconciliation to
sweep. A spawn failure fails the row here, hands the claim back, and
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
from collections.abc import Callable
from pathlib import Path

from ..dataset_task_sweeper import DatasetTaskSweeper
from ..dto import StartDatasetTaskCommand, TeacherTaskParams
from ..errors import (
    DatasetTaskActiveError,
    DatasetTaskLaunchError,
    InvalidQueryError,
)
from ..ports.dataset_library import DatasetLibrary
from ..memory_admission import LedgerSource, admit, pending_owner, task_owner
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

#: Stated device demands per task kind, in allocator MB. Deliberately
#: empty: the spec calls for "stated per task type with a *measured*
#: default (add to the hardware protocol)", and there is no measurement
#: of either kind on this machine yet. A kind with no number here is
#: **unknown**, and unknown is never a zero claim (task rule 2): it goes
#: down the exploratory-exclusive path -- admitted only when nothing
#: else holds the card, refused with a breakdown otherwise. The numbers
#: land with the hardware protocol (MEM-08); one map edit wires them in.
TASK_DEMAND_MB: dict[TaskKind, float] = {}


class StartDatasetTask:
    """The only place a dataset task is born (single-active per dataset)."""

    def __init__(
        self,
        *,
        library: DatasetLibrary,
        tasks: DatasetTasks,
        gateway: DatasetTaskGateway,
        checkpoints_dir: Callable[[], Path],
        memory_ledger: LedgerSource,
        sweeper: DatasetTaskSweeper | None = None,
    ) -> None:
        self._library = library
        self._tasks = tasks
        self._gateway = gateway
        # A callable, not a Path, and deliberately. The composition root
        # used to pass ``layout.checkpoints_dir``, which is a property read
        # once at wiring time -- so a ``checkpoints_dir`` changed in Settings
        # was reported by the settings API and not used here until the
        # server restarted. Worse, the refusal named the directory it was
        # really looking in, so the message pointed the user at the setting
        # they had just changed.
        #
        # Resolving per use costs a dict lookup and keeps the setting live
        # everywhere it is read. A callable rather than the layout itself
        # because ``application`` does not import ``infrastructure``.
        self._checkpoints_dir = checkpoints_dir
        # Required, never defaulted: every container that starts tasks
        # must say where admission lives. The provider may answer None
        # (device total unknown) -- then every start is refused
        # explicitly rather than admitted unchecked.
        self._memory_ledger = memory_ledger
        # Optional because only the *liveness* judgement needs it, and
        # the composition root always passes it; a container without one
        # simply keeps a dead predecessor's row until startup.
        self._sweeper = sweeper
        self._lock = threading.Lock()

    def _checkpoints(self) -> Path:
        return self._checkpoints_dir()

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

            # Admission, still under the start lock and still before the
            # row: a refusal writes nothing and holds nothing. A kind
            # with no measured default has an unknown demand, which the
            # ledger treats as an exploratory exclusive claim (see
            # TASK_DEMAND_MB) -- a dataset task blocks a graph that does
            # not fit with it, and vice versa, because they share the
            # one ledger.
            ledger = self._memory_ledger()
            stated = TASK_DEMAND_MB.get(kind)
            exploratory = stated is None
            if ledger is None:
                # admit() refuses with device-total-unknown before it
                # looks at the demand (rule 2).
                device_demand = 0.0
            elif stated is None:
                device_demand = ledger.capacity_mb
            else:
                # Stated is allocator MB; a grant is device MB (rule 4).
                device_demand = stated + ledger.process_overhead_mb
            provisional = pending_owner("task")
            grant = admit(
                ledger,
                provisional,
                device_demand,
                exploratory=exploratory,
                what=f"a '{kind.value}' task",
                # MEM-03H-03: while admit() finds no ledger, its rows
                # say who holds the device -- the refusal names them.
                source=self._memory_ledger,
            )

            task: DatasetTask | None = None
            try:
                task = self._tasks.add(
                    dataset=command.dataset,
                    kind=kind,
                    total=total,
                    params=params,
                    reserved_mb=grant.mb,
                )
                if ledger is not None:
                    ledger.rename(provisional, task_owner(task.id))
                try:
                    pid = self._gateway.spawn(
                        DatasetTaskLaunch(
                            task_id=task.id,
                            dataset_root=root,
                            kind=kind,
                            params=dict(
                                params, model=str(self._model_path(command.model))
                            ),
                        )
                    )
                except DatasetTaskLaunchError as exc:
                    self._tasks.finalize_if_active(
                        task.id, TaskStatus.FAILED, error=str(exc)
                    )
                    raise
            except BaseException:
                # Failed between the claim and the child (or at the
                # spawn itself): nothing stays held. Both owner forms
                # go, because release is idempotent and the rename may
                # or may not have happened. A hard crash cannot run
                # this -- there, the row's claim is rebuilt at startup
                # and released by the sweeper when the row turns out
                # to have no live child.
                if ledger is not None:
                    if task is not None:
                        ledger.release(task_owner(task.id))
                    ledger.release(provisional)
                raise
            # Record the pid immediately: stop() must be able to kill the
            # child during the seconds it spends importing torch before
            # its first progress tick, and reconciliation must recognise
            # it after a crash in this window. The child's own first
            # progress write repeats the same pid (self-identifying).
            # Deliberately outside the claim-guarded try: once the child
            # exists, its claim belongs to the child, and this write
            # failing must not hand the capacity back while it runs.
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
                f"{self._checkpoints()})"
            )
        return raw

    def _model_path(self, raw: str) -> Path:
        """Untrusted relative checkpoint path -> path inside the dir."""
        base = self._checkpoints().resolve()
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
