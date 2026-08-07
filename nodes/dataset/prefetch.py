"""PrefetchingBatchSource: decorates any TrainingBatchSource, overlapping
the *next* batch's host-side preparation with the *current* step's device
compute. docs/training_pipeline_design.md section 2.5.

Same pattern nodes/dataset/renoise.py's RenoiseBatchSource already
establishes for this domain (wrap, don't reimplement iteration) -- a
bounded background queue.Queue, one worker thread per __iter__() call
(not one for this object's whole lifetime -- see below for why that
matters), reading self._inner and handing batches to the training thread.

Explicitly NOT a VRAM optimization -- it trades a small, bounded amount
of extra host memory (maxsize=depth batches sitting in the queue at once)
for reduced wall-clock stall on the data side, a different axis than most
of this project's VRAM-focused work. Worth wiring in once
SupervisedLoRATrainerNode's own profile=True output (data_wait_ms, now
fetch_batch_ms -- see nodes/train/step_pipeline.py) actually shows data
loading stalling the step, not by default.

Not implemented here: pinning the host-side buffers (page-locked memory,
which makes the eventual host->device copy faster still). The design
doc's own section 2.5 mentions it only as a parenthetical, not a
requirement, and it adds a real platform-specific wrinkle (pinning
support/benefit isn't identical across CUDA and XPU) worth its own
follow-up once this is in real use, not bundled into first landing this.

**Per-__iter__() worker thread, not one shared for this object's whole
lifetime -- why:** nodes/train/step_pipeline.py's FetchBatchPhase calls
iter(batches) once at construction, then again on StopIteration (wrapping
to a new epoch once the dataset is exhausted -- see that phase's own
docstring). A TrainingBatchSource's __iter__() is expected to be
restartable, a fresh pass each time it's called (RenoiseBatchSource
already works this way: `for batch in self._inner: yield ...` inside
__iter__() itself calls self._inner's __iter__() fresh every time). This
class matches that: each __iter__() call starts a fresh worker thread
against a fresh iter(self._inner), and the worker (and its queue) are
torn down when that particular pass ends -- normally (the dataset
exhausts) or early (the caller stops consuming, e.g. FetchBatchPhase
never actually does this today, but a future caller might).

See docs/training_pipeline_design.md section 5.6 for the concurrency
contract this worker thread has to honor: the queue is the *only* thing
crossing the thread boundary, exactly as documented there.
"""

from __future__ import annotations

import queue
import threading
from typing import ClassVar, Iterator

from ..core import Port
from .handle import TrainingBatchSource
from .node import DataSourceNode

_SENTINEL = object()


class PrefetchingBatchSource(TrainingBatchSource):

    def __init__(self, inner: TrainingBatchSource, depth: int = 2):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self._inner = inner
        self._depth = depth  # bounded on purpose -- see module docstring

    def __iter__(self) -> Iterator[dict]:
        q: queue.Queue = queue.Queue(maxsize=self._depth)
        stop_event = threading.Event()

        def _worker():
            try:
                for batch in self._inner:
                    if stop_event.is_set():
                        return
                    q.put(batch)
            except Exception as e:
                # Queued, not raised here -- this is a background thread;
                # an uncaught exception here would just print to stderr
                # via the default threading excepthook and silently kill
                # the worker, leaving the consumer to see nothing but a
                # premature sentinel below, indistinguishable from normal
                # exhaustion. Handing it to the consumer thread to
                # re-raise (below) is what actually surfaces it.
                q.put(e)
            finally:
                # Always signal completion, even after an exception above
                # -- an unbounded consumer wait on a worker that's already
                # gone is a worse failure mode than a stray, unclaimed
                # sentinel left in an abandoned queue ever is.
                q.put(_SENTINEL)

        thread = threading.Thread(target=_worker, daemon=True, name="prefetch-batch-source")
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Reached on normal completion (harmless -- stop_event is
            # already moot by then) and on early abandonment (a caller
            # closing this generator, or a for-loop `break`) -- the case
            # that actually matters. Draining unblocks a worker possibly
            # stuck inside q.put() waiting for space that will now never
            # come from the consumer side; without this, that thread
            # would sit blocked forever (harmless to the process, since
            # it's a daemon thread, but it'd leak for the rest of this
            # run).
            stop_event.set()
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def __len__(self) -> int:
        return len(self._inner)

    def invalidate(self) -> None:
        self._inner.invalidate()


class PrefetchingBatchSourceNode(DataSourceNode):
    """Wraps a TrainingBatchSource so the next batch's host-side prep
    overlaps the current step's device compute, via a bounded background
    queue. See this module's docstring for why -- wire this between your
    dataset source and the trainer once profile=True's fetch_batch_ms
    actually shows data loading stalling steps, not as a default."""

    INPUTS: ClassVar[dict[str, Port]] = {
        "batches": Port(name="batches", type=TrainingBatchSource, required=True),
        "depth": Port(
            name="depth", type=int, required=False, default=2,
            doc="Max batches sitting in the background queue at once (bounded on "
                "purpose -- an unbounded queue would let the worker race arbitrarily "
                "far ahead, trading an unbounded amount of host memory for a benefit "
                "that saturates after a batch or two anyway). 2 covers the common case "
                "(one batch mid-transfer/prep, one ready to go) without holding much "
                "extra host memory; raise it only if profiling shows the worker itself "
                "is slower than one step and needs more headroom to stay ahead.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainingBatchSource]:
        self.validate_inputs(inputs)
        result = {"batches": PrefetchingBatchSource(
            inputs["batches"], depth=inputs.get("depth", self.INPUTS["depth"].default))}
        self.validate_outputs(result)
        return result
