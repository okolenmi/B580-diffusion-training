"""TrainerNode: the step loop. Consumes a TrainableModel + TrainingBatchSource
+ OptimizerHandle (+ TextEncoder, LRSchedule, LossWeighting) and runs the
requested number of optimizer updates."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, ClassVar

from ..core import Completion, Node, Port
from ..dataset.handle import TrainingBatchSource
from ..model.handle import TrainableModel
from ..model.text_encoder import TextEncoder
from ..monitor.handle import MonitorHandle
from ..ui import DisplayModule, Series
from ..optimizer.handle import OptimizerHandle
from .bucket_balance import BucketBalance
from .conditioning_blend import ConditioningBlend
from .loss import LossWeighting
from .schedule import LRSchedule


class TrainerNode(Node):

    # What a page can show of a running trainer. The keys are the ones the
    # step loops already put in their per-step report (managed.py's
    # MonitoringPhase; supervised.py reports a subset -- a key a loop does
    # not send is a gap in the chart, never a zero). This is the list the
    # dashboard used to keep privately in monitor.js (LOSS_SERIES,
    # VRAM_SERIES, the *_ms discovery); it now lives next to the code that
    # produces the numbers, so adding a metric is one edit, here.
    MODULES: ClassVar[tuple[DisplayModule, ...]] = (
        DisplayModule(
            name="progress", kind="progress", title="Progress",
            value_key="step", total_key="total_steps", size=(12, 2),
            doc="Optimizer step out of the run's total.",
        ),
        DisplayModule(
            name="loss", kind="chart", title="Loss", scale="symlog", size=(12, 6),
            series=(
                Series("loss", "loss", "#6c8cff"),
                Series("loss_t_low", "t low", "#4caf50"),
                Series("loss_t_mid", "t mid", "#ffb300"),
                Series("loss_t_high", "t high", "#ff5252"),
            ),
            doc="Training loss, plus the loss over the low / mid / high third "
                "of the timestep range (a bucket no sample hit is a gap).",
        ),
        DisplayModule(
            name="lr", kind="chart", title="Learning rate", size=(6, 4),
            series=(Series("lr", "lr", "#ab47bc"),),
        ),
        DisplayModule(
            name="grad_norm", kind="chart", title="Gradient norm", size=(6, 4),
            series=(Series("grad_norm", "grad norm", "#26c6da"),),
            doc="Only present while gradient clipping is on.",
        ),
        DisplayModule(
            name="vram", kind="chart", title="VRAM", size=(12, 6),
            reference_key="vram_budget_mb",
            series=(
                Series("vram_reserved_mb", "reserved", "#6c8cff", "MB"),
                Series("vram_peak_reserved_mb", "peak reserved", "#ff9800", "MB"),
                Series("vram_allocated_mb", "allocated", "#4caf50", "MB"),
                Series("resident_model_mb", "model", "#26c6da", "MB"),
                Series("resident_optimizer_mb", "optimizer", "#ab47bc", "MB"),
                Series("resident_text_encoder_mb", "text encoder", "#8d6e63", "MB"),
            ),
            doc="Allocator state and what each resident takes; the dashed "
                "line is the budget the numbers are held under.",
        ),
        DisplayModule(
            name="timing", kind="chart", title="Step timing", size=(12, 5),
            pattern="*_ms", exclude=("step_total_ms",),
            series=(Series("step_total_ms", "step total", "#cfd8dc", "ms"),),
            doc="Per-phase milliseconds (present when TRAIN_STEP_TIMING=1).",
        ),
        # The readouts the old Monitor page had beside its chart: the newest
        # loss, the loss per timestep third as bars, and the VRAM high-water
        # mark. Appended, not inserted: a saved page refers to a module by
        # (node, name), and the order of this tuple is not part of that.
        DisplayModule(
            name="loss_now", kind="scalar", title="Loss (latest)", size=(3, 2),
            series=(Series("loss", "loss", "#6c8cff"),),
            doc="The most recent step's loss.",
        ),
        DisplayModule(
            name="loss_by_t", kind="bars", title="Loss by timestep", size=(6, 3),
            series=(
                Series("loss_t_low", "t low", "#4caf50"),
                Series("loss_t_mid", "t mid", "#ffb300"),
                Series("loss_t_high", "t high", "#ff5252"),
            ),
            doc="Latest loss over the low / mid / high third of the timestep "
                "range (a third no sample hit is absent, not zero).",
        ),
        DisplayModule(
            name="vram_peak", kind="scalar", title="VRAM peak", size=(3, 2),
            series=(Series("vram_peak_reserved_mb", "peak reserved", "#ff9800", "MB"),),
            doc="Highest reserved device memory the run has reached.",
        ),
    )

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(name="model", type=TrainableModel, required=True,
                       doc="The same model instance passed in, now trained."),
    }

    COMMON_INPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(name="model", type=TrainableModel, required=True),
        "batches": Port(name="batches", type=TrainingBatchSource, required=True),
        "optimizer": Port(name="optimizer", type=OptimizerHandle, required=True),
        "text_encoder": Port(name="text_encoder", type=TextEncoder, required=True),
        "lr_schedule": Port(name="lr_schedule", type=LRSchedule, required=True),
        "steps": Port(name="steps", type=int, required=True),
        "loss_weighting": Port(name="loss_weighting", type=LossWeighting, required=False,
                                default=None, doc="None = uniform weighting."),
        "bucket_balance": Port(
            name="bucket_balance", type=BucketBalance, required=False, default=None,
            doc="Optional -- wire a Bucket Balance node's output here to rebalance "
                "per-t-bucket losses during training (nodes/train/bucket_balance.py: "
                "normalize / speed / dro modes; mode='off' wires the tracking only and "
                "changes no arithmetic). None = today's behavior, bit-identical. The "
                "same balance object can also be wired into the dataset source node "
                "for t_mode='adaptive' sampling -- either side alone, or both."),
        "conditioning_blend": Port(
            name="conditioning_blend", type=ConditioningBlend, required=False, default=None,
            doc="Optional -- wire a Conditioning Blend node's output here to train on "
                "ctx = p*ctx(caption) + (1-p)*ctx(\"\") with p drawn per sample per step "
                "(nodes/train/conditioning_blend.py). None, or the node at its default "
                "keep range 1.0..1.0, = captions exactly as stored, bit-identical."),
        "monitor": Port(name="monitor", type=MonitorHandle, required=False, default=None,
                         doc="Optional -- wire a MonitorNode here (e.g. "
                             "TrainingProgressMonitorNode) for a live step/loss/lr feed, "
                             "watchable from that node's 'Look inside' dashboard while "
                             "this is still running."),
        "on_step": Port(name="on_step", type=Callable, required=False, default=None,
                         doc="Optional callback(step, loss) -- for programmatic use "
                             "(tests, scripts); `monitor` above is the graph-editor path."),
    }

    def _report_stopped(self, step: int, total_steps: int) -> None:
        """A stop between steps: say so to whatever consumes the model.

        The model returned after a stop is valid -- trained for ``step`` of
        ``total_steps`` optimizer steps -- and this is how that fact reaches
        the nodes after it (nodes/core.py Completion): a saver can then
        write it out, named for the step it reached, instead of the stop
        discarding the work. Both trainer routes call this at the same
        place, so the wording and the numbers cannot drift apart.
        """
        self.report_completion(Completion.unfinished_at(
            step=step, total=total_steps, reason="stopped"))

    @abstractmethod
    def build(self, **inputs) -> dict[str, TrainableModel]:
        ...
