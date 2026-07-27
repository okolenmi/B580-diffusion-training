"""TrainerNode: the step loop. Consumes a TrainableModel + TrainingBatchSource
+ OptimizerHandle (+ TextEncoder, LRSchedule, LossWeighting) and runs the
requested number of optimizer updates."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, ClassVar

from ..core import Node, Port
from ..dataset.handle import TrainingBatchSource
from ..model.handle import TrainableModel
from ..model.text_encoder import TextEncoder
from ..monitor.handle import MonitorHandle
from ..optimizer.handle import OptimizerHandle
from .loss import LossWeighting
from .schedule import LRSchedule


class TrainerNode(Node):

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
        "monitor": Port(name="monitor", type=MonitorHandle, required=False, default=None,
                         doc="Optional -- wire a MonitorNode here (e.g. "
                             "TrainingProgressMonitorNode) for a live step/loss/lr feed, "
                             "watchable from that node's 'Look inside' dashboard while "
                             "this is still running."),
        "on_step": Port(name="on_step", type=Callable, required=False, default=None,
                         doc="Optional callback(step, loss) -- for programmatic use "
                             "(tests, scripts); `monitor` above is the graph-editor path."),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, TrainableModel]:
        ...
