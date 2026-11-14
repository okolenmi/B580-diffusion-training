"""TrainingProgressMonitorNode: live step/loss/lr feed for a training run.

Wire it into a TrainerNode's `monitor` input, spawn it in the graph, hit
"Look inside" -- the dashboard page (server/static/monitor_dashboard.html)
subscribes to this node's monitor_id over SSE and renders it with the
same chart.js the main dashboard tab uses.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .handle import LiveMonitorHandle, MonitorHandle
from .node import MonitorNode


class TrainingProgressMonitorNode(MonitorNode):

    INPUTS: ClassVar[dict[str, Port]] = {**MonitorNode.COMMON_INPUTS}

    def build(self, **inputs) -> dict[str, MonitorHandle]:
        self.validate_inputs(inputs)
        monitor_id = inputs["monitor_id"]
        if self.context.monitor_bus is not None:
            # Every build() is a *new* run starting to report to this monitor_id --
            # see MonitorBus.clear()'s own docstring for why this is the right
            # trigger and what it fixes (a real user report: re-running against the
            # same monitor_id overlaid the new run's line on top of the old one).
            self.context.monitor_bus.clear(monitor_id)
        result = {"monitor": LiveMonitorHandle(monitor_id, self.context.monitor_bus)}
        self.validate_outputs(result)
        return result
