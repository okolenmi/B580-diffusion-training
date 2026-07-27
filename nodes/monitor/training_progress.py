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
        result = {"monitor": LiveMonitorHandle(inputs["monitor_id"], self.context.monitor_bus)}
        self.validate_outputs(result)
        return result
