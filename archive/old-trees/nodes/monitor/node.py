"""MonitorNode: shared contract for nodes that expose a live data channel."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from ..core import Node, Port
from .handle import MonitorHandle


class MonitorNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "monitor": Port(name="monitor", type=MonitorHandle, required=True),
    }

    COMMON_INPUTS: ClassVar[dict[str, Port]] = {
        "monitor_id": Port(
            name="monitor_id", type=str, required=True,
            doc="Generated automatically when this node is spawned in the graph editor "
                "(editable if you want a stable ID to reopen the same dashboard later). "
                "Identifies this monitor's data stream independently of any particular "
                "graph run -- a dashboard opened on this ID shows data the moment any run "
                "reports to it, not just runs started from this tab.",
        ),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, MonitorHandle]:
        ...
