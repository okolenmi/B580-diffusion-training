"""DataSourceNode: shared contract for nodes that produce a TrainingBatchSource."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from ..core import Node, Port
from .handle import TrainingBatchSource


class DataSourceNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "batches": Port(name="batches", type=TrainingBatchSource, required=True,
                         doc="Iterable of training batches, each a dict of tensors."),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, TrainingBatchSource]:
        ...
