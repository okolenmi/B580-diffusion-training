"""OptimizerNode: the intermediate "these are all optimizers" layer.

This is the concrete answer to "there's a big similar part that can be an
intermediate node" -- every optimizer-family node shares exactly one
output shape (a single OptimizerHandle) and a couple of near-universal
inputs (params, lr), declared *once* here rather than repeated by hand in
every concrete subclass.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from ..core import Node, Port
from .handle import OptimizerHandle


class OptimizerNode(Node):
    """Shared contract for every node whose job is "given params and
    hyperparameters, construct a ready-to-use optimizer".

    OUTPUTS is fixed here, not re-declared per subclass: every optimizer
    node produces exactly one port, named "optimizer", typed as
    OptimizerHandle -- concrete subclasses inherit this unchanged.

    COMMON_INPUTS holds the couple of ports essentially every optimizer
    needs (the parameters to optimize, and a learning rate). Concrete
    subclasses build their own INPUTS as {**COMMON_INPUTS, <their own
    extra hyperparameter ports>}, so the shared ports are declared once
    and each subclass only has to write down what's actually different
    about it -- exactly the reduction in duplication the intermediate
    layer is supposed to provide, applied to *interface declarations*
    rather than *lifecycle method bodies* (contrast with the earlier,
    reverted attempt to share lifecycle *implementations* directly on
    core/optimizers.py's classes -- see docs/nodes_package_design.md for
    why that approach was abandoned in favor of this one).
    """

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "optimizer": Port(
            name="optimizer", type=OptimizerHandle, required=True,
            doc="A constructed, ready-to-use optimizer.",
        ),
    }

    COMMON_INPUTS: ClassVar[dict[str, Port]] = {
        "params": Port(
            name="params", type=Any, required=True,
            doc="Trainable parameters (or param groups with per-group lr) to optimize.",
        ),
        "lr": Port(
            name="lr", type=float, required=False, default=1e-5,
            doc="Learning rate.",
        ),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        ...
