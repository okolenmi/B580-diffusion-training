"""ExecutionLauncher port -- "run this graph in the background" as a dependency.

`GraphExecutionSupervisor` is the production implementation, but the two
use cases that start and stop an execution only need these two verbs.
Keeping them behind a port means the use cases no longer import a
concrete supervisor class (docs 08 S-01), and a test can assert on
"was it launched / cancelled" with a three-line recorder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.graph import GraphDefinition
from ...domain.value_objects import ExecutionId


class ExecutionLauncher(ABC):
    """Start and cooperatively cancel graph execution threads."""

    @abstractmethod
    def launch(self, execution_id: ExecutionId, graph: GraphDefinition) -> None:
        """Run ``graph`` in the background for ``execution_id``."""
        raise NotImplementedError

    @abstractmethod
    def cancel(self, execution_id: ExecutionId) -> None:
        """Ask the running execution to stop; no-op when not running here.

        Whether the row actually stops is decided by the stop use case's
        status compare-and-swap, not by this call.
        """
        raise NotImplementedError

    @abstractmethod
    def adopt(self, execution_id: ExecutionId) -> int | None:
        """Re-attach to a run still going after a restart; None if none.

        Only meaningful for a supervisor whose runs live in their own
        process, and the honest answer from one whose runs are threads is
        ``None`` -- those die with the server, so there is nothing to find.
        """
        raise NotImplementedError