"""SendExecutionAction -- a page asks a node of a live run to do something ("Save now").

The server checks what it can see from here: the run is live, the node is in the run's graph, the node declares the action, and
something in the graph can act on it. It sends nothing but the node and the action's name; the event itself is built inside the run
from the node's declaration, so a page cannot ask for anything the node did not offer. The answer is only "queued": whether the
node handled it comes back as frames on the node's stream.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.value_objects import GraphStatus
from ..actions import action_states
from ..errors import ActionNotFoundError, ActionUnavailableError, GraphExecutionNotActiveError, GraphExecutionNotFoundError
from ..ports.command_channel import CommandChannel
from ..ports.graph_execution_repository import GraphExecutionRepository
from ..ports.node_catalog import NodeCatalog


@dataclass(frozen=True, slots=True)
class ActionQueued:
    command_id: str
    node: str
    action: str


class SendExecutionAction:
    def __init__(self, *, executions: GraphExecutionRepository, catalog: NodeCatalog, channel: CommandChannel) -> None:
        self._executions = executions
        self._catalog = catalog
        self._channel = channel

    def execute(self, execution_id: int, node: str, action: str) -> ActionQueued:
        execution = self._executions.get(execution_id)
        if execution is None:
            raise GraphExecutionNotFoundError(f"execution {execution_id} does not exist")
        if execution.status is not GraphStatus.RUNNING:
            raise GraphExecutionNotActiveError(
                f"execution {execution_id} is {execution.status.value}; actions need a running one",
                details={"status": execution.status.value},
            )
        graph = execution.graph
        spec = graph.node(node)
        by_class = {n["class_name"]: n for n in self._catalog.catalog()["nodes"]}
        info = by_class.get(spec.class_name) if spec is not None else None
        states = action_states(graph, node, info, by_class) if info is not None else []
        state = next((s for s in states if s["name"] == action), None)
        if state is None:
            raise ActionNotFoundError(f"node {node!r} of execution {execution_id} has no action {action!r}")
        if not state["available"]:
            raise ActionUnavailableError(state["reason"], details={"reason": state["reason"]})
        return ActionQueued(self._channel.submit(execution_id, node, action), node, action)
