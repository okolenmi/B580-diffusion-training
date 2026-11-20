"""What a page may ask of a node while a graph runs: the node's declared actions, and whether each could do anything in this graph.

An action's effect on other nodes goes through ordinary events, so "Save now" does nothing unless something in the graph listens for
the checkpoint it produces. ``needs`` on the declaration names that event type; this module checks the graph for a listener (a node
that hears it by scope, or one wired to the action's node) and gives the reason when there is none.
"""

from __future__ import annotations

from ..domain.graph import GraphDefinition


def action_states(graph: GraphDefinition, uid: str, info: dict, by_class: dict[str, dict]) -> list[dict]:
    """The actions of node ``uid`` (catalog entry ``info``), each with ``available`` and, when not, ``reason``."""
    out = []
    for action in info.get("actions", ()):
        needs = action.get("needs")
        listener = needs is None or _has_listener(graph, uid, needs, by_class)
        out.append({
            **action, "available": listener,
            "reason": "" if listener else f"nothing in this graph listens for it: add {action.get('needs_label') or 'a node that does'}",
        })
    return out


def _has_listener(graph: GraphDefinition, uid: str, payload: str, by_class: dict[str, dict]) -> bool:
    wired = {e.to_node for e in graph.doc.event_edges if e.from_node == uid}
    for spec in graph.nodes:
        if spec.uid == uid:
            continue
        for event in (by_class.get(spec.class_name) or {}).get("events_in", ()):
            if payload in event.get("payload_mro", ()) and (event.get("scoped") or spec.uid in wired):
                return True
    return False
