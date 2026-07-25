"""Node-graph routes -- editor tab, isolated from the production
config/training path (see docs/node_architecture_refactor_plan.md).

/optimizers and /registry are pure introspection: read declared metadata,
build nothing, run nothing. /run is different in kind -- it actually
constructs and executes the submitted graph (see graph_executor.py's
docstring for why that's a deliberate, not accidental, distinction).
Neither touches config.py, config_model.py, or the training launch path.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import graph_executor
from .nodegraph_introspect import introspect_optimizer_nodes, introspect_registry, node_info_to_dict

router = APIRouter(prefix="/nodegraph")


@router.get("/optimizers")
async def list_optimizer_nodes():
    """Reads declared contracts directly off nodes/optimizer/'s real Node
    classes -- see nodegraph_introspect.py's introspect_optimizer_nodes()
    and docs/nodes_package_design.md. This replaced an earlier version that
    guessed ports from core.optimizers.py's constructor signatures; that
    approach is still available (introspect_legacy_class(), same module)
    for any future domain not yet migrated into nodes/.
    """
    try:
        infos = introspect_optimizer_nodes()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not introspect nodes/optimizer/ ({type(e).__name__}: {e}).",
        )
    return {"nodes": [node_info_to_dict(i) for i in infos]}


@router.get("/registry")
async def list_registry():
    """Every node across every migrated domain, grouped for the palette.
    See server.nodegraph_registry for the underlying class list."""
    try:
        groups = introspect_registry()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not introspect the node registry ({type(e).__name__}: {e}).",
        )
    return {domain: [node_info_to_dict(i) for i in infos] for domain, infos in groups.items()}


class GraphNodeIn(BaseModel):
    id: str
    class_name: str
    params: dict = {}


class GraphEdgeIn(BaseModel):
    from_node: str
    from_port: str
    to_node: str
    to_port: str


class GraphRunRequest(BaseModel):
    nodes: list[GraphNodeIn]
    edges: list[GraphEdgeIn]


@router.post("/run")
def run_graph(request: GraphRunRequest):
    """Executes the submitted graph in topological order. Declared as a
    plain `def`, not `async def`, so FastAPI runs it in its worker thread
    pool instead of the event loop -- a SupervisedLoRATrainerNode run can
    take a long time, and blocking the event loop for that long would
    freeze every other request (SSE progress for real training runs
    included) until it finished.

    No streaming progress yet -- the response only arrives once the whole
    graph finishes or a node fails. TrainerNode's `on_step` callback
    exists specifically to make a progress-streaming version possible
    later; wiring it to SSE is a real next step, not done here.
    """
    nodes = [graph_executor.NodeSpec(id=n.id, class_name=n.class_name, params=n.params)
             for n in request.nodes]
    edges = [graph_executor.EdgeSpec(from_node=e.from_node, from_port=e.from_port,
                                      to_node=e.to_node, to_port=e.to_port)
             for e in request.edges]
    try:
        executor = graph_executor.GraphExecutor(nodes, edges)
        results = executor.run()
    except graph_executor.GraphError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"results": [
        {"node_id": r.node_id, "ok": r.ok, "outputs": r.outputs, "error": r.error}
        for r in results
    ]}
