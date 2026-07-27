"""Node-graph routes -- editor tab, isolated from the production
config/training path (see docs/node_architecture_refactor_plan.md).

/optimizers and /registry are pure introspection: read declared metadata,
build nothing, run nothing. /run is different in kind -- it actually
constructs and executes the submitted graph (see graph_executor.py's
docstring for why that's a deliberate, not accidental, distinction).
Neither touches config.py, config_model.py, or the training launch path.
"""

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from nodes.core import ExecutionContext

from . import asset_paths, graph_executor
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
    result = {}
    for domain, infos in groups.items():
        entries = []
        for info in infos:
            entry = node_info_to_dict(info)
            entry["domain"] = domain  # same value as the grouping key, but on the node
            # itself too -- lets the frontend ask "is this a monitor-family node" without
            # re-deriving domain_of()'s module-path logic client-side in JS.
            entries.append(entry)
        result[domain] = entries
    return result


@router.get("/assets/{kind}")
async def list_assets(kind: str):
    """Options for a path_kind picker widget -- see server/asset_paths.py.
    kind: 'checkpoint', 'lora', or 'dataset'."""
    try:
        return asset_paths.list_options(kind)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/assets/{kind}/upload")
async def upload_asset(kind: str, file: UploadFile = File(...), relative_path: str | None = Form(None)):
    """Saves an uploaded file into the configured directory for `kind` --
    what makes the picker usable from a browser on a different machine
    than the server: there's no local filesystem to browse, so the file
    has to come over HTTP. relative_path lets the "Save As" dialog upload
    into a chosen subfolder; if omitted, falls back to the file's own
    name at the top level."""
    try:
        content = await file.read()
        target = relative_path or file.filename
        saved_path = asset_paths.save_upload(kind, target, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"kind": kind, "relative_path": target, "saved_path": saved_path}


@router.get("/assets/{kind}/browse")
async def browse_assets(kind: str, path: str = ""):
    """Immediate children of `path` (folders + .safetensors files) for the
    "Save As" dialog -- see server/asset_paths.py.browse(). Root is path=""."""
    try:
        return asset_paths.browse(kind, path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class MkdirRequest(BaseModel):
    relative_path: str


@router.post("/assets/{kind}/mkdir")
async def mkdir_asset(kind: str, request: MkdirRequest):
    try:
        created = asset_paths.make_subfolder(kind, request.relative_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"kind": kind, "relative_path": request.relative_path, "created": created}


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
def run_graph(payload: GraphRunRequest, request: Request):
    """Executes the submitted graph in topological order. Declared as a
    plain `def`, not `async def`, so FastAPI runs it in its worker thread
    pool instead of the event loop -- a SupervisedLoRATrainerNode run can
    take a long time, and blocking the event loop for that long would
    freeze every other request (SSE progress for real training runs
    included) until it finished.

    Progress streaming exists now for MonitorNode-family nodes (see
    nodes/monitor/) via the injected ExecutionContext's monitor_bus --
    this endpoint's own response still only arrives once the whole graph
    finishes or a node fails, but a node that reports through a monitor
    can be watched live at /nodegraph/monitor/{monitor_id} while this
    call is still blocked.
    """
    nodes = [graph_executor.NodeSpec(id=n.id, class_name=n.class_name, params=n.params)
             for n in payload.nodes]
    edges = [graph_executor.EdgeSpec(from_node=e.from_node, from_port=e.from_port,
                                      to_node=e.to_node, to_port=e.to_port)
             for e in payload.edges]
    context = ExecutionContext(monitor_bus=request.app.state.monitor_bus)
    try:
        executor = graph_executor.GraphExecutor(nodes, edges, context)
        results = executor.run()
    except graph_executor.GraphError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"results": [
        {"node_id": r.node_id, "ok": r.ok, "outputs": r.outputs, "error": r.error}
        for r in results
    ]}
