"""Node-graph routes -- editor tab, isolated from the production
config/training path (see docs/nodes_package_design.md).

/optimizers and /registry are pure introspection: read declared metadata,
build nothing, run nothing. /run is different in kind -- it actually
constructs and executes the submitted graph (see graph_executor.py's
docstring for why that's a deliberate, not accidental, distinction).
Neither touches config.py, config_model.py, or the training launch path.
"""

import threading
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from nodes.core import ExecutionContext

from . import asset_paths, graph_executor
from .nodegraph_introspect import introspect_optimizer_nodes, introspect_registry, node_info_to_dict

router = APIRouter(prefix="/nodegraph")


class _Execution:
    """One /run call's state -- a threading.Event a running
    SupervisedLoRATrainerNode polls cooperatively between steps
    (nodes/train/supervised.py, via ExecutionContext.should_cancel()),
    plus whatever's needed to answer a later status poll. Registered here
    specifically so /run/{id}/stop has something to set: the thread
    running executor.run() and the request that eventually asks "is it
    done yet" are different requests entirely, so this can't just be a
    local variable."""

    def __init__(self):
        self.cancel_event = threading.Event()
        self.status = "running"  # running -> finished | error | stopped
        self.results = None
        self.error = None
        self.started_at = time.time()


class _ExecutionRegistry:
    """Plain dict + lock -- no eviction policy, matching this project's
    existing DataTaskManager (server/routes_datasets.py): a long-running
    dev server accumulates a bounded number of these (one per graph run a
    person actually triggers), not an unbounded stream, so there's
    nothing here that needs pruning to stay healthy."""

    def __init__(self):
        self._lock = threading.Lock()
        self._executions: dict[str, _Execution] = {}

    def start(self, nodes, edges, monitor_bus) -> str:
        """Allocates the execution's id/cancel_event *before* building the
        GraphExecutor, and only starts the thread once the executor holds
        the real (correctly cancel_event-wired) context -- not built,
        then swapped in after the thread's already running, which would
        race the worker thread reading self.context before the swap
        happened. GraphExecutor's own construction doesn't validate
        anything (confirmed directly, not assumed) -- node classes and
        edges only get checked once run() actually starts, inside the
        thread, so any GraphError shows up through the normal status
        poll (status="error"), same path as a node's build() failing.
        """
        execution_id = uuid.uuid4().hex
        execution = _Execution()
        context = ExecutionContext(monitor_bus=monitor_bus, cancel_event=execution.cancel_event)
        executor = graph_executor.GraphExecutor(nodes, edges, context)

        with self._lock:
            self._executions[execution_id] = execution

        def _worker():
            try:
                results = executor.run()
                execution.results = [
                    {"node_id": r.node_id, "ok": r.ok, "outputs": r.outputs, "error": r.error}
                    for r in results
                ]
                execution.status = "stopped" if execution.cancel_event.is_set() else "finished"
            except graph_executor.GraphError as e:
                execution.error = str(e)
                execution.status = "error"
            except Exception as e:
                execution.error = f"{type(e).__name__}: {e}"
                execution.status = "error"
            finally:
                # By the time this runs, executor.run() has already
                # returned -- its own stack frame held the actual heavy
                # per-node outputs (models, tensors, optimizer state) and
                # released them the moment it returned, before this
                # `finally` block even starts. What's left is PyTorch's
                # caching allocator still holding that freed device memory
                # reserved for this process to reuse, rather than handing
                # it back to the driver -- which is why VRAM can still
                # look "full" in nvidia-smi/xpu-smi even once every
                # reference is really gone. empty_cache() is what actually
                # returns it; gc.collect() first matters because CPython's
                # refcounting alone won't free anything sitting in a
                # reference cycle (an nn.Module's parent/child
                # back-references, or an autograd graph, are exactly that
                # shape).
                import gc
                gc.collect()
                try:
                    from core.comfy_setup import xpu_empty_cache
                    xpu_empty_cache()
                except Exception as cleanup_exc:  # noqa: BLE001 -- best-effort, never mask the run's own result
                    print(f"[nodegraph] VRAM cleanup after run failed (non-fatal): {cleanup_exc}")

        threading.Thread(target=_worker, daemon=True).start()
        return execution_id

    def get(self, execution_id: str) -> _Execution | None:
        with self._lock:
            return self._executions.get(execution_id)

    def stop(self, execution_id: str) -> bool:
        execution = self.get(execution_id)
        if execution is None:
            return False
        execution.cancel_event.set()
        return True


_registry = _ExecutionRegistry()


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
    """Starts the submitted graph running in a background thread and
    returns immediately with an execution_id -- does not wait for the
    graph to finish, or for anything about the graph to be validated:
    GraphExecutor only checks node classes/edges once run() actually
    starts (inside the background thread here), not at construction, so
    a bad class name or a dangling edge reports through the same status
    poll as any other failure (status="error", error=<message>) rather
    than a synchronous 400 -- one reporting path for every kind of
    failure instead of two.

    GET /run/{execution_id} to poll status/results; POST
    /run/{execution_id}/stop to cancel a run in progress (cooperative --
    see nodes/train/supervised.py's step loop; stops between steps, not
    mid-backward-pass, and whatever's been trained so far is kept, not
    discarded).

    Progress streaming exists separately for MonitorNode-family nodes
    (see nodes/monitor/) via the injected ExecutionContext's monitor_bus,
    watchable at /nodegraph/monitor/{monitor_id}/stream regardless of
    this endpoint's own polling.
    """
    nodes = [graph_executor.NodeSpec(id=n.id, class_name=n.class_name, params=n.params)
             for n in payload.nodes]
    edges = [graph_executor.EdgeSpec(from_node=e.from_node, from_port=e.from_port,
                                      to_node=e.to_node, to_port=e.to_port)
             for e in payload.edges]

    execution_id = _registry.start(nodes, edges, request.app.state.monitor_bus)
    return {"execution_id": execution_id}


@router.get("/run/{execution_id}")
def run_status(execution_id: str):
    execution = _registry.get(execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail=f"No execution with id {execution_id}.")
    return {
        "status": execution.status,
        "results": execution.results,
        "error": execution.error,
        "started_at": execution.started_at,
    }


@router.post("/run/{execution_id}/stop")
def stop_run(execution_id: str):
    if not _registry.stop(execution_id):
        raise HTTPException(status_code=404, detail=f"No execution with id {execution_id}.")
    return {"ok": True}
