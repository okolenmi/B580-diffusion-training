"""FastAPI web server for the training UI."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .config import settings
from .process_manager import cleanup_orphaned_runs
from .routes_config import router as config_router
from .routes_history import router as history_router
from .routes_monitor import router as monitor_router
from .routes_settings import router as settings_router
from .routes_sse import router as sse_router
from .routes_training import router as training_router
from .routes_datasets import router as datasets_router
from .routes_nodegraph import router as nodegraph_router

import paths as _paths
from monitor_bus import MonitorBus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events."""
    # Initialize Main Server DB
    db.init_db(settings.db_path)

    # One MonitorBus for the process's lifetime, owned by app.state (not a
    # module-level singleton -- see monitor_bus.py's docstring). Route
    # handlers read it via request.app.state.monitor_bus; graph execution
    # receives it through nodes.core.ExecutionContext, not an import.
    app.state.monitor_bus = MonitorBus()

    # Sync paths.py's module-level overrides with the resolved settings (DB
    # override if set, else env var / ComfyUI default) so this process's own
    # path resolution -- e.g. the checkpoint/LoRA file-listing endpoint --
    # matches what gets injected into training subprocesses at launch.
    _paths.set_checkpoints_dir(settings.checkpoints_dir)
    _paths.set_loras_dir(settings.loras_dir)

    # Clean up orphaned runs
    killed = cleanup_orphaned_runs(settings.db_path)
    if killed:
        print(f"  Killed {killed} orphaned run(s) from previous session.")
    
    yield


import logging
import os
import re

# Uvicorn's own access log, one line per HTTP request -- useful for a request that
# actually did something (a POST, an error), noise for the same GET status poll
# repeated every second while a graph run or a training job is in progress (real
# complaint: the node-graph editor's own /nodegraph/run/{id} status poll -- not
# previously covered by the narrower, per-endpoint-string filter this replaces,
# which excluded two specific legacy polling paths by name and needed a new string
# added by hand every time a new polling endpoint showed up, exactly the gap that
# let this one through). Filters by shape instead: a successful (2xx) GET is
# routine, everything else (a real error, or a non-GET request -- starting a run,
# saving a setting, uploading a file) is worth seeing.
#
# COMFY_ACCESS_LOG controls this -- "filtered" (default): the rule above.
# "all": no filtering, uvicorn's own unmodified access log. "off": no access log
# at all. Doesn't touch uvicorn's own error/warning logging (a crash still prints
# its full traceback regardless of this setting) -- only the one-line-per-request
# access log.
_ACCESS_LOG_MODE = os.environ.get("COMFY_ACCESS_LOG", "filtered").strip().lower()
_SUCCESSFUL_GET_RE = re.compile(r'"GET [^"]*" 2\d\d')


class _FilteredAccessLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _SUCCESSFUL_GET_RE.search(record.getMessage())


if _ACCESS_LOG_MODE == "off":
    logging.getLogger("uvicorn.access").disabled = True
elif _ACCESS_LOG_MODE != "all":
    logging.getLogger("uvicorn.access").addFilter(_FilteredAccessLog())

app = FastAPI(
    title="Training Control Center",
    lifespan=lifespan
)

# Static files
app.mount("/static", StaticFiles(directory=str(settings.project_root / "server/static")), name="static")

# Ensure datasets directory exists for preview serving
datasets_dir = settings.project_root / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")

# Serve run-scoped output files (mid-training preview images, etc.) directly.
runs_dir = settings.runs_dir
runs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(runs_dir)), name="runs")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": exc.__class__.__name__}
    )


# Register routes
app.include_router(config_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(settings_router, prefix="/api")
app.include_router(sse_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(nodegraph_router, prefix="/api")
app.include_router(monitor_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = settings.project_root / "server/static/index.html"
    with open(index_path) as f:
        return f.read()


@app.get("/datasets", response_class=HTMLResponse)
async def dataset_manager():
    path = settings.project_root / "server/static/dataset_manager.html"
    with open(path) as f:
        return f.read()


@app.get("/nodegraph", response_class=HTMLResponse)
async def nodegraph_playground():
    """Interactive node-graph editor -- spawn nodes from the registry,
    connect ports, Run executes the submitted graph server-side (see
    server/graph_executor.py). Still isolated from the production
    config/training path -- executing a graph here only calls .build() on
    nodes/ classes, same boundary as everything else under nodes/. See
    docs/architecture.md."""
    path = settings.project_root / "server/static/nodegraph.html"
    with open(path) as f:
        return f.read()


@app.get("/nodegraph/monitor/{monitor_id}", response_class=HTMLResponse)
async def monitor_dashboard(monitor_id: str):
    """"Look inside" a MonitorNode -- live dashboard for one monitor_id,
    served independently of any running graph (see server/routes_monitor.py
    for the SSE stream it connects to). monitor_id isn't used server-side
    here; the page reads it from its own URL."""
    path = settings.project_root / "server/static/monitor_dashboard.html"
    with open(path) as f:
        return f.read()


def run(host: str = None, port: int = None):
    """Run the server."""
    import uvicorn
    uvicorn.run(
        app, 
        host=host or settings.host, 
        port=port or settings.port, 
        log_level="info"
    )
