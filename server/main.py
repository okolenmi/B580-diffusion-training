"""FastAPI web server for the distillation converter."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .config import settings
from .process_manager import cleanup_orphaned_runs
from .routes_config import router as config_router
from .routes_history import router as history_router
from .routes_settings import router as settings_router
from .routes_sse import router as sse_router
from .routes_training import router as training_router
from .routes_datasets import router as datasets_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events."""
    # Initialize Main Server DB
    db.init_db(settings.db_path)
    
    # Clean up orphaned runs
    killed = cleanup_orphaned_runs(settings.db_path)
    if killed:
        print(f"  Killed {killed} orphaned run(s) from previous session.")
    
    yield


import logging

# Filter out successful polling and preview logs to keep console clean
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/tasks/active" not in msg and "/previews/" not in msg

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI(
    title="Comfy Distillation Converter",
    lifespan=lifespan
)

# Static files
app.mount("/static", StaticFiles(directory=str(settings.project_root / "server/static")), name="static")

# Ensure datasets directory exists for preview serving
datasets_dir = settings.project_root / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")


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


def run(host: str = None, port: int = None):
    """Run the server."""
    import uvicorn
    uvicorn.run(
        app, 
        host=host or settings.host, 
        port=port or settings.port, 
        log_level="info"
    )
