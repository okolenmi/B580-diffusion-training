"""Integration test -- the curves of a finished run survive the server restart that deletes its event file.

Frames a run reported to a monitor are readable while the run's event file exists, are kept by the startup sweep in a
small archive, are thinned to a bounded count, and go away with the history. Real container, real files.

Run directly: python backend/tests/test_execution_metrics.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.use_cases import SweepExecutionScratch
from backend.infrastructure.execution_metrics import MAX_FRAMES, FileExecutionMetrics, thin
from backend.infrastructure.graph_event_stream import ExecutionEventWriter
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, build_services, check, finish, gdoc, gnode, wait_until

GRAPH = "/api/v1/graphs"

# -- thinning keeps the shape and the ends ---------------------------------------------
rows = [{"step": n} for n in range(1, 10_001)]
kept = thin(rows)
check(len(kept) == MAX_FRAMES and kept[0] == rows[0] and kept[-1] == rows[-1], "thin keeps the bound and both ends")
check([f["step"] for f in kept] == sorted(f["step"] for f in kept), "thin keeps the order")
check(thin(rows[:5]) == rows[:5], "a short run is kept whole")

# -- a run, then frames in its event file ----------------------------------------------
root = Path(tempfile.mkdtemp(prefix="backend-exec-metrics-"))
scratch = root / "test-graph-scratch"
services = build_services(project_root=root)
app = create_app(services)
status, _, body = asgi_request(app, f"{GRAPH}/run", method="POST", json_body={"graph": gdoc([gnode("v", "ScaleNode", value=3.0, factor=5.0)], [])})
run_id = body["execution_id"]
check(wait_until(lambda: asgi_request(app, f"{GRAPH}/executions/{run_id}")[2]["status"] in ("finished", "error", "stopped"), timeout=5.0), "the run ends")

writer = ExecutionEventWriter(scratch / f"execution_{run_id}.events.jsonl")
for step in range(1, 6001):
    writer.monitor("node:t1", {"step": step, "total": 6000, "loss": 1.0 / step})
writer.monitor("node:other", {"step": 1})

status, _, body = asgi_request(app, f"{GRAPH}/executions/{run_id}/monitor")
check(status == 200 and body["streams"] == ["node:other", "node:t1"], f"streams of the run ({body})")
status, _, body = asgi_request(app, f"{GRAPH}/executions/{run_id}/monitor/node:t1")
frames = body["frames"]
check(status == 200 and len(frames) == MAX_FRAMES and frames[0]["step"] == 1 and frames[-1]["step"] == 6000, "frames are thinned, first and last kept")
status, _, body = asgi_request(app, f"{GRAPH}/executions/{run_id}/monitor/node:missing")
check(status == 200 and body["frames"] == [], "a stream the run never used is empty, not an error")
status, _, _ = asgi_request(app, f"{GRAPH}/executions/99999/monitor")
check(status == 404, "an unknown run is 404")

# -- the startup sweep: the event file goes, the curves stay ---------------------------
events = scratch / f"execution_{run_id}.events.jsonl"
metrics = FileExecutionMetrics(scratch)
SweepExecutionScratch(executions=services.graphs.get_execution._executions, scratch_dir=scratch, metrics=metrics).execute()
check(not events.exists(), "the sweep removed the event file")
check((scratch / "metrics" / f"execution_{run_id}.json").is_file(), "its frames were archived first")
status, _, body = asgi_request(app, f"{GRAPH}/executions/{run_id}/monitor/node:t1")
check(status == 200 and len(body["frames"]) == MAX_FRAMES and body["frames"][-1]["step"] == 6000, "the curves are served from the archive")

# -- deleting the history takes the archive with it ------------------------------------
status, _, _ = asgi_request(app, f"{GRAPH}/executions", method="DELETE")
check(status == 200 and not list((scratch / "metrics").glob("*.json")), "deleting the history removes the archives")

finish()
