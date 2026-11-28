"""The journal behind every page's console: kept entries, numbering that survives a restart, the server's log and each run's end in it, and the two routes.

    python -m backend.tests.test_journal
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.execution_recorder import ExecutionRecorder
from backend.infrastructure.file_journal import FileJournal
from backend.infrastructure.journal_log_handler import JournalLogHandler
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, build_services, check, finish

API = "/api/v1/journal"
root = Path(tempfile.mkdtemp(prefix="backend-journal-"))

# the file behind it
path = root / "j.jsonl"
journal = FileJournal(path, keep=5)
for i in range(8):
    journal.add("info", "server", f"line {i}")
journal.add("bogus", "server", "x" * 9000)
last = journal.after(0, 100)
check([e.seq for e in last] == [5, 6, 7, 8, 9], f"only the newest few are held ({[e.seq for e in last]})")
check(last[-1].level == "info" and len(last[-1].text) < 4100, "an unknown level becomes info and a huge text is cut")
check([e.seq for e in journal.after(7, 100)] == [8, 9] and [e.seq for e in journal.after(0, 2)] == [8, 9], "'after' gives what came later; 'limit' keeps the newest")
for i in range(12):
    journal.add("warn", "run #3", f"more {i}")
check(len(path.read_text().splitlines()) <= 10, "the file is cut back, not kept whole")
path.write_text(path.read_text() + '{"seq": 99, "at"')  # a power cut in the middle of a line
again = FileJournal(path, keep=5)
check(again.last_seq() == journal.last_seq() and [e.text for e in again.after(0, 100)] == [e.text for e in journal.after(0, 100)], "after a restart the entries are back, numbering goes on, a torn last line is ignored")
check(again.add("info", "server", "next").seq == journal.last_seq() + 1, "and the next entry follows the last one")

# the server's log
logger = logging.getLogger("backend.test.journal")
handler = JournalLogHandler(journal)
logger.addHandler(handler)
logger.warning("disk %s", "low")
try:
    raise ValueError("boom")
except ValueError:
    logger.exception("catalog failed")
logger.info("not interesting")
logger.removeHandler(handler)
tail = journal.after(0, 3)
check([(e.level, e.source) for e in tail[-2:]] == [("warn", "server"), ("error", "server")] and "disk low" in tail[-2].text and "ValueError: boom" in tail[-1].text,
      f"warnings and errors of the server's log are filed, info is not ({[e.text for e in tail]})")

# routes
services = build_services(project_root=root / "app")
app = create_app(services)
status_, _, body = asgi_request(app, API, method="POST", json_body={"level": "success", "text": "Saved \"x\"."})
check(status_ == 201 and body["entries"][0]["source"] == "page" and body["entries"][0]["level"] == "success", f"a page's line is filed as coming from the page ({status_} {body})")
status_, _, body = asgi_request(app, API, method="POST", json_body={"level": "error", "text": "   "})
check(status_ == 422 and body["error"]["code"] == "invalid_query", "an empty line is refused")
status_, _, body = asgi_request(app, API, method="POST", json_body={"level": "loud", "text": "x"})
check(status_ == 422, "an unknown level is refused")
status_, _, body = asgi_request(app, f"{API}?after=0&limit=10")
check(status_ == 200 and [e["text"] for e in body["entries"]] == ['Saved "x".'] and body["last"] == body["entries"][-1]["seq"], f"the history is read from the start ({body})")
status_, _, body = asgi_request(app, f"{API}?after={body['last']}")
check(status_ == 200 and body["entries"] == [] , "and from the last entry on there is nothing new")

# a run's end is in it
from backend.domain.events import GraphExecutionFailed, GraphExecutionFinished, GraphExecutionStarted, GraphExecutionStopped
for event in (GraphExecutionStarted(execution_id=7), GraphExecutionFailed(execution_id=7, error="out of memory"), GraphExecutionStopped(execution_id=8, reason="user"), GraphExecutionFinished(execution_id=9, nodes=3)):
    services.events.publish_events(event) if hasattr(services.events, "publish_events") else services.event_bus.publish(event)
entries = asgi_request(app, f"{API}?after=1")[2]["entries"]
got = [(e["level"], e["source"], e["text"]) for e in entries]
check(got == [("info", "run #7", "started"), ("error", "run #7", "failed: out of memory"), ("warn", "run #8", "stopped: user"), ("success", "run #9", "finished")],
      f"how a run began and ended is in the journal ({got})")

# a node's warning during a run
recorder = ExecutionRecorder.__new__(ExecutionRecorder)
recorder._monitor_bus = None
recorder._journal = journal
before = journal.last_seq()
recorder.republish_monitor(12, {"monitor_id": "warning", "data": {"node": "n_x", "message": "Preview Image: no folder"}})
recorder.republish_monitor(12, {"monitor_id": "warning", "data": {"node": "n_x", "message": "again"}}, catch_up=True)
recorder.republish_monitor(12, {"monitor_id": "node:n_x", "data": {"step": 1}})
got = [(e.level, e.source, e.text) for e in journal.after(before, 10)]
check(got == [("warn", "run #12", "Preview Image: no folder")], f"a node's warning is filed once with its run, not again when a restarted server reads the run's events from the start ({got})")

finish()
