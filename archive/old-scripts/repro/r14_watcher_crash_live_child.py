"""N3-01: if the watcher thread crashes while the child is ALIVE, the row is marked failed and the child is left running."""
import os, sys, time, importlib.util
from pathlib import Path
sys.path.insert(0, os.environ.get("REPO","."))
spec = importlib.util.spec_from_file_location("T", "backend/tests/test_graph_adoption.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
from backend.domain.graph import GraphDefinition

class FlakyWriter(T.RecordingWriter):
    """The first node commit fails the way a locked / full database would; later commits work."""
    def __init__(self): super().__init__(); self.calls = 0
    def commit(self, execution, expected=None):
        self.calls += 1
        if getattr(execution, "results", ()) and not getattr(self, "_raised", False):
            self._raised = True
            raise RuntimeError("database is locked")          # transient: only this one commit fails
        return super().commit(execution, expected)

events = Path(T.TMP) / "execution_1.events.jsonl"; events.write_text("")
gw = T.RecordingGateway(events, alive=True); gw.found = 777              # a live child, pid 777
g = GraphDefinition.from_dict({"nodes":[{"id":"a","class_name":"X"}],"edges":[]}) if hasattr(GraphDefinition,"from_dict") else GraphDefinition()
class FlakyExecutions(T.StubExecutions):
    """A transient read failure (e.g. 'database is locked') on exactly one poll of the watch loop."""
    def __init__(self, **kw): super().__init__(**kw); self.reads = 0
    def get(self, execution_id):
        self.reads += 1
        if self.reads == 4: raise RuntimeError("database is locked")
        return super().get(execution_id)
ex = FlakyExecutions(graph=g)
sup = T._supervisor(gw, writer=T.RecordingWriter(), executions=ex)
print("adopt ->", sup.adopt(1))
time.sleep(0.2)
gw.write({"kind":"node","node_id":"a","ok":True,"outputs":{},"error":None,"duration_ms":5})   # a live record the watcher must persist
time.sleep(0.8)
row = ex._execution
print(f"row status      : {row.status.value}   error={row.error!r:.90}")
print(f"child alive     : {gw.alive}   stop signals sent: {gw.stopped}   kill signals sent: {gw.killed}")
print("=> row says FAILED, nothing stopped the child: single-active check now allows a second run on the same GPU" if row.status.value in ("failed","error") and gw.alive and not gw.killed and not gw.stopped else "not reproduced")
