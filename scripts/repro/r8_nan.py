"""F-03: a NaN/inf loss is serialized as the bare token NaN (invalid JSON)."""
import sys, json, datetime
sys.path.insert(0, __import__("os").environ.get("REPO","."))
from backend.domain.events import RunProgressed
from backend.presentation.sse import serialize_event
ev = RunProgressed(run_id=1, step=7, total_steps=100, loss=float("nan"), avg_loss=float("inf"), lr=1e-4,
                   phase="training", cache_done=None, cache_total=None,
                   occurred_at=datetime.datetime.now(datetime.timezone.utc))
s = serialize_event(ev)
print(s[:150])
def strict(c): raise ValueError(f"invalid JSON constant {c}")
try: json.loads(s, parse_constant=strict); print("strict parse: OK")
except ValueError as e: print("strict parse FAILED ->", e, "(browser JSON.parse throws the same)")
