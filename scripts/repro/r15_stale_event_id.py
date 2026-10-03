"""N3-02: Last-Event-ID from a PREVIOUS server process is accepted as valid once the new process has published more events."""
import os, sys, datetime
sys.path.insert(0, os.environ.get("REPO","."))
from backend.infrastructure.events.callback_event_bus import CallbackEventBus
from backend.domain import events as E
import inspect
life = [c for n,c in inspect.getmembers(E, inspect.isclass) if getattr(c,"__module__","")==E.__name__ and n.endswith(("Finished","Failed","Cancelled","Started")) and hasattr(c,"wire_name")]
print("lifecycle-ish event classes:", [c.__name__ for c in life][:6])
cls = life[0]
def mk():
    import dataclasses
    kw = {}
    for f in dataclasses.fields(cls):
        t = str(f.type)
        kw[f.name] = datetime.datetime.now(datetime.timezone.utc) if "datetime" in t else (3 if "int" in t and "None" not in t else (None if "None" in t else "x"))
    return cls(**kw)
bus = CallbackEventBus()                       # a freshly restarted server...
for _ in range(60): bus.publish(mk())          # ...that has since run something (60 lifecycle events, seq 1..60)
r = bus.replay_since(40)                       # a tab that last saw seq 40 -- from the PREVIOUS process
print(f"replay_since(40) on the new process -> complete={r.complete}, replayed seqs {r.events[0].seq}..{r.events[-1].seq if r.events else None}")
print("=> the client is told its history is consistent, but id 40 meant a different event in the old process" if r.complete else "not reproduced")
