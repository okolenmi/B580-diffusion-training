"""N3-04: a Last-Event-ID from a PREVIOUS server process must not be accepted.

Every process numbers its events from 1, so a client's cursor only means
something together with the epoch of the process that issued it. Before the
cursor existed, the bus could recognise a foreign id in exactly one case:
its number being *ahead* of everything published here. That stops being
true the moment the new process publishes past it -- which is precisely
what happens when a tab reconnects late after a restart -- and from then on
a dead process's id is served as if it were current, so the client is told
its history is continuous when it is not and skips the refetch that would
have corrected it.

This prints what the bus does with three cursors: one from this process,
one from a process that is gone, and one that is nonsense. Only the first
may come back `complete=True`.
"""
import datetime
import inspect
import os
import sys

sys.path.insert(0, os.environ.get("REPO", "."))

from backend.application.ports.event_bus import EventCursor
from backend.domain import events as E
from backend.infrastructure.events.callback_event_bus import CallbackEventBus

life = [
    c
    for n, c in inspect.getmembers(E, inspect.isclass)
    if getattr(c, "__module__", "") == E.__name__
    and n.endswith(("Finished", "Failed", "Cancelled", "Started"))
    and hasattr(c, "wire_name")
]
print("lifecycle-ish event classes:", [c.__name__ for c in life][:6])
cls = life[0]


def mk():
    import dataclasses

    kw = {}
    for f in dataclasses.fields(cls):
        t = str(f.type)
        kw[f.name] = (
            datetime.datetime.now(datetime.timezone.utc)
            if "datetime" in t
            else (3 if "int" in t and "None" not in t else (None if "None" in t else "x"))
        )
    return cls(**kw)


bus = CallbackEventBus()          # a freshly restarted server...
for _ in range(60):
    bus.publish(mk())             # ...that has since run something (seq 1..60)
print(f"this process's epoch: {bus.epoch}")
print(f"the wire form of a cursor: {EventCursor(bus.epoch, 40).wire()}\n")

cases = [
    ("this process, 40 events in", EventCursor(bus.epoch, 40), True),
    ("a dead process, id 40", EventCursor("0000deadbeef", 40), False),
    ("a dead process, id 1", EventCursor("0000deadbeef", 1), False),
    ("nothing usable (None)", None, False),
]

wrong = 0
for label, cursor, want_complete in cases:
    r = bus.replay_since(cursor)
    span = (
        f"{r.events[0].seq}..{r.events[-1].seq}" if r.events else "no events"
    )
    ok = r.complete == want_complete
    wrong += not ok
    print(f"  {label:<26} -> complete={str(r.complete):<5} {span:<12} "
          f"{'ok' if ok else 'WRONG (expected complete=' + str(want_complete) + ')'}")

print()
parsed = EventCursor.parse(f"{bus.epoch}:40")
print(f"a bare number parses to:        {EventCursor.parse('40')}")
print(f"'{bus.epoch}:40' parses to:     seq={parsed.seq}, epoch matches={parsed.epoch == bus.epoch}")

print()
if wrong:
    print(f"=> {wrong} case(s) answered wrongly")
    sys.exit(1)
print("=> a cursor from a previous process is refused whatever its number, so the")
print("   client is told to refetch instead of being handed a partial replay")
print("   that looks continuous.")