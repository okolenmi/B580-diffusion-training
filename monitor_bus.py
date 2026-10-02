"""Thread-safe pub-sub for MonitorNode live data.

Same core pattern as server/sse.py's SSEManager (asyncio.Queue per
subscriber, broadcasts marshaled onto the event loop via
call_soon_threadsafe since graph execution runs in a FastAPI worker
thread, not the event loop) -- not reused directly because the domain is
different enough to not share a class cleanly: string monitor_id keys
instead of int run_id, a generic payload dict instead of a fixed
progress/status shape, and a history buffer so a dashboard opened after a
run has already started (or reloaded mid-run) gets the run's data back,
not just future events -- capped at HISTORY_LIMIT, matched to the
dashboard's own record cap so the replay covers everything the charts
would have kept.

Deliberately NOT a module-level singleton the way SSEManager's `sse =
SSEManager()` is: no instance is created here. server/main.py's lifespan
creates exactly one and attaches it to app.state; server/routes_monitor.py
and GraphExecutor both receive it as an explicit constructor/dependency
parameter. Same reason this lives at the top level, not under server/:
nodes/ can accept a MonitorBus instance without importing server/, and
tests can hand any code here a fresh, isolated instance instead of
sharing process-wide state through an import.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque

# How much of a run a (re)connecting dashboard gets replayed. Matched to
# the monitor dashboard's own MAX_RECORDS (100k) on purpose: a page
# reload must restore exactly what the chart would have kept running --
# at 500, any refresh silently truncated the graph to "the newest 500
# steps", and shrank the hero's elapsed readout with it (elapsed spans
# the oldest report the dashboard has seen). Reports cost ~2KB each: a
# 3k-step run ~6MB; the cap's worst case (~200MB, plus ~0.8s of
# json.dumps in the subscribe path) only exists for runs that actually
# report 100k steps, and clear() drops it all when the next run starts.
HISTORY_LIMIT = 100000

# Per-subscriber backlog cap. A dashboard that cannot keep up (a
# backgrounded tab, a stalled laptop) used to grow this queue without
# bound, holding every report in RAM until the client came back. The
# live stream is telemetry: the NEWEST report is the one a chart needs,
# so a full queue gives up its oldest *step report* and keeps the
# frames that carry meaning (clear / run_end). History replay is a
# separate buffer above, so nothing is lost that was not already
# transient. docs 07 F-14.
QUEUE_MAX = 512


class MonitorBus:

    def __init__(self):
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None
        # Exception types already warned about, so a stream that is
        # broken in a loop says it once instead of once per frame (the
        # backend adapter sanitizes before reporting, so this only fires
        # for a caller that does not -- see _encode).
        self._warned: set[type] = set()

    def _encode(self, item: dict) -> str | None:
        """``data: {...}\n\n`` for one item, or None if it cannot be
        serialized.

        Both callers of json.dumps run on somebody else's thread: report()
        is called by graph execution from a FastAPI worker thread, and
        subscribe() runs on the event loop. An unserializable value --
        a torch tensor, a Path, a set -- would raise straight into the
        caller, and for report() that caller is a training step. A
        monitor feed losing one frame is not worth taking a run down
        for (docs 08 N-08).

        The frame is dropped rather than replaced: the review's rule is
        that a bad frame is data we cannot show, and inventing a
        substitute would be a number nobody measured.
        """
        try:
            return f"data: {json.dumps(item)}\n\n"
        except (TypeError, ValueError) as exc:
            if type(exc) not in self._warned:
                self._warned.add(type(exc))
                print(
                    f"monitor bus: dropping an unserializable frame "
                    f"({type(exc).__name__}: {exc}); further frames of this "
                    f"kind will be dropped silently"
                )
            return None

    def subscribe(self, monitor_id: str) -> asyncio.Queue:
        """Call only from within a running event loop (an async route
        handler) -- captures the loop on first use, same as SSEManager."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        for item in self._history[monitor_id]:
            payload = self._encode(item)
            if payload is not None:
                q.put_nowait(payload)
        self._subscribers[monitor_id].append(q)
        return q

    def unsubscribe(self, monitor_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(monitor_id)
        if subs and q in subs:
            subs.remove(q)

    def clear(self, monitor_id: str) -> None:
        """Drops this monitor_id's history and tells every currently-
        connected dashboard to reset its own chart, live -- called once
        by TrainingProgressMonitorNode.build() each time a *new* run
        starts reporting to a given monitor_id (build() runs exactly
        once per node per graph run, so "a new run is about to report
        here" and "this node is being constructed" are the same event).

        Without this, re-running training against the same monitor_id
        (the normal case -- a monitor_id is deliberately meant to
        outlive any one run, see MonitorNode.COMMON_INPUTS' own
        docstring) replayed the old run's full history to every new
        subscriber ahead of the new run's own data, and kept appending
        the new run's data onto the same deque behind it -- both runs
        landing on the same step axis at once, rendered as two
        overlapping lines with no visual indication they're different
        runs. A real report, not a hypothetical: confirmed exactly this
        shape from actual use. Clearing on every fresh build() removes
        that case entirely rather than trying to distinguish/label
        "which run" a given point belongs to -- simpler, and matches
        what was actually asked for (the old run's data gone, not kept
        and marked)."""
        self._history[monitor_id].clear()
        if self._loop is None:
            return
        payload = f"data: {json.dumps({'type': 'clear'})}\n\n"
        for q in list(self._subscribers.get(monitor_id, [])):
            self._loop.call_soon_threadsafe(self._safe_put, q, payload)

    def report(self, monitor_id: str, data: dict) -> None:
        """Safe to call from any thread -- this is the side graph
        execution actually calls, from a FastAPI worker thread.

        Cannot raise: a frame that will not serialize is dropped (and
        not stored), so a weird value in one report cannot end a
        training step or poison the replay for every later subscriber.
        """
        payload = self._encode(data)
        if payload is None:
            return
        self._history[monitor_id].append(data)
        if self._loop is None:
            return  # nothing subscribed yet; history above still keeps it for later
        for q in list(self._subscribers.get(monitor_id, [])):
            self._loop.call_soon_threadsafe(self._safe_put, q, payload)

    @staticmethod
    def _safe_put(q: asyncio.Queue, data: str) -> None:
        """Never block the publisher; on a full backlog, drop a report.

        Drops the oldest *step report* (telemetry) rather than the frame
        that arrived: a chart only ever needs the newest numbers, while
        `clear` / `run_end` reset state and must not be lost (docs 07
        F-14).
        """
        try:
            q.put_nowait(data)
            return
        except asyncio.QueueFull:
            pass
        try:
            oldest = q.get_nowait()
        except asyncio.QueueEmpty:  # raced with the consumer
            return
        if '"type"' in oldest and '"step"' not in oldest:
            # Not a step report: put it back and give up on this frame
            # rather than losing a state change.
            q.put_nowait(oldest)
            return
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass
