"""The child-process graph execution path, end to end.

Everything here spawns a real ``python -m
backend.infrastructure.graph_task_worker`` against the real ``nodes/``
tree. That is the point of the file: the in-process gateway shares the
producer, so a test that only exercised it would prove the producer works
and say nothing about the isolation the child exists to provide -- which is
precisely the part with no in-process equivalent.

The nodes are the four primitive constants because they are real, cheap,
and need no device: the child discovers them the same way it discovers
everything else, so "the child found its nodes" is genuinely tested.

Each spawn costs about two seconds, nearly all of it importing torch and
walking ``nodes/`` -- measured, not guessed (see the WP-22 note in
``docs/design/13-process-isolation.md``). That is the price of a
per-run process, and it is the number the default-flip step is judged on.

Run directly: python backend/tests/test_graph_task_gateway.py
"""

from __future__ import annotations

import json
import os
import signal
import sys
from dataclasses import replace
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.graph_task_gateway import (
    GraphTaskLaunch,
    GraphLaunchError,
)
from backend.domain.value_objects import ExecutionId
from backend.infrastructure.graph_event_stream import (
    EventKind,
    ExecutionEventTail,
)
from backend.infrastructure.graph_task_gateway import SubprocessGraphTaskGateway
from backend.infrastructure.process_identity import cmdline_mentions
from backend.infrastructure.workspace import WorkspaceLayout
from backend.tests.support import (
    check,
    run_tests_concurrently,
    use_temporary_comfy_dir,
    wait_until,
)

# Before anything resolves a ComfyUI path, and before this process
# spawns a child that will resolve one itself. Without it this file
# dies with "Cannot find ComfyUI directory" on a checkout that has no
# COMFY_DIR -- which is what a fresh clone looks like, and what a CI
# runner looks like. See support.use_temporary_comfy_dir.
COMFY = use_temporary_comfy_dir(prefix="backend-graph-child-comfy-")

TMP = Path(tempfile.mkdtemp(prefix="backend-graph-child-"))
GATEWAY = SubprocessGraphTaskGateway(WorkspaceLayout(Path(__file__).resolve().parents[2]))

#: Long enough to cover a real child reaching its first write, short
#: enough that a hung test is a failure rather than a stall.
WAIT = 90.0


def _launch(execution_id: int, nodes: list[dict], *, edges=None) -> GraphTaskLaunch:
    """Write a graph to disk and return the launch describing it."""
    graph = TMP / f"graph_{execution_id}.json"
    graph.write_text(
        json.dumps({"format": 1, "nodes": nodes, "edges": edges or []}),
        encoding="utf-8",
    )
    return GraphTaskLaunch(
        execution_id=ExecutionId(execution_id),
        graph_path=graph,
        event_path=TMP / f"events_{execution_id}.jsonl",
        log_path=TMP / f"run_{execution_id}.log",
    )


def _float_node(node_id: str, value: float) -> dict:
    return {"id": node_id, "class_name": "FloatConstantNode", "params": {"value": value}}


def _collect(
    tail: ExecutionEventTail,
    pid: int,
    until=None,
    timeout: float = WAIT,
) -> list:
    """Collect records, returning everything seen -- not just the new part.

    Both endings matter, and the accumulator matters most. A child that
    writes fewer records than expected and then exits must not cost the
    whole timeout: waiting it out turns a two-line mismatch into a
    two-minute stall and hides which one it was. (It hid this file's
    first two failures, which is why the helper knows about it.)

    Returning *everything* collected rather than only the last batch is
    what lets a caller synchronise on one record and still count it
    afterwards. An earlier version consumed the record it waited for, so a
    child that stopped the instant it got going reported zero nodes -- and
    "stopped before it built anything" is the one thing this helper
    exists to rule out.

    Polling rather than sleeping a guessed interval: the child's cost is
    real work -- importing torch, walking ``nodes/`` -- whose duration is
    not the test's to predict, and a fixed sleep is either slow or flaky.
    """
    deadline = time.monotonic() + timeout
    events: list = []
    while True:
        events.extend(tail.poll())
        if until is not None and until(events):
            return events
        if time.monotonic() >= deadline:
            return events
        if not GATEWAY.is_alive(pid):
            # Settle, do not read once. The child writes its outcome and
            # *then* exits, so "the process is gone" does not mean "the
            # record is visible" -- there is a window between the two, and a
            # single read landed in it about one gate run in three under
            # load, reporting zero records for a child that had written one.
            # Polls until something new arrives, or for half a second.
            previous = len(events)
            settle_until = time.monotonic() + 0.5
            while time.monotonic() < settle_until:
                events.extend(tail.poll())
                if len(events) > previous:
                    break
                time.sleep(0.02)
            return events
        time.sleep(0.02)


def _why_silent(launch: GraphTaskLaunch) -> str:
    """What the child said, for a check that found it silent.

    A child that writes no records has usually written *something* to its
    log -- an import error, a device fault, a traceback -- and a test that
    reports only "no records" throws that away. This fires about one suite
    run in five, on a loaded machine, so "it did not start" is not a
    diagnosis.
    """
    if not launch.log_path.exists():
        return "the child wrote no log at all"
    text = launch.log_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return "the child's log is empty"
    return "the child's log says: " + " | ".join(text.splitlines()[-3:])


def _is_node(event) -> bool:
    return event.kind is EventKind.NODE


def test_child_runs_a_graph_end_to_end() -> None:
    print("\n== a graph executed in a child, watched from the server ==")
    launch = _launch(
        1,
        [_float_node("a", 1.5), _float_node("b", 2.5)],
    )
    pid = GATEWAY.spawn(launch)
    check(pid > 0, f"a real pid came back (got {pid})")
    # Polled, not asserted once. Between fork and execve the child's
    # /proc/<pid>/cmdline is a copy of the *parent's*, and
    # `cmdline_mentions` reports that state as None rather than False --
    # deliberately, because False there is indistinguishable from a
    # recycled pid. `Popen` can return inside that window, so asserting
    # `is True` immediately after spawn is a coin flip on a loaded machine:
    # it failed about one gate run in ten, in this file and not only in the
    # test that added the most children to it.
    #
    # Polling is the honest form. The marker is a statement about the child
    # having exec'd, and exec takes as long as it takes.
    check(
        wait_until(
            lambda: cmdline_mentions(
                pid, "backend.infrastructure.graph_task_worker"
            ) is True,
            timeout=WAIT,
        ),
        f"and /proc agrees it is one of ours -- the marker the liveness "
        f"check and the signals both depend on (got "
        f"{cmdline_mentions(pid, 'backend.infrastructure.graph_task_worker')!r})",
    )
    check(GATEWAY.is_alive(pid), "and it is alive")

    events = _collect(ExecutionEventTail(launch.event_path), pid,
                     until=lambda seen: len(seen) >= 3)
    kinds = [e.kind for e in events]
    check(
        kinds == [EventKind.NODE, EventKind.NODE, EventKind.OUTCOME],
        f"two node records and one outcome, in order (got {kinds})",
    )
    by_id = {e.payload["node_id"]: e.payload for e in events if e.kind is EventKind.NODE}
    check(
        by_id["a"]["ok"] and by_id["a"]["outputs"]["value"] == 1.5,
        f"the child's first node's real output crossed the process boundary "
        f"(got {by_id.get('a')})",
    )
    check(
        by_id["b"]["ok"] and by_id["b"]["outputs"]["value"] == 2.5,
        "and the second's",
    )
    check(
        events[-1].payload == {"kind": "outcome", "error": None, "results_count": 2},
        f"the outcome record says it finished clean (got {events[-1].payload})",
    )
    check(
        wait_until(lambda: not GATEWAY.is_alive(pid), timeout=30.0),
        "and the child exits afterwards, so a watcher waiting on liveness ends",
    )


def test_a_refused_graph_is_reported_by_the_child() -> None:
    print("\n== a graph the child refuses is an outcome, not a lost run ==")
    # The runtime re-validates inside execute() before building anything,
    # so a graph the server would have rejected on the way in is refused
    # again in the child -- and refused *loudly*: the message reaches the
    # server as an outcome error instead of the child dying quietly and the
    # run looking successful.
    #
    # A per-node *build* failure is a different path and is covered in
    # test_graph_execution.py through BoomNode: provoking one here would
    # mean a node that builds and then raises on CPU, and no primitive in
    # nodes/ does that. What crosses the process boundary is the outcome
    # record either way, which is the part with no in-process equivalent.
    launch = _launch(
        2,
        [{"id": "bad", "class_name": "FloatConstantNode", "params": {"value": "not a float"}}],
    )
    pid = GATEWAY.spawn(launch)
    tail = ExecutionEventTail(launch.event_path)
    events = _collect(tail, pid)
    check(
        wait_until(lambda: not GATEWAY.is_alive(pid), timeout=30.0),
        "the child exited on its own",
    )
    outcome = next((e for e in events if e.kind is EventKind.OUTCOME), None)
    # _why_silent in the failure text, because "got []" is not a diagnosis.
    # A silent child has usually written something to its log -- an import
    # error, a device fault, a traceback -- and the run before this had a
    # check report only the empty list while the real cause sat unread.
    check(outcome is not None,
          f"and left an outcome record (got {events}; {_why_silent(launch)})")
    check(
        outcome is not None and outcome.payload["error"],
        f"carrying the failure, which is what the row is failed with "
        f"(got {outcome.payload if outcome else None}; {_why_silent(launch)})",
    )
    check(
        "FloatConstantNode" in str(outcome.payload["error"]),
        "naming the node that was wrong, so the message is actionable",
    )
    check(
        tail.poll() == [],
        "and nothing else followed it -- the refusal is one record, not a "
        "partial run plus a complaint",
    )


def test_a_killed_child_reports_no_outcome() -> None:
    print("\n== the headline: a child that dies does not look like a success ==")
    # This is the WP-22 property with no in-process equivalent.
    #
    # The graph is large on purpose. The run has to still be in progress when
    # the SIGKILL lands, and the only margin this test had was the ~2s of
    # torch import and node discovery before it. That is not a margin, it is a
    # hope: it failed once in a gate run, where the child ran its graph and
    # wrote a clean outcome while the kill was still in flight, and the
    # check then correctly reported an outcome record for a child that had
    # not died at all. Four thousand nodes, as its sibling signal tests use,
    # makes the child's work outlast any plausible scheduling delay instead
    # of racing it.
    nodes = [_float_node(f"n{i}", float(i)) for i in range(4000)]
    launch = _launch(3, nodes)
    pid = GATEWAY.spawn(launch)
    GATEWAY.kill(pid)
    check(
        wait_until(lambda: not GATEWAY.is_alive(pid), timeout=30.0),
        "the killed child is gone",
    )
    events = ExecutionEventTail(launch.event_path).poll()
    check(
        not any(e.kind is EventKind.OUTCOME for e in events),
        f"and it left no outcome record behind -- which is the whole reason "
        f"the supervisor can tell this from a clean finish (got {events})",
    )
    check(
        launch.log_path.exists(),
        "the child's stdout/stderr went to a log, so a real crash is "
        "diagnosable rather than invisible",
    )


def test_stopping_a_run_actually_stops_it() -> None:
    print("\n== a cooperative stop reaches the child's runtime ==")
    # Waited on the first node record, not on a timer: that is what makes
    # this a test of the signal path rather than of the child's startup.
    # Signalling earlier finds a process that has not installed its
    # handler yet, so the default SIGINT raises KeyboardInterrupt, the
    # child dies, "it stopped" passes -- and the cooperative path was
    # never executed at all.
    nodes = [_float_node(f"n{i}", float(i)) for i in range(4000)]
    launch = _launch(4, nodes)
    pid = GATEWAY.spawn(launch)
    tail = ExecutionEventTail(launch.event_path)
    events = _collect(tail, pid, until=lambda seen: any(map(_is_node, seen)))
    # `_why_silent` because this fires about one suite run in five on a
    # loaded machine, and a child that reached no node record has usually
    # said why in its log. A bare "got 0" is not a diagnosis -- and it is
    # how the 2026-10-04 gate run reported this one, in a file that already
    # had the helper for exactly this and simply was not calling it here.
    check(any(map(_is_node, events)),
          f"the child got past startup and started building nodes "
          f"(got {len([e for e in events if _is_node(e)])} node records; "
          f"{_why_silent(launch)})")

    GATEWAY.request_stop(pid)
    check(
        wait_until(lambda: not GATEWAY.is_alive(pid), timeout=60.0),
        "and stopped on SIGINT rather than running all 4000 nodes",
    )
    events += _collect(tail, pid)
    node_records = [e for e in events if e.kind is EventKind.NODE]
    check(
        0 < len(node_records) < len(nodes),
        f"having built some but not all: {len(node_records)} of {len(nodes)} "
        f"nodes reported before it stopped ({_why_silent(launch)})",
    )
    check(
        any(e.kind is EventKind.OUTCOME for e in events),
        f"and it still wrote an outcome, so the stop is distinguishable from "
        f"a crash (got {[e.kind for e in events][-3:]})",
    )


def test_a_sigterm_stops_the_run_the_same_way_a_sigint_does() -> None:
    print("\n== SIGTERM is a stop request, not a crash ==")
    # Round-3 N3-08. The gateway sends SIGINT and escalates to SIGKILL, so
    # the supervisor's own two signals were both handled correctly. SIGTERM
    # -- which is what `kill`, `docker stop` and systemd send -- had no
    # handler at all, so the default disposition killed the child on
    # arrival. Measured mid-run on a 60000-node graph before the fix:
    #
    #   SIGINT   exit 0,   outcome record, 200 node records
    #   SIGTERM  exit -15, no record,       86 node records lost
    #
    # With no record the supervisor falls back to "execution process exited
    # without reporting an outcome (crashed, or a device fault killed it)".
    # For a run somebody deliberately stopped that sentence is false, and
    # the results already on disk are thrown away with it.
    #
    # Deliberately not a gateway verb. `request_stop` sends SIGINT and
    # `kill` sends SIGKILL on purpose -- SIGKILL cannot be caught, and that
    # is what makes it the escalation. The question here is what a signal
    # from *outside* the supervisor does, so the test sends it itself.
    #
    # To the pid, not to the process group, and only after re-checking
    # whose it is. The first version of this test did
    # `os.killpg(os.getpgid(pid), SIGTERM)` and that was a way to kill the
    # test runner: `_collect` returns as soon as the child is gone, so the
    # pid it hands back may already be dead, and if the kernel has since
    # given that number to anything in *our* process group then
    # `os.getpgid` returns our own group and the killpg SIGTERMs every test
    # in the file. It did exactly that once: every check printed, and then
    # the process died with no traceback and a non-zero exit, which reads
    # in the suite report as a file that failed having passed everything.
    #
    # The gateway guards the same hazard with a cmdline marker and refuses
    # to signal when it does not match; the test uses that guard rather
    # than hand-rolling a weaker one. Signalling a single pid also caps
    # the damage if the check is simply wrong: one innocent process rather
    # than this whole process group.
    nodes = [_float_node(f"n{i}", float(i)) for i in range(4000)]
    launch = _launch(5, nodes)
    pid = GATEWAY.spawn(launch)
    tail = ExecutionEventTail(launch.event_path)
    events = _collect(tail, pid, until=lambda seen: any(map(_is_node, seen)))
    if not any(map(_is_node, events)):
        print(f"    DIAG {_why_silent(launch)}")
    # `_why_silent` because this fires about one suite run in five on a
    # loaded machine, and a child that reached no node record has usually
    # said why in its log. A bare "got 0" is not a diagnosis -- and it is
    # how the 2026-10-04 gate run reported this one, in a file that already
    # had the helper for exactly this and simply was not calling it here.
    check(any(map(_is_node, events)),
          f"the child got past startup and started building nodes "
          f"(got {len([e for e in events if _is_node(e)])} node records; "
          f"{_why_silent(launch)})")
    # By this point the child has written records, so it has long exec'd --
    # but `is not True` is used rather than `is True` anyway, because the
    # assertion that matters is the one that must not fire: that the pid is
    # ours before it is signalled. See the pre-exec window noted above.
    check(
        cmdline_mentions(pid, "backend.infrastructure.graph_task_worker")
        is not False,
        f"and pid {pid} is still our graph child, so it is safe to signal "
        f"({_why_silent(launch)})",
    )

    os.kill(pid, signal.SIGTERM)
    check(
        wait_until(lambda: not GATEWAY.is_alive(pid), timeout=60.0),
        "and a SIGTERM from the outside world stops it, rather than "
        "killing it where it stands",
    )
    events += _collect(tail, pid)
    node_records = [e for e in events if e.kind is EventKind.NODE]
    check(
        0 < len(node_records) < len(nodes),
        f"having built some but not all: {len(node_records)} of "
        f"{len(nodes)} nodes reported before it stopped",
    )
    outcomes = [e for e in events if e.kind is EventKind.OUTCOME]
    check(
        bool(outcomes),
        f"and it wrote an outcome, so the row is not told the process "
        f"crashed (got {[e.kind.name for e in events][-3:]})",
    )
    # `is not True`, not `is False`: `cmdline_mentions` has three answers,
    # and for a process that no longer exists it returns None ("nothing is
    # known about *this* process yet"), which is documented to err toward
    # alive. Asserting `is False` here therefore failed on a bare checkout
    # -- where the pid is reliably gone -- while passing on a developer
    # machine, where the number had usually been recycled onto something
    # whose cmdline reads as a definite no. What matters after a stop is
    # that the pid is no longer identified as a live graph child.
    check(
        cmdline_mentions(pid, "backend.infrastructure.graph_task_worker") is not True,
        f"and pid {pid} is no longer a live graph child, so the stop "
        f"reached the child rather than a recycled number",
    )


def test_signals_are_refused_for_a_pid_that_is_not_ours() -> None:
    print("\n== PID reuse: refusing to signal somebody else's process ==")
    # Our own test process. It is alive, its pid is a valid one, and its
    # cmdline says nothing about graph_task_worker -- exactly the state a
    # reused pid would be in. A bare kill(pid, 0) would call that alive.
    own = os.getpid()
    check(cmdline_mentions(own, "backend.infrastructure.graph_task_worker") is False,
          "the marker check says this pid is not a graph child")
    check(not GATEWAY.is_alive(own), "so liveness is False despite the process "
          "being very much alive")
    GATEWAY.request_stop(own)  # must be a no-op, not this test runner's death
    GATEWAY.kill(own)
    check(True, "and signalling it was refused rather than delivered -- this "
                "test process is still running, which is the proof")


def test_a_launch_failure_is_loud_and_typed() -> None:
    print("\n== a child that cannot start raises, rather than returning a pid ==")
    # The supervisor's row-repair path hangs off this: a spawn failure must
    # be an exception it can catch, because there is no watcher to notice
    # anything and the row would sit queued forever, blocking the
    # single-active check.
    #
    # The failure is forced through the log path rather than through
    # ``VENV_PYTHON``. The env var is the obvious way to point the gateway
    # at an interpreter that is not there, and it was the first thing
    # tried -- and it breaks these tests the moment they run
    # concurrently: the environment is process-global, so the other tests
    # in this file spawn their children while it is patched and hand Popen
    # a path that does not exist. It failed about one suite run in three,
    # in a *different* test, which is the worst place a flake can live.
    # A per-layout ``settings_kv`` does not help either: venv_python()
    # consults the environment before the settings tier, and the repo's
    # .env sets it.
    #
    # A log path that is a directory is a real spawn failure -- the scratch
    # tree is not writable -- and it needs no process-global state.
    blocked = TMP / "log_is_a_directory"
    blocked.mkdir(parents=True, exist_ok=True)
    launch = _launch(9, [_float_node("a", 1.0)])
    blocked_launch = replace(launch, log_path=blocked)
    try:
        GATEWAY.spawn(blocked_launch)
        check(False, "spawn raised GraphLaunchError")
    except GraphLaunchError as exc:
        check(
            "log_is_a_directory" in str(exc) or "Is a directory" in str(exc),
            f"and the message names what could not be launched (got {exc})",
        )
    except Exception as exc:  # noqa: BLE001 -- wrong type is the failure
        check(False, f"raised GraphLaunchError, got {type(exc).__name__}: {exc}")
    check(True, "and the row-repair path is reachable")


def main() -> None:
    # Concurrent because each test costs a real child-process startup and
    # the tests are independent processes by construction; see
    # support.run_tests_concurrently. Five serial startups took the backend
    # suite from 6.3s to 15s, and together they cost about one.
    tests = [
        test_child_runs_a_graph_end_to_end,
        test_a_refused_graph_is_reported_by_the_child,
        test_a_killed_child_reports_no_outcome,
        test_stopping_a_run_actually_stops_it,
        test_a_sigterm_stops_the_run_the_same_way_a_sigint_does,
        test_signals_are_refused_for_a_pid_that_is_not_ours,
        test_a_launch_failure_is_loud_and_typed,
    ]
    run_tests_concurrently(tests)


if __name__ == "__main__":
    main()
