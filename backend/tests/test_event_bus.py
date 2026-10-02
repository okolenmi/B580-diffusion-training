"""Unit tests -- CallbackEventBus semantics and thread safety.

Run directly: python backend/tests/test_event_bus.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.event_bus import Sequenced
from backend.domain.events import RunCompleted, RunsDeleted
from backend.infrastructure.events.callback_event_bus import CallbackEventBus
from backend.tests.support import check, finish


def test_pubsub_lifecycle() -> None:
    print("\n== publish / subscribe / close ==")
    bus = CallbackEventBus()
    received: list[Sequenced] = []
    subscription = bus.subscribe(received.append)

    first = RunCompleted(run_id=1, done_steps=3)
    bus.publish(first)
    check(
        [item.event for item in received] == [first],
        "subscriber receives the published event (same object)",
    )
    check(received[0].seq == 1, "first event is seq 1")
    check(received[0].event_type == "run_completed", "the type is readable unwrapped")

    late: list[Sequenced] = []
    bus.subscribe(late.append)
    second = RunCompleted(run_id=2, done_steps=4)
    bus.publish(second)
    check(
        [item.event for item in received] == [first, second],
        "existing subscriber sees later events too",
    )
    check(received[-1].seq == 2, "sequence numbers increase by one")
    check(
        [item.event for item in late] == [second],
        "late subscriber sees live events only, not the ring",
    )

    subscription.close()
    bus.publish(RunCompleted(run_id=3, done_steps=5))
    check(len(received) == 2, "closed subscription stops receiving")
    subscription.close()  # idempotent
    check(len(late) == 2, "other subscribers unaffected by someone else's close")


def test_handler_failure_is_isolated() -> None:
    print("\n== one bad handler must not break the rest ==")
    bus = CallbackEventBus()
    good: list[object] = []

    def explode(event) -> None:
        raise RuntimeError("handler bug")

    bus.subscribe(explode)
    bus.subscribe(good.append)
    event = RunCompleted(run_id=1, done_steps=1)
    try:
        bus.publish(event)
        check(
            [item.event for item in good] == [event],
            "remaining handlers still ran after a failure",
        )
    except RuntimeError:
        check(False, "publish must not propagate handler exceptions")


def test_thread_safety() -> None:
    print("\n== concurrent publishers from multiple threads ==")
    bus = CallbackEventBus()
    lock = Lock()
    collected: list[object] = []

    def handler(sequenced: Sequenced) -> None:
        with lock:
            collected.append(sequenced)

    bus.subscribe(handler)

    threads_per_worker = 250

    def worker(worker_id: int) -> None:
        for index in range(threads_per_worker):
            bus.publish(RunsDeleted(deleted=worker_id * 1000 + index))

    threads = [Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    check(
        len(collected) == 4 * threads_per_worker,
        f"all 1000 events delivered exactly once (got {len(collected)})",
    )
    check(
        len({item.event.deleted for item in collected}) == 4 * threads_per_worker,
        "no event was duplicated",
    )
    check(
        len({item.seq for item in collected}) == 4 * threads_per_worker,
        "no sequence number was handed out twice",
    )
    check(
        {item.seq for item in collected} == set(range(1, 4 * threads_per_worker + 1)),
        "the sequence numbers form one gapless run despite four threads",
    )


def main() -> None:
    test_pubsub_lifecycle()
    test_handler_failure_is_isolated()
    test_thread_safety()
    finish()


if __name__ == "__main__":
    main()
