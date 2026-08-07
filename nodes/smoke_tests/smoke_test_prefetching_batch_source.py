"""Correctness check for nodes/dataset/prefetch.py's PrefetchingBatchSource.

Ordinary iteration correctness matters less here than the concurrency
risks a background-thread decorator actually has (docs/training_pipeline_design.md
section 5.6): a consumer that stops early must not deadlock a worker
blocked inside queue.Queue.put(), and an exception from the wrapped
source must actually reach the consumer, not vanish into a dead daemon
thread. Both are exercised directly below, not just asserted safe by
reading the code.

Run this directly: `python nodes/smoke_tests/smoke_test_prefetching_batch_source.py`
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.dataset.handle import TrainingBatchSource
from nodes.dataset.prefetch import PrefetchingBatchSource

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class _ListSource(TrainingBatchSource):
    def __init__(self, items, invalidated_flag=None):
        self._items = items
        self._invalidated_flag = invalidated_flag

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def invalidate(self):
        if self._invalidated_flag is not None:
            self._invalidated_flag.append(True)


class _FailingSource(TrainingBatchSource):
    """Yields two real batches, then raises -- simulates a real I/O/decode
    error partway through a pass over the dataset."""

    def __iter__(self):
        yield {"n": 1}
        yield {"n": 2}
        raise RuntimeError("simulated dataset read failure")

    def __len__(self):
        return 3

    def invalidate(self):
        pass


def check_basic_order_and_contents_preserved():
    print("\n=== Batches come through in order, unchanged ===")
    items = [{"n": i} for i in range(10)]
    source = PrefetchingBatchSource(_ListSource(items), depth=2)
    got = list(source)
    record(got == items, "all 10 batches, same order, same contents", detail=str(got)[:80])


def check_len_and_invalidate_delegate():
    print("\n=== __len__/invalidate() delegate to the wrapped source ===")
    flag = []
    inner = _ListSource([{"n": 1}, {"n": 2}], invalidated_flag=flag)
    source = PrefetchingBatchSource(inner, depth=2)
    record(len(source) == 2, "__len__ delegates")
    source.invalidate()
    record(flag == [True], "invalidate() reached the inner source")


def check_multiple_iter_calls_each_get_a_fresh_pass():
    print("\n=== Two separate __iter__() calls each yield a full, independent pass ===")
    items = [{"n": i} for i in range(4)]
    source = PrefetchingBatchSource(_ListSource(items), depth=2)
    first_pass = list(source)
    second_pass = list(source)
    record(first_pass == items and second_pass == items,
           "both passes are complete and identical (matches FetchBatchPhase's "
           "epoch-wrap usage: iter(batches) called again after StopIteration)")


def check_early_abandonment_does_not_deadlock():
    """THE real risk: depth=2 with 100 items means the worker will block
    inside q.put() once it's 2 batches ahead of a consumer that only
    takes 3 and stops. If draining-on-abandonment (this class's __iter__
    finally block) didn't work, this test would hang forever instead of
    finishing -- a timeout-based thread join is the actual assertion."""
    print("\n=== Consumer stopping early doesn't deadlock the worker thread (real concurrency risk) ===")
    items = [{"n": i} for i in range(200)]
    source = PrefetchingBatchSource(_ListSource(items), depth=2)

    result = {"done": False}

    def _consume_a_few_then_stop():
        it = iter(source)
        for _ in range(3):
            next(it)
        it.close()  # explicit early abandonment, same as a for-loop `break`
        result["done"] = True

    t = threading.Thread(target=_consume_a_few_then_stop, daemon=True)
    t.start()
    t.join(timeout=5.0)
    record(result["done"], "consumer thread finished within 5s (no deadlock)",
           detail="still running -- worker thread is stuck" if not result["done"] else "")


def check_inner_exception_propagates_to_consumer():
    print("\n=== An exception from the wrapped source reaches the consumer, doesn't vanish ===")
    source = PrefetchingBatchSource(_FailingSource(), depth=2)
    got = []
    raised = None
    try:
        for batch in source:
            got.append(batch)
    except RuntimeError as e:
        raised = e
    record(got == [{"n": 1}, {"n": 2}], "the two real batches before the failure were delivered",
           detail=str(got))
    record(raised is not None and "simulated dataset read failure" in str(raised),
           "the RuntimeError itself reached the consumer, not just a premature stop",
           detail=repr(raised))


def check_depth_validation():
    print("\n=== depth < 1 is rejected at construction, not a confusing failure later ===")
    try:
        PrefetchingBatchSource(_ListSource([]), depth=0)
        ok = False
    except ValueError:
        ok = True
    record(ok, "depth=0 raises ValueError")


def main():
    check_basic_order_and_contents_preserved()
    check_len_and_invalidate_delegate()
    check_multiple_iter_calls_each_get_a_fresh_pass()
    check_early_abandonment_does_not_deadlock()
    check_inner_exception_propagates_to_consumer()
    check_depth_validation()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
