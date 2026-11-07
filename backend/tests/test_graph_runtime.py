"""ContractGraphRuntime -- the GraphRuntime port over the node tree.

The validator's issue codes and the runner's own behaviour are the tree's (next/tests); this file checks what the
adapter adds: the port's issue and result types, the run settings the server decided, skipped nodes left out of the
reported results, node failures as outcomes, the stop between nodes, and the memory release.

Run directly: python backend/tests/test_graph_runtime.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.ports.graph_runtime import IssueSeverity
from backend.infrastructure.graph.runtime import ContractGraphRuntime
from backend.tests.support import check, finish, fixture_tree_registry, gdef, gedge, gnode, uid

releases = {"n": 0}


def release() -> None:
    releases["n"] += 1


runtime = ContractGraphRuntime(fixture_tree_registry(), releaser=release)


def sum_graph(**settings):
    return gdef(
        [gnode("one", "SumNode", a=1.0, b=2.0), gnode("two", "ScaleNode", factor=3.0)],
        [gedge("one", "sum", "two", "value")],
        settings,
    )


# 1. Validation: the tree's issues arrive as the port's.
check(not [i for i in runtime.validate(sum_graph()) if IssueSeverity(i.severity).blocks], "a sound graph has no errors")
issues = runtime.validate(gdef([gnode("x", "NoSuchNode")]))
check([(i.code, i.node_id, IssueSeverity(i.severity)) for i in issues]
      == [("unknown_class", uid("x"), IssueSeverity.ERROR)], f"an unknown class names its node by uid (got {issues})")
check(all(i.plane == "data" for i in issues), "an issue carries its plane")

# 2. A run: results in order, each described, reported as they finish.
seen = []
outcome = runtime.execute(sum_graph(), cancel_event=threading.Event(), on_node_done=seen.append)
check(outcome.error is None and [r.node_id for r in outcome.results] == [uid("one"), uid("two")],
      f"both nodes ran, in dependency order (got {[r.node_id for r in outcome.results]})")
by_id = {r.node_id: r for r in outcome.results}
check(by_id[uid("two")].outputs == {"scaled": 9.0}, f"outputs are the described values (got {by_id[uid('two')].outputs})")
check([r.node_id for r in seen] == [r.node_id for r in outcome.results], "the callback saw every node, in order")
check(releases["n"] == 1, f"memory was released once after the run (got {releases['n']})")

# 3. A failing node is an outcome, not an exception; nothing after it runs or is reported.
failed = runtime.execute(
    gdef([gnode("boom", "BoomNode"), gnode("after", "ScaleNode", value=1.0)]), cancel_event=threading.Event())
check(failed.error is not None and "boom" in failed.error, f"the failure is the outcome's error (got {failed.error})")
check([(r.node_id, r.ok) for r in failed.results if not r.ok] == [(uid("boom"), False)], "the failed node is in the results")

# 4. An invalid graph never reaches a node.
invalid = runtime.execute(gdef([gnode("x", "NoSuchNode")]), cancel_event=threading.Event())
check(invalid.results == () and invalid.error.startswith("graph invalid"), f"invalid is an outcome (got {invalid})")

# 5. Stopped before it started: every node is skipped, and skipped nodes are not reported.
stop = threading.Event()
stop.set()
stopped = runtime.execute(sum_graph(), cancel_event=stop)
check(stopped.error is None and stopped.results == (), f"a stop leaves nothing in the results (got {stopped})")

# 6. The server's settings win over the graph's own.
recorded = {}


class Spy(ContractGraphRuntime):
    def _runner(self, observer=None):
        runner = super()._runner(observer)
        original = runner.run

        def run(doc, cancel_event=None):
            recorded.update(doc.settings)
            return original(doc, cancel_event)

        runner.run = run
        return runner


Spy(fixture_tree_registry(), releaser=release, overrides={"memory.vram_max_mb": 4096.0}).execute(
    sum_graph(**{"memory.vram_max_mb": "auto"}), cancel_event=threading.Event())
check(recorded.get("memory.vram_max_mb") == 4096.0, f"the admitted budget replaced the graph's (got {recorded})")

# 7. release_memory is the injected releaser.
before = releases["n"]
runtime.release_memory()
check(releases["n"] == before + 1, "release_memory calls the releaser")

finish()
