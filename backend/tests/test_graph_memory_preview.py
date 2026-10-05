"""PreviewGraphMemory -- what a run would need, before it runs.

The panel's whole value is that it cannot lie: a "fits" here must mean
the run endpoint would admit it, and a refusal preview must carry the
same numbers the real 409 does. So these check the arithmetic against
the ledger directly, and the edge cases that would make it lie --
unknown fingerprint, never-measured peak, unknown demand, no ledger --
are the ones pinned hardest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.graph_peak_source import ObservedPeak
from backend.application.memory_ledger import MemoryLedger
from backend.application.use_cases.preview_graph_memory import (
    VERDICT_DOES_NOT_FIT,
    VERDICT_EXPLORATORY,
    VERDICT_FITS,
    VERDICT_UNKNOWN,
    PreviewGraphMemory,
)
from backend.domain.graph import GraphDefinition
from backend.domain.memory_settings import MemorySettings
from backend.tests.support import check, finish

#: The card this project's admission numbers were measured on.
TOTAL_MB = 12216.0
FOREIGN_MB = 1024.0
OVERHEAD_MB = 600.0


def _ledger() -> MemoryLedger:
    return MemoryLedger(
        total_mb=TOTAL_MB,
        foreign_reserve_mb=FOREIGN_MB,
        process_overhead_mb=OVERHEAD_MB,
    )


def _graph(**memory) -> GraphDefinition:
    return GraphDefinition(memory=MemorySettings(**memory))


def _preview(ledger, peak_mb, fingerprint="fp-1"):
    return PreviewGraphMemory(
        memory_ledger=lambda: ledger,
        peak_source=lambda _graph: ObservedPeak(fingerprint, peak_mb),
    )


def test_a_measured_peak_that_fits() -> None:
    print("-- a stated demand that fits: fits, with the numbers --")
    preview = _preview(_ledger(), peak_mb=None)
    result = preview.execute(_graph(vram_max_mb=6000.0))
    check(result.verdict == VERDICT_FITS,
          f"verdict is fits (got {result.verdict}: {result.reason})")
    # 6000 allocator + 600 overhead is the device demand the ledger
    # compares against 11,192 MB of free capacity.
    check(result.device_demand_mb == 6600.0,
          f"device demand is allocator plus the per-process overhead "
          f"(got {result.device_demand_mb})")
    check(result.demand_mb == 6000.0 and result.demand_source == "stated",
          f"and the allocator demand is stated (got {result.demand_source})")
    check(result.peak_mb is None,
          f"a never-measured peak stays None, not 0.0 (got {result.peak_mb})")
    check(result.fingerprint_key == "fp-1",
          f"the fingerprint is carried through (got {result.fingerprint_key})")


def test_a_measured_peak_that_does_not_fit() -> None:
    print("-- a demand larger than the free card: refused, with the gap --")
    preview = _preview(_ledger(), peak_mb=None)
    result = preview.execute(_graph(vram_max_mb=11000.0))
    check(result.verdict == VERDICT_DOES_NOT_FIT,
          f"verdict is does_not_fit (got {result.verdict})")
    check(result.reason is not None and "11600" in result.reason,
          f"the reason names the device demand (got {result.reason!r})")
    check(result.reason is not None and "11192" in result.reason,
          f"and what is free (got {result.reason!r})")


def test_the_remembered_peak_feeds_the_demand() -> None:
    print("-- a remembered peak is what admission would use --")
    preview = _preview(_ledger(), peak_mb=7666.0)
    result = preview.execute(_graph())
    check(result.peak_mb == 7666.0,
          f"the remembered peak is reported (got {result.peak_mb})")
    # No stated max, so the observed peak becomes the demand: the panel
    # must quote the same number the run will be admitted against --
    # which includes effective_memory's 150 MB pillow on an observed
    # peak (7,666 measured + 150 = 7,816).
    check(result.demand_mb == 7816.0 and result.demand_source == "observed",
          f"and it becomes the observed demand, pillow included "
          f"(got {result.demand_source}, {result.demand_mb})")
    check(result.exploratory is False,
          "a measured configuration is not exploratory")


def test_an_unmeasured_configuration_is_exploratory_not_fits() -> None:
    print("-- never measured: exploratory, which is neither fits nor no --")
    preview = _preview(_ledger(), peak_mb=None)
    result = preview.execute(_graph())
    check(result.verdict == VERDICT_EXPLORATORY,
          f"verdict is exploratory (got {result.verdict})")
    check(result.exploratory is True,
          f"and it says so (got {result.exploratory})")
    check(result.reason is not None and "never been measured" in result.reason,
          f"with the reason a human can act on (got {result.reason!r})")


def test_an_unknown_fingerprint_is_named_as_unknown() -> None:
    print("-- a fingerprint that cannot be computed: explicit unknown --")
    preview = PreviewGraphMemory(
        memory_ledger=lambda: _ledger(),
        peak_source=lambda _graph: ObservedPeak(None, None),
    )
    result = preview.execute(_graph(vram_max_mb=6000.0))
    check(result.fingerprint_key is None,
          f"the fingerprint is None, not a fabricated key "
          f"(got {result.fingerprint_key!r})")
    check(result.peak_mb is None,
          f"and with no key there is no remembered peak (got {result.peak_mb})")
    # The stated demand still answers the fit question even so.
    check(result.verdict == VERDICT_FITS,
          f"a stated demand still gives a real verdict (got {result.verdict})")


def test_no_ledger_is_unknown_rather_than_fits() -> None:
    print("-- no ledger: unknown, never a reassuring fits --")
    preview = PreviewGraphMemory(
        memory_ledger=lambda: None,
        peak_source=lambda _graph: ObservedPeak("fp-1", 7666.0),
    )
    result = preview.execute(_graph(vram_max_mb=6000.0))
    check(result.verdict == VERDICT_UNKNOWN,
          f"verdict is unknown (got {result.verdict})")
    check(result.device_demand_mb is None and result.capacity_mb is None,
          "and every ledger-derived number is None rather than zero")
    check(result.peak_mb == 7666.0,
          f"while the remembered peak is still reported (got {result.peak_mb})")


def test_holders_come_from_the_live_ledger() -> None:
    print("-- the panel's 'who holds the card' is the ledger's own --")
    ledger = _ledger()
    ledger.reserve(owner="task:ingest", demand_mb=4000.0, exploratory=False)
    preview = _preview(ledger, peak_mb=None)
    result = preview.execute(_graph(vram_max_mb=6000.0))
    check(result.held_mb == 4000.0,
          f"the held total is reported (got {result.held_mb})")
    check("task:ingest" in result.holders,
          f"and the holder is named (got {list(result.holders)})")
    check(result.capacity_mb == 11192.0,
          f"capacity is total minus the foreign reserve (got {result.capacity_mb})")


def main() -> None:
    test_a_measured_peak_that_fits()
    test_a_measured_peak_that_does_not_fit()
    test_the_remembered_peak_feeds_the_demand()
    test_an_unmeasured_configuration_is_exploratory_not_fits()
    test_an_unknown_fingerprint_is_named_as_unknown()
    test_no_ledger_is_unknown_rather_than_fits()
    test_holders_come_from_the_live_ledger()
    finish()


if __name__ == "__main__":
    main()