"""PeakRecord -- a remembered peak is what makes observed mode safe *before*
a run starts.

Without it, the only way to know a run's peak is to watch it happen, so the
reservation is only as good as the steps already done and a run that grows past
its reservation dies **mid-flight**. That is the worst place for a resource
failure and the reason this file exists.

Every number here is one this session measured on the B580: rank-64 LoRA at
1024 with checkpointing peaks at 7,666 MB at batch 2 and 8,954 MB at batch 4;
residents are constant at 5,611 MB; reserved drift across runs is 0-14 MB, which
is what the 150 MB pillow has to cover and why it is not larger.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

from nodes.memory.peak_record import Fingerprint, PeakRecord

CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"    PASS: {message}")


FP = Fingerprint(model="sdxl", batch_size=2, latent_h=128, latent_w=128,
                 rank=64, checkpointing=True, optimizer="adamw")
FP_BATCH4 = Fingerprint(model="sdxl", batch_size=4, latent_h=128, latent_w=128,
                        rank=64, checkpointing=True, optimizer="adamw")
FP_NO_CKPT = Fingerprint(model="sdxl", batch_size=2, latent_h=128, latent_w=128,
                         rank=64, checkpointing=False, optimizer="adamw")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="peak-record-")) / "peaks.json"
    record = PeakRecord(tmp)

    print("[an unmeasured configuration is unknown, not free]")
    check(record.reservation_mb(FP) is None,
          "nothing measured means no reservation, and None rather than 0 -- "
          "0 would be a claim that a run needs nothing")

    print("\n[the reservation is the remembered peak plus the pillow]")
    record.record(FP, 7_666.0)
    check(record.reservation_mb(FP) == 7_666.0 + record.pillow_mb,
          f"a measured 7,666 MB reserves {record.reservation_mb(FP):,.0f} MB, "
          f"so step 0 is covered rather than discovered")

    print("\n[peaks never fall]")
    record.record(FP, 3_000.0)
    check(record.known()[FP.key()] == 7_666.0,
          "a later smaller peak does not lower the record -- a peak is a "
          "high-water mark, and lowering it would admit the next run on a "
          "number this one already exceeded")
    record.record(FP, 8_954.0)
    check(record.known()[FP.key()] == 8_954.0,
          "a higher one raises it, so a changed configuration is picked up")

    print("\n[configurations are distinguished by what changes the peak]")
    check(record.reservation_mb(FP_BATCH4) is None,
          "batch 4 is a different configuration and has not been measured, "
          "even though batch 2 has")
    record.record(FP_BATCH4, 8_954.0)
    check(record.known()[FP.key()] == 8_954.0
          and record.known()[FP_BATCH4.key()] == 8_954.0,
          "and recording it leaves batch 2's own number alone")
    check(record.reservation_mb(FP_NO_CKPT) is None,
          "so does turning checkpointing off -- which is the single biggest "
          "lever on workspace and the one most worth not conflating")

    print("\n[the file is a cache, so a damaged one costs a re-measure]")
    check(not tmp.with_suffix(".json.tmp").exists()
          and list(tmp.parent.glob("*.tmp")) == [],
          "no temp file is left behind by a normal write")
    tmp.write_text("{ not json", encoding="utf-8")
    check(record.reservation_mb(FP) is None,
          "a corrupt record reads as unknown rather than refusing to run -- a "
          "measurement cache must not become a dependency")
    record.record(FP, 7_666.0)
    check(record.reservation_mb(FP) is not None, "and rewrites cleanly")

    print("\n[the record is readable by a person]")
    raw = json.loads(tmp.read_text(encoding="utf-8"))
    check(all(isinstance(v, (int, float)) for v in raw.values()),
          f"it is plain JSON of numbers, editable by hand: {sorted(raw)}")

    print("\n[forgetting is the escape hatch]")
    record.forget(FP_BATCH4)
    check(record.reservation_mb(FP_BATCH4) is None,
          "a forgotten configuration is unknown again, so the next run "
          "re-measures -- safe, because being wrong this way only makes "
          "admission slower")
    check(record.reservation_mb(FP) is not None,
          "and forgetting one does not touch the others")

    print("\n[the record survives the process, which is the whole point]")
    reopened = PeakRecord(tmp)
    check(reopened.reservation_mb(FP) == reopened.known()[FP.key()] + reopened.pillow_mb,
          "a second PeakRecord over the same file sees the first one's "
          "measurement -- without this, observed mode can only ever be a "
          "within-run ratchet and a run dies mid-flight")

    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()
