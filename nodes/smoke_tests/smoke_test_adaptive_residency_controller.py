"""Checks nodes/train/managed.py's AdaptiveResidencyController in
isolation -- the actual decision logic (calibrate, then release nothing
if it was never needed, else release smallest-footprint-first until the
estimate fits; keep watching afterward and escalate if a later step's
own peak exceeds budget; a configurable safety margin shrinks the
effective ceiling before any of that math runs) -- directly, with fake
numbers, rather than only through a full ManagedLoRATrainerNode.build()
run (which can't exercise the "release something" branch at all in this
sandbox: DeviceContext.for_device() on a CPU tensor returns
_NullDeviceContext, whose memory_stats() is always None -- see this
class's own record_step_peak() docstring for why that's actually the
*correct* behavior to fall back to, not just a testing limitation, and
smoke_test_managed_trainer.py's own
check_ensure_loaded_always_fires_but_release_does_not_when_calibration_
cannot_resolve for the integration-level confirmation of that fallback).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.memory.handle import DeviceResident
from nodes.train.managed import AdaptiveResidencyController


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


class _FakeResident(DeviceResident):
    def __init__(self, footprint_mb: float):
        self._footprint_bytes = int(footprint_mb * 1024 ** 2)

    def footprint_bytes(self) -> int:
        return self._footprint_bytes

    def offload(self) -> None:
        pass

    def reload(self, device=None) -> None:
        pass

    def release(self) -> None:
        pass


def check_stays_resident_when_measured_peak_already_fits():
    print("[peak comfortably under budget: decides to release nothing, the exact "
          "real-world case that motivated this class -- see its own docstring]")
    controller = AdaptiveResidencyController(
        usable_mb=12000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=3,
        safety_margin=0.0,
    )
    for _ in range(3):
        check(controller.calibrating, "must still be calibrating before 3 steps are recorded")
        controller.record_step_peak({"peak_reserved_mb": 9000.0})
    check(not controller.calibrating, "must have decided after calibration_steps calls")
    check(controller.should_release("optimizer") is False, "9000MB already fits 12000MB")
    check(controller.should_release("text_encoder") is False, "9000MB already fits 12000MB")
    print("    PASS")


def check_releases_smallest_footprint_first_until_the_estimate_fits():
    print("[peak over budget: releases starting from the smallest footprint, only as "
          "many as the shortfall actually needs -- \"less performance costly options "
          "first\"]")
    controller = AdaptiveResidencyController(
        usable_mb=8000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=1,
        safety_margin=0.0,
    )
    # Peak 9000MB, usable 8000MB -- releasing optimizer alone (500MB) brings the
    # estimate to 8500MB, still over; releasing text_encoder too (1600MB) brings it
    # to 6900MB, under. Both must be selected, optimizer first by size.
    controller.record_step_peak({"peak_reserved_mb": 9000.0})
    check(controller.should_release("optimizer") is True, "smallest candidate, must go first")
    check(controller.should_release("text_encoder") is True,
          "optimizer alone (500MB) isn't enough to bring 9000MB under 8000MB")
    print("    PASS")


def check_releases_only_the_smallest_when_that_alone_is_enough():
    print("[peak over budget, but only barely: releasing just the smaller candidate is "
          "enough -- the larger, more expensive one must be left alone]")
    controller = AdaptiveResidencyController(
        usable_mb=8800.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=1,
        safety_margin=0.0,
    )
    # Peak 9000MB, usable 8800MB -- releasing optimizer (500MB) alone brings it to
    # 8500MB, already under. text_encoder, the more expensive one, must stay resident.
    controller.record_step_peak({"peak_reserved_mb": 9000.0})
    check(controller.should_release("optimizer") is True, "500MB alone is enough")
    check(controller.should_release("text_encoder") is False,
          "must not release the larger candidate once the smaller one was already enough")
    print("    PASS")


def check_releases_everything_when_even_that_is_not_enough():
    print("[peak far over budget: releases every candidate, still just an estimate -- "
          "before_step()'s own reactive check is the real backstop past this point, "
          "not this controller]")
    controller = AdaptiveResidencyController(
        usable_mb=1000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=1,
        safety_margin=0.0,
    )
    controller.record_step_peak({"peak_reserved_mb": 9000.0})
    check(controller.should_release("optimizer") is True, "must release everything it can")
    check(controller.should_release("text_encoder") is True, "must release everything it can")
    print("    PASS")


def check_decides_immediately_when_there_is_no_usable_ceiling():
    print("[usable_mb=None (ResourceControlHandle.usable_budget_mb() returned None -- "
          "no fixed-ceiling implementation): decides on the very first call, doesn't "
          "wait calibration_steps for a number that will never come]")
    controller = AdaptiveResidencyController(
        usable_mb=None,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=3,
    )
    controller.record_step_peak({"peak_reserved_mb": 9000.0})
    check(not controller.calibrating, "must decide immediately, not after 3 calls")
    check(controller.should_release("optimizer") is False, "nothing to plan against -- stay resident")
    check(controller.should_release("text_encoder") is False, "nothing to plan against -- stay resident")
    print("    PASS")


def check_decides_immediately_when_memory_stats_is_none():
    print("[memory_stats()=None (no device-memory concept at all, e.g. CPU): same "
          "immediate-decide fallback as usable_mb=None -- see this class's own "
          "record_step_peak() docstring for why both cases collapse to the same answer]")
    controller = AdaptiveResidencyController(
        usable_mb=8000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=3,
    )
    controller.record_step_peak(None)
    check(not controller.calibrating, "must decide immediately on a None reading")
    check(controller.should_release("optimizer") is False, "nothing to plan against -- stay resident")
    check(controller.should_release("text_encoder") is False, "nothing to plan against -- stay resident")
    print("    PASS")


def check_uses_the_max_peak_seen_across_calibration_not_the_last_one():
    print("[calibration takes the max reading across every step, not just the most "
          "recent -- a single unusually-high step must not get averaged away]")
    controller = AdaptiveResidencyController(
        usable_mb=8000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=3,
        safety_margin=0.0,
    )
    controller.record_step_peak({"peak_reserved_mb": 3000.0})
    controller.record_step_peak({"peak_reserved_mb": 9000.0})  # the real high-water mark
    controller.record_step_peak({"peak_reserved_mb": 4000.0})
    check(controller.should_release("optimizer") is True,
          "must have used 9000MB (the max), not 4000MB (the last), to decide")
    print("    PASS")


def check_escalates_when_a_later_steps_own_peak_exceeds_budget():
    print("[the actual new behavior: a later step's own peak (post-decision) exceeding "
          "budget escalates -- adds one more candidate to the release set -- rather than "
          "being ignored. The real motivating report: a variable-resolution dataset "
          "where calibration_steps happened to sample smaller images, so \"stay "
          "resident\" looked right until a genuinely larger one showed up later and "
          "OOM'd]")
    controller = AdaptiveResidencyController(
        usable_mb=12000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=1, safety_margin=0.0,
    )
    controller.record_step_peak({"peak_reserved_mb": 5000.0})  # decides: release nothing
    check(controller.should_release("optimizer") is False, "sanity: decided to stay resident")
    check(controller.should_release("text_encoder") is False, "sanity: decided to stay resident")

    controller.record_step_peak({"peak_reserved_mb": 13000.0})  # a later, larger step
    check(controller.should_release("optimizer") is True,
          "must have escalated to releasing the smallest candidate")
    check(controller.should_release("text_encoder") is False,
          "one candidate (500MB) is enough to bring 13000MB back under 12000MB")
    print("    PASS")


def check_escalation_does_not_reverse_once_usage_drops_back_down():
    print("[no de-escalation: once something's been added to the release set, a later, "
          "smaller reading must not remove it again -- avoids thrashing between "
          "resident/released every time usage happens to dip]")
    controller = AdaptiveResidencyController(
        usable_mb=12000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=1, safety_margin=0.0,
    )
    controller.record_step_peak({"peak_reserved_mb": 13000.0})  # decides: release optimizer
    check(controller.should_release("optimizer") is True, "sanity: escalated once")
    controller.record_step_peak({"peak_reserved_mb": 4000.0})  # a later, much smaller step
    check(controller.should_release("optimizer") is True,
          "must still be releasing optimizer -- a smaller reading must not undo it")
    print("    PASS")


def check_escalates_through_every_candidate_if_it_has_to():
    print("[repeated escalation: each new over-budget reading adds the next-smallest "
          "not-yet-released candidate, until either the budget is honored or nothing is "
          "left to escalate to]")
    controller = AdaptiveResidencyController(
        usable_mb=12000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=1, safety_margin=0.0,
    )
    controller.record_step_peak({"peak_reserved_mb": 5000.0})  # release nothing
    controller.record_step_peak({"peak_reserved_mb": 20000.0})  # way over -- needs both
    check(controller.should_release("optimizer") is True, "must escalate the first candidate")
    # Still over budget even with optimizer released (19500MB > 12000MB) -- must escalate
    # again on the *next* over-budget reading rather than adding both at once, since each
    # record_step_peak() call only knows about "the current release set wasn't enough".
    controller.record_step_peak({"peak_reserved_mb": 19500.0})
    check(controller.should_release("text_encoder") is True, "must escalate the second candidate too")
    print("    PASS")


def check_safety_margin_shrinks_the_effective_usable_ceiling():
    print("[safety_margin=0.1 (the real default): a 12000MB usable ceiling is really "
          "treated as 10800MB -- headroom for a somewhat-larger-than-calibrated step to "
          "still fit without needing to escalate at all]")
    controller = AdaptiveResidencyController(
        usable_mb=12000.0,
        candidates={"optimizer": _FakeResident(500), "text_encoder": _FakeResident(1600)},
        calibration_steps=1, safety_margin=0.1,
    )
    # 11000MB genuinely fits the raw 12000MB ceiling, but not 12000 * 0.9 = 10800MB.
    controller.record_step_peak({"peak_reserved_mb": 11000.0})
    check(controller.should_release("optimizer") is True,
          "must have compared against the margined 10800MB, not the raw 12000MB")
    print("    PASS")


def main():
    check_stays_resident_when_measured_peak_already_fits()
    check_releases_smallest_footprint_first_until_the_estimate_fits()
    check_releases_only_the_smallest_when_that_alone_is_enough()
    check_releases_everything_when_even_that_is_not_enough()
    check_decides_immediately_when_there_is_no_usable_ceiling()
    check_decides_immediately_when_memory_stats_is_none()
    check_uses_the_max_peak_seen_across_calibration_not_the_last_one()
    check_escalates_when_a_later_steps_own_peak_exceeds_budget()
    check_escalation_does_not_reverse_once_usage_drops_back_down()
    check_escalates_through_every_candidate_if_it_has_to()
    check_safety_margin_shrinks_the_effective_usable_ceiling()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
