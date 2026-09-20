"""Checks nodes/memory/control_handle.py's BudgetedResourceControlHandle
-- specifically the two additions this session made (see that module's
own docstring): strict-mode raising, and an explicit
DeviceContext.synchronize() around every offload/reload transition.
No existing smoke test exercised BudgetedResourceControlHandle's real
_make_room() logic at all before this file (checked directly: grep for
ResourceControlHandle/BudgetedResourceControlHandle across
nodes/smoke_tests/ turned up only smoke_test_text_encoder_cache.py's
own hand-rolled ABC fake, never the real implementation) -- a second
real gap found alongside the missing LoRATrainingConfigNode test (see
smoke_test_lora_training_config.py's own docstring).

DeviceContext.for_device() only ever returns a real XPU/CUDA/null
context based on a device string -- there was no way to script
memory_stats() into a real caller until this session added an optional
device_ctx= constructor parameter specifically to make this file
possible (BudgetedResourceControlHandle.__init__'s own docstring
comment). _FakeDeviceContext below is that script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.components.device import DeviceContext
from nodes.memory.control_handle import BudgetedResourceControlHandle
from nodes.memory.handle import DeviceResident
from nodes.resource_budget import ResourceBudget


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


class _FakeDeviceContext(DeviceContext):
    """A scripted sequence of memory_stats() reserved_mb readings, one
    per call, the last value repeating once the script runs out (so a
    test doesn't have to predict exactly how many times _make_room()
    will re-measure). Counts synchronize() calls -- the actual thing
    under test in most of this file."""

    def __init__(self, reserved_mb_sequence: list[float]):
        self._sequence = list(reserved_mb_sequence)
        self._calls = 0
        self.synchronize_call_count = 0

    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        self.synchronize_call_count += 1

    def memory_stats(self):
        value = self._sequence[min(self._calls, len(self._sequence) - 1)]
        self._calls += 1
        return {"reserved_mb": value}


class _FakeResident(DeviceResident):
    def __init__(self, name: str):
        self.name = name
        self.offload_calls = 0
        self.reload_calls = 0

    def footprint_bytes(self) -> int:
        return 0

    def offload(self) -> None:
        self.offload_calls += 1

    def reload(self, device=None) -> None:
        self.reload_calls += 1

    def release(self) -> None:
        pass


def check_offloading_brings_usage_back_under_budget():
    print("[before_step(): offloads registered-offloadable residents, in registration "
          "order, until measured reserved_mb is back under budget]")
    # 1000MB reserved to start (over the 500MB usable budget below), drops to 400MB
    # after offloading "a" -- "b" should never be touched.
    device_ctx = _FakeDeviceContext([1000.0, 400.0])
    budget = ResourceBudget(vram_budget_mb=600.0, vram_reserve_mb=100.0)  # usable = 500MB
    control = BudgetedResourceControlHandle(budget, device="cpu", device_ctx=device_ctx)
    a, b = _FakeResident("a"), _FakeResident("b")
    control.register("a", a, offloadable=True)
    control.register("b", b, offloadable=True)

    control.before_step(0)

    check(a.offload_calls == 1, a.offload_calls)
    check(b.offload_calls == 0, "must stop offloading once back under budget")
    check(device_ctx.synchronize_call_count == 1,
          "must synchronize() exactly once, right after the one offload() call")
    print("    PASS")


def check_non_offloadable_resident_is_never_touched():
    print("[a resident registered offloadable=False is never offloaded, even under "
          "sustained pressure]")
    device_ctx = _FakeDeviceContext([1000.0])  # never drops -- nothing offloadable to move
    budget = ResourceBudget(vram_budget_mb=600.0, vram_reserve_mb=100.0)
    control = BudgetedResourceControlHandle(budget, device="cpu", device_ctx=device_ctx)
    pinned = _FakeResident("pinned")
    control.register("pinned", pinned, offloadable=False)

    control.before_step(0)  # strict=False (default): must not raise

    check(pinned.offload_calls == 0, pinned.offload_calls)
    print("    PASS")


def check_strict_false_is_todays_behavior_unchanged():
    print("[strict=False (default): still over budget after exhausting every "
          "offloadable resident -- returns quietly, doesn't raise]")
    device_ctx = _FakeDeviceContext([1000.0, 700.0])  # offloads "a", still over 500MB usable
    budget = ResourceBudget(vram_budget_mb=600.0, vram_reserve_mb=100.0, strict=False)
    control = BudgetedResourceControlHandle(budget, device="cpu", device_ctx=device_ctx)
    a = _FakeResident("a")
    control.register("a", a, offloadable=True)

    control.before_step(0)  # must not raise

    check(a.offload_calls == 1, a.offload_calls)
    print("    PASS")


def check_strict_true_raises_when_budget_cannot_be_honored():
    print("[strict=True: raises RuntimeError instead of continuing, once nothing "
          "offloadable is left and usage is still over budget]")
    device_ctx = _FakeDeviceContext([1000.0, 700.0])  # same scenario, strict=True this time
    budget = ResourceBudget(vram_budget_mb=600.0, vram_reserve_mb=100.0, strict=True)
    control = BudgetedResourceControlHandle(budget, device="cpu", device_ctx=device_ctx)
    a = _FakeResident("a")
    control.register("a", a, offloadable=True)

    raised = False
    try:
        control.before_step(0)
    except RuntimeError as e:
        raised = True
        check("700" in str(e) and "500" in str(e),
              f"error message should name the actual and usable MB values: {e}")
    check(raised, "expected before_step() to raise RuntimeError")
    check(a.offload_calls == 1, "must still have offloaded everything it could before raising")
    print("    PASS")


def check_strict_true_does_not_raise_when_budget_is_actually_honored():
    print("[strict=True: no raise when offloading actually brings usage back under budget]")
    device_ctx = _FakeDeviceContext([1000.0, 400.0])
    budget = ResourceBudget(vram_budget_mb=600.0, vram_reserve_mb=100.0, strict=True)
    control = BudgetedResourceControlHandle(budget, device="cpu", device_ctx=device_ctx)
    a = _FakeResident("a")
    control.register("a", a, offloadable=True)

    control.before_step(0)  # must not raise

    check(a.offload_calls == 1, a.offload_calls)
    print("    PASS")


def check_ensure_loaded_reloads_and_synchronizes():
    print("[ensure_loaded(): reload()s an offloaded resident and synchronize()s "
          "before returning]")
    device_ctx = _FakeDeviceContext([400.0])  # under budget throughout -- nothing to offload
    budget = ResourceBudget(vram_budget_mb=600.0, vram_reserve_mb=100.0)
    control = BudgetedResourceControlHandle(budget, device="cpu", device_ctx=device_ctx)
    a = _FakeResident("a")
    control.register("a", a, offloadable=True)
    control.before_step(0)
    check(a.offload_calls == 0, "under budget -- must not have offloaded anything yet")

    # Manually offload, matching what before_step() would have done under real
    # pressure -- isolates ensure_loaded()'s own reload path from _make_room()'s.
    control._coordinator.offload("a")
    control._offloaded.add("a")
    sync_before = device_ctx.synchronize_call_count

    control.ensure_loaded("a")

    check(a.reload_calls == 1, a.reload_calls)
    check(device_ctx.synchronize_call_count > sync_before,
          "ensure_loaded() must synchronize() after reload(), before returning")
    check("a" not in control._offloaded, "must no longer be tracked as offloaded")
    print("    PASS")


def check_device_ctx_none_uses_real_factory_unchanged():
    print("[device_ctx omitted: falls back to the real DeviceContext.for_device(device), "
          "same as before this session's constructor change]")
    budget = ResourceBudget(vram_budget_mb=600.0, vram_reserve_mb=100.0)
    control = BudgetedResourceControlHandle(budget, device="cpu")
    check(type(control._device_ctx).__name__ == "_NullDeviceContext",
          type(control._device_ctx).__name__)
    control.before_step(0)  # _NullDeviceContext.memory_stats() -> None: must be a cheap no-op
    print("    PASS")


def check_release_offloads_unconditionally_and_rejects_non_offloadable():
    print("[release(): offloads a registered-offloadable resident regardless of "
          "current pressure, synchronizes, and is idempotent; raises for a resident "
          "registered offloadable=False instead of silently no-op-ing]")
    device_ctx = _FakeDeviceContext([100.0])  # well under budget -- no pressure at all
    budget = ResourceBudget(vram_budget_mb=600.0, vram_reserve_mb=100.0)
    control = BudgetedResourceControlHandle(budget, device="cpu", device_ctx=device_ctx)
    a, pinned = _FakeResident("a"), _FakeResident("pinned")
    control.register("a", a, offloadable=True)
    control.register("pinned", pinned, offloadable=False)

    control.release("a")
    check(a.offload_calls == 1, "must offload even though usage is nowhere near budget")
    check(device_ctx.synchronize_call_count == 1, device_ctx.synchronize_call_count)

    control.release("a")  # idempotent -- already offloaded
    check(a.offload_calls == 1, "must not offload a second time")

    raised = False
    try:
        control.release("pinned")
    except ValueError:
        raised = True
    check(raised, "expected release() to raise for a resident registered offloadable=False")
    check(pinned.offload_calls == 0, "must not have touched it before raising")
    print("    PASS")


def main():
    check_offloading_brings_usage_back_under_budget()
    check_non_offloadable_resident_is_never_touched()
    check_strict_false_is_todays_behavior_unchanged()
    check_strict_true_raises_when_budget_cannot_be_honored()
    check_strict_true_does_not_raise_when_budget_is_actually_honored()
    check_ensure_loaded_reloads_and_synchronizes()
    check_device_ctx_none_uses_real_factory_unchanged()
    check_release_offloads_unconditionally_and_rejects_non_offloadable()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
