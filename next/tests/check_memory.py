"""Memory runtime contract: the limit is settled honestly, pressure is answered between steps, nothing moves that must stay.

    python -m next.tests.check_memory

Uses a scripted fake device, so the numbers are exact: what the driver reports, what the allocator holds, who moves
off the device and in which order, and the refusals (floor, impossible budget, going over a strict budget).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..engine.runtime.memory import MB, MemoryBudgetExceeded, MemoryPolicy, ResidencyManager
from ..engine.runtime.residents import DeviceResident
from .check_engine import rig
from .support import check, raises, run


class FakeDevice:
    """Reports scripted driver numbers; ``foreign`` is memory other processes hold."""

    def __init__(self, total=12000.0, foreign=600.0, ours=0.0):
        self.total, self.foreign, self.ours, self.cache_emptied = total, foreign, ours, 0

    def memory(self):
        return {"device_total": self.total, "device_used": self.foreign + self.ours, "reserved": self.ours}

    def empty_cache(self):
        self.cache_emptied += 1

    def synchronize(self):
        pass


class Blob(DeviceResident):
    def __init__(self, device: FakeDevice, mb: float):
        self.device, self.mb, self.on_device = device, mb, True

    def footprint_bytes(self):
        return int(self.mb * MB) if self.on_device else 0

    def offload(self):
        self.on_device = False
        self.device.ours -= self.mb

    def reload(self, device=None):
        self.on_device = True
        self.device.ours += self.mb

    def release(self):
        pass


def manager(device, **policy):
    return ResidencyManager(device, MemoryPolicy(**policy))


def limits():
    device = FakeDevice(ours=2000)
    m = manager(device)
    m.start()
    check(m.limit_mb == 12000 - 600, f"auto budget is the card minus other processes ({m.limit_mb})")
    raises(MemoryBudgetExceeded, manager(device, limit_mb=12000).start, "a budget the card cannot give is refused", "can give")
    raises(MemoryBudgetExceeded, manager(device, floor_mb=11500).start, "a floor above what is free is refused", "floor")
    loose = manager(device, limit_mb=12000, strict=False)
    loose.start()
    check(loose.limit_mb == 12000, "a non-strict budget is taken as given")
    cpu = ResidencyManager(type("Cpu", (), {"memory": lambda self: {}, "empty_cache": lambda self: None})(), MemoryPolicy())
    cpu.start()
    cpu.after_step(0)
    check(cpu.limit_mb is None and cpu.used_mb() is None, "without driver numbers the manager measures nothing and never interferes")


def pressure():
    device = FakeDevice(ours=9000)
    model, encoder, opt = Blob(device, 5000), Blob(device, 1000), Blob(device, 3000)
    m = manager(device, limit_mb=11000, empty_cache_every=0, safety_margin=0.1, calibration_steps=3)
    m.register("model", model, rank=10, pinned=True)
    m.register("encoder", encoder, rank=0)
    m.register("optimizer", opt, rank=1)
    m.start()
    m.after_step(0)  # 9600 MB in use, ceiling 9900: fine
    check(m.empty_cache_every == 0 and not m.trace[1:], "no reaction below the ceiling")
    device.ours = 9600
    m.after_step(1)
    check(m.empty_cache_every == 1, "first lever: empty the allocator cache every step")
    m.after_step(2)
    check(encoder.on_device is False and m.trace[-1]["name"] == "park encoder between steps", "second lever parks the lowest-ranked resident")
    check(device.cache_emptied >= 1, "the cache was emptied after the step")
    m.before_step(3)
    check(encoder.on_device, "a parked resident comes back before the next step")
    device.ours = 9600
    m.after_step(3)
    check(not [t for t in m.trace if t["event"] == "lever"][2:], "after calibration no further lever is pulled")
    check(model.on_device, "a pinned resident never moves")
    device.ours = 11500
    raises(MemoryBudgetExceeded, lambda: m.after_step(4), "going over a strict budget stops the run", "budget")


def making_room():
    device = FakeDevice(ours=9000)
    model, encoder, opt = Blob(device, 5000), Blob(device, 1000), Blob(device, 3000)
    m = manager(device, limit_mb=11000)
    for name, blob, rank, pinned in (("model", model, 10, True), ("encoder", encoder, 0, False), ("optimizer", opt, 1, False)):
        m.register(name, blob, rank=rank, pinned=pinned)
    m.start()
    moved = m.make_room(800, "decode a preview")
    check(moved == ["encoder"] and not encoder.on_device and opt.on_device, f"makes room by moving the lowest rank first ({moved})")
    m.restore(moved)
    check(encoder.on_device, "restore brings residents back")
    raises(MemoryBudgetExceeded, lambda: m.make_room(5000, "huge"), "refuses when even moving everything movable is not enough", "do not fit")


def training():
    with tempfile.TemporaryDirectory() as raw:
        device = FakeDevice(ours=5000)
        m = manager(device, limit_mb=11000, empty_cache_every=1)
        m.start()
        trainer, *_ = rig(Path(raw), steps=2, residency=m)
        trainer.emit = lambda e, log=[]: log.append(e)
        trainer.run()
        check(device.cache_emptied == 2 and m.steps_seen == 2, "the trainer calls the manager around every step")


if __name__ == "__main__":
    run(limits, pressure, making_room, training)
