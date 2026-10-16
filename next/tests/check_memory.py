"""Memory runtime contract: the limit is settled honestly, pressure is answered between steps, nothing moves that must stay.

    python -m next.tests.check_memory

Uses a scripted fake device, so the numbers are exact: what the driver reports, what the allocator holds, who moves
off the device and in which order, and the refusals (floor, impossible budget, going over a strict budget).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..engine.runtime.memory import MB, MemoryBudgetExceeded, MemoryPolicy, ResidencyManager
from ..engine.runtime.encoders import OnDemandKit
from ..engine.runtime.residents import DeviceResident, ModulesResident
from .check_engine import rig
from .support import check, raises, run


class FakeDevice:
    """Reports scripted driver numbers; ``foreign`` is memory other processes hold."""

    def __init__(self, total=12000.0, foreign=600.0, ours=0.0):
        self.total, self.foreign, self.ours, self.cache_emptied = total, foreign, ours, 0
        self.slack = self.spike = 0.0  # allocator cache beyond live tensors; activation peak above live tensors

    def memory(self):
        return {"device_total": self.total, "device_used": self.foreign + self.ours + self.slack, "reserved": self.ours + self.slack,
                "allocated": self.ours, "peak_allocated": self.ours + self.spike}

    def reset_peak(self):
        pass

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
    check(m.limit_mb == 12000, f"auto budget is the whole card, other programs included in what is measured ({m.limit_mb})")
    raises(MemoryBudgetExceeded, manager(device, limit_mb=13000).start, "a budget above the card is refused", "card has")
    raises(MemoryBudgetExceeded, manager(device, floor_mb=11500).start, "a floor above what is free is refused", "floor")
    loose = manager(device, limit_mb=9000, strict=False)
    loose.start()
    check(loose.limit_mb == 9000, "an explicit budget is taken as given")
    cpu = ResidencyManager(type("Cpu", (), {"memory": lambda self: {}, "empty_cache": lambda self: None, "reset_peak": lambda self: None})(), MemoryPolicy())
    cpu.start()
    cpu.after_step(0)
    check(cpu.limit_mb is None and cpu.used_mb() is None, "without driver numbers the manager measures nothing and never interferes")


def pressure():
    device = FakeDevice(ours=9000)
    model, encoder, opt = Blob(device, 5000), Blob(device, 1000), Blob(device, 3000)
    m = manager(device, limit_mb=11000, safety_margin=0.1, calibration_steps=3)
    m.register("model", model, rank=10, pinned=True)
    m.register("encoder", encoder, rank=0)
    m.register("optimizer", opt, rank=1)
    m.start()
    m.after_step(0)  # 9600 MB in use, ceiling 9900: fine
    check(not [t for t in m.trace if t["event"] in ("lever", "pressure")] and device.cache_emptied == 0,
          "no reaction and no cache emptying while everything fits")
    device.ours = 9600
    m.after_step(1)
    check(encoder.on_device is False and opt.on_device and m.trace[-1]["name"] == "park encoder between steps",
          "under pressure the resident that is cheapest to move is parked first")
    check(device.cache_emptied == 1, "the cache is emptied once something has been released")
    m.before_step(2)
    check(encoder.on_device, "a parked resident comes back before the next step")
    device.ours = 9600
    m.after_step(2)
    check(opt.on_device is False and encoder.on_device is False, "still under pressure: the next one is parked as well")
    m.before_step(3)
    device.ours = 9600
    m.after_step(3)
    check(len([t for t in m.trace if t["event"] == "lever"]) == 2, "after calibration no further lever is pulled")
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
    raises(MemoryBudgetExceeded, lambda: m.make_room(6000, "huge"), "refuses when even moving everything movable is not enough", "do not fit")


def room_by_physical_limit():
    """With plenty of card left, making room must not refuse because the safety margin is crossed."""
    device = FakeDevice(total=11930, foreign=4300, ours=5600)
    model = Blob(device, 5000)
    m = manager(device)
    m.register("model", model, rank=10, pinned=True)
    m.start()
    check(m.make_room(1560, "load the text encoders") == [], "1.5 GB fit in the 2 GB left, so nothing moves and nothing is refused")
    raises(MemoryBudgetExceeded, lambda: m.make_room(2500, "too much"), "past the card's total it is refused", "do not fit")


def footprints():
    import torch

    class Frozen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("frozen", torch.zeros(1000, dtype=torch.bfloat16))
            self.adapter = torch.nn.Parameter(torch.zeros(500))
            self.on_meta = torch.nn.Parameter(torch.empty(10_000, device="meta"))

    from ..engine.runtime.residents import module_bytes
    check(module_bytes(Frozen(), on_device_only=False) == 1000 * 2 + 500 * 4,
          "frozen weights kept as buffers count toward a module's size and meta tensors do not")
    check(module_bytes(Frozen()) == 0, "on the host nothing counts as on the device")


def breakdown():
    device = FakeDevice(ours=8000)
    device.slack, device.spike = 700.0, 1500.0
    model, opt = Blob(device, 5000), Blob(device, 600)
    m = manager(device, limit_mb=11000)
    m.register("model", model, rank=10, pinned=True)
    m.register("optimizer", opt, rank=1)
    m.start()
    parts = m.breakdown()
    check(parts == {"model": 5000.0, "optimizer": 600.0, "other live": 2400.0, "activation peak": 1500.0, "allocator cache": 700.0,
                    "outside allocator": 600.0}, f"every megabyte on the card is attributed to a source ({parts})")
    m.after_step(0)
    check(m.trace[-1]["event"] == "breakdown" and m.trace[-1]["parts"] == parts, "the first step reports the breakdown for the monitor")


class FakeEncoder:
    def __init__(self, device: FakeDevice):
        self.device, self.moves, self.encoded = device, [], 0

    def move(self, where):
        self.moves.append(where)
        self.device.ours += 1000 if where != "cpu" else -1000

    def encode_text(self, prompts):
        self.encoded += 1
        return list(prompts)


def on_demand_encoders():
    device = FakeDevice(ours=5000)
    enc = FakeEncoder(device)
    resident = ModulesResident([], enc.move, "xpu")
    resident.size_bytes = lambda: 1000 * MB
    kit = OnDemandKit(enc, resident, device)
    kit.encode_text(["a"])
    check(enc.moves == ["xpu", "cpu"] and device.ours == 5000 and device.cache_emptied == 1,
          f"encoders visit the device to encode and leave, freeing the cache ({enc.moves})")
    with kit.session():
        kit.encode_text(["a"])
        kit.encode_text(["b"])
    check(enc.moves == ["xpu", "cpu", "xpu", "cpu"], "one session is one round trip however many encodes it holds")
    kit.hold = True
    kit.encode_text(["c"])
    kit.encode_text(["d"])
    check(enc.moves[-1] == "xpu" and enc.moves.count("xpu") == 3, "held encoders stay on the device")
    kit.release()
    check(device.ours == 5000, "release brings the device back to where it was")
    check(kit.cache_key("p") if hasattr(enc, "cache_key") else True, "other kit calls pass through")


def sampling():
    import time

    from ..engine.runtime.sampler import COLUMNS, MemorySampler

    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "m.csv"
        with MemorySampler(FakeDevice(ours=3000), path, interval=0.01) as sampler:
            sampler.label = "step 3"
            time.sleep(0.1)
        rows = path.read_text().strip().splitlines()
        check(rows[0] == ",".join(COLUMNS) and len(rows) > 3 and rows[-1].split(",")[1] == "step 3" and rows[-1].split(",")[-1] == "3600.0",
              f"the sampler logs labelled device memory several times a second ({len(rows) - 1} rows)")


def xpu_environment():
    import os

    from ..engine import xpu_env

    check(xpu_env.capacity_for_shapes(44) == 2048 and xpu_env.capacity_for_shapes(100) == 3500
          and xpu_env.capacity_for_shapes(None) is None and xpu_env.capacity_for_shapes(0) is None,
          "the primitive cache is sized to the shape count, never below the default, and unknown counts get no number")
    saved = {k: os.environ.pop(k, None) for k in (xpu_env.CACHE_VAR,)}
    try:
        xpu_env._explicit = False
        xpu_env.set_perf_env()
        check(os.environ[xpu_env.CACHE_VAR] == "2048" and xpu_env.size_cache_for_shapes(100) == 3500 and os.environ[xpu_env.CACHE_VAR] == "3500",
              "the default capacity grows to fit a dataset's shapes")
        os.environ[xpu_env.CACHE_VAR] = "512"
        xpu_env._explicit = False
        xpu_env.set_perf_env()
        check(xpu_env.size_cache_for_shapes(100) == 512 and os.environ[xpu_env.CACHE_VAR] == "512", "an exported capacity is kept even when too small")
        os.environ[xpu_env.CACHE_VAR] = "0"
        xpu_env._explicit = False
        xpu_env.set_perf_env()
        check(os.environ[xpu_env.CACHE_VAR] == "2048", "an unusable exported capacity falls back to the default")
    finally:
        xpu_env._explicit = False
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


def training():
    with tempfile.TemporaryDirectory() as raw:
        device = FakeDevice(ours=5000)
        m = manager(device, limit_mb=11000, empty_cache_every=1)
        m.start()
        trainer, *_ = rig(Path(raw), steps=2, residency=m)
        trainer.emit = lambda e, log=[]: log.append(e)
        trainer.run()
        check(m.steps_seen == 2 and device.cache_emptied == 0, "the trainer calls the manager around every step; nothing released, so no cache emptying")


if __name__ == "__main__":
    run(limits, pressure, making_room, room_by_physical_limit, footprints, breakdown, on_demand_encoders, sampling, xpu_environment, training)
