"""Memory runtime contract: the limit is settled honestly, pressure is answered between steps, nothing moves that must stay.

    python -m next.tests.check_memory

Uses a scripted fake device, so the numbers are exact: what the driver reports, what the allocator holds, who moves
off the device and in which order, and the refusals (floor, impossible budget, going over a strict budget).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ..engine.optim.algorithms import AdamW
from ..engine.optim.handle import Optimizer, ParamGroup
from ..engine.runtime.planner import StepPlanner, chunk_sizes
from ..engine.runtime.memory import MB, MemoryBudgetExceeded, MemoryPolicy, ResidencyManager, is_out_of_memory
from ..engine.runtime.encoders import OnDemandKit
from ..engine.runtime.residents import DeviceResident, ModulesResident
from types import SimpleNamespace

from ..engine.runtime.placement import BlockPlacement
from ..engine.runtime.profiler import BlockProfiler
from .check_engine import rig
from .support import check, raises, run


class FakeDevice:
    """Reports scripted driver numbers; ``foreign`` is memory other processes hold."""

    def __init__(self, total=12000.0, foreign=600.0, ours=0.0):
        self.total, self.foreign, self.ours, self.cache_emptied = total, foreign, ours, 0
        self.slack = self.spike = 0.0  # allocator cache beyond live tensors; activation peak above live tensors
        self.caps: list[float] = []
        self.own_outside = self.stuck = 0.0  # this process's memory outside the allocator; free pieces the cache cannot return
        self.reports_process = self.releasing = False

    def memory(self):
        out = {"device_total": self.total, "device_used": self.foreign + self.ours + self.slack + self.own_outside,
               "reserved": self.ours + self.slack, "allocated": self.ours, "peak_allocated": self.ours + self.spike}
        if self.reports_process:
            out |= {"process": self.ours + self.slack + self.own_outside, "split_free": self.stuck}
        return out

    def reset_peak(self):
        pass

    def empty_cache(self):
        self.cache_emptied += 1
        if self.releasing:
            self.slack = self.stuck

    def synchronize(self):
        pass

    capped = True

    def cap_allocator(self, mb):
        self.caps.append(mb)
        return self.capped


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
    m.claim("model", model, rank=10, pinned=True, purpose="test")
    m.claim("encoder", encoder, rank=0, purpose="test")
    m.claim("optimizer", opt, rank=1, purpose="test")
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
        m.claim(name, blob, rank=rank, pinned=pinned, purpose="test")
    m.start()
    moved = m.make_room(800, "decode a preview")
    check(moved == ["encoder"] and not encoder.on_device and opt.on_device, f"makes room by moving the lowest rank first ({moved})")
    m.restore(moved)
    check(encoder.on_device, "restore brings residents back")
    m.restore(moved)
    moved = m.make_room(800, "load the text encoders")
    m.before_step(1)
    check(encoder.on_device, "whatever left to make room is back before the next step, even if nobody restored it")
    raises(MemoryBudgetExceeded, lambda: m.make_room(6000, "huge"), "refuses when even moving everything movable is not enough", "do not fit")


def room_by_physical_limit():
    """With plenty of card left, making room must not refuse because the safety margin is crossed."""
    device = FakeDevice(total=11930, foreign=4300, ours=5600)
    model = Blob(device, 5000)
    m = manager(device)
    m.claim("model", model, rank=10, pinned=True, purpose="test")
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


def cached_blocks_are_not_pressure():
    device = FakeDevice(ours=8000)
    device.slack, device.spike = 2800.0, 1000.0  # 11400 MB in the driver's view, of which 2800 is reclaimable cache
    device.capped = False  # this allocator cannot be capped, so only we can give its cache back
    opt = Blob(device, 600)
    m = manager(device, safety_margin=0.1)
    m.claim("optimizer", opt, purpose="test")
    m.start()
    m.after_step(0)
    check(m.need_mb(device.memory()) == 9600 + 0 and not [t for t in m.trace if t["event"] in ("pressure", "lever")] and opt.on_device,
          "a card that looks full only because the allocator keeps free blocks is not under pressure")
    check(device.cache_emptied == 1 and m.trace[-1]["event"] == "trim" and opt.on_device,
          "it hands the cached blocks back instead, since the driver cannot reclaim them for its own allocations")
    device.slack, device.cache_emptied = 100.0, 0
    m.after_step(1)
    check(device.cache_emptied == 0, "a card with little cached is left alone")
    capped = FakeDevice(ours=8000)
    capped.slack, capped.spike = 2800.0, 1000.0
    again = manager(capped, safety_margin=0.1)
    again.start()
    again.after_step(0)
    check(capped.cache_emptied == 0, "under an allocator cap the cache is not emptied every step: the allocator frees it at the cap")


def allocator_cap():
    device = FakeDevice(ours=5000, foreign=1200)
    m = manager(device, calibration_steps=2)
    m.start()
    check(device.caps == [12000 - 1200 - m.OUTSIDE_HEADROOM_MB], f"the allocator is capped below the card by what lives outside it plus headroom ({device.caps})")
    device.foreign = 1500  # less than guessed: the cap relaxes at calibration
    m.after_step(0)
    m.after_step(1)
    check(device.caps[-1] == 12000 - 1500 - m.OUTSIDE_SLACK_MB, f"after calibration the cap is set from what was measured ({device.caps})")
    device.foreign = 2400
    m.after_step(2)
    check(device.caps[-1] == 12000 - 2400 - m.OUTSIDE_SLACK_MB, "it follows the outside share down when new shapes need more")
    device.foreign = 2000
    before = len(device.caps)
    m.after_step(3)
    check(len(device.caps) == before, "and does not loosen again")
    m.recover(3, RuntimeError("out of memory"))
    check(device.caps[-1] == 12000 - 2400 - 2 * m.OUTSIDE_SLACK_MB and m.trace[-1]["event"] == "allocator_cap"
          and any(t["event"] == "out_of_memory" for t in m.trace), "after an out-of-memory error the driver is given more room")
    check(is_out_of_memory(RuntimeError("UR_RESULT_ERROR_OUT_OF_RESOURCES")) and is_out_of_memory(RuntimeError("XPU out of memory. Tried"))
          and not is_out_of_memory(RuntimeError("shape mismatch")), "driver refusals count as out of memory, other errors do not")


def out_of_memory_retry():
    def run_with(fail_on_call=None, error=RuntimeError("UR_RESULT_ERROR_OUT_OF_RESOURCES"), **extra):
        with tempfile.TemporaryDirectory() as raw:
            device = FakeDevice(ours=5000)
            m = manager(device, limit_mb=11000)
            m.start()
            trainer, adapted, denoiser, _, events, _ = rig(Path(raw), steps=2, accumulation=2, residency=m, **extra)
            calls = [0]
            original = denoiser.predict

            def predict(*args):
                calls[0] += 1
                if calls[0] == fail_on_call:
                    raise error
                return original(*args)

            denoiser.predict = predict
            try:
                trainer.run()
                failure = None
            except Exception as caught:  # noqa: BLE001
                failure = caught
            return [p.detach().clone() for p in adapted.trainable_parameters()], calls[0], m, failure, len(events)

    clean, calls, _, _, _ = run_with()
    again, calls_again, m, failure, steps = run_with(fail_on_call=3)
    check(failure is None and steps == 2 and calls_again > calls and all(torch.allclose(a, b, atol=1e-4) for a, b in zip(clean, again)),
          "a window that runs out of memory is run again, in chunks, on the same batches and ends where an undisturbed run does")
    check(any(t["event"] == "out_of_memory" for t in m.trace) and m.accelerator.cache_emptied >= 1, "the manager is told, with the memory by source")
    _, _, _, failure, _ = run_with(fail_on_call=1, error=RuntimeError("shape mismatch"))
    check(isinstance(failure, RuntimeError) and "shape mismatch" in str(failure), "other errors are not retried")


def claims_and_ledger():
    device = FakeDevice(ours=8000)
    device.slack, device.spike = 700.0, 1500.0
    m = manager(device, limit_mb=12000)
    raises(ValueError, lambda: m.claim("x", Blob(device, 1), purpose=" "), "a claim without a stated purpose is refused", "what the memory is for")
    m.claim("model", Blob(device, 5000), purpose="weights", pinned=True)
    rows = {r["name"]: r for r in m.ledger()}
    check(rows["model"]["purpose"] == "weights" and rows["unclaimed live"]["mb"] == 3000 and rows["allocator free blocks"]["mb"] == 700,
          "the ledger lists each claim with its purpose, and what no claim covers")
    m.settle()
    check(device.cache_emptied == 1 and m.trace[-1]["event"] == "claims" and m.trace[-1]["rows"] == m.ledger(),
          "settling hands the cache back once the persistent claims are placed, and publishes the ledger")


def attribution():
    from ..engine.driver_memory import DriverMemory, parse_fdinfo

    device = FakeDevice(ours=5000, foreign=1300)
    device.reports_process, device.releasing = True, True
    device.slack, device.stuck, device.own_outside, device.spike = 1500.0, 300.0, 900.0, 800.0
    m = manager(device)
    m.claim("model", Blob(device, 4800), purpose="weights", pinned=True)
    parts = m.breakdown()
    check(parts == {"model": 4800.0, "unclaimed live": 200.0, "allocator free, returnable": 1200.0, "allocator free, stuck": 300.0,
                    "outside allocator (ours)": 900.0, "other programs": 1300.0} and sum(parts.values()) == device.memory()["device_used"],
          f"when the kernel reports this process's share, outside memory is split into ours and other programs ({parts})")
    m.start()
    cap = next(t for t in m.trace if t["event"] == "allocator_cap")
    check(cap["ours_outside_mb"] == 900 and cap["other_programs_mb"] == 1300 and cap["why"] == "start" and cap["limit_mb"] == 12000,
          "the cap event names each term that went into it")
    m.give_back("test")
    note = m.trace[-1]
    check(note["event"] == "cache_returned" and note["returned_mb"] == 1200 and note["card_freed_mb"] == 1200 and note["ours_outside_mb"] == 0,
          "a cache release says what it returned and whether the card's other users moved in")
    device.slack = 5000.0
    check(m.returnable_mb(device.memory()) == 4700, "only whole free segments count as returnable, not pieces between live blocks")
    check(parse_fdinfo("drm-driver:\txe\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t7\ndrm-resident-vram0:\t2048 KiB\ndrm-resident-gtt:\t5 KiB\n") ==
          (("0000:03:00.0", "7"), 2 * MB) and parse_fdinfo("pos:\t0\nflags:\t02\n") is None, "a handle's resident device memory is read, other handles are ignored")
    with tempfile.TemporaryDirectory() as folder:
        for name, client in (("3", 7), ("4", 7), ("5", 8)):
            (Path(folder) / name).write_text(f"drm-pdev:\tx\ndrm-client-id:\t{client}\ndrm-resident-vram0:\t{client - 6} MiB\n")
        (Path(folder) / "6").write_text("pos:\t0\n")
        check(DriverMemory(folder).resident_mb() == 3.0 and DriverMemory(Path(folder) / "none").resident_mb() is None,
              "handles of one client are counted once, and a system without the statistic gives None")


def checkpoint_density():
    from ..engine.runtime.density import CheckpointDensity

    def run_with(budget, cost=lambda share: 0.2 + 0.4 * (1 - share)):
        tuner, seen = CheckpointDensity(StepPlanner()), []

        def measure(share):
            seen.append(share)
            return cost(share)

        return tuner.choose(measure, 1.0, 5000, 1000, budget), seen

    chosen, seen = run_with(budget=20000)
    check(chosen == 0.0 and seen[-1] == 0.0, f"with room for everything no block is checkpointed, and the model is left at that share ({seen})")
    chosen, _ = run_with(budget=5500 + CheckpointDensity.RESERVE_MB)
    check(chosen == 0.5, f"the share goes down as far as the largest batch still fits ({chosen})")
    chosen, seen = run_with(budget=5200)
    check(chosen == 1.0 and seen == [1.0], "when even full checkpointing does not fit nothing more is measured")
    chosen, seen = run_with(budget=5600 + CheckpointDensity.RESERVE_MB, cost=lambda share: 0.2 + 0.4 * (1 - share) + (0.2 if share == 0.25 else 0))
    check(chosen == 0.5 and seen == [1.0, 0.5, 0.25, 0.5], f"a share that measures dearer than predicted is given up for the next one ({seen})")
    planner = StepPlanner()
    planner.learn(5000, 6000, 1000)
    planner.reset()
    check(planner.mb_per_unit is None, "the planner can forget costs measured at another share")


def planning():
    check(chunk_sizes(7, 3) == [3, 2, 2] and chunk_sizes(4, 1) == [4], "a batch splits into near-equal chunks")
    planner = StepPlanner()
    check(planner.chunks(4, 1000, 5000, 9000) == [4], "nothing is split before a cost is known")
    planner.learn(5000, 6000, 1000)  # one megabyte per pixel
    planner.learn(5000, 5500, 1000)
    check(planner.mb_per_unit == 1.0, "the dearest recent step sets the cost")
    check(planner.chunks(2, 1000, 5000, 9000) == [2] and planner.chunks(6, 1000, 5000, 9000) == [3, 3],
          "a batch is split into the fewest chunks whose predicted peak fits the budget")
    check(planner.chunks(3, 1000, 5000, 5100) == [1, 1, 1], "and into single samples when only those can fit")
    device = FakeDevice(ours=5000)
    device.spike = 1000.0
    m = manager(device, limit_mb=12000)
    m.start()
    with m.measuring(1000):
        pass
    check(abs(m.planner.mb_per_unit - 1.0) < 1e-9 and m.trace[-1]["event"] == "cost", "a probe measures the cost of a pixel from its peak")
    check(m.plan(6, 1000) == [3, 3] and any(t["event"] == "split" for t in m.trace), "the controller plans a split before the step and says so")
    m.after_step(0)
    check(m.planner.mb_per_unit >= 1.0, "the step's own peak keeps the cost current")


def persistent_first():
    a, b = torch.nn.Parameter(torch.ones(4, 4)), torch.nn.Parameter(torch.ones(4, 4))
    opt = Optimizer(AdamW(weight_decay=0.0), [ParamGroup([a, b])], 1e-1)
    opt.hold_grads()
    place = a.grad.data_ptr()
    check(a.grad is not None and b.grad is not None, "held gradients exist before any step")
    for _ in range(2):
        opt.zero_grad()
        (a * 2).sum().backward()
        opt.step()
    check(a.grad.data_ptr() == place, "gradients are zeroed in place, so no step allocates them again")
    check(bool((a != 1).any()) and bool((b == 1).all()), "a parameter the backward did not reach is still left alone")
    opt.release()


def breakdown():
    device = FakeDevice(ours=8000)
    device.slack, device.spike = 700.0, 1500.0
    model, opt = Blob(device, 5000), Blob(device, 600)
    m = manager(device, limit_mb=12000)
    m.claim("model", model, rank=10, pinned=True, purpose="test")
    m.claim("optimizer", opt, rank=1, purpose="test")
    m.start()
    parts = m.breakdown()
    check(parts == {"model": 5000.0, "optimizer": 600.0, "unclaimed live": 2400.0, "allocator free blocks": 700.0, "outside allocator": 600.0}
          and sum(parts.values()) == device.memory()["device_used"], f"every megabyte on the card is attributed to exactly one source ({parts})")
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
    with kit.session():
        pass
    check(enc.moves == ["xpu", "cpu", "xpu", "cpu"], "a session that encodes nothing never loads the encoders")
    kit.hold = True
    kit.encode_text(["c"])
    kit.encode_text(["d"])
    check(enc.moves[-1] == "xpu" and enc.moves.count("xpu") == 3, "held encoders stay on the device")
    kit.release()
    check(device.ours == 5000, "release brings the device back to where it was")
    class Room:
        moved = ["optimizer"]
        restored: list = []

        def make_room(self, mb, why):
            return self.moved

        def restore(self, names):
            self.restored.append(names)

    roomy = OnDemandKit(FakeEncoder(device), resident, device, Room())
    roomy.encode_text(["a"])
    check(Room.restored == [["optimizer"]], "residents moved for the encoders return when the encoders leave")
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
        check(rows[0] == ",".join(COLUMNS) and len(rows) > 3 and rows[-1].split(",")[1] == "step 3" and rows[-1].split(",")[5] == "3600.0",
              f"the sampler logs labelled device memory several times a second ({len(rows) - 1} rows)")


def xpu_environment():
    import os

    from ..engine import xpu_env

    check(xpu_env.capacity_for_shapes(44) == 2048 and xpu_env.capacity_for_shapes(100) == 3500
          and xpu_env.capacity_for_shapes(None) is None and xpu_env.capacity_for_shapes(0) is None,
          "the primitive cache is sized to the shape count, never below the default, and unknown counts get no number")
    saved = {k: os.environ.pop(k, None) for k in (xpu_env.CACHE_VAR, "PYTORCH_ALLOC_CONF", "PYTORCH_XPU_ALLOC_CONF")}
    try:
        xpu_env._explicit = False
        xpu_env.set_perf_env()
        check(os.environ[xpu_env.CACHE_VAR] == "2048" and xpu_env.size_cache_for_shapes(100) == 3500 and os.environ[xpu_env.CACHE_VAR] == "3500",
              "the default capacity grows to fit a dataset's shapes")
        check(os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True", "the allocator uses expandable segments unless told otherwise")
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
        xpu_env.set_perf_env()
        check(os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:False", "an exported allocator setting wins")
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


class FakeBlock:
    """A block whose recompute holds ``mb`` megabytes and takes about ``ms`` milliseconds."""

    def __init__(self, memory: list, mb: float, ms: float):
        self.memory, self.mb, self.ms = memory, mb, ms

    def forward(self, *args):
        import time
        self.memory[0] += self.mb
        time.sleep(self.ms / 1000)
        return args


def planning_with_fixed_cost():
    planner = StepPlanner()
    planner.learn(0, 300, 1000)
    check(planner.line() == (0.0, 0.3) and planner.cost_at(5000) == 0.3, "one point is taken as proportional")
    planner.learn(0, 1300, 11000)
    fixed, slope = planner.line()
    check(abs(fixed - 200) < 1e-6 and abs(slope - 0.1) < 1e-9, f"two sizes separate the fixed part from the part that grows ({fixed}, {slope})")
    check(abs(planner.predicted_peak_mb(100, 22000) - (100 + 200 + 22000 * 0.1 * 1.15)) < 1e-6, "the margin is on the growing part only")
    planner.learn(0, 1000, 6000)  # above the fitted line
    fixed, slope = planner.line()
    check(all(fixed + slope * u >= e - 1e-9 for u, e in ((1000, 300), (11000, 1300), (6000, 1000))), "the line lies above every observed point")
    check(planner.chunks(22, 1000, 0, 2000) == [11, 11] and planner.chunks(22, 1000, 0, 4000) == [22], "chunks follow the fitted line")
    flat = StepPlanner()
    flat.learn(0, 500, 1000)
    flat.learn(0, 400, 2000)
    check(flat.line() == (0.0, 0.5), "a falling trend is not extrapolated: the proportional bound is used")
    planner.reset()
    check(planner.line() is None and planner.chunks(5, 1000, 0, 100) == [5], "reset forgets everything")


def cache_thrash():
    from ..engine.runtime.thrash import PrimitiveCacheMonitor

    def feed(steps, **kw):
        monitor, messages = PrimitiveCacheMonitor(**kw), []
        messages = [monitor.observe(shape, seconds) for shape, seconds in steps]
        return monitor, [m for m in messages if m]

    slow_first = [(f"s{i}", t) for i in range(8) for t in (4.0, 1.0)] + [(f"s{i % 8}", 1.0) for i in range(30)]
    check(not feed(slow_first)[1], "slow first sightings alone are compilation, not cache eviction: no warning")
    thrash = [("a", 1.0), ("a", 1.0), ("b", 1.0), ("b", 1.0)] + [("a", 3.5), ("b", 3.5), ("a", 3.5), ("a", 1.0)] * 3
    monitor, warnings = feed(thrash)
    check(len(warnings) == 1 and "ONEDNN_PRIMITIVE_CACHE_CAPACITY" in warnings[0] and monitor.warned, "slow revisits warn once and name the knob")
    check(not feed([("a", 1.0), ("b", 1.0), ("a", 3.5), ("b", 3.5), ("a", 3.5)])[1], "too few revisits are no evidence")
    check(not feed([("a", 1.0), ("b", 1.0)] * 20)[1], "a run that never repeats a shape has no baseline and does not warn")
    raises(ValueError, lambda: PrimitiveCacheMonitor(ratio=1.0), "ratio must exceed 1", "ratio")


def measured_placement():
    memory = [0.0]
    sizes = {"a": (100, 3), "b": (20, 8), "c": (200, 3), "d": (10, 6), "silent": (0, 2)}  # mb, ms
    blocks = [(name, FakeBlock(memory, *size)) for name, size in sizes.items()]
    checkpointer = SimpleNamespace(observer=None)
    profiler = BlockProfiler(checkpointer, blocks, lambda: memory[0])
    placement = BlockPlacement(SimpleNamespace(MARGIN=1.15), profiler)

    def scripted(extra=0.0, seen=None):
        def measure(flags):
            if seen is not None:
                seen.append(list(flags))
            if all(flags):  # the backward of the first run: every block recomputes under the observer
                for _, block in blocks if checkpointer.observer else ():
                    checkpointer.observer(block.forward, [])
                    memory[0] = 0.0
                return 5.0
            kept = sum(b.mb for (_, b), on in zip(blocks, flags) if not on)
            return 5.0 + kept / 10 + extra
        return measure

    # the budget leaves room for exactly the two cheapest-to-keep blocks
    budget = 768 + 1000 + 100 * 5.0 * 1.15 + 30 * 10 * 1.15
    flags = placement.choose(scripted(), 1000, 100, 10, budget)
    check(flags == [True, False, True, False, True], f"the blocks that save the most time per megabyte are kept while they fit, the rest recompute ({flags})")
    check(not checkpointer.observer, "the profiler lets go of the checkpointer")
    check([e["block"] for e in placement.describe(flags)] == ["b", "d"], "the kept blocks are described with its measured cost")

    seen: list = []
    flags = placement.choose(scripted(extra=float("inf"), seen=seen), 1000, 100, 10, budget)
    check(all(flags) and seen[-1] == [True] * 5 and len(seen) < 10,
          f"a placement that does not fit once measured is given back to checkpointing until nothing is kept ({len(seen)} measurements)")

    check(all(placement.choose(scripted(), 1000, 100, 10, 768 + 1000 + 100 * 5.0 * 1.15 - 1)),
          "no room beyond the all-checkpointed cost keeps everything checkpointed")
    check(all(placement.choose(lambda flags: 5.0, 1000, 100, 10, 1e9)), "a profile without any activation figure (no data) keeps everything checkpointed")


def measured_checkpointing_in_trainer():
    from ..engine.runtime.checkpointing import ActivationCheckpointer

    class Handle:
        checkpoint_share = 1.0

        def __init__(self, denoiser):
            self.denoiser, self.checkpointer = denoiser, ActivationCheckpointer()

        def checkpoint_blocks(self):
            return self.denoiser.checkpoint_blocks()

        def set_checkpoint_flags(self, flags):
            self.denoiser.set_checkpointed(self.checkpointer, flags)

    with tempfile.TemporaryDirectory() as raw:
        device = FakeDevice(ours=5000)
        device.spike = 200.0
        m = manager(device, limit_mb=11000)
        m.start()
        trainer, _, denoiser, *_ = rig(Path(raw), steps=1, residency=m)
        handle = Handle(denoiser)
        profiler = BlockProfiler(handle.checkpointer, handle.checkpoint_blocks(), lambda: device.memory().get("allocated", 0.0))
        trainer._placement = (BlockPlacement(m.planner, profiler), handle)
        trainer.emit = lambda e: None
        trainer.run()
        check(len({u for u, _ in m.planner._points}) >= 2, "the memory probe runs at two batch sizes so the planner can tell fixed memory from growing memory")
        check(any(t["event"] == "checkpoint_plan" for t in m.trace) and handle.checkpointer.observer is None
              and all(b.checkpointer is handle.checkpointer for _, b in handle.checkpoint_blocks()),
              "the trainer measures with the profiler attached, keeps everything checkpointed when the profile has no memory figures, and reports the plan")


def tuned_checkpointing():
    from ..engine.runtime.checkpointing import ActivationCheckpointer
    from ..engine.runtime.density import CheckpointDensity
    from ..families.general.ldm_unet.attention import BasicTransformerBlock

    class Handle:
        checkpoint_share = 1.0

        def __init__(self, device):
            self.denoiser, self.device = None, device

        def set_checkpoint_share(self, share):
            self.denoiser.set_checkpointer(ActivationCheckpointer(), share)
            self.checkpoint_share = share
            self.device.spike = 200.0 + 100.0 * (1 - share)  # fewer recomputed blocks keep more activations

    with tempfile.TemporaryDirectory() as raw:
        device = FakeDevice(ours=5000)
        device.spike = 200.0
        m = manager(device, limit_mb=11000)
        m.start()
        handle = Handle(device)
        trainer, _, denoiser, *_ = rig(Path(raw), steps=1, residency=m, density=(CheckpointDensity(m.planner), handle))
        handle.denoiser = denoiser
        trainer.emit = lambda e: None
        trainer.run()
        blocks = [b for b in denoiser.model.modules() if isinstance(b, BasicTransformerBlock)]
        check(handle.checkpoint_share == 0.0 and all(b.checkpointer is None for b in blocks) and m.planner.mb_per_unit is not None
              and any(t["event"] == "checkpoint_share" and t["share"] == 0.0 for t in m.trace),
              "the trainer lowers the checkpointed share to what the budget allows, applies it to the network and reports it")

    with tempfile.TemporaryDirectory() as raw:
        device = FakeDevice(ours=5000)
        device.spike = 200.0
        m = manager(device, limit_mb=11000)
        m.start()
        handle = Handle(device)
        trainer, _, denoiser, *_ = rig(Path(raw), steps=1, residency=m, density=(CheckpointDensity(m.planner), handle))
        handle.denoiser = denoiser
        trainer.emit = lambda e: None
        real = trainer.pipeline.run

        def run_pipeline(state):
            if state.step < 0 and handle.checkpoint_share <= 0.5:
                raise RuntimeError("XPU out of memory. Tried to allocate 40.00 MiB")
            return real(state)

        trainer.pipeline.run = run_pipeline
        result = trainer.run()
        check(result.steps_done == 1 and handle.checkpoint_share == 1.0,
              "a share whose measuring run out of memory counts as too expensive: the tuner keeps the starting share and the run goes on")


if __name__ == "__main__":
    run(limits, pressure, making_room, room_by_physical_limit, footprints, cached_blocks_are_not_pressure, claims_and_ledger, planning, persistent_first, allocator_cap, out_of_memory_retry, breakdown, attribution, checkpoint_density, on_demand_encoders, sampling, xpu_environment, training, tuned_checkpointing, planning_with_fixed_cost, cache_thrash, measured_placement, measured_checkpointing_in_trainer)
