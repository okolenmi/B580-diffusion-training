"""Forward, loss and backward of one micro-step replayed as a captured device graph.

A step is bound by the one thread that issues its ~22,000 kernel launches; replaying them from a graph removes the
issuing cost (measured about 2x for forward plus backward). What is captured: the denoiser's forward, the masked
per-sample loss, the weighting and the backward, all reading static buffers that each replay overwrites with the
step's inputs. What stays eager: conditioning, noise, the optimizer step, the time probe.

Replay is not always the faster path: a captured step cannot use every kernel the eager one does (the flash attention
cannot be captured, so the capture runs the math attention). With a clock, each shape's eager time is taken from its
warm-up steps and its first replays are timed against it; a graph that does not win is dropped and the shape trains
eagerly.

A shape is captured only after it has been seen eagerly a few times (kernels compiled, memory settled), and only at
the first micro-step of a window, because capturing executes the computation once and its gradients are discarded
again. Graphs share one memory pool, so what they cost is the largest of them, not the sum.

Anything that cannot be replayed runs eagerly with the very same phases: the memory probe, batches the controller
split into chunks, shapes past the cap, shapes whose capture failed. Replays write gradients in place, so gradients
must be held (never reassigned), the optimizer must not update inside backward, nothing in the forward may draw
random numbers, and the adapter must not be gated per step (the gate is a fresh tensor every batch).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager

import torch
from torch import Tensor

from .loss import LossWeighting, per_sample_mse
from .phases import StepPhase
from .state import StepState


class GraphRefused(ValueError):
    """The run's configuration cannot be captured; says which part."""


@dataclass
class _Slot:
    sightings: int = 0
    graph: Any = None
    statics: dict = field(default_factory=dict)
    loss: Tensor | None = None
    per_sample: Tensor | None = None
    touched: frozenset = frozenset()
    eager_s: float = math.inf  # fastest timed warm-up step
    replay_s: float = math.inf
    replays_timed: int = 0


class GraphedCompute(StepPhase):
    """Stands in for the Forward, Loss and Backward phases.

    ``eager`` are those phases; ``graph_api`` is the device's ``XPUGraph`` / ``graph`` / ``graph_pool_handle`` triple
    (anything with the same three members, so the logic runs on a fake in tests), None where capture does not exist.
    """

    def __init__(self, *, eager: list[StepPhase], denoiser: Any, weighting: LossWeighting, balance: Any, optimizer: Any,
                 accumulation: int, graph_api: Any | None, synchronize: Callable[[], None], max_graphs: int = 8,
                 warmup: int = 3, wanted: frozenset | None = None, attentions: list[Callable[[], ContextManager]] | None = None, say: Callable[[str], None] = print,
                 clock: Callable[[], float] | None = None):
        self._eager, self._denoiser, self._weighting, self._balance = eager, denoiser, weighting, balance
        self._optimizer, self._scale, self._api, self._sync = optimizer, 1.0 / accumulation, graph_api, synchronize
        self._max, self._warmup, self._wanted = max_graphs, warmup, wanted  # wanted: (samples, h, w) allowed a graph; None = first come
        self._attentions = attentions or [contextlib.nullcontext]  # tried in order; the first that captures stays
        self._say, self._clock = say, clock  # a clock turns on the check that a graph is faster than eager
        self._slots: dict[tuple, _Slot] = {}
        self._off: set[tuple] = set()
        self._pool = None
        self.replays = self.captures = self.dropped = 0
        self.force_eager = False  # set by the profiler: a replay is one opaque launch to it, so the profiled steps run the eager phases

    MIN_GAIN = 0.05  # a replay must beat the eager step by this share to keep its graph
    TIMED_REPLAYS = 3
    FIRST_TIMED_SIGHTING = 2  # the first sighting pays for compiling kernels

    @property
    def label(self) -> str:
        return "graphed_compute"

    @staticmethod
    def refuse_unless_capturable(model: Any, optimizer: Any, fused: bool) -> None:
        if fused:
            raise GraphRefused("graph capture cannot be used with a fused optimizer: its update runs inside the backward")
        if not getattr(optimizer, "holds_grads", False):
            raise GraphRefused("graph capture needs gradients held in place (they are zeroed, never reassigned)")
        if not getattr(model, "deterministic", False):
            raise GraphRefused("graph capture needs a forward that draws no random numbers (dropout is on)")
        if getattr(model, "gate", None) is not None and getattr(model, "gate_window", None) is not None:
            raise GraphRefused("graph capture cannot be used with the timestep gate: the gate changes with every batch")

    def run(self, state: StepState) -> None:
        if self.force_eager:
            # the kernels a replay runs, made visible: the attention the capture settled on
            with self._attentions[0]() if self.captures else contextlib.nullcontext():
                return self._run_eager(state)
        if self._api is None or state.step < 0 or state.scale != 1.0:
            return self._run_eager(state)
        inputs = state.inputs
        if self._wanted is not None and (inputs.x_t.shape[0], *inputs.x_t.shape[-2:]) not in self._wanted:
            return self._run_eager(state)
        key = (tuple(inputs.x_t.shape), inputs.mask is not None, tuple(tuple(t.shape) for t in state.conditioning.tensors()))
        if key in self._off:
            return self._run_eager(state)
        slot = self._slots.get(key)
        if slot is None:
            if len(self._slots) >= self._max:
                self._off.add(key)
                self._say(f"[graph] shape {key[0]} is past the {self._max}-graph cap and runs eagerly")
                return self._run_eager(state)
            slot = self._slots[key] = _Slot()
        if slot.graph is None and not (state.micro == 0 and slot.sightings >= self._warmup):
            slot.sightings += 1
            return self._run_timed(slot, state) if slot.sightings >= self.FIRST_TIMED_SIGHTING else self._run_eager(state)
        if slot.graph is None and not self._capture(key, slot, state):
            return self._run_eager(state)
        self._replay(key, slot, state)

    def _run_timed(self, slot: _Slot, state: StepState) -> None:
        if self._clock is None:
            return self._run_eager(state)
        self._sync()
        start = self._clock()
        self._run_eager(state)
        self._sync()
        slot.eager_s = min(slot.eager_s, self._clock() - start)

    def _run_eager(self, state: StepState) -> None:
        for phase in self._eager:
            phase.run(state)

    def _extra(self, state: StepState) -> Tensor | None:
        if self._balance is None:
            return None
        weights = self._balance.weights_for(state.inputs.t_frac)
        return torch.ones(state.inputs.x_t.shape[0], device=state.inputs.x_t.device) if weights is None else weights

    def _fill(self, slot: _Slot, state: StepState) -> None:
        i, s = state.inputs, slot.statics
        for name, value in (("x_t", i.x_t), ("t", i.t), ("sigma", i.sigma), ("target", i.target)):
            s[name].copy_(value)
        if i.mask is not None:
            s["mask"].copy_(i.mask)
        for static, live in zip(s["conditioning"].tensors(), state.conditioning.tensors()):
            static.copy_(live)
        extra = self._extra(state)
        if extra is not None:
            s["extra"].copy_(extra)

    def _compute(self, s: dict) -> tuple[Tensor, Tensor]:
        pred = self._denoiser.predict(s["x_t"], s["t"], s["conditioning"])
        per_sample = per_sample_mse(pred, s["target"], s.get("mask"))
        loss = self._weighting.apply(per_sample, s["sigma"], s.get("extra"))
        (loss * self._scale).backward()
        return loss.detach(), per_sample.detach()

    def _capture(self, key: tuple, slot: _Slot, state: StepState) -> bool:
        """Capture this shape. The window's gradients are still zero (micro-step 0), and are zeroed again after the
        capture run, so the replay that follows is the step."""
        i = state.inputs
        slot.statics = {"x_t": torch.empty_like(i.x_t), "t": torch.empty_like(i.t), "sigma": torch.empty_like(i.sigma),
                        "target": torch.empty_like(i.target), "conditioning": state.conditioning.map(torch.empty_like)}
        if i.mask is not None:
            slot.statics["mask"] = torch.empty_like(i.mask)
        extra = self._extra(state)
        if extra is not None:
            slot.statics["extra"] = torch.empty_like(extra)
        self._fill(slot, state)
        failure = None
        for attention in list(self._attentions):
            try:
                self._sync()
                if self._pool is None:
                    self._pool = self._api.graph_pool_handle()
                graph = self._api.XPUGraph()
                with attention(), self._api.graph(graph, pool=self._pool):
                    loss, per_sample = self._compute(slot.statics)
                self._sync()
                self._attentions = [attention]
                failure = None
                break
            except Exception as error:  # noqa: BLE001 - a shape that cannot be captured trains eagerly instead
                failure = error
                self._optimizer.zero_grad()
        if failure is not None:
            self._slots.pop(key, None)
            self._off.add(key)
            self._say(f"[graph] capture of shape {key[0]} failed ({type(failure).__name__}: {str(failure).splitlines()[0] if str(failure) else ''}); it runs eagerly")
            return False
        slot.graph, slot.loss, slot.per_sample = graph, loss, per_sample
        slot.touched = self._optimizer.touched()
        self._optimizer.zero_grad()
        self.captures += 1
        self._say(f"[graph] captured shape {key[0]} ({self.captures} graph(s))")
        return True

    def _replay(self, key: tuple, slot: _Slot, state: StepState) -> None:
        timed = self._clock is not None and slot.replays_timed < self.TIMED_REPLAYS and slot.eager_s < math.inf
        if timed:
            self._sync()
            start = self._clock()
        self._fill(slot, state)
        slot.graph.replay()
        self._optimizer.touch(slot.touched)
        state.loss, state.per_sample = slot.loss.clone(), slot.per_sample.clone()
        self.replays += 1
        if timed:
            self._sync()
            slot.replay_s = min(slot.replay_s, self._clock() - start)
            slot.replays_timed += 1
            if slot.replays_timed == self.TIMED_REPLAYS:
                self._judge(key, slot)

    def _judge(self, key: tuple, slot: _Slot) -> None:
        kept = slot.replay_s <= slot.eager_s * (1 - self.MIN_GAIN)
        self._say(f"[graph] shape {key[0]}: replay {slot.replay_s * 1000:.0f} ms against {slot.eager_s * 1000:.0f} ms eager"
                  f"{'' if kept else '; the graph does not pay and is dropped'}")
        if kept:
            return
        self._slots.pop(key, None)
        self._off.add(key)
        self.dropped += 1
        if not self._slots:
            self._pool = None  # nothing left to share it; its memory goes back to the allocator
