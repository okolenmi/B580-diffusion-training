"""One implementation of each phase of the training step. Differences between runs (which optimizer, which
loss weighting, which family) are collaborators handed to the phase, never copies of it."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator

from .loss import LossWeighting, per_sample_mse
from .state import Inputs, StepState


class StepPhase(ABC):
    @abstractmethod
    def run(self, state: StepState) -> None: ...

    @property
    def label(self) -> str:
        name = type(self).__name__
        return "".join(("_" if c.isupper() and i else "") + c.lower() for i, c in enumerate(name))


class FetchBatch(StepPhase):
    """Next batch; when an epoch ends the batcher announces it and a fresh pass starts here."""

    def __init__(self, batches: Iterable[dict]):
        self._batches = batches
        self._iterator: Iterator[dict] = iter(batches)

    def run(self, state: StepState) -> None:
        try:
            state.batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._batches)
            state.batch = next(self._iterator)


class PrepareInputs(StepPhase):
    """Batch tensors to the device and the adapter gate for this batch's timesteps."""

    def __init__(self, gate: Any = None, window: Any = None):
        """``gate`` has ``set(tensor | None)``; ``window`` has ``gate(t) -> tensor`` (see the adapters' ``GateWindow``)."""
        self._gate, self._window = gate, window

    def run(self, state: StepState) -> None:
        batch, device = state.batch, state.device
        state.inputs = Inputs(
            x_t=batch["x_t"].to(device), t=batch["t"].to(device), sigma=batch["sigma"].to(device),
            target=batch["target"].to(device), t_frac=batch["t_frac"],
            mask=batch["valid_mask"].to(device) if batch.get("valid_mask") is not None else None)
        if self._gate is not None:
            self._gate.set(self._window.gate(state.inputs.t) if self._window is not None else None)


class EncodeConditioning(StepPhase):
    def __init__(self, conditioner):
        self._conditioner = conditioner

    def run(self, state: StepState) -> None:
        state.conditioning = self._conditioner.encode(state.batch["prompts"], state.batch["size_px"])


class Forward(StepPhase):
    def __init__(self, denoiser):
        self._denoiser = denoiser

    def run(self, state: StepState) -> None:
        i = state.inputs
        state.prediction = self._denoiser.predict(i.x_t, i.t, state.conditioning)


class Loss(StepPhase):
    def __init__(self, weighting: LossWeighting):
        self._weighting = weighting

    def run(self, state: StepState) -> None:
        i = state.inputs
        state.per_sample = per_sample_mse(state.prediction, i.target, i.mask)
        state.loss = self._weighting.apply(state.per_sample, i.sigma)


class Backward(StepPhase):
    """Backward of the loss scaled by ``1 / accumulation`` so the window's gradients average."""

    def __init__(self, accumulation: int = 1):
        self._scale = 1.0 / accumulation

    def run(self, state: StepState) -> None:
        (state.loss * self._scale).backward()


class Timed(StepPhase):
    """Records how long ``inner`` took, with the device idle at both ends."""

    def __init__(self, inner: StepPhase, synchronize=lambda: None):
        self.inner, self._sync = inner, synchronize

    @property
    def label(self) -> str:
        return self.inner.label

    def run(self, state: StepState) -> None:
        self._sync()
        started = time.perf_counter()
        self.inner.run(state)
        self._sync()
        state.timing_ms[self.label] = state.timing_ms.get(self.label, 0.0) + (time.perf_counter() - started) * 1000


class Pipeline:
    def __init__(self, phases: list[StepPhase]):
        self.phases = phases

    def run(self, state: StepState) -> StepState:
        for phase in self.phases:
            phase.run(state)
        return state
