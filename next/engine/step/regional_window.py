"""A step whose batches are split into timestep regions that each run on their own, and the combined update they give."""

from __future__ import annotations

import gc
from collections.abc import Callable

import torch
from torch import Tensor

from ..runtime.memory import is_out_of_memory
from .loss import T_THIRDS, thirds_of_range
from .regional import RegionalUpdate, RegionGradients, RegionStream
from .state import split_batch


class RegionalWindow:
    """Each region of a batch (its samples at one part of the timestep range) runs on its own from zero gradients; each region's
    gradients are averaged over the window's batches, the policy combines them, and the result is written back for the optimizer.
    A plain average of normalised regions over one batch is summed on the device as the passes finish (``RegionStream``); any
    other policy needs every region's vector, which waits in host memory until the passes are done.

    With ``calibrate_every`` only every Nth step runs the regions alone (``takes``); the others run as one batch with each region's
    loss weighted by the inverse of the gradient norm the calibrating steps measured. ``region_pass`` is the trainer's forward and
    backward of one region: ``(step, micro, part, attempt) -> (loss, per-sample losses, time fractions, timing)``."""

    def __init__(self, update: RegionalUpdate, optimizer, device, region_pass: Callable, balance=None, calibrate_every: int = 0):
        self.update, self.optimizer, self.device, self._region_pass = update, optimizer, device, region_pass
        self._balance, self._calibrate_every = balance, calibrate_every
        self._stream: RegionStream | None = None
        self.metrics: dict[str, float] = {}

    def takes(self, step: int) -> bool:
        if not self._calibrate_every:
            return True
        return step % self._calibrate_every == 0 or len(self._balance.weights) < len(T_THIRDS)

    def idle(self) -> None:
        """A step that ran as one batch: no per-region numbers to report."""
        self.metrics = {}

    def run(self, step: int, fetched: list[dict], attempt: int) -> tuple[list, list, dict, bool]:
        """The window's regions and the update direction written onto the parameters. The last item says whether an update is to be
        applied (the policy may skip a step on which the regions cancel)."""
        grads, pending = RegionGradients(self.optimizer), None
        stream = None
        if self.update.streams(len(fetched)):
            stream = self._stream = self._stream or RegionStream(self.optimizer)
            stream.begin()
        losses, thirds, timing, per_region = [], [], {}, {}
        for micro, batch in enumerate(fetched):
            region = batch.get("region")
            sizes = [int(n) for n in torch.bincount(region).tolist() if n] if region is not None else [batch["x_t"].shape[0]]
            matrix, loss, per_sample, fractions = [], 0.0, [], []
            for index, part in enumerate(split_batch(batch, sizes)):
                self.optimizer.zero_grad()
                value, losses_part, fractions_part, spent = self._region_pass(step, micro, part, attempt)
                if stream is not None:
                    stream.add()
                else:
                    matrix.append(grads.read())
                per_region.setdefault(index, []).append(float(value))
                loss, per_sample, fractions = loss + value / len(sizes), per_sample + losses_part, fractions + fractions_part
                for label, ms in spent.items():
                    timing[label] = timing.get(label, 0.0) + ms
            if stream is None:
                matrix = torch.stack(matrix)
                if pending is not None and pending.shape != matrix.shape:
                    raise ValueError("the batches of one window hold different numbers of regions")
                pending = matrix if pending is None else pending + matrix  # the policy sees each region's gradient over the whole window
            losses.append(loss)
            thirds.append(thirds_of_range(torch.cat([p.flatten() for p in per_sample]), torch.cat(fractions)))
        if stream is not None:
            norms, extra = stream.finish()
            total = True
        else:
            norms, total = self._combine(pending, len(fetched))
            extra = self.update.policy.last
        self.metrics = {f"loss_region_{i}": sum(v) / len(v) for i, v in per_region.items()}
        self.metrics.update({f"grad_norm_region_{i}": float(n) for i, n in enumerate(norms)})
        self.metrics.update(extra)
        self.metrics["update_skipped"] = float(total is None)
        if stream is None:
            self.optimizer.zero_grad()
            if total is not None:
                grads.write(total)
        if self._calibrate_every:
            self._learn_norms()
        return losses, thirds, timing, total is not None

    def _learn_norms(self) -> None:
        for i, name in enumerate(T_THIRDS):
            norm = self.metrics.get(f"grad_norm_region_{i}")
            if norm is not None:
                self._balance.observe_gradient({name: 0.0}, norm)

    def _combine(self, pending: Tensor, count: int) -> tuple[Tensor, Tensor | None]:
        """The regions' norms and the update direction from their gradients summed over ``count`` batches. The vectors wait in
        host memory while the passes run; here, with the activations gone, they go to the device for the arithmetic (the host
        does the same work many times slower) unless the card cannot hold them."""
        try:
            return self._combine_on(pending.to(self.device), count)
        except BaseException as error:
            if not is_out_of_memory(error):
                raise
            error.__traceback__ = None
            gc.collect()
            return self._combine_on(pending, count)

    def _combine_on(self, matrix: Tensor, count: int) -> tuple[Tensor, Tensor | None]:
        window = matrix / count if count > 1 else matrix
        return window.norm(dim=1).cpu(), self.update.direction(window)
