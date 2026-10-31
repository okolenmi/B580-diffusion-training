"""One update from the gradients of several timestep regions: reading them off the parameters, combining them with a
policy, and writing the result back for the optimizer."""

from __future__ import annotations

import torch
from torch import Tensor

from .policy import GradientPolicy


class RegionGradients:
    """The trainable parameters' gradients as one flat float32 vector, and back."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.params = optimizer.params
        self.reached: set[int] = set()  # parameters some region's backward reached

    def read(self) -> Tensor:
        self.reached |= self.optimizer.touched()
        return torch.cat([(torch.zeros(p.numel(), device=p.device) if p.grad is None else p.grad.detach().float().flatten()) for p in self.params])

    def write(self, vector: Tensor) -> None:
        offset = 0
        for p in self.params:
            piece = vector[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
            if p.grad is None:
                p.grad = piece.clone()
            else:
                p.grad.copy_(piece)
        self.optimizer.touch(frozenset(self.reached))


class RegionalUpdate:
    """Turns one micro-batch's regions into an update direction. With ``normalize`` every region's gradient is scaled to unit
    length first, so a region whose loss is small (heavily down-weighted timesteps) is not drowned out or favoured by size;
    the result is scaled back to the average length the regions had."""

    def __init__(self, policy: GradientPolicy, normalize: bool = True) -> None:
        self.policy, self.normalize = policy, normalize

    def direction(self, grads: Tensor) -> Tensor | None:
        norms = grads.norm(dim=1, keepdim=True)
        if grads.shape[0] == 1:
            return grads[0]
        matrix = grads / norms.clamp(min=1e-12) if self.normalize else grads
        out = self.policy.combine(matrix)
        if out is None:
            return None
        length = out.norm()
        if self.normalize and length > 0:
            out = out * (norms.mean() / length)
        return out
