"""One update from the gradients of several timestep regions: reading them off the parameters, combining them with a
policy, and writing the result back for the optimizer."""

from __future__ import annotations

import torch
from torch import Tensor

from .policy import GradientPolicy, Mean


class RegionGradients:
    """The trainable parameters' gradients as one flat float32 vector, and back. The vectors live in host memory: one per
    region, they would otherwise take a copy of the adapter's gradients each from the card's memory (and the matrix a second
    copy), which grows with the number of regions."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.params = optimizer.params
        self.reached: set[int] = set()  # parameters some region's backward reached

    def read(self) -> Tensor:
        self.reached |= self.optimizer.touched()
        flat = torch.cat([(torch.zeros(p.numel(), device=p.device) if p.grad is None else p.grad.detach().float().flatten()) for p in self.params])
        return flat.cpu()

    def write(self, vector: Tensor) -> None:
        offset = 0
        for p in self.params:
            piece = vector[offset:offset + p.numel()].view_as(p).to(device=p.device, dtype=p.dtype)
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

    def streams(self, batches: int) -> bool:
        """Whether ``RegionStream`` gives the same update: the plain average of normalised regions over a window of one batch."""
        return isinstance(self.policy, Mean) and self.normalize and batches == 1

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


class RegionStream:
    """The ``equal`` update without ever holding the regions' gradients apart: as each region's backward finishes, its gradient
    is scaled to unit length and added to a running sum that stays on the device, so nothing is copied between host and device
    and the card holds one extra copy of the adapter's gradients whatever the number of regions.

    The update is the same as ``RegionalUpdate`` with the plain average and normalising: the sum scaled back to the average
    length the regions had. What the single vectors would have given and the sum cannot is the smallest pairwise cosine; the
    mean pairwise cosine is exact, since the sum's length is ``regions + 2 * (sum of the pairwise cosines)``."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self._sum: list[Tensor] | None = None
        self._norms: list[Tensor] = []
        self._reached: set[int] = set()

    def begin(self) -> None:
        if self._sum is not None:
            torch._foreach_zero_(self._sum)
        self._norms, self._reached = [], set()

    def add(self) -> None:
        """Fold in the gradients the last backward left on the parameters."""
        self._reached |= self.optimizer.touched()
        params = self.optimizer.params
        grads = [torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.detach().float() for p in params]
        if self._sum is None:
            self._sum = [torch.zeros_like(g) for g in grads]
        norm = torch.stack(torch._foreach_norm(grads)).norm()
        self._norms.append(norm)
        torch._foreach_add_(self._sum, torch._foreach_mul(grads, 1.0 / norm.clamp(min=1e-12)))

    def finish(self) -> tuple[list[float], dict[str, float]]:
        """Write the update direction onto the parameters; the regions' gradient norms and the metrics of the combination."""
        count = len(self._norms)
        norms = torch.stack(self._norms)
        length = torch.stack(torch._foreach_norm(self._sum)).norm()
        scale = torch.where(length > 0, norms.mean() / length.clamp(min=1e-30), torch.zeros_like(length))
        for p, piece in zip(self.optimizer.params, torch._foreach_mul(self._sum, scale)):
            piece = piece.to(p.dtype)
            if p.grad is None:
                p.grad = piece
            else:
                p.grad.copy_(piece)
        self.optimizer.touch(frozenset(self._reached))
        metrics = {}
        if count > 1:
            metrics["region_cos_mean"] = float((length ** 2 - count) / (count * (count - 1)))
        return [float(n) for n in norms.cpu()], metrics
