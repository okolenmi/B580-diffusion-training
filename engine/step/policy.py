"""How the gradients of several timestep regions become one update.

Each region's gradient is the gradient of that region's mean loss alone, over every trainable parameter flattened into one
vector. A policy turns the ``regions x parameters`` matrix into the one vector the optimizer sees. The vectors are read as
directions only: how large an update is stays the optimizer's business."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class GradientPolicy(ABC):
    """``combine`` returns the update direction; ``report`` the numbers about the last call that belong in the step's metrics."""

    name = ""

    def __init__(self) -> None:
        self.last: dict[str, float] = {}

    @abstractmethod
    def combine(self, grads: Tensor) -> Tensor | None:
        """One vector from the ``regions x parameters`` matrix, or None when no update should be made."""

    @staticmethod
    def cosines(grads: Tensor) -> Tensor:
        unit = grads / grads.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return unit @ unit.T

    def _agreement(self, grads: Tensor) -> None:
        """Smallest cosine between two regions' gradients (1 = all alike, below 0 = at least two pull against each other)."""
        n = grads.shape[0]
        if n > 1:
            cos = self.cosines(grads)
            self.last["region_cos_min"] = float(cos[~torch.eye(n, dtype=torch.bool, device=cos.device)].min())


class Mean(GradientPolicy):
    """The plain average: what one batch holding every region gives."""

    name = "mean"

    def combine(self, grads):
        self._agreement(grads)
        return grads.mean(dim=0)


class Project(GradientPolicy):
    """A region's gradient loses the part that points against another region's gradient (PCGrad), so no region's share of the
    update pushes another region's loss up to first order; part of the signal is thrown away for it."""

    name = "project"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._generator = torch.Generator().manual_seed(seed)

    def combine(self, grads):
        self._agreement(grads)
        n = grads.shape[0]
        out = grads.clone()
        removed = 0.0
        for i in range(n):
            for j in torch.randperm(n, generator=self._generator).tolist():
                if j == i:
                    continue
                dot = torch.dot(out[i], grads[j])
                if dot < 0:
                    out[i] -= dot / grads[j].pow(2).sum().clamp(min=1e-12) * grads[j]
                    removed += 1
        self.last["regions_projected"] = removed
        return out.mean(dim=0)


class CommonDescent(GradientPolicy):
    """The shortest vector in the convex hull of the regions' gradients (MGDA): to first order a small step against it lowers
    every region's loss, unless the regions' gradients cancel out. ``floor`` is the share of the average squared gradient
    norm the vector must keep; below it the regions disagree too much, no update is made and the step is skipped
    (0 never skips)."""

    name = "common"
    ITERATIONS = 100

    def __init__(self, floor: float = 0.0) -> None:
        super().__init__()
        self.floor = floor

    @classmethod
    def weights(cls, gram: Tensor) -> Tensor:
        """Weights on the simplex minimising ``w^T gram w`` (Frank-Wolfe with exact line search)."""
        n = gram.shape[0]
        w = torch.full((n,), 1.0 / n, dtype=gram.dtype, device=gram.device)
        for _ in range(cls.ITERATIONS):
            grad = gram @ w
            vertex = int(grad.argmin())
            direction = -w.clone()
            direction[vertex] += 1.0
            curvature = direction @ gram @ direction
            if curvature <= 1e-30:
                break
            step = float(torch.clamp(-(w @ gram @ direction) / curvature, 0.0, 1.0))
            if step == 0.0:
                break
            w = w + step * direction
        return w

    def combine(self, grads):
        self._agreement(grads)
        gram = grads @ grads.T
        w = self.weights(gram.double())
        direction = (w.to(grads.dtype)[:, None] * grads).sum(dim=0)
        shared = float(direction.pow(2).sum() / gram.diagonal().mean().clamp(min=1e-30))
        self.last["region_shared"] = shared  # 1: identical gradients; 1 / regions: orthogonal ones; 0: they cancel
        if shared < self.floor:
            return None
        return direction


POLICIES = {"mean": Mean, "equal": Mean, "project": Project, "common": CommonDescent}


def make_policy(name: str, floor: float = 0.0, seed: int = 0) -> GradientPolicy:
    if name in ("mean", "equal"):
        return Mean()
    if name == "project":
        return Project(seed)
    if name == "common":
        return CommonDescent(floor)
    raise ValueError(f"unknown gradient policy {name!r}; choose from {tuple(POLICIES)}")
