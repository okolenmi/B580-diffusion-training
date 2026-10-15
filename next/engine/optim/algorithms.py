"""Update rules. Each works on a *stack* of same-shaped parameters at once: ``grad`` and every state tensor
carry a leading member axis, and a single parameter is a stack of one. The rule updates the state in place
and returns the amount to subtract from the parameters, plus a multiplicative weight decay (or None).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor


def _members(vector: Tensor | float, ndim: int) -> Tensor | float:
    """A per-member value shaped to broadcast against a stacked tensor of ``ndim`` dimensions."""
    return vector.view(-1, *([1] * (ndim - 1))) if isinstance(vector, Tensor) else vector


def _rms(stack: Tensor) -> Tensor:
    flat = stack.reshape(stack.shape[0], -1)
    return flat.norm(dim=1) / (flat.shape[1] ** 0.5 + 1e-8)


def _rows_cols(shape: Sequence[int]) -> tuple[int, int]:
    return shape[0], math.prod(shape[1:])


class Algorithm(ABC):
    name = ""

    def begin_step(self, n: int = 1) -> None:
        """Called once per optimizer step, before any update."""

    @abstractmethod
    def state_shapes(self, shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """State tensors for one parameter of ``shape`` (without the member axis)."""

    @abstractmethod
    def update(self, grad: Tensor, state: dict[str, Tensor], lr: float, params: Sequence[Tensor]) -> tuple[Tensor, Tensor | float | None]:
        """(delta, decay): the parameters become ``param * decay - delta``."""

    def decay_state(self, state: dict[str, Tensor], factor: float) -> None:
        for tensor in state.values():
            tensor.zero_() if factor <= 0 else tensor.mul_(factor)

    def reset_state(self, state: dict[str, Tensor]) -> None:
        for tensor in state.values():
            tensor.zero_()


class AdamW(Algorithm):
    name = "adamw"

    def __init__(self, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        self.beta1, self.beta2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.t = 0
        self._correction = 1.0

    def begin_step(self, n: int = 1) -> None:
        self.t += n
        self._correction = math.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

    def state_shapes(self, shape):
        return {"m": shape, "v": shape}

    def update(self, grad, state, lr, params):
        m, v = state["m"], state["v"]
        m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
        delta = (m / v.sqrt().add(self.eps)).mul_(lr * self._correction)
        return delta, (1.0 - lr * self.weight_decay) if self.weight_decay else None


class Adafactor(Algorithm):
    """Factored second moment for matrices; ``beta1`` adds a full first moment; ``scale_parameter`` scales the
    step by each parameter's own RMS (per member, so a stack behaves like its members one by one)."""

    name = "adafactor"

    def __init__(self, eps=(1e-8, 1e-3), clip_threshold: float = 1.0, beta1: float | None = None,
                 scale_parameter: bool = False, weight_decay: float = 0.0, tiny_parameter_threshold: int | None = None):
        self.eps1, self.eps2 = eps
        self.clip_threshold, self.beta1 = clip_threshold, beta1
        self.scale_parameter, self.weight_decay = scale_parameter, weight_decay
        self.tiny_parameter_threshold = tiny_parameter_threshold
        self.t = 0
        self._rho: float | None = None

    def factored(self, shape: Sequence[int]) -> bool:
        if len(shape) < 2:
            return False
        return self.tiny_parameter_threshold is None or math.prod(shape) >= self.tiny_parameter_threshold

    def begin_step(self, n: int = 1) -> None:
        self.t += n
        self._rho = max(1e-4, 1.0 - self.t ** -0.8)

    def state_shapes(self, shape):
        if self.factored(shape):
            rows, cols = _rows_cols(shape)
            state = {"vr": (rows,), "vc": (cols,)}
        else:
            state = {"vs": tuple(shape)}
        if self.beta1 is not None:
            state["exp_avg"] = tuple(shape)
        return state

    def update(self, grad, state, lr, params):
        if self._rho is None:
            raise RuntimeError("Adafactor.update() before begin_step(): the optimizer must call begin_step first")
        rho, k, shape = self._rho, grad.shape[0], grad.shape[1:]
        g = grad * _members(torch.clamp(self.clip_threshold / _rms(grad), max=1.0), grad.dim())
        if self.scale_parameter:
            p_rms = torch.stack([_rms(p.detach().float().unsqueeze(0))[0] for p in params])
            alpha = torch.clamp(p_rms.pow(2), min=max(self.eps1, self.eps2 ** 2)) * lr
        else:
            alpha = max(self.eps1, 1.0) * lr
        if self.factored(shape):
            rows, cols = _rows_cols(shape)
            gm = g.reshape(k, rows, cols)
            g2 = gm.pow(2)
            state["vr"].mul_(rho).add_(g2.mean(dim=2).add(self.eps1), alpha=1.0 - rho)
            state["vc"].mul_(rho).add_(g2.mean(dim=1).add(self.eps1), alpha=1.0 - rho)
            mean_root = state["vr"].mean(dim=1).add(self.eps1).sqrt()
            normalized = (gm / state["vr"].sqrt().add(self.eps1).unsqueeze(2) / state["vc"].sqrt().add(self.eps1).unsqueeze(1)
                          * mean_root.view(k, 1, 1)).reshape(grad.shape)
        else:
            state["vs"].mul_(rho).add_(g.pow(2).add(self.eps1), alpha=1.0 - rho)
            normalized = g / state["vs"].sqrt().add(self.eps1)
        if self.beta1 is not None:
            state["exp_avg"].mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)
            normalized = state["exp_avg"].clone()
        delta = normalized * _members(alpha, grad.dim())
        decay = None
        if self.weight_decay:
            decay = 1.0 - self.weight_decay * alpha
        return delta, decay


class CAME(Algorithm):
    """Confidence-guided Adafactor: the first moment is normalized by a factored estimate of its own error."""

    name = "came"

    def __init__(self, eps=(1e-30, 1e-16), clip_threshold: float = 1.0, betas=(0.9, 0.999, 0.9999),
                 weight_decay: float = 0.0):
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.beta1, self.beta2, self.beta3 = betas
        self.weight_decay = weight_decay

    def state_shapes(self, shape):
        if len(shape) >= 2:
            rows, cols = _rows_cols(shape)
            return {"r": (rows,), "c": (cols,), "ea": tuple(shape), "rr": (rows,), "rc": (cols,)}
        return {"s": tuple(shape), "ea": tuple(shape)}

    def _clip(self, normalized: Tensor) -> Tensor:
        return normalized / _members(torch.clamp(_rms(normalized) / self.clip_threshold, min=1.0), normalized.dim())

    def update(self, grad, state, lr, params):
        k, shape = grad.shape[0], grad.shape[1:]
        decay = (1.0 - self.weight_decay * lr) if self.weight_decay else None
        if len(shape) < 2:
            state["s"].mul_(self.beta2).add_(grad.pow(2).add(self.eps1), alpha=1.0 - self.beta2)
            normalized = self._clip(grad / state["s"].sqrt().add(self.eps1))
            state["ea"].mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)
            return state["ea"].clone() * lr, decay
        rows, cols = _rows_cols(shape)
        g = grad.reshape(k, rows, cols)
        g2 = g.pow(2).add(self.eps1)
        state["r"].mul_(self.beta2).add_(g2.mean(dim=2), alpha=1.0 - self.beta2)
        state["c"].mul_(self.beta2).add_(g2.mean(dim=1), alpha=1.0 - self.beta2)
        normalized = (g / state["r"].sqrt().add(self.eps1).unsqueeze(2) / state["c"].sqrt().add(self.eps1).unsqueeze(1)
                      * state["r"].mean(dim=1).add(self.eps1).sqrt().view(k, 1, 1))
        normalized = self._clip(normalized)
        ea = state["ea"].view(k, rows, cols)
        ea.mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)
        residual = (normalized - ea).pow(2).add(self.eps2)
        state["rr"].mul_(self.beta3).add_(residual.mean(dim=2), alpha=1.0 - self.beta3)
        state["rc"].mul_(self.beta3).add_(residual.mean(dim=1), alpha=1.0 - self.beta3)
        update = (ea / state["rr"].sqrt().add(self.eps1).unsqueeze(2) / state["rc"].sqrt().add(self.eps1).unsqueeze(1)
                  * state["rr"].mean(dim=1).add(self.eps1).sqrt().view(k, 1, 1))
        return (update * lr).reshape(grad.shape), decay


ALGORITHMS: dict[str, type[Algorithm]] = {"adamw": AdamW, "adafactor": Adafactor, "came": CAME}
