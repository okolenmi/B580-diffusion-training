"""Self-perceptual loss: the model's own prediction judged by what the frozen base network sees in it.

The squared error on the noise treats every position and every kind of mistake alike. Here the model's prediction is turned
into an image estimate, both that estimate and the true image are noised again at a fresh timestep, and the frozen network
(the same weights with the adapter switched off) compares the two by its middle-block features. Mistakes that change what the
network recognises count; ones it does not care about count little. Only eps prediction is supported: the estimate is
``x0 = x - sigma * eps`` in the sampler's convention."""

from __future__ import annotations

import torch
from torch import Tensor

from .phases import StepPhase
from .state import StepState


class SelfPerceptual:
    """The feature distance for one batch; ``objective`` noises, ``gate`` switches the adapter, ``denoiser`` gives the features."""

    def __init__(self, denoiser, objective, gate, seed: int = 0):
        if gate is None:
            raise ValueError("the self-perceptual loss compares against the model without its adapter: it needs an adapter with a gate")
        self._denoiser, self._objective, self._gate = denoiser, objective, gate
        self._generator = torch.Generator().manual_seed(seed)

    def distance(self, state: StepState) -> Tensor:
        """Mean over the batch of the relative squared feature distance, each sample weighted by ``1 / (sigma^2 + 1)``.

        The weight keeps the image estimate's error in units of the data: without it that error grows with sigma and the
        noisy end would decide the whole term."""
        i = state.inputs
        sigma = i.sigma.float().view(-1, *([1] * (i.x_t.ndim - 1)))
        x = i.x_t.float() * (sigma ** 2 + 1.0).sqrt()
        eps, eps_hat = i.target.float(), state.prediction.float()
        if i.mask is not None:
            eps_hat = torch.where(i.mask.bool().expand_as(eps_hat), eps_hat, eps)
        x0, x0_hat = x - sigma * eps, x - sigma * eps_hat
        t = self._objective.sample_t(x.shape[0], "uniform", *self._objective.t_range, self._generator).to(x.device)
        true_input = self._objective.noisy(x0, eps, t)[0]
        pred_input = self._objective.noisy(x0_hat, eps_hat, t)[0]
        with self._gate.override(torch.zeros(1, device=x.device)):
            with torch.no_grad():
                real = self._denoiser.mid_features(true_input, t, state.conditioning).float()
            pred = self._denoiser.mid_features(pred_input, t, state.conditioning).float()
        error = (pred - real).pow(2).flatten(1).mean(dim=1)
        scale = real.pow(2).flatten(1).mean(dim=1).clamp(min=1e-8)
        return ((error / scale) / (sigma.flatten() ** 2 + 1.0)).mean()


class SelfPerceptualLoss(StepPhase):
    """Puts ``weight`` times the self-perceptual distance on the state's penalty, which the backward adds to the loss."""

    def __init__(self, term: SelfPerceptual, weight: float):
        if weight <= 0:
            raise ValueError("the self-perceptual weight must be positive")
        self._term, self.weight = term, weight
        self._seen: list[float] = []

    def run(self, state: StepState) -> None:
        distance = self._term.distance(state)
        state.penalty = distance * self.weight
        self._seen.append(float(distance.detach()))

    def report(self) -> dict[str, float]:
        """The mean distance of the passes since the last report (not multiplied by the weight)."""
        if not self._seen:
            return {}
        mean, self._seen = sum(self._seen) / len(self._seen), []
        return {"loss_perceptual": mean}
