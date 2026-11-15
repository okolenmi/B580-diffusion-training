"""Anchor loss: the adapter may not change what the base network predicts under other prompts.

The training images are noised as usual, but the network is also asked about them under prompts that have nothing to do with the
dataset, once with the adapter and once without. The squared difference of the two predictions is the penalty: whatever the adapter
learned has to stay quiet for prompts the training never showed, while the training loss still shapes it for the dataset's own.
The extra pass runs after the main backward (its own forward and backward, gradients add up), so only one network graph is alive at a time."""

from __future__ import annotations

import torch
from torch import Tensor

from .phases import StepPhase
from .state import StepState

PROMPTS = ("a portrait of a woman, detailed face", "a red house on a green hill under a blue sky", "a photo of a dog running on grass",
           "a city street at night in the rain", "a bowl of fruit on a wooden table", "a mountain lake at sunrise",
           "an astronaut riding a horse", "a close-up of a mechanical watch")


class AnchorLoss(StepPhase):
    """Every ``every`` steps, backward of ``weight`` times the squared difference between the adapter's and the base network's predictions under a random anchor prompt per sample."""

    def __init__(self, denoiser, conditioner, gate, prompts: tuple[str, ...], weight: float, every: int, accumulation: int = 1, seed: int = 0):
        if gate is None:
            raise ValueError("the anchor loss compares against the model without its adapter: it needs an adapter with a gate")
        if weight <= 0 or every < 1 or not prompts:
            raise ValueError("the anchor loss needs a positive weight, an interval of at least 1 step and at least one prompt")
        self._denoiser, self._conditioner, self._gate = denoiser, conditioner, gate
        self.prompts, self.weight, self.every, self._seed = tuple(prompts), weight, every, seed
        self._scale = 1.0 / accumulation
        self._seen: list[float] = []
        conditioner.extra.update(self.prompts)  # kept encoded, so no pass has to load the text encoders

    def run(self, state: StepState) -> None:
        if state.step % self.every:
            return
        i = state.inputs
        count = i.x_t.shape[0]
        generator = torch.Generator().manual_seed(self._seed + state.step * 1009 + state.micro)  # the same picks after a resume
        picks = [self.prompts[k] for k in torch.randint(0, len(self.prompts), (count,), generator=generator).tolist()]
        conditioning = self._conditioner.encode(picks, state.batch["sizes_px"])
        with torch.no_grad(), self._gate.override(torch.zeros(1, device=i.x_t.device)):
            base = self._denoiser.predict(i.x_t, i.t, conditioning)
        distance = self.distance(self._denoiser.predict(i.x_t, i.t, conditioning), base)
        (distance * (self.weight * self._scale * state.scale)).backward()
        self._seen.append(float(distance.detach()))

    @staticmethod
    def distance(adapted: Tensor, base: Tensor) -> Tensor:
        return (adapted.float() - base.float().detach()).pow(2).mean()

    def report(self) -> dict[str, float]:
        """The mean distance since the last report (not multiplied by the weight)."""
        if not self._seen:
            return {}
        mean, self._seen = sum(self._seen) / len(self._seen), []
        return {"loss_anchor": mean}
