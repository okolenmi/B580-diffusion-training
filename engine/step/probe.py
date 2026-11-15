"""Is the adapter making some part of the timestep range worse than the frozen model?

The training loss is measured on whatever few samples a step held, so it cannot show a region getting worse than the
base model. The probe re-noises the same clean latents with the same noise at the same timesteps every time, in each
third of the range, once with the adapter live and once with it gated to zero (the base model). Differences between two
probe records are therefore the model's change, not sampling noise. The latents are training images: a healthy probe is
necessary, not sufficient (an adapter that memorises them looks fine here).

``probe_rel_t_*`` is adapter loss / base loss (1 = unchanged, above 1 = worse than not using the adapter);
``probe_drift_t_*`` is ``|pred_adapter - pred_base|^2 / |pred_base|^2``, how far the prediction moved whether or not it helped.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import Tensor

from .loss import T_THIRDS, per_sample_mse


@dataclass
class _Item:
    x0: Tensor
    mask: Tensor | None
    prompt: str
    size_px: tuple[int, int]


class TimeProbe:
    def __init__(self, *, denoiser, conditioner, gate, noiser, item_of: Callable[[str], tuple], device: str,
                 every: int, items: int = 2, points: int = 2, seed: int = 0, alignment=None):
        if every < 1 or items < 1 or points < 1:
            raise ValueError("every, items and points must each be at least 1")
        self.denoiser, self.conditioner, self.gate, self.noiser = denoiser, conditioner, gate, noiser
        self._item_of, self.device, self.every, self.items, self.points, self.seed = item_of, device, every, items, points, seed
        self._items: list[_Item] = []
        self._seen: set[str] = set()
        self._base: dict[tuple, tuple[float, Tensor]] | None = None
        self.alignment = alignment  # a GradientAlignment run with every record, or None

    @property
    def ready(self) -> bool:
        return len(self._items) >= self.items

    def capture(self, batch: dict) -> None:
        """Take the first samples seen as probe items, until there are enough."""
        for sample, prompt in zip(batch["samples"], batch["prompts"]):
            if self.ready:
                return
            if sample not in self._seen:
                self._seen.add(sample)
                x0, mask, size = self._item_of(sample)
                self._items.append(_Item(x0, mask, prompt, size))

    def due(self, step: int) -> bool:
        return self.ready and (step + 1) % self.every == 0

    def point(self, n: int, third: int, point: int) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Noisy input, timestep, target and mask of one probe point, on the device; the same ones every time."""
        item = self._items[n]
        frac = torch.tensor([(third + (point + 0.5) / self.points) / 3])
        t = self.noiser.t_at(frac)
        noise = torch.randn(item.x0.shape, generator=torch.Generator().manual_seed(self.seed + 1000 * n + 10 * third + point))
        x_t, target, _ = self.noiser.objective.noisy(item.x0, noise, t)
        mask = None if item.mask is None else item.mask.to(self.device)
        return x_t.to(self.device), t.to(self.device), target.to(self.device), mask

    def conditioning(self, n: int):
        item = self._items[n]
        return self.conditioner.encode([item.prompt], [item.size_px])

    def run(self) -> dict[str, float]:
        """Rel and drift per third (and the worst rel), in the metrics' naming."""
        live, base = {}, {}
        with torch.no_grad():
            for n in range(len(self._items)):
                conditioning = self.conditioning(n)
                for third in range(3):
                    for point in range(self.points):
                        key = (n, third, point)
                        x_t, t_dev, target, mask = self.point(n, third, point)
                        with self.gate.override(None):
                            pred = self.denoiser.predict(x_t, t_dev, conditioning)
                        live[key] = (float(per_sample_mse(pred, target, mask)), pred.float().cpu())
                        if self._base is None:
                            with self.gate.override(torch.zeros(1, device=self.device)):
                                pred0 = self.denoiser.predict(x_t, t_dev, conditioning)
                            base[key] = (float(per_sample_mse(pred0, target, mask)), pred0.float().cpu())
        if self._base is None:
            self._base = base
        out: dict[str, float] = {}
        for third, name in enumerate(T_THIRDS):
            keys = [k for k in live if k[1] == third]
            loss_live = sum(live[k][0] for k in keys)
            loss_base = sum(self._base[k][0] for k in keys)
            drift = sum(float((live[k][1] - self._base[k][1]).pow(2).sum() / self._base[k][1].pow(2).sum().clamp(min=1e-12)) for k in keys)
            suffix = name.removeprefix("loss_")
            out[f"probe_rel_{suffix}"] = loss_live / max(loss_base, 1e-12)
            out[f"probe_drift_{suffix}"] = drift / len(keys)
        out["probe_worst_rel"] = max(v for k, v in out.items() if k.startswith("probe_rel_"))
        if self.alignment is not None:
            out.update(self.alignment.run(self))
        return out
