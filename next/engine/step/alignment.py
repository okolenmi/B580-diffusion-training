"""Do the parts of the timestep range pull the adapter the same way?

For each third of the range, the gradient of the probe loss over that third's points with respect to the trainable
parameters. A third whose gradient has no positive cosine with the sum of all of them is one the combined update does
not help (to first order); strongly negative cosines between thirds are a conflict. Gradients come from a few samples,
so each third's gradient is also computed from two interleaved halves: their cosine (``gc_self_*``) is the noise floor,
and cross-third cosines near it say nothing. These are local statistics of the probe samples, meant to explain the
``probe_rel_*`` trajectory, not to replace it.

Keys: ``gc_norm_t_*``, ``gc_share_t_*`` (share of the summed norms), ``gc_self_t_*``, ``gc_cos_<a>_<b>``, ``gc_align_t_*``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from .loss import T_THIRDS, per_sample_mse


class GradientAlignment:
    def __init__(self, params: Callable[[], Sequence[nn.Parameter]], module: Callable[[], nn.Module]):
        self._params, self._module = params, module

    def run(self, probe) -> dict[str, float]:
        """``probe`` is the TimeProbe whose points are used. Leaves no gradient behind and the model in the mode it was in."""
        params = [p for p in self._params() if p.requires_grad]
        module = self._module()
        was_training = module.training
        module.eval()  # an adapter's dropout would make the same point give different gradients
        grads: dict[str, Tensor] = {}
        floor: dict[str, float] = {}
        try:
            conditioning = [probe.conditioning(n) for n in range(len(probe._items))]
            for third, name in enumerate(T_THIRDS):
                samples = [(n, third, point) for n in range(len(probe._items)) for point in range(probe.points)]
                if len(samples) >= 2:
                    first, second = self._gradient(probe, conditioning, params, samples[0::2]), self._gradient(probe, conditioning, params, samples[1::2])
                    norms = (first.norm().item(), second.norm().item())
                    if min(norms) > 0.0:
                        floor[name] = float(torch.dot(first, second)) / (norms[0] * norms[1])
                    share = len(samples[0::2]) / len(samples)
                    grads[name] = share * first + (1 - share) * second
                else:
                    grads[name] = self._gradient(probe, conditioning, params, samples)
        finally:
            module.train(was_training)
        return self._report(grads, floor)

    @staticmethod
    def _gradient(probe, conditioning, params: list[nn.Parameter], samples: list[tuple[int, int, int]]) -> Tensor:
        """Mean gradient of the loss over ``samples``, flattened, in float32 on the host."""
        total = torch.zeros(sum(p.numel() for p in params), dtype=torch.float32)
        for n, third, point in samples:
            x_t, t, target, mask = probe.point(n, third, point)
            with probe.gate.override(None):
                pred = probe.denoiser.predict(x_t, t, conditioning[n])
            loss = per_sample_mse(pred, target, mask).sum() / len(samples)
            found = torch.autograd.grad(loss, params, allow_unused=True)
            offset = 0
            for p, g in zip(params, found):
                if g is not None:
                    total[offset:offset + p.numel()] += g.detach().float().reshape(-1).cpu()
                offset += p.numel()
        return total

    @staticmethod
    def _report(grads: dict[str, Tensor], floor: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        norms = {n: float(g.norm()) for n, g in grads.items()}
        summed = sum(norms.values())
        for name, value in norms.items():
            suffix = name.removeprefix("loss_")
            out[f"gc_norm_{suffix}"] = value
            if summed > 0.0:
                out[f"gc_share_{suffix}"] = value / summed
        out.update({f"gc_self_{n.removeprefix('loss_')}": v for n, v in floor.items()})
        present = [n for n in T_THIRDS if norms.get(n, 0.0) > 0.0]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                out[f"gc_cos_{a.removeprefix('loss_t_')}_{b.removeprefix('loss_t_')}"] = float(torch.dot(grads[a], grads[b])) / (norms[a] * norms[b])
        if len(present) >= 2:
            total = sum(grads[n] for n in present)
            size = float(total.norm())
            if size > 0.0:
                for n in present:
                    out[f"gc_align_{n.removeprefix('loss_')}"] = float(torch.dot(grads[n], total)) / (norms[n] * size)
        return out
