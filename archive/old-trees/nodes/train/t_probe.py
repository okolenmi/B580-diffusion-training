"""TProbe: a *fixed* per-timestep probe of what training is doing to the
model, plus an optional per-bucket gradient-alignment diagnostic.

Why this exists. The per-bucket losses the monitor already charts
(loss.py's t_bucket_losses) are measured on whatever 1-2 random samples
the training batches happened to contain: different images, different
noise, different t inside the bucket, every step. That is fine for "is the
loss going down", but it cannot answer the question that decides whether a
LoRA is usable at inference -- *did this run make any timestep region
worse than the frozen base was?* -- because a batch-to-batch difference of
0.05 in a bucket's loss is indistinguishable from sampling noise, and a
fit-the-training-data loss cannot see collateral damage anyway.

Two independent tools, both off unless wired:

1. **Fixed-probe evaluation** (evaluate()). A handful of probe items
   (clean latent + cached conditioning, captured from the first batches the
   trainer sees) are re-noised with *fixed* seeded noise at a fixed grid of
   timesteps (points_per_bucket per T_BUCKETS third), forward-only. Every
   call sees exactly the same (x0, noise, t, conditioning), so a change
   between two calls is the model's change, not the sampler's. Each probe
   point is evaluated twice: with the LoRA live, and with the LoRA gated
   to exactly zero (the frozen base -- lora.py's gate=0 contract; cached
   after the first call because the base never changes). Reported per
   bucket:

     probe_t_*        raw eps/v MSE with the LoRA live
     probe_rel_t_*    that MSE / the frozen base's MSE on the same probe
                      inputs. > 1 means the LoRA made this region *worse
                      than not having it*; 1.0 = no change
     probe_drift_t_*  ||pred_lora - pred_base||^2 / ||pred_base||^2 -- how
                      far the LoRA moved the model's output, independent of
                      whether that movement helped
     probe_worst_rel  max over buckets of probe_rel_t_* -- the single
                      number that matters if one damaged region is enough
                      to ruin a sample

   Cost: forward-only, batch 1: (n_items * 3 * points_per_bucket) forwards
   on first call (base + LoRA), half that after.

2. **Gradient alignment** (alignment()). For each t bucket, the gradient of
   the probe loss w.r.t. the trainable parameters (mean over that bucket's
   probe points). Reports each bucket's gradient norm, the pairwise
   cosines between buckets, and each bucket's cosine with the *sum* of all
   bucket gradients (the direction a uniform-bucket-weight step would
   take). A bucket whose cosine with the sum is <= 0 is one the combined
   update does not help to first order; two buckets with a strongly
   negative pairwise cosine are in conflict. This is the direct test of
   "does a big win in one t region destroy another".

   The honest caveat is built into the output: every bucket's gradient is
   estimated from only n_items * points_per_bucket samples, so cosines
   between *different* buckets are only meaningful if each bucket's
   gradient agrees with *itself* across independent halves. alignment()
   reports that split-half self-cosine (gc_self_*); read cross-bucket
   numbers relative to it, not to +-1. If gc_self_* is ~0.05, then
   cross-bucket cosines of +-0.05 are noise.

   Cost: one forward+backward per probe point. Requires a non-fused
   optimizer -- a fused optimizer applies updates inside backward() hooks,
   so a diagnostic backward would silently train on probe data. Also
   clobbers .grad, so it must run only after the optimizer step and before
   the next zero_grad (ProbePhase's position in the managed pipeline).

Neither tool changes training arithmetic: evaluate() is torch.no_grad()
and alignment() runs after the step. Both leave the model in train mode
(the only mode the trainer ever runs it in). The training-time gate
(lora._current_gate) is saved and restored around both, and the
probes always run with the gate off -- i.e. as inference would apply the
LoRA (full strength at every t), which is the behavior being judged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..components.diffusion import DiffusionProcess, EpsParameterization
from .loss import T_BUCKETS


@dataclass
class ProbeItem:
    x0: torch.Tensor        # (1, C, H, W) clean latent, fp32, on the training device
    ctx_emb: torch.Tensor   # (1, N, 2048) cached text conditioning
    y: torch.Tensor         # (1, 2816) pooled text + size embedding


def _bucket_points(points_per_bucket: int) -> list[tuple[str, int]]:
    """[(bucket_name, t)] -- points_per_bucket evenly spaced t's strictly
    inside each T_BUCKETS third (midpoints of equal sub-intervals, so no
    point sits on a bucket edge and t=0 is never used)."""
    out: list[tuple[str, int]] = []
    for name, lo, hi in T_BUCKETS:
        for k in range(points_per_bucket):
            t = int(lo + (hi - lo) * (k + 0.5) / points_per_bucket)
            out.append((name, max(1, min(999, t))))
    return out


class TProbe:
    """See the module docstring. One instance per run; stateful (holds the
    probe items and the cached base-model reference)."""

    def __init__(self, n_items: int = 2, points_per_bucket: int = 2,
                 seed: int = 1234, grad_alignment: bool = False):
        if n_items < 1:
            raise ValueError(f"n_items must be >= 1, got {n_items}")
        if points_per_bucket < 1:
            raise ValueError(f"points_per_bucket must be >= 1, got {points_per_bucket}")
        self.n_items = int(n_items)
        self.points_per_bucket = int(points_per_bucket)
        self.seed = int(seed)
        self.grad_alignment = bool(grad_alignment)
        self._items: list[ProbeItem] = []
        self._points = _bucket_points(self.points_per_bucket)
        self._noise: dict[tuple[int, int], torch.Tensor] = {}
        # (item_idx, t) -> (base_pred, base_loss): the frozen base never
        # changes, so it is computed once.
        self._base: dict[tuple[int, int], tuple[torch.Tensor, float]] = {}

    # ---- collection ---------------------------------------------------------------

    def wants_items(self) -> bool:
        return len(self._items) < self.n_items

    def ready(self) -> bool:
        return len(self._items) >= self.n_items

    def collect(self, process: DiffusionProcess, x_t, target, t, sigma, ctx_emb, y) -> None:
        """Capture element 0 of a training batch as a probe item.

        x0 is *recovered* from the batch (x_t, target, sigma) through the
        run's own Parameterization.to_x0 -- exact for both eps and v
        targets (x_t = x0 + sigma*eps is the dataset convention), so the
        loader/dataset need no changes and the probe re-noises the same
        clean latent training saw.
        """
        if not self.wants_items():
            return
        with torch.no_grad():
            sig4 = sigma.detach().float().reshape(-1)[:1].view(1, 1, 1, 1)
            alpha = None  # unused by both Parameterization.to_x0 implementations
            x0 = process.parameterization.to_x0(
                target.detach()[:1].float(), x_t.detach()[:1].float(), alpha, sig4)
            self._items.append(ProbeItem(
                x0=x0.clone(),
                ctx_emb=ctx_emb.detach()[:1].clone(),
                y=y.detach()[:1].clone(),
            ))

    # ---- one probe point ----------------------------------------------------------

    def _noise_for(self, item_idx: int, t: int, like: torch.Tensor) -> torch.Tensor:
        key = (item_idx, t)
        if key not in self._noise:
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed + 1_000_003 * item_idx + 7919 * t)
            self._noise[key] = torch.randn(like.shape, generator=g).to(
                device=like.device, dtype=torch.float32)
        return self._noise[key]

    def _forward_point(self, model, process: DiffusionProcess, item_idx: int, t: int):
        """(pred, target) for one fixed probe point, using whatever gate
        is currently installed. Batch of 1."""
        item = self._items[item_idx]
        device = item.x0.device
        eps = self._noise_for(item_idx, t, item.x0)
        t_idx = torch.tensor([t], device=device, dtype=torch.long)
        alpha, sigma = process.schedule.alpha_sigma(t_idx)
        sig4 = sigma.float().view(-1, 1, 1, 1)
        x_t = item.x0 + sig4 * eps
        if isinstance(process.parameterization, EpsParameterization):
            target = eps
        else:
            target = EpsParameterization().convert_to(
                eps, x_t, alpha, sig4, process.parameterization)
        xc = process.input_transform.scale_input(x_t, sigma)
        pred = model.forward(xc, t_idx, item.ctx_emb, item.y)
        return pred, target

    @staticmethod
    def _mse(pred, target) -> torch.Tensor:
        return (pred.float() - target.float()).pow(2).mean()

    # ---- fixed-probe evaluation -----------------------------------------------------

    def evaluate(self, model, process: DiffusionProcess) -> tuple[dict[str, float], list[tuple[int, float, float]]]:
        """Forward-only probe. Returns (report_keys, detail) where detail is
        [(t, mean_lora_loss, mean_base_loss)] per grid t, averaged over
        items, for a human-readable printout."""
        if not self.ready():
            return {}, []
        from ..model.lora import lora_gate_override

        model.eval()
        try:
            per_bucket_loss: dict[str, list[float]] = {}
            per_bucket_base: dict[str, list[float]] = {}
            per_bucket_drift: dict[str, list[float]] = {}
            per_t: dict[int, list[tuple[float, float]]] = {}
            with torch.no_grad():
                for bucket, t in self._points:
                    for i in range(len(self._items)):
                        key = (i, t)
                        if key not in self._base:
                            with lora_gate_override(torch.zeros(1)):
                                bp, tgt = self._forward_point(model, process, i, t)
                            self._base[key] = (bp.detach().clone(), float(self._mse(bp, tgt)))
                        base_pred, base_loss = self._base[key]
                        with lora_gate_override(None):
                            lp, tgt = self._forward_point(model, process, i, t)
                        lora_loss = float(self._mse(lp, tgt))
                        bp32, lp32 = base_pred.float(), lp.float()
                        drift = float((lp32 - bp32).pow(2).sum()
                                      / bp32.pow(2).sum().clamp_min(1e-12))
                        per_bucket_loss.setdefault(bucket, []).append(lora_loss)
                        per_bucket_base.setdefault(bucket, []).append(base_loss)
                        per_bucket_drift.setdefault(bucket, []).append(drift)
                        per_t.setdefault(t, []).append((lora_loss, base_loss))
        finally:
            # Unconditional: the TrainableModel contract has no `training`
            # attribute to query (ComfyUNetTrainableModel exposes only
            # train()/eval()), and this probe only ever runs inside a
            # trainer that has put the model in train mode (build() calls
            # model.train() up front). Guessing "was it training?" via
            # getattr would return None on the real wrapper and leave it in
            # eval mode -- silently disabling LoRA dropout for the rest of
            # the run.
            model.train()

        report: dict[str, float] = {}
        rels: list[float] = []
        for name, _, _ in T_BUCKETS:
            if name not in per_bucket_loss:
                continue
            suffix = name.replace("loss_t_", "t_")
            lora = sum(per_bucket_loss[name]) / len(per_bucket_loss[name])
            base = sum(per_bucket_base[name]) / len(per_bucket_base[name])
            drift = sum(per_bucket_drift[name]) / len(per_bucket_drift[name])
            rel = lora / base if base > 0.0 else float("nan")
            report[f"probe_{suffix}"] = lora
            report[f"probe_rel_{suffix}"] = rel
            report[f"probe_drift_{suffix}"] = drift
            if math.isfinite(rel):
                rels.append(rel)
        if rels:
            report["probe_worst_rel"] = max(rels)
        detail = [(t, sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v))
                  for t, v in sorted(per_t.items())]
        return report, detail

    # ---- per-bucket gradient alignment ----------------------------------------------

    def _bucket_grad(self, model, process, params, samples) -> Optional[list[torch.Tensor]]:
        """Mean gradient (per-param fp32 CPU tensors) of the probe loss over
        `samples` [(item_idx, t)]. None when samples is empty."""
        if not samples:
            return None
        for p in params:
            p.grad = None
        for i, t in samples:
            pred, target = self._forward_point(model, process, i, t)
            (self._mse(pred, target) / len(samples)).backward()
        out = []
        for p in params:
            out.append(torch.zeros(p.shape, dtype=torch.float32) if p.grad is None
                       else p.grad.detach().float().cpu())
        return out

    @staticmethod
    def _dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
        return float(sum((x.double() * y.double()).sum() for x, y in zip(a, b)))

    def alignment(self, model, process: DiffusionProcess, params) -> dict[str, float]:
        """Per-bucket gradient norms / cosines. See the module docstring for
        how to read them (and the gc_self_* noise floor). Leaves every
        param's .grad as None on exit."""
        if not self.ready():
            return {}
        from ..model.lora import lora_gate_override

        params = [p for p in params if p.requires_grad]
        names = [name for name, _, _ in T_BUCKETS]
        grads: dict[str, list[torch.Tensor]] = {}
        self_cos: dict[str, float] = {}

        model.eval()
        try:
            with lora_gate_override(None):
                for name in names:
                    samples = [(i, t) for (b, t) in self._points if b == name
                               for i in range(len(self._items))]
                    if not samples:
                        continue
                    if len(samples) >= 2:
                        # Independent halves (interleaved, so both cover every t
                        # in the bucket) -> the bucket's own reliability.
                        half_a, half_b = samples[0::2], samples[1::2]
                        ga = self._bucket_grad(model, process, params, half_a)
                        gb = self._bucket_grad(model, process, params, half_b)
                        na, nb = math.sqrt(self._dot(ga, ga)), math.sqrt(self._dot(gb, gb))
                        if na > 0.0 and nb > 0.0:
                            self_cos[name] = self._dot(ga, gb) / (na * nb)
                        wa, wb = len(half_a) / len(samples), len(half_b) / len(samples)
                        grads[name] = [wa * x + wb * y for x, y in zip(ga, gb)]
                        del ga, gb
                    else:
                        grads[name] = self._bucket_grad(model, process, params, samples)
        finally:
            for p in params:
                p.grad = None
            model.train()  # unconditional -- see evaluate()'s finally

        report: dict[str, float] = {}
        norms = {n: math.sqrt(self._dot(g, g)) for n, g in grads.items()}
        total_norm_sum = sum(norms.values())
        for n, v in norms.items():
            suffix = n.replace("loss_t_", "t_")
            report[f"gc_norm_{suffix}"] = v
            if total_norm_sum > 0.0:
                report[f"gc_share_{suffix}"] = v / total_norm_sum
        for n, v in self_cos.items():
            report[f"gc_self_{n.replace('loss_t_', 't_')}"] = v
        present = [n for n in names if n in grads and norms[n] > 0.0]
        for a_i in range(len(present)):
            for b_i in range(a_i + 1, len(present)):
                a, b = present[a_i], present[b_i]
                cos = self._dot(grads[a], grads[b]) / (norms[a] * norms[b])
                sa, sb = a.replace("loss_t_", ""), b.replace("loss_t_", "")
                report[f"gc_cos_{sa}_{sb}"] = cos
        if len(present) >= 2:
            total = [sum(parts) for parts in zip(*(grads[n] for n in present))]
            tnorm = math.sqrt(self._dot(total, total))
            if tnorm > 0.0:
                for n in present:
                    report[f"gc_align_{n.replace('loss_t_', 't_')}"] = (
                        self._dot(grads[n], total) / (norms[n] * tnorm))
        return report


def format_probe_line(step: int, report: dict[str, float], detail) -> str:
    """One human-readable console line for a probe result."""
    parts = [f"[probe step {step}]"]
    for name, _, _ in T_BUCKETS:
        s = name.replace("loss_t_", "t_")
        if f"probe_rel_{s}" in report:
            parts.append(f"{s}: rel={report[f'probe_rel_{s}']:.3f} "
                         f"loss={report[f'probe_{s}']:.4f} drift={report[f'probe_drift_{s}']:.4f}")
    if "probe_worst_rel" in report:
        parts.append(f"worst_rel={report['probe_worst_rel']:.3f}")
    line = "  ".join(parts)
    if detail:
        line += "\n    per-t (lora/base): " + " ".join(
            f"t={t}:{(a / b if b > 0 else float('nan')):.2f}" for t, a, b in detail)
    gc_keys = sorted(k for k in report if k.startswith("gc_"))
    if gc_keys:
        line += "\n    grad-align: " + " ".join(f"{k[3:]}={report[k]:.3f}" for k in gc_keys)
    return line
