"""A9 matrix (imported by a9_many_prompts_prewarm.py --matrix, not run alone).

Faithful managed-route runs under memory pressure with a many-prompt
dataset, at empty_cache_every_n_steps=1 vs 100:

* real SDXL CLIP fp16 on XPU, pre-wrapped in CachingTextEncoder with
  max_entries=8 (the LRU then thrashes on a 60-prompt source -- every
  step past the warmed head misses, the many-prompts condition);
* real ManagedLoRATrainerNode loop, real BudgetedResourceControlHandle,
  real ComposedAdamW on a 150M-param fake (fp32 states ~1.2 GB -- the
  optimizer churn is real hundreds-of-MB traffic when released, which
  a tiny fake would not reproduce, and allocator fragmentation needs
  real moving pieces, not a static standing allocation);
* static ballast (pinned, never moves) standing in for the frozen UNet
  base, sized so the total approaches the card;
* budget = standing + 0.5 GB: fits the CLIP-free calibration peak, not
  a CLIP-resident miss -- the "calibrated fine, failed partway" regime
  AdaptiveResidencyController's own docstring describes.

Source order is hits-first (the 8 warmed prompts, then the 52 others
cycling): calibration sees hits, the misses start mid-run, so any
escalation is miss-driven. One extra arm runs misses-from-step-0
(the tight-from-the-start regime).

Each arm records, per step: vram_reserved_mb, vram_peak_reserved_mb
(the monitor's own series), wall seconds, and cumulative misses.
OOMs are caught per arm and recorded, not fatal to the matrix.
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.components.device import DeviceContext
from nodes.core import ExecutionContext
from nodes.memory.control_handle import BudgetedResourceControlHandle
from nodes.model.handle import TrainableModel
from nodes.model.text_encoder_cache import CachingTextEncoder
from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
from nodes.resource_budget import ResourceBudget
from nodes.train.loss import UniformLossWeighting
from nodes.train.managed import ManagedLoRATrainerNode
from nodes.train.schedule import ConstantLRSchedule

PROMPTS = ([f"a photo of object number {i}, dramatic lighting, high detail"
             for i in range(52)] +
           [f"zz_top calibration still life {i}, studio light"
            for i in range(8)])
#: the warmed head sorts AFTER the tail, so after discovery warms the
#: sorted key set into an 8-entry LRU, exactly these 8 are retained --
#: calibration (which runs on the head first) sees hits, the tail misses.
WARMED_HEAD = PROMPTS[52:]
TAIL = PROMPTS[:52]
WARMED_HEAD_N = 8
STEPS = 25


class _XPUFakeModel(TrainableModel):
    """150M fp32 params on XPU: AdamW states ~1.2 GB, the real churn
    size when the controller releases the optimizer. forward() is
    trivial but device-real (xc arrives on XPU, p lives on XPU)."""

    def __init__(self):
        self.p = torch.nn.Parameter(
            torch.randn(150_000_000, device="xpu") * 0.01)

    def forward(self, xc, t, ctx_emb, y):
        return xc + self.p.sum()

    def trainable_parameters(self):
        return [self.p]

    def train(self):
        return self

    def eval(self):
        return self

    def to(self, device=None, **kwargs):
        return self

    def trained_state_dict(self):
        return {}

    def footprint_bytes(self):
        return self.p.numel() * self.p.element_size()

    def offload(self):
        raise AssertionError("model must never be offloaded by design")

    def reload(self, device=None):
        pass

    def release(self):
        pass


class _ManyPrompts:
    """Finite, re-iterable. hits_first=True: the warmed head first (so
    calibration sees hits), then the unwarmed tail cycling -- misses
    start mid-run. False: tail first (tight from step 0)."""

    def __init__(self, warmed: list[str], tail: list[str], hits_first=True):
        self._warmed = warmed
        self._tail = tail
        self._hits_first = hits_first

    def _batch(self, prompt):
        return {"x_t": torch.randn(1, 4, 32, 32, device="xpu"),
                "target": torch.randn(1, 4, 32, 32, device="xpu"),
                "t": torch.tensor([500]),
                "prompt": prompt}

    def __iter__(self):
        first, second = (self._warmed, self._tail) if self._hits_first \
            else (self._tail, self._warmed)
        if not second:
            # control arm: everything is warmed, cycle the head
            second = first
        for p in first:
            yield self._batch(p)
        i = 0
        # more than STEPS so the run never starves
        for _ in range(STEPS + 5):
            yield self._batch(second[i % len(second)])
            i += 1

    def __len__(self):
        return len(self._warmed) + STEPS + 5

    def invalidate(self):
        pass


class _Mon:
    def __init__(self):
        self.rows = []

    def report(self, d):
        self.rows.append((time.time(), dict(d)))


def _run_arm(clip_inner, warmed, tail, *, standing_mb, interval,
             hits_first, label, max_entries=WARMED_HEAD_N):
    from nodes.model.trainer_parameters import TrainerParametersNode
    dev = DeviceContext.for_device("xpu")
    model = _XPUFakeModel()
    trainer = SimpleNamespace(unet=model, clip=None)
    params = TrainerParametersNode().build(trainer=trainer)["params"]
    opt = ComposedAdamWOptimizerNode().build(
        params=params, lr=1e-4, device="xpu",
        state_precision="float32")["optimizer"]
    # Top up to the target standing AFTER model+optimizer exist, so the
    # ballast plus the real 0.6 GB params + 1.2 GB fp32 states sum to it
    # rather than overshooting by ~1.8 GB.
    cur = dev.memory_stats()["reserved_mb"]
    ballast_mb = max(0.0, standing_mb - cur)
    ballast = torch.empty(int(ballast_mb * 1024 ** 2 // 4),
                          dtype=torch.float32, device="xpu")
    b0 = dev.memory_stats()
    print(f"[{label}] standing reserved={b0['reserved_mb']:.0f} MB, "
          f"interval={interval}", flush=True)

    events: dict = {"miss_ts": [], "miss_prompts": []}
    inner_encode = CachingTextEncoder.encode_prompt_only

    def counting(self, prompt, batch_size):
        # Timestamped: the warm pass also flows through here (its first
        # prompt), so warm-time and train-time misses are split post-hoc
        # at the first monitor row instead of trusting a bare count.
        events["miss_ts"].append(time.time())
        if len(events["miss_prompts"]) < 5:
            events["miss_prompts"].append(prompt)
        return inner_encode(self, prompt, batch_size)

    CachingTextEncoder.encode_prompt_only = counting
    mon = _Mon()
    rc = BudgetedResourceControlHandle(
        ResourceBudget(vram_budget_mb=standing_mb + 500,
                       vram_reserve_mb=0.0), device="xpu")
    cached = CachingTextEncoder(clip_inner, max_entries=max_entries)
    trainer.clip = cached
    node = ManagedLoRATrainerNode()
    node.context = ExecutionContext()
    outcome = "completed"
    try:
        node.build(
            trainer=trainer, batches=_ManyPrompts(warmed, tail, hits_first),
            optimizer=opt, lr_schedule=ConstantLRSchedule(lr=1e-4),
            loss_weighting=UniformLossWeighting(), steps=STEPS,
            resource_control=rc, monitor=mon, calibration_steps=2,
            empty_cache_every_n_steps=interval)
    except Exception as exc:  # noqa: BLE001 -- an OOM is a data point here
        msg = f"{type(exc).__name__}: {exc}".strip()
        outcome = ("OOM" if "out of memory" in msg.lower() else
                   f"FAILED({msg[:120]})")
        print(f"[{label}] {outcome}\n{traceback.format_exc(limit=3)}",
              flush=True)
    finally:
        CachingTextEncoder.encode_prompt_only = inner_encode
    steps = [r for _, r in mon.rows if r.get("type") != "run_end"]
    peaks = [r.get("vram_peak_reserved_mb", 0) or 0 for r in steps]
    walls = [t for t, _ in mon.rows]
    per_step = ([walls[i + 1] - walls[i] for i in range(len(walls) - 1)]
                if len(walls) > 1 else [])
    peak = max(peaks) if peaks else 0.0
    train_start = walls[0] if walls else float("inf")
    warm_misses = sum(1 for t in events["miss_ts"] if t < train_start)
    train_misses = sum(1 for t in events["miss_ts"] if t >= train_start)
    print(f"[{label}] outcome={outcome} steps={len(steps)} "
          f"misses_warm={warm_misses} misses_train={train_misses} "
          f"max_step_peak={peak:.0f} MB "
          + (f"slowest_step={max(per_step):.1f}s median={sorted(per_step)[len(per_step)//2]:.1f}s"
             if per_step else ""), flush=True)
    print(f"[{label}] first train-miss prompts: "
          f"{events['miss_prompts'][:3]}", flush=True)
    del trainer, cached, model, opt, node, rc, mon, ballast
    gc.collect()
    dev.synchronize()
    torch.xpu.empty_cache()
    return {"label": label, "outcome": outcome, "steps": len(steps),
            "misses": events["miss"], "max_step_peak": peak,
            "slowest_step": max(per_step) if per_step else 0.0}


def run_matrix():
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.text_encoder import SDXLTextEncoderNode
    ctx = ExecutionContext()
    weights = SafetensorsCheckpointNode(ctx).build(
        path="div_4.safetensors")["weights"]
    clip_inner = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    warmed, tail = WARMED_HEAD, TAIL
    print(f"CLIP resident fp16; {len(PROMPTS)} prompts; "
          f"LRU retains the {len(warmed)} zz_top heads", flush=True)

    results = []
    # control: everything warmed (no capacity pressure) -- must stay flat
    results.append(_run_arm(clip_inner, PROMPTS, [], standing_mb=8600,
                            interval=100, hits_first=True, max_entries=512,
                            label="control_all_hit_S8600_N100"))
    for s in (8000, 8600, 9200):
        for n in (1, 100):
            results.append(_run_arm(
                clip_inner, warmed, tail, standing_mb=s, interval=n,
                hits_first=True, label=f"miss_midrun_S{s}_N{n}"))
    results.append(_run_arm(clip_inner, warmed, tail, standing_mb=8600,
                            interval=100, hits_first=False,
                            label="miss_from_step0_S8600_N100"))
    print("\nMATRIX:", flush=True)
    for r in results:
        print(f"  {r['label']}: {r['outcome']} steps={r['steps']} "
              f"misses={r['misses']} max_step_peak={r['max_step_peak']:.0f} "
              f"slowest={r['slowest_step']:.1f}s", flush=True)
