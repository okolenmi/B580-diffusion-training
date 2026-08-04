"""SupervisedLoRATrainerNode: v1 training loop for the dataset -> LoRA case.

Scope, explicit -- deliberately deferred, not forgotten (see
docs/nodes_package_design.md's TrainerNode section for the running list):
single conditioning pass per batch (no CFG cond/uncond dual pass), no
gradient accumulation, no cyclic/teacher-rollout caching, no DAgger, no
adversarial pre-conditioning, no timestep gating, no resume/checkpoint
cadence (use `on_step` for that). Assumes the dataset's stored `target` is
already in the student's own prediction parameterization -- no
teacher/student eps<->vpred conversion at train time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional

from ..components.device import DeviceContext
from ..components.diffusion import (DiffusionProcess, DiscreteLinearNoiseSchedule,
                                     EpsParameterization, KarrasInputScaler)
from ..core import Port
from ..dataset.handle import TrainingBatchSource
from ..model.handle import TrainableModel
from ..model.text_encoder import TextEncoder
from ..monitor.handle import MonitorHandle
from ..optimizer.handle import FusedOptimizerHandle, OptimizerHandle
from .loss import LossWeighting, UniformLossWeighting
from .node import TrainerNode
from .schedule import LRSchedule


@dataclass
class _StepContext:
    """Everything about a run that stays fixed step to step -- bundled so
    _run_step takes (ctx, batch, step) instead of an ever-growing
    parameter list every time the loop needs one more thing (this is the
    second time; `monitor` is what pushed it over into "bundle this")."""
    model: TrainableModel
    optimizer: OptimizerHandle
    text_encoder: TextEncoder
    lr_schedule: LRSchedule
    loss_weighting: LossWeighting
    diffusion_process: DiffusionProcess
    device_ctx: DeviceContext
    is_fused: bool
    device: object
    total_steps: int
    on_step: Optional[Callable] = None
    monitor: Optional[MonitorHandle] = None
    profile: bool = False


class SupervisedLoRATrainerNode(TrainerNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        **TrainerNode.COMMON_INPUTS,
        "diffusion_process": Port(
            name="diffusion_process", type=DiffusionProcess, required=False, default=None,
            doc="None = today's actual behavior: DiscreteLinearNoiseSchedule's default "
                "linear beta schedule, epsilon prediction, ComfyUI's calculate_input "
                "scaling. Wire a different DiffusionProcess (e.g. built around "
                "RescaledZeroTerminalSNRSchedule + VPredParameterization) to change any "
                "of the three -- see nodes/components/diffusion.py.",
        ),
        "profile": Port(
            name="profile", type=bool, required=False, default=False,
            doc="Per-phase step timing (data wait / text encode / forward / backward / "
                "optimizer step) printed every step, and included in monitor.report() if a "
                "monitor is wired -- the breakdown this project didn't have a way to see. "
                "Off by default: correct phase timing needs a device synchronize() between "
                "phases (DeviceContext.synchronize()), which blocks the async pipeline and "
                "makes steps measurably slower than a normal run while this is on. Use it "
                "for a short diagnostic run, not for real training. Also reports "
                "vram_allocated_mb/vram_reserved_mb each step when profiling -- allocated "
                "growing over many steps is a real, live reference leak; reserved growing "
                "while allocated stays flat is just the caching allocator's own bookkeeping, "
                "not a leak (see nodes/components/device.py's DeviceContext.memory_stats).",
        ),
        "empty_cache_every_n_steps": Port(
            name="empty_cache_every_n_steps", type=int, required=False, default=0,
            doc="0 disables this. When >0, calls gc.collect() + DeviceContext.empty_cache() "
                "every N steps. This only returns *unused* cached memory to the driver -- it "
                "cannot free memory still genuinely referenced by something, so it won't help "
                "a real reference leak (check profile=True's vram_allocated_mb for that), only "
                "caching-allocator fragmentation/bookkeeping. Costs a device sync each time "
                "it runs (same as any other explicit synchronize), so a very small N will "
                "cost real step time -- start high (e.g. 50) and go lower only if needed.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)

        model: TrainableModel = inputs["model"]
        batches: TrainingBatchSource = inputs["batches"]
        steps: int = inputs["steps"]
        empty_cache_every_n_steps: int = inputs.get(
            "empty_cache_every_n_steps", self.INPUTS["empty_cache_every_n_steps"].default)

        model.train()
        device = next(iter(model.trainable_parameters())).device
        optimizer = inputs["optimizer"]
        ctx = _StepContext(
            model=model,
            optimizer=optimizer,
            text_encoder=inputs["text_encoder"],
            lr_schedule=inputs["lr_schedule"],
            loss_weighting=inputs.get("loss_weighting") or UniformLossWeighting(),
            diffusion_process=inputs.get("diffusion_process") or DiffusionProcess(
                DiscreteLinearNoiseSchedule(), EpsParameterization(), KarrasInputScaler()),
            device_ctx=DeviceContext.for_device(device),
            is_fused=isinstance(optimizer, FusedOptimizerHandle),
            device=device,
            total_steps=steps,
            on_step=inputs.get("on_step"),
            monitor=inputs.get("monitor"),
            profile=inputs.get("profile", self.INPUTS["profile"].default),
        )

        step = 0
        batch_ready_at = time.perf_counter()
        while step < steps:
            for batch in batches:
                if step >= steps:
                    break
                if self.context.should_cancel():
                    # Cooperative stop, between steps only -- never mid
                    # backward/optimizer-step. Not a failure: the model
                    # trained so far is a normal, valid output, same as a
                    # run that finished all its steps, just fewer of them.
                    result = {"model": model}
                    self.validate_outputs(result)
                    return result
                wait_ms = (time.perf_counter() - batch_ready_at) * 1000
                self._run_step(ctx, batch, step, wait_ms)
                step += 1
                if empty_cache_every_n_steps > 0 and step % empty_cache_every_n_steps == 0:
                    import gc
                    gc.collect()
                    ctx.device_ctx.empty_cache()
                batch_ready_at = time.perf_counter()

        result = {"model": model}
        self.validate_outputs(result)
        return result

    @staticmethod
    def _run_step(ctx: _StepContext, batch: dict, step: int, wait_ms: float = 0.0) -> None:
        import torch

        timing = None
        t0 = None
        if ctx.profile:
            ctx.device_ctx.synchronize()
            timing = {"data_wait_ms": wait_ms}
            t0 = time.perf_counter()

        x_t = batch["x_t"].to(ctx.device)
        target = batch["target"].to(ctx.device)
        t = batch["t"].to(device=ctx.device, dtype=torch.long).view(-1)
        _, sigma = ctx.diffusion_process.schedule.alpha_sigma(t)
        xc = ctx.diffusion_process.input_transform.scale_input(x_t, sigma)

        batch_h, batch_w = x_t.shape[2] * 8, x_t.shape[3] * 8
        ctx_emb, y = ctx.text_encoder.encode(batch["prompt"], batch_size=x_t.shape[0],
                                              height=batch_h, width=batch_w)
        ctx_emb = ctx_emb.to(device=ctx.device, dtype=torch.bfloat16)
        y = y.to(device=ctx.device, dtype=torch.bfloat16)

        t1 = None
        if ctx.profile:
            ctx.device_ctx.synchronize()
            t1 = time.perf_counter()
            timing["encode_ms"] = (t1 - t0) * 1000

        lr = ctx.lr_schedule.value(step)
        ctx.optimizer.update_lr(lr)
        if ctx.is_fused:
            ctx.optimizer.begin_step(sub_steps=1)
        else:
            ctx.optimizer.zero_grad()

        pred = ctx.model.forward(xc, t, ctx_emb, y)

        t2 = None
        if ctx.profile:
            ctx.device_ctx.synchronize()
            t2 = time.perf_counter()
            timing["forward_ms"] = (t2 - t1) * 1000

        per_sample = (pred.float() - target.float()).pow(2)
        per_sample = per_sample.view(per_sample.shape[0], -1).mean(dim=1)
        weight = ctx.loss_weighting.weight(float(sigma.float().mean().item()))
        loss = per_sample.mean() * weight
        loss.backward()

        t3 = None
        if ctx.profile:
            ctx.device_ctx.synchronize()
            t3 = time.perf_counter()
            timing["backward_ms"] = (t3 - t2) * 1000

        if not ctx.is_fused:
            ctx.optimizer.step(n_steps=1)

        if ctx.profile:
            ctx.device_ctx.synchronize()
            t4 = time.perf_counter()
            timing["optim_ms"] = (t4 - t3) * 1000
            timing["step_total_ms"] = (t4 - t0) * 1000 + wait_ms
            mem = ctx.device_ctx.memory_stats()
            if mem is not None:
                timing["vram_allocated_mb"] = mem["allocated_mb"]
                timing["vram_reserved_mb"] = mem["reserved_mb"]

        loss_value = float(loss.item())
        if ctx.on_step is not None:
            ctx.on_step(step, loss_value)
        if ctx.monitor is not None or ctx.profile:
            report = {
                "step": step, "total_steps": ctx.total_steps,
                "loss": loss_value, "lr": lr, "t": time.time(),
            }
            if timing is not None:
                report.update(timing)
            if ctx.monitor is not None:
                ctx.monitor.report(report)
        if ctx.profile:
            vram_part = ""
            if "vram_allocated_mb" in timing:
                vram_part = (f" vram_allocated={timing['vram_allocated_mb']:.0f}MB "
                             f"vram_reserved={timing['vram_reserved_mb']:.0f}MB")
            print(f"  [step {step}] wait={timing['data_wait_ms']:.0f}ms "
                  f"encode={timing['encode_ms']:.0f}ms forward={timing['forward_ms']:.0f}ms "
                  f"backward={timing['backward_ms']:.0f}ms optim={timing['optim_ms']:.0f}ms "
                  f"total={timing['step_total_ms']:.0f}ms" + vram_part)
