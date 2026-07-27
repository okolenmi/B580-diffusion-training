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
    is_fused: bool
    device: object
    total_steps: int
    on_step: Optional[Callable] = None
    monitor: Optional[MonitorHandle] = None


class SupervisedLoRATrainerNode(TrainerNode):

    INPUTS: ClassVar[dict[str, Port]] = {**TrainerNode.COMMON_INPUTS}

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)

        model: TrainableModel = inputs["model"]
        batches: TrainingBatchSource = inputs["batches"]
        steps: int = inputs["steps"]

        model.train()
        device = next(iter(model.trainable_parameters())).device
        optimizer = inputs["optimizer"]
        ctx = _StepContext(
            model=model,
            optimizer=optimizer,
            text_encoder=inputs["text_encoder"],
            lr_schedule=inputs["lr_schedule"],
            loss_weighting=inputs.get("loss_weighting") or UniformLossWeighting(),
            is_fused=isinstance(optimizer, FusedOptimizerHandle),
            device=device,
            total_steps=steps,
            on_step=inputs.get("on_step"),
            monitor=inputs.get("monitor"),
        )

        step = 0
        while step < steps:
            for batch in batches:
                if step >= steps:
                    break
                self._run_step(ctx, batch, step)
                step += 1

        result = {"model": model}
        self.validate_outputs(result)
        return result

    @staticmethod
    def _run_step(ctx: _StepContext, batch: dict, step: int) -> None:
        import torch
        from core.model_io import comfy_input_transform
        from core.noise_schedule import get_alpha_sigma

        x_t = batch["x_t"].to(ctx.device)
        target = batch["target"].to(ctx.device)
        t = batch["t"].to(device=ctx.device, dtype=torch.long).view(-1)
        _, sigma = get_alpha_sigma(t)
        xc = comfy_input_transform(x_t, sigma)

        batch_h, batch_w = x_t.shape[2] * 8, x_t.shape[3] * 8
        ctx_emb, y = ctx.text_encoder.encode(batch["prompt"], batch_size=x_t.shape[0],
                                              height=batch_h, width=batch_w)
        ctx_emb = ctx_emb.to(device=ctx.device, dtype=torch.bfloat16)
        y = y.to(device=ctx.device, dtype=torch.bfloat16)

        lr = ctx.lr_schedule.value(step)
        ctx.optimizer.update_lr(lr)
        if ctx.is_fused:
            ctx.optimizer.begin_step(sub_steps=1)
        else:
            ctx.optimizer.zero_grad()

        pred = ctx.model.forward(xc, t, ctx_emb, y)
        per_sample = (pred.float() - target.float()).pow(2)
        per_sample = per_sample.view(per_sample.shape[0], -1).mean(dim=1)
        weight = ctx.loss_weighting.weight(float(sigma.float().mean().item()))
        loss = per_sample.mean() * weight
        loss.backward()

        if not ctx.is_fused:
            ctx.optimizer.step(n_steps=1)

        loss_value = float(loss.item())
        if ctx.on_step is not None:
            ctx.on_step(step, loss_value)
        if ctx.monitor is not None:
            ctx.monitor.report({
                "step": step, "total_steps": ctx.total_steps,
                "loss": loss_value, "lr": lr, "t": time.time(),
            })
