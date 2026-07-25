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

from typing import ClassVar

from ..core import Port
from ..dataset.handle import TrainingBatchSource
from ..model.handle import TrainableModel
from ..model.text_encoder import TextEncoder
from ..optimizer.handle import FusedOptimizerHandle, OptimizerHandle
from .loss import LossWeighting, UniformLossWeighting
from .node import TrainerNode
from .schedule import LRSchedule


class SupervisedLoRATrainerNode(TrainerNode):

    INPUTS: ClassVar[dict[str, Port]] = {**TrainerNode.COMMON_INPUTS}

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)

        model: TrainableModel = inputs["model"]
        batches: TrainingBatchSource = inputs["batches"]
        optimizer: OptimizerHandle = inputs["optimizer"]
        text_encoder: TextEncoder = inputs["text_encoder"]
        lr_schedule: LRSchedule = inputs["lr_schedule"]
        loss_weighting: LossWeighting = inputs.get("loss_weighting") or UniformLossWeighting()
        steps: int = inputs["steps"]
        on_step = inputs.get("on_step")

        model.train()
        device = next(iter(model.trainable_parameters())).device
        is_fused = isinstance(optimizer, FusedOptimizerHandle)

        step = 0
        while step < steps:
            for batch in batches:
                if step >= steps:
                    break
                self._run_step(batch, step, model, optimizer, text_encoder,
                                lr_schedule, loss_weighting, is_fused, device, on_step)
                step += 1

        result = {"model": model}
        self.validate_outputs(result)
        return result

    @staticmethod
    def _run_step(batch, step, model, optimizer, text_encoder, lr_schedule,
                   loss_weighting, is_fused, device, on_step):
        import torch
        from core.model_io import comfy_input_transform
        from core.noise_schedule import get_alpha_sigma

        x_t = batch["x_t"].to(device)
        target = batch["target"].to(device)
        t = batch["t"].to(device=device, dtype=torch.long).view(-1)
        _, sigma = get_alpha_sigma(t)
        xc = comfy_input_transform(x_t, sigma)

        batch_h, batch_w = x_t.shape[2] * 8, x_t.shape[3] * 8
        ctx, y = text_encoder.encode(batch["prompt"], batch_size=x_t.shape[0],
                                      height=batch_h, width=batch_w)
        ctx = ctx.to(device=device, dtype=torch.bfloat16)
        y = y.to(device=device, dtype=torch.bfloat16)

        optimizer.update_lr(lr_schedule.value(step))
        if is_fused:
            optimizer.begin_step(sub_steps=1)
        else:
            optimizer.zero_grad()

        pred = model.forward(xc, t, ctx, y)
        per_sample = (pred.float() - target.float()).pow(2)
        per_sample = per_sample.view(per_sample.shape[0], -1).mean(dim=1)
        weight = loss_weighting.weight(float(sigma.float().mean().item()))
        loss = per_sample.mean() * weight
        loss.backward()

        if not is_fused:
            optimizer.step(n_steps=1)

        if on_step is not None:
            on_step(step, float(loss.item()))
