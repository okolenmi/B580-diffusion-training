"""Two AdamW options -- deliberately kept separate, not one node with a
"cpu_offload" flag, because they have genuinely different intended use
cases (full fine-tune vs. LoRA), not just a performance knob.

AdamWOptimizerNode wraps core.optimizers.CPUAdamW. Unlike the other
adapters in this package, this one is not a pure pass-through: CPUAdamW
does not implement decay_states/reset_states at all (confirmed by reading
core/optimizers.py directly -- a real, latent bug, since core/trainer.py
calls optimizer.decay_states(...) unconditionally in cyclic-tuning mode,
so combining optimizer="adamw" with cyclic tuning would raise
AttributeError the first time anyone actually tried it). Because
OptimizerHandle declares decay_states/reset_states as required abstract
methods, this adapter is *forced* to implement them for
AdamWOptimizerHandle to be instantiable at all -- see
docs/nodes_package_design.md's "worked example" section for the reasoning.
The implementation is new code, written and verified fresh here, not
copied from anywhere -- core/optimizers.py's CPUAdamW itself is never
touched.

One genuine correctness point, not just a style choice: CPUAdamW.step()
does self.m[i].mul_(self.b1)... unconditionally, with no "is this None"
guard (unlike the GPU optimizers' lazily-populated state lists, which do
guard). So a reset here must set m[i]/v[i] back to *zero tensors*, not
None -- setting them to None would make the very next step() call crash
with AttributeError('NoneType' object has no attribute 'mul_'). Verified
by reading CPUAdamW.step()'s body directly before writing this.

SimpleAdamWOptimizerNode wraps torch.optim.AdamW directly instead. See
SimpleAdamWOptimizerHandle's own docstring below for why this is the one
to actually use for LoRA -- CPUAdamW's CPU residency (right for a full
2.6B-parameter fine-tune, where Adam state genuinely can't fit on the
device) is pure, measured overhead for LoRA's tiny parameter count.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from ..memory.handle import sum_tensor_bytes
from .handle import OptimizerHandle
from .node import OptimizerNode


class AdamWOptimizerHandle(OptimizerHandle):

    def __init__(self, legacy_optimizer):
        self._legacy = legacy_optimizer

    @property
    def lr(self) -> float:
        return self._legacy.lr

    def update_lr(self, new_lr: float) -> None:
        # CPUAdamW has no param_lr list at all (confirmed by reading its
        # __init__ -- only a single scalar self.lr), so this is simpler
        # than the param_lr-list adapters.
        self._legacy.lr = new_lr

    def step(self, n_steps: int = 1) -> None:
        self._legacy.step(n_steps=n_steps)

    def zero_grad(self) -> None:
        self._legacy.zero_grad()

    def offload_states_to_cpu(self) -> None:
        self._legacy.offload_states_to_cpu()  # already a no-op on the legacy class -- states are CPU-resident always

    def reload_states_to_device(self, device: str | None = None) -> None:
        self._legacy.reload_states_to_device(device)  # also already a no-op

    def decay_states(self, factor: float) -> None:
        if factor <= 0:
            self.reset_states()
            return
        for i in range(len(self._legacy.m)):
            if self._legacy.m[i] is not None:
                self._legacy.m[i].mul_(factor)
            if self._legacy.v[i] is not None:
                self._legacy.v[i].mul_(factor)
        print(f"    [AdamW] Optimizer states decayed by factor {factor:.2f}.")

    def reset_states(self) -> None:
        # Must zero in place, not set to None -- see this module's
        # docstring for why (CPUAdamW.step() has no None-guard).
        for i in range(len(self._legacy.m)):
            self._legacy.m[i].zero_()
            self._legacy.v[i].zero_()
        print("    [AdamW] Optimizer states reset.")

    def free_states(self) -> None:
        self._legacy.free_states()

    def footprint_bytes(self) -> int:
        # getattr(..., ()), not self._legacy.m/.v directly: CPUAdamW.free_states()
        # does `del self.m, self.v` (confirmed by reading it), not clear-in-place --
        # so after release() these attributes don't exist at all, and the
        # correct "best-effort current usage" answer at that point is 0, not
        # an AttributeError.
        return sum_tensor_bytes(getattr(self._legacy, "m", ()), getattr(self._legacy, "v", ()))
    """CPU-resident AdamW -- see core.optimizers.CPUAdamW's own module
    comment (FP32 states on CPU, saved to disc as BF16). Right for full
    fine-tuning; a measured trap for LoRA -- see SimpleAdamWOptimizerNode
    below instead."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999)),
        "eps": Port(name="eps", type=float, required=False, default=1e-8),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=1e-2),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        from core.optimizers import CPUAdamW
        legacy = CPUAdamW(
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            betas=inputs.get("betas", self.INPUTS["betas"].default),
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
        )
        result = {"optimizer": AdamWOptimizerHandle(legacy)}
        self.validate_outputs(result)
        return result


class SimpleAdamWOptimizerHandle(OptimizerHandle):
    """Wraps torch.optim.AdamW directly (foreach=True) -- PyTorch's own
    batched per-dtype/per-device kernels do the update, not a Python loop
    with a CPU<->device round trip per parameter tensor like
    AdamWOptimizerHandle/CPUAdamW above does. That round trip is why this
    class exists: CPUAdamW was built for full fine-tuning, where 2.6B
    parameters' worth of Adam state genuinely can't fit on the device
    alongside the model -- CPU residency is the right call there. LoRA's
    total trainable parameter count is tiny and easily device-resident, so
    that same design is pure overhead for it: one synchronous .to("cpu")
    and one .to(device) call *per parameter tensor*, every step, for
    however many LoRA layers exist (order of 100+ small tensors is
    normal), with no compute to hide the transfer latency behind. Measured
    directly, not theorized: a user's real profiler output showed a
    1186ms optimizer step next to a 273+311ms forward+backward, using
    CPUAdamW for LoRA training."""

    def __init__(self, legacy_optimizer, device):
        self._legacy = legacy_optimizer
        self._device = device

    @property
    def lr(self) -> float:
        return self._legacy.param_groups[0]["lr"]

    def update_lr(self, new_lr: float) -> None:
        for group in self._legacy.param_groups:
            group["lr"] = new_lr

    def step(self, n_steps: int = 1) -> None:
        # n_steps > 1 (gradient accumulation) isn't meaningful here yet --
        # SupervisedLoRATrainerNode v1 never calls this with n_steps != 1
        # (no grad accum -- see that node's own documented scope list).
        self._legacy.step()

    def zero_grad(self) -> None:
        self._legacy.zero_grad(set_to_none=True)

    def _each_state_tensor(self):
        import torch
        for state in self._legacy.state.values():
            for key, val in state.items():
                if key != "step" and torch.is_tensor(val):
                    yield state, key, val

    def offload_states_to_cpu(self) -> None:
        for state, key, val in list(self._each_state_tensor()):
            state[key] = val.to("cpu")

    def reload_states_to_device(self, device: str | None = None) -> None:
        target = device or self._device
        for state, key, val in list(self._each_state_tensor()):
            state[key] = val.to(target)

    def decay_states(self, factor: float) -> None:
        if factor <= 0:
            self.reset_states()
            return
        for _state, _key, val in self._each_state_tensor():
            val.mul_(factor)

    def reset_states(self) -> None:
        for _state, _key, val in self._each_state_tensor():
            val.zero_()

    def free_states(self) -> None:
        self._legacy.state.clear()

    def footprint_bytes(self) -> int:
        return sum(val.numel() * val.element_size()
                   for _state, _key, val in self._each_state_tensor())


class SimpleAdamWOptimizerNode(OptimizerNode):
    """Plain, fully device-resident torch.optim.AdamW -- the right default
    for LoRA. See SimpleAdamWOptimizerHandle above for why
    AdamWOptimizerNode (this file's other optimizer) is a trap for this
    specific case despite the name; that one is for full fine-tuning."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999)),
        "eps": Port(name="eps", type=float, required=False, default=1e-8),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=1e-2),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        import torch

        params = inputs["params"]
        legacy = torch.optim.AdamW(
            params,
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            betas=inputs.get("betas", self.INPUTS["betas"].default),
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
            foreach=True,
        )
        device = params[0].device if len(params) > 0 else None
        result = {"optimizer": SimpleAdamWOptimizerHandle(legacy, device)}
        self.validate_outputs(result)
        return result
