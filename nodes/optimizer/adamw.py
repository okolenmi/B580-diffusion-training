"""AdamWOptimizerNode: wraps core.optimizers.CPUAdamW.

Unlike the other four adapters in this package, this one is not a pure
pass-through: CPUAdamW does not implement decay_states/reset_states at all
(confirmed by reading core/optimizers.py directly -- a real, latent bug,
since core/trainer.py calls optimizer.decay_states(...) unconditionally in
cyclic-tuning mode, so combining optimizer="adamw" with cyclic tuning would
raise AttributeError the first time anyone actually tried it). Because
OptimizerHandle declares decay_states/reset_states as required abstract
methods, this adapter is *forced* to implement them for
AdamWOptimizerHandle to be instantiable at all -- see
docs/nodes_package_design.md's "worked example" section for the reasoning.

The implementation below is new code, written and verified fresh here, not
copied from anywhere -- core/optimizers.py's CPUAdamW itself is never
touched and remains exactly as it is for any other current caller.

One genuine correctness point, not just a style choice: CPUAdamW.step()
does self.m[i].mul_(self.b1)... unconditionally, with no "is this None"
guard (unlike the GPU optimizers' lazily-populated state lists, which do
guard). So a reset here must set m[i]/v[i] back to *zero tensors*, not
None -- setting them to None would make the very next step() call crash
with AttributeError('NoneType' object has no attribute 'mul_'). Verified
by reading CPUAdamW.step()'s body directly before writing this.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
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


class AdamWOptimizerNode(OptimizerNode):
    """CPU-resident AdamW -- see core.optimizers.CPUAdamW's own module
    comment (FP32 states on CPU, saved to disc as BF16)."""

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
