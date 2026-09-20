"""BudgetedLoRATrainerNode: SupervisedLoRATrainerNode's exact training
loop (nodes/train/loop.py's run_supervised_lora_training_loop -- shared,
not duplicated, see that module's own docstring), with two Port-level
differences that together make VRAM safety the default instead of an
opt-in someone has to remember to wire:

- resource_control is required here, not optional. The concrete reason
  this is a second Node class rather than a doc comment on
  SupervisedLoRATrainerNode recommending you wire one: a Port default
  of None is easy to leave unwired on a long run, and this project's
  own docs/known-issues/open.md already has a real (if not yet
  root-caused, and on the legacy core/ pipeline rather than nodes/) user
  report of "device lost" errors and silent hangs after VRAM-pressure
  events on this project's own B580/XPU hardware. Making the budget
  mandatory here means a graph that forgot to wire one fails at Run,
  loudly, before any training happens -- not partway through a long run
  once it's too late to matter. See VRAMBudgetControllerNode
  (nodes/memory/vram_budget_controller.py) for the budget itself,
  including its new `strict` input (nodes/memory/control_handle.py) for
  turning "best-effort, offload what's safe and continue anyway" into
  "raise instead of training on over budget."
- empty_cache_every_n_steps defaults to 50, not 0. Periodically returns
  unused cached memory to the driver without being asked -- real, small
  step-time cost (a device sync every 50 steps), traded for keeping this
  run's own peak reserved footprint further from whatever ceiling
  resource_control is enforcing, rather than only reacting once
  something's already close to it. Still a plain int Port, same as
  SupervisedLoRATrainerNode's identically-named one -- 0 still disables
  it, for anyone who measures and decides they don't want it.

Everything else (gating, profiling, diffusion_process, loss_weighting,
monitor, on_step) is identical to SupervisedLoRATrainerNode -- this is a
drop-in replacement for it wherever a VRAM budget should be mandatory,
not a narrower, separate thing.

Route-agnostic on purpose: model/text_encoder here are plain
TrainableModel/TextEncoder ports, exactly like SupervisedLoRATrainerNode's
own -- wire them directly from the existing main route
(ComfyUNetLoRANode/SDXLTextEncoderNode), or from the newer Resources
Controller route via TrainerResourcesUnpackNode
(nodes/model/trainer_unpack.py) unpacking a LoRATrainingConfigNode's
`trainer` output. This class has no idea which, and doesn't need to --
see that module's own docstring for why the bundled `trainer` type
can't be wired into a plain TrainableModel/TextEncoder port directly
(server/graph_executor.py's real issubclass() check on every edge).

Why this is a subclass of TrainerNode directly, not of
SupervisedLoRATrainerNode: every concrete Node in this project
subclasses its domain-family ABC directly and shares real logic via a
plain, non-Node object instead (see e.g. nodes/optimizer/composed_fused_adamw.py
next to nodes/optimizer/composed_adamw.py, sharing AdamWAlgorithm/
ComposedOptimizerHandle rather than one inheriting the other) --
nodes/train/loop.py's extraction follows that same, consistently-applied
pattern rather than being the one place in the codebase where a
concrete Node inherits from another concrete Node.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from ..memory.control_handle import ResourceControlHandle
from ..model.handle import TrainableModel
from .loop import run_supervised_lora_training_loop
from .node import TrainerNode
from .supervised import SupervisedLoRATrainerNode

_DEFAULT_EMPTY_CACHE_EVERY_N_STEPS = 50


class BudgetedLoRATrainerNode(TrainerNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        **SupervisedLoRATrainerNode.INPUTS,
        "resource_control": Port(
            name="resource_control", type=ResourceControlHandle, required=True,
            doc="Wire a VRAM Budget Controller node's own output here -- required on this "
                "node, unlike SupervisedLoRATrainerNode's identically-named optional port "
                "(see this module's own docstring for why). model/optimizer are registered "
                "non-offloadable (needed every step, nothing here can safely offload and "
                "reload either mid-run yet); text_encoder is offloadable exactly when it's "
                "wired from a caching text encoder. Set VRAMBudgetControllerNode's own "
                "`strict` input True to raise instead of continuing once nothing left is "
                "safe to offload and usage is still over budget.",
        ),
        "empty_cache_every_n_steps": Port(
            name="empty_cache_every_n_steps", type=int, required=False,
            default=_DEFAULT_EMPTY_CACHE_EVERY_N_STEPS,
            doc="Same mechanism as SupervisedLoRATrainerNode's identically-named port (0 "
                "disables it, gc.collect() + DeviceContext.empty_cache() every N steps "
                "otherwise) -- defaults to "
                f"{_DEFAULT_EMPTY_CACHE_EVERY_N_STEPS} here instead of 0, since this node's "
                "whole reason to exist is staying clear of a VRAM ceiling rather than only "
                "reacting to it once already close.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)
        result = run_supervised_lora_training_loop(
            self.context,
            model=inputs["model"],
            batches=inputs["batches"],
            optimizer=inputs["optimizer"],
            text_encoder=inputs["text_encoder"],
            lr_schedule=inputs["lr_schedule"],
            steps=inputs["steps"],
            diffusion_process=inputs.get("diffusion_process"),
            gate_enabled=inputs.get("gate_enabled", self.INPUTS["gate_enabled"].default),
            gate_train_low=inputs.get("gate_train_low", self.INPUTS["gate_train_low"].default),
            gate_train_high=inputs.get("gate_train_high", self.INPUTS["gate_train_high"].default),
            gate_width=inputs.get("gate_width", self.INPUTS["gate_width"].default),
            profile=inputs.get("profile", self.INPUTS["profile"].default),
            empty_cache_every_n_steps=inputs.get(
                "empty_cache_every_n_steps", self.INPUTS["empty_cache_every_n_steps"].default),
            profile_memory_per_phase=inputs.get(
                "profile_memory_per_phase", self.INPUTS["profile_memory_per_phase"].default),
            resource_control=inputs.get("resource_control"),
            loss_weighting=inputs.get("loss_weighting"),
            monitor=inputs.get("monitor"),
            on_step=inputs.get("on_step"),
            log_prefix=type(self).__name__,
        )
        self.validate_outputs(result)
        return result
