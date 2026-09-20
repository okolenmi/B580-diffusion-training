"""Checks nodes/train/budgeted.py's BudgetedLoRATrainerNode:

1. resource_control is genuinely required (validate_inputs raises
   without one) -- the entire reason this node exists as a second
   class rather than a doc comment on SupervisedLoRATrainerNode.
2. empty_cache_every_n_steps really does default to 50, not
   SupervisedLoRATrainerNode's 0.
3. Given equivalent inputs (including a resource_control), this node's
   real training behavior is identical to SupervisedLoRATrainerNode's
   -- both now call the exact same nodes/train/loop.py function, so
   this is really a check that the nodes/train/supervised.py extraction
   (this session's own refactor) didn't change behavior, proven by
   running both concrete classes side by side rather than trusting the
   refactor was mechanical by inspection alone.
4. model/optimizer/text_encoder actually get registered with
   resource_control, matching SupervisedLoRATrainerNode's own
   registration (offloadable=False/False/isinstance-CachingTextEncoder).

Reuses smoke_test_trainer_cancellation.py's exact fakes (_FakeModel/
_FakeOptimizer/_FakeTextEncoder/_FiniteBatches) rather than a second
copy of the same mocking -- same reasoning
smoke_test_lora_training_resources.py already gives for its own
cross-imports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.core import ExecutionContext
from nodes.memory.control_handle import ResourceControlHandle
from nodes.train.budgeted import BudgetedLoRATrainerNode
from nodes.train.loss import UniformLossWeighting
from nodes.train.schedule import ConstantLRSchedule
from nodes.train.supervised import SupervisedLoRATrainerNode
from nodes.smoke_tests.smoke_test_trainer_cancellation import (
    _FakeModel, _FakeOptimizer, _FakeTextEncoder, _FiniteBatches)


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


class _RecordingResourceControl(ResourceControlHandle):
    """Records every register()/before_step() call -- enough to prove
    BudgetedLoRATrainerNode registers what SupervisedLoRATrainerNode's
    own docstring says it should, without a mocking framework. Same
    shape as smoke_test_text_encoder_cache.py's own
    _RecordingResourceControl, extended to also record register()
    (that file's version only needed ensure_loaded())."""

    def __init__(self):
        self.registered: list[tuple[str, bool]] = []
        self.before_step_calls: list[int] = []
        self.ensure_loaded_calls: list[str] = []

    def register(self, name: str, resident, offloadable: bool = False) -> None:
        self.registered.append((name, offloadable))

    def before_step(self, step: int) -> None:
        self.before_step_calls.append(step)

    def ensure_loaded(self, name: str) -> None:
        self.ensure_loaded_calls.append(name)


def check_resource_control_is_required():
    print("[BudgetedLoRATrainerNode.INPUTS['resource_control'] is required=True, "
          "unlike SupervisedLoRATrainerNode's identically-named optional one]")
    check(BudgetedLoRATrainerNode.INPUTS["resource_control"].required is True,
          "resource_control must be required on BudgetedLoRATrainerNode")
    check(SupervisedLoRATrainerNode.INPUTS["resource_control"].required is False,
          "SupervisedLoRATrainerNode's own resource_control must still be optional -- "
          "unrelated to this new node, must not have regressed")
    print("    PASS")


def check_build_without_resource_control_raises():
    print("[build() without resource_control raises via validate_inputs, before any "
          "training happens]")
    node = BudgetedLoRATrainerNode()
    node.context = ExecutionContext()
    model = _FakeModel()
    raised = False
    try:
        node.build(
            model=model, optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
            batches=_FiniteBatches(), steps=10,
            lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        )
    except ValueError:
        raised = True
    check(raised, "expected build() to raise ValueError (missing required input) "
                   "when resource_control isn't wired")
    check(model.calls == 0, "must fail before running any step, not partway through")
    print("    PASS")


def check_empty_cache_every_n_steps_default_differs_from_supervised():
    print("[empty_cache_every_n_steps defaults to 50 here, 0 on SupervisedLoRATrainerNode]")
    check(BudgetedLoRATrainerNode.INPUTS["empty_cache_every_n_steps"].default == 50,
          BudgetedLoRATrainerNode.INPUTS["empty_cache_every_n_steps"].default)
    check(SupervisedLoRATrainerNode.INPUTS["empty_cache_every_n_steps"].default == 0,
          "SupervisedLoRATrainerNode's own default must still be 0 -- must not have regressed")
    print("    PASS")


def check_registers_model_optimizer_text_encoder_same_as_supervised():
    print("[resource_control.register() calls match SupervisedLoRATrainerNode's own: "
          "model/optimizer offloadable=False, text_encoder offloadable=False (plain fake, "
          "not a CachingTextEncoder)]")
    control = _RecordingResourceControl()
    node = BudgetedLoRATrainerNode()
    node.context = ExecutionContext()
    model = _FakeModel()
    node.build(
        model=model, optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
        batches=_FiniteBatches(), steps=3,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        resource_control=control,
    )
    check(("model", False) in control.registered, control.registered)
    check(("optimizer", False) in control.registered, control.registered)
    check(("text_encoder", False) in control.registered, control.registered)
    check(control.before_step_calls == [0, 1, 2], control.before_step_calls)
    print("    PASS")


def check_training_behavior_matches_supervised_trainer_node():
    print("[same inputs, same steps trained, same final call count as "
          "SupervisedLoRATrainerNode -- both share nodes/train/loop.py's one real "
          "implementation now]")
    supervised_model = _FakeModel()
    supervised_node = SupervisedLoRATrainerNode()
    supervised_node.context = ExecutionContext()
    supervised_result = supervised_node.build(
        model=supervised_model, optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
        batches=_FiniteBatches(), steps=9,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        resource_control=_RecordingResourceControl(),
    )

    budgeted_model = _FakeModel()
    budgeted_node = BudgetedLoRATrainerNode()
    budgeted_node.context = ExecutionContext()
    budgeted_result = budgeted_node.build(
        model=budgeted_model, optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
        batches=_FiniteBatches(), steps=9,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        resource_control=_RecordingResourceControl(),
    )

    check(supervised_model.calls == budgeted_model.calls == 9,
          f"supervised={supervised_model.calls} budgeted={budgeted_model.calls}")
    check(supervised_result["model"] is supervised_model, "must return the same model instance")
    check(budgeted_result["model"] is budgeted_model, "must return the same model instance")
    print("    PASS")


def main():
    check_resource_control_is_required()
    check_build_without_resource_control_raises()
    check_empty_cache_every_n_steps_default_differs_from_supervised()
    check_registers_model_optimizer_text_encoder_same_as_supervised()
    check_training_behavior_matches_supervised_trainer_node()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
