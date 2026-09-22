"""Checks nodes/train/managed.py's ManagedLoRATrainerNode -- specifically
the one thing that's actually new and risky here: that optimizer/
text_encoder residency really is bracketed (ensure_loaded before use,
release right after) in the right order relative to encode/backward/
step, for both a plain and a fused optimizer, and that model is never
released at all. Everything else this node does (diffusion math,
gating, monitoring) is either a straight, low-risk read of the main
route's own equivalent, or already covered by this node's own
docstrings' worked reasoning -- this file deliberately doesn't
re-verify all of that, only the choreography that's genuinely new.

One more thing this file checks, added after the rest: every check
above uses a fake optimizer that never actually looks at `params` --
real coverage of the residency call order, but zero coverage of
whether params (nodes/model/trainer_parameters.py's TrainerParametersNode,
the piece that actually gets a real OptimizerNode its required `params`
input on this route) really is the same object model.forward()/
backward() update, all the way through a real ComposedAdamWOptimizerHandle
and a real optimizer.step(). check_real_optimizer_via_trainer_parameters_node_
actually_updates_the_trained_parameter below closes that gap directly,
end to end, rather than trusting that identity holds by inspection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from types import SimpleNamespace

import torch

from nodes.core import ExecutionContext
from nodes.memory.control_handle import BudgetedResourceControlHandle, ResourceControlHandle
from nodes.model.handle import TrainableModel
from nodes.model.trainer_parameters import TrainerParametersNode
from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
from nodes.optimizer.handle import FusedOptimizerHandle, OptimizerHandle
from nodes.resource_budget import ResourceBudget
from nodes.train.loss import UniformLossWeighting
from nodes.train.managed import ManagedLoRATrainerNode
from nodes.train.schedule import ConstantLRSchedule
from nodes.smoke_tests.smoke_test_trainer_cancellation import _FiniteBatches


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


class _FakeResourceControl(ResourceControlHandle):
    """Logs every call into one shared, ordered list -- enough to prove
    the actual bracketing order without a mocking framework."""

    def __init__(self, events: list):
        self._events = events
        self._offloadable: set = set()

    def register(self, name, resident, offloadable: bool = False) -> None:
        if offloadable:
            self._offloadable.add(name)
        self._events.append(f"register:{name}:offloadable={offloadable}")

    def before_step(self, step: int) -> None:
        self._events.append(f"before_step:{step}")

    def ensure_loaded(self, name: str) -> None:
        self._events.append(f"ensure_loaded:{name}")

    def release(self, name: str) -> None:
        check(name in self._offloadable, f"release({name!r}) called but not registered offloadable")
        self._events.append(f"release:{name}")


class _FakeModel(TrainableModel):
    def __init__(self, events: list):
        self._events = events
        # Non-zero start, and forward()'s return value genuinely depends on p (no
        # `* 0`, unlike smoke_test_trainer_cancellation.py's own _FakeModel -- that
        # one only needs call counts, not real learning; this file's own
        # check_real_optimizer_... below needs an actual, non-zero, data-dependent
        # gradient to reach p, or "did the value change" would be true for the
        # wrong reason (decoupled weight decay alone can move a non-zero p even
        # with a structurally-zeroed gradient) or trivially false (zero p, zero
        # grad, zero decay-of-zero -- no observable change either way, whether or
        # not the real wiring under test is correct).
        self.p = torch.nn.Parameter(torch.randn(4, 4) * 0.1)

    def forward(self, xc, t, ctx_emb, y):
        self._events.append("forward")
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
        raise AssertionError("model must never be offloaded by this node's own design")

    def reload(self, device=None):
        pass

    def release(self):
        pass


class _FakeTextEncoder:
    def __init__(self, events: list):
        self._events = events

    def encode(self, prompt, batch_size, height, width):
        self._events.append("encode")
        return torch.zeros(batch_size, 1, 4), torch.zeros(batch_size, 4)

    def footprint_bytes(self):
        return 0

    def offload(self):
        pass

    def reload(self, device=None):
        pass

    def release(self):
        pass


def _make_optimizer_class(base):
    class _FakeOptimizer(base):
        def __init__(self, events: list):
            self._events = events

        @property
        def lr(self):
            return 1e-4

        def update_lr(self, new_lr):
            pass

        def step(self, n_steps=1):
            self._events.append("optimizer_step")

        def zero_grad(self):
            pass

        def begin_step(self, sub_steps=1):
            self._events.append("begin_step")

        def prepare_next_pass(self):
            pass

        def offload_states_to_cpu(self):
            pass

        def reload_states_to_device(self, device=None):
            pass

        def decay_states(self, factor):
            pass

        def reset_states(self):
            pass

        def free_states(self):
            pass

        def footprint_bytes(self):
            return 8

    return _FakeOptimizer


_FakeOptimizer = _make_optimizer_class(OptimizerHandle)
_FakeFusedOptimizer = _make_optimizer_class(FusedOptimizerHandle)


def _run(optimizer, events) -> dict:
    node = ManagedLoRATrainerNode()
    node.context = ExecutionContext()
    model = _FakeModel(events)
    trainer = SimpleNamespace(unet=model, clip=_FakeTextEncoder(events))
    resource_control = _FakeResourceControl(events)
    result = node.build(
        trainer=trainer, batches=_FiniteBatches(), optimizer=optimizer,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        steps=2, resource_control=resource_control,
    )
    check(result["model"] is model, "must return the exact unet instance")
    return events


def check_contracts():
    print("[contracts]")
    check(not getattr(ManagedLoRATrainerNode, "__abstractmethods__", None),
          "must be concretely instantiable")
    check("resource_control" in ManagedLoRATrainerNode.INPUTS
          and ManagedLoRATrainerNode.INPUTS["resource_control"].required is True,
          "resource_control must be required")
    check("model" not in ManagedLoRATrainerNode.INPUTS and "text_encoder" not in ManagedLoRATrainerNode.INPUTS,
          "must take `trainer` bundled, not separate model/text_encoder ports")
    print("    PASS")


def check_model_is_registered_non_offloadable_and_never_released():
    print("[model: registered offloadable=False, offload() never called]")
    events: list = []
    _run(_FakeOptimizer(events), events)
    check("register:model:offloadable=False" in events, events)
    check(not any("release:model" in e for e in events), events)
    print("    PASS")


def check_text_encoder_and_optimizer_bracket_their_own_phase_each_step():
    print("[non-fused: ensure_loaded/encode/release for text_encoder, then "
          "ensure_loaded/[backward]/optimizer_step/release for optimizer -- once "
          "per step, both steps]")
    events: list = []
    _run(_FakeOptimizer(events), events)

    step_boundaries = [i for i, e in enumerate(events) if e.startswith("before_step:")]
    check(len(step_boundaries) == 2, events)
    for start, end in zip(step_boundaries, step_boundaries[1:] + [len(events)]):
        step_events = events[start:end]
        check(step_events == [
            step_events[0],  # before_step:N
            "ensure_loaded:text_encoder", "encode", "release:text_encoder",
            "forward", "ensure_loaded:optimizer", "optimizer_step", "release:optimizer",
        ], step_events)
    print("    PASS")


def check_fused_optimizer_still_brackets_around_backward_but_never_calls_step():
    print("[fused: optimizer.step() is never called (the real update happens in a "
          "backward hook this fake doesn't model, matching FusedOptimizerHandle's own "
          "documented no-op step()) -- but ensure_loaded/release still bracket "
          "backward the same as the non-fused case, just without an optimizer_step "
          "event in between]")
    events: list = []
    _run(_FakeFusedOptimizer(events), events)

    step_boundaries = [i for i, e in enumerate(events) if e.startswith("before_step:")]
    check(len(step_boundaries) == 2, events)
    for start, end in zip(step_boundaries, step_boundaries[1:] + [len(events)]):
        step_events = events[start:end]
        check(step_events == [
            step_events[0],  # before_step:N
            "ensure_loaded:text_encoder", "encode", "release:text_encoder",
            "begin_step", "forward", "ensure_loaded:optimizer", "release:optimizer",
        ], step_events)
    print("    PASS")


def check_real_optimizer_via_trainer_parameters_node_actually_updates_the_trained_parameter():
    print("[end-to-end, no fakes for the optimizer side: TrainerParametersNode's own "
          "params -> a real ComposedAdamWOptimizerNode -> ManagedLoRATrainerNode -- the "
          "same tensor object the whole way through, and a real optimizer.step() "
          "actually changes it]")
    events: list = []
    model = _FakeModel(events)
    trainer = SimpleNamespace(unet=model, clip=_FakeTextEncoder(events))

    params = TrainerParametersNode().build(trainer=trainer)["params"]
    check(params[0] is model.p, "TrainerParametersNode must hand back the model's own "
                                 "parameter object, not a copy")

    optimizer = ComposedAdamWOptimizerNode().build(params=params, lr=0.5, device="cpu")["optimizer"]
    check(optimizer.params[0] is model.p,
          "the constructed optimizer must be tracking the model's own parameter object")

    # A real BudgetedResourceControlHandle too, not the logging fake above -- proves
    # optimizer's own offload_states_to_cpu()/reload_states_to_device() (which
    # ensure_loaded()/release() call into) work against a real ComposedOptimizerHandle,
    # not just the DeviceResident interface in the abstract.
    resource_control = BudgetedResourceControlHandle(
        ResourceBudget(vram_budget_mb=1e9, vram_reserve_mb=0.0), device="cpu")

    node = ManagedLoRATrainerNode()
    node.context = ExecutionContext()
    before = model.p.detach().clone()

    result = node.build(
        trainer=trainer, batches=_FiniteBatches(), optimizer=optimizer,
        lr_schedule=ConstantLRSchedule(lr=0.5), loss_weighting=UniformLossWeighting(),
        steps=3, resource_control=resource_control,
    )

    after = model.p.detach().clone()
    check(not torch.equal(before, after),
          "model.p must have actually changed -- proves gradients flowed into the real "
          "parameter object and a real optimizer.step() applied them, not just that "
          "nothing raised")
    check(result["model"] is model, "must return the exact same model instance")
    print("    PASS")


def check_profile_prints_residency_lines_at_the_right_moments():
    print("[profile=True: EncodeConditioningPhase/BackwardAndOptimizerStepPhase each "
          "print their own loaded/released line -- MonitoringPhase runs too late to "
          "ever show this (both are already released by the time it runs), which is "
          "exactly the bug this closes]")
    import io
    import contextlib

    events: list = []
    model = _FakeModel(events)
    trainer = SimpleNamespace(unet=model, clip=_FakeTextEncoder(events))
    node = ManagedLoRATrainerNode()
    node.context = ExecutionContext()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        node.build(
            trainer=trainer, batches=_FiniteBatches(), optimizer=_FakeOptimizer(events),
            lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
            steps=1, resource_control=_FakeResourceControl(events), profile=True,
        )
    output = buf.getvalue()
    check("[residency] text_encoder loaded:" in output, output)
    check("[residency] text_encoder released:" in output, output)
    check("[residency] optimizer loaded:" in output, output)
    check("[residency] optimizer released:" in output, output)
    print("    PASS")


def main():
    check_contracts()
    check_model_is_registered_non_offloadable_and_never_released()
    check_text_encoder_and_optimizer_bracket_their_own_phase_each_step()
    check_fused_optimizer_still_brackets_around_backward_but_never_calls_step()
    check_real_optimizer_via_trainer_parameters_node_actually_updates_the_trained_parameter()
    check_profile_prints_residency_lines_at_the_right_moments()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
