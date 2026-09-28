"""Checks nodes/train/managed.py's ManagedLoRATrainerNode -- specifically
that optimizer/text_encoder residency is correctly wired to
AdaptiveResidencyController's own decision (ensure_loaded always fires;
release only fires when the controller actually decided to), for both a
plain and a fused optimizer, and that model is never released at all.
Everything else this node does (diffusion math, gating, monitoring) is
either a straight, low-risk read of the main route's own equivalent, or
already covered by this node's own docstrings' worked reasoning -- this
file deliberately doesn't re-verify all of that, only the choreography
that's genuinely new.

AdaptiveResidencyController's own decision logic (calibrate, then
release nothing vs. release smallest-first) is tested directly, with
fake numbers, in smoke_test_adaptive_residency_controller.py -- this
file can't drive that branch through a full ManagedLoRATrainerNode.build()
run at all: every check here runs on CPU tensors, where
DeviceContext.for_device() returns _NullDeviceContext (memory_stats()
always None), so the controller always falls back to "stay resident"
immediately (see its own record_step_peak() docstring for why that's
the correct fallback, not a gap). check_phases_actually_call_release_
when_the_controller_decides_to and check_profile_prints_residency_lines_
at_the_right_moments below drive a controller directly instead, to
cover the "release actually happens" side of the wiring too.

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

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from types import SimpleNamespace

import torch

from nodes.core import ExecutionContext
from nodes.memory.control_handle import BudgetedResourceControlHandle, ResourceControlHandle
from nodes.model.handle import TrainableModel
from nodes.model.text_encoder_cache import CachingTextEncoder
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

    def usable_budget_mb(self):
        return 8000.0  # never actually consulted in this file's own tests: every one
        # of them runs on CPU tensors, where DeviceContext.for_device() returns
        # _NullDeviceContext (memory_stats() always None) -- AdaptiveResidencyController
        # decides immediately, ignoring usable_budget_mb() entirely, and always decides
        # "stay resident" (see its own record_step_peak() docstring, and
        # smoke_test_adaptive_residency_controller.py for the real, direct coverage of
        # its decision logic with fake, non-None numbers).


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
        self.prompt_encodes = 0
        self.unloaded = False

    def encode(self, prompt, batch_size, height, width):
        self._events.append("encode")
        return torch.zeros(batch_size, 1, 4), torch.zeros(batch_size, 4)

    def encode_prompt_only(self, prompt, batch_size):
        # Reached only through a CachingTextEncoder's cache-miss path --
        # the two halves below are what prewarm warms and what a
        # step-time cache hit skips entirely.
        self._events.append("encode_prompt_only")
        self.prompt_encodes += 1
        return torch.zeros(batch_size, 1, 4), torch.zeros(batch_size, 4)

    def resolution_embedding(self, height, width, batch_size):
        self._events.append("resolution_embedding")
        return torch.zeros(batch_size, 2)

    def unload(self):
        self._events.append("unload")
        self.unloaded = True

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


class _FiniteEpoch:
    """Finite per iteration (one pass = two batches), re-iterable --
    the shape ManagedDatasetSourceNode's real output has, and what
    prewarm_text_encoder's full-dataset discovery pass assumes.
    Deliberately not _FiniteBatches (smoke_test_trainer_cancellation's),
    which yields forever per iter() and would hang any full pass.
    Two distinct prompts -> two unique (prompt, batch_size, h, w) keys
    at identical resolution -> one shared resolution-cache entry."""

    def __iter__(self):
        yield {"x_t": torch.randn(2, 4, 4, 4), "target": torch.randn(2, 4, 4, 4),
               "t": torch.tensor([500, 500]), "prompt": "a"}
        yield {"x_t": torch.randn(2, 4, 4, 4), "target": torch.randn(2, 4, 4, 4),
               "t": torch.tensor([500, 500]), "prompt": "b"}


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


def check_ensure_loaded_always_fires_but_release_does_not_when_calibration_cannot_resolve():
    print("[non-fused, on CPU (no memory_stats concept -- see AdaptiveResidencyController's "
          "own record_step_peak() docstring): ensure_loaded fires every step for both "
          "text_encoder and optimizer, release never fires for either -- calibration "
          "immediately falls back to \"stay resident\", the correct answer when there's "
          "nothing to measure against, not a gap in this test]")
    events: list = []
    _run(_FakeOptimizer(events), events)

    step_boundaries = [i for i, e in enumerate(events) if e.startswith("before_step:")]
    check(len(step_boundaries) == 2, events)
    for start, end in zip(step_boundaries, step_boundaries[1:] + [len(events)]):
        step_events = events[start:end]
        check(step_events == [
            step_events[0],  # before_step:N
            "ensure_loaded:text_encoder", "encode",
            "forward", "ensure_loaded:optimizer", "optimizer_step",
        ], step_events)
    print("    PASS")


def check_fused_optimizer_same_fallback_never_calls_step_either_way():
    print("[fused: same \"stay resident\" fallback, and optimizer.step() is still never "
          "called regardless (the real update happens in a backward hook this fake "
          "doesn't model, matching FusedOptimizerHandle's own documented no-op step())]")
    events: list = []
    _run(_FakeFusedOptimizer(events), events)

    step_boundaries = [i for i, e in enumerate(events) if e.startswith("before_step:")]
    check(len(step_boundaries) == 2, events)
    for start, end in zip(step_boundaries, step_boundaries[1:] + [len(events)]):
        step_events = events[start:end]
        check(step_events == [
            step_events[0],  # before_step:N
            "ensure_loaded:text_encoder", "encode",
            "begin_step", "forward", "ensure_loaded:optimizer",
        ], step_events)
    print("    PASS")


def check_phases_actually_call_release_when_the_controller_decides_to():
    print("[the other half of the wiring: when AdaptiveResidencyController *has* decided "
          "to release something (driven directly here with a fake peak, since CPU can't "
          "produce one), EncodeConditioningPhase/BackwardAndOptimizerStepPhase actually "
          "call release() -- not exercised by any check above, all of which hit the "
          "always-stays-resident CPU fallback instead]")
    from nodes.train.managed import (AdaptiveResidencyController, BackwardAndOptimizerStepPhase,
                                      EncodeConditioningPhase, ManagedStepState)

    events: list = []
    model = _FakeModel(events)
    text_encoder = _FakeTextEncoder(events)
    optimizer = _FakeOptimizer(events)
    resource_control = _FakeResourceControl(events)
    resource_control.register("text_encoder", text_encoder, offloadable=True)
    resource_control.register("optimizer", optimizer, offloadable=True)

    controller = AdaptiveResidencyController(
        usable_mb=100.0, candidates={"text_encoder": text_encoder, "optimizer": optimizer},
        calibration_steps=1)
    controller.record_step_peak({"peak_reserved_mb": 99999.0})  # forces "release everything"
    check(controller.should_release("text_encoder") and controller.should_release("optimizer"),
          "sanity: the controller must actually have decided to release both")

    state = ManagedStepState(step=0, batch={"x_t": torch.zeros(1, 4, 4, 4), "prompt": "x"},
                              model=model, device=torch.device("cpu"))
    state.extras["x_t"] = state.batch["x_t"]
    EncodeConditioningPhase(text_encoder, resource_control, controller).run(state)
    check("release:text_encoder" in events, events)

    state.extras["loss"] = model.p.sum()
    BackwardAndOptimizerStepPhase(optimizer, is_fused=False,
                                   resource_control=resource_control, controller=controller).run(state)
    check("release:optimizer" in events, events)
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
          "ever show this even when release() does fire, which is the bug this "
          "closes (see this module's own docstring). Forces a real release decision "
          "directly (CPU can't produce one through the full node -- see "
          "check_phases_actually_call_release_when_the_controller_decides_to above) "
          "so both the loaded and released lines actually get exercised, not just "
          "loaded.]")
    import contextlib
    import io

    from nodes.train.managed import (AdaptiveResidencyController, BackwardAndOptimizerStepPhase,
                                      DeviceContext, EncodeConditioningPhase, ManagedStepState)

    events: list = []
    model = _FakeModel(events)
    text_encoder = _FakeTextEncoder(events)
    optimizer = _FakeOptimizer(events)
    resource_control = _FakeResourceControl(events)
    resource_control.register("text_encoder", text_encoder, offloadable=True)
    resource_control.register("optimizer", optimizer, offloadable=True)
    controller = AdaptiveResidencyController(
        usable_mb=100.0, candidates={"text_encoder": text_encoder, "optimizer": optimizer},
        calibration_steps=1)
    controller.record_step_peak({"peak_reserved_mb": 99999.0})
    device_ctx = DeviceContext.for_device("cpu")

    state = ManagedStepState(step=0, batch={"x_t": torch.zeros(1, 4, 4, 4), "prompt": "x"},
                              model=model, device=torch.device("cpu"))
    state.extras["x_t"] = state.batch["x_t"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        EncodeConditioningPhase(text_encoder, resource_control, controller,
                                 device_ctx=device_ctx, profile=True).run(state)
        state.extras["loss"] = model.p.sum()
        BackwardAndOptimizerStepPhase(optimizer, is_fused=False, resource_control=resource_control,
                                       controller=controller, device_ctx=device_ctx,
                                       profile=True).run(state)
    output = buf.getvalue()
    check("[residency] text_encoder loaded:" in output, output)
    check("[residency] text_encoder released:" in output, output)
    check("[residency] optimizer loaded:" in output, output)
    check("[residency] optimizer released:" in output, output)
    print("    PASS")


def check_step_timing_off_by_default_and_on_when_requested():
    print("[TRAIN_STEP_TIMING: off by default (identical to the pre-instrumentation "
          "loop -- no prints, no sync calls), on when the env var is set]")
    from nodes.train.managed import ManagedStepPhase, ManagedStepState, ManagedTrainingStepPipeline

    class _RecordingPhase(ManagedStepPhase):
        def __init__(self, name):
            self.name = name
            self.ran = False

        def run(self, state):
            self.ran = True
            state.extras.setdefault("order", []).append(self.name)
            return state

    phase_a, phase_b = _RecordingPhase("a"), _RecordingPhase("b")
    pipeline = ManagedTrainingStepPipeline([phase_a, phase_b])
    state = ManagedStepState(step=0, batch=None, model=None, device="cpu")

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = pipeline.run_step(state)
    assert phase_a.ran and phase_b.ran
    assert result.extras["order"] == ["a", "b"], "phases must still run in the given order"
    assert buf.getvalue() == "", f"must print nothing when TRAIN_STEP_TIMING is unset: {buf.getvalue()!r}"
    print("    PASS: default path unchanged, no output")

    original = os.environ.get("TRAIN_STEP_TIMING")
    os.environ["TRAIN_STEP_TIMING"] = "1"
    try:
        phase_c, phase_d = _RecordingPhase("c"), _RecordingPhase("d")
        timed_pipeline = ManagedTrainingStepPipeline([phase_c, phase_d])
        state2 = ManagedStepState(step=7, batch=None, model=None, device="cpu")
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            result2 = timed_pipeline.run_step(state2)
        assert phase_c.ran and phase_d.ran
        assert result2.extras["order"] == ["c", "d"]
        out = buf2.getvalue()
        assert "step 7 timing" in out and "_RecordingPhase=" in out, out
        print(f"    PASS: timing line printed when enabled: {out.strip()}")
    finally:
        if original is None:
            os.environ.pop("TRAIN_STEP_TIMING", None)
        else:
            os.environ["TRAIN_STEP_TIMING"] = original


def check_prewarm_text_encoder_warms_unloads_and_skips_ensure_loaded():
    print("[prewarm_text_encoder: wraps clip in a cache sized to the dataset, warms "
          "every key from the same batches object, unloads the encoder, and the "
          "encode phase stops forcing ensure_loaded]")

    # Part 1: fresh wrap (no cache_text_encoder set on the config node).
    events = []
    node = ManagedLoRATrainerNode()
    node.context = ExecutionContext()
    model = _FakeModel(events)
    inner = _FakeTextEncoder(events)
    trainer = SimpleNamespace(unet=model, clip=inner)
    rc = _FakeResourceControl(events)
    result = node.build(
        trainer=trainer, batches=_FiniteEpoch(), optimizer=_FakeOptimizer(events),
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        steps=2, resource_control=rc, prewarm_text_encoder=True)
    check(result["model"] is model, "must return the exact unet instance")
    check(isinstance(trainer.clip, CachingTextEncoder),
          "fresh wrap expected when cache_text_encoder wasn't set")
    check(trainer.clip._resource_control is rc,
          "the freshly built cache must be bound to the training handle")
    check(trainer.clip._max_entries >= 2,
          "cache must be sized to the dataset's key count so the warm pass can't evict itself")
    check(inner.unloaded, "encoder must be unloaded once the warm pass completes")
    first_step = events.index("before_step:0")
    step_ensures = [e for e in events[first_step:] if e == "ensure_loaded:text_encoder"]
    check(not step_ensures,
          f"no ensure_loaded('text_encoder') may fire during training when prewarmed "
          f"(would re-upload the just-unloaded encoder); got {step_ensures}")
    warm_inner = [i for i, e in enumerate(events)
                  if e in ("encode_prompt_only", "resolution_embedding")]
    check(warm_inner and max(warm_inner) < first_step,
          "every inner-encoder call must happen during the warm pass, before step 0")
    check(inner.prompt_encodes == 2,
          f"one inner CLIP pass per distinct prompt during warm (2), got {inner.prompt_encodes}")
    check(len([e for e in events if e == "forward"]) == 2, "both steps must still train")

    # Part 2: an existing wrap (LoRATrainingConfigNode's cache_text_encoder=True
    # shape) is kept as-is -- not double-wrapped -- and gets its handle late-bound.
    events2 = []
    node2 = ManagedLoRATrainerNode()
    node2.context = ExecutionContext()
    model2 = _FakeModel(events2)
    inner2 = _FakeTextEncoder(events2)
    pre_wrapped = CachingTextEncoder(inner2)
    trainer2 = SimpleNamespace(unet=model2, clip=pre_wrapped)
    rc2 = _FakeResourceControl(events2)
    node2.build(
        trainer=trainer2, batches=_FiniteEpoch(), optimizer=_FakeOptimizer(events2),
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        steps=1, resource_control=rc2, prewarm_text_encoder=True)
    check(trainer2.clip is pre_wrapped, "an existing cache wrap must be kept, not replaced")
    check(pre_wrapped._resource_control is rc2,
          "an existing wrap must get the handle late-bound (bind_resource_control)")
    check(inner2.unloaded, "existing wrap's inner encoder must still be unloaded after warm")
    first_step2 = events2.index("before_step:0")
    step_ensures2 = [e for e in events2[first_step2:] if e == "ensure_loaded:text_encoder"]
    check(not step_ensures2, f"same skip rule applies through an existing wrap; got {step_ensures2}")
    print("    PASS")


def main():
    check_contracts()
    check_model_is_registered_non_offloadable_and_never_released()
    check_ensure_loaded_always_fires_but_release_does_not_when_calibration_cannot_resolve()
    check_fused_optimizer_same_fallback_never_calls_step_either_way()
    check_phases_actually_call_release_when_the_controller_decides_to()
    check_real_optimizer_via_trainer_parameters_node_actually_updates_the_trained_parameter()
    check_profile_prints_residency_lines_at_the_right_moments()
    check_step_timing_off_by_default_and_on_when_requested()
    check_prewarm_text_encoder_warms_unloads_and_skips_ensure_loaded()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
