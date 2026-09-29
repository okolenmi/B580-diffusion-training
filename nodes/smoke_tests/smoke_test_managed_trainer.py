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
    for port_name in ("grad_accum", "grad_clip_max_norm", "save_every_n_steps",
                      "save_prefix", "project_layout"):
        check(port_name in ManagedLoRATrainerNode.INPUTS,
              f"{port_name} port must exist on ManagedLoRATrainerNode")
    check(ManagedLoRATrainerNode.INPUTS["grad_accum"].default == 1
          and ManagedLoRATrainerNode.INPUTS["grad_clip_max_norm"].default == 0.0
          and ManagedLoRATrainerNode.INPUTS["save_every_n_steps"].default == 0,
          "all three cadence/clip ports must default to the pre-existing "
          "behavior (no accumulation, no clip, no saves)")
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
    assert "timing_ms" not in result.extras, (
        "unset TRAIN_STEP_TIMING must leave no timing_ms in extras at all "
        "(zero overhead means zero writes, not just zero prints)")
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
        # extras["timing_ms"] is the channel MonitoringPhase reads to build the
        # report's {label}_ms / step_total_ms keys -- snake_case labels from
        # the shared _phase_label (both instances here are the same class, so
        # they share one label and the second write wins; real phase lists are
        # all distinct classes, and it's the key *shape* under test).
        timing = result2.extras.get("timing_ms")
        assert timing and all(v >= 0 for v in timing.values()), timing
        # "_RecordingPhase" -> strip "Phase" -> "_Recording" -> snake, so the
        # private class's own leading underscore survives as a double one.
        assert list(timing) == ["__recording"], timing

        # End-to-end: those extras keys must actually reach the monitor report
        # as {label}_ms + step_total_ms (managed MonitoringPhase's report
        # path), step_total_ms exactly their sum.
        from nodes.monitor.handle import MonitorHandle

        class _Cap(MonitorHandle):
            def __init__(self):
                self.reports = []

            def report(self, d):
                self.reports.append(dict(d))

        cap = _Cap()
        _run_managed([], _recording_optimizer([], fused=False), steps=1, monitor=cap)
        step_reports = [r for r in cap.reports if "type" not in r]
        assert len(step_reports) == 1, step_reports
        rep = step_reports[0]
        phase_keys = [k for k in rep if k.endswith("_ms") and k != "step_total_ms"]
        assert phase_keys and "step_total_ms" in rep, sorted(rep)
        assert rep["step_total_ms"] >= 0 and all(rep[k] >= 0 for k in phase_keys)
        assert abs(rep["step_total_ms"] - sum(rep[k] for k in phase_keys)) < 1e-6, (
            f"step_total_ms ({rep['step_total_ms']}) != sum of {phase_keys}")
        print(f"    PASS: timing line printed when enabled: {out.strip()}")
        print(f"    PASS: report carries timing: { {k: round(v, 1) for k, v in rep.items() if k.endswith('_ms')} }")
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


def _recording_optimizer(events, fused: bool):
    """Recording optimizer for the grad_accum/clip/save cadence checks:
    counts *every* call those phases make -- update_lr (and the LR values),
    zero_grad, step, begin_step including its sub_steps argument, and
    prepare_next_pass. _FakeOptimizer above records only some of these and
    can't tell begin_step(sub_steps=1) from begin_step(sub_steps=K), which
    is exactly the distinction the fused accumulation path turns on."""
    base = FusedOptimizerHandle if fused else OptimizerHandle

    class _Recording(base):
        def __init__(self, events):
            self._events = events
            self.update_lrs = []

        @property
        def lr(self):
            return 1e-4

        def update_lr(self, new_lr):
            self.update_lrs.append(new_lr)

        def step(self, n_steps=1):
            self._events.append("optimizer_step")

        def zero_grad(self):
            self._events.append("zero_grad")

        def begin_step(self, sub_steps=1):
            self._events.append(f"begin_step:{sub_steps}")

        def prepare_next_pass(self):
            self._events.append("prepare_next_pass")

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

    return _Recording(events)


def _run_managed(events, optimizer, *, steps, grad_accum=1, extra=None, batches=None,
                 model=None, resource_control=None, monitor=None, on_step=None):
    node = ManagedLoRATrainerNode()
    node.context = ExecutionContext()
    model = model or _FakeModel(events)
    trainer = SimpleNamespace(unet=model, clip=_FakeTextEncoder(events))
    inputs = dict(
        trainer=trainer, batches=batches or _FiniteBatches(), optimizer=optimizer,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        steps=steps, resource_control=resource_control or _FakeResourceControl(events),
        grad_accum=grad_accum, monitor=monitor)
    if on_step is not None:
        inputs["on_step"] = on_step
    if extra:
        inputs.update(extra)
    result = node.build(**inputs)
    check(result["model"] is model, "must return the exact unet instance")
    return model


def check_grad_accum_runs_one_optimizer_update_per_window():
    print("[grad_accum=2, non-fused, steps=3: 6 micro-steps of forward/"
          "before_step, but zero_grad/update_lr/optimizer_step/on_step each "
          "fire exactly once per window]")
    events: list = []
    optimizer = _recording_optimizer(events, fused=False)
    on_steps = []
    _run_managed(events, optimizer, steps=3, grad_accum=2,
                 on_step=lambda s, l: on_steps.append(s))
    check(len([e for e in events if e == "forward"]) == 6,
          f"one forward per micro-step (3*2=6); got {len([e for e in events if e == 'forward'])}")
    check(len([e for e in events if e.startswith("before_step")]) == 6,
          "resource-control safety net runs per micro-step, not per window")
    check(len([e for e in events if e == "optimizer_step"]) == 3,
          f"one optimizer.step per window (3); got {len([e for e in events if e == 'optimizer_step'])}")
    check(len([e for e in events if e == "zero_grad"]) == 3,
          "zeroing every micro-step would wipe the window's gradients -- must be once per window")
    check(len(optimizer.update_lrs) == 3,
          f"update_lr fires once per window (3), got {len(optimizer.update_lrs)}")
    check(on_steps == [0, 1, 2],
          f"on_step once per optimizer step with its index; got {on_steps}")
    print("    PASS")


def check_fused_grad_accum_spans_sub_steps():
    print("[grad_accum=3, fused, steps=2: begin_step(sub_steps=3) once per "
          "window, prepare_next_pass between passes, step() never called "
          "(the boundary backward's hook is the update)]")
    events: list = []
    optimizer = _recording_optimizer(events, fused=True)
    _run_managed(events, optimizer, steps=2, grad_accum=3)
    begins = [e for e in events if e.startswith("begin_step")]
    check(begins == ["begin_step:3", "begin_step:3"],
          f"begin_step must be called once per window with sub_steps=grad_accum; got {begins}")
    check(len([e for e in events if e == "prepare_next_pass"]) == 2 * 2,
          "one prepare_next_pass per non-boundary micro-step (window of 3 -> 2 each)")
    check(not [e for e in events if e == "optimizer_step"],
          "fused must never call step() -- the hook already applied the update")
    check(len([e for e in events if e == "forward"]) == 6,
          "all 6 micro-steps still ran")
    print("    PASS")


def check_grad_accum_flows_real_gradients_into_real_optimizer():
    print("[grad_accum=2 through a real ComposedAdamWOptimizerHandle: the "
          "loss/K scaling + boundary step still moves the real parameter -- "
          "the fake-optimizer checks above prove call counts, this proves "
          "the arithmetic produces a real update]")
    events: list = []
    model = _FakeModel(events)
    trainer = SimpleNamespace(unet=model, clip=_FakeTextEncoder(events))
    params = TrainerParametersNode().build(trainer=trainer)["params"]
    optimizer = ComposedAdamWOptimizerNode().build(params=params, lr=0.5, device="cpu")["optimizer"]
    resource_control = BudgetedResourceControlHandle(
        ResourceBudget(vram_budget_mb=1e9, vram_reserve_mb=0.0), device="cpu")
    before = model.p.detach().clone()
    node = ManagedLoRATrainerNode()
    node.context = ExecutionContext()
    node.build(
        trainer=trainer, batches=_FiniteBatches(), optimizer=optimizer,
        lr_schedule=ConstantLRSchedule(lr=0.5), loss_weighting=UniformLossWeighting(),
        steps=2, resource_control=resource_control, grad_accum=2)
    check(not torch.equal(before, model.p.detach()),
          "real parameter must change under grad_accum=2 (window gradient reached step())")
    print("    PASS")


def check_grad_accum_loss_scaling_stays_internal():
    print("[LossPhase backward_scale: extras['loss'] stays the reportable "
          "unscaled loss; extras['loss_for_backward'] carries loss/grad_accum "
          "and is only set when grad_accum > 1]")
    from nodes.train.managed import LossPhase as ManagedLossPhase

    def state():
        return SimpleNamespace(extras={
            "pred": torch.zeros(2, 4),
            # per-sample MSE: sample0 = (4^2)/4 = 4.0, sample1 = 0.0
            "target": torch.tensor([[0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 0.0]]),
            "sigma": torch.tensor([1.0, 1.0])})

    st = state()
    ManagedLossPhase(UniformLossWeighting(), backward_scale=0.5).run(st)
    check(abs(st.extras["loss"].item() - 2.0) < 1e-6,
          f"reported loss must stay unscaled (2.0); got {st.extras['loss'].item()}")
    check("loss_for_backward" in st.extras
          and abs(st.extras["loss_for_backward"].item() - 1.0) < 1e-6,
          "loss_for_backward must be loss/grad_accum (2.0/2 = 1.0)")

    st2 = state()
    ManagedLossPhase(UniformLossWeighting()).run(st2)
    check("loss_for_backward" not in st2.extras,
          "grad_accum=1 must keep the old single-tensor path (no extra graph node)")
    print("    PASS")


def check_grad_clip_applies_once_per_optimizer_step():
    print("[grad_clip_max_norm: clip_grad_norm_ called exactly once per "
          "optimizer step (boundary only, never on interior micro-steps), "
          "with the model's own trainable parameters and the port's value; "
          "clip=0 never calls it]")
    import torch.nn.utils as torch_utils

    from nodes.monitor.handle import MonitorHandle

    class _CaptureMonitor(MonitorHandle):
        def __init__(self):
            self.reports = []

        def report(self, data):
            self.reports.append(dict(data))

    original = torch_utils.clip_grad_norm_
    calls = []

    def spy(params, max_norm):
        calls.append((list(params), float(max_norm)))
        return original(params, max_norm)

    torch_utils.clip_grad_norm_ = spy
    try:
        events: list = []
        optimizer = _recording_optimizer(events, fused=False)
        monitor = _CaptureMonitor()
        model = _run_managed(events, optimizer, steps=2, grad_accum=2,
                             extra={"grad_clip_max_norm": 3.5}, monitor=monitor)
        check(len(calls) == 2,
              f"clip must fire once per optimizer step (2), got {len(calls)}")
        check(all(m == 3.5 for _, m in calls),
              f"clip must use the port's max_norm (3.5); got {[m for _, m in calls]}")
        check(all(p and p[0] is model.p for p, _ in calls),
              "clip must run over the model's own trainable parameters")
        # clip_grad_norm_ already measured the pre-clip total norm; the stash
        # must put it on every step report as a plain float -- the value the
        # monitor dashboard's grad-norm readout plots.
        step_reports = [r for r in monitor.reports if "type" not in r]
        check(len(step_reports) == 2
              and all(isinstance(r.get("grad_norm"), float) for r in step_reports),
              "clip measured the norm -> every step report carries a float "
              f"grad_norm; got {[r.get('grad_norm') for r in step_reports]}")

        events2: list = []
        optimizer2 = _recording_optimizer(events2, fused=False)
        calls.clear()
        monitor2 = _CaptureMonitor()
        _run_managed(events2, optimizer2, steps=2, grad_accum=2, monitor=monitor2)
        check(not calls, "grad_clip_max_norm=0 (default) must not clip at all")
        check(all("grad_norm" not in r for r in monitor2.reports if "type" not in r),
              "no clip -> no grad_norm key (absent means unmeasured, never a "
              "fabricated 0)")
    finally:
        torch_utils.clip_grad_norm_ = original
    print("    PASS")


def check_build_rejects_invalid_training_shapes():
    print("[build-time contract checks: grad_accum < 1, negative clip, "
          "clip on a fused optimizer, empty save_prefix with saving on -- "
          "each raises instead of silently doing the wrong thing]")

    def expect_valueerror(extra, fragment):
        events: list = []
        node = ManagedLoRATrainerNode()
        node.context = ExecutionContext()
        model = _FakeModel(events)
        trainer = SimpleNamespace(unet=model, clip=_FakeTextEncoder(events))
        inputs = dict(
            trainer=trainer, batches=_FiniteBatches(),
            optimizer=_recording_optimizer(events, fused=bool(extra.get("_fused"))),
            lr_schedule=ConstantLRSchedule(lr=1e-4),
            loss_weighting=UniformLossWeighting(), steps=1,
            resource_control=_FakeResourceControl(events))
        inputs.update({k: v for k, v in extra.items() if not k.startswith("_")})
        try:
            node.build(**inputs)
        except ValueError as e:
            check(fragment in str(e),
                  f"error for {extra} must mention {fragment!r}; got: {e}")
            return
        raise AssertionError(f"build() with {extra} should have raised ValueError")

    expect_valueerror({"grad_accum": 0}, "grad_accum")
    expect_valueerror({"grad_clip_max_norm": -1.0}, "grad_clip_max_norm")
    expect_valueerror({"_fused": True, "grad_clip_max_norm": 1.0}, "fused")
    expect_valueerror({"save_every_n_steps": 1, "save_prefix": "   "}, "save_prefix")
    print("    PASS")


def check_intermediate_save_writes_numbered_checkpoints():
    print("[save_every_n_steps: numbered LoRA safetensors land in the "
          "sandboxed loras dir at each boundary step (atomic, no .tmp "
          "leftovers), and cadence counts optimizer steps]")
    import shutil
    import tempfile

    from nodes.components.layout import ProjectLayout

    tmp = Path(tempfile.mkdtemp(prefix="smoke_managed_save_"))
    try:
        layout = ProjectLayout(
            comfy_dir=tmp, checkpoints_dir=tmp / "ckpts", loras_dir=tmp / "loras",
            datasets_dir=tmp / "ds", runs_dir=tmp / "runs")

        class _SaveableModel(_FakeModel):
            def trained_state_dict(self):
                return {"smoke.w": torch.zeros(2, 2)}

        events: list = []
        model = _SaveableModel(events)
        _run_managed(events, _recording_optimizer(events, fused=False), steps=2,
                     model=model, extra={"save_every_n_steps": 1,
                                         "save_prefix": "smoke_step",
                                         "project_layout": layout})
        f1 = layout.loras_dir / "smoke_step_000001.safetensors"
        f2 = layout.loras_dir / "smoke_step_000002.safetensors"
        check(f1.exists() and f2.exists(),
              f"both step checkpoints must exist; got {sorted(p.name for p in layout.loras_dir.glob('*'))}")
        check(not list(layout.loras_dir.glob("*.tmp")),
              "atomic write must leave no .tmp file behind")

        # save_every_n_steps=2: only the boundary optimizer step lands.
        events2: list = []
        _run_managed(events2, _recording_optimizer(events2, fused=False), steps=2,
                     model=_SaveableModel(events2),
                     extra={"save_every_n_steps": 2, "save_prefix": "smoke_two",
                            "project_layout": layout})
        check((layout.loras_dir / "smoke_two_000002.safetensors").exists()
              and not (layout.loras_dir / "smoke_two_000001.safetensors").exists(),
              "cadence must count optimizer steps: only step 2 saved with save_every=2")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("    PASS")


def check_monitor_reports_once_per_optimizer_step_with_t_buckets():
    from nodes.monitor.handle import MonitorHandle

    class _CaptureMonitor(MonitorHandle):
        def __init__(self):
            self.reports = []

        def report(self, data):
            self.reports.append(dict(data))

    class _TBatches:
        """t=[100, 800] every batch -> loss_t_low + loss_t_high present,
        loss_t_mid absent (no mid-t samples -> no key, not a zero)."""
        def __iter__(self):
            while True:
                yield {"x_t": torch.randn(2, 4, 4, 4), "target": torch.randn(2, 4, 4, 4),
                       "t": torch.tensor([100, 800]), "prompt": "x"}

    print("[monitor diagnostics: one report per optimizer step under "
          "grad_accum (not one per micro-step), each carrying the raw "
          "per-t-bucket losses for the buckets the window's batches "
          "actually sampled, plus exactly one terminal run_end]")
    events: list = []
    monitor = _CaptureMonitor()
    _run_managed(events, _recording_optimizer(events, fused=False), steps=2,
                 grad_accum=2, batches=_TBatches(), monitor=monitor)
    # Step-shaped reports only: run_end is a *different* message type (a
    # terminal status the dashboard needs to tell "finished" from "hung"),
    # not a step report -- counting it would be counting the wrong thing.
    step_reports = [r for r in monitor.reports if "type" not in r]
    check(len(step_reports) == 2,
          f"one report per optimizer step (2), got {len(step_reports)}")
    ends = [r for r in monitor.reports if r.get("type") == "run_end"]
    check(len(ends) == 1 and ends[0]["step"] == 2 and ends[0]["cancelled"] is False,
          f"exactly one run_end after 2 completed steps, not cancelled; got {ends}")
    for rep in step_reports:
        check(rep["step"] in (0, 1) and isinstance(rep["loss"], float),
              f"report must carry the optimizer step and window-averaged loss; got {rep.get('step')}")
        check("loss_t_low" in rep and "loss_t_high" in rep,
              f"window sampled t=100 and t=800 -> both bucket keys required; got {sorted(rep)}")
        check("loss_t_mid" not in rep,
              "no mid-t samples in the window -> key omitted (chart gap, not a fabricated 0)")
        check(isinstance(rep["loss_t_low"], float) and isinstance(rep["loss_t_high"], float),
              "bucket values must be plain floats for JSON transport")
        # New report keys (monitor dashboard upgrade): the fake handle states
        # a usable budget, so the ceiling must ride along on every report;
        # grad_norm/timing keys stay absent because this run clips nothing
        # and never opted into TRAIN_STEP_TIMING -- present-or-absent is the
        # contract, and absent must not be a fabricated 0.
        check(rep.get("vram_budget_mb") == 8000.0,
              f"vram_budget_mb must carry the handle's usable budget; got {rep.get('vram_budget_mb')}")
        check("grad_norm" not in rep and "step_total_ms" not in rep,
              "grad_norm/timing keys must stay absent when clipping is off and "
              "TRAIN_STEP_TIMING is unset (no fabricated values)")
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
    check_grad_accum_runs_one_optimizer_update_per_window()
    check_fused_grad_accum_spans_sub_steps()
    check_grad_accum_flows_real_gradients_into_real_optimizer()
    check_grad_accum_loss_scaling_stays_internal()
    check_grad_clip_applies_once_per_optimizer_step()
    check_build_rejects_invalid_training_shapes()
    check_intermediate_save_writes_numbered_checkpoints()
    check_monitor_reports_once_per_optimizer_step_with_t_buckets()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
