"""Numerical correctness check for nodes/train/step_pipeline.py's phases,
run through the real SupervisedLoRATrainerNode.build() entry point --
the highest-risk test in this backlog item, since this is the core
training loop everything else feeds into.

Not covered by smoke_test_trainer_cancellation.py (loop control flow) or
smoke_test_trainer_profiling.py (timing dict shape): whether the actual
MATH is right -- xc/sigma computed correctly, loss computed correctly
from them, gradients correct, the resulting parameter update correct.
Checked against an independent reference computed by hand in this file,
calling the same real DiffusionProcess/LossWeighting objects directly
(not through the pipeline), not by re-deriving what the pipeline should
do from memory.

A plain hand-written SGD OptimizerHandle is used deliberately instead of
a real one (AdamW etc.) -- decouples "is the pipeline's wiring correct"
(this test's job) from "is Adam's math correct" (already proven by
smoke_test_adamw_equivalence.py); SGD's update rule (p -= lr * grad) is
simple enough to hand-verify with no risk of the reference itself being
subtly wrong.

Run this directly: `python nodes/smoke_tests/smoke_test_step_pipeline_correctness.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.core import ExecutionContext
from nodes.components.diffusion import (DiffusionProcess, DiscreteLinearNoiseSchedule,
                                         EpsParameterization, KarrasInputScaler)
from nodes.model.handle import TrainableModel
from nodes.optimizer.handle import OptimizerHandle
from nodes.train.loss import UniformLossWeighting
from nodes.train.schedule import ConstantLRSchedule
from nodes.train.supervised import SupervisedLoRATrainerNode

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


class _RecordingModel(TrainableModel):
    """pred = p, broadcast across the batch -- a simple, hand-differentiable
    function of the one trainable parameter, decoupled from xc/t/ctx_emb/y's
    actual values (recorded, not used to compute pred) so this test can
    check "was the diffusion/encoding math threaded through correctly"
    (via what got recorded) separately from "was the loss/gradient/
    optimizer-step math correct" (via pred's simple, known dependence on p)."""

    def __init__(self, p_init: torch.Tensor):
        self.p = torch.nn.Parameter(p_init.clone())
        self.received = None

    def forward(self, xc, t, ctx_emb, y):
        self.received = (xc, t, ctx_emb, y)
        return self.p.unsqueeze(0).expand(xc.shape[0], *self.p.shape)

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
        pass

    def reload(self, device=None):
        pass

    def release(self):
        pass


class _SGDOptimizerHandle(OptimizerHandle):
    """p -= lr * grad. Nothing fancier -- see this module's docstring."""

    def __init__(self, params, lr: float):
        self.params = params
        self._lr = lr

    @property
    def lr(self):
        return self._lr

    def update_lr(self, new_lr):
        self._lr = new_lr

    def step(self, n_steps=1):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self._lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None

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
        return 0


class _FakeTextEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, prompt, batch_size, height, width):
        self.calls += 1
        torch.manual_seed(123)  # deterministic, independent of global RNG state
        return torch.randn(batch_size, 1, 4), torch.randn(batch_size, 4)

    def unload(self):
        pass

    def footprint_bytes(self):
        return 0

    def offload(self):
        pass

    def reload(self, device=None):
        pass

    def release(self):
        pass


class _FixedBatchSource:
    """Repeats one exact, pre-built batch forever."""

    def __init__(self, x_t, target, t, prompt):
        self._batch = {"x_t": x_t, "target": target, "t": t, "prompt": prompt}

    def __iter__(self):
        while True:
            yield self._batch


def check_step_math_matches_independent_reference():
    print("\n=== One real training step matches an independent hand-computed reference ===")
    torch.manual_seed(42)
    shape = (4, 4, 4)  # (C, H, W) for a batch element -- x_t/target's per-sample shape
    batch_size = 2
    lr = 1e-2

    x_t = torch.randn(batch_size, *shape)
    target = torch.randn(batch_size, *shape)
    t = torch.tensor([300, 700])
    p_init = torch.randn(*shape)

    process = DiffusionProcess(DiscreteLinearNoiseSchedule(), EpsParameterization(),
                                KarrasInputScaler())
    loss_weighting = UniformLossWeighting()

    # --- Independent reference: same real DiffusionProcess/LossWeighting
    # objects, called directly here, not through any StepPhase. ---
    p_ref = p_init.clone().requires_grad_(True)
    _, sigma_ref = process.schedule.alpha_sigma(t)
    xc_ref = process.input_transform.scale_input(x_t, sigma_ref)
    pred_ref = p_ref.unsqueeze(0).expand(batch_size, *shape)
    per_sample_ref = (pred_ref.float() - target.float()).pow(2).view(batch_size, -1).mean(dim=1)
    weight_ref = loss_weighting.weight(float(sigma_ref.float().mean().item()))
    loss_ref = per_sample_ref.mean() * weight_ref
    loss_ref.backward()
    with torch.no_grad():
        p_ref_after = p_ref - lr * p_ref.grad

    # --- Real pipeline path, through the actual public build() entry point. ---
    model = _RecordingModel(p_init)
    optimizer = _SGDOptimizerHandle([model.p], lr=lr)
    text_encoder = _FakeTextEncoder()
    node = SupervisedLoRATrainerNode()
    node.context = ExecutionContext()
    node.build(
        model=model, optimizer=optimizer, text_encoder=text_encoder,
        batches=_FixedBatchSource(x_t, target, t, "a test prompt"), steps=1,
        lr_schedule=ConstantLRSchedule(lr=lr), loss_weighting=loss_weighting,
        diffusion_process=process,
    )

    record(torch.allclose(model.p.detach(), p_ref_after, atol=1e-6),
           "final parameter value matches the hand-computed reference exactly",
           detail=f"got {model.p.detach()}, expected {p_ref_after}")

    recv_xc, recv_t, recv_ctx_emb, recv_y = model.received
    record(torch.allclose(recv_xc, xc_ref),
           "xc actually passed to forward() matches the independently-computed value")
    record(torch.equal(recv_t, t), "t passed to forward() is unchanged (just moved/cast)")
    record(recv_ctx_emb.shape == (batch_size, 1, 4) and recv_y.shape == (batch_size, 4),
           "ctx_emb/y shapes match what the text encoder actually returned")
    record(text_encoder.calls == 1, "text encoder called exactly once for the one step")


def check_epoch_wrapping_reuses_dataset():
    """A dataset with fewer batches than `steps` needs must be trained
    over multiple passes (FetchBatchPhase wrapping back to the start),
    not treated as an error or silently truncated."""
    print("\n=== A 2-batch dataset trained for 5 steps wraps around, doesn't crash or truncate ===")
    shape = (2, 2, 2)
    batches_seen = []

    class _TwoBatchSource:
        def __init__(self):
            self._batches = [
                {"x_t": torch.full((1, *shape), 1.0), "target": torch.zeros(1, *shape),
                 "t": torch.tensor([500]), "prompt": "a"},
                {"x_t": torch.full((1, *shape), 2.0), "target": torch.zeros(1, *shape),
                 "t": torch.tensor([500]), "prompt": "b"},
            ]

        def __iter__(self):
            for b in self._batches:
                yield b

    class _RecordingModel2(TrainableModel):
        def __init__(self):
            self.p = torch.nn.Parameter(torch.zeros(*shape))

        def forward(self, xc, t, ctx_emb, y):
            batches_seen.append(xc[0, 0, 0, 0].item())
            return xc + self.p.sum() * 0

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
            return 0

        def offload(self):
            pass

        def reload(self, device=None):
            pass

        def release(self):
            pass

    node = SupervisedLoRATrainerNode()
    node.context = ExecutionContext()
    node.build(
        model=_RecordingModel2(), optimizer=_SGDOptimizerHandle([], lr=0.0),
        text_encoder=_FakeTextEncoder(), batches=_TwoBatchSource(), steps=5,
        lr_schedule=ConstantLRSchedule(lr=0.0), loss_weighting=UniformLossWeighting(),
    )
    record(len(batches_seen) == 5, "ran exactly 5 steps despite only 2 distinct batches",
           detail=str(batches_seen))
    # xc's magnitude tracks x_t's magnitude (scale_input divides by a
    # sigma-derived constant, doesn't reorder) -- so alternating small/
    # large values confirms the two distinct batches, not one repeated.
    distinct = len(set(round(v, 4) for v in batches_seen))
    record(distinct == 2, "exactly two distinct batch values appeared, alternating",
           detail=str(batches_seen))


def main():
    check_step_math_matches_independent_reference()
    check_epoch_wrapping_reuses_dataset()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
