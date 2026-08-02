"""Verifies nodes/train/supervised.py's cooperative-cancellation check
actually works: a cancel_event set before build() starts must produce a
zero-step run (checked before consuming the first batch), and one set
mid-run must stop before finishing all requested steps but keep whatever
was trained. Real risk here (new logic, not covered elsewhere): an
off-by-one in "check before or after running the step" would either burn
one extra step past cancellation or never check at all.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.core import ExecutionContext
from nodes.model.handle import TrainableModel
from nodes.optimizer.handle import OptimizerHandle
from nodes.train.loss import UniformLossWeighting
from nodes.train.schedule import ConstantLRSchedule
from nodes.train.supervised import SupervisedLoRATrainerNode


class _FakeModel(TrainableModel):
    def __init__(self):
        self.p = torch.nn.Parameter(torch.zeros(4, 4))
        self.calls = 0

    def forward(self, xc, t, ctx_emb, y):
        self.calls += 1
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


class _FakeOptimizer(OptimizerHandle):
    @property
    def lr(self):
        return 1e-4

    def update_lr(self, new_lr):
        pass

    def step(self, n_steps=1):
        pass

    def zero_grad(self):
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


class _FakeTextEncoder:
    def encode(self, prompt, batch_size, height, width):
        return torch.zeros(batch_size, 1, 4), torch.zeros(batch_size, 4)

    def unload(self):
        pass


class _FiniteBatches:
    """Cycles the same batch forever -- SupervisedLoRATrainerNode's outer
    while-loop is what actually bounds step count (or cancellation)."""

    def __iter__(self):
        while True:
            yield {
                "x_t": torch.randn(2, 4, 4, 4),
                "target": torch.randn(2, 4, 4, 4),
                "t": torch.tensor([500, 500]),
                "prompt": "x",
            }


def _make_node(cancel_event):
    node = SupervisedLoRATrainerNode()
    node.context = ExecutionContext(cancel_event=cancel_event)
    return node


def check_cancel_before_first_step_trains_zero_steps():
    print("[cancel_event already set: build() returns before any step runs]")
    model = _FakeModel()
    cancel_event = threading.Event()
    cancel_event.set()
    node = _make_node(cancel_event)
    result = node.build(
        model=model, optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
        batches=_FiniteBatches(), steps=100,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
    )
    assert result["model"] is model
    assert model.calls == 0, f"expected zero forward calls, got {model.calls}"
    print("    PASS")


def check_cancel_mid_run_stops_but_keeps_progress():
    print("[cancel_event set mid-run: stops before all steps, keeps what trained so far]")
    model = _FakeModel()
    cancel_event = threading.Event()
    node = _make_node(cancel_event)

    class _CancelAfterNSteps(_FiniteBatches):
        def __iter__(self):
            for batch in super().__iter__():
                yield batch
                if model.calls >= 5:
                    cancel_event.set()

    result = node.build(
        model=model, optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
        batches=_CancelAfterNSteps(), steps=100,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
    )
    assert result["model"] is model
    assert 0 < model.calls < 100, f"expected somewhere between 1 and 99 steps, got {model.calls}"
    print(f"    PASS: stopped after {model.calls} steps, not all 100")


def check_no_cancel_event_runs_normally():
    print("[no cancel_event at all (e.g. a direct call, no server): runs to completion]")
    model = _FakeModel()
    node = SupervisedLoRATrainerNode()
    node.context = ExecutionContext()  # cancel_event=None, the default
    result = node.build(
        model=model, optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
        batches=_FiniteBatches(), steps=7,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
    )
    assert model.calls == 7
    print("    PASS")


def check_empty_cache_every_n_steps_does_not_crash():
    print("[empty_cache_every_n_steps runs cleanly (no XPU here, so it's a no-op path, but must not raise)]")
    model = _FakeModel()
    node = SupervisedLoRATrainerNode()
    node.context = ExecutionContext()
    result = node.build(
        model=model, optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
        batches=_FiniteBatches(), steps=10,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        empty_cache_every_n_steps=3,
    )
    assert model.calls == 10
    print("    PASS")


def main():
    check_cancel_before_first_step_trains_zero_steps()
    check_cancel_mid_run_stops_but_keeps_progress()
    check_no_cancel_event_runs_normally()
    check_empty_cache_every_n_steps_does_not_crash()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
