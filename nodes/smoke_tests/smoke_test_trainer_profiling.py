"""Targeted test for the profile=True instrumentation added to
nodes/train/supervised.py's _run_step -- real risk of a bug here (several
new conditional branches, t0..t4 locals only set when profiling), not
covered by the existing contract-only train test. Fakes model/optimizer/
text_encoder (no real ComfyUI needed) since only _run_step's control flow
and the timing dict's shape are being checked, not real UNet numerics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.train.supervised import SupervisedLoRATrainerNode, _StepContext
from nodes.train.loss import UniformLossWeighting
from nodes.train.schedule import ConstantLRSchedule
from nodes.monitor.handle import MonitorHandle
from nodes.components.device import DeviceContext
from nodes.components.diffusion import (DiffusionProcess, DiscreteLinearNoiseSchedule,
                                         EpsParameterization, KarrasInputScaler)


class _FakeModel:
    def __init__(self):
        self.p = torch.nn.Parameter(torch.zeros(4, 4))

    def forward(self, xc, t, ctx_emb, y):
        return xc + self.p.sum() * 0  # depends on p so backward has something to do

    def trainable_parameters(self):
        return [self.p]


class _FakeOptimizer:
    def __init__(self):
        self.stepped = 0

    def update_lr(self, lr):
        pass

    def zero_grad(self):
        self.p_grad_cleared = True

    def step(self, n_steps=1):
        self.stepped += 1


class _FakeTextEncoder:
    def encode(self, prompt, batch_size, height, width):
        return torch.zeros(batch_size, 1, 4), torch.zeros(batch_size, 4)


class _RecordingMonitor(MonitorHandle):
    def __init__(self):
        self.reports = []

    def report(self, data):
        self.reports.append(data)


def _make_batch():
    return {
        "x_t": torch.randn(2, 4, 4, 4),
        "target": torch.randn(2, 4, 4, 4),
        "t": torch.tensor([500, 500]),
        "prompt": "x",
    }


def _make_ctx(profile: bool, monitor=None) -> _StepContext:
    device = torch.device("cpu")
    return _StepContext(
        model=_FakeModel(), optimizer=_FakeOptimizer(), text_encoder=_FakeTextEncoder(),
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        diffusion_process=DiffusionProcess(
            DiscreteLinearNoiseSchedule(), EpsParameterization(), KarrasInputScaler()),
        device_ctx=DeviceContext.for_device(device),
        is_fused=False, device=device, total_steps=10,
        monitor=monitor, profile=profile,
    )


def check_contracts():
    print("[contracts]")
    assert "profile" in SupervisedLoRATrainerNode.INPUTS
    assert SupervisedLoRATrainerNode.INPUTS["profile"].default is False
    print("    PASS")


def check_profile_off_unchanged_behavior():
    print("[profile=False: runs, no timing dict, monitor gets a plain report]")
    monitor = _RecordingMonitor()
    ctx = _make_ctx(profile=False, monitor=monitor)
    SupervisedLoRATrainerNode._run_step(ctx, _make_batch(), step=0, wait_ms=5.0)
    assert len(monitor.reports) == 1
    assert "encode_ms" not in monitor.reports[0]
    assert ctx.optimizer.stepped == 1
    print("    PASS")


def check_profile_on_produces_full_timing_breakdown():
    print("[profile=True: full timing breakdown, all phases present and non-negative]")
    monitor = _RecordingMonitor()
    ctx = _make_ctx(profile=True, monitor=monitor)
    SupervisedLoRATrainerNode._run_step(ctx, _make_batch(), step=0, wait_ms=3.0)
    assert len(monitor.reports) == 1
    report = monitor.reports[0]
    for key in ("data_wait_ms", "encode_ms", "forward_ms", "backward_ms", "optim_ms", "step_total_ms"):
        assert key in report, f"missing {key}"
        assert report[key] >= 0, f"{key} was negative: {report[key]}"
    assert report["data_wait_ms"] == 3.0
    assert report["step_total_ms"] >= report["encode_ms"] + report["forward_ms"]
    print(f"    PASS: {report}")


def check_profile_on_without_monitor_does_not_crash():
    print("[profile=True with no monitor wired -- still prints, doesn't need one]")
    ctx = _make_ctx(profile=True, monitor=None)
    SupervisedLoRATrainerNode._run_step(ctx, _make_batch(), step=0, wait_ms=0.0)
    print("    PASS")


def main():
    check_contracts()
    check_profile_off_unchanged_behavior()
    check_profile_on_produces_full_timing_breakdown()
    check_profile_on_without_monitor_does_not_crash()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
