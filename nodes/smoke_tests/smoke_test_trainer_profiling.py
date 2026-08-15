"""Targeted test for SupervisedLoRATrainerNode's profile=True instrumentation
(nodes/train/step_pipeline.py's TimedPhase, wrapped around every phase
except MonitoringPhase) -- real risk here: several new per-phase timing
keys, a step_total_ms that has to equal their sum, and a
profile=False path that must add zero overhead and zero extra keys.
Fakes model/optimizer/text_encoder (no real ComfyUI needed) since only
the pipeline's control flow and the timing dict's shape are being
checked, not real UNet numerics -- same approach as
smoke_test_trainer_cancellation.py, reusing its fakes' shape rather than
constructing internals directly (there's no equivalent of the old
_StepContext to construct anymore -- see nodes/train/step_pipeline.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.model.handle import TrainableModel
from nodes.optimizer.handle import OptimizerHandle
from nodes.train.loss import UniformLossWeighting
from nodes.train.schedule import ConstantLRSchedule
from nodes.monitor.handle import MonitorHandle
from nodes.train.supervised import SupervisedLoRATrainerNode

EXPECTED_PHASE_LABELS = (
    "fetch_batch", "prepare_diffusion_inputs", "encode_conditioning",
    "optimizer_begin_step", "forward", "loss", "backward", "optimizer_step",
)


class _FakeModel(TrainableModel):
    def __init__(self):
        self.p = torch.nn.Parameter(torch.zeros(4, 4))

    def forward(self, xc, t, ctx_emb, y):
        return xc + self.p.sum() * 0  # depends on p so backward has something to do

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


class _FakeOptimizer(OptimizerHandle):
    def __init__(self):
        self.stepped = 0

    @property
    def lr(self):
        return 1e-4

    def update_lr(self, new_lr):
        pass

    def step(self, n_steps=1):
        self.stepped += 1

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

    def footprint_bytes(self):
        return 0


class _FakeTextEncoder:
    def encode(self, prompt, batch_size, height, width):
        return torch.zeros(batch_size, 1, 4), torch.zeros(batch_size, 4)

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


class _RecordingMonitor(MonitorHandle):
    def __init__(self):
        self.reports = []

    def report(self, data):
        self.reports.append(data)


class _OneBatch:
    """Repeats the same single batch -- steps=1 in every check below, so
    only ever fetched once regardless."""

    def __iter__(self):
        while True:
            yield {
                "x_t": torch.randn(2, 4, 4, 4),
                "target": torch.randn(2, 4, 4, 4),
                "t": torch.tensor([500, 500]),
                "prompt": "x",
            }


def _run(profile: bool, monitor=None, steps: int = 1) -> dict:
    node = SupervisedLoRATrainerNode()
    from nodes.core import ExecutionContext
    node.context = ExecutionContext()
    optimizer = _FakeOptimizer()
    node.build(
        model=_FakeModel(), optimizer=optimizer, text_encoder=_FakeTextEncoder(),
        batches=_OneBatch(), steps=steps,
        lr_schedule=ConstantLRSchedule(lr=1e-4), loss_weighting=UniformLossWeighting(),
        profile=profile, monitor=monitor,
    )
    return optimizer


def check_contracts():
    print("[contracts]")
    assert "profile" in SupervisedLoRATrainerNode.INPUTS
    assert SupervisedLoRATrainerNode.INPUTS["profile"].default is False
    print("    PASS")


def check_profile_off_unchanged_behavior():
    print("[profile=False: runs, no timing keys, monitor gets a plain report]")
    monitor = _RecordingMonitor()
    optimizer = _run(profile=False, monitor=monitor)
    assert len(monitor.reports) == 1
    report = monitor.reports[0]
    for label in EXPECTED_PHASE_LABELS:
        assert f"{label}_ms" not in report, f"{label}_ms should not be present when profile=False"
    assert "step_total_ms" not in report
    assert "tracked_footprint_mb" not in report
    assert "resident_model_mb" not in report
    assert optimizer.stepped == 1
    print("    PASS")


def check_profile_on_produces_full_timing_breakdown():
    print("[profile=True: full per-phase timing breakdown, all present, non-negative, sums correctly]")
    monitor = _RecordingMonitor()
    _run(profile=True, monitor=monitor)
    assert len(monitor.reports) == 1
    report = monitor.reports[0]
    for label in EXPECTED_PHASE_LABELS:
        key = f"{label}_ms"
        assert key in report, f"missing {key}"
        assert report[key] >= 0, f"{key} was negative: {report[key]}"
    expected_total = sum(report[f"{label}_ms"] for label in EXPECTED_PHASE_LABELS)
    assert abs(report["step_total_ms"] - expected_total) < 1e-6, (
        f"step_total_ms ({report['step_total_ms']}) != sum of phase times ({expected_total})")
    print(f"    PASS: {report}")


def check_profile_on_without_monitor_does_not_crash():
    print("[profile=True with no monitor wired -- still prints, doesn't need one]")
    _run(profile=True, monitor=None)
    print("    PASS")


def check_profile_on_reports_resident_footprint_breakdown():
    print("[profile=True: resident_<name>_mb breakdown (ResourceProfile, backlog item 1) "
          "present and sums to tracked_footprint_mb]")
    monitor = _RecordingMonitor()
    _run(profile=True, monitor=monitor)
    report = monitor.reports[0]
    for key in ("resident_model_mb", "resident_optimizer_mb", "resident_text_encoder_mb"):
        assert key in report, f"missing {key}"
        assert report[key] >= 0, f"{key} was negative: {report[key]}"
    resident_sum = (report["resident_model_mb"] + report["resident_optimizer_mb"]
                     + report["resident_text_encoder_mb"])
    assert abs(report["tracked_footprint_mb"] - resident_sum) < 1e-9, (
        f"tracked_footprint_mb ({report['tracked_footprint_mb']}) != sum of "
        f"resident_*_mb ({resident_sum})")
    # _FakeModel's one 4x4 float32 parameter: 16 * 4 bytes; _FakeOptimizer and
    # _FakeTextEncoder both hardcode footprint_bytes() -> 0 (see their classes
    # above) -- real, known values, not just "some positive number".
    expected_model_mb = (4 * 4 * 4) / (1024 ** 2)
    assert abs(report["resident_model_mb"] - expected_model_mb) < 1e-9, (
        f"resident_model_mb ({report['resident_model_mb']}) != expected ({expected_model_mb})")
    assert report["resident_optimizer_mb"] == 0.0
    assert report["resident_text_encoder_mb"] == 0.0
    print(f"    PASS: resident_model_mb={report['resident_model_mb']:.6f} "
          f"tracked_footprint_mb={report['tracked_footprint_mb']:.6f}")


def main():
    check_contracts()
    check_profile_off_unchanged_behavior()
    check_profile_on_produces_full_timing_breakdown()
    check_profile_on_without_monitor_does_not_crash()
    check_profile_on_reports_resident_footprint_breakdown()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
