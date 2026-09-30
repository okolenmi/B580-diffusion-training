"""Checks nodes/train/t_probe.py (TProbe) and its wiring into
ManagedLoRATrainerNode (ProbePhase + probe_* Ports).

What is verified, and how -- everything runs on CPU with toy models whose
correct answers are known in closed form, so a failure means the probe is
wrong, not that a number "looks odd":

  * probe t grid: strictly inside each T_BUCKETS third, t=0 never used;
  * collect(): x0 is recovered exactly from (x_t, target, sigma) for both
    eps and v-prediction batches;
  * evaluate(): deterministic across calls (fixed noise), rel == 1 and
    drift == 0 for a LoRA that is a no-op, rel != 1 / drift > 0 once it is
    not, the frozen-base reference is computed once and reused, and both
    core.lora's gate and the model's train/eval flag are restored;
  * alignment(): with a single scalar parameter and per-bucket sign
    constructed to conflict, cosines come out as -1 and the combined
    direction's alignment flags the losing bucket; orthogonal per-bucket
    parameters give cosine 0; .grad is left clear; split-half self-cosine
    is reported;
  * trainer wiring: probe_every_n_steps=0 adds no phase and no forward call
    (bit-for-bit the previous pipeline), >0 runs the probe and merges
    probe_* into the monitor report, and probe_grad_alignment with a fused
    optimizer is a build-time ValueError.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from types import SimpleNamespace

import torch

from nodes.components.diffusion import (DiffusionProcess, DiscreteLinearNoiseSchedule,
                                         EpsParameterization, KarrasInputScaler,
                                         VPredParameterization)
from nodes.core import ExecutionContext
from nodes.train.loss import UniformLossWeighting
from nodes.train.managed import ManagedLoRATrainerNode, ProbePhase
from nodes.train.schedule import ConstantLRSchedule
from nodes.train.t_probe import TProbe, _bucket_points, format_probe_line

PROCESS = DiffusionProcess(DiscreteLinearNoiseSchedule(), EpsParameterization(), KarrasInputScaler())


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


# ---- toy models --------------------------------------------------------------------

class _GatedToyModel:
    """pred = base(xc) + gate * delta * xc. Reads core.lora._current_gate the
    way LoRALinear does, so gate=0 is *exactly* the base output."""

    def __init__(self, delta: float):
        self.base_scale = 0.5
        self.delta = torch.nn.Parameter(torch.tensor(float(delta)))
        self.training = True
        self.forwards = 0

    def forward(self, xc, t, ctx_emb, y):
        import core.lora as lora
        self.forwards += 1
        out = self.base_scale * xc.float()
        gate = lora._current_gate
        d = self.delta * xc.float()
        if gate is not None:
            d = d * gate.to(d.dtype).view(-1, 1, 1, 1)
        return out + d

    def trainable_parameters(self):
        return [self.delta]

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


class _BucketSignModel:
    """One scalar parameter w. pred = o + w*c(t) with c = +1 (low t),
    0 (mid), -2 (high). Target is eps (mean ~ 0), so d loss/dw ~ 2*o*c:
    +2 for low, -4 for high, 0 for mid -- a constructed conflict."""

    def __init__(self, offset: float = 1.0):
        self.w = torch.nn.Parameter(torch.tensor(0.0))
        self.offset = offset
        self.training = True

    def forward(self, xc, t, ctx_emb, y):
        c = torch.where(t < 333, torch.tensor(1.0),
                        torch.where(t >= 666, torch.tensor(-2.0), torch.tensor(0.0)))
        return self.offset + self.w * c.view(-1, 1, 1, 1) * torch.ones_like(xc, dtype=torch.float32)

    def trainable_parameters(self):
        return [self.w]

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


class _OrthogonalModel:
    """Three parameters, one per bucket; each bucket's loss touches only its own."""

    def __init__(self):
        self.p = torch.nn.Parameter(torch.zeros(3))
        self.training = True

    def forward(self, xc, t, ctx_emb, y):
        idx = (t >= 333).long() + (t >= 666).long()
        return self.p[idx].view(-1, 1, 1, 1) * torch.ones_like(xc, dtype=torch.float32) + 0.3

    def trainable_parameters(self):
        return [self.p]

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


def _make_probe(n_items=2, points=2, grad_alignment=False, process=PROCESS, seed_data=0):
    probe = TProbe(n_items=n_items, points_per_bucket=points, grad_alignment=grad_alignment)
    g = torch.Generator().manual_seed(seed_data)
    for i in range(n_items):
        x0 = torch.randn(1, 4, 8, 8, generator=g)
        t = 500
        alpha, sigma = process.schedule.alpha_sigma(torch.tensor([t]))
        sig4 = sigma.view(-1, 1, 1, 1)
        eps = torch.randn(1, 4, 8, 8, generator=g)
        x_t = x0 + sig4 * eps
        probe.collect(process, x_t, eps, torch.tensor([t]), sigma,
                      torch.zeros(1, 3, 4), torch.zeros(1, 4))
    return probe


# ---- checks ------------------------------------------------------------------------

def check_grid():
    print("[grid: strictly inside each third, never t=0]")
    for n in (1, 2, 3, 5):
        pts = _bucket_points(n)
        check(len(pts) == 3 * n, pts)
        for name, lo, hi in __import__("nodes.train.loss", fromlist=["T_BUCKETS"]).T_BUCKETS:
            ts = [t for b, t in pts if b == name]
            check(len(ts) == n and all(lo < t < hi for t in ts), (name, ts))
            check(ts == sorted(set(ts)), f"grid t's must be distinct/ascending: {ts}")
    check(all(1 <= t <= 999 for _, t in _bucket_points(50)), "t range")
    print("    PASS")


def check_collect_recovers_x0_eps_and_vpred():
    print("[collect: x0 recovered exactly from (x_t, target, sigma), eps and v]")
    for param in (EpsParameterization(), VPredParameterization()):
        proc = DiffusionProcess(DiscreteLinearNoiseSchedule(), param, KarrasInputScaler())
        for t in (5, 300, 700, 990):
            x0 = torch.randn(2, 4, 8, 8)
            eps = torch.randn(2, 4, 8, 8)
            tt = torch.tensor([t, t])
            alpha, sigma = proc.schedule.alpha_sigma(tt)
            sig4 = sigma.view(-1, 1, 1, 1)
            x_t = x0 + sig4 * eps
            target = eps if isinstance(param, EpsParameterization) else \
                EpsParameterization().convert_to(eps, x_t, alpha, sig4, param)
            probe = TProbe(n_items=1)
            probe.collect(proc, x_t, target, tt, sigma, torch.zeros(2, 3, 4), torch.zeros(2, 4))
            got = probe._items[0].x0
            check(got.shape == (1, 4, 8, 8), got.shape)
            err = (got - x0[:1]).abs().max().item()
            check(err < 2e-4 * max(1.0, float(sigma[0])), (type(param).__name__, t, err))
    # item cap
    probe = TProbe(n_items=1)
    probe.collect(PROCESS, torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8), torch.tensor([10]),
                  torch.tensor([0.1]), torch.zeros(1, 3, 4), torch.zeros(1, 4))
    probe.collect(PROCESS, torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8), torch.tensor([10]),
                  torch.tensor([0.1]), torch.zeros(1, 3, 4), torch.zeros(1, 4))
    check(len(probe._items) == 1 and probe.ready() and not probe.wants_items(), "n_items cap")
    print("    PASS")


def check_evaluate_noop_lora_is_rel_one_drift_zero():
    print("[evaluate: no-op LoRA -> rel == 1, drift == 0; deterministic; base cached]")
    import core.lora as lora
    probe = _make_probe(n_items=2, points=2)
    model = _GatedToyModel(delta=0.0)
    r1, detail = probe.evaluate(model, PROCESS)
    for s in ("t_low", "t_mid", "t_high"):
        check(_close(r1[f"probe_rel_{s}"], 1.0, 1e-6), r1)
        check(r1[f"probe_drift_{s}"] == 0.0, r1)
    check(_close(r1["probe_worst_rel"], 1.0, 1e-6), r1)
    check(len(detail) == 6, detail)
    fwd_first = model.forwards
    check(fwd_first == 2 * 6 * 2, f"base+lora forwards on first call: {fwd_first}")
    r2, _ = probe.evaluate(model, PROCESS)
    check(model.forwards - fwd_first == 2 * 6, "second call must reuse the cached base reference")
    check(r1 == r2, "fixed noise -> identical probe on an unchanged model")
    print("    PASS")


def check_evaluate_sees_a_real_change_and_restores_state():
    print("[evaluate: LoRA that changes the output -> rel != 1, drift > 0; gate & train flag restored]")
    import core.lora as lora
    probe = _make_probe(n_items=2, points=2)
    model = _GatedToyModel(delta=0.0)
    base_report, _ = probe.evaluate(model, PROCESS)
    model.delta.data.fill_(0.8)
    sentinel = torch.tensor([0.25])
    lora.set_lora_gate(sentinel)
    model.train()
    try:
        r, _ = probe.evaluate(model, PROCESS)
        check(lora._current_gate is sentinel, "training-time gate must be restored")
        check(model.training is True, "train flag must be restored")
        for s in ("t_low", "t_mid", "t_high"):
            check(r[f"probe_drift_{s}"] > 0.05, r)
            check(abs(r[f"probe_rel_{s}"] - 1.0) > 1e-3, r)
            # base reference is what the LoRA-off report measured, unchanged
            check(_close(r[f"probe_t_{s[2:]}"] / r[f"probe_rel_{s}"],
                         base_report[f"probe_t_{s[2:]}"], 1e-5), (r, base_report))
        # the probe ignores the training gate: same numbers with the sentinel gate off
        lora.set_lora_gate(None)
        r_nogate, _ = probe.evaluate(model, PROCESS)
        check(r == r_nogate, "probe must be independent of the training-time gate")
    finally:
        lora.set_lora_gate(None)
    print("    PASS")


class _NoTrainingAttrModel:
    """Mirrors ComfyUNetTrainableModel: train()/eval() exist, but there is NO
    `.training` attribute to query. Records the last mode it was put in."""

    def __init__(self):
        self.delta = torch.nn.Parameter(torch.tensor(0.3))
        self.mode = "train"

    def forward(self, xc, t, ctx_emb, y):
        return 0.5 * xc.float() + self.delta * xc.float()

    def trainable_parameters(self):
        return [self.delta]

    def train(self):
        self.mode = "train"
        return self

    def eval(self):
        self.mode = "eval"
        return self


def check_model_left_in_train_mode_even_without_training_attr():
    print("[regression: model without a .training attribute is left in train mode "
          "after evaluate() and alignment()]")
    probe = _make_probe(n_items=2, points=1, grad_alignment=True)
    model = _NoTrainingAttrModel()
    check(not hasattr(model, "training"), "test model must mimic the real wrapper")
    probe.evaluate(model, PROCESS)
    check(model.mode == "train", f"evaluate() left the model in {model.mode!r}")
    model.eval()
    probe.alignment(model, PROCESS, model.trainable_parameters())
    check(model.mode == "train", f"alignment() left the model in {model.mode!r}")
    print("    PASS")


def check_not_ready_returns_empty():
    print("[not enough probe items yet -> empty result, no forwards]")
    probe = TProbe(n_items=2)
    model = _GatedToyModel(0.0)
    check(probe.evaluate(model, PROCESS) == ({}, []), "evaluate before ready")
    check(probe.alignment(model, PROCESS, model.trainable_parameters()) == {}, "alignment before ready")
    check(model.forwards == 0, "no forwards")
    print("    PASS")


def check_alignment_detects_constructed_conflict():
    print("[alignment: constructed low(+2)/high(-4) conflict -> cos -1; combined step "
          "flagged as hurting low]")
    probe = _make_probe(n_items=2, points=2, grad_alignment=True)
    model = _BucketSignModel(offset=1.0)
    r = probe.alignment(model, PROCESS, model.trainable_parameters())
    # 2*o*|c| up to the sample mean of the (8x8x4) fixed noise, ~ +-0.06 -> loose tolerance
    check(_close(r["gc_norm_t_low"], 2.0, 0.4), r)
    check(_close(r["gc_norm_t_high"], 4.0, 0.8), r)
    check("gc_norm_t_mid" in r and r["gc_norm_t_mid"] < 1e-6, r)       # c = 0 -> no gradient
    check("gc_cos_low_mid" not in r and "gc_cos_mid_high" not in r, "zero-gradient bucket excluded")
    check(_close(r["gc_cos_low_high"], -1.0, 1e-3), r)
    # g_total = +2 - 4 = -2 -> low is anti-aligned (-1), high is aligned (+1)
    check(_close(r["gc_align_t_low"], -1.0, 1e-3), r)
    check(_close(r["gc_align_t_high"], 1.0, 1e-3), r)
    check(r["gc_share_t_high"] > r["gc_share_t_low"], "high dominates by norm here")
    check(model.w.grad is None, ".grad must be left clear")
    check(_close(r["gc_self_t_low"], 1.0, 1e-3) and _close(r["gc_self_t_high"], 1.0, 1e-3), r)
    print("    PASS")


def check_alignment_orthogonal_buckets_and_self_cosine():
    print("[alignment: bucket-private parameters -> cross cosine 0; self-cosine reported]")
    probe = _make_probe(n_items=2, points=2, grad_alignment=True)
    model = _OrthogonalModel()
    r = probe.alignment(model, PROCESS, model.trainable_parameters())
    for k in ("gc_cos_low_mid", "gc_cos_low_high", "gc_cos_mid_high"):
        check(abs(r[k]) < 1e-6, (k, r))
    for s in ("t_low", "t_mid", "t_high"):
        check(_close(r[f"gc_self_{s}"], 1.0, 1e-3), r)   # one param per bucket -> halves agree
        check(0.0 < r[f"gc_align_{s}"] <= 1.0 + 1e-6, r)
    # identity: sum_b ||g_b|| * cos(g_b, g_tot) == ||g_tot||; orthogonal -> ||g_tot||^2 = sum ||g_b||^2
    n = [r[f"gc_norm_{s}"] for s in ("t_low", "t_mid", "t_high")]
    lhs = sum(ni * r[f"gc_align_{s}"] for ni, s in zip(n, ("t_low", "t_mid", "t_high")))
    check(_close(lhs, sum(x * x for x in n) ** 0.5, 1e-4), (lhs, n))
    check(model.p.grad is None, ".grad cleared")
    print("    PASS")


def check_format_line_smoke():
    print("[format_probe_line: renders without error on a full report]")
    probe = _make_probe(n_items=2, points=2, grad_alignment=True)
    model = _GatedToyModel(0.3)
    rep, detail = probe.evaluate(model, PROCESS)
    rep.update(probe.alignment(model, PROCESS, model.trainable_parameters()))
    line = format_probe_line(7, rep, detail)
    check("[probe step 7]" in line and "worst_rel" in line and "grad-align" in line, line)
    print("    PASS")


# ---- trainer wiring ----------------------------------------------------------------

class _WiredModel(_GatedToyModel):
    """_GatedToyModel + the TrainableModel/DeviceResident surface build() touches."""

    def __init__(self, delta=0.0):
        super().__init__(delta)
        self.p = self.delta

    def to(self, device=None, **kwargs):
        return self

    def trained_state_dict(self):
        return {}

    def footprint_bytes(self):
        return 8

    def offload(self):
        pass

    def reload(self, device=None):
        pass

    def release(self):
        pass


class _Batches:
    """Finite epoch of 4 batches (batch 2, 8x8 latents), fresh noise each pass."""

    def __iter__(self):
        for i in range(4):
            x0 = torch.randn(2, 4, 8, 8)
            eps = torch.randn(2, 4, 8, 8)
            t = torch.tensor([200 + 100 * i, 250 + 100 * i])
            _, sigma = PROCESS.schedule.alpha_sigma(t)
            yield {"x_t": x0 + sigma.view(-1, 1, 1, 1) * eps, "target": eps, "t": t, "prompt": "p"}


class _Encoder:
    def encode(self, prompt, batch_size, height, width):
        return torch.zeros(batch_size, 3, 4), torch.zeros(batch_size, 4)

    def footprint_bytes(self):
        return 0

    def offload(self):
        pass

    def reload(self, device=None):
        pass

    def release(self):
        pass


def _optimizer(fused: bool):
    from nodes.optimizer.handle import FusedOptimizerHandle, OptimizerHandle
    base = FusedOptimizerHandle if fused else OptimizerHandle

    class _Opt(base):
        lr = 1e-4

        def update_lr(self, new_lr): pass
        def step(self, n_steps=1): pass
        def zero_grad(self): pass
        def begin_step(self, sub_steps=1): pass
        def prepare_next_pass(self): pass
        def offload_states_to_cpu(self): pass
        def reload_states_to_device(self, device=None): pass
        def decay_states(self, factor): pass
        def reset_states(self): pass
        def free_states(self): pass
        def footprint_bytes(self): return 8

    return _Opt()


def _resource_control():
    from nodes.memory.control_handle import ResourceControlHandle

    class _RC(ResourceControlHandle):
        def register(self, name, resident, offloadable=False): pass
        def before_step(self, step): pass
        def ensure_loaded(self, name): pass
        def release(self, name): pass
        def usable_budget_mb(self): return None

    return _RC()


class _Monitor:
    def __init__(self):
        self.reports = []

    def report(self, r):
        self.reports.append(r)


def _build(model, steps=4, fused=False, monitor=None, **extra):
    node = ManagedLoRATrainerNode()
    node.context = ExecutionContext()
    return node.build(
        trainer=SimpleNamespace(unet=model, clip=_Encoder()), batches=_Batches(),
        optimizer=_optimizer(fused), lr_schedule=ConstantLRSchedule(lr=1e-4),
        loss_weighting=UniformLossWeighting(), steps=steps,
        resource_control=_resource_control(), monitor=monitor, **extra)


def check_ports_default_off():
    print("[ports: probe defaults are off, and off adds nothing]")
    ins = ManagedLoRATrainerNode.INPUTS
    for name in ("probe_every_n_steps", "probe_items", "probe_points_per_bucket",
                 "probe_grad_alignment"):
        check(name in ins, name)
    check(ins["probe_every_n_steps"].default == 0 and ins["probe_grad_alignment"].default is False,
          "probe must default off")
    plain = _WiredModel()
    _build(plain, steps=3)
    check(plain.forwards == 3, f"probe off -> exactly one forward per step, got {plain.forwards}")
    print("    PASS")


def check_probe_runs_in_trainer_and_reaches_the_monitor():
    print("[trainer: probe_every_n_steps>0 runs, prints, and merges probe_* into the step report]")
    model = _WiredModel(delta=0.2)
    mon = _Monitor()
    _build(model, steps=4, monitor=mon, probe_every_n_steps=2, probe_items=2,
           probe_points_per_bucket=1)
    with_probe = [r for r in mon.reports if "step" in r and any(k.startswith("probe_") for k in r)]
    check(with_probe, mon.reports)
    for r in with_probe:
        check({"probe_rel_t_low", "probe_rel_t_mid", "probe_rel_t_high", "probe_worst_rel"} <= set(r), r)
    plain_steps = [r["step"] for r in mon.reports if "step" in r
                   and not any(k.startswith("probe_") for k in r)]
    check(plain_steps, "steps between probes must carry no probe_* keys (absent, not carried)")
    # first probe fires as soon as 2 items exist (step index 1), then every 2nd step
    check(with_probe[0]["step"] == 1, [r["step"] for r in with_probe])
    print("    PASS")


def check_alignment_with_fused_optimizer_is_rejected():
    print("[trainer: probe_grad_alignment + fused optimizer -> ValueError at build]")
    try:
        _build(_WiredModel(), fused=True, probe_every_n_steps=2, probe_grad_alignment=True)
    except ValueError as e:
        check("fused" in str(e), str(e))
    else:
        raise AssertionError("expected ValueError")
    # forward-only probe is fine with a fused optimizer
    _build(_WiredModel(), fused=True, probe_every_n_steps=2)
    print("    PASS")


def main():
    check_grid()
    check_collect_recovers_x0_eps_and_vpred()
    check_evaluate_noop_lora_is_rel_one_drift_zero()
    check_evaluate_sees_a_real_change_and_restores_state()
    check_model_left_in_train_mode_even_without_training_attr()
    check_not_ready_returns_empty()
    check_alignment_detects_constructed_conflict()
    check_alignment_orthogonal_buckets_and_self_cosine()
    check_format_line_smoke()
    check_ports_default_off()
    check_probe_runs_in_trainer_and_reaches_the_monitor()
    check_alignment_with_fused_optimizer_is_rejected()
    print("ALL PASS")


if __name__ == "__main__":
    main()
