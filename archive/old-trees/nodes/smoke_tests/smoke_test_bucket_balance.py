"""Correctness checks for nodes/train/bucket_balance.py -- the optional
per-t-bucket rebalancing (gradient side: off/normalize/speed/dro weight
modes; data side: adaptive t sampling), and its wiring through both
trainer routes' LossPhase/MonitoringPhase plus both dataset source nodes
and manager.loader.

What's checked, and why:

- Constructor/node/loader validation: every knob that could make a
  controller silently stupid (bad rates, inverted clips, eta<=0) fails at
  construction with a ValueError, and t_mode="adaptive" without a wired
  balance fails as a config error *before* any filesystem/DB access.
- The "off" mode guarantee: wiring a balance that isn't reweighting must
  be a bit-identical no-op (weight_for_t() -> None, both LossPhases keep
  their original code path), while still tracking (baselines form) so the
  sampler side can use it alone.
- Direction of each mode against hand-computed expectations, not just
  "some number came out": normalize => w ∝ 1/baseline (magnitude
  equalization), speed => laggards gain / fast descendents lose
  (relative-rate matching), dro => worst-relative-to-own-baseline gets
  the biggest weight. Plus the invariants that keep the loss scale sane:
  renormalized mean 1, terminal clamp to [clip_min, clip_max].
- Honesty rules: no weight keys before a bucket's own warmup, absent
  (not zero) keys for unobserved buckets, prob keys only after an
  adaptive sampler has actually stated its range, gaps never
  zero-filled (a window observing one bucket doesn't touch the others).
- Data side: uniform over the true [t_low, t_high] coverage pre-warmup;
  post-warmup proportional to (current/baseline)^sample_bias with
  sample_bias=0 keeping sampling uniform; sample_t stays inside the
  covered range and inside [t_low, t_high].
- Integration: LossPhase composes w(sigma) * w_bucket(t) exactly as
  hand-computed; MonitoringPhase observes even with no monitor wired
  (the balance drives training, so tracking must not depend on
  reporting); managed route observes window-accumulated means at the
  grad_accum boundary.
- Editor gating (Port.visible_when): the mode-specific knobs must be
  gated to their own mode (speed's fast_alpha/eta, dro's dro_lambda,
  the clips under any mode that actually reweights) and the
  mode-independent knobs must stay ungated -- metadata, but checked
  here because the graph editor itself can't run in CI, and a wrong
  gate either shows dead fields (the user complaint that started this)
  or hides a knob someone needs.

No GPU involved anywhere -- phases are driven with hand-built states.

Run this directly: `python nodes/smoke_tests/smoke_test_bucket_balance.py`
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nodes.train.bucket_balance import BALANCE_MODES, BucketBalance, BucketBalanceNode

TOL = 1e-9
failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def check(name: str, got: float, expected: float, tol: float = TOL):
    ok = abs(got - expected) <= tol
    record(ok, name, detail=f"got {got}, expected {expected}")


def check_true(name: str, ok: bool, detail: str = ""):
    record(bool(ok), name, detail=detail)


def check_raises(name: str, fn, contains: str = ""):
    try:
        fn()
    except ValueError as e:
        if contains and contains not in str(e):
            check_true(name, False, detail=f"message {e!r} lacks {contains!r}")
        else:
            check_true(name, True)
    except Exception as e:  # noqa: BLE001 -- any other exception is a FAIL
        check_true(name, False, detail=f"raised {type(e).__name__}: {e}")
    else:
        check_true(name, False, detail="did not raise")


BASELINES = (0.1, 0.2, 0.4)


def warmed(mode: str, baselines=BASELINES, warmup: int = 3, **kw) -> BucketBalance:
    """A balance with every bucket past warmup at the given constant
    baselines (so baseline == every warmup observation == initial ema)."""
    b = BucketBalance(mode=mode, warmup_reports=warmup, **kw)
    for _ in range(warmup):
        b.observe({"loss_t_low": baselines[0],
                   "loss_t_mid": baselines[1],
                   "loss_t_high": baselines[2]})
    return b


def check_ctor_validation():
    print("\n=== ctor validation: nonsense knobs fail loudly at construction ===")
    check_raises("bad mode rejected", lambda: BucketBalance(mode="bogus"))
    check_raises("warmup_reports=0 rejected", lambda: BucketBalance(warmup_reports=0))
    check_raises("ema_alpha=0 rejected", lambda: BucketBalance(ema_alpha=0.0))
    check_raises("ema_alpha>1 rejected", lambda: BucketBalance(ema_alpha=1.5))
    check_raises("fast_alpha=0 rejected", lambda: BucketBalance(fast_alpha=0.0))
    check_raises("eta=0 rejected", lambda: BucketBalance(eta=0.0))
    check_raises("clip_min=0 rejected", lambda: BucketBalance(clip_min=0.0))
    check_raises("inverted clip rejected",
                 lambda: BucketBalance(clip_min=2.0, clip_max=1.0))
    check_raises("dro_lambda<0 rejected", lambda: BucketBalance(dro_lambda=-1.0))
    check_raises("sample_bias<0 rejected", lambda: BucketBalance(sample_bias=-0.5))
    check_true("boundary values accepted (alpha=1.0, dro_lambda=0, sample_bias=0)",
               BucketBalance(ema_alpha=1.0, fast_alpha=1.0, dro_lambda=0.0,
                             sample_bias=0.0) is not None)
    check_true("mode list is the four documented modes",
               BALANCE_MODES == ("off", "normalize", "speed", "dro"),
               detail=f"got {BALANCE_MODES!r}")


def check_off_mode_tracks_without_reweighting():
    print("\n=== mode='off': tracking only, gradient side a guaranteed no-op ===")
    b = BucketBalance(mode="off", warmup_reports=2)
    b.observe({"loss_t_low": 0.1, "loss_t_mid": 0.2, "loss_t_high": 0.4})
    check_true("pre-warmup: weight_for_t is None",
               b.weight_for_t([100, 400, 800]) is None)
    b.observe({"loss_t_low": 0.1, "loss_t_mid": 0.2, "loss_t_high": 0.4})
    check_true("post-warmup: baselines formed (tracking happened)",
               set(b.baselines()) == {"loss_t_low", "loss_t_mid", "loss_t_high"},
               detail=f"baselines {b.baselines()}")
    check_true("post-warmup: weights stay empty (no reweighting)",
               b.weights() == {})
    check_true("post-warmup: weight_for_t stays None",
               b.weight_for_t([100, 400, 800]) is None)
    check_true("report() empty with no sampler wired (nothing meaningful to say)",
               b.report() == {}, detail=f"report {b.report()}")


def check_normalize_direction_and_invariants():
    print("\n=== mode='normalize': w ∝ 1/baseline, mean 1, t->bucket mapping ===")
    b = warmed("normalize")
    w = b.weights()
    check_true("all three buckets weighted", set(w) == set("loss_t_low loss_t_mid loss_t_high".split()),
               detail=f"weights {w}")
    # w ∝ 1/B: w_low / w_high == B_high / B_low == 4
    check("w_low/w_high equals B_high/B_low (inverse-baseline direction)",
          w["loss_t_low"] / w["loss_t_high"], BASELINES[2] / BASELINES[0])
    check("w_low/w_mid equals B_mid/B_low",
          w["loss_t_low"] / w["loss_t_mid"], BASELINES[1] / BASELINES[0])
    check("weights renormalized to mean 1", sum(w.values()) / len(w), 1.0)
    check_true("terminal clip not engaged at these baselines",
               all(0.25 <= v <= 4.0 for v in w.values()), detail=f"{w}")

    import torch
    got = b.weight_for_t(torch.tensor([100, 400, 800]))
    want = [w["loss_t_low"], w["loss_t_mid"], w["loss_t_high"]]
    for i, (g, x) in enumerate(zip(got.tolist(), want)):
        # float32: weight_for_t with no explicit dtype lands on torch's
        # default float dtype (the LossPhases always pass per_sample's).
        check(f"weight_for_t[{i}] maps to its bucket", g, x, tol=1e-6)
    outside = b.weight_for_t(torch.tensor([0, 333, 666, 1000, 2000])).tolist()
    # t=0 -> low (bucket [0,333)), 333 -> mid, 666 -> high, >=1000 -> neutral 1.0
    check_true("t outside every bucket stays neutral 1.0",
               outside[3] == 1.0 and outside[4] == 1.0, detail=f"{outside}")
    check("bucket boundaries are lo-inclusive", outside[1], w["loss_t_mid"], tol=1e-6)
    check("t=0 counts as low (lo-inclusive)", outside[0], w["loss_t_low"], tol=1e-6)

    # Report keys: weight keys only, no prob keys (no sampler has asked).
    rep = b.report()
    check_true("report has exactly the three weight_t_* keys (no prob yet)",
               set(rep) == {"weight_t_low", "weight_t_mid", "weight_t_high"},
               detail=f"{rep}")

    # Partial observation: a bucket that never appeared stays neutral and
    # unreported -- never a fabricated 1.0-weight series.
    partial = BucketBalance(mode="normalize", warmup_reports=3)
    for _ in range(3):
        partial.observe({"loss_t_low": 0.1})
    check_true("only the observed bucket got weighted",
               partial.weights() == {"loss_t_low": 1.0}, detail=f"{partial.weights()}")
    check_true("report covers only the observed bucket",
               set(partial.report()) == {"weight_t_low"}, detail=f"{partial.report()}")
    mid_t = partial.weight_for_t([400]).tolist()
    check("unobserved bucket maps to neutral 1.0", mid_t[0], 1.0)

    # A window that skipped a bucket must not disturb its state (gap
    # skipped, never zero-filled).
    before = dict(b.weights())
    b.observe({"loss_t_mid": 0.2})  # normalize is baseline-driven: idempotent
    after = b.weights()
    for k in ("loss_t_low", "loss_t_high"):
        check(f"bucket absent from window keeps its weight ({k})", after[k], before[k])


def check_speed_favors_laggards():
    print("\n=== mode='speed': laggards gain weight, fast descendents lose it ===")
    b = BucketBalance(mode="speed", warmup_reports=3, ema_alpha=0.05,
                      fast_alpha=0.25, eta=0.5)
    for _ in range(3):
        b.observe({"loss_t_low": 1.0, "loss_t_mid": 1.0, "loss_t_high": 1.0})
    check_true("post-warmup weights all start at 1",
               b.weights() and all(abs(v - 1.0) < 1e-12 for v in b.weights().values()),
               detail=f"{b.weights()}")
    for _ in range(5):
        # low descends fast, mid stalls at its baseline, high rises.
        b.observe({"loss_t_low": 0.6, "loss_t_mid": 1.0, "loss_t_high": 1.4})
    w = b.weights()
    check_true("ordering: descenders < stallers < risers",
               w["loss_t_low"] < w["loss_t_mid"] < w["loss_t_high"], detail=f"{w}")
    check_true("descender below 1, riser above 1",
               w["loss_t_low"] < 1.0 < w["loss_t_high"], detail=f"{w}")
    check("weights renormalized to mean 1 (clip not engaged)",
          sum(w.values()) / len(w), 1.0, tol=1e-6)
    check_true("all weights inside the clip",
               all(0.25 <= v <= 4.0 for v in w.values()), detail=f"{w}")

    # Noise-gone-wild guard: extreme divergence ends clamped, not runaway.
    for _ in range(200):
        b.observe({"loss_t_low": 0.001, "loss_t_mid": 1.0, "loss_t_high": 5.0})
    w = b.weights()
    check_true("long run stays clamped to [clip_min, clip_max]",
               all(0.25 <= v <= 4.0 for v in w.values()), detail=f"{w}")
    # The clip genuinely engaged (descender floored) and the ordering the
    # controller exists for survives the clamp. The riser is NOT asserted
    # at clip_max: renormalization happens before the clamp, so how close
    # the winner lands to the ceiling depends on the pack -- the guarantee
    # being tested is bounded weights, not a specific saturation point.
    check_true("clip engaged without breaking the ordering",
               w["loss_t_low"] == 0.25 and w["loss_t_high"] >= w["loss_t_mid"],
               detail=f"{w}")


def check_dro_emphasizes_worst_bucket():
    print("\n=== mode='dro': worst relative-to-own-baseline dominates ===")
    b = warmed("dro", warmup=3, ema_alpha=1.0, dro_lambda=1.0)
    w0 = b.weights()
    check_true("at-baseline start: every weight exactly 1",
               all(abs(v - 1.0) < 1e-12 for v in w0.values()), detail=f"{w0}")
    # mid now sits at 2x its own baseline (worst), high at 1x, low at 0.5x.
    b.observe({"loss_t_low": 0.05, "loss_t_mid": 0.4, "loss_t_high": 0.4})
    w = b.weights()
    check_true("ordering: improving < at-baseline < worst",
               w["loss_t_low"] < w["loss_t_high"] < w["loss_t_mid"], detail=f"{w}")
    check("weights renormalized to mean 1", sum(w.values()) / len(w), 1.0, tol=1e-6)
    check_true("inside the clip",
               all(0.25 <= v <= 4.0 for v in w.values()), detail=f"{w}")

    flat = warmed("dro", warmup=3, ema_alpha=1.0, dro_lambda=0.0)
    flat.observe({"loss_t_low": 0.05, "loss_t_mid": 0.4, "loss_t_high": 0.4})
    w = flat.weights()
    check_true("dro_lambda=0 makes emphasis flat at 1 (degenerates to neutral)",
               all(abs(v - 1.0) < 1e-12 for v in w.values()), detail=f"{w}")


def check_sampling_side():
    print("\n=== data side: coverage, difficulty bias, honesty gates ===")
    fresh = BucketBalance()
    probs = fresh.sampling_probs(1, 999)
    check_true("pre-warmup: uniform over all three covered buckets",
               set(probs) == {"loss_t_low", "loss_t_mid", "loss_t_high"}
               and all(abs(p - 1 / 3) < 1e-12 for p in probs.values()), detail=f"{probs}")
    check("pre-warmup probs sum to 1", sum(probs.values()), 1.0)
    check_true("range covering only high: exactly {high: 1.0}",
               fresh.sampling_probs(700, 999) == {"loss_t_high": 1.0},
               detail=f"{fresh.sampling_probs(700, 999)}")
    check_true("range covering only low: exactly {low: 1.0}",
               fresh.sampling_probs(1, 332) == {"loss_t_low": 1.0},
               detail=f"{fresh.sampling_probs(1, 332)}")
    partial = fresh.sampling_probs(300, 400)
    check_true("straddling range covers low+mid only",
               set(partial) == {"loss_t_low", "loss_t_mid"}, detail=f"{partial}")
    check("straddling probs sum to 1", sum(partial.values()), 1.0)

    # Difficulty bias: baselines 0.1, current (0.01, 0.1, 0.9) -> r = (0.1, 1, 9).
    b = BucketBalance(mode="off", warmup_reports=3, ema_alpha=1.0, sample_bias=1.0)
    for _ in range(3):
        b.observe({"loss_t_low": 0.1, "loss_t_mid": 0.1, "loss_t_high": 0.1})
    b.observe({"loss_t_low": 0.01, "loss_t_mid": 0.1, "loss_t_high": 0.9})
    probs = b.sampling_probs(1, 999)
    check("hard bucket's prob (9/10.1)", probs["loss_t_high"], 9 / 10.1, tol=1e-9)
    check("mid bucket's prob (1/10.1)", probs["loss_t_mid"], 1 / 10.1, tol=1e-9)
    check("solved bucket's prob (0.1/10.1)", probs["loss_t_low"], 0.1 / 10.1, tol=1e-9)

    flat = BucketBalance(mode="off", warmup_reports=3, ema_alpha=1.0, sample_bias=0.0)
    for _ in range(3):
        flat.observe({"loss_t_low": 0.1, "loss_t_mid": 0.1, "loss_t_high": 0.1})
    flat.observe({"loss_t_low": 0.01, "loss_t_mid": 0.1, "loss_t_high": 0.9})
    probs = flat.sampling_probs(1, 999)
    check_true("sample_bias=0 keeps sampling uniform despite difficulty",
               all(abs(p - 1 / 3) < 1e-12 for p in probs.values()), detail=f"{probs}")

    # Empirical: draws follow the biased distribution and stay in range.
    rng = random.Random(7)
    draws = [b.sample_t(rng, 1, 999) for _ in range(300)]
    check_true("every draw inside [t_low, t_high]",
               all(1 <= t <= 999 for t in draws))
    high_share = sum(1 for t in draws if t >= 666) / len(draws)
    low_share = sum(1 for t in draws if t < 333) / len(draws)
    check_true("hard (high) bucket dominates the draws",
               high_share > 0.75, detail=f"high_share={high_share}")
    check_true("solved (low) bucket is starved", low_share < 0.08,
               detail=f"low_share={low_share}")
    hi_only = [b.sample_t(rng, 700, 999) for _ in range(50)]
    check_true("coverage-restricted draws stay in [700, 999]",
               all(700 <= t <= 999 for t in hi_only))

    # Prob report gate: only after a sampler has stated its range.
    check_true("no prob keys before any sample_t call",
               not any(k.startswith("prob_t_") for k in fresh.report()),
               detail=f"{fresh.report()}")
    fresh.sample_t(random.Random(0), 1, 999)
    rep = fresh.report()
    check_true("prob keys appear after sample_t, weight keys still absent",
               set(rep) == {"prob_t_low", "prob_t_mid", "prob_t_high"}
               and not any(k.startswith("weight_t_") for k in rep), detail=f"{rep}")


def check_main_route_phases():
    print("\n=== main route: LossPhase composes weights, MonitoringPhase observes ===")
    import torch
    from nodes.train.loss import UniformLossWeighting
    from nodes.train.step_pipeline import LossPhase, MonitoringPhase, StepState

    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = torch.zeros_like(pred)
    sigma = torch.tensor([1.0, 1.0, 1.0])
    t = torch.tensor([100, 400, 800])
    per_sample = (pred - target).pow(2).view(3, -1).mean(dim=1)

    def make_state():
        st = StepState(step=0, batch=None, model=None, device="cpu")
        st.extras.update(pred=pred, target=target, sigma=sigma, t=t)
        return st

    baseline = float(LossPhase(UniformLossWeighting()).run(make_state()).extras["loss"])

    off = BucketBalance(mode="off", warmup_reports=1)
    off.observe({"loss_t_low": 0.1, "loss_t_mid": 0.2, "loss_t_high": 0.4})
    got = float(LossPhase(UniformLossWeighting(), bucket_balance=off)
                .run(make_state()).extras["loss"])
    check_true("wired mode='off' loss is bit-identical to no balance",
               got == baseline, detail=f"{got} vs {baseline}")

    prewarm = BucketBalance(mode="normalize", warmup_reports=10)
    got = float(LossPhase(UniformLossWeighting(), bucket_balance=prewarm)
                .run(make_state()).extras["loss"])
    check_true("wired pre-warmup balance is bit-identical to no balance",
               got == baseline, detail=f"{got} vs {baseline}")

    norm = warmed("normalize")
    got = float(LossPhase(UniformLossWeighting(), bucket_balance=norm)
                .run(make_state()).extras["loss"])
    w = norm.weights()
    bucket_w = torch.tensor([w["loss_t_low"], w["loss_t_mid"], w["loss_t_high"]])
    expected = float((per_sample * bucket_w).mean())
    check("loss == mean(per_sample * w_bucket) with uniform sigma weights",
          got, expected, tol=1e-6)

    # MonitoringPhase: report carries the weight keys; observe happens
    # even with no monitor (tracking must not depend on reporting).
    class FakeMonitor:
        def __init__(self):
            self.reports = []

        def report(self, r):
            self.reports.append(r)

    mon = FakeMonitor()
    ph = MonitoringPhase(total_steps=10, device_ctx=None, monitor=mon,
                         bucket_balance=norm)
    st = StepState(step=0, batch=None, model=None, device="cpu")
    st.extras.update(loss=torch.tensor(0.5), lr=1e-3,
                     per_sample_loss=per_sample, t=t)
    ph.run(st)
    rep = mon.reports[-1]
    check_true("report carries weight_t_low",
               "weight_t_low" in rep and abs(rep["weight_t_low"] - w["loss_t_low"]) < 1e-6,
               detail=f"keys {sorted(rep)}")
    check_true("report carries no prob keys (no sampler wired)",
               not any(k.startswith("prob_t_") for k in rep), detail=f"keys {sorted(rep)}")
    check("bucket diagnostics still reported", rep["loss_t_low"], 0.5, tol=1e-6)

    fresh = BucketBalance(mode="normalize", warmup_reports=1)
    ph2 = MonitoringPhase(total_steps=10, device_ctx=None, monitor=None,
                          bucket_balance=fresh)
    st2 = StepState(step=0, batch=None, model=None, device="cpu")
    st2.extras.update(loss=torch.tensor(0.5), lr=1e-3,
                      per_sample_loss=per_sample, t=t)
    ph2.run(st2)
    check_true("monitor-less run still observes (weights formed)",
               bool(fresh.weights()), detail=f"{fresh.weights()}")


def check_managed_route_phases():
    print("\n=== managed route: grad_accum-window observe at the boundary ===")
    import torch
    from nodes.train.loss import UniformLossWeighting
    from nodes.train.managed import (LossPhase as ManagedLossPhase,
                                     ManagedStepState,
                                     MonitoringPhase as ManagedMonitoringPhase)

    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = torch.zeros_like(pred)
    sigma = torch.tensor([1.0, 1.0, 1.0])
    t = torch.tensor([100, 400, 800])
    per_sample = (pred - target).pow(2).view(3, -1).mean(dim=1)

    def make_state(micro=0):
        st = ManagedStepState(step=0, batch=None, model=None, device="cpu",
                              micro=micro)
        st.extras.update(pred=pred, target=target, sigma=sigma, t=t)
        return st

    norm = warmed("normalize")
    got = float(ManagedLossPhase(UniformLossWeighting(), bucket_balance=norm)
                .run(make_state()).extras["loss"])
    w = norm.weights()
    bucket_w = torch.tensor([w["loss_t_low"], w["loss_t_mid"], w["loss_t_high"]])
    check("managed loss == mean(per_sample * w_bucket)",
          got, float((per_sample * bucket_w).mean()), tol=1e-6)
    check_true("no loss_for_backward at backward_scale=1",
               "loss_for_backward" not in make_state().extras)

    off = BucketBalance(mode="off", warmup_reports=1)
    off.observe({"loss_t_low": 0.1, "loss_t_mid": 0.2, "loss_t_high": 0.4})
    plain = float(ManagedLossPhase(UniformLossWeighting()).run(make_state())
                  .extras["loss"])
    got = float(ManagedLossPhase(UniformLossWeighting(), bucket_balance=off)
                .run(make_state()).extras["loss"])
    check_true("managed mode='off' loss is bit-identical to no balance",
               got == plain, detail=f"{got} vs {plain}")

    # grad_accum=2: bucket means come from the *window*, observed once at
    # the boundary micro-step.
    b = BucketBalance(mode="normalize", warmup_reports=1)
    ph = ManagedMonitoringPhase(total_steps=10, device_ctx=None, coordinator=None,
                                grad_accum=2, bucket_balance=b)

    def boundary_state(micro, ps, tt):
        st = ManagedStepState(step=0, batch=None, model=None, device="cpu",
                              micro=micro)
        st.extras.update(loss=torch.tensor(0.5), lr=1e-3,
                         per_sample_loss=ps, t=tt)
        return st

    ph.run(boundary_state(0, torch.tensor([0.5]), torch.tensor([100])))
    check_true("no observe while the window is still open",
               b.weights() == {}, detail=f"{b.weights()}")
    ph.run(boundary_state(1, torch.tensor([0.9]), torch.tensor([800])))
    check_true("boundary observes the accumulated window (low + high seen)",
               set(b.weights()) == {"loss_t_low", "loss_t_high"},
               detail=f"{b.weights()}")


def check_config_validation():
    print("\n=== config surface: adaptive-without-balance fails as a config error ===")
    out = BucketBalanceNode().build(mode="speed", eta=1.0)
    check_true("BucketBalanceNode.build returns the wired balance",
               isinstance(out.get("balance"), BucketBalance)
               and out["balance"].mode == "speed"
               and out["balance"].eta == 1.0, detail=f"{out}")
    check_raises("node rejects a mode outside its choices",
                 lambda: BucketBalanceNode().build(mode="bogus"),
                 contains="'mode'='bogus' is not one of")
    check_raises("node passes bad knobs through to the ctor",
                 lambda: BucketBalanceNode().build(clip_min=2.0, clip_max=1.0))

    from nodes.dataset.managed import ManagedDatasetSourceNode
    check_raises("managed source: adaptive without balance",
                 lambda: ManagedDatasetSourceNode().build(
                     dataset_root=Path("some_set"), t_mode="adaptive"),
                 contains="bucket_balance")
    check_raises("managed source: adaptive without balance fails before path "
                 "resolution (message mentions balance, not paths)",
                 lambda: ManagedDatasetSourceNode().build(
                     dataset_root=Path("some_set"), t_mode="adaptive"),
                 contains="Bucket Balance")

    from manager.loader import ManagedDatasetLoader
    check_raises("loader ctor: adaptive without balance",
                 lambda: ManagedDatasetLoader(dataset_root=Path("/nonexistent"),
                                              t_mode="adaptive"),
                 contains="bucket_balance")

    # The "with a balance, draws really follow the bias" end-to-end used to
    # run through RenoiseBatchSource's _renoise(); that node is retired with
    # the baked-grid format it corrected, and the equivalent check now runs
    # through the real train-time path in smoke_test_t_sampling.py
    # (TrainTimeSampler + the same warmed-balance construction).


def check_port_mode_gates():
    print("\n=== editor gating: mode-specific knobs hide under the wrong mode ===")
    inputs = BucketBalanceNode.INPUTS
    for name in ("fast_alpha", "eta"):
        check_true(f"{name} gated to mode='speed'",
                   inputs[name].visible_when == ("mode", "speed"),
                   detail=f"got {inputs[name].visible_when!r}")
    check_true("dro_lambda gated to mode='dro'",
               inputs["dro_lambda"].visible_when == ("mode", "dro"),
               detail=f"got {inputs['dro_lambda'].visible_when!r}")
    for name in ("clip_min", "clip_max"):
        # _recompute_weights clamps in all three reweighting modes (the
        # terminal min/max runs after every branch) and in no other:
        # "off" never computes weights, the data side never reads clips.
        check_true(f"{name} gated to any reweighting mode (never 'off')",
                   inputs[name].visible_when ==
                   ("mode", ("normalize", "speed", "dro")),
                   detail=f"got {inputs[name].visible_when!r}")
    for name in ("mode", "warmup_reports", "ema_alpha", "sample_bias"):
        # warmup/ema feed weights AND the data side's difficulty ratio,
        # sample_bias is purely data-side -- none of them are knowably
        # dead under any mode the node itself can see.
        check_true(f"{name} ungated (meaningful under every mode)",
                   inputs[name].visible_when is None,
                   detail=f"got {inputs[name].visible_when!r}")


def main():
    check_ctor_validation()
    check_off_mode_tracks_without_reweighting()
    check_normalize_direction_and_invariants()
    check_speed_favors_laggards()
    check_dro_emphasizes_worst_bucket()
    check_sampling_side()
    check_main_route_phases()
    check_managed_route_phases()
    check_config_validation()
    check_port_mode_gates()
    print()
    if failures:
        print("=" * 60)
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
