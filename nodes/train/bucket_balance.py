"""BucketBalance: optional per-t-bucket rebalancing -- both sides of training.

The problem it exists for, stated plainly: the optimizer minimizes ONE
scalar (the batch mean of per-sample loss), so a sum freely trades one t
region's progress against another's, and the per-t-bucket numbers
(loss_t_low/mid/high, loss.py's t_bucket_losses) that the monitor charts
plot are diagnostics only -- a bucket that stalls or regresses never
changes what the gradient does. t_mode sampling and Min-SNR/P2 weighting
are the only region levers, and both are fixed at config time: neither
can react to what is actually happening per region mid-run.

This object closes that loop. It observes the window's per-bucket means
once per optimizer step (both trainer routes' MonitoringPhase call
observe()), tracks each bucket against its own history, and exposes two
independent, optional sides:

Gradient side -- weight_for_t(): a per-sample multiplier (keyed on the
sample's t bucket) applied in LossPhase alongside the existing
sigma-based LossWeighting. Modes (chosen at build, all optional):

- "off" (default): tracking only. weight_for_t() returns None and both
  LossPhases fall through to their exact pre-existing code path --
  wiring a balance in this mode changes no arithmetic at all. Exists so
  the sampler side can use the tracking without any reweighting.
- "normalize": static inverse-baseline weights after warmup --
  w ∝ 1/baseline, renormalized to mean 1. Equalizes contribution
  *magnitude*: a bucket whose raw loss lives at 0.2 can't outshout one
  at 0.02 just by being 10x the number. No controller, no feedback
  after warmup.
- "speed": training-rate matching (the GradNorm idea, minus its
  per-layer gradient norms -- here the "training rate" is measured
  directly off the reported bucket losses): each bucket's recent rate
  (fast EMA / slow EMA; < 1 means descending) is compared to the mean
  rate across buckets, and weights step by (relative rate)^eta:
  buckets descending slower than the pack gain weight, faster ones lose
  it, renormalized to mean 1 then clamped to [clip_min, clip_max]. The
  control target is the literal "all losses should go down at the same
  speed".
- "dro": worst-bucket emphasis (Group-DRO flavored) -- w ∝
  exp(lambda * current/baseline) over buckets, renormalized: the bucket
  furthest above its own baseline dominates the objective, so one
  broken region pulls the run's capacity instead of being averaged
  away.

Honesty rules, matching the rest of this codebase: a bucket with no
baseline yet (fewer than warmup_reports observations) gets a neutral 1.0
multiplier and reports no weight key -- nothing is fabricated for a
region the run hasn't measured. A window that sampled no bucket X
leaves X's state untouched (gaps skipped, never zero-filled).

Data side -- sample_t()/sampling_probs(): adaptive t sampling for the
dataset sources. Chooses a bucket with probability ∝
(current/baseline)^sample_bias over whichever buckets actually intersect
[t_low, t_high], then draws uniformly inside that intersection -- hard
regions (relative to their own baseline) get more samples, solved ones
starve toward the floor, and pre-warmup it is plain uniform over the
coverage. Independent of `mode`: sample_bias=0 keeps sampling uniform
whatever the gradient side is doing, so either side can be tested
alone. The dataset source nodes (t_mode="adaptive") *read* this object;
the trainer *writes* it via observe() -- one shared instance wired to
both, either side optional.

Reports: MonitoringPhase adds weight_t_low/mid/high (mode != off, only
buckets past warmup) and -- once an adaptive sampler has actually asked
for a range -- prob_t_low/mid/high for the buckets that range covers.
Keys absent, not zero, for buckets that don't apply: the monitor
chart's gap rule, applied to these too.

Threading: observe() runs on the trainer's thread while sample_t() can
run inside a prefetch thread (PrefetchingBatchSourceNode). Both only
read/write plain float dicts under the GIL with no read-modify-write
spanning both sides, so no locking is needed -- and none is taken.
"""

from __future__ import annotations

import math
from typing import ClassVar

from ..core import Node, Port
from .loss import T_BUCKETS

# "off" first: it is the default, and the only mode where the gradient
# side is a guaranteed no-op.
BALANCE_MODES = ("off", "normalize", "speed", "dro")

# ((name, lo, hi), ...) with lo inclusive / hi exclusive -- T_BUCKETS's
# own convention, reused for both t->bucket lookup and [t_low, t_high]
# coverage math so the two can't disagree.
BUCKETS = T_BUCKETS


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class BucketBalance:
    """Stateful per-t-bucket tracker + rebalancer; see the module
    docstring for the problem, the four modes, and the honesty rules.

    One instance per run, wired to the trainer (observe + weight) and/or
    to the dataset source (sample_t). State accumulates for the object's
    lifetime: rebuild the graph (which rebuilds this node's output) for
    a fresh warmup."""

    def __init__(self, mode: str = "off", warmup_reports: int = 10,
                 ema_alpha: float = 0.05, fast_alpha: float = 0.25,
                 eta: float = 0.5, clip_min: float = 0.25,
                 clip_max: float = 4.0, dro_lambda: float = 1.0,
                 sample_bias: float = 1.0):
        if mode not in BALANCE_MODES:
            raise ValueError(f"mode must be one of {BALANCE_MODES}, got {mode!r}")
        if warmup_reports < 1:
            raise ValueError(f"warmup_reports must be >= 1, got {warmup_reports}")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if not (0.0 < fast_alpha <= 1.0):
            raise ValueError(f"fast_alpha must be in (0, 1], got {fast_alpha}")
        if eta <= 0.0:
            raise ValueError(f"eta must be > 0, got {eta}")
        if clip_min <= 0.0 or clip_max < clip_min:
            raise ValueError(
                f"need 0 < clip_min <= clip_max, got clip_min={clip_min}, clip_max={clip_max}")
        if dro_lambda < 0.0:
            raise ValueError(f"dro_lambda must be >= 0, got {dro_lambda}")
        if sample_bias < 0.0:
            raise ValueError(f"sample_bias must be >= 0, got {sample_bias}")

        self.mode = mode
        self.warmup_reports = int(warmup_reports)
        self.ema_alpha = float(ema_alpha)
        self.fast_alpha = float(fast_alpha)
        self.eta = float(eta)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.dro_lambda = float(dro_lambda)
        self.sample_bias = float(sample_bias)

        self._warmup: dict[str, list[float]] = {}   # bucket -> observations until baseline
        self._baseline: dict[str, float] = {}        # bucket -> baseline (own-scale anchor)
        self._ema: dict[str, float] = {}             # bucket -> slow EMA of window means
        self._fast: dict[str, float] = {}            # bucket -> fast EMA (speed mode's rate)
        self._weights: dict[str, float] = {}         # bucket -> current multiplier (mode != off)
        self._last_sample_range: tuple[int, int] | None = None  # set by sample_t(), gates prob report

    # ---- trainer side -------------------------------------------------------------

    def observe(self, bucket_means: dict[str, float]) -> None:
        """Fold one optimizer step's window means ({loss_t_*: mean raw
        MSE}) into the per-bucket tracking, then (mode != "off") refresh
        the weights.

        Keys outside T_BUCKETS and non-finite/non-positive means are
        ignored rather than poisoning the EMAs; buckets absent from this
        window are simply not updated (the window sampled nothing there,
        which is data, not zero)."""
        for key, mean in bucket_means.items():
            if key not in {name for name, _, _ in BUCKETS}:
                continue
            mean = float(mean)
            if not math.isfinite(mean) or mean <= 0.0:
                continue
            self._ema[key] = (mean if key not in self._ema else
                              self.ema_alpha * mean + (1.0 - self.ema_alpha) * self._ema[key])
            self._fast[key] = (mean if key not in self._fast else
                               self.fast_alpha * mean + (1.0 - self.fast_alpha) * self._fast[key])
            if key not in self._baseline:
                warm = self._warmup.setdefault(key, [])
                warm.append(mean)
                if len(warm) >= self.warmup_reports:
                    self._baseline[key] = _mean(warm)
                    self._warmup.pop(key, None)
        if self.mode == "off" or not self._baseline:
            return
        self._recompute_weights()

    def _eligible(self) -> list[str]:
        """Buckets past their own warmup -- the only ones with a real
        baseline, hence the only ones any mode may weight."""
        return [name for name, _, _ in BUCKETS if name in self._baseline]

    def _recompute_weights(self) -> None:
        eligible = self._eligible()
        if not eligible:
            return
        if self.mode == "normalize":
            # 1/baseline so every bucket's *scaled* loss starts at the same
            # magnitude, then mean-1 over the eligible set.
            raw = {k: 1.0 / self._baseline[k] for k in eligible}
        elif self.mode == "dro":
            # exp(lambda * current/baseline): the worst relative-to-own-
            # baseline bucket dominates; the max subtraction is only for
            # float overflow safety (it cancels in the renormalization).
            rel = {k: self._ema[k] / self._baseline[k] for k in eligible}
            peak = max(rel.values())
            raw = {k: math.exp(self.dro_lambda * (v - peak)) for k, v in rel.items()}
        else:  # "speed"
            # rate = fast/slow EMA; < 1 = descending, ~1 = stalled, > 1 =
            # rising. Relative to the pack's mean rate, then step the
            # *previous* weights by (relative)^eta -- incremental by
            # construction, so a transient noisy window nudges rather than
            # flings.
            rates = {}
            for k in eligible:
                slow = self._ema[k]
                rates[k] = (self._fast[k] / slow) if slow > 0.0 else 1.0
            pack = _mean(rates.values()) or 1.0
            raw = {k: self._weights.get(k, 1.0) * (rates[k] / pack) ** self.eta
                   for k in eligible}
        total = _mean(raw.values()) or 1.0
        self._weights = {k: min(self.clip_max, max(self.clip_min, v / total))
                         for k, v in raw.items()}

    def weight_for_t(self, t, *, dtype=None, device=None):
        """Per-sample multiplier tensor for this window's t values, or
        None when this balance applies no reweighting at all (mode
        "off", nothing observed yet, or no t) -- None is what lets both
        LossPhases keep their exact original code path."""
        if self.mode == "off" or not self._weights or t is None:
            return None
        import torch
        vals = t.detach().reshape(-1).tolist() if hasattr(t, "detach") else list(t)
        out = [self._weight_for_value(float(v)) for v in vals]
        return torch.tensor(out, dtype=dtype, device=device)

    def _weight_for_value(self, value: float) -> float:
        for name, lo, hi in BUCKETS:
            if lo <= value < hi:
                return self._weights.get(name, 1.0)
        return 1.0  # outside every bucket: neutral, never penalized for it

    def report(self) -> dict[str, float]:
        """Monitor keys for this step: weight_t_* for weighted buckets
        (absent pre-warmup and in "off" mode), and prob_t_* only once an
        adaptive sampler has stated its range -- before that there is no
        real sampling distribution to report."""
        out: dict[str, float] = {}
        for key, weight in self._weights.items():
            out[key.replace("loss_t_", "weight_t_")] = float(weight)
        if self._last_sample_range is not None:
            for key, prob in self.sampling_probs(*self._last_sample_range).items():
                out[key.replace("loss_t_", "prob_t_")] = float(prob)
        return out

    # ---- data side (dataset sources) ----------------------------------------------

    def _active_buckets(self, t_low: int, t_high: int) -> list[tuple[str, int, int]]:
        """[(bucket, inclusive_lo, inclusive_hi)] for buckets with real
        coverage of [t_low, t_high] -- T_BUCKETS' hi is exclusive, the
        t_range is inclusive at both ends (sample_timestep's randint)."""
        active = []
        for name, lo, hi in BUCKETS:
            clo, chi = max(lo, t_low), min(hi - 1, t_high)
            if clo <= chi:
                active.append((name, clo, chi))
        return active

    def sampling_probs(self, t_low: int, t_high: int) -> dict[str, float]:
        """P(bucket) over the buckets [t_low, t_high] actually covers.

        ∝ (current/baseline)^sample_bias per warmed bucket -- current and
        baseline are both the bucket's own raw-MSE scale, so the ratio is
        dimensionless and comparable across buckets. Unwarmed buckets sit
        at ratio 1.0 ("at its own baseline" -- all we honestly know).
        sample_bias=0 makes every exponent 1: plain uniform over the
        coverage, i.e. the adaptive sampler then changes nothing."""
        active = self._active_buckets(t_low, t_high)
        if not active:
            return {}
        raw = []
        for name, _, _ in active:
            if name in self._baseline and self._baseline[name] > 0.0:
                ratio = max(self._ema.get(name, self._baseline[name]) /
                            self._baseline[name], 1e-6)
                raw.append(ratio ** self.sample_bias)
            else:
                raw.append(1.0)
        total = sum(raw) or 1.0
        return {name: w / total for (name, _, _), w in zip(active, raw)}

    def sample_t(self, rng, t_low: int, t_high: int) -> int:
        """Adaptive draw for the dataset sources: bucket ~ sampling_probs,
        then uniform inside that bucket's coverage. Remembers the range so
        report() can publish the real distribution. rng only needs
        .random() and .randint() (random.Random satisfies both)."""
        self._last_sample_range = (int(t_low), int(t_high))
        active = self._active_buckets(t_low, t_high)
        if not active:
            # No bucket covers the range at all -- fall back to the plain
            # range draw (raises exactly as sample_timestep would for an
            # inverted range).
            return rng.randint(t_low, t_high)
        probs = self.sampling_probs(t_low, t_high)
        x = rng.random()
        acc = 0.0
        chosen = active[-1]
        for entry in active:
            acc += probs.get(entry[0], 0.0)
            if x <= acc:
                chosen = entry
                break
        return rng.randint(chosen[1], chosen[2])

    # ---- test/introspection helpers (cheap, read-only) ----------------------------

    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def baselines(self) -> dict[str, float]:
        return dict(self._baseline)


class BucketBalanceNode(Node):
    """Graph-editor producer for BucketBalance -- wire its `balance`
    output to a trainer's `bucket_balance` port (gradient side) and/or a
    dataset source node's `bucket_balance` port with t_mode="adaptive"
    (data side). Either side or both; mode="off" means tracking only."""

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "balance": Port(
            name="balance", type=BucketBalance, required=True,
            doc="Wire to a trainer's bucket_balance port (per-sample loss "
                "reweighting) and/or a dataset source node's bucket_balance "
                "with t_mode='adaptive' (sampling). One instance, shared: "
                "the trainer's observe() feeds what the sampler reads.",
        ),
    }

    INPUTS: ClassVar[dict[str, Port]] = {
        "mode": Port(
            name="mode", type=str, required=False, default="off",
            choices=BALANCE_MODES,
            doc="'off' = tracking only, gradient side is a guaranteed no-op "
                "(bit-identical to not wiring this); 'normalize' = static "
                "inverse-baseline weights (equalize contribution magnitude); "
                "'speed' = equalize each bucket's relative descent rate "
                "(laggers gain weight); 'dro' = worst-bucket emphasis "
                "(Group-DRO style, broken region dominates). See "
                "nodes/train/bucket_balance.py's module docstring.",
        ),
        "warmup_reports": Port(
            name="warmup_reports", type=int, required=False, default=10,
            doc="Observations a bucket needs before it gets a baseline (and "
                "before any mode weights it). Larger = stabler baselines, "
                "later start. A bucket first seen mid-run starts its own "
                "warmup clock.",
        ),
        "ema_alpha": Port(
            name="ema_alpha", type=float, required=False, default=0.05,
            doc="Slow EMA rate for the per-bucket loss level (all modes + "
                "the sampling difficulty). Smaller = smoother, slower to "
                "notice real change.",
        ),
        "fast_alpha": Port(
            name="fast_alpha", type=float, required=False, default=0.25,
            doc="'speed' mode only: fast EMA rate; its ratio to the slow "
                "EMA is the bucket's descent rate (fast/slow < 1 = falling).",
        ),
        "eta": Port(
            name="eta", type=float, required=False, default=0.5,
            doc="'speed' mode only: how hard to steer. 0.5 = square-root "
                "step (deliberate), 1.0 = full relative correction per "
                "report -- large values chase noise.",
        ),
        "clip_min": Port(
            name="clip_min", type=float, required=False, default=0.25,
            doc="Lower clamp on a bucket's multiplier: the 'no bucket gets "
                "ignored' guarantee (weight never below 0.25x).",
        ),
        "clip_max": Port(
            name="clip_max", type=float, required=False, default=4.0,
            doc="Upper clamp: keeps one noisy window from flinging the loss "
                "scale. Applied after the mean-1 renormalization.",
        ),
        "dro_lambda": Port(
            name="dro_lambda", type=float, required=False, default=1.0,
            doc="'dro' only: sharpness of the worst-bucket emphasis. "
                "0 = all buckets equal (weights flat at 1), larger = the "
                "worst bucket dominates harder.",
        ),
        "sample_bias": Port(
            name="sample_bias", type=float, required=False, default=1.0,
            doc="Data side: exponent on each bucket's current/baseline "
                "difficulty when t_mode='adaptive'. 0 = sample uniform "
                "(adaptive sampling off), 1 = proportional to difficulty, "
                ">1 = concentrate harder.",
        ),
    }

    def build(self, **inputs) -> dict[str, BucketBalance]:
        self.validate_inputs(inputs)
        balance = BucketBalance(
            mode=inputs.get("mode", self.INPUTS["mode"].default),
            warmup_reports=inputs.get(
                "warmup_reports", self.INPUTS["warmup_reports"].default),
            ema_alpha=inputs.get("ema_alpha", self.INPUTS["ema_alpha"].default),
            fast_alpha=inputs.get("fast_alpha", self.INPUTS["fast_alpha"].default),
            eta=inputs.get("eta", self.INPUTS["eta"].default),
            clip_min=inputs.get("clip_min", self.INPUTS["clip_min"].default),
            clip_max=inputs.get("clip_max", self.INPUTS["clip_max"].default),
            dro_lambda=inputs.get("dro_lambda", self.INPUTS["dro_lambda"].default),
            sample_bias=inputs.get("sample_bias", self.INPUTS["sample_bias"].default),
        )
        result = {"balance": balance}
        self.validate_outputs(result)
        return result
