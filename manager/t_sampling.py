"""Train-time timestep selection for single-latent ("lora_raw") datasets.

Ingestion stores exactly one clean latent (x0) per image -- no noise, no
timestep -- so every sample's t is decided at draw time, and this module
is the one place `t_mode` is interpreted (manager/loader.py calls
`draw()` once per materialized sample):

  * the five static distributions (uniform/low/mid/high/logit) are
    delegated to core.noise_schedule.sample_timestep, which is where
    they are actually implemented;
  * "adaptive" reads a wired BucketBalance's live per-bucket difficulty
    (nodes/train/bucket_balance.py's data side): bucket ~
    (current/baseline)^sample_bias over whichever buckets [t_low, t_high]
    covers, uniform inside the picked bucket;
  * "exact" pins t to an explicit list of timesteps -- t_values="500"
    for one precise value, or t_values="200,500,800" cycled in draw
    order (one value per sample drawn, so every listed t gets an equal
    long-run share no matter how the epoch shuffles). The direct
    counterpart of t_low/t_high: that narrows the range, this removes it.
    A wired BucketBalance can steer that list: each listed value's draw
    weight becomes the difficulty of the bucket it falls in --
    (current/baseline)^sample_bias, the data side's own ratio, read
    through the balance's exact_probs() -- so your precise t's get
    pulled toward whichever zone the balance measures as behind
    (exact targeting inside the bucket-balance loop, not a separate
    knob). When the balance has nothing to say -- none wired, all
    buckets unwarmed, sample_bias=0, or every listed value in one
    bucket -- the plain pinned cycle is untouched.

Everything is validated in the constructor, deliberately: a
misconfigured run (unknown mode, exact with missing/invalid values,
adaptive without a balance, t_low > t_high) must fail as a config error
before any dataset access, never mid-iteration -- and never silently.
sample_timestep's alpha_beta.get(mode, ...) degrades modes it doesn't
know to Beta(1, 1) == uniform, and a silently-uniform mode is a lie.

Ownership/sync: manager/ never imports nodes/ (the BucketBalance is
duck-typed by contract -- its `observe()`/`sample_t()`/`exact_probs()`
are documented here, not typed), and nodes/dataset/timestep_modes.py -- where the
graph editor's `choices` list lives -- can't import this module (a
Port's `choices` is needed at class-definition time, before torch/core
are importable there). That node-side constant,
T_MODES_TRAIN_TIME = (*T_MODES, "adaptive", "exact"), is therefore a
second, documented copy of the accepted set, kept honest by
nodes/smoke_tests/smoke_test_t_sampling.py.
"""

from __future__ import annotations

from core.noise_schedule import T_MODES, sample_timestep

# The modes this module interprets beyond core's five static
# distributions -- mirrored (as a copy, for the reason above) by
# nodes/dataset/timestep_modes.py's T_MODES_TRAIN_TIME.
TRAIN_TIME_MODES = ("adaptive", "exact")


def parse_exact_t_values(spec, t_low: int, t_high: int) -> list[int]:
    """Parse and validate `t_values` for t_mode="exact" (the Port's raw
    string form: "500" or "200,500,800").

    Raises ValueError -- a config error -- on anything that isn't a
    usable list of timesteps inside both the noise schedule (1..999;
    t=0 would be zero-noise) and the run's own [t_low, t_high] range.
    Duplicates are allowed on purpose: "200,200,800" is a legitimate
    way to weight the cycle.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(
            "t_mode='exact' needs t_values: comma-separated timesteps to pin t to, "
            "e.g. t_values='500' (every sample at t=500) or t_values='200,500,800' "
            "(cycled in draw order).")
    values = []
    for part in spec.split(","):
        text = part.strip()
        try:
            v = int(text)
        except ValueError:
            raise ValueError(
                f"t_values={spec!r}: {text!r} is not an integer timestep -- expected "
                "comma-separated integers like '200,500,800'.") from None
        if not (1 <= v <= 999):
            raise ValueError(
                f"t_values={spec!r}: {v} is outside the noise schedule (valid t is "
                "1..999; t=0 would be zero-noise).")
        if not (t_low <= v <= t_high):
            raise ValueError(
                f"t_values={spec!r}: {v} is outside [t_low={t_low}, t_high={t_high}] "
                "-- exact timesteps must lie inside the run's training range (widen "
                "t_low/t_high, or fix the value).")
        values.append(v)
    return values


class TrainTimeSampler:
    """A run's validated (t_mode, t_range[, t_values][, balance]) config
    plus the draw() dispatch. One instance per loader, which matters for
    `exact`: the cycle cursor lives here, so it advances across batches
    *and* across epochs (the loader's __iter__ restarts every epoch but
    this object persists), giving each listed t an equal long-run share
    regardless of shuffling. (While a wired balance is actively steering
    the list, draws are weighted picks instead of cycle positions --
    also per-loader, just no longer sequential.)"""

    def __init__(self, t_mode: str, t_low: int, t_high: int,
                 bucket_balance=None, t_values: str = ""):
        if t_low > t_high:
            raise ValueError(f"t_low={t_low} > t_high={t_high} -- empty t range.")
        self.t_mode = t_mode
        self.t_low = t_low
        self.t_high = t_high
        self._bucket_balance = bucket_balance
        if t_mode == "adaptive":
            if bucket_balance is None:
                raise ValueError(
                    "t_mode='adaptive' requires a bucket_balance (a "
                    "nodes/train/bucket_balance.BucketBalance instance) -- without it "
                    "there is no progress signal to be adaptive with. Wire a Bucket "
                    "Balance node, or pick a static t_mode.")
            self._exact_values = None
        elif t_mode == "exact":
            self._exact_values = parse_exact_t_values(t_values, t_low, t_high)
        elif t_mode in T_MODES:
            self._exact_values = None
        else:
            raise ValueError(
                f"Unknown t_mode={t_mode!r} -- valid modes: "
                f"{tuple(T_MODES) + TRAIN_TIME_MODES}. (Rejected here because "
                "sample_timestep would silently degrade an unknown mode to uniform.)")
        self._cursor = 0

    def draw(self, rng) -> int:
        """The next sample's t. `rng` supplies randomness for the static,
        adaptive, and balance-steered exact paths (the loader passes the
        `random` module -- the global stream, as before); an exact list
        nothing steers is deterministic by construction (cycle
        position), so `rng` goes unused there."""
        if self.t_mode == "exact":
            return self._draw_exact(rng)
        if self.t_mode == "adaptive":
            # Bounds pass straight through: sample_t() picks a bucket inside
            # the [t_low, t_high] coverage, then draws uniform inside it.
            return self._bucket_balance.sample_t(rng, self.t_low, self.t_high)
        return sample_timestep(rng, self.t_mode, self.t_low, self.t_high)

    def _draw_exact(self, rng) -> int:
        """Exact mode: the plain pinned cycle, unless a wired balance can
        tell the listed values' buckets apart -- then a weighted pick
        over the list (exact_probs' all-equal contract = "nothing to
        steer" = the cycle below). The cursor only advances on the cycle
        path; a steered draw has no position to keep."""
        values = self._exact_values
        weights = None
        if self._bucket_balance is not None:
            weights = self._bucket_balance.exact_probs(
                self.t_low, self.t_high, values)
            if max(weights) - min(weights) <= 1e-12:
                weights = None
        if weights is None:
            v = values[self._cursor % len(values)]
            self._cursor += 1
            return v
        # Weighted pick: value ~ its bucket's (current/baseline)^sample_bias
        # share; duplicates in the list carry their multiplied share, and
        # the trailing fallback only covers the float round-off gap at 1.0.
        x = rng.random()
        acc = 0.0
        for v, w in zip(values, weights):
            acc += w
            if x < acc:
                return v
        return values[-1]
