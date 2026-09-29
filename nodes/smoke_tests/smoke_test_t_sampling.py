"""Correctness checks for manager/t_sampling.py -- train-time timestep
selection for single-latent ("lora_raw") datasets: the five static
distributions (delegated to core.noise_schedule.sample_timestep),
"adaptive" (a wired BucketBalance's per-bucket difficulty draw), and
"exact" (t pinned to a cycled t_values list) -- plus the node-side
surface that feeds it (ManagedDatasetSourceNode's t_mode/t_values Ports)
and the list-sync guarantee between nodes/dataset/timestep_modes.py's
T_MODES_TRAIN_TIME and what t_sampling actually accepts.

Why each check:

- Config-error discipline: every misconfiguration (unknown mode, exact
  without/with invalid values, inverted range, adaptive without a
  balance) must fail at construction -- in the node's build() *before*
  path resolution, in the loader's ctor *before* DB access -- with an
  actionable message, never mid-iteration and never by silently
  degrading to uniform (sample_timestep's alpha_beta.get(mode, ...)
  would happily do that for an unknown mode).
- Exact cycling is the whole point of exact mode, so it's checked as an
  exact sequence ([200,500,800,200,500,800]), not a distribution.
- Balance-steered exact: with a wired BucketBalance that can tell the
  listed values' buckets apart, draws must follow the balance's own
  (current/baseline)^sample_bias share -- and with no balance, a cold
  balance, sample_bias=0, or all listed values in one bucket, the exact
  cycle above must come back untouched (the pin is the default; the
  balance only earns the list when it has a real opinion).
- Editor gating: t_values must only exist under t_mode="exact" and
  bucket_balance only under ("adaptive", "exact") -- the mode fields
  the graph editor hides/shows via Port.visible_when, checked as
  metadata here because the editor itself can't run in CI.
- Adaptive is checked end-to-end through a real BucketBalance: a warmed
  balance biased to the hard high bucket must dominate actual draws
  (this is the check that used to run through RenoiseBatchSource's
  _renoise() -- that node retired with the baked-grid format, this is
  the same check through the real train-time path).
- Static modes stay inside [t_low, t_high] (bounds pass through).

No GPU involved. Run directly:
`python nodes/smoke_tests/smoke_test_t_sampling.py`
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.noise_schedule import T_MODES as CORE_T_MODES
from manager.t_sampling import TrainTimeSampler, parse_exact_t_values
from nodes.dataset.managed import ManagedDatasetSourceNode
from nodes.dataset.timestep_modes import T_MODES as NODES_T_MODES, T_MODES_TRAIN_TIME
from nodes.train.bucket_balance import BucketBalance

failures = []


def check_true(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


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


def check_lists_synced():
    print("\n=== node-side choices == what t_sampling actually accepts ===")
    check_true("nodes' static copy equals core's T_MODES",
               NODES_T_MODES == CORE_T_MODES,
               detail=f"nodes={NODES_T_MODES!r} core={CORE_T_MODES!r}")
    expected = (*CORE_T_MODES, "adaptive", "exact")
    check_true("T_MODES_TRAIN_TIME is a pure extension (static five + train-time two)",
               T_MODES_TRAIN_TIME == expected,
               detail=f"{T_MODES_TRAIN_TIME!r} != {expected!r}")
    check_true("t_mode Port.choices reads that constant",
               ManagedDatasetSourceNode.INPUTS["t_mode"].choices == T_MODES_TRAIN_TIME,
               detail=f"{ManagedDatasetSourceNode.INPUTS['t_mode'].choices!r}")
    check_true("t_values Port exists, defaults to empty",
               ManagedDatasetSourceNode.INPUTS["t_values"].default == "")
    # The two accepted sets really match: T_MODES + TRAIN_TIME_MODES from
    # t_sampling's own view of the world.
    from manager.t_sampling import TRAIN_TIME_MODES
    check_true("t_sampling's accepted set == (*core.T_MODES, *TRAIN_TIME_MODES) "
               "reconstructed from the node constant",
               T_MODES_TRAIN_TIME == (*CORE_T_MODES, *TRAIN_TIME_MODES),
               detail=f"{T_MODES_TRAIN_TIME!r}")


def check_parse_exact_values():
    print("\n=== parse_exact_t_values: valid specs parse, nonsense fails loudly ===")
    check_true("'500' -> [500]", parse_exact_t_values("500", 1, 999) == [500])
    check_true("'200,500,800' -> [200, 500, 800]",
               parse_exact_t_values("200,500,800", 1, 999) == [200, 500, 800])
    check_true("whitespace tolerated ('200, 500 ,800')",
               parse_exact_t_values("200, 500 ,800", 1, 999) == [200, 500, 800])
    check_true("duplicates allowed (a weighting, not a typo)",
               parse_exact_t_values("200,200,800", 1, 999) == [200, 200, 800])
    check_raises("empty spec rejected",
                 lambda: parse_exact_t_values("", 1, 999),
                 contains="t_values")
    check_raises("None rejected",
                 lambda: parse_exact_t_values(None, 1, 999),
                 contains="t_values")
    check_raises("non-integer rejected",
                 lambda: parse_exact_t_values("200,five,800", 1, 999),
                 contains="not an integer")
    check_raises("empty list entry rejected",
                 lambda: parse_exact_t_values("200,,500", 1, 999),
                 contains="not an integer")
    check_raises("t=0 rejected (zero-noise)",
                 lambda: parse_exact_t_values("0", 0, 999),
                 contains="outside the noise schedule")
    check_raises("t=1000 rejected (off-schedule)",
                 lambda: parse_exact_t_values("1000", 1, 999),
                 contains="outside the noise schedule")
    check_raises("value outside [t_low, t_high] rejected",
                 lambda: parse_exact_t_values("500", 660, 680),
                 contains="outside [t_low")


def check_static_modes():
    print("\n=== static modes: five real distributions, bounds respected ===")
    rng = random.Random(0)
    for mode in CORE_T_MODES:
        s = TrainTimeSampler(mode, 100, 900)
        ts = [s.draw(rng) for _ in range(200)]
        check_true(f"{mode}: all draws inside [t_low=100, t_high=900]",
                   all(100 <= v <= 900 for v in ts),
                   detail=f"min={min(ts)} max={max(ts)}")
    check_raises("unknown mode fails at construction (never silently uniform)",
                 lambda: TrainTimeSampler("bogus", 1, 999),
                 contains="Unknown t_mode")
    check_raises("inverted range fails at construction",
                 lambda: TrainTimeSampler("uniform", 500, 100),
                 contains="empty t range")


def check_exact_cycles():
    print("\n=== exact mode: pinned values, cycled in draw order ===")
    s = TrainTimeSampler("exact", 1, 999, t_values="200,500,800")
    seq = [s.draw(random.Random(i)) for i in range(6)]
    check_true("cycle is exact and repeats ([200,500,800] x2)",
               seq == [200, 500, 800, 200, 500, 800], detail=f"{seq}")
    s1 = TrainTimeSampler("exact", 1, 999, t_values="42")
    check_true("single value: every draw is that value",
               all(s1.draw(None) == 42 for _ in range(10)))
    s2 = TrainTimeSampler("exact", 660, 680, t_values="660,680")
    check_true("values at the range's own endpoints are legal",
               [s2.draw(None), s2.draw(None)] == [660, 680])
    check_raises("exact without t_values fails at construction",
                 lambda: TrainTimeSampler("exact", 1, 999),
                 contains="t_values")
    check_raises("exact value outside the run's range fails at construction",
                 lambda: TrainTimeSampler("exact", 660, 680, t_values="500"),
                 contains="outside [t_low")


def check_adaptive():
    print("\n=== adaptive mode: config errors + real-balance end-to-end ===")
    check_raises("adaptive without a balance fails at construction",
                 lambda: TrainTimeSampler("adaptive", 1, 999),
                 contains="bucket_balance")

    # The e2e that used to run through RenoiseBatchSource._renoise(): a
    # warmed balance biased to the hard high bucket must dominate actual
    # draws through TrainTimeSampler.draw -- the loader's real path.
    b = BucketBalance(mode="off", warmup_reports=3, ema_alpha=1.0)
    for _ in range(3):
        b.observe({"loss_t_low": 0.1, "loss_t_mid": 0.1, "loss_t_high": 0.1})
    b.observe({"loss_t_low": 0.01, "loss_t_mid": 0.1, "loss_t_high": 0.9})
    s = TrainTimeSampler("adaptive", 1, 999, bucket_balance=b)
    ts = [s.draw(random) for _ in range(200)]
    check_true("every adaptive draw inside [t_low=1, t_high=999]",
               all(1 <= v <= 999 for v in ts),
               detail=f"min={min(ts)} max={max(ts)}")
    high_share = sum(1 for v in ts if v >= 666) / len(ts)
    check_true("adaptive draws follow the balance's bias (hard high bucket dominates)",
               high_share > 0.75, detail=f"high_share={high_share}")

    # sample_bias=0 must neutralize the bias (uniform over coverage) --
    # the "either side testable alone" guarantee, from the data side.
    b2 = BucketBalance(mode="off", warmup_reports=3, ema_alpha=1.0, sample_bias=0.0)
    for _ in range(3):
        b2.observe({"loss_t_low": 0.1, "loss_t_mid": 0.1, "loss_t_high": 0.1})
    b2.observe({"loss_t_low": 0.01, "loss_t_mid": 0.1, "loss_t_high": 0.9})
    s2 = TrainTimeSampler("adaptive", 1, 999, bucket_balance=b2)
    ts2 = [s2.draw(random) for _ in range(600)]
    share2 = sum(1 for v in ts2 if v >= 666) / len(ts2)
    check_true("sample_bias=0 keeps sampling ~uniform despite the skew",
               0.2 < share2 < 0.45, detail=f"high_share={share2}")


def check_exact_steered():
    print("\n=== balance-steered exact: the balance's opinion over your pinned list ===")

    def warmed(sample_bias=1.0):
        # Same shaping as check_adaptive: baselines at 0.1, then one
        # report putting ema at (0.01, 0.1, 0.9) -- ema_alpha=1.0 makes
        # the ratios exact. mode="off" on purpose: the data side is
        # mode-independent, so steering must work whatever the gradient
        # side is doing (the adaptive e2e relies on the same fact).
        b = BucketBalance(mode="off", warmup_reports=3, ema_alpha=1.0,
                          sample_bias=sample_bias)
        for _ in range(3):
            b.observe({"loss_t_low": 0.1, "loss_t_mid": 0.1, "loss_t_high": 0.1})
        b.observe({"loss_t_low": 0.01, "loss_t_mid": 0.1, "loss_t_high": 0.9})
        return b

    # 1) Wired but uninformed balance: the pinned cycle is untouched.
    cold = BucketBalance()
    s0 = TrainTimeSampler("exact", 1, 999, bucket_balance=cold,
                          t_values="200,500,800")
    seq0 = [s0.draw(random.Random(i)) for i in range(6)]
    check_true("cold balance: exact sequence stays [200,500,800] x2",
               seq0 == [200, 500, 800, 200, 500, 800], detail=f"{seq0}")

    # 2) Warmed, skewed balance: actual draws follow the balance's own
    #    distribution over the list (computed from sampling_probs, so
    #    the check can't drift from what the balance really says).
    b = warmed()
    s1 = TrainTimeSampler("exact", 1, 999, bucket_balance=b, t_values="500,900")
    ts = [s1.draw(random) for _ in range(3000)]
    probs = b.sampling_probs(1, 999)
    want = probs["loss_t_high"] / (probs["loss_t_high"] + probs["loss_t_mid"])
    got = sum(1 for v in ts if v == 900) / len(ts)
    check_true("steered draws hit the bad bucket's t at its expected share",
               abs(got - want) < 0.04, detail=f"share(t=900)={got:.3f} want={want:.3f}")
    check_true("every steered draw is still a listed value",
               all(v in (500, 900) for v in ts))

    # 3) exact_probs' contract: normalized per value, duplicates keep
    #    their multiplied share, harder bucket outweighs the other.
    w = b.exact_probs(1, 999, [500, 900, 900])
    check_true("exact_probs sums to 1; duplicate t=900 shares weight and beats t=500",
               abs(sum(w) - 1.0) < 1e-12 and abs(w[1] - w[2]) < 1e-15 and w[2] > w[0],
               detail=f"{w}")

    # 4) sample_bias=0 neutralizes the ratio: back to the plain cycle.
    s2 = TrainTimeSampler("exact", 1, 999, bucket_balance=warmed(sample_bias=0.0),
                          t_values="500,900")
    seq2 = [s2.draw(random.Random(i)) for i in range(4)]
    check_true("sample_bias=0: the plain [500,900] cycle again",
               seq2 == [500, 900, 500, 900], detail=f"{seq2}")

    # 5) One bucket can't be told apart from itself -- nothing to steer,
    #    so the cycle stands even against a fully warmed, skewed balance.
    s3 = TrainTimeSampler("exact", 1, 999, bucket_balance=warmed(),
                          t_values="100,200")
    seq3 = [s3.draw(random.Random(i)) for i in range(4)]
    check_true("all listed values in one bucket: cycle, not noise",
               seq3 == [100, 200, 100, 200], detail=f"{seq3}")


def check_visible_when_gates():
    print("\n=== editor gating: mode-specific inputs hide under the wrong t_mode ===")
    inputs = ManagedDatasetSourceNode.INPUTS
    check_true("t_values gated to t_mode='exact'",
               inputs["t_values"].visible_when == ("t_mode", "exact"),
               detail=f"got {inputs['t_values'].visible_when!r}")
    check_true("bucket_balance gated to t_mode in ('adaptive', 'exact')",
               inputs["bucket_balance"].visible_when ==
               ("t_mode", ("adaptive", "exact")),
               detail=f"got {inputs['bucket_balance'].visible_when!r}")
    for name in ("t_mode", "t_low", "t_high"):
        check_true(f"{name} ungated (meaningful under every t_mode)",
                   inputs[name].visible_when is None,
                   detail=f"got {inputs[name].visible_when!r}")


def check_config_errors_surface_before_fs():
    print("\n=== node/loader config errors fire before any path or DB work ===")
    check_raises("node: exact without t_values, before path resolution",
                 lambda: ManagedDatasetSourceNode().build(
                     dataset_root=Path("some_set"), t_mode="exact"),
                 contains="t_values")
    check_raises("node: exact value out of range, before path resolution",
                 lambda: ManagedDatasetSourceNode().build(
                     dataset_root=Path("some_set"), t_mode="exact",
                     t_values="500", t_low=660, t_high=680),
                 contains="outside [t_low")
    check_raises("node: unknown t_mode rejected by Port choices",
                 lambda: ManagedDatasetSourceNode().build(
                     dataset_root=Path("some_set"), t_mode="bogus"),
                 contains="is not one of")

    from manager.loader import ManagedDatasetLoader
    check_raises("loader ctor: exact without t_values, before DB access",
                 lambda: ManagedDatasetLoader(
                     dataset_root=Path("/nonexistent"), t_mode="exact"),
                 contains="t_values")
    check_raises("loader ctor: unknown t_mode, before DB access",
                 lambda: ManagedDatasetLoader(
                     dataset_root=Path("/nonexistent"), t_mode="bogus"),
                 contains="Unknown t_mode")


def main():
    check_lists_synced()
    check_parse_exact_values()
    check_static_modes()
    check_exact_cycles()
    check_adaptive()
    check_exact_steered()
    check_visible_when_gates()
    check_config_errors_surface_before_fs()

    print("=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
