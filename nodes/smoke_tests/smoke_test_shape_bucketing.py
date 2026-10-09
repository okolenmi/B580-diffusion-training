"""Checks optional shape bucketing: loader padding, the validity mask, and a
loss that excludes padding without rescaling.

Shape bucketing pads each latent up to the next multiple of N so a
multi-resolution dataset trains on few shapes. On `non-square` a multiple of
32 collapses 44 shapes to 3, which matters because a first sighting costs
~3.85 s and every graph run is a fresh process (MEM-05), so 44 shapes cost
~169 s per run that 3 shapes cost ~12 s.

It is a knob rather than a behaviour change, and three things have to be true
for "off by default" to mean anything:

  1. with the knob off, the loader is byte-identical to before -- same shapes,
     no mask, and the loss takes its original expression;
  2. with the knob on, padding is excluded from the loss;
  3. and the exclusion does not *rescale* the loss. Masking the squared error
     and taking a plain mean would report ~87% of the real loss for a padded
     batch and train 13% too slowly -- a silent under-training that looks like
     a bad learning rate. So the sum is divided by the valid element count,
     which keeps a padded sample's loss on the same scale as an unpadded one.

(3) is the one that would survive review and still be wrong, so it is checked
against an analytically-known case rather than a recorded number.

**Conditioning** (the last block of checks) is a separate bug with the same
cause and a worse failure mode. Both trainers read the size off x_t's shape,
which for a bucketed batch is the *bucket*, and told the UNet one resolution
for the whole batch. So a 520x480 image padded into a 768x512 bucket was
described as 768x512 -- and since a bucket mixes true sizes by construction,
that was every batch of every bucketed run, not a corner of it. Measured on
`non-square` at batch 4 with a multiple of 32, 10 of the first 10 batches had
mixed true sizes. The mask already carries the real extent, so the checks
below require the conditioning to come from it and to be byte-identical to
before when there is no mask at all.
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from manager.loader import ManagedDatasetLoader
from nodes.model.text_encoder import TextEncoder
from nodes.model.text_encoder_cache import CachingTextEncoder
from nodes.train.bucket_sizes import (encode_with_true_sizes,
                                      true_sizes_from_mask)
from nodes.train.step_pipeline import LossPhase
from nodes.train.loss import UniformLossWeighting


class _RecordingControl:
    """ResourceControlHandle stand-in that only records ensure_loaded().

    Real enough for the property under test: CachingTextEncoder calls
    ensure_loaded() *before* touching the inner encoder on a miss, and
    counts of those calls are what a cache-key mismatch looks like from the
    outside (an extra call is a reload the cache was supposed to avoid).
    """

    def __init__(self):
        self.ensure_loaded_calls: list[str] = []

    def ensure_loaded(self, name):
        self.ensure_loaded_calls.append(name)


def _discovered_keys(loader):
    """The (prompt, batch_size, height, width, count) keys text-encoder
    prewarm would find for this dataset -- the real discovery pass, not a
    re-derivation of it, so a change to the production code cannot be
    dodged by the test computing the same thing differently."""
    from nodes.model.text_encoder_prewarm import discover_dataset_keys
    return discover_dataset_keys(loader, max_batches=1000).keys


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _loader(multiple: int, dataset: str = "non-square",
           shuffle: bool = True, keep_incomplete: bool = False
           ) -> ManagedDatasetLoader:
    from paths import resolve_safe_dataset_path
    return ManagedDatasetLoader(
        dataset_root=resolve_safe_dataset_path(dataset), batch_size=2,
        shuffle=shuffle, keep_incomplete=keep_incomplete,
        shape_bucket_multiple=multiple)


def _every_sample_loader(multiple: int) -> ManagedDatasetLoader:
    """A loader that yields every sample, for checks about per-sample
    quantities.

    keep_incomplete=True, because with shuffle=True and batch_size=2 the
    loader skips a random 3 samples per epoch (its own warning says so), and
    a check that compares the dataset against the emitted batches would then
    fail on sampling noise -- two shapes hold only the skipped samples, so
    they are absent from a shuffled epoch and present in the dataset. That is
    a real property of the loader and worth knowing, but it is not what these
    checks are about.
    """
    return _loader(multiple, shuffle=True, keep_incomplete=True)


def _all_batches(multiple: int):
    return list(_loader(multiple))


def check_off_is_unchanged():
    print("[knob off: the shapes the dataset stores, and no mask anywhere]")
    shapes, masks = set(), 0
    for b in _all_batches(0):
        shapes.add(tuple(b["x_t"].shape[-2:]))
        if b.get("valid_mask") is not None:
            masks += 1
    check(masks == 0,
          f"off must emit no mask at all, got {masks} batches carrying one -- "
          f"a mask would change the loss's arithmetic for a graph that never "
          f"asked for bucketing")
    check(len(shapes) == 44,
          f"expected the dataset's own 44 shapes, got {len(shapes)}")
    print(f"    {len(shapes)} shapes, 0 masks")
    print("    PASS")


def check_on_collapses_the_shape_count():
    print("[knob on: a multiple of 32 collapses the shape count]")
    for multiple, expected in ((32, 3), (64, 3)):
        shapes = {tuple(b["x_t"].shape[-2:]) for b in _all_batches(multiple)}
        check(len(shapes) == expected,
              f"multiple of {multiple}: expected {expected} shapes, got "
              f"{len(shapes)} {sorted(shapes)}")
        for h, w in shapes:
            check(h % multiple == 0 and w % multiple == 0,
                  f"shape {h}x{w} is not a multiple of {multiple}")
    print("    multiple of 32 -> 3 shapes; multiple of 64 -> 3 shapes")
    print("    PASS")


def check_every_bucketed_batch_carries_a_mask():
    print("[every batch in a bucketed run carries a mask, including samples "
          "that needed no padding -- a bucket mixes both, and a mask dropped "
          "for the batch would train on padding]")
    for b in _all_batches(32):
        vm = b.get("valid_mask")
        check(vm is not None,
              f"a bucketed batch arrived without a mask: shape "
              f"{tuple(b['x_t'].shape[-2:])}")
        check(vm.shape == b["x_t"].shape,
              f"mask {tuple(vm.shape)} does not cover the latent "
              f"{tuple(b['x_t'].shape)}")
        check(set(vm.unique().tolist()) <= {0.0, 1.0},
              f"mask must be 0/1, got {vm.unique().tolist()[:5]}")
    print("    PASS")


def check_padding_never_discards_a_pixel():
    print("[padding keeps every real pixel: the mask covers at least the "
          "sample's own area -- rounding DOWN would reach fewer shapes by "
          "throwing data away, which is why this rounds up]")
    total_valid, total_elements = 0, 0
    for b in _all_batches(32):
        vm = b["valid_mask"]
        total_valid += float(vm.sum())
        # The mask covers every element of the latent (B, C, H, W), so the
        # denominator is the latent's element count -- not just H*W, which is
        # what makes this fraction come out larger than 1.
        total_elements += int(b["x_t"].numel())
    frac = total_valid / total_elements
    check(0.80 < frac < 0.92,
          f"valid fraction {frac:.3f} is not the measured ~0.87 (+15% compute "
          f"at a multiple of 32)")
    print(f"    mean valid fraction {frac:.3f} (i.e. ~+15% latent compute)")
    print("    PASS")


def _loss_for(pred, target, mask=None):
    """The main route's LossPhase over a synthetic state, returning the loss.

    Only the main route, because the analytic checks below are about the loss
    *mathematics*, and check_both_routes_agree() is what holds both routes to
    it. Routing every analytic case through both would have caught the managed
    route's missing mask; routing them through one and adding a separate
    equivalence check would not have, which is exactly what happened.
    """
    return _loss_both_routes(pred, target, mask)[0]


def _loss_both_routes(pred, target, mask):
    """(loss, loss) from each route's own LossPhase, on identical inputs."""
    from nodes.train.managed import LossPhase as ManagedLoss
    from nodes.train.managed import ManagedStepState
    from nodes.train.step_pipeline import LossPhase as MainLoss
    from nodes.train.step_pipeline import StepState

    def _fill(state):
        state.extras["pred"] = pred
        state.extras["target"] = target
        state.extras["sigma"] = torch.full((pred.shape[0],), 0.5)
        if mask is not None:
            state.extras["valid_mask"] = mask
        return state

    main = MainLoss(UniformLossWeighting()).run(
        _fill(StepState(step=0, batch=None, model=None, device=pred.device)))
    managed = ManagedLoss(UniformLossWeighting()).run(
        _fill(ManagedStepState(step=0, batch=None, model=None,
                              device=pred.device)))
    return (float(main.extras["loss"]), float(managed.extras["loss"]))


def check_the_loss_ignores_padded_elements():
    print("[padded elements contribute nothing to the loss: garbage in the pad "
          "region must not move the number]")
    torch.manual_seed(0)
    pred = torch.zeros(1, 4, 8, 8)
    target = torch.zeros(1, 4, 8, 8)
    target[:, :, :4, :] = 1.0            # real content: top half
    mask = torch.zeros(1, 4, 8, 8)
    mask[:, :, :4, :] = 1.0
    base = _loss_for(pred, target, mask)

    pred_padded = pred.clone()
    pred_padded[:, :, 4:, :] = 99.0      # nonsense in the padded half
    after = _loss_for(pred_padded, target, mask)
    check(abs(base - after) < 1e-9,
          f"loss moved from {base} to {after} when only padding changed -- the "
          f"mask is not excluding the padded region")
    print(f"    loss {base:.6f} -> {after:.6f} after corrupting the pad")
    print("    PASS")


def check_masking_does_not_rescale_the_loss():
    print("[masking does not rescale: a padded batch reports the same loss as "
          "the same content unpadded -- otherwise a 13%-padded batch would "
          "train 13% too slowly and look like a bad learning rate]")
    torch.manual_seed(0)
    # Half the tensor is real content worth 1.0; the rest is padding worth 0.
    target = torch.zeros(2, 4, 8, 8)
    target[:, :, :4, :] = 1.0
    pred = torch.zeros_like(target)

    mask = torch.zeros(2, 4, 8, 8)
    mask[:, :, :4, :] = 1.0

    masked = _loss_for(pred, target, mask)

    # The equivalent unpadded tensor: just the real half.
    small_target = target[:, :, :4, :].contiguous()
    small_pred = pred[:, :, :4, :].contiguous()
    unpadded = _loss_for(small_pred, small_target)

    check(abs(masked - unpadded) < 1e-6,
          f"padded loss {masked} != unpadded loss {unpadded}: masking is "
          f"dividing by the wrong count, so padding silently slows training")
    print(f"    padded {masked:.6f} == unpadded {unpadded:.6f}")
    print("    PASS")


def check_an_all_ones_mask_is_a_no_op():
    print("[an all-ones mask is exactly the unmasked loss -- the un-padded "
          "samples in a bucketed batch rely on this]")
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    plain = _loss_for(pred, target)
    ones = _loss_for(pred, target, torch.ones(2, 4, 8, 8))
    check(abs(plain - ones) < 1e-6,
          f"unmasked {plain} != all-ones-masked {ones}")
    print(f"    {plain:.6f} == {ones:.6f}")
    print("    PASS")


def check_no_mask_key_means_the_original_expression():
    print("[no mask key in the batch means LossPhase takes its original "
          "expression -- the un-bucketed path must not merely be numerically "
          "close to before]")
    import inspect
    src = inspect.getsource(LossPhase.run)
    check('extras.get("valid_mask")' in src,
          "LossPhase must read the mask defensively, not assume it exists")
    check("is not None" in src,
          "LossPhase must branch on absence rather than letting a missing key "
          "raise")
    print("    PASS")


# --- conditioning of padded samples ---------------------------------------
#
# A bucketed batch's conditioning used to be derived from x_t's shape, which
# is the *bucket*: every sample in a 96x64 bucket was described to the UNet
# as 768x512 regardless of its real extent. Measured on `non-square` at
# batch 4 with a multiple of 32, that was true of **10 of the first 10
# batches** -- a bucket mixes true sizes by construction, so this was not an
# edge case, it was every batch of every bucketed run.


class _RecordingEncoder(TextEncoder):
    """Minimal TextEncoder whose resolution embedding is a *readable* function
    of (height, width), so a wrong row is a wrong number rather than a wrong
    number nobody can attribute."""

    def __init__(self):
        self.calls: list[tuple[int, int, int]] = []

    def encode_prompt_only(self, prompt, batch_size):
        g = torch.Generator().manual_seed(len(prompt))
        return (torch.randn(batch_size, 77, 4, generator=g),
                torch.zeros(batch_size, 2))

    def resolution_embedding(self, height, width, batch_size):
        self.calls.append((height, width, batch_size))
        # One column, the size encoded as a single number, so a test can read
        # each row of y back as the (h, w) the model was told.
        return torch.full((batch_size, 1), float(height) * 10000 + float(width))

    def unload(self): ...
    def offload(self): ...
    def reload(self): ...
    def release(self): ...
    def footprint_bytes(self): return 0


def _rows_of(y) -> list[tuple[int, int]]:
    """Read each row of y back as the (height, width) it encodes."""
    return [(int(v) // 10000, int(v) % 10000) for v in y[:, -1].tolist()]


def _mask_for(h, w, H, W, top=0, left=0, channels=4):
    """A (1, channels, H, W) mask with the real region placed at (top, left) --
    the random-offset shape the loader actually emits."""
    m = torch.zeros(1, channels, H, W)
    m[:, :, top:top + h, left:left + w] = 1.0
    return m


def check_true_sizes_come_from_the_mask():
    print("[true sizes are recovered from the mask's valid region, not from "
          "the padded shape -- and not assuming the region starts at 0,0]")
    # (B, C, H, W) -> (B, H, W) -> the bounding box of the non-zero region,
    # in pixels (latent * 8, the VAE factor).
    m = torch.cat([_mask_for(6, 8, 9, 8, top=1, left=0),      # 48x64
                   _mask_for(9, 8, 9, 8)], 0)                  # 72x64, no pad
    got = true_sizes_from_mask(m)
    check(got == [(48, 64), (72, 64)],
          f"expected [(48, 64), (72, 64)] from the mask's extents, got {got}")
    print(f"    {got}")
    # A mask that is not (B, C, H, W) must not silently produce something:
    # the loader emits 4 channels, but a batch merged from (1, 4, H, W)
    # samples has B of them, and the channel count is not the point.
    m1 = torch.cat([_mask_for(6, 8, 9, 8, top=1, channels=1),
                    _mask_for(9, 8, 9, 8, channels=1)], 0)
    check(true_sizes_from_mask(m1) == [(48, 64), (72, 64)],
          "one-channel masks must give the same answer as four-channel ones")
    print("    PASS")


def check_an_all_zero_mask_is_refused():
    print("[an all-zero mask is refused, not turned into a 0x0 resolution "
          "embedding -- that would be a plausible-looking wrong value "
          "fed straight into the UNet]")
    mask = torch.zeros(1, 4, 8, 8)
    try:
        true_sizes_from_mask(mask)
    except ValueError as e:
        check("all-zero" in str(e), f"unhelpful refusal: {e}")
        print(f"    PASS: {str(e).split('.')[0]}")
        return
    raise AssertionError("an all-zero mask produced a size instead of "
                         "refusing: the loader cannot emit one, so this means "
                         "the mask did not come from the loader")


def check_a_padded_sample_is_conditioned_by_its_true_size():
    print("[the bug itself: a padded sample's conditioning equals what the "
          "SAME sample gets unpadded]")
    prompt = "a cat"
    # Unpadded: a 6x8 latent, described as 48x64.
    enc = _RecordingEncoder()
    x_plain = torch.zeros(1, 4, 6, 8)
    _, y_plain = encode_with_true_sizes(enc, prompt, 1, x_plain, None)
    check(_rows_of(y_plain) == [(48, 64)],
          f"unpadded 6x8 latent should be conditioned as 48x64, got "
          f"{_rows_of(y_plain)}")

    # Same sample, padded up to a 9x8 bucket (72x64). Its mask says 6x8 is
    # real, so its conditioning must be 48x64 -- the number above.
    enc2 = _RecordingEncoder()
    x_pad = torch.zeros(1, 4, 9, 8)
    mask = _mask_for(6, 8, 9, 8, top=2)
    _, y_pad = encode_with_true_sizes(enc2, prompt, 1, x_pad, mask)
    check(_rows_of(y_pad) == [(48, 64)],
          f"a 6x8 sample padded into a 9x8 bucket was conditioned as "
          f"{_rows_of(y_pad)}, not 48x64 -- the padded shape leaked in")
    check(torch.equal(y_plain, y_pad),
          f"padded {y_pad.tolist()} != unpadded {y_plain.tolist()} for the "
          f"same sample")
    print(f"    padded batch shape {tuple(x_pad.shape[-2:])} -> "
          f"{_rows_of(y_pad)}, identical to the unpadded run")
    print("    PASS")


def check_a_mixed_bucket_gives_different_rows():
    print("[samples of different true sizes in one bucket get different rows "
          "-- one value for the whole batch is the bug's other half]")
    mask = torch.cat([_mask_for(6, 8, 9, 8, top=1),     # 48x64
                      _mask_for(7, 8, 9, 8, top=0),     # 56x64
                      _mask_for(9, 8, 9, 8)], 0)        # 72x64, no padding
    x_t = torch.zeros(3, 4, 9, 8)                       # all one bucket
    _, y = encode_with_true_sizes(_RecordingEncoder(), "p", 3, x_t, mask)
    rows = _rows_of(y)
    check(rows == [(48, 64), (56, 64), (72, 64)],
          f"expected one row per true size, got {rows}")
    check(len(set(rows)) == 3,
          f"rows must differ per sample, got {rows}")
    # And the order must be the batch's order, not sorted or grouped.
    mask2 = torch.cat([_mask_for(9, 8, 9, 8), _mask_for(6, 8, 9, 8)], 0)
    _, y2 = encode_with_true_sizes(_RecordingEncoder(), "p", 2, x_t[:2], mask2)
    check(_rows_of(y2) == [(72, 64), (48, 64)],
          f"rows must follow batch order, got {_rows_of(y2)}")
    print(f"    {rows}")
    print("    PASS")


def check_the_unbucketed_path_is_byte_identical():
    print("[no mask = no padding = the exact tensors encode() always "
          "returned, byte for byte -- not merely close]")
    torch.manual_seed(0)
    for (h, w) in ((6, 8), (12, 12), (9, 8)):
        x_t = torch.zeros(2, 4, h, w)
        enc_a, enc_b = _RecordingEncoder(), _RecordingEncoder()
        _, via_helper = encode_with_true_sizes(enc_a, "p", 2, x_t, None)
        _, via_encode = enc_b.encode("p", 2, h * 8, w * 8)
        check(torch.equal(via_helper, via_encode),
              f"{h}x{w}: unbucketed path changed -- "
              f"{via_helper.tolist()} != {via_encode.tolist()}")
        # ...and it must have asked the encoder exactly the one call it
        # always did, so no extra device work appears in every step.
        check(enc_a.calls == [(h * 8, w * 8, 2)],
              f"{h}x{w}: expected one resolution_embedding call, got "
              f"{enc_a.calls}")
    print("    PASS: identical tensors, identical call count")


def check_both_trainers_use_the_correction():
    print("[both trainer routes use it, and neither reads x_t's shape as the "
          "sample's size]")
    import inspect
    from nodes.train import managed, step_pipeline
    for mod, cls in ((step_pipeline, "EncodeConditioningPhase"),
                     (managed, "EncodeConditioningPhase")):
        src = inspect.getsource(getattr(mod, cls).run)
        check("encode_with_true_sizes" in src,
              f"{mod.__name__}.{cls} must condition from the mask")
        check("x_t.shape[2] * 8" not in src and "x_t.shape[2]*8" not in src,
              f"{mod.__name__}.{cls} still derives the size from the padded "
              f"shape")
    # The helper is the single place that knows the rule, so a third route
    # cannot re-derive it wrongly without this failing.
    import nodes.train.bucket_sizes as bs
    check(bs.encode_with_true_sizes.__doc__ is not None,
          "the helper must carry the rule, not just implement it")
    print("    PASS")


def check_the_cache_key_matches_what_is_requested():
    print("[CachingTextEncoder checks the keys it will request -- a key "
          "checked but not filled is a silent permanent cache miss]")
    inner = _RecordingEncoder()
    control = _RecordingControl()
    cache = CachingTextEncoder(inner, resource_control=control)
    cache.encode_per_sample("p", 2, [(480, 640), (560, 640)])
    first = len(control.ensure_loaded_calls)
    check(first == 1,
          f"a both-cold per-sample encode must ensure_loaded once, got {first}")
    cache.encode_per_sample("p", 2, [(480, 640), (560, 640)])
    check(len(control.ensure_loaded_calls) == first,
          "a repeat of the same mixed batch re-loaded the encoder: the "
          "resolution keys checked are not the ones filled")
    # A new size at the same count is a resolution miss only.
    cache.encode_per_sample("p", 2, [(480, 640), (720, 640)])
    check(len(control.ensure_loaded_calls) == first + 1,
          "a new size must be one more ensure_loaded")
    # int sizes and 0-dim tensors must be the same key, or a caller that
    # derives sizes from a mask never hits a key warmed from plain ints.
    inner2 = _RecordingEncoder()
    control2 = _RecordingControl()
    cache2 = CachingTextEncoder(inner2, resource_control=control2)
    cache2.encode_per_sample("p", 2, [(480, 640), (480, 640)])
    calls_before = len(inner2.calls)
    cache2.encode_per_sample(
        "p", 2, [(torch.tensor(480), torch.tensor(640)),
                 (torch.tensor(480), torch.tensor(640))])
    check(len(inner2.calls) == calls_before,
          f"tensor sizes made a different cache key: {inner2.calls}")
    print("    PASS")


def check_pad_fraction_report_is_off_unless_asked_for():
    print("[no pad report when bucketing is off -- the knob is off by default, "
          "so the common build must print nothing new]")
    check(_loader(0).pad_fraction_stats() is None,
          "bucketing off must not produce pad statistics")
    check(_loader(0).report_pad_fraction() is None,
          "bucketing off must not print a pad report")
    print("    PASS")


def check_pad_fraction_report_matches_what_is_actually_padded():
    print("[the reported pad fraction is the fraction the loader really pads "
          "-- checked against the emitted batches, not against a formula]")
    for multiple in (16, 24, 32, 64):
        stats = _loader(multiple).pad_fraction_stats()
        check(stats is not None, f"x{multiple}: no stats with bucketing on")
        # Every shape in the report must be one a batch's mask actually
        # recovers, and every trained shape must be in the report. If the
        # report used a different rounding than _bucket_size, one of the two
        # sets would have an element the other lacks.
        reported = {(s["latent_hw"][0], s["latent_hw"][1])
                    for s in stats["per_shape"]}
        trained = set()
        counts = collections.Counter()
        for batch in _every_sample_loader(multiple):
            for h_px, w_px in true_sizes_from_mask(batch["valid_mask"]):
                trained.add((h_px // 8, w_px // 8))
                counts[(h_px // 8, w_px // 8)] += 1
        check(reported == trained,
              f"x{multiple}: report covers {len(reported)} shape(s), training "
              f"sees {len(trained)}; "
              f"{sorted(reported ^ trained)[:4]}")
        # And the per-shape sample counts must be the real ones, which is the
        # part a shape-set comparison cannot see.
        table_counts = {(s["latent_hw"][0], s["latent_hw"][1]): s["samples"]
                        for s in stats["per_shape"]}
        check(table_counts == dict(counts),
              f"x{multiple}: per-shape sample counts disagree with the "
              f"batches")
        check(stats["samples"] == sum(counts.values()),
              f"x{multiple}: sample total {stats['samples']} != "
              f"{sum(counts.values())}")
        print(f"    x{multiple}: {stats['shapes_in']} -> {stats['shapes_out']} "
              f"buckets, {stats['samples']} samples, pad median "
              f"{stats['pad_fraction_median']:.1%} max "
              f"{stats['pad_fraction_max']:.1%}")
    print("    PASS")


def check_pad_fraction_arithmetic_is_right():
    print("[the fraction itself: 1 - real/canvas, per sample, weighted over "
          "samples rather than over shapes]")
    # One shape at a time, so the numbers are checkable by hand.
    loader = _loader(16)
    stats = loader.pad_fraction_stats()
    for s in stats["per_shape"]:
        h, w = s["latent_hw"]
        H, W = s["bucket_hw"]
        check(H >= h and W >= w,
              f"{h}x{w} -> {H}x{W} shrank; bucketing pads up only")
        want = 1.0 - (h * w) / float(H * W)
        check(abs(s["pad_fraction"] - want) < 1e-12,
              f"{h}x{w}: pad fraction {s['pad_fraction']} != {want}")
        # A sample already on a multiple must report exactly zero, not a
        # rounding crumb that would show up as "1 padded sample" for a
        # shape that never padded.
        if (h, w) == (H, W):
            check(s["pad_fraction"] == 0.0,
                  f"{h}x{w} is already a bucket but reports "
                  f"{s['pad_fraction']} pad")
    check(stats["padded_samples"] ==
          sum(s["samples"] for s in stats["per_shape"] if s["pad_fraction"] > 0),
          "padded_samples must count samples, not shapes")
    check(0.0 <= stats["pad_fraction_min"] <= stats["pad_fraction_max"] <= 1.0,
          f"distribution out of range: "
          f"{stats['pad_fraction_min']}..{stats['pad_fraction_max']}")
    check(stats["pad_fraction_min"] <= stats["pad_fraction_median"]
          <= stats["pad_fraction_max"],
          "median outside [min, max]")
    check(stats["pad_fraction_median"] <= stats["pad_fraction_p90"]
          <= stats["pad_fraction_max"],
          "p90 outside [median, max]")
    # The mean is over samples, so a shape with many samples must pull it.
    weighted = sum(s["pad_fraction"] * s["samples"]
                   for s in stats["per_shape"]) / stats["samples"]
    check(abs(stats["pad_fraction_mean"] - weighted) < 1e-12,
          f"mean {stats['pad_fraction_mean']} is not the sample-weighted "
          f"{weighted}")
    unweighted = sum(s["pad_fraction"] for s in stats["per_shape"]) / len(
        stats["per_shape"])
    check(abs(stats["pad_fraction_mean"] - unweighted) > 1e-9
          or len(stats["per_shape"]) == 1,
          "mean is being taken over shapes; the shape sample counts differ, "
          "so the two cannot coincide")
    print(f"    mean {stats['pad_fraction_mean']:.4f} = sample-weighted "
          f"(shape-weighted would be {unweighted:.4f})")
    print("    PASS")


def check_per_axis_pad_report_names_the_modal_side():
    print("[per-axis report: modal side, who keeps it, who pads both axes -- "
          "the numbers that make a bad multiple legible]")
    import contextlib
    import io
    # Ground truth from the dataset, independent of the Counter logic inside
    # pad_fraction_stats: sum the per_shape table (itself checked against
    # emitted batches above) per axis and take the argmax.
    for multiple in (24, 32, 48):
        stats = _loader(multiple).pad_fraction_stats()
        n = stats["samples"]
        h_by: dict[int, int] = {}
        w_by: dict[int, int] = {}
        for s in stats["per_shape"]:
            h, w = s["latent_hw"]
            h_by[h] = h_by.get(h, 0) + s["samples"]
            w_by[w] = w_by.get(w, 0) + s["samples"]
        want_mh = max(h_by, key=lambda k: h_by[k])
        want_mw = max(w_by, key=lambda k: w_by[k])
        check(stats["modal_latent_h"] == want_mh
              and stats["modal_latent_h_samples"] == h_by[want_mh],
              f"x{multiple}: modal h {stats['modal_latent_h']} "
              f"({stats['modal_latent_h_samples']}) != table {want_mh} "
              f"({h_by[want_mh]})")
        check(stats["modal_latent_w"] == want_mw
              and stats["modal_latent_w_samples"] == w_by[want_mw],
              f"x{multiple}: modal w {stats['modal_latent_w']} "
              f"({stats['modal_latent_w_samples']}) != table {want_mw} "
              f"({w_by[want_mw]})")
        check(stats["modal_divisible"] ==
              {"h": want_mh % multiple == 0, "w": want_mw % multiple == 0},
              f"x{multiple}: divisibility flags {stats['modal_divisible']} "
              f"wrong for modal {want_mh}x{want_mw}")
        # Preserved counts recomputed from the table, same definition the
        # implementation must use (bucket == own size).
        want_hp = sum(s["samples"] for s in stats["per_shape"]
                      if s["bucket_hw"][0] == s["latent_hw"][0])
        want_wp = sum(s["samples"] for s in stats["per_shape"]
                      if s["bucket_hw"][1] == s["latent_hw"][1])
        want_mhp = sum(s["samples"] for s in stats["per_shape"]
                       if s["latent_hw"][0] == want_mh
                       and s["bucket_hw"][0] == s["latent_hw"][0])
        want_mwp = sum(s["samples"] for s in stats["per_shape"]
                       if s["latent_hw"][1] == want_mw
                       and s["bucket_hw"][1] == s["latent_hw"][1])
        want_both = sum(s["samples"] for s in stats["per_shape"]
                        if s["bucket_hw"][0] > s["latent_hw"][0]
                        and s["bucket_hw"][1] > s["latent_hw"][1])
        for key, want in (("height_preserved_samples", want_hp),
                          ("width_preserved_samples", want_wp),
                          ("modal_height_preserved_samples", want_mhp),
                          ("modal_width_preserved_samples", want_mwp),
                          ("padded_both_axes_samples", want_both)):
            check(stats[key] == want,
                  f"x{multiple}: {key}={stats[key]} != table {want}")
        # Partition: untouched + one-axis + both-axes == every sample.
        one_axis = sum(s["samples"] for s in stats["per_shape"]
                       if (s["bucket_hw"][0] > s["latent_hw"][0])
                       != (s["bucket_hw"][1] > s["latent_hw"][1]))
        untouched = sum(s["samples"] for s in stats["per_shape"]
                        if s["pad_fraction"] == 0.0)
        check(untouched + one_axis + want_both == n,
              f"x{multiple}: {untouched}+{one_axis}+{want_both} != {n}")
        check(want_both <= stats["padded_samples"] <= n,
              f"x{multiple}: both-axes {want_both} outside [0, padded]")
        # The wording the spec asks for: the report must say in words when
        # the multiple does not divide the modal side -- and stay silent
        # when it does (no recommendation either way, per the spec).
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _loader(multiple).report_pad_fraction()
        out = buf.getvalue()
        if not stats["modal_divisible"]["w"]:
            check(f"does not divide the modal width {want_mw}" in out,
                  f"x{multiple}: report never says the multiple misses the "
                  f"modal width {want_mw}")
        else:
            check("does not divide" not in out,
                  f"x{multiple}: report claims a miss that is not there")
        print(f"    x{multiple}: modal {want_mh}x{want_mw}, kept "
              f"{want_mhp}/{h_by[want_mh]} x {want_mwp}/{w_by[want_mw]}, "
              f"both-axes {want_both}/{n}")
    # Exact pins on non-square, the dataset every other check here uses:
    # x32 keeps every modal sample, x24 keeps none, x48 pads both axes on
    # nearly all. If the dataset changes these move with it -- update them,
    # do not delete the check.
    s32 = _loader(32).pad_fraction_stats()
    check((s32["modal_latent_h"], s32["modal_latent_h_samples"]) == (64, 85)
          and (s32["modal_latent_w"], s32["modal_latent_w_samples"]) == (64, 204),
          "non-square modal sides moved; re-pin the numbers, do not drop them")
    check(s32["modal_height_preserved_samples"] == 85
          and s32["modal_width_preserved_samples"] == 204
          and s32["padded_both_axes_samples"] == 0,
          "x32 must keep all 85+204 modal samples and pad both axes on none")
    s24 = _loader(24).pad_fraction_stats()
    check(s24["modal_height_preserved_samples"] == 0
          and s24["modal_width_preserved_samples"] == 0
          and s24["padded_both_axes_samples"] == 254,
          "x24 must keep no modal sample and pad both axes on 254/273")
    s48 = _loader(48).pad_fraction_stats()
    check(s48["padded_both_axes_samples"] == 256
          and s48["modal_width_preserved_samples"] == 0,
          "x48 must pad both axes on 256/273 and keep no modal width")
    print("    PASS")


def check_bucketing_cost_is_visible_at_build():
    print("[build() reports the pad fraction once, before any step -- the "
          "recurring cost of the knob is invisible in a step time, and only "
          "the one-time saving shows up there]")
    import inspect
    from nodes.dataset.managed import ManagedDatasetSourceNode
    src = inspect.getsource(ManagedDatasetSourceNode.build)
    check("report_pad_fraction" in src,
          "build() must report bucketing's cost, since it is the only point "
          "that knows the dataset's shapes without iterating it")
    # Swallowed on failure, like the cache sizing above it: a report must
    # never be the reason a dataset fails to load.
    check("except Exception" in src,
          "the report must not be able to fail a build")
    # And it must fire only when there is padding to report, so the default
    # build prints nothing new. The loader side of that is checked above.
    stats = _loader(32).report_pad_fraction()
    check(isinstance(stats, dict) and stats["pad_fraction_max"] > 0,
          "x32 must report a nonzero pad fraction")
    print("    PASS")


def check_prewarm_derives_the_keys_training_asks_for():
    print("[text-encoder prewarm discovers the TRUE sizes, so a bucketed "
          "dataset does not warm keys training never asks for while the ones "
          "it does ask for go cold -- each miss re-loads CLIP]")
    # shuffle=False here, and deliberately: with bucketing the loader groups
    # by (prompt, neg_prompt, bucketed size) and drops incomplete groups, so
    # two different shuffles of `non-square` do not train the same samples
    # and a second shuffle legitimately contains a size the first does not.
    # Comparing discovery against a reference pass needs the same pass.
    keys = _discovered_keys(_loader(32, shuffle=False))
    sizes = {(h, w) for _, _, h, w, _ in keys}
    padded = {(96 * 8, 64 * 8), (64 * 8, 64 * 8), (80 * 8, 64 * 8)}
    check(sizes != padded,
          f"discovery returned only the {len(padded)} padded bucket sizes, so "
          f"the true sizes training asks for would all miss")
    check(len(sizes) > len(padded),
          f"expected the true sizes, got {len(sizes)}")
    # Every discovered key must be one a real batch actually asks for, or
    # warming spends its budget on entries nothing reads.
    asked = set()
    for batch in _loader(32, shuffle=False):
        if batch.get("valid_mask") is None:
            asked.add((batch["x_t"].shape[2] * 8, batch["x_t"].shape[3] * 8))
        else:
            asked.update(true_sizes_from_mask(batch["valid_mask"]))
    check(sizes <= asked,
          f"discovery invented sizes no batch asks for: "
          f"{sorted(sizes - asked)[:4]}")
    check(sizes == asked,
          f"discovery missed sizes batches ask for: "
          f"{sorted(asked - sizes)[:4]}")
    # And the unbucketed case must be untouched: the same keys as before.
    plain = _discovered_keys(_loader(0, shuffle=False))
    plain_sizes = {(h, w) for _, _, h, w, _ in plain}
    plain_padded = {(tuple(b["x_t"].shape[-2:])) for b in _loader(0, shuffle=False)}
    check(plain_sizes == {(h * 8, w * 8) for h, w in plain_padded},
          "the unbucketed discovery keys changed")
    print(f"    bucketed: {len(sizes)} true sizes "
          f"(vs {len(padded)} padded buckets); unbucketed: "
          f"{len(plain_sizes)}, unchanged")
    print("    PASS")


def check_both_routes_agree():
    print("[the two trainers' LossPhase are the SAME function on the mask -- the "
          "managed one had none, and every bucketing measurement in this "
          "project was taken on the managed route]")
    torch.manual_seed(0)
    cases = []
    # analytic cases first: identical content, different pad fractions
    for pad in (0.0, 0.25, 0.5):
        pred = torch.zeros(2, 4, 8, 8)
        target = torch.zeros(2, 4, 8, 8)
        mask = torch.ones(2, 4, 8, 8)
        target[:, :, :int(8 * (1 - pad)), :] = 1.0
        mask[:, :, int(8 * (1 - pad)):, :] = 0.0
        cases.append((f"pad {pad:.0%}", pred, target, mask))
    # a real-shaped mask with a random offset, which is what the loader emits
    m = torch.zeros(2, 4, 12, 8)
    m[:, :, 3:9, 1:6] = 1.0
    cases.append(("offset 12x8", torch.randn(2, 4, 12, 8),
                  torch.randn(2, 4, 12, 8), m))
    # garbage in the pad region must change nothing. Only the PAD is
    # corrupted -- adding a constant to `pred` everywhere would corrupt the
    # real content too, and both routes would (correctly) report a different
    # number, which is the mistake this check first made.
    pred = torch.zeros(1, 4, 8, 8)
    target = torch.zeros(1, 4, 8, 8)
    target[:, :, :4, :] = 1.0
    mask = torch.zeros(1, 4, 8, 8)
    mask[:, :, :4, :] = 1.0
    pred_corrupt = pred.clone()
    pred_corrupt[:, :, 4:, :] = 99.0        # pad rows only
    cases.append(("clean pad", pred, target, mask))
    cases.append(("corrupt pad", pred_corrupt, target, mask))
    # and no mask at all, which must reduce to the plain mean
    cases.append(("no mask", torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8),
                  None))

    for label, p, t, msk in cases:
        main_loss, managed_loss = _loss_both_routes(p, t, msk)
        check(abs(main_loss - managed_loss) < 1e-6,
              f"{label}: main route {main_loss} != managed route "
              f"{managed_loss} -- the two LossPhase have drifted, and only one "
              f"of them is exercised by hw_validate")
    print(f"    {len(cases)} case(s), both routes agree to 1e-6")

    # The specific failure that shipped: a padded batch whose pad region is
    # pure garbage must still report the unpadded loss. Unfixed, the managed
    # route reported a number that moved with the garbage.
    _, clean_p, clean_t, clean_mask = cases[-3]
    _, dirty_p, dirty_t, dirty_mask = cases[-2]
    clean, clean_m = _loss_both_routes(clean_p, clean_t, clean_mask)
    dirty, dirty_m = _loss_both_routes(dirty_p, dirty_t, dirty_mask)
    check(abs(clean - dirty) < 1e-6 and abs(clean_m - dirty_m) < 1e-6,
          f"corrupting the pad moved the managed route's loss "
          f"({clean_m} -> {dirty_m}); it is still scoring the padding")
    print(f"    pad corruption moves neither route's loss ({clean:.6f})")
    print("    PASS")


def main():
    check_off_is_unchanged()
    check_on_collapses_the_shape_count()
    check_every_bucketed_batch_carries_a_mask()
    check_padding_never_discards_a_pixel()
    check_the_loss_ignores_padded_elements()
    check_masking_does_not_rescale_the_loss()
    check_an_all_ones_mask_is_a_no_op()
    check_no_mask_key_means_the_original_expression()
    check_both_routes_agree()
    check_true_sizes_come_from_the_mask()
    check_an_all_zero_mask_is_refused()
    check_a_padded_sample_is_conditioned_by_its_true_size()
    check_a_mixed_bucket_gives_different_rows()
    check_the_unbucketed_path_is_byte_identical()
    check_both_trainers_use_the_correction()
    check_the_cache_key_matches_what_is_requested()
    check_pad_fraction_report_is_off_unless_asked_for()
    check_pad_fraction_report_matches_what_is_actually_padded()
    check_pad_fraction_arithmetic_is_right()
    check_per_axis_pad_report_names_the_modal_side()
    check_bucketing_cost_is_visible_at_build()
    check_prewarm_derives_the_keys_training_asks_for()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
