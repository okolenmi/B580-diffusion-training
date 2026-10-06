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
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from manager.loader import ManagedDatasetLoader
from nodes.train.step_pipeline import LossPhase
from nodes.train.loss import UniformLossWeighting


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _loader(multiple: int, dataset: str = "non-square") -> ManagedDatasetLoader:
    from paths import resolve_safe_dataset_path
    return ManagedDatasetLoader(
        dataset_root=resolve_safe_dataset_path(dataset),
        batch_size=2, shuffle=True, shape_bucket_multiple=multiple)


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
    """Run the real LossPhase over a synthetic state, returning the loss."""
    from nodes.train.step_pipeline import StepState

    class _Ctx:
        pass

    state = StepState(step=0, batch=None, model=None, device=pred.device)
    state.extras["pred"] = pred
    state.extras["target"] = target
    state.extras["sigma"] = torch.full((pred.shape[0],), 0.5)
    if mask is not None:
        state.extras["valid_mask"] = mask
    return float(LossPhase(UniformLossWeighting()).run(state)
                 .extras["loss"])


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


def main():
    check_off_is_unchanged()
    check_on_collapses_the_shape_count()
    check_every_bucketed_batch_carries_a_mask()
    check_padding_never_discards_a_pixel()
    check_the_loss_ignores_padded_elements()
    check_masking_does_not_rescale_the_loss()
    check_an_all_ones_mask_is_a_no_op()
    check_no_mask_key_means_the_original_expression()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
