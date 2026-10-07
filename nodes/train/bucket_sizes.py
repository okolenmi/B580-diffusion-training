"""Per-sample true sizes for a shape-bucketed batch.

Shape bucketing (manager/loader.py's `_apply_bucket`) pads a latent up to
its bucket and ships a `valid_mask` marking the real pixels. Both trainers'
EncodeConditioningPhase used to read the size off `x_t.shape` -- the
**padded** shape -- and hand it to `TextEncoder.encode()`, which puts one
resolution embedding on every row of `y`. So for a bucketed batch the model
was told "768x512" about content that may be 520x480, and one value served
samples whose true sizes differ. That is a conditioning bug, not a
performance one: the model is being described as something it is not, on
exactly the samples whose real extent is being hidden.

This module recovers the true size per sample from the mask that already
travels with the batch, so the conditioning describes each sample's real
extent. The mask is the only source of truth available at encode time --
the loader's pre-padding `h`/`w` is not carried in the batch dict -- and
deriving it here means the two trainers cannot disagree about what the mask
means.

Deliberately not a loser's fix. The alternative -- conditioning every sample
in a bucket by the padded size -- is what the code already does, and it is
wrong in a way no loss correction can undo.
"""

from __future__ import annotations


def true_sizes_from_mask(valid_mask, downscale: int = 8) -> list[tuple[int, int]]:
    """Each sample's true (height, width) in **pixels**, from its mask.

    `valid_mask` is (B, C, H, W) float, 1 on real pixels and 0 on padding.
    The true extent is the bounding box of the non-zero region, so this is
    shape-generic over both the channel dim and the pad offset: it does not
    assume the real pixels start at row 0 / column 0, because
    `_apply_bucket` draws that offset at random per sample precisely so
    borders are not systematically real.

    `downscale` is the VAE factor between latent and pixels (8), the same
    one EncodeConditioningPhase has always multiplied by -- so a sample with
    no padding gives back exactly the number it would have computed from
    `x_t.shape` before bucketing existed.

    Returns one (height, width) per sample, in batch order.
    """
    import torch
    if valid_mask is None:
        raise ValueError("true_sizes_from_mask: valid_mask is None; a batch "
                         "with no mask has no true size to recover -- the "
                         "caller must use the padded shape in that case")
    if downscale < 1:
        raise ValueError(f"true_sizes_from_mask: downscale={downscale}, "
                         f"must be >= 1")
    mask = valid_mask
    # A mask that is 0 everywhere for a sample would make the extent 0 and
    # produce a "0x0 image" resolution embedding. The loader cannot emit one
    # (it only pads up, and an unpadded sample gets an all-ones mask), so
    # rather than trust that silently, say so -- a 0-sized resolution
    # embedding is a wrong conditioning value that would look like a
    # plausible tensor all the way into the UNet.
    occupied = mask.reshape(mask.shape[0], -1, mask.shape[-2],
                            mask.shape[-1]).any(dim=1)
    h = occupied.any(dim=-1).sum(dim=-1)
    w = occupied.any(dim=-2).sum(dim=-1)
    if bool((h == 0).any()) or bool((w == 0).any()):
        bad = torch.nonzero((h == 0) | (w == 0)).flatten().tolist()
        raise ValueError(
            f"true_sizes_from_mask: sample(s) {bad} have an all-zero "
            f"valid_mask, so their true extent is 0x0. The loader pads up "
            f"from the stored size and emits all-ones for an unpadded "
            f"sample, so this means the mask did not come from the loader")
    return [(int(hi) * downscale, int(wi) * downscale)
            for hi, wi in zip(h.tolist(), w.tolist())]


def encode_with_true_sizes(text_encoder, prompt: str, batch_size: int,
                           x_t, valid_mask, downscale: int = 8):
    """(ctx, y) for a batch, describing each sample's *true* extent.

    The one-call form when there is nothing to correct -- no mask, so no
    padding and every sample's true size is the padded one -- routes to
    `encode()` unchanged rather than to `encode_per_sample()`, so an
    unbucketed batch is byte-identical to what this always produced. That is
    a requirement, not a preference: bucketing is default-off, and turning
    it on must not perturb a graph that never asked for it.

    With a mask, the sizes come from it. A bucketed batch whose samples all
    share a true size still goes through `encode_per_sample()` rather than
    being collapsed to `encode()` on the padded size: the two agree in value
    for such a batch, but only the per-sample call is *derived* from the
    right number, and the padded size it would have used is exactly the
    number this function exists to stop using.
    """
    if valid_mask is None:
        return text_encoder.encode(prompt, batch_size=batch_size,
                                   height=x_t.shape[2] * downscale,
                                   width=x_t.shape[3] * downscale)
    sizes = true_sizes_from_mask(valid_mask, downscale=downscale)
    return text_encoder.encode_per_sample(prompt, batch_size, sizes)
