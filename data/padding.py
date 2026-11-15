"""What fills the padded part of a latent canvas.

Zero padding (the default) leaves a blank strip that the network's normalisation and attention see but that never
exists when generating. Mirror padding fills it with a reflection of the sample's own noisy input, so the strip
looks like image content and adds no information: it is the same noisy values as the ones it copies, not a second
noise draw of the same picture. The loss mask excludes the padded part either way.
"""

from __future__ import annotations

import torch
from torch import Tensor

FILLS = ("zero", "mirror")


def mirror_index(size: int, start: int, length: int) -> Tensor:
    """For each of ``size`` canvas positions, the position of ``length`` valid ones (starting at ``start``) it copies:
    itself inside the valid part, its reflection outside, repeating when the padding is longer than the picture."""
    position = torch.arange(size) - start
    if length == 1:
        return torch.full((size,), start)
    period = 2 * (length - 1)
    folded = position % period
    return torch.where(folded < length, folded, period - folded) + start


def mirror_fill(canvas: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """``canvas`` (C, H, W) with everything outside the valid block replaced by the reflection of the block."""
    rows = mirror_index(canvas.shape[-2], top, height)
    cols = mirror_index(canvas.shape[-1], left, width)
    return canvas[:, rows][:, :, cols]
