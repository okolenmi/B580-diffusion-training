"""Meta-device construction guard: nothing uncovered stays uninitialised.

Both big models (UNet in unet_wrapper.py, CLIP towers in clip_encoder.py)
are now constructed on the `meta` device and materialised with `to_empty()`
to skip ~15 s of random init that the checkpoint overwrites. The hazard
that buys: any parameter or buffer the checkpoint does NOT supply is
uninitialised memory (NaN/Inf/garbage, different every run) where it used
to be deterministic init. Comparing loaded state dicts cannot see it --
only the tensors the checkpoint covered are compared -- so the check has
to be on coverage itself.

verify_loaded(model, missing, unexpected, ...) runs right after every
load_state_dict(strict=False) on a meta-built model:
- allowlisted-missing tensors get a deterministic zero fill immediately
  (they must be proven-unused, like CLIP-L's pooled projection whose
  output the encoder discards -- see clip.py's `l_out, _ = ...`);
- any other missing tensor raises, naming the first 10 keys;
- unexpected keys raise unless allowlisted (e.g. CLIP position_ids, which
  this project synthesises because newer transformers derives it).

An allowlist entry is a claim that a tensor is provably unused. Do not add
one because a checkpoint you have never loaded needs it -- find out why it
is missing first.
"""

from __future__ import annotations


def verify_loaded(model, missing: list, unexpected: list, *,
                  allow_fill: frozenset = frozenset(),
                  allow_unexpected: frozenset = frozenset(),
                  owner: str = "model") -> None:
    """Zero-fill allowlisted gaps, raise on everything else. See module doc."""
    import torch
    missing = list(missing)
    unexpected = list(unexpected)
    fill = [k for k in missing if k in allow_fill]
    bad_missing = [k for k in missing if k not in allow_fill]
    bad_unexpected = [k for k in unexpected if k not in allow_unexpected]
    if bad_missing or bad_unexpected:
        parts = []
        if bad_missing:
            shown = ", ".join(bad_missing[:10])
            parts.append(f"{len(bad_missing)} parameter(s)/buffer(s) not "
                         f"covered by the checkpoint (first: {shown})")
        if bad_unexpected:
            shown = ", ".join(bad_unexpected[:10])
            parts.append(f"{len(bad_unexpected)} unexpected checkpoint key(s) "
                         f"(first: {shown})")
        raise ValueError(
            f"{owner}: refusing a meta-built model with uncovered tensors -- "
            "they would be uninitialised memory, not init. " + "; ".join(parts))
    if fill:
        with torch.no_grad():
            state = model.state_dict()
            for k in fill:
                state[k].zero_()
