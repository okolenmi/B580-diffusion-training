"""Gradient-aligned start for an adapter's down-projection factors.

A fresh LoRA has ``B = 0`` and a random ``A``: the first updates move ``B`` along ``G A^T`` (``G`` = the gradient of the loss
with respect to the full weight), so how fast it learns depends on how well the random rows of ``A`` happen to line up with
``G``. Here ``A`` starts as the top right-singular directions of ``G`` measured on a few real batches (the idea of LoRA-GA and
LoRA-One), found with a randomized SVD that needs nothing but the gradients the adapter already produces:

1. with ``B = 0`` the gradient of ``B`` is ``s G A^T``, a random sketch of ``G``'s column space; its orthonormal basis is ``Q``;
2. with ``B = eps Q / s`` (``eps`` small, so the function barely moves) the gradient of ``A`` is ``eps Q^T G``;
3. the right-singular vectors of that small matrix are the directions wanted; they become the rows of ``A`` (at the old rows'
   average length), and ``B`` goes back to zero, so the network's function at step 0 is unchanged.

Both passes see the same batches. The linear algebra runs on the host, one layer at a time."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor


class AlignedInit:
    PROBE = 1e-3  # ``eps``: how far B is moved for the second measurement

    def __init__(self, layers: Mapping[str, Any], batches: int):
        if batches < 1:
            raise ValueError("the gradient-aligned initialisation needs at least one batch")
        for path, layer in layers.items():
            if layer.kind != "lora":
                raise ValueError(f"the gradient-aligned initialisation is defined for LoRA, not {layer.kind} ({path})")
        self._layers, self.batches = dict(layers), batches

    def run(self, fetch: Callable[[], dict], backward: Callable[[dict], None], zero: Callable[[], None]) -> dict[str, float]:
        """``fetch`` gives a batch, ``backward`` adds its gradients to the adapter's, ``zero`` clears them."""
        if any(bool(layer.lora_B.detach().ne(0).any()) for layer in self._layers.values()):
            raise ValueError("the gradient-aligned initialisation needs a fresh adapter (B = 0), not a loaded one")
        batches = [fetch() for _ in range(self.batches)]
        zero()
        for batch in batches:
            backward(batch)
        bases = self._set_probe_factors()
        zero()
        for batch in batches:
            backward(batch)
        summary = self._set_directions(bases)
        zero()
        return summary

    def _set_probe_factors(self) -> dict[str, Tensor]:
        """B := eps * Q / scaling from the first measurement; returns the layers' bases (a layer the loss never reached gets none)."""
        bases: dict[str, Tensor] = {}
        for path, layer in self._layers.items():
            grad = layer.lora_B.grad
            sketch = None if grad is None else grad.detach().flatten(1).float().cpu()
            with torch.no_grad():
                layer.lora_B.zero_()
                if sketch is None or not bool(sketch.ne(0).any()):
                    continue
                basis = torch.linalg.qr(sketch).Q  # (out, min(out, rank))
                bases[path] = basis
                probe = torch.zeros(layer.lora_B.flatten(1).shape)
                probe[:, :basis.shape[1]] = basis * (self.PROBE / layer.scaling)
                layer.lora_B.copy_(probe.view_as(layer.lora_B))
        return bases

    def _set_directions(self, bases: Mapping[str, Tensor]) -> dict[str, float]:
        """A's first rows := the right-singular vectors of eps Q^T G; B := 0."""
        changed, skipped = 0, 0
        for path, layer in self._layers.items():
            grad = layer.lora_A.grad
            with torch.no_grad():
                layer.lora_B.zero_()
                if path not in bases or grad is None:
                    skipped += 1
                    continue
                small = grad.detach().flatten(1).float().cpu()[:bases[path].shape[1]]
                if not bool(small.ne(0).any()):
                    skipped += 1
                    continue
                directions = torch.linalg.svd(small, full_matrices=False).Vh
                old = layer.lora_A.detach().flatten(1).float().cpu()
                scale = old.norm(dim=1).mean()
                old[:directions.shape[0]] = directions * scale
                layer.lora_A.copy_(old.view_as(layer.lora_A).to(layer.lora_A.device))
                changed += 1
        return {"layers": float(changed), "skipped": float(skipped)}
