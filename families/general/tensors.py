"""Reading a slice of a safetensors file: the tensors under one key prefix, nothing else."""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from safetensors import safe_open
from torch import Tensor


def read_prefixed(path: str | os.PathLike, prefix: str = "", *, exclude: Iterable[str] = (),
                  dtype: torch.dtype | None = None, device: str | torch.device = "cpu") -> dict[str, Tensor]:
    """Tensors whose key starts with ``prefix``, with the prefix removed.

    ``exclude`` lists sub-prefixes (relative to ``prefix``) to leave out. Tensors are cast to ``dtype``
    when given. Only the requested tensors are read from disk.
    """
    skip = tuple(exclude)
    out: dict[str, Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if any(name.startswith(s) for s in skip):
                continue
            tensor = handle.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            out[name] = tensor.to(device) if str(device) != "cpu" else tensor
    return out
