"""How an optimizer's per-parameter state is held between steps.

A store wraps one group's state (a dict of stacked tensors) into a handle, hands out plain float32
tensors for the update (``checkout``) and takes them back (``commit``). The float32 store is the identity;
the int8 store keeps the state quantized, trading a small, bounded precision loss per step for about a
quarter of the memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


def quantize_u8(flat: Tensor, blocksize: int) -> tuple[Tensor, Tensor, Tensor]:
    """Per-block linear quantization to uint8: (codes, block minimum, block scale)."""
    n = flat.numel()
    blocks_n = -(-n // blocksize)
    padded = flat.new_zeros(blocks_n * blocksize)
    padded[:n] = flat
    blocks = padded.view(blocks_n, blocksize)
    low = blocks.min(dim=1).values
    scale = (blocks.max(dim=1).values - low).clamp_min(1e-12) / 255.0
    codes = ((blocks - low.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 255).to(torch.uint8)
    return codes, low.float(), scale.float()


def dequantize_u8(codes: Tensor, low: Tensor, scale: Tensor, n: int) -> Tensor:
    return (codes.float() * scale.unsqueeze(1) + low.unsqueeze(1)).view(-1)[:n]


class StateStore(ABC):
    @abstractmethod
    def wrap(self, state: dict[str, Tensor]) -> Any: ...

    @abstractmethod
    def checkout(self, handle: Any) -> dict[str, Tensor]: ...

    @abstractmethod
    def commit(self, handle: Any, state: dict[str, Tensor]) -> None: ...

    @abstractmethod
    def to(self, handle: Any, device: str) -> Any: ...

    @abstractmethod
    def footprint_bytes(self, handle: Any) -> int: ...

    in_place = False  # True when checkout hands out the live tensors (views of them can be updated directly)


class Float32Store(StateStore):
    in_place = True

    def wrap(self, state):
        return state

    def checkout(self, handle):
        return handle

    def commit(self, handle, state) -> None:
        pass

    def to(self, handle, device):
        return {k: t.to(device) for k, t in handle.items()}

    def footprint_bytes(self, handle) -> int:
        return sum(t.numel() * t.element_size() for t in handle.values())


class _Quantized:
    def __init__(self, tensor: Tensor, blocksize: int):
        self.shape, self.blocksize = tensor.shape, blocksize
        self.codes, self.low, self.scale = quantize_u8(tensor.reshape(-1).float(), blocksize)

    def dequantize(self) -> Tensor:
        return dequantize_u8(self.codes, self.low, self.scale, self.shape.numel()).view(self.shape)

    def requantize(self, tensor: Tensor) -> None:
        self.codes, self.low, self.scale = quantize_u8(tensor.reshape(-1).float(), self.blocksize)

    def to(self, device: str) -> _Quantized:
        moved = object.__new__(_Quantized)
        moved.shape, moved.blocksize = self.shape, self.blocksize
        moved.codes, moved.low, moved.scale = self.codes.to(device), self.low.to(device), self.scale.to(device)
        return moved

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.codes, self.low, self.scale))


class Int8BlockStore(StateStore):
    def __init__(self, blocksize: int = 256):
        self.blocksize = blocksize

    def wrap(self, state):
        return {k: _Quantized(v, self.blocksize) for k, v in state.items()}

    def checkout(self, handle):
        return {k: q.dequantize() for k, q in handle.items()}

    def commit(self, handle, state) -> None:
        for k, q in handle.items():
            q.requantize(state[k])

    def to(self, handle, device):
        return {k: q.to(device) for k, q in handle.items()}

    def footprint_bytes(self, handle) -> int:
        return sum(q.nbytes() for q in handle.values())


STORES: dict[str, type[StateStore]] = {"float32": Float32Store, "int8_blockwise": Int8BlockStore}


def make_store(name: str) -> StateStore:
    if name not in STORES:
        raise ValueError(f"unknown state precision {name!r}; choose from {list(STORES)}")
    return STORES[name]()
