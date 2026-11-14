"""OptimizerStateStore: how an Algorithm's own per-parameter state dict
is actually held in memory between steps -- plain fp32
(Float32StateStore, the only real behavior before this class existed)
or block-wise quantized 8-bit (Int8BlockStateStore, ~4x smaller).
nodes/optimizer/composed.py's ComposedOptimizerHandle is the one place
this plugs in -- wrap() right after Algorithm.init_state(),
checkout()/commit() around every step() -- so every Algorithm/
ExecutionStrategy pair gets this for free, the same "written exactly
once, generically" reasoning that class's own docstring already gives
for its other lifecycle methods (offload/reload/decay/reset/footprint).
Neither strategies/*.py nor algorithms/*.py need to know this exists --
Algorithm.compute_update()/decay_state()/reset_state() are always
handed real, mutable fp32 tensors by checkout(), exactly as if state
were never wrapped at all.

Real tradeoff, not a free lunch: Int8BlockStateStore dequantizes to
real fp32 before compute_update() touches it and requantizes the
result afterward -- extra compute and real, if small and block-local,
precision loss reintroduced every single step, in exchange for roughly
4x less resident optimizer-state memory. Whether that trade is worth it
for this project's own LoRA training specifically -- where optimizer
state is already the smallest of the three residents next to the
frozen base weights and activations -- is a real, open question this
class doesn't answer on its own; it exists so the choice is a Port, not
an assumption baked in either direction.

Not the same technique, and not a substitute for, ResourceControlHandle's
own offload/reload (nodes/memory/control_handle.py): that moves a whole
resident to a different device, paying a real transfer cost on every
reload; this shrinks a resident's own footprint in place, on the same
device, every step, with no transfer at all. Direct comparison already
made and rejected for CPU-offloading optimizer state specifically (real
per-step PCIe cost, small absolute LoRA optimizer-state size to begin
with) -- this is a genuinely different lever, not a second attempt at
the same one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..quantization import dequantize_blockwise_linear_u8, quantize_blockwise_linear_u8


class OptimizerStateStore(ABC):

    @abstractmethod
    def wrap(self, state: dict[str, torch.Tensor]) -> Any:
        """Called once per parameter, right after Algorithm.init_state()
        -- converts a real fp32 state dict into whatever this store
        actually holds between steps (a "handle" -- opaque outside this
        class, ComposedOptimizerHandle only ever passes it back to
        checkout()/commit()/to()/footprint_bytes() below, never
        inspects it directly)."""

    @abstractmethod
    def checkout(self, handle: Any) -> dict[str, torch.Tensor]:
        """Real, mutable fp32 tensors -- whatever Algorithm method runs
        next (compute_update()/decay_state()/reset_state()) is always
        called with exactly this, never with a store's own internal
        representation, so none of them need to know or care which
        store is in use."""

    @abstractmethod
    def commit(self, handle: Any, state: dict[str, torch.Tensor]) -> None:
        """Write checkout()'s tensors -- mutated in place by whatever
        just ran -- back into this store's own representation."""

    @abstractmethod
    def to(self, handle: Any, device: str) -> Any:
        """Move this store's own representation to device directly, no
        checkout()/commit() round trip -- there's nothing to dequantize/
        requantize for a plain device move, only to relocate."""

    @abstractmethod
    def footprint_bytes(self, handle: Any) -> int:
        ...


class Float32StateStore(OptimizerStateStore):
    """The only real behavior before this module existed -- no
    compression at all. wrap()/checkout() are the identity; commit() is
    a no-op (compute_update() already mutated the same tensors in
    place, nothing to write back); to() is the plain per-tensor .to()
    ComposedOptimizerHandle's own lifecycle methods already did
    directly before this existed. The default -- constructing a
    ComposedOptimizerHandle without a state_store behaves exactly as it
    always did, bit for bit, nothing about this module changes that
    path at all."""

    def wrap(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return state

    def checkout(self, handle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return handle

    def commit(self, handle: dict[str, torch.Tensor], state: dict[str, torch.Tensor]) -> None:
        pass

    def to(self, handle: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
        return {k: t.to(device) for k, t in handle.items()}

    def footprint_bytes(self, handle: dict[str, torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in handle.values())


class _QuantizedTensor:
    """One state tensor (e.g. one parameter's own "m" or "v" from
    AdamW), stored block-wise quantized. Deliberately not a
    torch.Tensor subclass and never handed to an Algorithm directly --
    checkout()/commit() are the only door in or out, so nothing is
    tempted to run an in-place op (.mul_(), .add_(), ...) straight
    against quantization codes, which would silently corrupt them
    rather than the values they represent."""

    def __init__(self, q: torch.Tensor, lo: torch.Tensor, scale: torch.Tensor,
                 shape: torch.Size, dtype: torch.dtype):
        self.q = q
        self.lo = lo
        self.scale = scale
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def from_tensor(cls, t: torch.Tensor, blocksize: int) -> "_QuantizedTensor":
        q, lo, scale = quantize_blockwise_linear_u8(t.reshape(-1).to(torch.float32), blocksize)
        return cls(q, lo, scale, t.shape, t.dtype)

    def dequantize(self) -> torch.Tensor:
        n = 1
        for d in self.shape:
            n *= d
        flat = dequantize_blockwise_linear_u8(self.q, self.lo, self.scale, n)
        return flat.view(self.shape).to(self.dtype)

    def requantize_(self, t: torch.Tensor, blocksize: int) -> None:
        self.q, self.lo, self.scale = quantize_blockwise_linear_u8(
            t.reshape(-1).to(torch.float32), blocksize)

    def to(self, device: str) -> "_QuantizedTensor":
        return _QuantizedTensor(self.q.to(device), self.lo.to(device), self.scale.to(device),
                                 self.shape, self.dtype)

    def footprint_bytes(self) -> int:
        return (self.q.numel() * self.q.element_size()
                + self.lo.numel() * self.lo.element_size()
                + self.scale.numel() * self.scale.element_size())


class Int8BlockStateStore(OptimizerStateStore):
    """Block-wise linear 8-bit quantization (nodes/quantization.py) of
    every tensor in an Algorithm's own state dict -- see this module's
    own top docstring for the real tradeoff. blocksize=256 matches
    nodes/quantization.py's own default, itself bitsandbytes' real,
    current block size for exactly this kind of quantization -- see
    that module's docstring for the source."""

    def __init__(self, blocksize: int = 256):
        self._blocksize = blocksize

    def wrap(self, state: dict[str, torch.Tensor]) -> dict[str, _QuantizedTensor]:
        return {k: _QuantizedTensor.from_tensor(v, self._blocksize) for k, v in state.items()}

    def checkout(self, handle: dict[str, _QuantizedTensor]) -> dict[str, torch.Tensor]:
        return {k: qt.dequantize() for k, qt in handle.items()}

    def commit(self, handle: dict[str, _QuantizedTensor], state: dict[str, torch.Tensor]) -> None:
        for k, qt in handle.items():
            qt.requantize_(state[k], self._blocksize)

    def to(self, handle: dict[str, _QuantizedTensor], device: str) -> dict[str, _QuantizedTensor]:
        return {k: qt.to(device) for k, qt in handle.items()}

    def footprint_bytes(self, handle: dict[str, _QuantizedTensor]) -> int:
        return sum(qt.footprint_bytes() for qt in handle.values())


# Single source of truth for every composed optimizer node's own state_precision
# Port -- STRATEGIES/resolve_strategy() in strategy_registry.py already established
# this "one dict of classes + one resolver constructing fresh, not the same choices
# hand-typed on every node" shape for a different Port (strategy); same reasoning
# and same shape applied here for a second one. Classes, not shared instances, even
# though neither store below actually holds per-instance mutable state today --
# matching resolve_strategy()'s own convention exactly rather than deviating because
# it happens to be safe not to right now.
STATE_PRECISIONS: dict[str, type[OptimizerStateStore]] = {
    "float32": Float32StateStore,
    "int8_blockwise": Int8BlockStateStore,
}

STATE_PRECISION_DOC = (
    "How this optimizer's own per-parameter state (e.g. AdamW's m/v) is held in "
    "memory between steps. 'float32' (the default): no compression, today's "
    "original behavior, bit-for-bit. 'int8_blockwise': block-wise quantized to "
    "8 bits (~4x smaller resident state) -- real, small, bounded precision loss "
    "reintroduced every step in exchange for the memory saved; for LoRA training "
    "specifically, optimizer state is already the smallest of the resident pieces "
    "next to the frozen base weights and activations, so the absolute VRAM freed "
    "will be real but modest, not the difference between fitting and not fitting."
)


def resolve_state_store(name: str) -> OptimizerStateStore:
    if name not in STATE_PRECISIONS:
        raise ValueError(f"Unknown state_precision {name!r} -- choose one of {list(STATE_PRECISIONS)}")
    return STATE_PRECISIONS[name]()
