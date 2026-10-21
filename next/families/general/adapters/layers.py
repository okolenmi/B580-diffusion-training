"""LoRA and DoRA wrappers for ``nn.Linear`` and ``nn.Conv2d``.

The wrapped layer's weights become frozen buffers (``base_weight``, ``base_bias``); the adapter's factors are float32
parameters, because a bfloat16 adapter rounds small updates to nothing. The forward adds the low-rank update to the
base result:

* LoRA: ``base(x) + gate * scaling * B(A(dropout(x)))``.
* DoRA: the update is rescaled so the weight's per-output-row norm follows a learned magnitude vector; the norm is
  detached (section 4.3 of the DoRA paper).

``training_weight`` multiplies ``scaling`` (per-region weighting). A quantized family of layers would subclass these
and override ``frozen_weight``.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..quantize.backends import shape_problem
from ..quantize.ops import Int8Weight
from .gate import TimestepGate
from .scaling import scaling_factor

KINDS = ("lora", "dora")


class AdaptedLayer(nn.Module):
    """Shared behaviour; subclasses define the factor shapes and the base operation."""

    kind: str
    rank: int
    alpha: float
    training_weight: float

    def _setup(self, original: nn.Module, kind: str, rank: int, alpha: float, policy: str, dropout: float,
               weight: float, gate: TimestepGate | None) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown adapter kind {kind!r}; choose from {KINDS}")
        self.kind, self.rank, self.alpha, self.policy = kind, rank, alpha, policy
        self.training_weight = weight
        self.gate = gate
        self.register_buffer("base_weight", original.weight.detach())
        self.register_buffer("base_bias", original.bias.detach() if original.bias is not None else None)
        self.register_buffer("base_scale", None)  # set when the weight is stored in int8
        self._int8 = None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def scaling(self) -> float:
        return scaling_factor(self.policy, self.alpha, self.rank) * self.training_weight

    @property
    def is_int8(self) -> bool:
        return self._int8 is not None

    @property
    def frozen_weight(self) -> Tensor:
        """The weight as a dense tensor; for an int8 layer this rebuilds it, so hot paths must not call it."""
        return self._int8.backend.dequant(self.base_weight, self.base_scale) if self._int8 is not None else self.base_weight

    def _init_factors(self, a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
        device = self.base_weight.device
        self.lora_A = nn.Parameter(torch.empty(*a_shape, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(*b_shape, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.kind == "dora":
            with torch.no_grad():
                norm = self._weight_norm()
            self.magnitude = nn.Parameter(norm.to(torch.float32))

    def adapter_state(self) -> dict[str, Tensor]:
        state = {"lora_A": self.lora_A, "lora_B": self.lora_B}
        if self.kind == "dora":
            state["magnitude"] = self.magnitude
        return state

    def load_adapter_state(self, state: Mapping[str, Tensor]) -> None:
        for name in ("lora_A", "lora_B"):
            param = getattr(self, name)
            if state[name].shape != param.shape:
                raise ValueError(f"{name} shape {tuple(state[name].shape)} does not fit {tuple(param.shape)}")
            param.data.copy_(state[name].to(device=param.device, dtype=param.dtype))
        if self.kind == "dora":
            if "magnitude" in state:
                self.magnitude.data.copy_(state["magnitude"].to(device=self.magnitude.device, dtype=self.magnitude.dtype))
            else:
                with torch.no_grad():
                    self.magnitude.copy_(self._weight_norm().to(self.magnitude.dtype))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.adapter_state().values())

    def merge(self) -> None:
        """Fold the update into the base weight and zero the factors. LoRA only."""
        if self.kind != "lora":
            raise NotImplementedError("merging a DoRA adapter into its weight is not supported")
        if self.is_int8:
            raise NotImplementedError("merging into an int8 base weight is not supported")
        self.base_weight.copy_(self.base_weight + self._delta().to(self.base_weight.dtype))
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def _gated(self, out: Tensor) -> Tensor:
        return self.gate.apply(out) if self.gate is not None else out

    def _weight_norm(self) -> Tensor:
        raise NotImplementedError

    def _delta(self) -> Tensor:
        raise NotImplementedError


class AdaptedLinear(AdaptedLayer):
    def __init__(self, original: nn.Linear, kind: str = "lora", rank: int = 64, alpha: float = 1.0,
                 policy: str = "classic", dropout: float = 0.0, weight: float = 1.0,
                 gate: TimestepGate | None = None) -> None:
        super().__init__()
        self.in_features, self.out_features = original.in_features, original.out_features
        self._setup(original, kind, rank, alpha, policy, dropout, weight, gate)
        self._init_factors((rank, self.in_features), (self.out_features, rank))

    def _delta(self) -> Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def _weight_norm(self) -> Tensor:
        merged = self.frozen_weight.to(self.lora_A.dtype) + self.scaling * (self.lora_B @ self.lora_A)
        return torch.linalg.norm(merged, dim=1)

    def use_int8(self, backend) -> None:
        """Keep the base weight as int8 from now on (LoRA only: DoRA needs the dense weight in every forward)."""
        problem = shape_problem(self.out_features, self.in_features)
        if self.kind != "lora" or problem is not None:
            raise ValueError(problem or "only LoRA layers can keep their base weight in int8")
        w8, scale = backend.prepare(self.base_weight)
        self.base_weight, self.base_scale = w8, scale
        self._int8 = Int8Weight(backend)

    def _base(self, x: Tensor) -> Tensor:
        if self._int8 is not None:
            return self._int8.apply(x, self.base_weight, self.base_scale, self.base_bias)
        return F.linear(x, self.base_weight, self.base_bias)

    def forward(self, x: Tensor) -> Tensor:
        base = self._base(x)
        down = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T
        if self.kind == "lora":
            # the small factor takes the scale: one full-size multiply fewer per layer
            return base + self._gated(down @ (self.lora_B.T * self.scaling)).to(base.dtype)
        raw = down @ self.lora_B.T
        scale = (self.magnitude / self._weight_norm().detach()).to(x.dtype).view(1, -1)
        no_bias = base - self.base_bias if self.base_bias is not None else base
        update = (scale - 1) * no_bias + scale * raw.to(base.dtype) * self.scaling
        return base + self._gated(update)


class AdaptedConv2d(AdaptedLayer):
    def __init__(self, original: nn.Conv2d, kind: str = "lora", rank: int = 64, alpha: float = 1.0,
                 policy: str = "classic", dropout: float = 0.0, weight: float = 1.0,
                 gate: TimestepGate | None = None) -> None:
        super().__init__()
        self.in_channels, self.out_channels = original.in_channels, original.out_channels
        self.kernel_size, self.stride, self.padding = original.kernel_size, original.stride, original.padding
        self.dilation, self.groups = original.dilation, original.groups
        self._setup(original, kind, rank, alpha, policy, dropout, weight, gate)
        self._init_factors((rank, self.in_channels // self.groups, *self.kernel_size), (self.out_channels, rank, 1, 1))

    def _conv(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def _delta(self) -> Tensor:
        a = self.lora_A.reshape(self.rank, -1)
        b = self.lora_B.reshape(self.out_channels, self.rank)
        return (b @ a).view(self.base_weight.shape) * self.scaling

    def _weight_norm(self) -> Tensor:
        a = self.lora_A.reshape(self.rank, -1)
        b = self.lora_B.reshape(self.out_channels, self.rank)
        merged = self.frozen_weight.to(self.lora_A.dtype) + self.scaling * (b @ a).view(self.base_weight.shape)
        return merged.norm(p=2, dim=tuple(range(1, merged.dim())))

    def forward(self, x: Tensor) -> Tensor:
        base = self._conv(x, self.frozen_weight, self.base_bias)
        down = self._conv(self.dropout(x).to(self.lora_A.dtype), self.lora_A, None)
        if self.kind == "lora":
            return base + self._gated(F.conv2d(down, self.lora_B * self.scaling)).to(base.dtype)
        raw = F.conv2d(down, self.lora_B)
        scale = (self.magnitude / self._weight_norm().detach()).to(x.dtype).view(1, -1, 1, 1)
        no_bias = base - self.base_bias.view(1, -1, 1, 1) if self.base_bias is not None else base
        update = (scale - 1) * no_bias + scale * raw.to(base.dtype) * self.scaling
        return base + self._gated(update)
