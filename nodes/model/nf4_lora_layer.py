"""NF4LoRALinear/NF4LoRAConv2d: the real forward-path wiring
NF4WeightStore (nf4_weight_store.py) has been missing since it landed --
see that module's own docstring, "Not yet wired into a real forward
pass." See docs/training_pipeline_design.md section 3.3/10 for the
design rationale (this was the single remaining construction item in
the original backlog).

**Not built via composition over core.lora.LoRALinear**, unlike
DoRALinear -- deliberately. LoRALinear.__init__ captures the frozen
base weight as a real bf16 buffer directly from `original.weight`
(`self.register_buffer("base_weight", original.weight.detach())`), which
would materialize exactly the full-size bf16 copy NF4WeightStore exists
to avoid holding at all. These classes hold the NF4WeightStore itself
instead (a plain attribute, not a buffer or parameter -- it owns no
gradient-tracked or device-resident-in-the-usual-sense state of its
own) and call `frozen.materialize()` fresh inside forward() to get the
dequantized weight for that one call, exactly matching
NF4WeightStore's own documented caching decision ("re-dequantize on
every materialize() call, never cache the dequantized bf16 result
across calls").

Otherwise matches core.lora.LoRALinear/LoRAConv2d's real math and
conventions as closely as the composition constraint above allows,
confirmed against that source directly (not recalled): same fp32
lora_A/lora_B convention and the same reasoning for it (bf16 rounds an
LoRA-scale update away entirely at typical LoRA learning rates -- see
LoRALinear.__init__'s own comment), same kaiming_uniform_ A / zero B
init, same "scale lora_B by `scaling` before the matmul, not after"
optimization, same dropout handling, and the same LoRA timestep gate
(nodes/model/lora_gate.py's apply_lora_gate(), shared with
dora_layer.py rather than a third copy of it).

Bias is kept as a plain, unquantized buffer, cast to match
frozen.materialize()'s own dtype (not necessarily original.bias's own
dtype -- NF4WeightStore always normalizes its dequantized output to
bf16 regardless of the source tensor's dtype, except for an fp16
source; casting bias to match keeps F.linear/F.conv2d's two operands
dtype-consistent regardless of what dtype the original layer happened
to be constructed in). Bias itself is `out_features` elements, a tiny
fraction of a layer's total parameter count next to the
`out_features * in_features` weight matrix NF4 actually compresses;
quantizing it would trade real precision risk for negligible VRAM
savings.

**Real, expected consequence of quantization, not a bug:** unlike
PlainLoRAAdapter/DoRAAdapter's BF16WeightStore-backed layers, these are
NOT numerically identical to a plain bf16 layer even before any LoRA
training happens -- NF4WeightStore's own real quantization error (see
that module's docstring: ~9% relative RMSE on realistic weight-like
data) means `frozen.materialize()` is already a lossy approximation of
the original weight. What IS held to the same bar as every other layer
in this codebase: given whatever `frozen.materialize()` actually
returns, the LoRA forward/backward math built on top of it is exactly
what core.lora.LoRALinear/LoRAConv2d would compute given that same
tensor as their own base_weight -- see
nodes/smoke_tests/smoke_test_nf4_lora_layer.py for that equivalence
check, done against a fixed dequantized reference specifically so
quantization error itself (already NF4WeightStore's own, separately
tested concern) doesn't get conflated with this layer's own forward math.

**Not yet using a MemoryManager-backed scratch buffer for the
dequantized tensor** -- a fresh allocation every forward call for now,
which NF4WeightStore's own docstring already establishes as correct
(if not maximally VRAM-efficient): "changes *where* the dequantized
tensor's memory comes from... not *whether* it's recomputed each call."
Real, separate follow-up, not blocking this from being a real, usable
forward path today.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_gate import apply_lora_gate
from .nf4_weight_store import NF4WeightStore


class NF4LoRALinear(nn.Module):
    """See this module's docstring for the full derivation and grounding."""

    def __init__(self, original: nn.Linear, frozen: NF4WeightStore, rank: int = 64,
                 alpha: float = 1.0, dropout: float = 0.0, weight: float = 1.0):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = (alpha / rank) * weight
        self.frozen = frozen

        if original.bias is not None:
            self.register_buffer("base_bias", original.bias.detach().to(frozen.materialize().dtype))
        else:
            self.register_buffer("base_bias", None)

        device = original.weight.device
        param_dtype = torch.float32  # see this module's docstring
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=device, dtype=param_dtype))
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=device, dtype=param_dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self.frozen.materialize()
        result = F.linear(x, base_weight, self.base_bias)

        lora_out = (self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T) \
            @ (self.lora_B.T * self.scaling)
        lora_out = apply_lora_gate(lora_out)
        return result + lora_out.to(result.dtype)

    def get_lora_weights(self):
        return self.lora_A, self.lora_B

    def load_lora_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        with torch.no_grad():
            self.lora_A.copy_(lora_A.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
            self.lora_B.copy_(lora_B.to(device=self.lora_B.device, dtype=self.lora_B.dtype))


class NF4LoRAConv2d(nn.Module):
    """See this module's docstring for the full derivation and grounding."""

    def __init__(self, original: nn.Conv2d, frozen: NF4WeightStore, rank: int = 64,
                 alpha: float = 1.0, dropout: float = 0.0, weight: float = 1.0):
        super().__init__()
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = original.kernel_size
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.rank = rank
        self.alpha = alpha
        self.scaling = (alpha / rank) * weight
        self.frozen = frozen

        if original.bias is not None:
            self.register_buffer("base_bias", original.bias.detach().to(frozen.materialize().dtype))
        else:
            self.register_buffer("base_bias", None)

        device = original.weight.device
        param_dtype = torch.float32
        self.lora_A = nn.Parameter(torch.empty(
            rank, self.in_channels // self.groups, *self.kernel_size,
            device=device, dtype=param_dtype))
        self.lora_B = nn.Parameter(torch.zeros(
            self.out_channels, rank, 1, 1, device=device, dtype=param_dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self.frozen.materialize()
        result = F.conv2d(x, base_weight, self.base_bias, self.stride,
                           self.padding, self.dilation, self.groups)

        x_in = self.dropout(x).to(self.lora_A.dtype)
        adapter = F.conv2d(x_in, self.lora_A, None, self.stride,
                            self.padding, self.dilation, self.groups)
        adapter = F.conv2d(adapter, self.lora_B * self.scaling)
        adapter = apply_lora_gate(adapter)
        return result + adapter.to(result.dtype)

    def get_lora_weights(self):
        return self.lora_A, self.lora_B

    def load_lora_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        with torch.no_grad():
            self.lora_A.copy_(lora_A.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
            self.lora_B.copy_(lora_B.to(device=self.lora_B.device, dtype=self.lora_B.dtype))
