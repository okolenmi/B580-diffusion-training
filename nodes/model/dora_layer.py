"""DoRALinear/DoRAConv2d: Liu et al., "DoRA: Weight-Decomposed Low-Rank
Adaptation" (arXiv:2402.09353, ICML 2024 Oral). See
docs/training_pipeline_design.md section 3.1 for the design rationale.

Grounded directly in HuggingFace PEFT's real implementation
(peft/src/peft/tuners/lora/dora.py, fetched and read directly, not
recalled from training data or the paper's own notation -- which is
genuinely ambiguous about which axis "column-wise" norm means relative
to nn.Linear's [out_features, in_features] weight layout). Cross-checked
against Meta's torchtune implementation, which computes the exact same
thing independently. Both compute `torch.linalg.norm(weight, dim=1)` on
a [out_features, in_features] weight -- one magnitude scalar per OUTPUT
channel, matching standard weight-normalization intuition (Salimans &
Kingma, 2016), not literally "per column" the way you'd read it off the
matrix on paper.

**The efficient forward formulation** (not the naive "merge the full
weight, then run a normal linear/conv", which would defeat LoRA's
efficiency and this project's own VRAM-first design goal) is PEFT's,
verified here by algebraic expansion, not just copied. Let
`mag_norm_scale = magnitude / ||base_weight + scaling*(lora_B @
lora_A)||_c` (detached from the gradient graph -- DoRA paper, section
4.3, quoted directly in PEFT's own comment: "we suggest treating
||V+deltaV||_c ... as a constant, thereby detaching it from the
gradient graph"). PEFT computes
`(mag_norm_scale - 1) * base_result_no_bias + mag_norm_scale *
lora_result * scaling` as "the extra output to add on top of the base
layer's own output." Adding base_result to that expands to exactly
`mag_norm_scale * (base_result_no_bias + lora_result * scaling) + bias`
-- the textbook DoRA formula, restructured two ways: bias is never
itself magnitude-scaled (bias isn't part of the weight being decomposed
at all -- a real, easy-to-get-wrong detail if implementing from the
paper's weight-only equations directly), and base_result is reused
rather than recomputed.

**Real, deliberate extension beyond PEFT, needed for this project
specifically: the LoRA timestep gate** (core.lora.py's
set_lora_gate()/compute_lora_gate(), docs/suspicious_findings.md's
"Pending user testing" entry). PEFT has no such concept -- LLM
fine-tuning has no analogous "only some timesteps were in the training
data" idea. Applied here to the *entire* DoRA delta (both the
magnitude-reweighted base contribution and the magnitude-reweighted LoRA
contribution), matching LoRALinear's own gate semantics exactly: gate=0
must produce exactly the frozen base output, nothing else -- which for
DoRA means gating result_dora as a whole, not just the raw LoRA term
inside it. mag_norm_scale itself carries trainable state (magnitude),
and it has to vanish outside the trained t-range for the same reason the
LoRA delta does, or DoRA's magnitude parameter would keep influencing
generation at timesteps this training run never supervised -- exactly
the failure mode the gate exists to prevent for plain LoRA. The actual
gate application (`apply_lora_gate()`) lives in `lora_gate.py` now --
extracted once `nf4_lora_layer.py` needed the identical logic too.

**Built via composition over a real core.lora.LoRALinear/LoRAConv2d**
(accessed through lora_class_cache.py's _real_lora_classes(), the same
patch-immune accessor PlainLoRAAdapter uses, for the same reason:
constructing one from inside adapter_strategy_scope's own patched path
must not recurse) rather than reimplementing parameter setup (dtype
choices, kaiming init, buffer registration for the frozen base) a second
time -- only the forward math is genuinely new here, so only the forward
math is new code. Held as a real submodule (self._lora), so
DoRALinear.parameters() correctly includes lora_A/lora_B via ordinary
nn.Module recursion, plus this class's own new magnitude parameter --
and correctly excludes base_weight/base_bias (LoRALinear's own buffers,
not parameters), no extra bookkeeping needed for either.

**Checkpoint save/load: now a real round-trip for the common case,
honestly-scoped gap for one real edge case.** load_lora_weights(lora_A,
lora_B) still loads the directional component only and recomputes
magnitude fresh from it -- a real, useful thing to do (start DoRA
training from an existing plain-LoRA checkpoint's direction), but NOT a
full DoRA checkpoint round-trip, since a trained magnitude is
independent state a freshly-recomputed value can't recover.
load_dora_weights(lora_A, lora_B, magnitude) is the real round-trip, and
restore_alpha(alpha) keeps alpha/scaling consistent with a
checkpoint-restored value the same way core.lora.load_lora_into_model
does for a plain layer. nodes/model/lora_saver.py (via
nodes/model/lora_phases.py's extract_combined_weights/
extract_own_generation_weights) and LoRACheckpointLoaderNode (via its
own _load_dora_layers()) both know about `.dora_scale` now -- key name
matches ComfyUI's own comfy/lora.py convention (`{key}.dora_scale`,
read alongside `.alpha`), not invented here. DoRAAdapter is trainable in
a real run today, and saving/loading that training's real result
(direction + magnitude + alpha) correctly is now wired for an unsplit
DoRA layer, the overwhelmingly common case.

The one real edge case still open, by design rather than oversight: a
DoRA layer that's been phase-split (nodes/model/lora_phases.py's
LoRAPhaseSplitNode) can't have its magnitude folded into a combined,
multi-generation checkpoint -- magnitude scales the *entire*
frozen-base-plus-delta result (see this docstring's efficient-forward
section above), not expressible as "one more rank-stacked generation"
the way a plain LoRAGeneration's own delta is. extract_combined_weights
raises a clear error for this case rather than silently emitting a
checkpoint that quietly drops the trained magnitude's effect -- see
that function's own docstring. extract_own_generation_weights (the
"just this phase" snapshot LoRAPhaseSplitNode's completed_generation
output actually uses) has no such limitation and round-trips magnitude
fine either way, since it never combines. How phase-splitting and a
DoRA base's magnitude should even combine, if at all, is a real,
separate design question -- not attempted here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_class_cache import _real_lora_classes
from .lora_gate import apply_lora_gate


def _weight_norm_linear(base_weight: torch.Tensor, lora_A: torch.Tensor,
                         lora_B: torch.Tensor, scaling: float) -> torch.Tensor:
    """||base_weight + scaling * (lora_B @ lora_A)||, one scalar per
    output row (dim=1 on a [out_features, in_features] matrix) -- see
    this module's docstring for the PEFT/torchtune grounding. Always
    computed fresh, never cached: the merged matrix is only ever needed
    for this one reduction and is never itself kept around, matching
    PEFT's own choice to pay a full-size intermediate rather than
    persist a merged weight."""
    merged = base_weight.to(lora_A.dtype) + scaling * (lora_B @ lora_A)
    return torch.linalg.norm(merged, dim=1)


def _weight_norm_conv2d(base_weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor,
                         scaling: float, out_channels: int, rank: int) -> torch.Tensor:
    """Same idea as _weight_norm_linear, generalized to Conv2d's 4D
    weight -- norm over every dim except dim 0 (out_channels), matching
    PEFT's _DoraConvNdLayer.get_weight_norm exactly (dim=(1,2,3) on a
    [out_channels, in_channels/groups, kh, kw] weight)."""
    re_A = lora_A.reshape(rank, -1)
    re_B = lora_B.reshape(out_channels, rank)
    delta = (re_B @ re_A).view(base_weight.shape)
    merged = base_weight.to(lora_A.dtype) + scaling * delta
    dims = tuple(range(1, merged.dim()))
    return merged.norm(p=2, dim=dims)





class DoRALinear(nn.Module):
    """See this module's docstring for the full derivation and grounding."""

    def __init__(self, original: nn.Linear, rank: int = 64, alpha: float = 1.0,
                 dropout: float = 0.0, weight: float = 1.0):
        super().__init__()
        LoRALinear, _ = _real_lora_classes()
        self._lora = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout, weight=weight)
        self.in_features = self._lora.in_features
        self.out_features = self._lora.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = self._lora.scaling

        with torch.no_grad():
            init_norm = _weight_norm_linear(
                self._lora.base_weight, self._lora.lora_A, self._lora.lora_B, self.scaling)
        # fp32, matching lora_A/lora_B's own dtype -- see LoRALinear.__init__'s
        # comment on why (bf16 rounds small updates away entirely at this lr
        # scale). magnitude is exactly this kind of small-update-sensitive
        # trainable parameter.
        self.magnitude = nn.Parameter(init_norm.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = self._lora
        base_result = F.linear(x, lora.base_weight, lora.base_bias)

        x_dropped = lora.dropout(x).to(lora.lora_A.dtype)
        lora_raw = (x_dropped @ lora.lora_A.T) @ lora.lora_B.T  # not yet scaled

        weight_norm = _weight_norm_linear(
            lora.base_weight, lora.lora_A, lora.lora_B, self.scaling
        ).detach()  # section 4.3 of the DoRA paper -- see this module's docstring
        mag_norm_scale = (self.magnitude / weight_norm).to(x.dtype).view(1, -1)

        base_result_no_bias = base_result
        if lora.base_bias is not None:
            base_result_no_bias = base_result - lora.base_bias

        result_dora = ((mag_norm_scale - 1) * base_result_no_bias
                        + mag_norm_scale * lora_raw.to(base_result.dtype) * self.scaling)
        result_dora = apply_lora_gate(result_dora)
        return base_result + result_dora

    def merge(self):
        raise NotImplementedError(
            "DoRALinear.merge() -- merging magnitude+direction back into a single "
            "weight for inference isn't implemented yet. Train unmerged; see "
            "docs/training_pipeline_design.md section 3.1."
        )

    def get_lora_weights(self):
        return self._lora.lora_A, self._lora.lora_B

    def load_lora_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """Loads the directional component only, recomputing magnitude
        fresh from it -- see this module's docstring for exactly what
        this does and doesn't recover. Use load_dora_weights() for a
        real DoRA checkpoint round-trip."""
        self._lora.load_lora_weights(lora_A, lora_B)
        with torch.no_grad():
            self.magnitude.copy_(_weight_norm_linear(
                self._lora.base_weight, self._lora.lora_A, self._lora.lora_B, self.scaling
            ).to(self.magnitude.dtype))

    def load_dora_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor,
                           magnitude: torch.Tensor):
        """The real round-trip -- direction and the independently-trained
        magnitude both restored, not the latter recomputed."""
        self._lora.load_lora_weights(lora_A, lora_B)
        with torch.no_grad():
            self.magnitude.copy_(magnitude.to(device=self.magnitude.device,
                                               dtype=self.magnitude.dtype))

    def restore_alpha(self, alpha: float) -> None:
        """Update alpha/scaling to a checkpoint-restored value -- same
        formula core.lora.load_lora_into_model uses for a plain
        LoRALinear/LoRAConv2d, kept as this class's own method rather
        than something outside it recomputing the formula itself, since
        it needs self._lora.training_weight and has to land in
        self.scaling specifically: forward() reads self.scaling, not
        self._lora.scaling (this class's own copy, set once at __init__
        -- see this module's docstring)."""
        self.alpha = alpha
        self.scaling = (alpha / self.rank) * self._lora.training_weight


class DoRAConv2d(nn.Module):
    """See this module's docstring for the full derivation and grounding."""

    def __init__(self, original: nn.Conv2d, rank: int = 64, alpha: float = 1.0,
                 dropout: float = 0.0, weight: float = 1.0):
        super().__init__()
        _, LoRAConv2d = _real_lora_classes()
        self._lora = LoRAConv2d(original, rank=rank, alpha=alpha, dropout=dropout, weight=weight)
        self.in_channels = self._lora.in_channels
        self.out_channels = self._lora.out_channels
        self.kernel_size = self._lora.kernel_size
        self.stride = self._lora.stride
        self.padding = self._lora.padding
        self.dilation = self._lora.dilation
        self.groups = self._lora.groups
        self.rank = rank
        self.alpha = alpha
        self.scaling = self._lora.scaling

        with torch.no_grad():
            init_norm = _weight_norm_conv2d(
                self._lora.base_weight, self._lora.lora_A, self._lora.lora_B,
                self.scaling, self.out_channels, self.rank)
        self.magnitude = nn.Parameter(init_norm.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = self._lora
        base_result = F.conv2d(x, lora.base_weight, lora.base_bias, lora.stride,
                                lora.padding, lora.dilation, lora.groups)

        x_dropped = lora.dropout(x).to(lora.lora_A.dtype)
        adapter = F.conv2d(x_dropped, lora.lora_A, None, lora.stride,
                            lora.padding, lora.dilation, lora.groups)
        lora_raw = F.conv2d(adapter, lora.lora_B)  # not yet scaled

        weight_norm = _weight_norm_conv2d(
            lora.base_weight, lora.lora_A, lora.lora_B, self.scaling,
            self.out_channels, self.rank
        ).detach()  # section 4.3 of the DoRA paper -- see this module's docstring
        # weight_norm: [out_channels] (no keepdim -- see _weight_norm_conv2d) --
        # reshape to (1, out_channels, 1, 1) to broadcast against a conv's
        # NCHW output.
        mag_norm_scale = (self.magnitude / weight_norm).to(x.dtype).view(1, -1, 1, 1)

        base_result_no_bias = base_result
        if lora.base_bias is not None:
            base_result_no_bias = base_result - lora.base_bias.view(1, -1, 1, 1)

        result_dora = ((mag_norm_scale - 1) * base_result_no_bias
                        + mag_norm_scale * lora_raw.to(base_result.dtype) * self.scaling)
        result_dora = apply_lora_gate(result_dora)
        return base_result + result_dora

    def merge(self):
        raise NotImplementedError(
            "DoRAConv2d.merge() -- merging magnitude+direction back into a single "
            "weight for inference isn't implemented yet. Train unmerged; see "
            "docs/training_pipeline_design.md section 3.1."
        )

    def get_lora_weights(self):
        return self._lora.lora_A, self._lora.lora_B

    def load_lora_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """See DoRALinear.load_lora_weights's docstring -- same caveat."""
        self._lora.load_lora_weights(lora_A, lora_B)
        with torch.no_grad():
            self.magnitude.copy_(_weight_norm_conv2d(
                self._lora.base_weight, self._lora.lora_A, self._lora.lora_B,
                self.scaling, self.out_channels, self.rank
            ).to(self.magnitude.dtype))

    def load_dora_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor,
                           magnitude: torch.Tensor):
        """See DoRALinear.load_dora_weights's docstring -- the real
        round-trip."""
        self._lora.load_lora_weights(lora_A, lora_B)
        with torch.no_grad():
            self.magnitude.copy_(magnitude.to(device=self.magnitude.device,
                                               dtype=self.magnitude.dtype))

    def restore_alpha(self, alpha: float) -> None:
        """See DoRALinear.restore_alpha's docstring -- identical
        reasoning."""
        self.alpha = alpha
        self.scaling = (alpha / self.rank) * self._lora.training_weight
