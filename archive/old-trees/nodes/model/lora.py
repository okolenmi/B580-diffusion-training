"""LoRA (Low-Rank Adaptation): the adapter layers, the injection walk,
and the timestep gate.

**Moved here from `archive/core/lora.py` on 2026-10-02.** The implementation is
unchanged -- only the location, and the two intra-package imports below
(`archive/core/clip_encode.py`-style relative imports had to become
`nodes.model.*`). What moved is not incidental: `nodes/model/` had been
*patching this module's `LoRALinear`/`LoRAConv2d` names in place* to
substitute its own DoRA and NF4 layers (see `adapter_injection.py`),
because `_inject_lora` resolves them as bare module-level names at call
time. Owning the file makes that patch unnecessary -- `_inject_lora`
now takes the layer classes it builds as an explicit argument. See its
`layer_classes` parameter for what that replaced.

`archive/core/lora.py` remains as a re-export shim: `core/`'s trainer, its two
cache builders, `manager/builder.py`, and `archive/core/unet_wrapper.py`'s own
imports all still resolve through it.

The comments throughout are the originals and carry the reasoning that
is easy to lose -- the fp32-adapter derivation (why a bf16 adapter
rounds to a standstill), base weights as buffers rather than
attributes, and the gate's signed-distance formulation and its naming
rationale. They are worth reading before changing any of it."""

import math
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep-conditional gating.
#
# Motivation: if training data only covers some sub-range of the noise
# schedule, timesteps outside that range never get supervised, and there's
# nothing structurally stopping the LoRA delta from drifting there anyway
# (shared weights, optimizer momentum, etc. don't respect "this timestep
# wasn't sampled"). Gating scales the LoRA contribution toward ~1 for
# timesteps inside [gate_train_low, gate_train_high] -- your dataset's actual
# t range -- and toward ~0 outside it, so timesteps you aren't training on
# stay close to the frozen base model instead of drifting, by construction,
# not by hoping nothing touches them.
#
# Naming matches this codebase's existing t_low/t_high convention (see
# CommonSettings.t_low/t_high, manager/builder.py's t_low/t_high) rather than
# a "protect" framing: "the range I'm training on" is the number you already
# know (it's your dataset's t bounds), whereas "the range to protect" needs
# an extra mental inversion step every time -- which is exactly the mistake
# that happened twice in a row building this feature. Matching the
# already-established convention removes that failure mode instead of
# documenting around it.
#
# Set via set_lora_gate() once per training step (all LoRALinear instances
# in the same forward pass share the same gate, since they're processing the
# same batch of samples) -- not per-layer, since there's no reason for it to
# differ across layers.
# ---------------------------------------------------------------------------

_current_gate: Optional[torch.Tensor] = None


def set_lora_gate(gate: Optional[torch.Tensor]):
    """Set the current per-sample gate (shape (B,), or None to disable
    gating entirely -- every LoRALinear.forward() call will multiply its
    delta by this until it's changed again."""
    global _current_gate
    _current_gate = gate


@contextlib.contextmanager
def lora_gate_override(gate: Optional[torch.Tensor]):
    """Temporarily force the LoRA gate to `gate` for the duration of the
    `with` block, restoring whatever gate was active before on exit (even if
    the block raises). Safe to call on any model, including one with zero
    LoRALinear layers -- the gate is simply unused in that case.

    Primary use: forcing gate=0 during teacher-trajectory generation when
    the "teacher" is the same live model object as the student (LoRA mode
    with no separate base checkpoint) -- gate=0 makes LoRALinear.forward()
    produce exactly base-model output (result + 0), so the same weights
    serve as both teacher and student without a second model in VRAM.
    Nests correctly, so student-forward (real gate) and teacher-forward
    (gate=0) calls can be interleaved within the same rollout (e.g. DAgger
    chain-mixing) without one leaking into the other.
    """
    global _current_gate
    prev = _current_gate
    _current_gate = gate
    try:
        yield
    finally:
        _current_gate = prev


def compute_lora_gate(t: torch.Tensor, train_low: float, train_high: float,
                       width: float) -> torch.Tensor:
    """Smooth gate: ~1 (train normally) for t inside [train_low, train_high]
    -- your dataset's actual t range -- fading to ~0 (protect the frozen
    base weights) the further t is outside that range on either side.
    `width` controls how sharp the transition is at both edges (smaller =
    sharper cutoff right at the boundary, larger = more gradual handoff).

    Worked example matching a real case: dataset covers t=120-450.
    compute_lora_gate(t, train_low=120, train_high=450, width=40) gives
    gate≈1 for t=120-450 (full LoRA training, this IS your data),
    gate≈0.5 exactly at t=120 and t=450 (the edges of your data),
    gate≈0 for t well below 120 or well above 450 (protected -- your
    dataset has nothing to say about those timesteps).

    t: (B,) tensor of per-sample timesteps (whatever scale/convention the
    rest of the pipeline already uses for t).
    """
    t = t.float()
    below = train_low - t   # positive when t is below the training range
    above = t - train_high  # positive when t is above the training range
    inside_depth = torch.minimum(t - train_low, train_high - t)  # positive when inside
    # Signed distance from the training interval: negative (how far outside)
    # when t is outside, positive (how deep) when t is inside, ~0 at an edge.
    dist = torch.where(t < train_low, -below,
                        torch.where(t > train_high, -above, inside_depth))
    return torch.sigmoid(dist / max(width, 1e-6))


#: Named target sets for LoRAConfig.target_modules.
#: ATTENTION is this project's historical default. KOHYA_DEFAULT matches what
#: kohya sd-scripts' default SDXL LoRA adapts (every Linear inside each
#: Transformer2DModel): 722 modules on SDXL. KOHYA_PLUS is the same UNet set
#: with the conditioning-path embeddings kept (target_conditioning_path=True,
#: the default): 722 + time_embed/label_emb = 726 modules on SDXL.
TARGETS_ATTENTION = ["to_q", "to_k", "to_v", "to_out.0"]
TARGETS_KOHYA_DEFAULT = TARGETS_ATTENTION + ["ff.net.0.proj", "ff.net.2", "proj_in", "proj_out"]
TARGETS_KOHYA_PLUS = list(TARGETS_KOHYA_DEFAULT)
#: The 722 set plus `label_emb` alone (724 modules with the conditioning
#: path off). The B study's arm 6 -- constant LR, rank 64/alpha 48,
#: test2 -- rendered best of six arms of the maintainer's own judgement
#: (backgrounds most detailed, dataset vibe best represented), and the
#: isolation reads: label_emb helps (724 > 722), time_embed hurts
#: (726 < 724 with label held constant). Because it is the *default for
#: new runs* (C3, 2026-10-11), it is a named preset rather than the
#: comma-joined "kohya_default,label_emb" string it started as.
TARGETS_KOHYA_LABEL = TARGETS_KOHYA_DEFAULT + ["label_emb"]


@dataclass
class LoRAConfig:
    rank: int = 64
    alpha: float = 1.0
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    block_weights: Optional[Dict[str, float]] = None
    target_all: bool = False
    # time_embed.* and label_emb.* sit on the global conditioning path
    # (emb = time_embed(t) + label_emb(y) is added into EVERY ResBlock).
    # True keeps the historical behaviour (they are always adapted, 564 modules
    # on SDXL). False restricts adaptation to attention like kohya / diffusers
    # LoRA defaults (560 modules). Added as an A/B switch for quality debugging.
    target_conditioning_path: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = list(TARGETS_ATTENTION)
        elif isinstance(self.target_modules, str):
            # Convenience for graph/UI ports, which carry strings: a preset name
            # ("attention", "kohya_default", "kohya_plus", "kohya_label"), a
            # comma-separated pattern list, or preset+extras joined by ","
            # ("kohya_default,label_emb" = the 722 set plus label_emb alone =
            # 724, the time-vs-label isolation form behind "kohya_label").
            # "kohya_plus" resolves to the same UNet patterns as
            # "kohya_default"; the "+" is target_conditioning_path keeping
            # time_embed.*/label_emb.*: 726 modules on SDXL instead of 722.
            presets = {"attention": TARGETS_ATTENTION, "kohya_default": TARGETS_KOHYA_DEFAULT,
                       "kohya_plus": TARGETS_KOHYA_PLUS,
                       "kohya_label": TARGETS_KOHYA_LABEL}
            raw = self.target_modules.strip()
            if raw in presets:
                self.target_modules = list(presets[raw])
            else:
                expanded: list[str] = []
                for tok in (t.strip() for t in raw.split(",") if t.strip()):
                    if tok in presets:
                        expanded.extend(presets[tok])
                    else:
                        expanded.append(tok)
                self.target_modules = expanded
            if not self.target_modules:
                raise ValueError("target_modules string selected no patterns")


# ComfyUI SDXL UNet attention structure:
#   attn1 / attn2 (CrossAttention)
#     ├── to_q:   Linear(query_dim, inner_dim)
#     ├── to_k:   Linear(context_dim, inner_dim)
#     ├── to_v:   Linear(context_dim, inner_dim)
#     └── to_out: Sequential
#           ├── 0: Linear(inner_dim, query_dim)   ← Linear we wrap
#           └── 1: Dropout
#
# The hierarchy inside UNetModel:
#   input_blocks.N.M.transformer_blocks.K.attn1/2.to_q/k/v
#   input_blocks.N.M.transformer_blocks.K.attn1/2.to_out.0
#   middle_block.M.transformer_blocks.K.attn1/2.to_q/k/v
#   middle_block.M.transformer_blocks.K.attn1/2.to_out.0
#   output_blocks.N.M.transformer_blocks.K.attn1/2.to_q/k/v
#   output_blocks.N.M.transformer_blocks.K.attn1/2.to_out.0


class LoRALinear(nn.Module):
    """A frozen linear layer with low-rank trainable adapters (A and B matrices).

    Design notes
    ------------
    The frozen base weight and bias are stored as **buffers** (not parameters
    and not plain tensor attributes).  This means:

    * ``module.parameters()`` returns ONLY lora_A and lora_B — iterating
      the full model for weight-tracking or optimizer construction never
      accidentally includes frozen base weights.
    * ``module.state_dict()`` still includes them under ``base_weight`` /
      ``base_bias`` so checkpoints can be saved and loaded correctly.
    * No extra memory is used — buffers are views of the original tensors.
    """

    def __init__(self, original: nn.Linear, rank: int = 64, alpha: float = 1.0,
                 dropout: float = 0.0, weight: float = 1.0):
        super().__init__()
        self.in_features  = original.in_features
        self.out_features = original.out_features
        self.rank    = rank
        self.alpha   = alpha
        self.scaling = (alpha / rank) * weight
        self.training_weight = weight

        # Store frozen tensors as buffers so they appear in state_dict() but
        # NOT in parameters().  This is the critical fix: the old code assigned
        # them as plain attributes which caused them to leak into parameters()
        # in some PyTorch versions / ComfyUI patching scenarios.
        self.register_buffer("base_weight", original.weight.detach())
        if original.bias is not None:
            self.register_buffer("base_bias", original.bias.detach())
        else:
            self.register_buffer("base_bias", None)

        device = self.base_weight.device

        # CRITICAL: lora_A/lora_B are always fp32, regardless of the frozen base
        # model's dtype (bf16). This is not cosmetic.
        #
        # Adafactor's normalized update is ~O(lr) per step, essentially
        # independent of the raw gradient's magnitude (it divides by the
        # running RMS of the gradient itself). lr here is typically 1e-5-1e-4.
        # bf16 only has ~7-8 bits of mantissa (~0.4-0.8% relative precision), so
        # at a parameter's natural magnitude (e.g. ~0.02-0.1 for a
        # kaiming-initialized lora_A), an update smaller than that ~0.4-0.8%
        # threshold rounds straight back to the unchanged value -- every single
        # step, forever. Verified numerically: a bf16 parameter at realistic
        # init magnitude receiving 2000 consecutive lr=1e-5 updates ends up
        # *bit-for-bit unchanged*; at lr=1e-4, ~73% of steps are silently
        # dropped. Only once lr is cranked far past the parameter's own scale
        # (e.g. 1e-2) do updates clear the rounding floor -- but by then each
        # step is comparable in size to the whole weight, so instead of
        # gradient descent you get near-random overwriting, i.e. exactly the
        # "no effect, then destructive" behavior this was chasing.
        #
        # This is why every mainstream LoRA implementation (HF PEFT, diffusers)
        # keeps trainable adapter weights in fp32 even when the frozen base
        # model is fp16/bf16, casting down only for the forward matmul. A/B are
        # tiny (rank x dim), so the extra memory here is negligible.
        param_dtype = torch.float32

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features,  device=device, dtype=param_dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=param_dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero so the adapter is a no-op at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base_weight is a buffer (not a Parameter) so no gradient accumulates on
        # it — no_grad is unnecessary here and would wrongly block the gradient
        # path through x back to earlier layers, leaving only the rank-limited
        # LoRA branch to propagate gradients upstream.
        result = F.linear(x, self.base_weight, self.base_bias)

        # LoRA branch computed in fp32 (lora_A/lora_B's own dtype -- see
        # __init__) regardless of x's dtype (bf16), then cast back down to
        # match `result` before adding, so this layer's output dtype is
        # unchanged from the caller's point of view.
        #
        # Optimization: Multiply the small lora_B by scaling BEFORE the matrix
        # multiplication. This saves one full-size tensor multiplication kernel
        # launch and millions of ops on large images.
        lora_out = (self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T) @ (self.lora_B.T * self.scaling)

        gate = _current_gate
        if gate is not None:
            # Reshape (B,) -> (B, 1, ..., 1) to match lora_out's actual rank,
            # so this works whether x is (B, C) or (B, N, C) without needing
            # to know which kind of layer this is.
            g = gate.to(device=lora_out.device, dtype=lora_out.dtype)
            g = g.view(-1, *([1] * (lora_out.dim() - 1)))
            lora_out = lora_out * g

        return result + lora_out.to(result.dtype)

    def merge(self):
        """Merge adapter weights into base weight in-place (for inference)."""
        delta   = (self.lora_B @ self.lora_A) * self.scaling
        merged  = self.base_weight + delta.to(self.base_weight.dtype)
        self.base_weight.copy_(merged)
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def get_lora_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lora_A, self.lora_B

    def load_lora_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        if lora_A.shape != self.lora_A.shape:
            # Handle standard LoRA (2816) being loaded into CFG-aware model (3072)
            if lora_A.shape[1] == 2816 and self.lora_A.shape[1] == 3072:
                print(f"[WARNING] Loading legacy 2816-dim LoRA into 3072-dim model; "
                      f"zero-padding the extra 256 dims. Verify this checkpoint is meant "
                      f"for this model config.")
                padding = torch.zeros((lora_A.shape[0], 256), device=lora_A.device, dtype=lora_A.dtype)
                lora_A = torch.cat([lora_A, padding], dim=1)
            else:
                assert lora_A.shape == self.lora_A.shape, \
                    f"A shape mismatch: {lora_A.shape} vs {self.lora_A.shape}"
        
        assert lora_B.shape == self.lora_B.shape, \
            f"B shape mismatch: {lora_B.shape} vs {self.lora_B.shape}"
        
        self.lora_A.data.copy_(lora_A.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
        self.lora_B.data.copy_(lora_B.to(device=self.lora_B.device, dtype=self.lora_B.dtype))


class GroupedLoRALinear(nn.Module):
    """Fuses multiple LoRA linear projections into a single batch operation.
    
    Used to mitigate CPU bottleneck in 'Full LoRA' mode.
    """
    def __init__(self, layers: List[Tuple[str, nn.Module, str, LoRALinear]]):
        super().__init__()
        self.layer_metadata = []
        
        # Collect info
        for full_name, model, name, lora in layers:
            self.layer_metadata.append({
                "full_name": full_name,
                "model": model,
                "attr_name": name,
                "lora": lora,
                "in_features": lora.in_features,
                "out_features": lora.out_features,
            })
            # Remove individual lora layers from their parents
            # and replace them with a proxy that calls this group?
            # Actually, it's easier to just use the registry for the forward pass
            # if we are doing manual grouping. But for standard training,
            # we want the model to stay transparent.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation of grouped forward...
        # For now, this is a placeholder for future optimization.
        # True grouping requires rewriting the UNet forward pass or using hooks.
        pass


class LoRAConv2d(nn.Module):
    """A frozen conv2d layer with low-rank trainable adapters."""

    def __init__(self, original: nn.Conv2d, rank: int = 64, alpha: float = 1.0,
                 dropout: float = 0.0, weight: float = 1.0):
        super().__init__()
        self.in_channels  = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size  = original.kernel_size
        self.stride       = original.stride
        self.padding      = original.padding
        self.dilation     = original.dilation
        self.groups       = original.groups
        self.rank         = rank
        self.alpha        = alpha
        self.scaling      = (alpha / rank) * weight
        self.training_weight = weight

        self.register_buffer("base_weight", original.weight.detach())
        if original.bias is not None:
            self.register_buffer("base_bias", original.bias.detach())
        else:
            self.register_buffer("base_bias", None)

        device = self.base_weight.device

        # fp32, not base_weight.dtype (bf16) -- see the matching comment in
        # LoRALinear.__init__. Same rounding-to-a-standstill failure mode
        # applies here.
        param_dtype = torch.float32

        # A: (rank, in_channels * k * k), B: (out_channels, rank * 1 * 1)
        # For 1x1 conv, it's just like Linear. For 3x3, it's more complex but
        # Kohya/ComfyUI convention for LoRA-Conv2d is often to use
        # (rank, in_channels, k, k) and (out_channels, rank, 1, 1).
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_channels // self.groups, 
                                               *self.kernel_size, device=device, dtype=param_dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank, 1, 1, 
                                               device=device, dtype=param_dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.conv2d(x, self.base_weight, self.base_bias, self.stride,
                          self.padding, self.dilation, self.groups)
        
        # LoRA branch: (x * lora_A) * lora_B, computed in fp32 (lora_A/B's own
        # dtype) then cast back down before adding -- see LoRALinear.forward.
        x_in = self.dropout(x).to(self.lora_A.dtype)
        adapter = F.conv2d(x_in, self.lora_A, None, self.stride,
                           self.padding, self.dilation, self.groups)
        
        # Optimization: Multiply the small lora_B by scaling BEFORE the conv2d.
        # This saves one full-size tensor multiplication kernel launch.
        adapter = F.conv2d(adapter, self.lora_B * self.scaling)
        
        return result + adapter.to(result.dtype)

    def merge(self):
        """Merge adapter weights into base weight in-place."""
        # lora_A: (rank, in_C/groups, k, k)
        # lora_B: (out_C, rank, 1, 1)
        # We want delta: (out_C, in_C/groups, k, k)
        re_A = self.lora_A.view(self.rank, -1)
        re_B = self.lora_B.view(self.out_channels, self.rank)
        delta = (re_B @ re_A).view(self.out_channels, self.in_channels // self.groups, 
                                   *self.kernel_size)
        
        merged = self.base_weight + (delta * self.scaling).to(self.base_weight.dtype)
        self.base_weight.copy_(merged)
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def get_lora_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lora_A, self.lora_B

    def load_lora_weights(self, lora_A: torch.Tensor, lora_B: torch.Tensor):
        assert lora_A.shape == self.lora_A.shape, \
            f"A shape mismatch: {lora_A.shape} vs {self.lora_A.shape}"
        assert lora_B.shape == self.lora_B.shape, \
            f"B shape mismatch: {lora_B.shape} vs {self.lora_B.shape}"
        self.lora_A.data.copy_(lora_A.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
        self.lora_B.data.copy_(lora_B.to(device=self.lora_B.device, dtype=self.lora_B.dtype))


# ---------------------------------------------------------------------------
# Key naming — matches ComfyUI's model_lora_keys_unet / Kohya convention
#   state_dict key:        model.diffusion_model.input_blocks.3.1.transformer_blocks.0.attn1.to_q.weight
#   LoRA key:              lora_unet_input_blocks_3_1_transformer_blocks_0_attn1_to_q
#   Saved tensor keys:     lora_unet_....lora_down.weight  (A: rank x in_dim)
#                          lora_unet_....lora_up.weight    (B: out_dim x rank)
#                          lora_unet_....alpha             (scalar)
# ---------------------------------------------------------------------------

_LORA_PREFIX = "lora_unet_"
_DOWN_SUFFIX = ".lora_down.weight"
_UP_SUFFIX   = ".lora_up.weight"
_ALPHA_SUFFIX = ".alpha"


def _key_to_lora_key(full_module_name: str) -> str:
    """Convert a full module name to the LoRA key prefix."""
    path = full_module_name
    if path.startswith("model.diffusion_model."):
        path = path[len("model.diffusion_model."):]
    path = path.replace(".", "_")
    return f"{_LORA_PREFIX}{path}"


def _segment_match(full_name: str, block_id: str) -> bool:
    """True if full_name equals block_id or starts with block_id + '.'

    This is a segment-boundary check so 'input_blocks.1' cannot match
    'input_blocks.10' or 'input_blocks.3.1'.
    """
    return full_name == block_id or full_name.startswith(block_id + ".")


def parse_block_weights(spec: str | dict | None) -> dict[str, float] | None:
    """The `block_weighting` config string -> `LoRAConfig.block_weights`.

    Accepts `"input_blocks.0:0.5,middle_block:1.0"` (comma-separated
    `block_id:weight` pairs; `block_id` is any UNet module-path prefix, so
    `input_blocks.3` weights everything under that block) or an already
    parsed dict, which is returned as-is. `None`/empty -> None, meaning
    "no weighting", which is what every run used before this existed.

    Returns None rather than an empty dict so callers can distinguish
    "off" from "parsed but empty", and raises ValueError on a malformed
    pair instead of silently dropping it -- a typo'd block id would
    otherwise look like it had been applied.
    """
    if spec is None or spec == "":
        return None
    if isinstance(spec, dict):
        return {str(k): float(v) for k, v in spec.items()}
    out: dict[str, float] = {}
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        block_id, sep, weight = token.rpartition(":")
        if not sep:
            raise ValueError(
                f"block_weighting: {token!r} is not 'block_id:weight' "
                f"(e.g. 'input_blocks.0:0.5,middle_block:1.0')")
        block_id = block_id.strip()
        try:
            value = float(weight)
        except ValueError:
            raise ValueError(
                f"block_weighting: {token!r} has a non-numeric weight "
                f"{weight!r}") from None
        if not block_id:
            raise ValueError(f"block_weighting: {token!r} has an empty block id")
        if value < 0:
            raise ValueError(
                f"block_weighting: {block_id} has a negative weight {value}")
        out[block_id] = value
    return out or None


#: The layers `_inject_lora` builds by default: plain LoRA on a frozen
#: bf16 base. Any (linear_cls, conv_cls) pair with the same constructor
#: signature works -- that is the seam `nodes/model/adapter_strategy.py`
#: uses to substitute DoRA and NF4 layers.
_DEFAULT_LAYER_CLASSES = (LoRALinear, LoRAConv2d)


def _inject_lora(model: nn.Module, config: LoRAConfig, prefix: str = "",
                 registry: Optional[List[Tuple[str, nn.Module, str, nn.Module]]] = None,
                 matched_block_ids: Optional[set] = None,
                 layer_classes=None,
                 ) -> List[Tuple[str, nn.Module, str, nn.Module]]:
    """Walk *model* named_children recursively.  When we find a module that
    corresponds to a target, wrap it with LoRA.

    `layer_classes`: `(linear_cls, conv_cls)` to build targets with,
    defaulting to `(LoRALinear, LoRAConv2d)`. Both take the same
    constructor arguments (`original, rank, alpha, dropout, weight`).

    This parameter used to not exist, which is why
    `nodes/model/adapter_injection.py` had to monkeypatch this module's
    own `LoRALinear`/`LoRAConv2d` names for the duration of a build: the
    walk resolved them as module-level globals, so patching them was the
    only way to change what it constructed. That worked, but it made two
    module globals the mechanism, which in turn required
    `lora_class_cache.py` to defeat the patch from inside recursive
    strategies, and left a documented race between two concurrent
    builds. Passing the classes explicitly removes the need for all three:
    a build now says what it wants rather than arranging for something
    else to look up the wrong thing."""
    if layer_classes is None:
        layer_classes = _DEFAULT_LAYER_CLASSES
    linear_cls, conv_cls = layer_classes
    if registry is None:
        registry = []
    if matched_block_ids is None:
        matched_block_ids = set()

    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, (nn.Linear, nn.Conv2d)):
            parent_name = prefix.rpartition(".")[2] if prefix else ""
            is_target = False

            # Honour config.target_modules (it used to be read nowhere: the set
            # was hard-coded to q/k/v/to_out.0). A pattern matches a layer whose
            # dotted name equals it or ends with "." + it, so the defaults
            # ("to_q", "to_k", "to_v", "to_out.0") select exactly the layers the
            # old hard-coded test did, and "ff.net.0.proj", "ff.net.2",
            # "proj_in", "proj_out" now select the feed-forward and projection
            # Linears that kohya's default SDXL LoRA also adapts (722 modules
            # total instead of 560).
            if any(full_name == pat or full_name.endswith("." + pat)
                   for pat in config.target_modules):
                is_target = True
            # Extra blocks: time_embed and label_emb use segment-boundary match.
            # Each fires on the flag OR on being named explicitly in
            # target_modules (so "kohya_default,label_emb" adapts label_emb
            # alone with the flag off -- 724 modules -- isolating label vs
            # time conditioning without a third flag).
            if not is_target:
                time_hit = (_segment_match(full_name, "time_embed")
                            or full_name.startswith("time_embed."))
                label_hit = (_segment_match(full_name, "label_emb")
                             or full_name.startswith("label_emb."))
                explicit = set(config.target_modules or ())
                if ((config.target_conditioning_path or "time_embed" in explicit)
                        and time_hit) or \
                   ((config.target_conditioning_path or "label_emb" in explicit)
                        and label_hit):
                    is_target = True
            
            # Support targeting ANY linear/conv layer inside a block that has weighting
            # ONLY if target_all is enabled.
            if config.block_weights:
                for block_id in config.block_weights:
                    if _segment_match(full_name, block_id):
                        matched_block_ids.add(block_id)
                        if config.target_all:
                            is_target = True
                        break

            if is_target:
                # Determine block weight using segment-aware prefix matching.
                block_weight = 1.0
                if config.block_weights:
                    for block_id, w in config.block_weights.items():
                        if _segment_match(full_name, block_id):
                            block_weight = w
                            break

                if block_weight <= 0:
                    continue

                if isinstance(child, nn.Linear):
                    lora_layer = linear_cls(
                        child, rank=config.rank,
                        alpha=config.alpha, dropout=config.dropout,
                        weight=block_weight,
                    )
                else: # Conv2d
                    lora_layer = conv_cls(
                        child, rank=config.rank,
                        alpha=config.alpha, dropout=config.dropout,
                        weight=block_weight,
                    )
                setattr(model, name, lora_layer)
                registry.append((full_name, model, name, lora_layer))
        else:
            _inject_lora(child, config, full_name, registry, matched_block_ids,
                         layer_classes)

    return registry


def inject_lora_into_unet(unet_model: nn.Module, config: LoRAConfig, layer_classes=None
                          ) -> List[Tuple[str, nn.Module, str, nn.Module]]:
    """Walk `unet_model` and wrap every target with an adapter layer.

    `layer_classes`: see `_inject_lora`'s own docstring. `core/`'s trainer
    calls this with no third argument and gets plain LoRA, unchanged."""
    matched_ids = set()
    registry = _inject_lora(unet_model, config, matched_block_ids=matched_ids,
                            layer_classes=layer_classes)

    # Warn about block_weight keys that matched no injected layer.
    if config.block_weights:
        unmatched = set(config.block_weights.keys()) - matched_ids
        if unmatched:
            print(f"    [LoRA] WARNING: the following block_weighting keys matched "
                  f"no injected LoRA layers (check spelling or module structure): "
                  f"{sorted(unmatched)}")

    return registry


def _is_adapter_layer(layer, layer_classes) -> bool:
    """Whether `layer` is one of the adapter classes this build injected.

    Previously each of the four helpers below gated on
    `isinstance(layer, (LoRALinear, LoRAConv2d))` against the module
    globals. Under the old monkeypatch that gate silently answered the
    wrong question: the names had been rebound to `_AdapterPatchedLayer`,
    whose `__new__` returns whatever the strategy built, so a DoRA or NF4
    layer was never an instance of it and was skipped -- the reason
    `adapter_injection.py` needs `reenable_dora_requires_grad()` and
    `dora_trainable_parameters()` to fix up what this gate let through.
    Taking the classes as an argument makes the gate state what was
    actually injected instead of what a global currently points at."""
    if layer_classes is None:
        layer_classes = _DEFAULT_LAYER_CLASSES
    return isinstance(layer, tuple(layer_classes))


def layer_training_weight(layer) -> float:
    """The block weight an adapter layer trains at, for any layer kind.

    `LoRALinear`/`LoRAConv2d` keep it as `training_weight`. A `DoRA*` layer
    wraps a LoRA layer and keeps it on `_lora`; the NF4 layers and the
    phase-split generations never stored it, only the product
    `scaling == (alpha / rank) * weight`, so it is recovered from that.
    One accessor, so save and load cannot disagree about which layers have
    the attribute (the first version read `layer.training_weight` and raised
    AttributeError on a DoRA layer)."""
    direct = getattr(layer, "training_weight", None)
    if direct is not None:
        return float(direct)
    inner = getattr(layer, "_lora", None)
    wrapped = getattr(inner, "training_weight", None)
    if wrapped is not None:
        return float(wrapped)
    alpha = getattr(layer, "alpha", 0.0)
    rank = getattr(layer, "rank", 0)
    scaling = getattr(layer, "scaling", None)
    if scaling is not None and alpha and rank:
        return float(scaling) * rank / float(alpha)
    return 1.0


def _set_layer_training_weight(layer, weight: float) -> None:
    """Counterpart of `layer_training_weight`: the weight lands where the
    layer's own code reads it (a DoRA layer's `restore_alpha` reads
    `_lora.training_weight`)."""
    layer.training_weight = weight
    inner = getattr(layer, "_lora", None)
    if inner is not None:
        inner.training_weight = weight


def extract_lora_weights(registry: List[Tuple[str, nn.Module, str, nn.Module]],
                         layer_classes=None
                         ) -> Dict[str, torch.Tensor]:
    weights = {}
    for full_name, parent, attr_name, lora_layer in registry:
        if not _is_adapter_layer(lora_layer, layer_classes):
            continue
        lora_key = _key_to_lora_key(full_name)
        A, B = lora_layer.get_lora_weights()
        weights[f"{lora_key}{_DOWN_SUFFIX}"] = A.detach().cpu().contiguous()
        weights[f"{lora_key}{_UP_SUFFIX}"]   = B.detach().cpu().contiguous()
        # EFFECTIVE alpha = alpha * block_weight, not the nominal one.
        #
        # A reader (ComfyUI, diffusers, peft -- i.e. everything outside this
        # project) reconstructs the delta as `(lora_B @ lora_A) * (alpha /
        # rank)`. Training applied `(alpha / rank) * training_weight`
        # (`LoRALinear.forward`). Writing the nominal alpha therefore made
        # every external reader apply the delta `1/training_weight` times
        # too strongly: measured exactly, a 0.5 block weight came back at
        # 2x the strength it was trained at.
        #
        # `alpha` is the one strength field every LoRA reader already
        # honours, so folding the weight into it is what makes the file
        # mean the same thing everywhere. The `.block_weight` key below is
        # kept as well -- this project's own loader needs it to restore
        # per-module weights for a *resume* (see load_lora_into_model).
        effective_alpha = lora_layer.alpha * layer_training_weight(lora_layer)
        weights[f"{lora_key}{_ALPHA_SUFFIX}"] = torch.tensor([effective_alpha],
                                                              dtype=torch.float32)
        # The block weight is part of the trained strength (forward applies
        # (alpha/rank)*training_weight), so it must round-trip like alpha:
        # without this key a resume/inference with block_weights set trains
        # at one strength and applies another. Readers that do not know the
        # key (ComfyUI, older versions of this loader) ignore it, and loads
        # of files written before it existed keep the old behaviour (config
        # weight applies) -- see load_lora_into_model.
        weights[f"{lora_key}.block_weight"] = torch.tensor(
            [layer_training_weight(lora_layer)], dtype=torch.float32)
    return weights


def load_lora_into_model(registry: List[Tuple[str, nn.Module, str, nn.Module]],
                         state_dict: Dict[str, torch.Tensor], layer_classes=None):
    for full_name, parent, attr_name, lora_layer in registry:
        if not _is_adapter_layer(lora_layer, layer_classes):
            continue
        lora_key = _key_to_lora_key(full_name)
        A_key = f"{lora_key}{_DOWN_SUFFIX}"
        B_key = f"{lora_key}{_UP_SUFFIX}"
        alpha_key = f"{lora_key}{_ALPHA_SUFFIX}"
        if A_key in state_dict and B_key in state_dict:
            lora_layer.load_lora_weights(state_dict[A_key], state_dict[B_key])
            # Restore the alpha/scaling from the checkpoint so that a run resumed
            # with a different alpha in the config still uses the trained scaling.
            block_key = f"{lora_key}.block_weight"
            saved_weight = (float(state_dict[block_key].item())
                            if block_key in state_dict else None)
            if alpha_key in state_dict:
                saved_alpha = state_dict[alpha_key].item()
                if abs(saved_alpha - lora_layer.alpha) > 1e-6:
                    print(f"    [LoRA] {full_name}: alpha mismatch "
                          f"(checkpoint={saved_alpha}, config={lora_layer.alpha}). "
                          f"Using checkpoint value.")
                lora_layer.alpha = saved_alpha
            if saved_weight is not None:
                # The file carries the EFFECTIVE alpha (already
                # alpha*weight -- see extract_lora_weights), so the weight
                # is folded into the alpha we just restored. Leaving
                # training_weight at its trained value would apply it a
                # second time (alpha*w/rank * w) and make a resume train
                # at w^2 the strength it was trained at, so it is
                # neutralised here. `saved_weight` is still recorded on the
                # layer for anything that reports the per-module weight.
                current = layer_training_weight(lora_layer)
                if abs(saved_weight - current) > 1e-6:
                    print(f"    [LoRA] {full_name}: block_weight differs "
                          f"(checkpoint={saved_weight}, config={current}); the "
                          f"file's alpha already includes it.")
                _set_layer_training_weight(lora_layer, 1.0)
            # A file that predates the `.block_weight` key carries the
            # NOMINAL alpha, so the config's weight still applies (the
            # legacy behaviour, kept so old files load unchanged).
            lora_layer.scaling = ((lora_layer.alpha / lora_layer.rank)
                                  * layer_training_weight(lora_layer))


def merge_lora_into_unet(registry: List[Tuple[str, nn.Module, str, nn.Module]],
                         layer_classes=None):
    for full_name, parent, attr_name, lora_layer in registry:
        if _is_adapter_layer(lora_layer, layer_classes):
            lora_layer.merge()


def lora_param_count(registry: List[Tuple[str, nn.Module, str, nn.Module]],
                     layer_classes=None) -> int:
    total = 0
    for _, _, _, layer in registry:
        if _is_adapter_layer(layer, layer_classes):
            total += layer.lora_A.numel() + layer.lora_B.numel()
    return total


def lora_module_group(module_key: str) -> str:
    """Group name for a LoRA module key (without the down/up/alpha suffix),
    same grouping as lora_forensics.py: time_embed, label_emb, attn_input,
    attn_middle, attn_output, other_unet, text_encoder, unknown."""
    if module_key.startswith(("lora_te", "lora_te1", "lora_te2")):
        return "text_encoder"
    if not module_key.startswith("lora_unet_"):
        return "unknown"
    p = module_key[len("lora_unet_"):]
    if p.startswith("time_embed"):
        return "time_embed"
    if p.startswith("label_emb"):
        return "label_emb"
    if "attn" in p or "transformer_blocks" in p:
        if p.startswith("input_blocks"):
            return "attn_input"
        if p.startswith("middle_block"):
            return "attn_middle"
        if p.startswith("output_blocks"):
            return "attn_output"
    return "other_unet"


def lora_health_summary(state_dict: dict) -> dict:
    """Per-group median ||dW||_F over an exported LoRA state dict (D3).

    dW = (alpha/rank) * block_weight * (B @ A), materialised directly
    (rank <= 256, tiny -- no trace trick needed); alpha defaults to rank
    and block_weight to 1.0 when their keys are absent, matching the
    loader's own defaults. Groups per lora_module_group.

    Warnings (the A1 signatures): attn_output median > 1.8x attn_input
    median (the attention-only run showed 2.0x); any group median > 5x
    the attention reference (median of the three attn group medians).
    Pure function of the state dict -- no model needed, so the saver and
    offline inspection share it. Unit-tested on synthetic data.
    """
    import statistics
    by_group: dict[str, list] = {}
    for key, down in state_dict.items():
        if not key.endswith(_DOWN_SUFFIX):
            continue
        mod = key[: -len(_DOWN_SUFFIX)]
        up = state_dict.get(mod + _UP_SUFFIX)
        if up is None:
            continue
        rank = down.shape[0]
        alpha = state_dict.get(mod + _ALPHA_SUFFIX)
        alpha = float(alpha.reshape(-1)[0]) if alpha is not None else float(rank)
        bw = state_dict.get(mod + ".block_weight")
        bw = float(bw.reshape(-1)[0]) if bw is not None else 1.0
        with torch.no_grad():
            # Conv adapters carry 4-D tensors (down: rank,in,kh,kw; up:
            # out,rank,1,1); the delta's norm is that of the flattened
            # product, and `@` on the 4-D tensors raised, which skipped the
            # whole summary for any Conv2d LoRA (target_all / block weights).
            dw = ((alpha / rank) * bw
                  * (up.float().reshape(up.shape[0], -1)
                     @ down.float().reshape(rank, -1))).norm().item()
        by_group.setdefault(lora_module_group(mod), []).append(dw)
    groups = {g: {"n": len(v), "median_dW": statistics.median(v)}
              for g, v in sorted(by_group.items()) if v}
    warnings = []
    att = [groups[g]["median_dW"] for g in
           ("attn_input", "attn_middle", "attn_output") if g in groups]
    if att:
        ref = statistics.median(att)
        if ("attn_input" in groups and "attn_output" in groups and ref > 0
                and groups["attn_output"]["median_dW"]
                / max(groups["attn_input"]["median_dW"], 1e-12) > 1.8):
            warnings.append(
                "attn_output median is >1.8x attn_input: the output "
                "projections carry the adaptation (attention-only signature)")
        for g, s in groups.items():
            if g not in ("attn_input", "attn_middle", "attn_output") \
                    and ref > 0 and s["median_dW"] > 5.0 * ref:
                warnings.append(
                    f"{g} median is >5x the attention reference: check what "
                    f"the LoRA put on the global conditioning path")
    return {"groups": groups, "warnings": warnings}
