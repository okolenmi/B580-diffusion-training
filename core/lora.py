"""LoRA (Low-Rank Adaptation) module for injecting trainable adapters into UNet."""

import math
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


@dataclass
class LoRAConfig:
    rank: int = 64
    alpha: float = 1.0
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    block_weights: Optional[Dict[str, float]] = None
    target_all: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]


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

        dtype  = self.base_weight.dtype
        device = self.base_weight.device

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features,  device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero so the adapter is a no-op at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base_weight is a buffer (not a Parameter) so no gradient accumulates on
        # it — no_grad is unnecessary here and would wrongly block the gradient
        # path through x back to earlier layers, leaving only the rank-limited
        # LoRA branch to propagate gradients upstream.
        result = F.linear(x, self.base_weight, self.base_bias)
        
        # Optimization: Multiply the small lora_B by scaling BEFORE the matrix
        # multiplication. This saves one full-size tensor multiplication kernel
        # launch and millions of ops on large images.
        lora_out = (self.dropout(x) @ self.lora_A.T) @ (self.lora_B.T * self.scaling)

        gate = _current_gate
        if gate is not None:
            # Reshape (B,) -> (B, 1, ..., 1) to match lora_out's actual rank,
            # so this works whether x is (B, C) or (B, N, C) without needing
            # to know which kind of layer this is.
            g = gate.to(device=lora_out.device, dtype=lora_out.dtype)
            g = g.view(-1, *([1] * (lora_out.dim() - 1)))
            lora_out = lora_out * g

        return result + lora_out

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

        dtype  = self.base_weight.dtype
        device = self.base_weight.device

        # A: (rank, in_channels * k * k), B: (out_channels, rank * 1 * 1)
        # For 1x1 conv, it's just like Linear. For 3x3, it's more complex but
        # Kohya/ComfyUI convention for LoRA-Conv2d is often to use
        # (rank, in_channels, k, k) and (out_channels, rank, 1, 1).
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_channels // self.groups, 
                                               *self.kernel_size, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank, 1, 1, 
                                               device=device, dtype=dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.conv2d(x, self.base_weight, self.base_bias, self.stride,
                          self.padding, self.dilation, self.groups)
        
        # LoRA branch: (x * lora_A) * lora_B
        # lora_A uses the same groups as original to match input channel structure
        adapter = F.conv2d(self.dropout(x), self.lora_A, None, self.stride,
                           self.padding, self.dilation, self.groups)
        
        # Optimization: Multiply the small lora_B by scaling BEFORE the conv2d.
        # This saves one full-size tensor multiplication kernel launch.
        adapter = F.conv2d(adapter, self.lora_B * self.scaling)
        
        return result + adapter

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


def _inject_lora(model: nn.Module, config: LoRAConfig, prefix: str = "",
                 registry: Optional[List[Tuple[str, nn.Module, str, nn.Module]]] = None,
                 matched_block_ids: Optional[set] = None
                 ) -> List[Tuple[str, nn.Module, str, nn.Module]]:
    """Walk *model* named_children recursively.  When we find a module that
    corresponds to a target, wrap it with LoRA."""
    if registry is None:
        registry = []
    if matched_block_ids is None:
        matched_block_ids = set()

    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, (nn.Linear, nn.Conv2d)):
            parent_name = prefix.rpartition(".")[2] if prefix else ""
            is_target = False

            # to_q / to_k / to_v : direct attributes of CrossAttention
            if name in ("to_q", "to_k", "to_v"):
                is_target = True
            # to_out.0 : the first element inside nn.Sequential named "to_out"
            elif parent_name == "to_out" and name.isdigit() and name == "0":
                is_target = True
            # Extra blocks: time_embed and label_emb use segment-boundary match
            elif (_segment_match(full_name, "time_embed")
                 or full_name.startswith("time_embed.")
                 or _segment_match(full_name, "label_emb")
                 or full_name.startswith("label_emb.")):
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
                    lora_layer = LoRALinear(
                        child, rank=config.rank,
                        alpha=config.alpha, dropout=config.dropout,
                        weight=block_weight,
                    )
                else: # Conv2d
                    lora_layer = LoRAConv2d(
                        child, rank=config.rank,
                        alpha=config.alpha, dropout=config.dropout,
                        weight=block_weight,
                    )
                setattr(model, name, lora_layer)
                registry.append((full_name, model, name, lora_layer))
        else:
            _inject_lora(child, config, full_name, registry, matched_block_ids)

    return registry


def inject_lora_into_unet(unet_model: nn.Module, config: LoRAConfig
                          ) -> List[Tuple[str, nn.Module, str, nn.Module]]:
    matched_ids = set()
    registry = _inject_lora(unet_model, config, matched_block_ids=matched_ids)

    # Warn about block_weight keys that matched no injected layer.
    if config.block_weights:
        unmatched = set(config.block_weights.keys()) - matched_ids
        if unmatched:
            print(f"    [LoRA] WARNING: the following block_weighting keys matched "
                  f"no injected LoRA layers (check spelling or module structure): "
                  f"{sorted(unmatched)}")

    return registry


def extract_lora_weights(registry: List[Tuple[str, nn.Module, str, nn.Module]]
                         ) -> Dict[str, torch.Tensor]:
    weights = {}
    for full_name, parent, attr_name, lora_layer in registry:
        if not isinstance(lora_layer, (LoRALinear, LoRAConv2d)):
            continue
        lora_key = _key_to_lora_key(full_name)
        A, B = lora_layer.get_lora_weights()
        weights[f"{lora_key}{_DOWN_SUFFIX}"] = A.detach().cpu().contiguous()
        weights[f"{lora_key}{_UP_SUFFIX}"]   = B.detach().cpu().contiguous()
        weights[f"{lora_key}{_ALPHA_SUFFIX}"] = torch.tensor([lora_layer.alpha],
                                                              dtype=torch.float32)
    return weights


def load_lora_into_model(registry: List[Tuple[str, nn.Module, str, nn.Module]],
                         state_dict: Dict[str, torch.Tensor]):
    for full_name, parent, attr_name, lora_layer in registry:
        if not isinstance(lora_layer, (LoRALinear, LoRAConv2d)):
            continue
        lora_key = _key_to_lora_key(full_name)
        A_key = f"{lora_key}{_DOWN_SUFFIX}"
        B_key = f"{lora_key}{_UP_SUFFIX}"
        alpha_key = f"{lora_key}{_ALPHA_SUFFIX}"
        if A_key in state_dict and B_key in state_dict:
            lora_layer.load_lora_weights(state_dict[A_key], state_dict[B_key])
            # Restore the alpha/scaling from the checkpoint so that a run resumed
            # with a different alpha in the config still uses the trained scaling.
            if alpha_key in state_dict:
                saved_alpha = state_dict[alpha_key].item()
                if abs(saved_alpha - lora_layer.alpha) > 1e-6:
                    print(f"    [LoRA] {full_name}: alpha mismatch "
                          f"(checkpoint={saved_alpha}, config={lora_layer.alpha}). "
                          f"Using checkpoint value.")
                lora_layer.alpha = saved_alpha
                lora_layer.scaling = (saved_alpha / lora_layer.rank) * lora_layer.training_weight


def merge_lora_into_unet(registry: List[Tuple[str, nn.Module, str, nn.Module]]):
    for full_name, parent, attr_name, lora_layer in registry:
        if isinstance(lora_layer, (LoRALinear, LoRAConv2d)):
            lora_layer.merge()


def lora_param_count(registry: List[Tuple[str, nn.Module, str, nn.Module]]) -> int:
    total = 0
    for _, _, _, layer in registry:
        if isinstance(layer, (LoRALinear, LoRAConv2d)):
            total += layer.lora_A.numel() + layer.lora_B.numel()
    return total
