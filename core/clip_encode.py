"""CLIP text encoder for SDXL — loads from full checkpoint conditioner keys.

Extracts CLIP weights from the teacher checkpoint's conditioner section,
tokenizes text with SDXL's dual tokenizer (clip_l + clip_g),
encodes to (ctx, pooled) tensors, and builds (ctx, y) for the UNet.

CLIP is loaded once, used for conditioning, then unloaded.
"""

import torch


def _extract_and_convert_clip_state_dict(state_dict: dict) -> dict:
    """Extract conditioner CLIP keys and convert to clip_l/clip_g prefix format."""
    import comfy.utils as utils

    cond_sd = {k: v for k, v in state_dict.items()
               if k.startswith("conditioner.")}

    replace_prefix = {
        "conditioner.embedders.0.transformer.text_model": "clip_l.transformer.text_model",
        "conditioner.embedders.1.model.": "clip_g.",
    }
    cond_sd = utils.state_dict_prefix_replace(cond_sd, replace_prefix, filter_keys=True)
    cond_sd = utils.clip_text_transformers_convert(cond_sd, "clip_g.", "clip_g.transformer.")

    if "clip_l.transformer.text_model.embeddings.position_ids" not in cond_sd:
        cond_sd["clip_l.transformer.text_model.embeddings.position_ids"] = torch.arange(77).expand((1, -1))

    return cond_sd


class SDXLClipEncoder:
    """Standalone SDXL CLIP encoder — loads from teacher checkpoint conditioner."""

    def __init__(self, full_state_dict: dict, device: str):
        self.device = device
        self.dtype = torch.float16  # CLIP runs in fp16
        self.out_dtype = torch.bfloat16  # Output matches UNet dtype
        self._embedder = None

        # Build tokenizer and model
        from comfy.sdxl_clip import SDXLClipModel, SDXLTokenizer
        self.tokenizer = SDXLTokenizer()
        self.clip_model = SDXLClipModel(device="cpu", dtype=self.dtype)
        self.clip_model.eval()

        # Extract and load CLIP weights using load_state_dict (same as ComfyUI)
        clip_sd = _extract_and_convert_clip_state_dict(full_state_dict)

        # Add position IDs if missing (required for clip_l)
        if 'clip_l.transformer.text_model.embeddings.position_ids' not in clip_sd:
            clip_sd['clip_l.transformer.text_model.embeddings.position_ids'] = torch.arange(77).expand((1, -1))

        # Convert all weights to model's dtype — load_state_dict silently skips
        # dtype-mismatched keys, leaving the model with random weights (→ NaN)
        for k in list(clip_sd.keys()):
            if isinstance(clip_sd[k], torch.Tensor):
                clip_sd[k] = clip_sd[k].to(dtype=self.dtype)

        # Use load_state_dict with strict=False — this is what ComfyUI does
        # (CLIP.load_sd with full_model=True). Missing keys like logit_scale
        # and text_projection are expected and harmless for SDXL.
        missing, unexpected = self.clip_model.load_state_dict(clip_sd, strict=False)
        # Filter out expected-but-missing keys (not used by SDXL)
        _expected_missing = {
            'clip_l.logit_scale',
            'clip_l.transformer.text_projection.weight',
        }
        missing = [k for k in missing if k not in _expected_missing]
        # Filter out expected-but-unexpected keys (we add position_ids manually)
        _expected_unexpected = {'clip_l.transformer.text_model.embeddings.position_ids'}
        unexpected = [k for k in unexpected if k not in _expected_unexpected]
        if missing:
            print(f"    Warning: {len(missing)} CLIP keys missing in checkpoint")
        if unexpected:
            print(f"    Warning: {len(unexpected)} unexpected CLIP keys")
        del clip_sd

        # Move to device
        self.clip_model = self.clip_model.to(device=device, dtype=self.dtype)
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single prompt. Returns (ctx, pooled)."""
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        with torch.no_grad():
            ctx, pooled = self.clip_model.encode_token_weights(tokens)
        return ctx, pooled

    def encode_prompt_and_pool(self, prompt: str, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """encode_prompt() plus the padding/dtype/device/batch handling
        encode_for_unet() below always did right after it -- pulled out
        on its own for the same reason resolution_embedding() was: the
        expensive half (the real CLIP forward pass, inside encode_prompt()
        above) depends only on the prompt string, never on height/width,
        so a caller with real cached (ctx, pooled) for this exact prompt
        (nodes/model/text_encoder_cache.py's CachingTextEncoder, as of
        this session) can skip straight to resolution_embedding() on a
        cache hit instead of re-running this. encode_for_unet() below
        calls this too, unchanged in what it returns."""
        ctx, pooled = self.encode_prompt(prompt)

        # Ensure at least 77 tokens (standard SDXL base length)
        if ctx.shape[1] < 77:
            padding = torch.zeros((ctx.shape[0], 77 - ctx.shape[1], ctx.shape[2]),
                                 device=ctx.device, dtype=ctx.dtype)
            ctx = torch.cat([ctx, padding], dim=1)

        pooled = pooled.to(device=self.device, dtype=self.out_dtype)
        ctx = ctx.to(device=self.device, dtype=self.out_dtype).repeat(batch_size, 1, 1)
        pooled = pooled.repeat(batch_size, 1)
        return ctx, pooled

    def _get_embedder(self):
        if self._embedder is None:
            from comfy.model_base import Timestep
            self._embedder = Timestep(256).to(device=self.device, dtype=self.out_dtype)
        return self._embedder

    def resolution_embedding(self, height: int, width: int, batch_size: int = 1,
                              crop_w: int = 0, crop_h: int = 0,
                              target_width: int = None, target_height: int = None) -> torch.Tensor:
        """The (batch, 1536) SDXL time-embedding half of encode_for_unet()'s
        own `y` output -- same computation (same as ComfyUI's
        SDXL.encode_adm), pulled out on its own specifically so a caller
        that already has real, cached pooled text conditioning for this
        prompt (nodes/model/text_encoder_cache.py's CachingTextEncoder,
        as of this session) can get just the resolution-dependent part
        recomputed, without re-running the CLIP forward pass encode_prompt()
        does -- CLIP's own encoding depends only on the prompt string, never
        on height/width, so the two were always independent pieces of work,
        just not separately callable before this. encode_for_unet() below
        calls this too, unchanged in what it returns, not a second copy of
        this math."""
        if target_width is None:
            target_width = width
        if target_height is None:
            target_height = height

        embedder = self._get_embedder()

        time_embs = []
        # original_h, original_w, crop_h, crop_w, target_h, target_w
        for val in [height, width, crop_h, crop_w, target_height, target_width]:
            time_embs.append(embedder(torch.tensor([val], device=self.device, dtype=self.out_dtype)))
        return torch.cat(time_embs, dim=-1).repeat(batch_size, 1)

    def encode_for_unet(self, prompt: str, batch_size: int = 1,
                        height: int = 1024, width: int = 1024,
                        crop_w: int = 0, crop_h: int = 0,
                        target_width: int = None, target_height: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt and return (ctx, y) ready for UNet forward pass.

        ctx: (batch, N*77, 2048)
        y:   (batch, 2816) = 1280 pooled text + 1536 SDXL time embeddings

        Time embeddings encode: height, width, crop_h, crop_w, target_height, target_width
        Each embedded to 256 dims using sinusoidal timestep embedding (total 6×256=1536).

        Thin combination of encode_prompt_and_pool() + resolution_embedding()
        (both above) -- kept as one call for every caller except
        CachingTextEncoder, which needs the two pieces separately instead.
        """
        ctx, pooled = self.encode_prompt_and_pool(prompt, batch_size)
        time_emb_flat = self.resolution_embedding(
            height, width, batch_size, crop_w=crop_w, crop_h=crop_h,
            target_width=target_width, target_height=target_height)
        y = torch.cat([pooled, time_emb_flat], dim=-1)
        return ctx, y

    def unload(self):
        """Free CLIP and embedder from GPU memory."""
        if self.clip_model:
            self.clip_model = self.clip_model.cpu()
        if self._embedder:
            self._embedder = self._embedder.cpu()
        self.device = "cpu"
        import gc
        gc.collect()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
