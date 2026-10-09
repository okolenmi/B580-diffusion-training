"""CLIP text encoder for SDXL -- loads from full checkpoint conditioner keys.

Extracts CLIP weights from the teacher checkpoint's conditioner section,
tokenizes text with SDXL's dual tokenizer (clip_l + clip_g),
encodes to (ctx, pooled) tensors, and builds (ctx, y) for the UNet.

CLIP is loaded once, used for conditioning, then unloaded.

**Moved here from `core/clip_encode.py` on 2026-10-02, unchanged.**
Text encoding was the last place `nodes/model/` still borrowed a
`core/` implementation outright -- `SDXLTextEncoderNode.build()` and
`SDXLArchitecture.build_text_encoder()` both constructed this class, so
the node graph's conditioning path ran through `core/`. The module was
self-contained (torch only, nothing from `core/`), which made this a
relocation rather than a rewrite, consistent with the project's rule
that verified code gets wrapped rather than re-derived.

`core/clip_encode.py` is kept as a re-export shim rather than deleted,
because `core/`'s own trainer, its two cache builders, and
`manager/builder.py` all construct this class too, and `core/` is still
the production training path the backend spawns. Point those at this
module directly when `core/` is eventually retired; the shim exists so
that retirement is a one-line change per call site instead of a search
for every caller.

The comments below are the originals and still describe the real
reasons for the non-obvious lines; see `nodes/model/text_encoder.py`
for what wraps this, and `nodes/model/text_encoder_cache.py` for why
`encode_prompt_and_pool()` and `resolution_embedding()` are separate
methods rather than one.
"""

import torch

from .clip_state_dict import (
    clip_text_transformers_convert,
    state_dict_prefix_replace,
)
from .clip import SDXLClipModel
from .timestep_embedding import Timestep
from .tokenizer import SDXLTokenizer


def _extract_and_convert_clip_state_dict(state_dict: dict) -> dict:
    """Extract conditioner CLIP keys and convert to clip_l/clip_g prefix format.

    `filter_keys=True` drops any conditioner key matching neither prefix,
    which reads like a hazard and is not one: checked against a real SDXL
    checkpoint, all 587 `conditioner.` keys match one of the two (197 under
    the CLIP-L prefix, 390 under the CLIP-G one, zero unmatched). ComfyUI
    passes `filter_keys=False` here, which keeps the old spellings too; the
    difference only shows on a checkpoint whose conditioner keys are not in
    the format this loader expects.
    """
    cond_sd = {k: v for k, v in state_dict.items()
               if k.startswith("conditioner.")}

    replace_prefix = {
        "conditioner.embedders.0.transformer.text_model": "clip_l.transformer.text_model",
        "conditioner.embedders.1.model.": "clip_g.",
    }
    cond_sd = state_dict_prefix_replace(cond_sd, replace_prefix, filter_keys=True)
    cond_sd = clip_text_transformers_convert(cond_sd, "clip_g.", "clip_g.transformer.")

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

        # The tokenizer and both CLIP towers are this project's
        # (nodes/model/tokenizer.py and nodes/model/clip.py). They were
        # comfy's sdxl_clip; see design doc 12 section 7.3.
        self.tokenizer = SDXLTokenizer()
        # Meta-device construction, same reason as the UNet wrapper's:
        # eager init randomly fills 818M params (~4 s, measured) that the
        # load_state_dict below overwrites immediately.
        with torch.device("meta"):
            self.clip_model = SDXLClipModel(device="meta", dtype=self.dtype)
        self.clip_model.to_empty(device="cpu")
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
        token_ids = self.tokenizer.tokenize(prompt)
        with torch.no_grad():
            ctx, pooled = self.clip_model.encode_token_ids(token_ids)
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

    def batching_available(self) -> bool:
        """True when `encode_prompts` will actually batch.

        Float32 towers, and only then -- see `encode_prompts`'s docstring for
        the measurement and for why casting here instead would be worse than
        not batching.

        A *query* rather than a constant, because the answer depends on how
        this encoder was constructed and a module-level flag would be a lie
        the moment someone builds one with a different dtype.
        """
        try:
            dtype = next(self.clip_model.parameters()).dtype
        except StopIteration:  # pragma: no cover -- an empty tower is broken
            return False
        return dtype == torch.float32

    def encode_prompts(self, prompts, batch_size: int = 1):
        """Encode N independent prompts, one per row: [(ctx, pooled), ...].

        The bulk sibling of `encode_prompt_and_pool`, and the reason it is
        not just a loop over that: one prompt per forward is 32.2 ms on the
        B580 where 64 at a time is 2.53 ms in float16 -- but float16
        batching diverges 19% from serial float16 and float32 diverges
        0.17% (`SDXLClipModel.encode_token_rows`'s own docstring has the
        numbers and what they are normalised by). So this goes through the
        float32 path, at 10.99 ms/prompt, which is the only batching that
        agrees with the per-prompt path closely enough for a warm cache and
        a cache miss to be interchangeable.

        Per-prompt results, each shaped exactly as
        `encode_prompt_and_pool(prompt, batch_size)` would return it, so a
        caller cannot tell which path produced them. A one-prompt call is a
        batch of one, not a special case -- that is what keeps the two paths
        from disagreeing, and it is why this is not "loop and concatenate".

        **Refuses to batch unless the towers are float32, and falls back to
        the per-prompt loop when they are not.** `self.dtype` is float16
        (line above: "CLIP runs in fp16"), so *this is the production path*
        and the fallback is what actually runs today -- the batching is
        dormant until something loads CLIP in float32. That is deliberate
        and it is the only safe option:

        * casting to float32 inside this method would make a warmed prompt
          differ from a missed one by the float16-vs-float32 gap, which is
          the incoherence the whole design exists to avoid -- a run's
          conditioning would depend on whether its cache was warm;
        * loading CLIP in float32 unconditionally would fix that at 2x the
          resident footprint (about 3.1 GB) for a speedup that, at a
          5,000-entry budget, saves 2.7 min of a run that takes an hour.

        So the batched path is opt-in by dtype rather than wrong by default,
        and `batching_available()` below is what the warm pass reports so a
        slow warm is explicable rather than mysterious.
        """
        if not prompts:
            return []
        if not self.batching_available():
            return [self.encode_prompt_and_pool(prompt, batch_size)
                    for prompt in prompts]
        # The model's input shape is one dict holding one row per prompt per
        # tower -- `{"l": [ids, ...], "g": [ids, ...]}` -- the same
        # `encode_token_ids` takes. Kept identical on purpose: a second
        # convention for "several prompts" would be a second thing to get
        # right, and the rows-per-tower form is already what the towers
        # validate and stack.
        l_rows, g_rows = [], []
        for prompt in prompts:
            token_ids = self.tokenizer.tokenize(prompt)
            l_rows.append(list(token_ids["l"][0]))
            g_rows.append(list(token_ids["g"][0]))
        context, pooled = self.clip_model.encode_token_rows(
            {"l": l_rows, "g": g_rows})
        results = []
        for row_ctx, row_pooled in zip(context, pooled):
            if row_ctx.shape[0] < 77:
                padding = torch.zeros(
                    (row_ctx.shape[0], 77 - row_ctx.shape[1], row_ctx.shape[2]),
                    device=row_ctx.device, dtype=row_ctx.dtype)
                row_ctx = torch.cat([row_ctx, padding], dim=1)
            ctx = row_ctx.unsqueeze(0).to(device=self.device, dtype=self.out_dtype)
            y = row_pooled.unsqueeze(0).to(device=self.device, dtype=self.out_dtype)
            results.append((
                ctx.repeat(batch_size, 1, 1) if batch_size > 1 else ctx,
                y.repeat(batch_size, 1) if batch_size > 1 else y,
            ))
        return results

    def _get_embedder(self):
        if self._embedder is None:
            # Owned here rather than imported from ComfyUI; see
            # nodes/model/timestep_embedding.py for the provenance and the
            # measurements. It used to read
            # `Timestep(256).to(device=self.device, dtype=self.out_dtype)`,
            # which controlled nothing: Timestep has no parameters and no
            # buffers, so `.to()` on it does not even move a device. The
            # output is float32 whatever `out_dtype` says, and the device
            # comes from the tensor handed in below. `torch.cat` promotes,
            # so `y` comes out float32 either way -- as it does in ComfyUI.
            self._embedder = Timestep(256)
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
