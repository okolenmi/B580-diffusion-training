"""Mid-training preview image generation.

Runs a short sampling pass with the current (in-progress) student weights so
training quality can be watched visually without waiting for the run to
finish. Deliberately reuses the exact stepping formula and helper functions
already proven correct in manager/builder.py's teacher trajectory sampler
(x_t = x0 + sigma*eps forward process; deterministic Euler-in-x0 step),
rather than deriving new sampling math -- this is inference-only code but
the underlying parameterization must match training exactly or the preview
would silently be measuring something other than what was trained.

Uses a fixed seed so successive previews across a run are directly
comparable to each other: only the weights change between them, not the
starting noise.
"""

import json
from pathlib import Path

import torch
from PIL import Image

from .model_io import comfy_input_transform, raw_to_denoised, make_init_noise
from .noise_schedule import get_alpha_sigma
from .vae_decode import VAEDecoder
from .lora import set_lora_gate


class PreviewGenerator:
    """Generates and saves a grid of preview images at a point in training."""

    def __init__(self, conds, neg_cond, non_unet_sd, device, out_dir,
                 steps=16, cfg=4.0, resolution=512, seed=12345, student_type="eps",
                 max_batch_size=8):
        """
        Parameters
        ----------
        conds : list[tuple[Tensor, Tensor]]
            (ctx, y) pairs, one per preview prompt. Pinned CPU tensors are
            fine -- moved to device here.
        neg_cond : tuple[Tensor, Tensor] or None
            Shared (ctx_u, y_u) for CFG. None disables CFG (cfg treated as 1.0).
        non_unet_sd : dict
            Non-UNet portion of a checkpoint state dict (contains VAE weights
            under "first_stage_model."). Expected to already be resident in
            CPU RAM (Trainer keeps this for its whole lifetime as self.non_unet)
            -- the VAE itself is only instantiated on `device` for the brief
            decode step below and freed immediately after, the same load/free
            pattern Trainer._dump_cache_samples already uses.
        out_dir : str or Path
            Directory to write step_{N}/ subfolders and manifest.json into.
        max_batch_size : int
            Max prompts processed in a single batched forward pass (doubled
            internally when CFG is active). Prompts beyond this run in
            additional sequential chunks, capping worst-case VRAM usage when
            many prompts are configured.
        """
        self.conds = conds
        self.neg_cond = neg_cond
        self.non_unet_sd = non_unet_sd
        self.device = device
        self.out_dir = Path(out_dir)
        self.steps = steps
        self.cfg = cfg
        self.resolution = resolution
        self.seed = seed
        self.student_type = student_type
        self.max_batch_size = max(1, max_batch_size)
        self.latent_size = max(1, resolution // 8)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.out_dir / "manifest.json"

    def _denoise_chunk(self, student, conds_chunk, t_grid):
        """Run the full denoising loop for one chunk of prompts, batched
        into a single forward call per step (2x that with CFG). Returns the
        chunk's final latents as a CPU float32 tensor, shape (len(conds_chunk), 4, H, W).
        """
        n_steps = self.steps
        n_prompts = len(conds_chunk)
        use_cfg = self.cfg > 1.0 and self.neg_cond is not None

        ctx_batch = torch.cat([c.to(self.device) for c, _ in conds_chunk], dim=0)
        y_batch = torch.cat([y.to(self.device) for _, y in conds_chunk], dim=0)
        if use_cfg:
            ctx_u, y_u = self.neg_cond
            ctx_u_batch = ctx_u.to(self.device).repeat(n_prompts, 1, 1)
            y_u_batch = y_u.to(self.device).repeat(n_prompts, 1)

        # Same starting noise for every prompt (matches original per-prompt
        # behavior: the seed was reset before each prompt, so every prompt
        # already started from the identical noise tensor).
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        st0 = get_alpha_sigma(t_grid[0])[1]
        single_noise = make_init_noise((1, 4, self.latent_size, self.latent_size),
                                        self.device, torch.float32, st0, generator=gen)
        x_t = single_noise.repeat(n_prompts, 1, 1, 1)

        for step_i in range(n_steps):
            t_val = t_grid[step_i]
            at_cur, st_cur = get_alpha_sigma(t_val)
            xc = comfy_input_transform(x_t, st_cur)

            if use_cfg:
                t_tensor = torch.tensor([t_val] * (n_prompts * 2), dtype=torch.long, device=self.device)
                batched_x = torch.cat([xc, xc], dim=0)
                batched_ctx = torch.cat([ctx_batch, ctx_u_batch], dim=0)
                batched_y = torch.cat([y_batch, y_u_batch], dim=0)
                out = student.forward(batched_x, t_tensor, batched_ctx, batched_y)
                out_pos, out_neg = out.chunk(2)
                raw = out_neg + (out_pos - out_neg) * self.cfg
            else:
                t_tensor = torch.tensor([t_val] * n_prompts, dtype=torch.long, device=self.device)
                raw = student.forward(xc, t_tensor, ctx_batch, y_batch)

            at_t = torch.tensor([at_cur.item()], device=self.device, dtype=torch.bfloat16).view(-1, 1, 1, 1)
            st_t = torch.tensor([st_cur.item()], device=self.device, dtype=torch.bfloat16).view(-1, 1, 1, 1)
            denoised = raw_to_denoised(raw, x_t, at_t, st_t, self.student_type)

            if step_i < n_steps - 1:
                st_next = get_alpha_sigma(t_grid[step_i + 1])[1]
                x_t = x_t + ((x_t - denoised) / st_cur) * (st_next - st_cur)
            else:
                # Final step: use the model's own x0 prediction directly
                # rather than stepping to t=0's (near-zero) sigma.
                x_t = denoised

        return x_t.float().cpu()

    @torch.no_grad()
    def generate(self, student, global_step):
        """Sample one image per configured prompt using the current student
        weights, decode, and save under out_dir/step_{global_step}/.

        Prompts are split into chunks of at most max_batch_size, each chunk
        batched into a single forward call per denoising step (2x that with
        CFG) -- this caps worst-case VRAM usage when many prompts are
        configured, while still batching within each chunk for speed.

        Caller is responsible for the model's train()/eval() mode around
        this call -- generate() itself doesn't mutate it, so this stays
        usable for both the training-time hook and any future standalone
        preview tooling that might want train-mode dropout active or not.
        """
        step_dir = self.out_dir / f"step_{global_step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        set_lora_gate(None)

        n_steps = self.steps
        t_high = 999
        # Same construction as builder.py's uniform t_grid: n_steps+1 points
        # from t_high down to (and including) 0.
        t_grid = [round(t_high - j * t_high / n_steps) for j in range(n_steps + 1)]

        chunk_latents = []
        for start in range(0, len(self.conds), self.max_batch_size):
            chunk = self.conds[start:start + self.max_batch_size]
            chunk_latents.append(self._denoise_chunk(student, chunk, t_grid))
        final_latents = torch.cat(chunk_latents, dim=0)

        # Load the VAE once, decode everything in one call, then free immediately.
        vae = VAEDecoder.from_checkpoint(self.non_unet_sd, self.device)
        if vae is None:
            raise RuntimeError("No VAE weights found in the loaded checkpoint (paths.base_model) "
                               "-- cannot decode preview images.")
        img_tensor = vae.decode(final_latents.to(self.device, dtype=torch.bfloat16))
        saved = []
        for i in range(len(self.conds)):
            img_np = img_tensor[i].permute(1, 2, 0).numpy()
            fname = f"prompt_{i}.png"
            Image.fromarray(img_np).save(step_dir / fname)
            saved.append(fname)
        vae.free()
        del vae
        from .comfy_setup import xpu_empty_cache
        xpu_empty_cache()

        self._update_manifest(global_step, saved)
        return step_dir

    def _update_manifest(self, global_step, filenames):
        """Append this step's result to previews/manifest.json.

        Format: [{"step": int, "files": [str, ...]}, ...], sorted by step.
        Trimmed to the most recent 200 entries so a very long run doesn't
        grow this file (or the previews directory it tracks) unbounded.
        """
        manifest = []
        if self._manifest_path.exists():
            try:
                manifest = json.loads(self._manifest_path.read_text())
            except Exception:
                manifest = []
        manifest = [m for m in manifest if m.get("step") != global_step]
        manifest.append({"step": global_step, "files": filenames})
        manifest.sort(key=lambda m: m["step"])
        manifest = manifest[-200:]
        self._manifest_path.write_text(json.dumps(manifest))
