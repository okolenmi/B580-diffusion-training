"""Dataset generation and ingestion engine."""

import gc
import json
import math
import random
import time
import csv
import os
import uuid
from pathlib import Path
from typing import Any, Optional, Union

import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from safetensors.torch import load_file

from core.clip_encode import SDXLClipEncoder
from core.comfy_setup import xpu_empty_cache
from core.model_io import comfy_input_transform, make_init_noise, raw_to_denoised, raw_to_target
from core.noise_schedule import get_alpha_sigma, sample_timestep, eps_to_vpred
from core.seed import derive_seed
from core.unet_wrapper import ComfyUNetWrapper
from core.vae_decode import VAEDecoder
from .db import _connect, add_source, get_trajectories, update_task_progress, update_task_status
from .preview import PreviewGenerator
from .storage import ShardWriter

_MAX_TRAJS_PER_SHARD = 50


class DataTaskRunner:
    """Manages generation of synthetic trajectories and real data encoding."""

    def __init__(self, device: str):
        self.device = device

    def _prepare_keywords(self, cfg: dict) -> list[str]:
        keywords = []
        # Add static keywords if any
        raw_kws = cfg.get("keywords")
        if isinstance(raw_kws, list):
            keywords.extend(raw_kws)
        elif isinstance(raw_kws, str) and raw_kws.strip():
            keywords.extend([k.strip() for k in raw_kws.split("\n") if k.strip()])

        kw_file = cfg.get("keywords_file")
        if kw_file:
            p_kw_file = Path(kw_file)
            if p_kw_file.exists():
                if p_kw_file.suffix.lower() == ".csv":
                    with open(p_kw_file, "r", encoding="utf-8") as f:
                        content = f.read(4096)
                        f.seek(0)
                        # Try to detect delimiter and header
                        try:
                            sniffer = csv.Sniffer()
                            dialect = sniffer.sniff(content)
                            has_header = sniffer.has_header(content)
                            reader = csv.reader(f, dialect)
                            if has_header: next(reader)
                        except Exception:
                            f.seek(0)
                            reader = csv.reader(f)
                            
                        for row in reader:
                            if row and row[0].strip():
                                kw = row[0].strip().replace("_", " ")
                                if not kw.lower().startswith(("http", "www")):
                                    keywords.append(kw)
                else:
                    with open(p_kw_file, "r", encoding="utf-8") as f:
                        file_kws = [line.strip().replace("_", " ") for line in f 
                                    if line.strip() and not line.startswith("#")]
                        keywords.extend(file_kws)
        
        # Deduplicate while preserving order for stability (rng will handle randomness)
        seen = set()
        return [x for x in keywords if not (x in seen or seen.add(x))]

    def _generate_prompts(self, cfg: Union[list, dict, str], count: int, rng: random.Random) -> list[str]:
        if isinstance(cfg, list):
            return [cfg[i % len(cfg)] for i in range(count)]
        if isinstance(cfg, str):
            return [cfg] * count
        
        # Dictionary mode
        keywords = self._prepare_keywords(cfg)
        if not keywords:
            return [""] * count
            
        template = cfg.get("template") or "{keywords}"
        min_k, max_k = cfg.get("min", 3), cfg.get("max", 10)

        print(f"    Keyword pool: {len(keywords)} unique, sampling {count} prompts ({min_k}-{max_k} each)")

        results = []
        for i in range(count):
            n = rng.randint(min_k, max_k)
            sampled = rng.sample(keywords, min(n, len(keywords)))
            kw_str = ", ".join(sampled)
            results.append(template.replace("{keywords}", kw_str) if "{keywords}" in template else f"{template}, {kw_str}")
        return results

    def _finalize_shard(self, dataset_root: Path, shard_writer: ShardWriter, shard_file: Path, source_id: int,
                        pending_trajs: list[dict] = None):
        """Write shard to disk and insert DB records in one transaction."""
        sample_count, size_bytes = shard_writer.write()
        
        with _connect(dataset_root / "metadata.db") as conn:
            cur = conn.execute(
                "INSERT INTO shards (file_path, sample_count, size_bytes, is_temporary, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(shard_file.relative_to(dataset_root)), sample_count, size_bytes, 1, time.time())
            )
            shard_id = cur.lastrowid
            
            if pending_trajs:
                for td in pending_trajs:
                    conn.execute(
                        "INSERT INTO trajectories (source_id, shard_id, shard_index, sample_count, seed, prompt, preview_path, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (source_id, shard_id, td["shard_index"], td["sample_count"],
                         td["seed"], td["prompt"], td["preview_path"], td["metadata_json"])
                    )
            conn.commit()

    def run_teacher_task(self, dataset_root: Path, model_path: Path, 
                         pos_cfg: Union[list, dict], neg_cfg: Union[str, dict],
                         n_conditions: int, n_samples_per_cond: int,
                         steps_range: tuple[int, int],
                         cfg_range: tuple[float, float],
                         batch_size: int = 1, latent_size: int = 64, seed: int = 42,
                         model_type: str = "eps",
                         task_id: int = None,
                         t_mode: str = "uniform",
                         t_low: int = 20,
                         t_high: int = 999):
        """Generate synthetic trajectories from a teacher model with dual-randomized prompts.

        model_type: prediction type of the teacher model — "eps" or "vpred".
        Must match the checkpoint.  Getting this wrong causes the Euler denoising
        steps to walk in the wrong direction and stores targets in the wrong space.
        """
        try:
            if task_id: update_task_progress(dataset_root / "metadata.db", task_id, 0, pid=os.getpid())

            source_id = add_source(dataset_root / "metadata.db", f"{model_path.name}_distill", "teacher", 
                                str(model_path), {
                                    "pos": pos_cfg, "neg": neg_cfg,
                                    "cfg_range": cfg_range, "n_cond": n_conditions, "n_samples": n_samples_per_cond,
                                    "model_type": model_type,
                                    "t_mode": t_mode, "t_low": t_low, "t_high": t_high,
                                })

            print(f"  Loading teacher model: {model_path.name}")
            sd = load_file(str(model_path))
            
            total_trajs = n_conditions * n_samples_per_cond
            rng = random.Random(seed)
            pos_prompts = self._generate_prompts(pos_cfg, n_conditions, rng)
            neg_prompts = self._generate_prompts(neg_cfg, n_conditions, rng)

            print(f"  Encoding conditioning pool...")
            encoder = SDXLClipEncoder(sd, device=self.device)
            px = latent_size * 8
            
            all_unique_texts = set(pos_prompts) | set(neg_prompts)
            text_cache = {}
            for txt in all_unique_texts:
                ctx, y = encoder.encode_for_unet(txt, batch_size=1, height=px, width=px)
                text_cache[txt] = (ctx.cpu().float().contiguous(), y.cpu().float().contiguous())
            
            encoder.unload(); del encoder; xpu_empty_cache(); gc.collect()

            vae_sd = {k.replace("first_stage_model.", ""): v
                    for k, v in sd.items() if k.startswith("first_stage_model")}
            previewer = PreviewGenerator(self.device, vae_sd)
            teacher = ComfyUNetWrapper(sd, device=self.device, dtype=torch.bfloat16)
            teacher.eval()
            for p in teacher.parameters(): p.requires_grad_(False)
            del sd; xpu_empty_cache(); gc.collect()

            # Check for duplicate (prompt, seed) pairs already in the archive
            db_path = dataset_root / "metadata.db"
            existing_trajs = get_trajectories(db_path)
            existing_keys = {(t["prompt"], t["seed"]) for t in existing_trajs if t["seed"] is not None}
            dup_count = 0

            shard_file = dataset_root / "staging" / f"teacher_{uuid.uuid4().hex[:12]}.safetensors"
            shard_writer = ShardWriter(shard_file)
            pending_trajs = []
            trajs_in_shard = 0
            preview_tasks = []  # (latent, output_path)
            
            traj_idx = 0
            with tqdm(total=total_trajs, desc="Generating Trajectories") as pbar:
                for cond_i in range(n_conditions):
                    p_text = pos_prompts[cond_i]
                    p_ctx_base, p_y_base = text_cache[p_text]

                    for sample_i in range(0, n_samples_per_cond, batch_size):
                        this_bs = min(batch_size, n_samples_per_cond - sample_i)
                        n_text = rng.choice(neg_prompts)
                        n_ctx_base, n_y_base = text_cache[n_text]
                        
                        p_ctx = p_ctx_base.repeat(this_bs, 1, 1).to(self.device, dtype=torch.bfloat16)
                        p_y = p_y_base.repeat(this_bs, 1).to(self.device, dtype=torch.bfloat16)
                        n_ctx = n_ctx_base.repeat(this_bs, 1, 1).to(self.device, dtype=torch.bfloat16)
                        n_y = n_y_base.repeat(this_bs, 1).to(self.device, dtype=torch.bfloat16)
                        
                        batch_base_seed = seed + traj_idx
                        for b in range(this_bs):
                            if (p_text, batch_base_seed + b) in existing_keys:
                                dup_count += 1
                        cfg_val = rng.uniform(*cfg_range)
                        n_steps = rng.randint(*steps_range)
                        
                        if t_mode == "uniform":
                            t_grid = [round(t_high - j * (t_high - 0) / n_steps) for j in range(n_steps + 1)]
                        else:
                            # Non-linear trajectory spacing: sample boundaries from distribution
                            sampled = [sample_timestep(rng, t_mode, 1, t_high) for _ in range(n_steps - 1)]
                            t_grid = sorted([t_high, 0] + sampled, reverse=True)
                            
                        _, st_first = get_alpha_sigma(t_grid[0])

                        # Create unique noise per sample
                        x_t_list = []
                        for b in range(this_bs):
                            cpu_gen = torch.Generator(device="cpu")
                            cpu_gen.manual_seed(derive_seed(batch_base_seed + b, 0, "traj_init"))
                            x_t_list.append(make_init_noise((1, 4, latent_size, latent_size), self.device, 
                                                           torch.bfloat16, st_first, cpu_gen))
                        x_t = torch.cat(x_t_list, dim=0)
                        batch_trajectories = [[] for _ in range(this_bs)]
                        
                        with torch.no_grad():
                            for step_i in range(n_steps):
                                t_val = t_grid[step_i]
                                at_cur, st_cur = get_alpha_sigma(t_val)
                                t_tensor = torch.tensor([t_val] * this_bs, dtype=torch.long, device=self.device)
                                xc = comfy_input_transform(x_t, st_cur)

                                if p_ctx.shape[1] != n_ctx.shape[1]:
                                    max_len = max(p_ctx.shape[1], n_ctx.shape[1])
                                    for target_ctx in [p_ctx, n_ctx]:
                                        if target_ctx.shape[1] < max_len:
                                            pad = torch.zeros((target_ctx.shape[0], max_len - target_ctx.shape[1], target_ctx.shape[2]), 
                                                             device=target_ctx.device, dtype=target_ctx.dtype)
                                            if target_ctx is p_ctx: p_ctx = torch.cat([p_ctx, pad], dim=1)
                                            else: n_ctx = torch.cat([n_ctx, pad], dim=1)

                                batched_x = torch.cat([xc, xc], dim=0)
                                batched_t = torch.cat([t_tensor, t_tensor], dim=0)
                                batched_ctx = torch.cat([p_ctx, n_ctx], dim=0)
                                batched_y = torch.cat([p_y, n_y], dim=0)
                                out = teacher.forward(batched_x, batched_t, batched_ctx, batched_y)
                                out_pos, out_neg = out.chunk(2)
                                raw = out_neg + (out_pos - out_neg) * cfg_val
                                
                                at_t = torch.tensor([at_cur.item()] * this_bs, device=self.device, dtype=torch.bfloat16).view(-1, 1, 1, 1)
                                st_t = torch.tensor([st_cur.item()] * this_bs, device=self.device, dtype=torch.bfloat16).view(-1, 1, 1, 1)
                                denoised = raw_to_denoised(raw, x_t, at_t, st_t, model_type)
                                target = raw_to_target(raw, x_t, at_t, st_t, model_type, model_type)
                                target_p = raw_to_target(out_pos, x_t, at_t, st_t, model_type, model_type)
                                target_n = raw_to_target(out_neg, x_t, at_t, st_t, model_type, model_type)

                                for b in range(this_bs):
                                    batch_trajectories[b].append({
                                        "t": t_val, "x_t": x_t[b:b+1].cpu().float().contiguous(),
                                        "target": target[b:b+1].cpu().float().contiguous(),
                                        "target_p": target_p[b:b+1].cpu().float().contiguous(),
                                        "target_n": target_n[b:b+1].cpu().float().contiguous(),
                                        "at": at_cur.item(), "st": st_cur.item()
                                    })
                                
                                st_next = get_alpha_sigma(t_grid[step_i + 1])[1]
                                x_t = x_t + ((x_t - denoised) / st_cur) * (st_next - st_cur)
                                del out, out_pos, out_neg, raw, denoised, target, target_p, target_n, t_tensor, xc, at_t, st_t

                            # NOTE: we intentionally do NOT append a t=0 "final image" sample.
                            # At t=0 sigma≈0.029 — the latent is already nearly clean — and the
                            # model output there is numerically unstable.  Adding it would inject
                            # noise into the training signal without useful distillation content.

                        for b in range(this_bs):
                            # Stack entire trajectory into sequence tensors
                            xt_seq = torch.cat([s["x_t"] for s in batch_trajectories[b]], dim=0)
                            p_seq = torch.cat([s["target_p"] for s in batch_trajectories[b]], dim=0)
                            n_seq = torch.cat([s["target_n"] for s in batch_trajectories[b]], dim=0)
                            t_grid_full = [s["t"] for s in batch_trajectories[b]]
                            meta_list = [{"at": s["at"], "st": s["st"]} for s in batch_trajectories[b]]

                            traj_id_in_shard = shard_writer.add_compressed_trajectory(
                                xt_seq, p_seq, n_seq, t_grid_full, meta_list
                            )
                            
                            # Use the 0-timestep latent for preview (deferred)
                            cleanest = batch_trajectories[b][-1]["x_t"]
                            rel_preview_path = f"previews/traj_{uuid.uuid4().hex[:16]}.webp"
                            preview_tasks.append((cleanest, dataset_root / rel_preview_path))

                            pending_trajs.append({
                                "shard_index": traj_id_in_shard,
                                "sample_count": len(t_grid_full),
                                "seed": batch_base_seed + b,
                                "prompt": p_text,
                                "preview_path": str(rel_preview_path),
                                "metadata_json": json.dumps({"cfg": cfg_val, "neg": n_text, "batch_idx": b, "compressed": True, "type": "good", "model_type": model_type})
                            })
                            
                            traj_idx += 1
                            pbar.update(1)
                            if task_id: update_task_progress(dataset_root / "metadata.db", task_id, traj_idx)
                        
                        trajs_in_shard += this_bs
                        if trajs_in_shard >= _MAX_TRAJS_PER_SHARD:
                            self._finalize_shard(dataset_root, shard_writer, shard_file, source_id, pending_trajs)
                            shard_file = dataset_root / "staging" / f"teacher_{uuid.uuid4().hex[:12]}.safetensors"
                            shard_writer = ShardWriter(shard_file)
                            pending_trajs = []
                            trajs_in_shard = 0
                        
                        del n_ctx, n_y, p_ctx, p_y

            if dup_count:
                print(f"  Warning: {dup_count} trajectory(s) have the same (prompt, seed) as existing data. "
                      f"Consider stopping if this is unintended.")

            if pending_trajs:
                self._finalize_shard(dataset_root, shard_writer, shard_file, source_id, pending_trajs)

            # Batch generate previews after the main loop
            if preview_tasks:
                print(f"  Generating {len(preview_tasks)} previews...")
                for latent, preview_path in preview_tasks:
                    previewer.generate_preview(latent, preview_path)
            previewer.free(); del teacher, text_cache, previewer; xpu_empty_cache(); gc.collect()
            if task_id: update_task_status(dataset_root / "metadata.db", task_id, 'finished')

        except Exception as e:
            if task_id: update_task_status(dataset_root / "metadata.db", task_id, 'failed', error=str(e))
            raise e

    def _preprocess_image(self, img_path: Path, px: int, resize_mode: str) -> torch.Tensor:
        """Load and preprocess an image, returning a (3, H, W) tensor on CPU.

        resize_mode:
          "fit"         — resize so smallest dim = px preserving aspect ratio,
                          then crop ≤7 px from longer side to make divisible by 8.
                          Result may be non-square — no distortion, minimal data loss.
          "center_crop" — resize so smallest dim = px, then center-crop to px×px square.
          "pad"         — resize so largest dim = px preserving aspect ratio,
                          then pad to px×px square with black borders.
        """
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if resize_mode == "fit":
            scale = px / min(w, h)
            new_w, new_h = round(w * scale), round(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Crop up to 7 px from the longer side to be divisible by 8
            adj_w = new_w - (new_w % 8)
            adj_h = new_h - (new_h % 8)
            left = (new_w - adj_w) // 2
            top = (new_h - adj_h) // 2
            img = img.crop((left, top, left + adj_w, top + adj_h))
        elif resize_mode == "center_crop":
            scale = px / min(w, h)
            new_w, new_h = round(w * scale), round(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            left = (new_w - px) // 2
            top = (new_h - px) // 2
            img = img.crop((left, top, left + px, top + px))
        elif resize_mode == "pad":
            scale = px / max(w, h)
            new_w, new_h = round(w * scale), round(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            padded = Image.new("RGB", (px, px), (0, 0, 0))
            left = (px - new_w) // 2
            top = (px - new_h) // 2
            padded.paste(img, (left, top))
            img = padded
        else:  # "resize" (old stretch — distort to square)
            img = img.resize((px, px), Image.Resampling.LANCZOS)

        return torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)

    def run_ingestion_task(self, dataset_root: Path, model_path: Path, image_dir: Path,
                           latent_size: int = 64, recursive: bool = True,
                           resize_mode: str = "resize",
                           n_timesteps: int = 20,
                           cfg_range: tuple = (1.0, 7.5),
                           neg_prompt: str = "",
                           model_type: str = "eps",
                           seed: int = 42,
                           task_id: int = None,
                           t_mode: str = "uniform",
                           t_low: int = 20,
                           t_high: int = 999):
        """VAE-encode real images and store denoising reconstruction targets.

        Each image is encoded to x0, then noised at n_timesteps uniformly-spaced
        timesteps.  The reconstruction target at each step is the true noise eps
        (for eps models) or the v-prediction equivalent (for vpred models).
        This is standard diffusion fine-tuning — no teacher forward is needed,
        and the loss starts near 1.0 rather than near 0.

        Previously the teacher was run on each noised latent to produce targets.
        Since the teacher is the same model as the LoRA base, its outputs are
        identical to the student's outputs at LoRA init (lora_B=0), producing a
        loss of zero from step 0 with no gradient signal to drive learning.

        resize_mode:
          "fit"         — resize so smallest dim = px*8 preserving aspect ratio,
                          then crop to be divisible by 8. Output may be non-square.
          "center_crop" — resize so smallest dim = px*8, then center-crop to square.
          "pad"         — resize so largest dim = px*8, pad to square with black.
          "resize"      — hard stretch to (px*8, px*8), distorts aspect ratio.

        model_type: "eps" or "vpred" — must match the checkpoint used for training.
        n_timesteps: number of timesteps sampled per image (default 20).
        t_mode: distribution to sample from (uniform, logit, low, mid, high).
        t_low, t_high: bounds for timestep sampling.
        cfg_range: kept for API compatibility; not used (no teacher forward).
        neg_prompt: negative prompt stored in metadata for training conditioning.
        """
        try:
            if task_id: update_task_progress(dataset_root / "metadata.db", task_id, 0, pid=os.getpid())

            source_id = add_source(dataset_root / "metadata.db", f"{image_dir.name}_real", "real",
                                str(model_path), {
                                    "image_dir": str(image_dir), "recursive": recursive,
                                    "resize_mode": resize_mode, "model_type": model_type,
                                    "n_timesteps": n_timesteps,
                                    "t_mode": t_mode, "t_low": t_low, "t_high": t_high,
                                })

            print(f"  Loading VAE from: {model_path.name}")
            sd = load_file(str(model_path))
            vae_sd = {k.replace("first_stage_model.", ""): v
                    for k, v in sd.items() if k.startswith("first_stage_model")}
            vae = VAEDecoder.from_vae_sd(vae_sd, device=self.device)
            previewer = PreviewGenerator(self.device, vae_sd)
            del sd; xpu_empty_cache(); gc.collect()

            px = latent_size * 8

            img_exts = {".png", ".jpg", ".jpeg", ".webp"}
            if recursive:
                image_files = [p for p in image_dir.glob("**/*")
                               if p.is_file() and p.suffix.lower() in img_exts]
            else:
                image_files = [p for p in image_dir.glob("*")
                               if p.is_file() and p.suffix.lower() in img_exts]

            # Read per-image prompts from .txt sidecar files
            image_prompts = []
            for img_path in image_files:
                cap_path = img_path.with_suffix(".txt")
                image_prompts.append(cap_path.read_text().strip() if cap_path.exists() else "")

            rng = random.Random(seed)

            # Generate timestep grid
            if t_mode == "uniform":
                # Uniformly-spaced timesteps with jitter
                # (t=0 excluded — sigma≈0 produces unreliable targets)
                # A small random jitter (±half the step size) is added so the model
                # sees a spread of timestep values rather than the same 20 discrete
                # points every epoch — this improves generalisation to arbitrary
                # inference timesteps that fall between the training grid points.
                _step = (t_high - t_low) / max(n_timesteps - 1, 1)
                _half = _step / 2.0
                t_grid = []
                for i in range(n_timesteps):
                    base = t_high - i * _step
                    jitter = rng.uniform(-_half, _half) if i not in (0, n_timesteps - 1) else 0.0
                    t_grid.append(int(round(max(t_low, min(t_high, base + jitter)))))
            else:
                # Distribution-based sampling (logit, low, mid, high)
                t_grid_raw = []
                for _ in range(n_timesteps):
                    t_grid_raw.append(sample_timestep(rng, t_mode, t_low, t_high))
                # Sort descending to maintain consistent order in shards
                t_grid = sorted(t_grid_raw, reverse=True)

            shard_file = dataset_root / "staging" / f"real_{uuid.uuid4().hex[:12]}.safetensors"
            shard_writer = ShardWriter(shard_file)
            pending_trajs = []
            trajs_in_shard = 0
            preview_tasks = []

            for i, img_path in enumerate(tqdm(image_files, desc="Ingest")):
                try:
                    img_tensor = self._preprocess_image(img_path, px, resize_mode)
                    with torch.no_grad():
                        x0 = vae.encode(img_tensor).to(device=self.device, dtype=torch.float32)
                    del img_tensor

                    prompt = image_prompts[i]

                    traj_samples = []
                    for t_val in t_grid:
                        at_cur, st_cur = get_alpha_sigma(t_val)
                        at_f = at_cur.item()
                        st_f = st_cur.item()

                        # Sample noise and compute noised latent
                        cpu_gen = torch.Generator(device="cpu")
                        cpu_gen.manual_seed(derive_seed(seed, i * len(t_grid) + t_val, "ingest_noise"))
                        eps = torch.randn(x0.shape, generator=cpu_gen,
                                         device="cpu", dtype=torch.float32)

                        x_t = x0.cpu() + st_f * eps

                        # Reconstruction target: the true noise we added.
                        # For eps model: target = eps.
                        # For vpred model: target = eps_to_vpred(eps, x_t, alpha, sigma),
                        # i.e. (eps - sigma*x0) / sqrt(sigma^2+1) -- see the derivation
                        # in noise_schedule.py. This matches ComfyUI's V_PREDICTION
                        # class under this codebase's x_t = x0 + sigma*eps forward
                        # process. NOTE: this is *not* the textbook DDPM v-formula
                        # (v = alpha*eps - sigma*x0), which assumes a different
                        # forward process (x_t = alpha*x0 + sigma*eps) and gives a
                        # wrong target here, especially at high-noise timesteps
                        # where alpha is small.
                        if model_type == "vpred":
                            target = eps_to_vpred(eps, x_t, at_cur, st_cur)
                        else:
                            target = eps

                        traj_samples.append({
                            "t": t_val,
                            "x_t": x_t.contiguous(),
                            "target": target.contiguous(),
                            "target_p": target.contiguous(),
                            "target_n": target.contiguous(),
                            "at": at_f, "st": st_f,
                        })
                        del eps, x_t, target

                    # Stack into compressed format
                    xt_seq = torch.cat([s["x_t"]      for s in traj_samples], dim=0)
                    p_seq  = torch.cat([s["target_p"] for s in traj_samples], dim=0)
                    n_seq  = torch.cat([s["target_n"] for s in traj_samples], dim=0)
                    t_list = [s["t"]  for s in traj_samples]
                    m_list = [{"at": s["at"], "st": s["st"]} for s in traj_samples]

                    traj_id = shard_writer.add_compressed_trajectory(xt_seq, p_seq, n_seq,
                                                                      t_list, m_list)
                    # Least-noisy step (last in t_grid) used as preview
                    rel_preview = f"previews/real_{uuid.uuid4().hex[:16]}.webp"
                    preview_tasks.append((traj_samples[-1]["x_t"], dataset_root / rel_preview))

                    pending_trajs.append({
                        "shard_index": traj_id,
                        "sample_count": len(t_list),
                        "seed": seed + i,
                        "prompt": prompt,
                        "preview_path": rel_preview,
                        "metadata_json": json.dumps({
                            "neg": neg_prompt,
                            "compressed": True, "type": "good",
                            "model_type": model_type,
                        }),
                    })
                    trajs_in_shard += 1
                    if trajs_in_shard >= _MAX_TRAJS_PER_SHARD:
                        self._finalize_shard(dataset_root, shard_writer, shard_file,
                                             source_id, pending_trajs)
                        shard_file = dataset_root / "staging" / f"real_{uuid.uuid4().hex[:12]}.safetensors"
                        shard_writer = ShardWriter(shard_file)
                        pending_trajs = []
                        trajs_in_shard = 0
                    if task_id: update_task_progress(dataset_root / "metadata.db", task_id, i + 1)
                    del x0, xt_seq, p_seq, n_seq, traj_samples

                except Exception as e:
                    print(f"    Failed {img_path.name}: {e}")

            if pending_trajs:
                self._finalize_shard(dataset_root, shard_writer, shard_file,
                                     source_id, pending_trajs)

            if preview_tasks:
                print(f"  Generating {len(preview_tasks)} previews...")
                for latent, preview_path in preview_tasks:
                    previewer.generate_preview(latent, preview_path)

            vae.free(); previewer.free(); del vae, previewer
            xpu_empty_cache(); gc.collect()
            if task_id: update_task_status(dataset_root / "metadata.db", task_id, 'finished')

        except Exception as e:
            if task_id: update_task_status(dataset_root / "metadata.db", task_id, 'failed', error=str(e))
            raise e