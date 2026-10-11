#!/usr/bin/env python3
"""Train the A2-matched config with THIS trainer and save the LoRA.

Why this exists: `scripts/hw_validate.py` trains and measures, but it
never calls save_trained_weights(), so the matched 201-step run left no
file to compare against the Kohya one. Everything needed is already in
the graph -- LoRACheckpointSaverNode and lora_saver.save_trained_weights()
both exist and share one implementation -- so this drives the same nodes
the validated runs use and adds the save.

The config is deliberately identical to the Kohya arm
(`a2_minimal.py`): same dataset, batch 1, rank 64, alpha 32, lr 1e-4,
AdamW, 201 steps, UNet attention only (this trainer's default target set
is to_q/to_k/to_v/to_out.0, the same modules Kohya's
--network_train_unet_only keeps). So the two LoRAs differ in
implementation, not in settings.

One asymmetry, recorded rather than hidden: Kohya resizes to a bucket
(384x384 here) while this trainer uses the dataset's native shapes
(64x48 latent = 512x384, and the dataset is non-square). Same images,
different geometry. That affects loss comparability, not the value of
having both files to look at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# parents[2]: this file was archived from bad_LoRA_investigation/ (one level
# deeper) to archive/bad-LoRA-investigation/ on 2026-10-09; parents[1] is
# now the archive folder, not the repo root.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="non-square")
    ap.add_argument("--steps", type=int, default=201)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "constant"],
                    help="cosine = CosineLRSchedule decaying to 5%% (the default "
                         "this trainer ships); constant = flat lr, matching "
                         "kohya's --lr_scheduler=constant default. The A1 "
                         "comparison showed cosine undertrains ~1.5-1.8x vs "
                         "kohya's flat schedule at the same nominal lr.")
    ap.add_argument("--out", default="a2match_mine")
    ap.add_argument("--target-conditioning-path", type=lambda s: s.lower() != "false",
                    default=True,
                    help="False excludes time_embed.* and label_emb.* (564 -> 560 "
                         "modules), matching what kohya's --network_train_unet_only "
                         "actually trains. A/B switch from "
                         "conditioning_path_switch.patch.")
    ap.add_argument("--target-modules", default="attention",
                    choices=["attention", "kohya_default", "kohya_plus",
                             "kohya_default,label_emb"],
                    help="attention (default): to_q/to_k/to_v/to_out.0 only (560 "
                         "modules + cond path = 564). kohya_default: also "
                         "ff.net.0.proj, ff.net.2, proj_in, proj_out (722 + cond "
                         "path), matching what kohya sd-scripts' default SDXL "
                         "LoRA adapts. kohya_plus: same UNet set with the "
                         "conditioning-path embeddings kept (726 with the "
                         "default --target-conditioning-path). "
                         "kohya_default,label_emb (724): 722 plus label_emb "
                         "alone (cond path off) -- isolates label-vs-time "
                         "conditioning. Targets A/B from "
                         "lora_targets_and_cond_switch.patch.")
    ap.add_argument("--progress", type=int, default=0,
                    help="print one loss line every N steps (flush=True), so a "
                         "wedged run shows exactly where it stopped. 0 = silent.")
    ap.add_argument("--unload-encoder-after", type=int, default=0,
                    help="N > 0: call encoder.unload() after step N (towers to "
                         "CPU, ~1.5 GB freed). Later steps run on cache hits, "
                         "which never touch the towers, so training math is "
                         "identical -- a genuinely new key after unload "
                         "degrades to a slow CPU encode, never a wrong one. "
                         "0 (default) = historical resident behaviour. Added "
                         "for the 722/726 B arms: +78M trainable params push "
                         "sustained VRAM to ~11.2 GB, where this card wedges "
                         "instead of raising OOM.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save a checkpoint LoRA every N steps as "
                         "<out>_stepNNNNN.safetensors (plus the final save). "
                         "Killed runs keep their last checkpoint instead of "
                         "nothing -- added after arm 3 died at step 350 "
                         "unsaved. 0 = final save only.")
    ap.add_argument("--prewarm-unload", action="store_true",
                    help="Encode every distinct (prompt, bucket shape) in the "
                         "dataset up front at the training batch size, then "
                         "unload the towers BEFORE step 0. Same math (later "
                         "steps are cache hits either way), but the 1.5 GB is "
                         "free from the first update -- needed because the "
                         "first optimizer step itself can OOM at ~11.9 GB "
                         "occupied, before any after-N unload could fire.")
    args = ap.parse_args()

    from nodes.core import ExecutionContext
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.lora_injector import ComfyUNetLoRANode
    from nodes.model.lora_saver import save_trained_weights
    from nodes.model.parameters import ModelParametersNode
    from nodes.model.text_encoder import SDXLTextEncoderNode
    from nodes.model.text_encoder_cache import CachingTextEncoderNode
    from nodes.train.schedule import ConstantLRScheduleNode, CosineLRScheduleNode
    from nodes.train.supervised import SupervisedLoRATrainerNode
    from scripts.hw_validate import MemProbe

    ctx = ExecutionContext()
    probe = MemProbe()
    # Bare filename, not an absolute path: both the checkpoint loader and
    # save_trained_weights resolve model paths *relative* to a sandboxed
    # base dir (paths.get_checkpoints_dir() / get_loras_dir(), both of
    # which are ComfyUI's own models/ subdirs). An absolute path raises
    # "Invalid relative path". Useful side effect: the LoRA lands directly
    # in ComfyUI's loras/ folder, so there is nothing to copy afterwards.
    weights = SafetensorsCheckpointNode(ctx).build(path="div_4.safetensors")["weights"]

    from nodes.dataset.managed import ManagedDatasetSourceNode
    batches = ManagedDatasetSourceNode(ctx).build(
        dataset_root=args.dataset, batch_size=args.batch, shuffle=True,
        keep_incomplete_batches=False, shape_bucket_multiple=0)["batches"]

    if args.schedule == "constant":
        schedule = ConstantLRScheduleNode(ctx).build(lr=args.lr)["schedule"]
    else:
        schedule = CosineLRScheduleNode(ctx).build(
            lr=args.lr, total_steps=args.steps)["schedule"]
    model = ComfyUNetLoRANode(ctx).build(
        weights=weights, rank=args.rank, alpha=args.alpha,
        use_checkpoint=True, target_modules=args.target_modules,
        target_conditioning_path=args.target_conditioning_path)["model"]
    encoder = CachingTextEncoderNode(ctx).build(
        encoder=SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    )["encoder"]
    params = ModelParametersNode(ctx).build(model=model)["params"]
    from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
    optimizer = ComposedAdamWOptimizerNode(ctx).build(
        params=params, lr=args.lr, state_precision="float32")["optimizer"]

    if args.prewarm_unload:
        # Warm every distinct (prompt, bucket shape) at the training batch
        # size straight from trajectory metadata (no latent loads), then
        # unload the towers before step 0. Later steps are cache hits
        # either way; this only moves the 1.5 GB earlier, ahead of the
        # first optimizer update that otherwise OOMs near the ceiling.
        loader = batches._loader
        by_shape: dict[tuple[int, int], None] = {}
        prompts: dict[str, None] = {}
        for t in loader.trajectories:
            h, w = int(t.get("latent_h") or 0), int(t.get("latent_w") or 0)
            if h and w:
                by_shape[loader._bucket_size(h, w)] = None
            if t.get("prompt") is not None:
                prompts[t["prompt"]] = None
        n = 0
        for p in prompts:
            for (H, W) in by_shape:
                encoder.encode(p, batch_size=args.batch,
                               height=H * 8, width=W * 8)
                n += 1
        encoder.unload()
        # Towers moved to CPU frees allocations, but the caching
        # allocator keeps the reserved pool -- without this the card
        # sits at its old ceiling and the pressure never lifts.
        import torch
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        print(f"  [encoder] prewarmed {n} key(s), unloaded before step 0",
              flush=True)

    print(f"training {args.steps} steps, batch {args.batch}, "
          f"rank {args.rank}, alpha {args.alpha}, lr {args.lr} "
          f"({args.schedule}), targets={args.target_modules}, "
          f"target_conditioning_path={args.target_conditioning_path}",
          flush=True)

    def on_step(step, loss, shape=None):
        # step_notify calls back as (step, loss[, shape]); unbuffered.
        if args.progress and isinstance(step, int) and step % args.progress == 0:
            print(f"  [progress] step {step} loss={loss:.5f} shape={shape}",
                  flush=True)
        if args.save_every and isinstance(step, int) and step > 0 \
                and step % args.save_every == 0:
            ckpt = save_trained_weights(model, f"{args.out}_step{step:05d}.safetensors")
            print(f"  [checkpoint] step {step} -> {ckpt}", flush=True)
        if args.unload_encoder_after and step == args.unload_encoder_after:
            before = encoder.footprint_bytes() if hasattr(
                encoder, "footprint_bytes") else -1
            encoder.unload()
            # Moving towers to CPU frees allocations but the caching
            # allocator keeps the reserved pool -- without this the card
            # stays at its ceiling and the pressure that wedges
            # near-full runs never lifts. Safe here: between steps, only
            # referenced tensors survive (grads/params/statics all live).
            import torch
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
            print(f"  [encoder] unloaded after step {step} "
                  f"(footprint was {before / 1024 / 1024:.0f} MB)", flush=True)

    SupervisedLoRATrainerNode(ctx).build(
        model=model, batches=batches, optimizer=optimizer, text_encoder=encoder,
        lr_schedule=schedule, steps=args.steps, on_step=on_step, profile=False)

    path = save_trained_weights(model, f"{args.out}.safetensors")
    print(f"saved LoRA -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())