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

REPO = Path(__file__).resolve().parents[1]
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
                    choices=["attention", "kohya_default", "kohya_plus"],
                    help="attention (default): to_q/to_k/to_v/to_out.0 only (560 "
                         "modules + cond path = 564). kohya_default: also "
                         "ff.net.0.proj, ff.net.2, proj_in, proj_out (722 + cond "
                         "path), matching what kohya sd-scripts' default SDXL "
                         "LoRA adapts. kohya_plus: same UNet set with the "
                         "conditioning-path embeddings kept (726 with the "
                         "default --target-conditioning-path). Targets A/B from "
                         "lora_targets_and_cond_switch.patch.")
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

    print(f"training {args.steps} steps, batch {args.batch}, "
          f"rank {args.rank}, alpha {args.alpha}, lr {args.lr} "
          f"({args.schedule}), targets={args.target_modules}, "
          f"target_conditioning_path={args.target_conditioning_path}")
    SupervisedLoRATrainerNode(ctx).build(
        model=model, batches=batches, optimizer=optimizer, text_encoder=encoder,
        lr_schedule=schedule, steps=args.steps, on_step=None, profile=False)

    path = save_trained_weights(model, f"{args.out}.safetensors")
    print(f"saved LoRA -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())