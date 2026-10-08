#!/usr/bin/env bash
# A2: the same images through kohya sd-scripts, matched to this trainer.
#
# The task calls this "the single most informative missing fact": every
# number so far measures this project against itself or against diffusers
# component-by-component, and none of it can say whether kohya -- a
# trainer that demonstrably produces LoRAs the maintainer likes -- turns
# these same images into a good LoRA. If it does, the defect is in this
# trainer. If it does not, the defect is in the data.
#
# Matched deliberately, because an unmatched comparison proves nothing:
#   pretrained_model_name_or_path
#                        div_4.safetensors      (same base; this kohya
#                                             version renamed the flag)
#   network_dim       64                     (same as Test_02)
#   network_alpha     32                     (same; the forensics show
#                                             Test_02 at scale 1.000)
#   resolution        512,512                (the modal training width in
#                                             this dataset, NOT SDXL native)
#   enable_bucket     yes, bucket_no_upscale  (required: source images are up
#                                             to 2480x3508 and Kohya raises
#                                             "image too large" without it)
#   min/max_timestep  1 / 999                (this trainer's t_low/t_high)
#   max_train_epochs  1, --train_batch_size 2
#   learning_rate     1e-4                   (the trainer's default)
#
# Resolution is the one deliberate concession: kohya buckets to a fixed
# resolution, and this dataset's native shapes are odd (720x512, 392x512),
# so matching exact shapes is not expressible. 512x512 is the modal width
# and is what a kohya user would pick for this data by default. That
# difference is recorded, not hidden.
#
# Empty captions are given a placeholder rather than left blank: kohya
# skips the caption entirely when the field is empty, which is not what
# this trainer does (it encodes ""). Using a fixed neutral caption also
# matches the intended usage -- sample with the same fixed neutral text.
set -euo pipefail

KOHYA=/home/okolenmi/Desktop/kohya/kohya_ss/sd-scripts
PY=/home/okolenmi/comfy/venv/bin/python
REPO=/home/okolenmi/Desktop/B580-diffusion-training
OUT="$REPO/runs/a2_kohya"
IMG=/home/okolenmi/Downloads/datas

# Kohya wants one caption file per image, named <stem>.txt, beside it.
# The originals are read-only and must not be written to, so the dataset
# is assembled by symlink into a scratch dir with captions alongside.
WORK="$OUT/images"
mkdir -p "$WORK"
CAPTION="style"

# Kohya's FineTuningDataset reads its file list from a JSONL metadata
# file (--in_json), one {"image_path", "caption"} per line. It does NOT
# scan a directory, and --caption_extension applies to ControlNet
# datasets, not to this path. Sidecar .txt files would be silently
# ignored, so the listing has to be built explicitly.
META="$OUT/metadata.jsonl"
: > "$META"
count=0
while IFS= read -r -d '' f; do
    stem="$(basename "$f")"
    ln -sf "$f" "$WORK/$stem"
    printf '{"image_path": "%s/%s", "caption": "%s"}\n' "$WORK" "$stem" "$CAPTION" >> "$META"
    count=$((count + 1))
done < <(find "$IMG" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) -print0)

echo "staged $count images with caption '$CAPTION' in $WORK (metadata: $META)"

cd "$KOHYA"
exec "$PY" sdxl_train_network.py \
    --pretrained_model_name_or_path=/home/okolenmi/comfy/ComfyUI/models/checkpoints/div_4.safetensors \
    --train_data_dir="$WORK" \
    --in_json="$META" \
    --output_dir="$OUT" \
    --output_name=a2_kohya \
    --network_dim=64 \
    --network_alpha=32 \
    --resolution=512,512 \
    --enable_bucket \
    --bucket_no_upscale \
    --min_timestep=1 \
    --max_timestep=999 \
    --train_batch_size=2 \
    --max_train_epochs=1 \
    --learning_rate=1e-4 \
    --optimizer_type=adamw \
    --mixed_precision=bf16 \
    --full_bf16 \
    --seed=1234 \
    "$@"
# --full_bf16 is here for VRAM, not tuning. Without it this OOMs on a
# 12,216 MB B580 at the very first backward:
#   RuntimeError: level_zero backend failed with error: 40
#   (UR_RESULT_ERROR_OUT_OF_RESOURCES)
# Kohya keeps the UNet plus both CLIPs plus the VAE resident, and the UNet
# alone is ~10.3 GB in fp32, so bf16 weights are what makes it fit. This is
# the same pressure this project's own trainer relieves structurally instead
# (it unloads the text encoder for the whole run, -1,561 MB measured).
#
# --gradient_checkpointing is deliberately NOT used: enabling it walks into
# another broken path, `prepare_text_encoder_grad_ckpt_workaround` doing
# `text_encoder.text_model.embeddings.requires_grad_(True)` on a bare
# CLIPTextModel that has no `.text_model` attribute ->
#   AttributeError: 'CLIPTextModel' object has no attribute 'text_model'
# --full_bf16 alone is what fits, so the checkpointing trade is not needed.
#
# No --cache_device: it does not exist in this kohya version. Verified by
# grepping library/args.py, not assumed.
#
# No device flag: there is no --xpu. library/device_utils.py's
# get_preferred_device() picks cuda, then xpu, then mps, then cpu, and
# the training scripts use accelerator.device rather than asking. On this
# box HAS_CUDA is False and HAS_XPU is True, so it lands on xpu on its
# own. That was checked in the source rather than assumed, because a
# wrong flag here would not fail loudly -- it would just not exist.