#!/bin/sh
# usage: REPO=/path/to/B580-diffusion-training sh run_all.sh      (needs: torch diffusers transformers safetensors)
for t in unet_vs_diffusers lora_export_vs_merge grad_ckpt_vs_plain vae_vs_diffusers clip_vs_hf noise_vs_ddpm; do
  echo "=================== $t"; python3 -W ignore $t.py 2>&1 | grep -v -i "warn\|deprecat" | tail -12
done
