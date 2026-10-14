# Lessons carried over: CLIP, VAE, schedules

Facts that the old comments recorded and the new code must keep honouring. Measured on the B580 unless noted.

## Text encoders
* Batched CLIP in float16 diverges from encoding prompts one at a time by 19% of the mean activation magnitude; in float32 it is 0.17% (1.5e-6 of the largest activation). Cache warming and the per-prompt miss path must agree, so batched encoding is valid only in float32.
* Float32 costs about 3.1 GB while both towers are resident. Batch 64 in float32: 10.99 ms/prompt, against 32.2 ms/prompt serial in float16 (5,000 prompts: 55 s versus 2.7 min). Neither is a bottleneck for ordinary datasets.
* A single-row encode and a batch-of-one encode agree to 1.2e-7 relative.
* Conditioning is the penultimate layer without the final norm; pooled is the final-normed state at the first EOS, projected. Wrong choices load and run without any error.
* CLIP-L pads with EOS, CLIP-G pads with 0; both share one vocabulary.
* Token weights (`(word:1.2)`) were a ComfyUI wildcard-syntax feature and are not supported; a weight other than 1.0 must be refused at the boundary.

## VAE
* The downsample pads asymmetrically (0,1,0,1); symmetric padding shifts the image half a pixel per level.
* Attention treats channels as features and H*W as sequence; the middle block always has attention.
* SDXL latent scale is 0.13025, ddconfig `ch=128, ch_mult=(1,2,4,4), num_res_blocks=2`.
* Loading a VAE non-strictly hid mismatches in the old code; the codec now refuses missing or unexpected keys.

## Schedules
* SDXL's scaled-linear schedule gives sigma_max 14.6146. Zero-terminal-SNR makes sigma infinite at the last step and pairs only with v-prediction.
* The input scaling is `x / sqrt(sigma^2 + 1)`.

## Tokenizer
* The BPE data (`vocab.json`, `merges.txt`) is not in the repository; it is found through `CLIP_TOKENIZER_DIR`, `assets/clip_tokenizer`, or `$COMFY_DIR/sd1_tokenizer`.
* The old tokenizer and the new one agree token for token on a synthetic vocabulary here; the real vocabulary comparison is run on the owner's machine (`scripts/compare_tokenizers.py`).

## Adapters
* The old `LoRAConv2d` ignored the timestep gate while `LoRALinear` and the DoRA layers applied it. The new layers apply the gate to every adapter kind; this is a deliberate change, not a port.
* Adapter factors are float32 even when the base is bfloat16: a bfloat16 adapter rounds small updates to a standstill.
* Base weights are buffers, not parameters, so the frozen model never enters the optimizer or autograd.
