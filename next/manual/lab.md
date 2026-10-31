# The lab

`python -m next.lab` compares variants of a training on the same footing: it trains each variant (an *arm*) in its own process, renders the same images with each result, measures what changed, and writes one report.

## Commands
- `python -m next.lab template > experiment.json` prints an experiment file to edit.
- `python -m next.lab run experiment.json [--phase train render report] [--arms a b] [--force]` runs it. Finished work is kept and skipped on the next run (`--force` redoes it), so a stopped run continues where it was.
- `python -m next.lab render --models M --file CKPT --lora a.safetensors b.safetensors --prompt "..." --seeds 1 2 --strengths 0.5 1 --out DIR` renders a grid with adapter files, without training.
- `python -m next.lab engines --models M --file CKPT --comfy <ComfyUI folder> [--comfy-python PY] [--lora L]` renders the same recipes with this project and with ComfyUI and compares the images (is the renderer the same as ComfyUI's?).

## The experiment file
`train_args` are given to every arm (`python -m next.run sdxl-lora` arguments, `--set node.port=value` included); each arm adds its own `args`. `product` lists groups of choices and adds every combination (names joined with `+`). `grid` says which images to render (prompts, seeds, strengths, steps, cfg, sampler `euler`/`dpmpp_2m`, scheduler `normal`/`simple`/`karras`, size). `images.train` and `images.holdout` are folders of images (the training ones, and ones the training never saw) for the loss by timestep.

## What is kept (`<output>/<experiment>/`)
`experiment.json`; `reference/images` (the grid without any adapter); per arm: `train.log`, `train.json` (command, losses, probe lines), `train/lora_stepNNNNNN.safetensors`, `images/*.png` with `index.json`, `timestep_loss.json`, `image_metrics.json`; `report.md`, `sheets/`, and `<experiment>-share.zip` (report, measurements, logs and the sheets as JPEG, under 25 MB).
Every PNG carries a `lab` text chunk (JSON: the recipe, the arm and its arguments, the shared training arguments, the checkpoint, the adapter file with its hash, step and rank, the time) and an A1111-style `parameters` chunk; `ImageStore.read_meta(path)` reads it.

## What the report measures
- **Detail** of each image from the radial power spectrum of the luma: the spectral slope (natural photographs are near -2; more negative = less fine detail), the share of power in the two finest octaves, the variance of the Laplacian (sharpness). Against the reference image of the same prompt and seed: the finest octave and the sharpness as ratios (below 1 = less fine detail), the coarse change (the picture reduced 16 times) and the fine change (what a blur leaves out), and their ratio.
- **Loss by timestep**: the squared error of the predicted noise at fixed timesteps on the same noise, with the adapter and without, on training and held-out images; the report prints adapter / base per timestep (above 1: worse than the base model there).
- **Probe lines** of the training (`trainer.probe_every`): adapter / base loss by third of the range.
- **Contact sheets**: per prompt, seed and strength with the reference first, and native-pixel crops of strength 1 (the size at which the eye sees the loss of detail).

## Rendering
The sampler follows ComfyUI's (k-diffusion) exactly: the same sigma tables, `normal`/`simple`/`karras` schedules, Euler and DPM++ 2M, the same noise for a seed, the nearest integer timestep given to the network. The UNet runs in bf16 (ComfyUI picks fp16 on its own), text encoding and VAE as in training. Strength scales the adapter's output. The network, the text encoders and the decoder are never on the device together; a decode that does not fit is done in overlapping tiles.


## Rendering without training: `lora`, `reuse`, `eps_scale`

An arm can be given an adapter file made earlier (`"lora": "/path/file.safetensors"`: rendered as it is, nothing is trained) or can render the adapter of an earlier arm (`"reuse": "<arm>"`, copies that arm's training record and per-timestep table). `"eps_scale": 1.01` divides the predicted noise (after guidance) by that factor at every sampling step (epsilon scaling, Ning et al. 2024: sampling errors make the prediction too large; a factor slightly above 1 pulls the trajectory back). Images of such an arm carry `_e<factor>` in the file name and the factor in their metadata. `python -m next.lab template eps-scale` starts an experiment of this kind.
