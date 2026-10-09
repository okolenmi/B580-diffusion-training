# MEASURED: running kohya sd-scripts on Intel XPU (4 patches, 2 of them upstream bugs)

Task item: **A2** (`archive/bad-LoRA-investigation/TASK-lora-quality-and-multishape.md`).
Status: toolchain working. **No A2 verdict yet** — see "not yet measured".
Hardware: Intel Arc B580, 12,216 MB. torch 2.12.1+xpu, kohya at commit
`6721028c79ee85a78b3a06dfd8954dae310a1cce` (dev, 2026-06-16).

Getting A2 runnable took four fixes. Two are Intel/XPU porting, one is a
checkpoint-format gap, and **two are genuine upstream bugs that make
`sdxl_train_network.py` unrunnable as shipped.**

Patches: `kohya_xpu_patches.diff` (against the commit above, applies with
`git apply` inside `sd-scripts/`).

## 1. `accelerate` reads CUDA properties for an XPU device (XPU porting)

Every entry point died before any training code ran:

    AttributeError: 'torch._C._XpuDeviceProperties' object has no attribute 'major'

`Accelerator.__init__` calls `is_bf16_available()`, which does
`torch.cuda.get_device_properties(device).major`. On Intel,
`torch.cuda.get_device_properties` returns an `_XpuDeviceProperties` with
attributes `architecture, device_id, driver_version, has_fp64,
max_compute_units, ...` and **no `major`/`minor` at all** (verified by
enumerating `dir()`).

Initially patched `library/accelerator_setup.py` to answer it for the real
device. **Then dropped that patch**: fix 2 below makes it redundant,
because Kohya's own hijack already sets `torch.cuda.is_bf16_supported`.

## 2. The CUDA→XPU hijack aborts halfway on torch 2.12 (XPU porting)

Kohya calls `init_ipex()` early, which hijacks `torch.cuda` to mean XPU.
It is written as ~90 unguarded assignments in one block, so it is
**all-or-nothing**: on torch 2.12 `torch.xpu.Optional` no longer exists,
line 49 throws `AttributeError`, and every assignment after it is skipped.

The failure is *silent and partial*, which is what makes it expensive:
`torch.cuda.is_available()` (line 40, before the throw) now answers for
XPU, while `torch.Tensor.cuda` (line 42, before the throw — later
overwritten by `hijacks.py`) does not. Downstream, code that reads the
device as CUDA gets

    RuntimeError: PyTorch is not linked with support for cuda devices

from a torch that is working perfectly on XPU. That error names CUDA,
which is the one thing that is not happening, and sends you looking in the
wrong place entirely.

Patched each assignment to skip names absent on this build. Verified
after the fix:

    t.cuda()                        -> xpu:0
    t.to("cuda")                    -> xpu:0
    torch.zeros(4, device="cuda")   -> xpu:0
    torch.cuda.is_bf16_supported()  -> True

## 3. CLIP-L stored in diffusers layout fails to load (checkpoint format)

`div_4.safetensors` stores CLIP-L as

    conditioner.embedders.0.transformer.text_model.encoder.layers.N...

Kohya strips only `conditioner.embedders.0.transformer.`, leaving
`text_model.encoder.layers.N...`, but `CLIPTextModel` wants
`encoder.layers.N...`. Kohya's CLIP-L path otherwise assumes an **OpenCLIP**
checkpoint (`transformer.resblocks.*`, converted by `convert_sd_clip`), so
there is no conversion for this shape.

Symptom listed every key as both unexpected and missing — ~300 lines that
say nothing about the cause. Patch strips the `text_model.` prefix when
present, guarded on the keys actually looking like diffusers CLIP keys so
OpenCLIP checkpoints are untouched. Also needed `embeddings.position_ids`
dropped unprefixed: Kohya pops only the `text_model.`-prefixed spelling,
and that key is a buffer newer transformers derives rather than stores.

After both: `text encoder 1: <All keys matched successfully>`.

**A note on a wrong first attempt.** I first assumed a ComfyUI-native vs
A1111 prefix mismatch and renamed `conditioner.embedders.*` keys. That was
wrong — `div_4` already *is* in the A1111 CLIP layout
(`conditioner.embedders.0.transformer.` 197 keys,
`conditioner.embedders.1.model.` 390 keys), and the rename double-prefixed
to `text_model.text_model.*`. Reverted. Checked the keys directly instead
of reasoning about which convention "should" apply.

## 4. `sdxl_train_network.py` cannot run at all (UPSTREAM BUG)

    AttributeError: 'module '__main__' has no attribute 'create_network'

`train_network.py` does `importlib.import_module(args.network_module)` then
calls `create_network(...)`, but **no training script registers a
`--network_module` argument** — only the `*_gen_img.py` generators do. So
`args.network_module` is undefined and the shipped entry point crashes.

Setting it to `__name__` is also wrong: `import_module("__main__")` returns
a fresh empty module, not the running one. And this file has no module-level
`create_network` at all — it defines only `SdxlNetworkTrainer` and
`setup_parser`.

Correct value is **`"networks.lora"`**, the module whose `create_network`
has the signature `train_network.py` calls. With that, the network builds:

    create LoRA for U-Net: 722 modules.
    create LoRA for Text Encoder: 264 modules.

A LoRA user hitting this would conclude their install is broken; it is a
one-line omission at this commit.

## Also required (not bugs, just missing)

`sd-scripts/` is an **empty uninitialised submodule** — `kohya_ss` is only
the GUI, every training script lives in `sd-scripts`. Fixed with
`git submodule update --init --depth 1 sd-scripts`.

Missing Python deps, installed into this project's venv: `diffusers`,
`accelerate`, `opencv-python-headless`, `imagesize`, `voluptuous`.

Dataset wiring, worth recording because two of these are silent traps:
- Kohya's `FineTuningDataset` **reads its file list from a JSONL**
  (`--in_json`) and does **not** scan a directory. Sidecar `.txt` captions
  are ignored, and `--caption_extension` applies to ControlNet datasets, not
  this path.
- Without `--enable_bucket`, Kohya raises `AssertionError: image too
  large, but cropping and bucketing are disabled` on these sources (up to
  2480×3508).

## What this unblocked (A2 verdict, 2026-10-09)

**A2 is done; see `MEASURED-targets-and-schedule.md` for the numbers.**
Kohya trained to completion at rank 16/alpha 16 on the A1 single image
(500 steps, reproduces the scene) and ran the matched 201-step arm at
384px; this trainer ran the same arms. The comparison resolved to two
configuration gaps, both fixed and measured: cosine-vs-constant LR
schedule (~2x undertraining) and LoRA targets (560 vs kohya's 722;
`kohya_plus` 726 renders best). The residual structural difference in
§"Not yet measured" below (resize-to-bucket vs pad-to-bucket) stands as
recorded but did not block the verdict — A1 uses identical pixels.

## Not yet measured (original note, kept as written)

**This establishes that A2 is runnable. It is not an A2 result.** No kohya
LoRA has been trained to completion and none has been compared to anything.

Still open, and unchanged by any of the above:
- no image has been rendered, by either trainer
- the LoRA-quality question is unanswered; this only removes a blocker

The comparison is also not yet apples-to-apples, and one difference is
structural rather than a tuning choice: Kohya with `--enable_bucket`
**resizes** each image to fit a bucket, whereas this trainer stores
`resize_mode=fit` latents and **pads** to a bucket (shape bucketing). Same
images in, different geometry out. Recorded here so the A2 result is read
with it in mind.