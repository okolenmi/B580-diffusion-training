# To test on the B580

Everything the sandbox can check is checked there: `python run_tests.py` runs the trainer, backend, frontend and **browser** suites (the browser suite drives a real
server and Chromium on tiny CPU stand-ins; see `tests/README.md`). What is left is what only your card and real weights can answer. It is sorted by cost of being
wrong, not by when it was written. Do **Critical** and stop; the rest is optional and nothing is blocked by it. Delete an entry when it passes; if one fails, keep it
and paste the console lines. An AI on the PC can run most of it: `docs/tester-brief.md`.

## Critical (about 1 hour of your attention, plus one unattended run)

A change that breaks these breaks the foundation: a wrong picture, a wrong loss, a run that eats the machine.

### A. The unattended lab (start it and leave; an hour or two)
```
python -m accept run --out accept --models <models folder> --datasets <datasets folder> --dataset <dataset name> --file <checkpoint file name> \
  --output <output folder> --batch 2 --resolution 512 -- --param buckets.max_shapes=4
```
About 35 short runs in separate processes; finished scenarios are kept, so the same command continues after a stop. It writes `accept/verdict.md`: a pass or fail row
per check with the reason. Send `accept/verdict.md` and the `accept/matrix/*/run.log` of anything marked FAIL. Memory, nf4, features and allocator passed on 2026-11-24
(`docs/internals/acceptance-lab.md`); open from it: **determinism** (a FAIL names the first save where two same-seed runs part ways) and **speed** (two attention limits
were changed, expect green). On a SIGSEGV near the end of a profile send the core backtrace line.

### B. Quantization probe (about 10 minutes, no training)
`python -m accept quant-probe --models <models> --file <checkpoint> --precision nf4 int8 --resolution 512 --out probe`. Send `probe/quant-nf4.md` and `quant-int8.md`.
Pass: it finishes (the earlier run died of memory; fixed) and the whole-network error is small for int8 and sane for nf4 (the table names the worst block). If it still dies,
send the traceback and say whether it was the dense or the NF4 phase.

### C. Real weights through the apps page (about 45 minutes)
The pages, buttons, queue, Stop, Save now, history and the RAM admission are machine-checked; here only what needs the real model. Make apps from the starters (Apps, new,
pick the graph) with your checkpoint and datasets.
1. **SDXL picture** (starter *SDXL image generation*, 1024x1024, 25 steps, your prompt). The PNG is a picture that matches the prompt (not noise or a flat colour), the console
   has no error. The same recipe (seed, sampler, scheduler, guidance) in ComfyUI gives the same composition. Then 1536x1536: the decode falls back to tiles with a log line instead of failing.
2. **SDXL LoRA** (starter *SDXL LoRA*, your dataset, about 30 steps). Loss is finite and does not climb; *Save now* writes a file within a step or two; the finished adapter, put
   into *Adapter Stack* of the picture app at strength 0.8, changes the picture (strength 0 equals none).
3. **Anima picture** (starter `anima-image`, DiT, Qwen3 and VAE from your ComfyUI folders, 1024x1024, 30 steps, guidance 4, Euler / simple). A picture with the composition ComfyUI
   gives for the same prompt and seed (not identical pixels; this path was never compared). If it is wrong, the first suspects are the timestep given to the DiT (`s` as is) and
   the 16-channel latent normalisation. Report peak VRAM of sampling and of the Wan VAE decode (may need the tiled fallback at large sizes).
4. **RAM ceiling** (a short SDXL LoRA run with `memory.ram_max_mb` 3000): ends as **error** "host memory: the run holds N MB, over its ceiling of 3000 MB", outputs written before are
   intact; report N against `top`. Then a ceiling just above the real use (peak RES minus the mapped checkpoint): the run completes. The count is `RssAnon + RssShmem`, not RES.

## Optional (when it interests you; none of it blocks anything)

- **17. Padding default: compare three LoRAs.** Full runs, same seed, dataset and settings with `--pad-fill zero`, `mirror`, `mirror-signal` and `--param buckets.max_shapes=4` (padding is
  3 % of a canvas on average otherwise). The quick check passed (mirror costs x0.99 of zero). Compare loss at high noise and, more important, generations from the three with the same prompts
  and seeds: borders, symmetry artifacts, style strength. A symmetry measure helps: correlation of each generation with its horizontal flip, averaged over about 20 seeds. Tell me which
  default you pick and I set it. For a first look before spending hours: `python -m accept padding-sheet --images <a few images> --canvas 128x128 --count 4 --out sheet.png` (also `--sigma 0.2`).
- **29. NF4 training arms** (after B): `--param adapter.base_precision=nf4` (and once with `--param adapter.nf4_double_quant=true`), eager and `--set trainer.graph=true`, 100 steps with `--perf`:
  step time, peak memory, the `quantization` line, loss against native and int8. The accept lab's nf4 suite already passed (662 layers, 3012 MB saved, loss equal to int8's). Risk: `xpu-kernels`
  older than 0.3 has no NF4 kernel and `auto` falls back to the slow torch chain (a warning says so).
- **6b. Merge Adapter.** `Merge Adapter` node (Load Model -> Merge Adapter -> Inject Adapter) with a kohya LoRA, strength 1. Task line "Merging <file>"; step-1 loss differs from the unmerged
  run; strength 0 equals no merge node; another family's LoRA stops with "nothing ... matches".
- **6d. Auto caption.** On a dataset with uncaptioned images: `--auto-caption --set autocaption.limit=5`, then without the limit (needs `transformers` and network once). Captions written, a rerun says
  "0 captioned", the caption model's memory is gone before training.
- **90. Anima speed and memory.** `python -m run anima-lora ... --resolution 512 --steps 50 --graph` against the same without `--graph` (time per step, losses); then 1024 px batch 1 (peak, time per
  step); one 1024 px picture through `get_family("anima").load_vae(part, device="xpu")` (encode and decode time and peak).
- **103. Load Model dtype per part.** Denoiser bf16, VAE fp32, text encoders auto: the VRAM readout differs from all-bf16 by about the VAE's size; changing a text encoder's dtype rebuilds a persisted
  conditioning cache once.
- **107.2 with a real model.** the Sampler's *Live preview* addon at 1024x1024, on SDXL and on Anima: with `latent2rgb` the Preview Image card shows a blurry version of the final picture (same colours and layout, not noise or wrong hues) every 2 steps: report the extra time per step. With `vae` (VAE wired to the addon): report the extra time per preview and the peak VRAM; with a tight *VRAM budget* the card says once that previews stopped and the run finishes.
  (Wiring, frames, and same final picture are machine-checked.)
- **102.1 Windows only.** The server console restores without taking focus on an error, and not again within 30 seconds (classic console and Windows Terminal). Skip while Windows is not used.

Experiments of the lab (hours each, research questions rather than checks) moved to `docs/experiments-to-run.md`.

## Not listed on purpose

Machine-checked by `python -m tests.browser` (what a failing run would show is in each file's docstring), or settled by ordinary use, which will hit them far sooner than a checklist:

| was | now |
|---|---|
| 96 start, Settings, Help | `check_pages` (tiles, getting started, recent runs, refused folder on its field, delete history asks twice and is off during a run, help search and sidebar). Not checkable here: `different environment` row |
| 97 image generation | `check_generation` (picture, recipe chunk, batch seeds, adapter stack, Stop, size refusal) plus Critical C.1 for the real weights |
| 98, 104 apps editor, field cards | `check_board` (sections, tabs, Place in, Undo, Delete, save/reload, Phone view, File drop-down). Dragging cards by hand is left to use: a bad drag is seen at once |
| 99, 100 Save now, status, history, queue | `check_runs` |
| 101 RAM ceiling | `check_runs` (enforced, admission refusal); real use in C.4 |
| 102.2-102.4 dataset click, console | `check_pages` (the 3000-picture scroll is left to use) |
| 105 Anima generation | `check_generation` (runs, size rule) plus C.3 |
| 106 apps own their graph | `check_apps` |
| 107.1 pictures and resize | `check_editor` |
| 18 epochs cosine bullet | `check_engine` (a schedule without a length follows the run) |
| 76 import a console graph, 92 loss chart, 93 Build/Watch | use; report what annoys you when it does |
| 81 shared weights on the card | answered in `docs/internals/parallel-runs.md` (G1 failed) |
| 30 padding sheet by eye | folded into 17 |

`python -m accept judge <folder with matrix/> speed` judges a bundle from `python -m perf hwcheck` without running anything.
