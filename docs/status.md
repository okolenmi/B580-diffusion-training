# Status

The single place that says what exists. Last checked against the code on 2026-11-19.
Hardware results are not guessed here: `TESTING.md` (root) lists what still has to run on the B580, and every lever in `manual/` carries
`verified` or `untested`.

## Built

| Area | State |
|---|---|
| Graph core: nodes, ports, groups, events, settings, addons, validation, runner, catalog | built, CPU-tested (`tests/check_graph_core.py`, `check_addons.py`) |
| Families: SDXL and Anima (detection from headers, parts, conditioning, latents, objective, adapters) | built, CPU-tested; Anima is written from ComfyUI's layout and checked against real file headers, its hardware items are in `TESTING.md` |
| Data: image libraries, per-VAE latent cache, prompts (tag shuffle and swap, caption pool, dropout, auto-caption), buckets, batcher with epoch barrier, conditioning cache and store | built, CPU-tested |
| Training engine: step phases, optimizers (AdamW, Adafactor, CAME; int8 state), loss weighting, time balance, regional passes, EMA, resume, graph capture, activation checkpointing, profiler, memory budget and residency | built, CPU-tested; speed and memory behaviour checked on the card for the levers `manual/` marks `verified` |
| Frozen base weights in int8 and NF4 | built; int8 measured on the card, NF4 CPU-tested (`internals/kernels.md`) |
| Generation (SDXL text to image): Encode Prompt, Sampler (Euler, DPM++ 2M; normal, simple, karras; guidance, epsilon scaling), Decode Image (tiled when it does not fit), Save Image (PNGs with the recipe inside), Adapter Stack (several LoRA files with a strength each, baked in when the Sampler loads the denoiser); the `sdxl-image` starter. The sampler, tiling and recipe store live in `engine/generate` and the lab uses them from there | built, CPU-tested on tiny stand-ins (`check_generate`); not run on the card (`TESTING.md` 97); no image preview in views yet; Anima is not supported |
| Nodes: Load Model, Inject/Load/Save/Merge Adapter, Adapter EMA, Conditioner, Blend Conditioning, optimizers, schedules, Resume, triggers, constants | built |
| Addons on the trainer and noiser: anchor, cleaner, self-perceptual loss, aligned start, trend weighting, gradient policy, region ladder, batch ramp | built, CPU-tested |
| Backend: validate and run graphs in a child process, memory admission, parallel runs (`auto`, `parallel`, `queue`), planned runs, run history and per-run curves, image libraries, settings, installer (graphics card, new or existing virtualenv with pinned install of torch from the Intel XPU index, tokenizer files, models folder; the server restarts into the chosen environment and setup reopens when that environment is deleted; ComfyUI optional) | built; the backend suite needs Python 3.14 |
| Frontend: `/graph` editor with views and monitor, run mode switch, instance view; `/datasets`; start page (`/`: server, card, PyTorch and node-library status, a getting-started list, latest runs, card memory); settings (sections: folders, Python environment with a live check and prompt-file downloads, interface, housekeeping, server); setup (locks the rest of the app until finished, reopens when the chosen environment is gone); help (searchable guide with troubleshooting entries) | built; browser-checked against the real backend |
| Labs: `perf` (profile, matrix, micro-benchmarks, hardware bundle), `accept` (automatic verdicts for hardware runs), `lab` (train variants, render, measure) | built, CPU-tested |
| Apps (`/apps`, `contracts/apps.md`): frozen, versioned snapshot of a graph plus a page of cards over it (cards bind to fields and live views, one page, edit and use modes, run with typed values, "update from graph" with a reviewed diff; runs record their origin). The old `/templates` page and its saved templates are gone | built, CPU-tested (`test_apps.py`), browser-checked against fixture nodes; tabs, nested sections and more card kinds are not built yet |
| Command line: `python -m run sdxl-lora` and `anima-lora` | built |

## Declared but not working

- Graph setting `memory.ram_max_mb` is in the registry and the editor and is recorded with the run, but nothing enforces it
  (`manual/persistence-and-unbuilt.md`); no host-memory sensing exists yet.

## Not built

- Event-trigger buttons in views (they need a control channel), image blocks in views.
- Block swap (denoiser blocks held on the host; Anima may need it, SDXL does not): no setting, no family block layout, no engine code (`internals/memory.md`, "Open").
- Weights shared between processes: measured not possible with plain torch (`internals/parallel-runs.md`); every run holds its own copy.
- Custom kernels for the frozen-weight path beyond what `xpu-kernels` provides (`internals/kernels.md`, `internals/capturable-attention.md`).

## Waiting on hardware

`TESTING.md`: item 94 first (does the swapped tree start and train), then the older experiment items. On a machine without the card the suites
that cannot pass are the ones needing Python 3.14 or an XPU (`test_environment_port`, `test_event_contract`, `test_graph_task_gateway`,
two checks of `test_memory_wiring`); everything else should be green (`python run_tests.py`).

## Keeping this page true

Change it in the same commit that builds, removes or verifies something. Do not copy this table into other pages; link here.
