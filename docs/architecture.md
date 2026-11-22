# Architecture

How the code is laid out and how a run flows. Formats are in `contracts/`, rules for writing code in `developing.md`, the harder
mechanisms in `internals/`.

## Packages (repository root)

| Package | Holds | Imports |
|---|---|---|
| `graph/` | the node contract (`Node`, `Port`, `Group`), the graph document, validator, runner, events, settings, addons, catalog | stdlib only; the server imports it, so no torch |
| `families/api`, `families/registry` | model-file headers, detection, scopes (`ModelLibrary`), the `Family` interfaces; the registry is the only module that names families | stdlib only |
| `families/general` | shared parts: CLIP, VAE, UNet, adapter layers, schedules, quantization (int8, NF4) | torch |
| `families/sdxl`, `families/anima` | everything true of one model: fingerprint, parts, conditioning, latents, objective, adapters | api + general |
| `data/` | image library, latent cache (per VAE), prompt table, buckets, batcher with the epoch barrier, noise, conditioning cache | graph, families api/registry |
| `engine/` | the training step (`engine/step`: phases, loss, regions, EMA, graph capture) and runtime services (`engine/runtime`: residency, budget, planner, allocator cap, profiler); optimizers (`engine/optim`) | below |
| `nodes/` | thin node classes by domain: `model`, `conditioning`, `dataset`, `optimizer`, `train`, `primitive` | graph, engine, data, families api/registry |
| `perf/`, `accept/`, `lab/` | the performance lab, the acceptance lab, the experiment lab (each has `python -m <name>`) | see `scripts/check_imports.py --list` |
| `backend/` | the web server: `domain`, `application` (use cases, ports), `infrastructure` (adapters, the graph child process), `presentation` (FastAPI routes) | its own layers; torch never at import |
| `frontend/` | plain JS pages: `/graph` (editor and monitor), `/datasets`, `/apps`, settings, setup | talks to `/api/v1` only |
| root modules | `run.py` (command line), `console.py`, `starters.py` (the graphs the line and the editor start from), `paths.py` | |

`scripts/check_imports.py` enforces the layering (`--list` prints the table).

## A run

1. A **graph document** (`contracts/graph-json.md`) lists nodes, data wires, event wires, addons, settings and views. Starters build one
   (`starters.sdxl_lora`, `starters.anima_lora`); the editor, `python -m run` and the server all use the same document.
2. `graph.validate` checks it; `graph.runtime` / `graph.run` build the nodes in dependency order. The server does this in a **child process**
   (`backend/infrastructure/graph_task_worker.py`) so a crash or an out-of-memory cannot take the server down; progress comes back as
   an event file and per-node frames (`contracts/runs.md`).
3. Memory **admission** (`backend/infrastructure/admission/graph_memory.py`, `internals/memory.md`) decides whether the run fits next to the other runs;
   inside the run, `engine/runtime` keeps residents and levers within the budget.
4. The trainer node (`nodes/train/trainer.py`) hands the built model, batches, optimizer and addons to `engine/step`, which runs the phases.
   A run saves adapters through the Save Adapter node, triggered by events.

## The ideas and where they live

- **Families.** Nodes never name a family: they call the interfaces in `families/api` on the family named by the model part's `family_id` (`families/api/parts.py`; the registry returns the family).
  Adding a model = a new `families/<name>/` with a `fingerprint.py` and an entry in `families/registry.py` (see `developing.md`).
- **Load Model** (`nodes/model/load_model.py`). One loader: it reads only file headers, detects what a file holds and gives one output
  per part (an SDXL checkpoint: UNet, both CLIPs, VAE; an Anima setup: DiT, Qwen3, VAE). Scopes are folder lists, `families/api/library.py`.
- **Resolve.** A node's shape can depend on its parameters or files: `Node.resolve(params) -> Resolution` (`contracts/resolve.md`),
  asked by the editor while it shows "processing".
- **Events.** A second plane beside the data wires for going backward (end of epoch, prompts changed, checkpoint due, stop). `graph/events.py`, `contracts/events.md`.
- **Settings.** Environment facts (device, precision, VRAM limits, seeds) are graph settings, not ports: `graph/settings.py`, `contracts/settings.md`.
- **Addons.** Optional parts of a node instead of separate nodes: `internals/addons.md`.
- **Data chain.** library → latent cache → prompts (captions, tag shuffle/swap, pool, dropout) → buckets → noiser → batcher → conditioner → trainer
  (`nodes/dataset`, `nodes/conditioning`, `data/`). The cache refuses latents made by another VAE.
- **Display modules.** A node declares what is worth watching (`MODULES`); the trainer reports a flat per-step frame; `/graph` draws them
  with the shared board renderers. No per-node code in the frontend.
- **Libraries and runs.** Image libraries (`/api/v1/datasets`), run history, planned and parallel runs (`internals/parallel-runs.md`).
