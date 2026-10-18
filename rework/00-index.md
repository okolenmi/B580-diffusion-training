# Rework: index and resume point

Status file for the nodes/graph/UI rework. Plans live in this folder, one file per
domain. **Nothing here is implemented yet.** A fresh agent (or a session after
compaction) starts by reading this file, then `90-build-plan.md`, then the plan it
is assigned.

User-facing lookup of the implemented levers (by VRAM, speed, quality, data): `next/MANUAL.md`. What the owner still has to run on hardware: `TESTING.md` in the repo root. Keep both current when a lever is added or its status changes.

## Ground rules for working on this folder

- Work in small steps; each step ends with its file written to disk and its row
  below updated. A step that is not on disk did not happen.
- Plans are refined in place. Decisions the owner has made are marked **DECIDED**
  with the date; everything else is a proposal.
- **No code until the design is settled and the owner says "start".**
- Facts about the current code go to `audit/`, with file:line references. Plans
  cite those files instead of repeating them.
- Sub-agents: at most one at a time (the owner's usage limit is per 5 hours; a
  parallel fan-out can fail all at once).
- **Compatibility is not a goal** (DECIDED 2026-10-13 18:21): no migration, shims,
  aliases, deprecation windows or old-versus-new parity gates. The new code is built
  in a parallel tree (`next/`) beside the old `nodes/`; see `90-build-plan.md`.
- Testing: one acceptance test after the rewrite plus cheap structural checks
  (DECIDED direction, same message).
- **Working method for AI** (owner suggested recording it): read the old file, then
  write the new one against the plan; do not port line by line; harvest hardware
  lessons from old comments first; contracts before bodies; keep `PORTING.md`
  current. Details: `90-build-plan.md` section 3.

## Owner's requirements (2026-10-13)

1. Model families as classes/folders: `SDXL` and `Anima` (Cosmos-based, Qwen text
   encoder). Nodes call a family's methods through one shared method structure,
   like importing the same function from different libraries.
2. One universal **Load Model** node: detects what a file contains, spawns outputs
   dynamically (server computes, node shows "processing... please wait").
   A checkpoint with 3 parts gives 3 outputs; a file with 1 part gives 1.
3. Strict resource **scopes** like ComfyUI (checkpoints, UNet models, CLIPs, VAEs)
   that only list folders; the node does not promise what the file will yield.
4. LoRA (and other adapter) structures are defined by the family library; which
   parts get trained must be choosable (a node is not required yet).
5. Text encoders: SDXL checkpoint gives 4 outputs (UNet, CLIP_L, CLIP_G, VAE) plus
   an SDXL-specific packing node. Encoders that are not model-specific (Qwen, CLIP)
   live in a shared "general" folder.
6. Events: a graph event system so the graph can go "backward" (end of epoch
   re-triggers a caption-swapping node; dataset announces prompt changes).
7. Environment things (VRAM budget, device) move to a **graph settings** menu,
   not nodes. Monitors are not nodes: nodes own their UI elements; templates are
   subscribers to events that update widgets. "graph -> templates" needs rethinking.
8. Nodes: better live update, **groups of fields**, collapsible without losing the
   ability to receive wires.
9. A developer guide so every new agent follows one vision.
10. Dataset plan accepted; improve where there is room.
11. (18:21) Migration unimportant; parallel nodes folder, rewrite file by file;
    one big test after; do things as I see fit outside `nodes/`.

## Steps

| # | Step | File | State |
|---|------|------|-------|
| 0 | Index and process | `00-index.md` | written |
| 1 | Audit: frontend graph UI | `audit/frontend-graph-ui.md` | done |
| 2 | Audit: backend graph engine | `audit/backend-graph-engine.md` | done |
| 3 | Audit: model loading and SDXL coupling | `audit/model-loading-and-sdxl-coupling.md` | done |
| 4 | Research: Anima facts | `audit/anima-model-facts.md` | done |
| 5 | Plan: events | `10-events.md` | draft 1 |
| 6 | Plan: graph settings | `20-graph-settings.md` | draft 1 |
| 7 | Plan: node UI | `30-node-ui.md` | draft 1 |
| 8 | Plan: model families | `40-model-families.md` | draft 1 |
| 9 | Plan: Load Model and scopes | `41-load-model.md` | draft 1 |
| 10 | Plan: text encoders | `42-text-encoders.md` | draft 1 |
| 11 | Plan: adapters | `43-adapters.md` | draft 1 |
| 12 | Plan: dataset, latents, prompts | `50-dataset-and-latents.md` | draft 1 |
| 13 | Plan: templates and monitors | `60-templates-monitors.md` | draft 1 |
| 14 | Plan: trainer and runtime | `70-trainer-runtime.md` | draft 1 |
| 15 | Developer guide | `80-developer-guide.md` | draft 1 |
| 16 | Build plan (parallel tree; replaced the migration plan) | `90-build-plan.md` | draft 1 |
| 17 | Porting ledger (generated, triage still `?`) | `PORTING.md` | skeleton |
| 18 | Consistency pass 1 (names, compatibility purge, open questions) | all | done; repeat after owner answers |
| 19 | Phase 0: `contracts/`, import-check, comment-lint, `next/` skeleton | `contracts/`, `scripts/` | done |
| 20 | Phase 1: `next/graph` | `next/graph/` | done (91 checks) |
| 21 | Phase 2: families API, detection, Load Model | `next/families/api`, `next/nodes/` | done; reviewed and pushed |
| 22 | Phase 3: general + SDXL families (schedules, VAE, CLIP, tokenizer, conditioning, UNet, adapters, family object) | `next/families/general`, `next/families/sdxl` | done except: adapter phases and model nodes (phase 5, they need the engine runtime), NF4 weight store (phase 5) |
| 23 | Phase 4: data (library, per-VAE latent cache, prompt stages, bucketing, batcher with epoch barrier, noiser, conditioning cache) and dataset nodes | `next/data`, `next/nodes/dataset` | done; unverified on XPU. Not ported: auto-caption (`manager/captioning.py`, `nodes/dataset/auto_caption.py`), dataset preview, prewarm driver (phase 5) |
| 24 | Phase 5: training engine (optimizers with stacked state, fused updates, loss weighting, schedules, step loop, activation checkpointing, device handles, residency) and trainer-graph nodes (Inject Adapter, Save/Load Adapter, Conditioner, optimizers, schedules, loss weighting, Diffusion Trainer, Every N Steps) | `next/engine`, `next/nodes/{model,conditioning,optimizer,train}` | CPU-verified end to end through the graph runner (`check_pipeline`); unverified on XPU. Residency manager with budget, calibration and levers (`engine/runtime/memory.py`). Not ported: NF4, graph capture, block swap, profiler, probes, bucket balance, conditioning blend node |

Earlier map of parts and misplaced logic: the "Node graph component map" doc
(nine parts around one trainer). Its recommendations are inputs, not decisions.

## Glossary (names fixed for the build; change here first)

| Name | Meaning | Defined in |
|---|---|---|
| `next/` | working name of the parallel tree; relative imports inside | 90 |
| `graph/` | node base, ports, events, settings, validator, runtime, graph JSON | 90, 80 |
| `families/{api,general,sdxl,anima}` | model knowledge; `Family` with kits `detect, parts, conditioning, latents, objective, adapters, memory` | 40 |
| `FamilyRef` | family id carried in handles | 40 |
| `ModelPart` and role subclasses `DenoiserPart, ClipLPart, ClipGPart, Qwen3Part, VAEPart, AdapterPart` | what Load Model outputs | 41 |
| `LoadModelNode` | the universal loader; scopes `checkpoints, diffusion_models, text_encoders, vae` (+ `loras` for adapters) | 41 |
| `Conditioning` / `conditioner` | opaque prompt encoding value / the handle wired to the trainer | 40, 42 |
| `SDXLConditionerNode`, `AnimaConditionerNode`, `TextEncoderOptionsNode`, `ConditioningBlendNode` | text-side nodes (blend wraps a conditioner) | 42, 70 |
| `AdapterPlan`, `TrainableSelection` (group x region x pattern), `InjectAdapterNode`, `LoadAdapterNode` | adapters | 43 |
| `EventType`, `EventHub`, `EVENTS_IN/OUT`, `event_edges` | event plane | 10 |
| `SettingSpec`, `settings`, `READS_SETTINGS`, `[graph_defaults]` | graph settings | 20 |
| `Group`, `Port.ui`, `multi`, `DYNAMIC`, pins | node UI metadata | 30 |
| `PromptTable`, `LatentSet`, `ImageSet`, `BatchPlan`, `Noiser`, `PromptsChanged`, `EpochEnding` | data chain | 50, 10 |
| `ObjectiveNode` (outputs `objective`, `noiser`), `DiffusionTrainerNode`, `engine/{step,runtime}` | training | 70 |
| `uid`, `views`, Graph / View / Starter / App | templates split | 60 |

## Open questions for the owner

Collected from every plan (each plan's own section has the context). **Blocking** =
needed before the build starts; the rest can be answered when that area is reached.

### Answered 2026-10-14

Old app need not keep working; new dataset format fine; tests classified by region
(no separate spot checks); optimizers and quantization kept with comments cleaned;
phases 0 to 2 first then review (start approved). Recorded in `90-build-plan.md`
section 0.1.

### Blocking (answer before the heavy phases)

1. (40) Is `families/` as a top-level package acceptable?
2. (10) Explicit dashed event wires, or events addressable by name without wires?
3. (10) Tag swap before the next epoch begins (barrier), or may it lag one epoch?
4. (80) Comment policy (measurements to `docs/known-issues`, no history in code), and
   the import-boundary check required from the first commit (done in phase 0)?

### Per area (can wait)

- **Events (10):** a failing handler stops the run (default) or is skippable per wire?
- **Settings (20):** is my reading of "environment" right (`device`, `precision`,
  residency, graph capture, profiling, cache size, seeds as settings;
  `use_checkpoint` a node field)? `[graph_defaults]` in `config.toml` or per-graph only?
- **Node UI (30):** pins on the group header with spring-loaded expand while dragging,
  or always show every socket and collapse widgets only? Undo/redo in this rework?
  Keep frames and quick-add extras?
- **Families (40):** for Anima, blend on the final adapter output or on the Qwen states
  before the adapter (needs an experiment; start with final)? Stubs for SD1.5/Flux, or
  only two families with interface tests against a fake third (recommended)?
- **Load Model (41):** labels "UNet" for SDXL and "DiT" for Anima over neutral port
  names? `.gguf` someday? Family guess badge per file in the picker, or a plain list?
- **Encoders (42):** pack node takes parts or encoders, or always require an Encoder
  node? May Qwen2/T5 tokenizer assets enter the repository once licences are
  confirmed? `fill=blank` (recommended) or `zeros` for a single-CLIP SDXL route?
- **Adapters (43):** is "group x region x pattern" enough, or per-layer-index control
  (blocks 10 to 20 at one rank, others at another) in the first UI? Text-encoder
  adapters in the first pass? Anima native key format acceptable until the ComfyUI key
  scheme is confirmed?
- **Dataset (50):** Tag Swap random per epoch or walking each image's tags in order
  (which is the experiment)? Caption channel naming (`image.txt` plus `.tags.txt` /
  `.desc.txt`, or another)?
- **Templates (60):** is "Apps" right for beginners, or Starters only? Compare runs
  (persisted per-run metrics) now or later? May a monitor window outlive its graph?
- **Trainer (70):** a single trainer as the only training node? `grad_clip` on the
  optimizer and `grad_accum` on the trainer? Optional B580 numbers for a final
  comparison.
- **Anima (90, phase 8):** will you supply a real Anima safetensors header (script
  provided that extracts it without weights) and a short real run?
- **Process (90):** parallel editor work beside the Python phases when I work across
  sessions, or strictly one phase at a time?
