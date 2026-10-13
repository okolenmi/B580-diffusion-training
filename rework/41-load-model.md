# 41. The universal Load Model node, scopes, and dynamic outputs

Status: **proposal.** Needs: 40 (families), 30 (node UI: pending state and
dynamic ports), 20 (settings: dtype). Inputs: `audit/model-loading-and-sdxl-coupling.md`,
`audit/backend-graph-engine.md`, `audit/frontend-graph-ui.md`.

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What the owner asked for

* **One** node, "Load Model", that loads any supported file.
* Its outputs appear **dynamically**: the server inspects the file and the node
  changes on the graph. While the server works, the node shows
  "processing... please wait".
* A checkpoint with three parts gives three outputs at once; a file with one
  part gives one output. SDXL checkpoint: UNet, CLIP_L, CLIP_G, VAE (4).
* **Strict scopes** like ComfyUI (checkpoints, UNet models, CLIPs, VAEs): the
  scope only chooses a **folder**. The node does not promise what the output
  looks like.

## 2. Behaviour, step by step

1. The node has `scope` (folder list), `file` (picker over that folder, subfolders
   allowed), `family` (`auto` by default).
2. When `scope`, `file` or `family` changes, the editor puts the node in state
   **resolving** (title bar shows "processing... please wait", fields stay
   editable, existing ports stay but are dimmed) and asks the server:
   `POST /api/v1/graphs/nodes/LoadModelNode/resolve {params}`.
3. The server calls `LoadModelNode.resolve_outputs(params)` (the hook already
   in `nodes/core.py:431`), which calls `families.detect_file(scope, file)`:
   reads the safetensors header only (no tensor data), asks every registered
   family's `detect.match`, merges the answers, and returns the part list.
4. The response patches **only this node**: its output ports are replaced by
   one port per detected part. State becomes **ready** (or **error** with the
   reason, or **needs_choice** if two families tie).
5. A run calls `build()`, which repeats the (cached) detection, creates
   `ModelPart` handles for the parts present, and validates that the result
   matches `resolve_outputs`. The server's graph validation calls the same
   hook, so a file that changed on disk between editing and running is caught
   before any node starts.

## 3. Ports

Inputs (all optional except `file`):

| Port | Type | Notes |
|---|---|---|
| `scope` | choice: `checkpoints`, `diffusion_models`, `text_encoders`, `vae` | folder list only; default `checkpoints` |
| `file` | path under the scope's folder | picker; relative names only |
| `family` | choice: `auto` + families | becomes required when detection is ambiguous; the choices are computed by the resolve response |
| `dtype` | choice: `auto`, `bf16`, `fp16`, `fp32` | advanced group; `auto` = the graph setting (plan 20), then the file's own dtype |
| `prefix` | text | advanced group; only for files whose keys carry an unusual prefix |

Outputs: one per detected part; **port names are stable, labels are friendly.**

| Port name | Label shown | Type | Appears for |
|---|---|---|---|
| `denoiser` | "UNet" for SDXL, "DiT" for Anima | `DenoiserPart` | SDXL checkpoint or UNet file; Anima DiT file |
| `clip_l` | CLIP_L | `ClipLPart` | SDXL checkpoint; a CLIP-L file |
| `clip_g` | CLIP_G | `ClipGPart` | SDXL checkpoint; a CLIP-G file |
| `qwen3` | Qwen3 | `Qwen3Part` | Anima text encoder file |
| `vae` | VAE | `VAEPart` | checkpoint or VAE file |
| `adapter` | LLM adapter | `AdapterPart` | an Anima DiT file (the adapter lives inside it) |

Types are real Python classes, subclasses of `ModelPart`, defined in the pure
`families/api/detection.py` (`role -> class` table). This means **the existing
wire-type checks (`type_mro`, `issubclass`) keep working** for dynamic ports:
an SDXL pack node declares `clip_l: ClipLPart` and `clip_g: ClipGPart`, and
the editor refuses to wire a Qwen part into it with no new machinery.

Same-role duplicates (a file that holds two UNets) are reported as a warning
and only the best-scoring one is exposed.

## 4. Detection

Layered, each layer pure Python:

1. **Container.** Parse the safetensors header by hand (8-byte little-endian
   length, then JSON) rather than through `safe_open`; it is a stable
   documented format, avoids importing torch or `safetensors` in the server
   process, and lets us cap the header size (refuse above, say, 100 MB).
   Result: `HeaderIndex(keys -> (dtype, shape), metadata, file_size)`.
   Only `.safetensors` is accepted. **`.ckpt` / `.pt` are refused**: they are
   pickles that run code on load. GGUF and sharded `*.index.json` are future
   container readers behind the same `HeaderIndex`.
2. **Part classification.** Each family supplies key-prefix rules for its parts
   (`families/<name>/fingerprint.py`). Everything a family recognises becomes a
   `PartInfo(kind, role, prefix, key_count, dtype, bytes, config)`.
3. **Family scoring.** A family's `match` returns a score from decisive keys
   (Anima: `llm_adapter.blocks.0.cross_attn.q_proj.weight`; SDXL:
   `conditioner.embedders.1.model.` plus `label_emb`), widths read from shapes
   (so SD 1.5 vs SDXL is decided by the cross-attention width, not by a name).
4. **Resolution.** Zero matches: error with a report (key counts per prefix,
   nearest family). One match: ready. Several with equal best score: `needs_choice`.
5. **Cache.** Keyed by `(real path, size, mtime_ns)`; held in the server
   process; bounded (LRU of 64). Detection of a 7 GB file reads about 1 MB.

The `Detection` object is also the source for the **memory preview**:
`family.memory.estimate(part, dtype)` from header sizes, so the budget panel
can show a number before anything is loaded.

## 5. Scopes and folders

A scope is a name and a list of root folders; it never says what files mean.

| Scope | Default folder | ComfyUI folder |
|---|---|---|
| `checkpoints` | `<models>/checkpoints` | `checkpoints` |
| `diffusion_models` | `<models>/diffusion_models` | `diffusion_models` (formerly `unet`) |
| `text_encoders` | `<models>/text_encoders` | `text_encoders` (formerly `clip`) |
| `vae` | `<models>/vae` | `vae` |
| `loras` | `<models>/loras` | `loras` (used by Load Adapter, plan 43) |

* `ProjectLayout` (today: `checkpoints_dir`, `loras_dir`, `datasets_dir`,
  `runs_dir`) gains `models_dir(scope) -> tuple[Path, ...]`; config keys under
  `[models]` in `config.toml`; fallback to `<comfy_dir>/models/<scope>`.
* `path_kind` values become the scope names. The assets API
  (`GET /assets/{kind}`) already serves a list per kind; it needs the new kinds
  and an optional `?detail=1` that adds size and mtime.
* Sandboxing is unchanged and mandatory: names are relative, `..`, absolute
  paths and escaping symlinks are rejected (`ProjectLayout.resolve_safe_model_path`).
* **Not done in the first pass:** reading ComfyUI's `extra_model_paths.yaml`.
  It is the natural compatibility step (the project already depends on a
  ComfyUI folder); listed as decision L4.

## 6. Dangling wires: the rule that makes dynamic ports safe

A dynamic node must never silently destroy the user's wiring.

* A wire whose port still exists after a resolve is untouched.
* A wire whose port disappeared stays in the document as **dangling**: drawn
  dashed red at the node edge with the old port's name; the graph validator
  reports `unknown_output_port` for it, so a run is refused with a clear
  message; deleting the wire or switching the file back repairs it.
* While **resolving**, the old ports remain (dimmed) so a slow answer or a
  transient error never makes wires jump.
* A failed or aborted resolve keeps the last good shape.

This rule belongs to the editor (plan 30) and applies to every node with
dynamic ports, not only this one.

## 7. API additions

| Call | Purpose |
|---|---|
| `POST /graphs/nodes/{class}/resolve` | body `{params}`; returns `{state, inputs, outputs, choices, info, detail}`. `state` is `ready`, `error` or `needs_choice`. `choices` carries computed dropdowns (e.g. the families to pick between). `info` is a short read-only line ("SDXL, unet + clip_l + clip_g + vae, bf16, 6.9 GB"). Generic for any dynamic node; sequence-tokened by the client like diagnostics |
| `GET /assets/{scope}` | list files in a scope (exists for `checkpoint` and `lora`, extended) |
| `POST /models/detect` | `{scope, file}` -> `Detection` JSON; thin wrapper over the same function; used for optional picker badges |
| catalog: `NodeInfo.dynamic` | `{inputs, outputs, depends_on: [port names], debounce_ms}`; tells the editor when to call resolve |

`Node` gains a declarative marker, not new base-class behaviour:
`DYNAMIC = DynamicShape(outputs=True, depends_on=("scope","file","family"))`.
The framework's rule "OUTPUTS must not be empty" (`nodes/core.py`) is relaxed
for dynamic nodes; their static `OUTPUTS` is `{}`.

## 8. Failure modes and how each is shown

| Case | State | What the user sees |
|---|---|---|
| no file chosen | `ready` with no outputs | node with no output ports, hint "choose a file" |
| file missing or outside the folder | `error` | red line under `file` |
| header unreadable or truncated | `error` | "not a valid safetensors file (header ...)" |
| pickle file (.ckpt) | `error` | "refused: pickle files can execute code; convert to safetensors" |
| no family matches | `error` | key-prefix report |
| two families tie | `needs_choice` | `family` dropdown highlighted, ports not exposed until chosen |
| a LoRA/adapter file | `error` | "this is an adapter; use Load Adapter" |
| server slow or down | `resolving` then retry notice | node stays dimmed, console note; no wire changes |

## 9. Tests (planned, all runnable without weights)

* Synthetic safetensors **headers** (JSON only, zero-length tensors) for: SDXL
  checkpoint, SDXL UNet only, SD 1.5, Anima DiT, Qwen3, Qwen-Image VAE,
  CLIP-G only, a LoRA, a corrupted header, a 200 MB header.
* Fingerprint tests per family; ambiguity test with two fake families.
* `resolve` endpoint tests: sequence ordering, cache hit/miss on mtime change.
* Editor tests (`node --test`): dangling wire retention, resolving state,
  superseded responses ignored.
* Sandbox tests: traversal, symlink escape, absolute path, pickle extension.

## 10. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| L1 | Output naming | `unet` / `denoiser` | Port `denoiser` (stable across families), label per family ("UNet", "DiT") |
| L2 | Typing of dynamic outputs | one `ModelPart` type; one subclass per role | Per role: reuses wire checks, makes pack nodes precise |
| L3 | Header parsing | `safe_open`; hand parser | Hand parser: no torch in the server, size cap, same cost |
| L4 | `extra_model_paths.yaml` | skip; read | Read it in a later step; design the `models_dir(scope)` tuple now so it is additive |
| L5 | Picker badges | none; lazy family badge per file | Lazy badge on hover/selection only; the scope stays a pure folder list as asked |
| L6 | Old nodes | delete; keep as shims | Not ported: `SafetensorsCheckpointNode` has no successor class besides Load Model (plan 90, no compatibility) |
| L7 | Several families in one file | refuse; choose | Choose (`needs_choice`), never guess |

## 11. Open questions for the owner

1. Is the denoiser label "UNet" for SDXL (and "DiT" for Anima) what you want on
   the node, with the neutral port name underneath?
2. Do you want `.gguf` accepted at some point (it would need its own loader
   per family)? The design leaves the door open.
3. Should the picker show the family guess per file (decision L5), or stay a
   plain list as you described?
