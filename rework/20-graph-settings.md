# 20. Graph settings: environment out of nodes

Status: **proposal.** Feeds every other plan (devices, precision, residency,
cache, seeds). Inputs: `audit/backend-graph-engine.md` section 5,
`audit/frontend-graph-ui.md` section 6, ADR 0005 (memory admission).

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows or old-port support; the new tree is built clean beside `nodes/`.

## 1. What the owner asked for

* VRAM budget must **not** be a node. A **specific graph settings menu** holds
  everything "related more to environment than to process".
* Monitor is not a thing to attach to the graph either (plan 60).

## 2. Three tiers, one rule

| Tier | Belongs here when | Stored in | Edited in |
|---|---|---|---|
| **App** | it describes the machine or install | `config.toml` | Config page (exists) |
| **Graph** | it describes the environment of *this run*; nobody would wire it | the graph JSON (`settings`) | **Graph settings dialog** (new) |
| **Node field** | it changes *what the process does* | the node's params | the node |

Rule of thumb: **if two nodes in a graph must agree on it, or a person asks "why
did I have to set this five times", it is a graph setting.** If one node's
behaviour is what a person is experimenting with, it is a node field.

## 3. What exists, and what is wrong with it

* `GraphDoc.memory` (`state.js:67`) and `backend/domain/modules.py GRAPH_FIELDS`:
  four memory fields, shown in a section of the editor. Stored in the graph,
  carried into templates. Good, and it should be kept as the seed.
* `VRAMBudgetControllerNode` still exists beside it (4 ports). ADR 0005 planned to lift
  it into graph settings but did not decide which value wins; the two numbers are different quantities (allocator ceiling minus
  reserve vs device claim plus overhead).
* **The same environment value is a port on many nodes**, each with its own
  default. Counted in the code:

| Setting | Where it is a port today | Default |
|---|---|---|
| device | `ComfyUNetLoRANode`, `ResourcesControllerNode`, `SDXLTextEncoderNode`, `VRAMBudgetControllerNode`, `AutoCaptionNode` | `"xpu"` (AutoCaption: `"auto"`) |
| dtype | `ComfyUNetLoRANode.dtype`, `ResourcesControllerNode.unet_dtype` | `None` / `"inherited"` |
| frozen weight store | `ComfyUNetLoRANode.frozen_weight_store` | type |
| residency | `ManagedLoRATrainerNode`: `residency_safety_margin`, `calibration_steps`, `empty_cache_every_n_steps` | |
| text-encoder prewarm | `ManagedLoRATrainerNode.prewarm_text_encoder` | |
| graph capture | `use_xpu_graph` on both trainers | |
| profiling | `profile`, `profile_memory_per_phase` | |
| seeds | `ConditioningBlend.seed`, `CaptionPool.seed`, loader shuffle | -1 / None |
| memory | graph settings already | |

## 4. The settings registry

A **setting** is data, not code: the same shape as a node `Port` so one renderer
draws both (already how `GRAPH_FIELDS` works).

```python
@dataclass(frozen=True)
class SettingSpec:
    key: str                  # dotted: "runtime.graph_capture"
    label: str; doc: str
    type: str                 # bool | int | float | str | choice
    default: Any
    choices: tuple | None = None
    unit: str | None = None
    minimum/maximum/step
    group: str                # dialog tab: Memory, Device, Runtime, Conditioning, Reproducibility
    advanced: bool = False
    allow_auto: bool = False
    contributed_by: str = "core"     # "core", a family id, or a node class
```

* Core settings live in `backend/domain/` (pure Python), as `GRAPH_FIELDS` does
  today, so the backend keeps its layering.
* **Nodes and families contribute** specs with a class-level `SETTINGS`
  tuple (a family: `Family.settings`). The infrastructure catalog (which may
  import nodes) merges them and serves them; the domain never imports them.
* Unknown keys in a saved graph are kept and warned about (same posture as
  unknown node classes today).

### Proposed initial keys

| Group | Key | Replaces | Notes |
|---|---|---|---|
| Memory | `memory.vram_max_mb`, `memory.vram_min_mb`, `memory.strict`, `memory.ram_max_mb` | exists | unchanged semantics |
| Device | `device.default` | the five `device` ports | `xpu`/`cpu`/`auto`; a node may override (below) |
| Device | `precision.denoiser`, `precision.encoders`, `precision.vae` | `dtype`, `unet_dtype` | `auto` = the file's own dtype (Load Model info line shows it) |
| Device | `precision.autocast` | trainer-internal constants | compute dtype during the step |
| Runtime | `runtime.graph_capture` | `use_xpu_graph` | |
| Runtime | `runtime.residency.safety_margin`, `runtime.residency.calibration_steps` | managed trainer ports | plan 70 |
| Runtime | `runtime.empty_cache_every_n_steps` | trainer ports | the knob involved in the prewarm OOM |
| Runtime | `runtime.block_swap` | new | count of blocks held on host, validated against the family's block layout (Anima needs it) |
| Runtime | `runtime.profile` | trainer `timing` port starts from it | per-phase memory is the ledger (plan 71), so there is no `profile_memory` setting |
| Conditioning | `conditioning.cache_mb`, `conditioning.prewarm` | `prewarm_text_encoder`, `prompt_capacity()` | plan 42 |
| Reproducibility | `seed.base` | per-node seeds | a node seed left empty derives from `seed.base` and the node id, so one number reproduces the whole graph |

Kept as **node fields** (they change the process): optimizer, learning rate,
rank and groups, loss weighting, timestep policy, caption stages, save and
probe cadence (these become event wiring, plan 10), LoRA timestep gate,
`use_checkpoint` (activation checkpointing; decision G6).

## 5. Resolution order and node access

`node explicit field > graph setting > app default > code default`

* Nodes read settings, not ports: `self.context.settings.device`,
  `...precision.denoiser`, typed accessors over an immutable snapshot built
  once at run start (like `memory` on `ExecutionContext` today).
* A node may keep an **override** only where mixing is legitimate: `device` on
  `AutoCaptionNode` (captioning on CPU while training on the accelerator),
  `precision` per Load Model (a fp32 VAE next to a bf16 UNet). Overrides sit in
  an **Override** field group, empty by default, labelled "inherits graph
  setting".
* Validation: unknown key, wrong type, out of range (`min`, `max`), and
  cross-key rules (`vram_min <= vram_max`) are reported as graph issues before
  any run, with the key named.
* Each setting spec lists **who reads it**; nodes declare
  `READS_SETTINGS = ("device.default", ...)` so the dialog can say "used by:
  Load Model n1, Trainer n7" and the editor can warn about a field that
  overrides a setting.

## 6. Storage

* Graph JSON: `settings: {"memory.vram_max_mb": "auto", ...}`. There is no old
  `memory` object and no reader for one (plan 90: no compatibility).
* App defaults: a `[graph_defaults]` table in `config.toml`: a person's machine
  (device, precision, vram floor) is entered once and every new graph starts from
  it. Graphs keep the values they stored.
* The old per-node ports (`device`, `dtype`, `use_xpu_graph`, ...) are **not
  carried into the new node classes**. Only the declared overrides of section 5
  exist.

## 7. The dialog (UI)

A real menu, not a section squeezed beside the canvas: a gear in the toolbar
opens a dialog with tabs (Memory, Device, Runtime, Conditioning, Reproducibility),
a search box, and **advanced** collapsed per tab.

Each row: label, control, unit, an **origin badge** (default, app, this graph),
a reset button, the doc as a tooltip, and the "used by" line. A row whose value
is overridden by a node shows that node. The memory preview panel moves into the
Memory tab (it already reads these numbers).

Templates keep exposing graph settings as blocks, as now.

## 8. `VRAMBudgetControllerNode`: not ported

The new tree has no such node. Graph settings hold the budget; the runtime builds
the controller from them (`ctx.runtime.resource_control()`, plan 70). The
ADR 0005 question "which of two different quantities wins" is settled by
construction: only the settings exist, defined in the dialog with their exact
meaning (allocator ceiling minus reserve versus device claim plus overhead are two
named settings, not one number).

## 9. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| G1 | Tiers | two (app, node); three | Three; graph tier is the missing one |
| G2 | Key naming | flat; dotted groups | Dotted (`group.key`); the dialog tabs are the first segment |
| G3 | Node overrides | none; allowed everywhere; allowed where declared | Only where a node declares a reason (device, precision) |
| G4 | Seeds | per node; derived from one base | Derived from `seed.base`, node seed optional |
| G5 | Stored as | `memory` object; `settings` map | `settings` map |
| G6 | Activation checkpointing | node field; graph setting | Node field (advanced) for now; it changes numerics slightly and per-adapter choices exist |
| G7 | Where the dialog lives | editor section; separate dialog | Separate dialog with origin badges |
| G8 | VRAM node | keep; drop | Not ported (section 8) |

## 10. Risks

* A hidden setting changes behaviour silently. Mitigation: origin badge,
  "used by", and an info line on nodes that inherit.
* Specs contributed by nodes and families can collide. Mitigation: keys must be
  namespaced by group, and the catalog refuses duplicates at load (reported as
  `load_errors`, like import failures).

## 11. Open questions for the owner

1. Is my reading of "environment" right? In particular: `device`, `precision`,
   residency, graph capture, profiling, cache size, seeds as graph settings;
   `use_checkpoint` stays a node field.
2. Do you want `[graph_defaults]` in `config.toml` (machine-wide defaults for new
   graphs), or per-graph only?
