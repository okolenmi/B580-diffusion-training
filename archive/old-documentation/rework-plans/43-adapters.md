# 43. Adapters (LoRA and relatives) and the choice of what trains

Status: **proposal.** Needs 40 (family `adapters` kit), 41 (parts), 42 (encoder
adapters). Feeds 70 (parameter groups). Facts: `audit/model-loading-and-sdxl-coupling.md`
sections 2 and 4, `audit/anima-model-facts.md` section 5.

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What the owner asked for

LoRAs, and other supported adapter types, get their structure from the family
library (SDXL, Anima). It must be adaptive enough to **choose manually which
parts are trained**. A dedicated node is *not* required yet; this is mostly
about code that is comfortable to extend.

## 2. What exists today (short)

* `build_lora_injected_unet(...)`: 12 options (rank, alpha, dropout, scaling
  policy, target modules, checkpointing, adapter strategy, frozen weight store,
  conditioning-path flag, block weights) walking a `ComfyUNetWrapper`.
  `ComfyUNetLoRANode` is its node (13 ports).
* Strategies already general in spirit: `AdapterStrategy` (`PlainLoRAAdapter`,
  `DoRAAdapter`), `LoRAScalingPolicy` (`Classic`, `RankStabilized`),
  `FrozenWeightStore` (`BF16`, nf4), `LoRAGeneration` stacking (phases).
* "Which modules": a list of name patterns or a named set (`TARGETS_ATTENTION`,
  `TARGETS_KOHYA_DEFAULT`, `+label_emb`, `+time_embed`: 722/724/726 modules on
  SDXL) and `block_weights` (`"input_blocks.3:0.5,middle_block:1.0"`, a UNet
  module-path prefix with a multiplier).
* File format: `lora_unet_<path with _>.lora_down/up.weight/alpha`.
* Saver (`LoRACheckpointSaverNode`), loader (`LoRACheckpointLoaderNode`),
  `LoRAPhaseSplitNode`.

Problem: target names, block names and the key scheme are baked into SDXL
files (`lora.py`, `lora_phases.lora_key`). Anima's layers are named
`blocks.N.self_attn.q_proj`, `cross_attn`, `mlp`, plus optional embedders,
and its default excludes by regex (`.*(_modulation|_norm|_embedder|final_layer).*`).

## 3. The model: **group x region x pattern**

A selection of what trains has three independent ingredients.

| Ingredient | Meaning | Example (SDXL) | Example (Anima) |
|---|---|---|---|
| **Module group** | a named kind of layer, defined by the family | `attn_self`, `attn_cross`, `ff`, `transformer_proj`, `label_emb`, `time_embed` | `self_attn`, `cross_attn`, `mlp`, `patch_embed`, `timestep_embed`, `final_layer`, `llm_adapter` |
| **Region** | where in the network, defined by the family's block layout | `input_blocks.N`, `middle_block`, `output_blocks.N` | `blocks.N` (N = 0..27 for the 28-block model) |
| **Pattern** | an escape hatch: include / exclude by `re.fullmatch` on the module path | | |

```python
@dataclass(frozen=True)
class TrainableSelection:
    groups: tuple[str, ...]                 # which groups are on (default = layout default)
    regions: dict[str, float | None]        # region prefix -> weight multiplier, None = off
    include: tuple[str, ...] = ()           # extra modules to add, regex fullmatch
    exclude: tuple[str, ...] = ()           # modules to remove, regex fullmatch
    per_group: dict[str, GroupOverride] = {}   # rank / lr multiplier / alpha per group
```

`regions` generalises today's `block_weights` (same prefix idea, but valid for
any family's block names and able to say "off"). `per_group` is sd-scripts'
`network_reg_dims` / `network_reg_lrs` idea and is how the Anima rule
"**do not train the LLM adapter; if you do, use a lower learning rate**" is
expressed: group `llm_adapter` is off by default; turning it on may carry an
`lr_mult`.

## 4. Family interface

```python
class ModuleGroup:   name; label; default_on: bool; doc: str; excluded_note: str|None
class AdapterLayout(Protocol):
    def groups(self) -> tuple[ModuleGroup, ...]
    def regions(self, info: PartInfo) -> tuple[str, ...]     # prefixes, from header config
    def classify(self, module_path: str, module_kind: str) -> str | None   # -> group or None
class AdapterKit(Protocol):
    layout: AdapterLayout
    def plan_from_keys(self, index: HeaderIndex, part: PartInfo,
                       sel: TrainableSelection) -> AdapterPlan       # HEADER ONLY
    def inject(self, part_model, spec: AdapterSpec, plan: AdapterPlan) -> AdaptedPart
    def key_schemes(self) -> tuple[KeyScheme, ...]          # file formats this family can write/read
```

* **`plan_from_keys` is the important idea.** Weight key names already contain
  module paths (`...attn1.to_q.weight` is module `...attn1.to_q`) and shapes
  give parameter counts. So the editor can show, **without loading the model**,
  "this selection trains 722 modules, 23.1 M adapter parameters at rank 64,
  about 90 MB" from the Load Model detection alone. The same plan drives
  injection, so what the preview shows is what gets built.
* `AdapterPlan`: per-module list `(path, group, region, in/out features,
  rank, lr_mult)`, plus totals per group. It is JSON-friendly for the UI.
* The kind-agnostic parts stay in `general/adapters`: LoRA and DoRA layers,
  strategies, scaling policies, weight stores, phase stacking, checkpointing.
  Kinds declare what layer types they support (Linear, Conv2d); a family
  `classify` simply returns no group for an unsupported module.

## 5. The node surface (no dedicated "select parts" node yet)

`InjectAdapterNode` replaces `ComfyUNetLoRANode` and works on any
`DenoiserPart` **or** text-encoder part (plan 42). Fields are **grouped** (plan 30):

| Group | Fields |
|---|---|
| **Adapter** | `kind` (choices from general: lora, dora), `rank`, `alpha`, `dropout`, `scaling` |
| **Targets** | `groups` (multi-select; choices **computed by the resolve call** from the family layout and the wired part), `regions` (a small table: prefix -> weight or off), `include`, `exclude` |
| **Memory** (advanced) | `checkpointing`, `weight_store` |
| **Schedule** (advanced) | timestep gate `low`/`high` (the existing `lora_gate`) |

Output: `AdaptedPart` (trainable parameters, `trained_state_dict`, device
residency) -- the successor of `TrainableModel`. Its info line shows the plan
totals. A future dedicated "what to train" node is then just a producer of
`TrainableSelection`: the field group becomes a handle input without changing
the injection node, which is why the selection is a value type now.

Other nodes:

* `LoadAdapterNode` (scope `loras`): reads the file header with the family
  `key_schemes`, returns `AdapterFile` (family, kind, rank, groups present,
  scheme). Modes: `continue` (resume training these weights) or `frozen`
  with `strength` (apply while training another). Compatible with the earlier
  frozen-LoRA ports of `ResourcesControllerNode`.
* `SaveAdapterNode` (existing saver): `format` choices come from the family's
  `key_schemes` of the adapted part; metadata written into the file: family id,
  kind, rank, alpha, selection, step, base model name (so Load Adapter can warn
  on a mismatch before injection).
* `LoRAPhaseSplitNode` stays; it works on layer objects and is family-agnostic.

## 6. Parameter groups for the optimizer

`AdaptedPart.param_groups()` returns one group per `(module group, lr_mult)`.
This is the producer the handle `ParameterGroupPolicy` never had: the optimizer
node (plan 70) consumes `param_groups`, so LoRA+ ratios, an adapter-specific
learning rate and the Anima "adapter lr" are expressed once, here, rather than
by editing optimizer nodes. `ModelParametersNode` / `TrainerParametersNode`
become thin collectors.

## 7. File formats

| Family | Scheme | State |
|---|---|---|
| SDXL | kohya/ComfyUI: `lora_unet_<path>.lora_down.weight`, `.lora_up.weight`, `.alpha` (and `lora_te1_`, `lora_te2_` for encoders) | UNet half is shipped; encoder prefixes to verify |
| Anima | **unknown for ComfyUI** (audit gap 1); sd-scripts has a conversion script | ship a `native` scheme first (module-path keys + metadata), add the ComfyUI scheme after reading the reference implementations |

The saver must **refuse** a scheme it cannot write correctly rather than
produce a file that loads as "no effect". Detection of an existing adapter
(`inspect_lora` today) moves to the same `key_schemes`.

## 8. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| A1 | Selection model | raw regex list; named sets; group x region x pattern | Group x region x pattern: groups for the common case, regions for block weighting, patterns as the escape hatch |
| A2 | Where layouts live | one general table; per family | Per family (class names are family facts); general code only sees groups |
| A3 | Preview | after load; from header | From header: Load Model already has the key list |
| A4 | Anima file format | wait for ComfyUI scheme; ship native first | Native first, refuse unknown scheme explicitly |
| A5 | Dedicated select node | now; later | Later; the value type exists now so it is additive |
| A6 | Optimizer groups | per optimizer node; from the adapted part | From the adapted part |
| A7 | Same node for denoiser and encoder adapters | two nodes; one | One node, layouts supply the groups |
| A8 | UI wording | "LoRA"; "Adapter" | Node is "Inject Adapter", kind defaults to LoRA; the word LoRA stays in the kind choice and in file names |

## 9. Tests planned

* Layout classification tables for SDXL against the **722/724/726** counts
  that the code already documents (regression: the new `plan_from_keys` must
  reproduce them from key lists alone).
* Header-only plan equals the plan built from a loaded toy model (parity test
  on a small synthetic UNet and a small synthetic DiT).
* Round trip export -> import per scheme; refusal on an unsupported scheme.
* Selection edge cases: group off but included by pattern, exclude wins over
  include, region off, unknown group name (error lists valid names).

## 10. Open questions for the owner

1. Does "group x region x pattern" cover what you mean by choosing parts, or do
   you also want per-layer-index control (e.g. only blocks 10 to 20, rank
   16 there and 64 elsewhere)? The model supports it; the question is what the
   first UI exposes.
2. Should text-encoder adapters be in the first pass, or only the denoiser?
   (Design supports both; encoder training conflicts with cached encoder
   outputs, section 8 of plan 42.)
3. For Anima, is a native-format file acceptable until the ComfyUI key scheme
   is confirmed from the reference code?
