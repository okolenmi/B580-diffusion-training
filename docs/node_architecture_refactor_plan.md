# Node-based architecture refactor -- plan

Started 2026-07. This is a multi-session initiative. Written to survive
context resets between sessions -- read this file before resuming the work,
regardless of which Claude session picks it up next.

## Why (grounded in what actually happened this session, not abstract taste)

Two concrete, felt costs from this session alone:

1. **The `cache.student_mix` visibility bug.** Answering "why doesn't this
   checkbox show up in LoRA mode" required tracing through five files and
   three separate layers: `config_model.py` (Pydantic schema) ->
   `config_schema.py` (auto-derived UI metadata from Pydantic introspection,
   including an auto-generated `visible_when` for discriminated-union
   members) -> `config_ui.py` (hand-authored label/group/*extra*
   `visible_when` conditions, merged on top) -> `options.py` (server-side
   merge of the two visibility-condition sources into one dict) ->
   `option-tree.js` (client-side evaluator for the merged conditions). Every
   layer, read in isolation, looked correct. The actual answer was never
   found by static reading alone -- it needed live reproduction, which
   wasn't available. That's not a one-off; it's what happens whenever a
   single piece of behavior ("is this option relevant right now") is encoded
   as cross-cutting conditions spread across independently-evolving files
   with no single owner.

2. **`Trainer`'s ad-hoc flag accumulation.** Implementing the unified-teacher
   LoRA change this session meant adding `self._lora_unified_teacher` to an
   already-long list of instance flags (`self._is_lora`, `self._is_cyclic`,
   ...) each gating slightly different behavior at different points across
   an 850+ line `train()` method. Every new capability this session (CAME
   optimizer, opt-step semantics, unified teacher, DAgger chain-mixing) had
   to be threaded through this one god-object correctly, and a real bug
   (`elif self.teacher:` becoming unreachable) slipped through mid-session
   specifically *because* of this threading -- not because the underlying
   idea was wrong.

The diagnosis: **the codebase's actual failure mode isn't any single bad
line, it's coupling.** Understanding or safely changing one piece of
behavior requires holding several other files/flags in your head
simultaneously, and that cost scales *up*, not down, as more sessions add
more features to the same god-objects. Given this project is developed
almost entirely by successive, context-isolated AI sessions with no
persistent memory of prior reasoning, that coupling cost is a much bigger
deal here than in an ordinary human-maintained codebase -- each new session
starts by re-deriving understanding from scratch, and coupling directly
multiplies how much has to be re-derived correctly before a safe change can
be made.

## What a "node" actually buys here

A node's whole value proposition is: **explicit, typed input/output
boundaries that make "does this configuration make sense" a structural
question (is this node present / is this port connected) instead of a
cross-file conditional-logic question.** The `cache.student_mix` bug
couldn't exist in a node system, structurally -- if "Teacher Rollout Cache"
is a node type, its "Student Mix" input either exists on that node or it
doesn't; there's no separate visibility-condition system to get out of sync
with the underlying schema.

**Important distinction, worth being explicit about because it changes cost
and risk by an order of magnitude:** "node-based" has two separable parts,
and only one of them fixes what's actually hurting:

- **(A) An execution graph with typed node/port boundaries in the backend
  Python code.** This is what directly fixes the coupling problem above --
  it's a *code organization* change. Moderate-to-large effort, but doesn't
  require building new infrastructure Anthropic/this project doesn't
  already have -- it's a disciplined refactor.
- **(B) A visual, draggable-node canvas UI** (what ComfyUI actually looks
  like) for a human to build graphs by hand, live, in a browser. This is a
  *separate, much bigger* engineering project (a graph-editor frontend,
  live execution/preview wiring, node palette, save/load of visual layouts
  -- ComfyUI's own node-canvas layer, `litegraph.js`-based, is a
  substantial piece of software in its own right, built by a team, over
  a long time).

(A) is what makes this codebase safer for AI sessions to work in. (B) is a
UX layer for the human on top of (A), and is optional -- (A) alone already
gets you: a Trainer that's a small piece of code wiring together
already-defined nodes, config validation that's structural rather than
cross-file, and swappable components (data source, optimizer, teacher
strategy) with real interfaces instead of ad-hoc flags. **Recommendation:
build (A) first, fully, and treat (B) as a later, optional, lower-priority
addition -- possibly never, if a well-organized config file/CLI on top of
(A)'s graph is sufficient.** This isn't reneging on the request; (B) can
absolutely be built later, but sequencing it after (A) means every session
spent on (B) is spent on top of an already-solid foundation instead of
racing to build UI plumbing before the underlying coupling problem is fixed.
Flagging this now for explicit buy-in before committing many future
sessions to a particular order.

## Proposed node/port abstraction, mapped to what actually exists today

A concrete sketch, not a final design -- meant to be validated/revised in
Phase 1, not treated as fixed.

```python
class Port:
    """A typed input or output slot on a node."""
    name: str
    type: type  # e.g. TrainableModel, TrainingDataSource, Optimizer, ...

class Node(Protocol):
    """A single-responsibility unit. No node knows about any other node's
    internals -- only the typed values on its own declared ports."""
    inputs:  dict[str, Port]
    outputs: dict[str, Port]
    def run(self, **inputs) -> dict[str, Any]: ...
```

Concrete node types, chosen because they map onto boundaries that *already
exist* in the codebase today, just not made explicit:

- **`ModelProviderNode`** -- loads a checkpoint, outputs typed UNet/CLIP/VAE
  weight bundles. (Currently: scattered across `load_models()`'s state-dict
  slicing logic in `trainer.py`.)
- **`LoRAInjectorNode`** -- takes a `ModelWeights` input + LoRA config,
  outputs a `TrainableModel` (with the gate mechanism this session already
  built and verified -- `lora_gate_override()` becomes a first-class part of
  this node's contract, not an ad-hoc context manager threaded through
  three files by hand).
- **`TeacherSourceNode`** (interface) -- outputs something that can produce
  teacher predictions. Two concrete implementations:
  - `SeparateModelTeacherNode` (today's distillation case: a second loaded
    model)
  - `GatedSelfTeacherNode` (this session's unified-LoRA discovery: the *same*
    `TrainableModel`, gate forced to 0) Both implement the same output port
    type, so `TrainerNode` genuinely doesn't need to know or care which one
    it's connected to -- this is the exact ad-hoc
    `self._lora_unified_teacher` branching from this session, promoted to a
    real interface instead of a flag threaded through five call sites.
- **`DataSourceNode`** (interface) -- `TeacherRolloutCacheNode` (wraps
  `cache_trajectory.py`), `RandomCacheNode` (wraps `cache_random.py`),
  `DatasetLoaderNode` (wraps `manager/loader.py`). All produce the same
  `TrainingBatchSource` output type.
- **`OptimizerNode`** -- **this one is nearly free.** `optimizers.py`
  already has 5 classes (`CPUAdamW`, `ChunkedXPUAdafactor`, `ChunkedXPUCAME`,
  `ForeachXPUAdafactor`, `FusedXPUAdafactor`) sharing a real common
  interface (`step(n_steps=)`, `zero_grad()`, offload/reload/decay/reset
  hooks) already, verified directly this session while porting CAME to match
  it. `optimizer_builder.py`'s if/elif string dispatch is already 90% of a
  node -- it just needs a declared port type and to stop living inside
  `Trainer.build_optimizer()`.
- **`TrainerNode`** -- given `TrainableModel` + `TeacherSource` (optional) +
  `DataSource` + `Optimizer` + schedule params, runs the step loop, emits
  progress/checkpoint/preview *events* other nodes can subscribe to instead
  of `Trainer` calling `self.preview_gen.generate()`/`save_callback()`
  directly by hardcoded reference.
- **`CheckpointSaverNode`**, **`PreviewGeneratorNode`** -- subscribe to
  `TrainerNode`'s step events, own their own logic (`save.py`,
  `preview_sampler.py` mostly already are this, just not wired as
  subscribers to a generic event rather than direct callback params).

Config validation becomes structural under this model: an option "existing"
or "making sense" reduces to "is the node present in the graph" /
"is this port connected" -- no separate cross-file visibility-condition
system needed at all, which is a direct fix for pain point #1 above.

## Migration strategy: incremental (strangler fig), not big-bang

Given: (a) the current system is a *working, debugged* pipeline (CAME
verified against the reference implementation, opt-step semantics fixed,
SNR bug fixed, DAgger chain-mixing wired and verified this session -- real,
hard-won correctness that a rewrite risks silently regressing), and (b)
future sessions may be short/interrupted (confirmed recurring problem this
whole conversation), a big-bang rewrite is close to the worst-case plan --
if a session runs out mid-rewrite, a future session inherits two
half-integrated architectures and has to understand *both* to make any safe
change, which is strictly worse than today's single (if coupled) system.

Proposed instead: introduce the `Node`/`Port` interfaces, then migrate one
subsystem at a time *behind* those interfaces while the rest of the system
keeps working unchanged, in this order (chosen for lowest risk / highest
confidence-building value first):

**Phase 1 (small, safe, proves the concept):** Wrap the optimizer subsystem
as real nodes. Lowest risk because the underlying classes already share a
clean interface (see above) -- this phase is close to "add a thin
`OptimizerNode` wrapper and a tiny graph-resolution shim," not a rewrite of
`optimizers.py` itself. Success criterion: `Trainer` asks a graph for "the
optimizer" instead of calling `build_optimizer()`'s if/elif directly, with
*zero* behavior change (same classes, same math, same checkpoints) --
provable by running the exact same CAME numerical verification harness
built earlier this session against the wrapped version.

**Phase 2:** `TeacherSourceNode` interface, with `GatedSelfTeacherNode` and
`SeparateModelTeacherNode` as the two implementations -- directly replaces
this session's `self._lora_unified_teacher` flag-threading with a real
interface. Good second phase because the two implementations and their
contract are already fully understood and verified (this session's work),
so this phase is "promote working ad-hoc code to a real interface," not new
design work.

**Phase 3:** `DataSourceNode` interface over `cache_trajectory.py` /
`cache_random.py` / `manager/loader.py`.

**Phase 4:** `TrainerNode` itself -- extract the step loop out of the
`Trainer` god-object into a node that only knows about its typed input
ports, with `CheckpointSaverNode`/`PreviewGeneratorNode` as event
subscribers instead of direct callback parameters. Highest-risk phase
(touches the most surface area), done last, once phases 1-3 have already
validated the interfaces against real, working code.

**Only after 1-4 are stable:** revisit whether a visual canvas UI (B, above)
is worth building, or whether a clean config file/CLI over the now-explicit
graph is sufficient on its own.

At every phase boundary, the system should be fully runnable and the
existing correctness work (CAME's numerical match, the opt-step counting
invariants, SNR formulas) should be re-verified against the wrapped version
before moving to the next phase -- a phase that breaks something already
fixed is a regression, not progress, regardless of how much closer it gets
to the end architecture.

## Refined design decisions (after follow-up discussion)

- **Custom lightweight canvas, not ComfyUI's.** Explicit reason: ComfyUI's
  canvas has a real, measured performance cost (~30% generation slowdown
  just from having the tab visible/watched) -- almost certainly a
  persistent per-frame redraw loop (`requestAnimationFrame` or equivalent)
  that burns CPU/GPU cycles continuously regardless of whether anything on
  screen actually changed, competing with real generation/training work for
  the same hardware. Design rule for this project's version, stated
  explicitly so it doesn't drift: **DOM-based node rendering (real
  `<div>` elements, browser-native layout/compositing), not a `<canvas>`
  being manually redrawn every frame. No polling loop, no RAF loop.
  Render on actual state change (a fetch resolving, a user click), then go
  fully idle.** This is cheap to build correctly and expensive to fix later
  if built wrong, so it's worth being strict about from the first line of
  code, not just an aspiration.
- **Node UI is derived from the real class, not hand-duplicated.** Same
  principle as `config_schema.py`'s Pydantic introspection, applied to
  nodes: a node's displayed ports come from introspecting the actual Python
  class (`inspect.signature()`), not a separately-maintained metadata file.
  This is a structural guarantee against recreating the exact bug category
  that started this whole discussion (`config_ui.py`'s hand-authored
  conditions drifting out of sync with the real schema) -- there's no
  second file for the node graph's rendering to drift out of sync with.
- **Separate dev/testing tab, isolated from the production config path
  until deliberately switched over.** A new route (`/nodegraph`) and a new
  API router (`/api/nodegraph/*`), touching the existing config/training
  code paths not at all beyond one router-registration line in `main.py`.
  Old system keeps working unchanged for as long as needed; nothing is
  switched over until the new system is actually ready.
- **The canvas is also a development aid for the refactor itself, not just
  an end-user feature.** Being able to see a node's real, auto-derived
  ports and (once execution is wired up) run it in isolation is valuable
  *during* Phase 1-4 of the migration -- it catches interface mistakes by
  direct inspection/testing of one node, rather than only via full
  end-to-end training runs. This is why it's being built early/alongside
  the backend work rather than deferred to the very end.

## First slice: shipped

A minimal, real, working proof-of-concept of the "auto-derived, no-drift"
principle above -- deliberately small, touches nothing in the production
path:

- `server/nodegraph_introspect.py` -- pure introspection (`inspect.
  signature()`-based), zero side effects, zero coupling. Given any class,
  derives its ports from the real `__init__` signature. Functionally
  verified (not just syntax-checked) against a dummy class mirroring
  `ChunkedXPUCAME`'s actual signature -- correct first-line-only docstring
  extraction, correct required-vs-default detection, correct type display
  for both annotated and unannotated parameters.
- `server/routes_nodegraph.py` -- new, isolated `APIRouter`
  (`/api/nodegraph/optimizers`), returning the introspected
  `optimizers.py` classes (`CPUAdamW`, `ChunkedXPUAdafactor`,
  `ChunkedXPUCAME`, `ForeachXPUAdafactor`, `FusedXPUAdafactor`) as JSON.
  Chosen as the very first target because it's already the closest thing
  in the codebase to real node-shaped code (see Phase 1 above).
- `server/static/nodegraph.html` -- the actual playground page at
  `/nodegraph`. Fetches the JSON once, renders each class as a DOM node
  box with its ports (name, type, default, required/optional) laid out
  inside -- no canvas, no redraw loop, matching the design rule above.
  HTML structure and embedded JS both verified (tag-balance check via
  Python's `html.parser`, syntax check via `node --check`) before shipping.
- `main.py` -- three additive lines (import, router registration, page
  route), following the exact existing pattern used for `/datasets`. No
  existing route touched or modified.

This slice deliberately does *not* yet include: interactive dragging,
wire/connection rendering, or actual graph execution -- those come with
Phase 2+ once there's a second node type (e.g. `TeacherSourceNode`) to
meaningfully connect to something. The goal of this slice was narrowly to
prove the core "ports auto-derived from real code, rendered without a
ComfyUI-style performance cost" idea end-to-end before building anything
more elaborate on top of it.

**Follow-up, same session**: outputs added. Standardized rule adopted:
a node wrapping a constructor has exactly one output, an instance of that
class -- but *inputs* are 100% derivable from the real signature alone
(structural fact), while an output's semantic *category* (e.g. that
`ChunkedXPUCAME` specifically is "an Optimizer") is domain knowledge the
class can't self-report, so it's supplied explicitly by whichever
introspection call already knows the domain (`introspect_optimizers()`
passes `category="optimizer"`). A class introspected without a supplied
category correctly shows zero outputs -- rendered honestly in the UI as
"not yet standardized" rather than a fabricated guess. Sidebar link added
(`/nodegraph`, one click from the main dashboard, mirroring the existing
`/datasets` link).

## Open questions for the user

1. ~~Sequencing buy-in~~ -- **Resolved**: custom lightweight canvas (not
   ComfyUI's), built early as a dev-testing scaffold alongside the backend
   refactor rather than deferred to the end, on an isolated `/nodegraph`
   tab that doesn't affect the production config path until deliberately
   switched over. See "Refined design decisions" above.
2. Any existing config files / running training setups that need to keep
   working *unchanged* throughout the migration (i.e. is backward TOML
   compatibility a hard requirement, or is a config format change acceptable
   as part of this)? **Still open.**
3. ~~Comfortable starting Phase 1~~ -- **Resolved**: first slice shipped
   (optimizer classes introspected and rendered on the new `/nodegraph`
   tab). Next: decide whether to continue deepening Phase 1 (e.g. add
   `step()`'s runtime signature too, not just `__init__`; add a way to
   actually instantiate + smoke-test a node from the playground) or move to
   defining the formal `Node`/`Port` base classes and wrapping the first
   real node (`OptimizerNode`) behind them, per the Phase 1 description
   above. Needs the user's steer on which feels more valuable to see next.
