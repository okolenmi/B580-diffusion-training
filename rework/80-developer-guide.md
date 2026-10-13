# 80. Developer guide: how to write nodes (draft)

Status: **draft of the document that will ship** as `docs/developing-nodes.md`
once the design is settled. Until then it describes the *target* shape: sections
marked **(target)** use APIs that the plans introduce and that do not exist yet.
Sections marked **(today)** describe rules that already hold.

Why this exists: the owner's complaint is that every new agent arrives with its
own "vision". This guide is the one vision. It is short on philosophy and long on
concrete rules, because a rule an agent can check is followed and a value an
agent has to interpret is not. Every rule says **why**, and where possible **what
test pins it** (the repo's own convention from `docs/decisions/README.md`: "a
decision with no test is a comment that will be edited").

Read in this order: section 1 (the picture), section 2 (where code goes),
section 3 (anatomy), then the section for what you are building.

---

## 1. The picture in one page

* A **graph** is data: nodes, data wires, event wires, settings. It is run once,
  in a child process, forward through the data plane. (today)
* A **node** declares *what it takes and gives* (ports, groups, events, modules,
  settings it reads) and does its work in `build()`. It knows nothing about how
  it is drawn. (today for ports and modules; target for the rest)
* A **family** (`families/sdxl`, `families/anima`) is the only place that knows
  how a specific model is built, encoded, noised and adapted. Nodes call a
  family through one shared interface and never name a family. (target, plan 40)
* A **handle** is what travels on a wire: a real Python object with a class.
  The wire's type is that class. (today)
* **Events** are the second plane: how a graph goes backward (end of epoch,
  prompts changed). They are explicit wires, handled synchronously at safe
  points. (target, plan 10)
* **Settings** describe the environment of a run (device, precision, VRAM
  limits, seeds). Nodes read them; nodes do not ask for them as ports.
  (target, plan 20)
* **Views** (monitors, pages) and the editor draw nodes from catalog data. No
  per-node-type code exists in the frontend, and none must be added. (today)

The three questions to ask before any change:

1. *Which layer owns this?* (section 2)
2. *Is this a process choice (node field), an environment fact (graph setting) or
   a machine fact (app config)?* (plan 20, section 2)
3. *Is there an event for it, or is a wire the honest model?* (section 8)

---

## 2. Where code goes

### 2.1 Packages and their import rules

| Package | May import | Must not import | Notes |
|---|---|---|---|
| `families/api` | stdlib only | torch, safetensors, anything in the repo except itself | the server process must be able to import it (resolve hooks, detection) |
| `families/general` | `families/api`, torch | families, nodes, backend, manager | shared building blocks: CLIP, Qwen, VAE, LoRA layers, schedules |
| `families/<name>` | `families/api`, `families/general`; its own `nodes/` subfolder also `graph` | other families, `nodes/`, backend | everything that is true of one model (plan 42: family node files live in `families/<name>/nodes/`) |
| `graph/` | stdlib only (no torch) | everything else in the repo | node base, ports, events, settings, validator, runtime; the server imports it |
| `engine/` | `graph`, `families/api`, `families/registry`, torch | nodes, backend, concrete families | step phases and runtime services (residency, memory, capture) |
| `nodes/` | `graph`, `families/api`, `families/registry`, `data`, `engine`, `nodes/*` | a concrete family, `backend/` | thin; wires family calls and handles together |
| `data/` (replaces `manager/`, plan 90) | stdlib, torch, `families/api`, `families/registry` | nodes, backend | dataset library, latent cache, prompt table, batching; family constants come from `LatentSpec` |
| `backend/` | its own layers (docs/design/backend/01-architecture.md, the ten rules); `graph`, `families/api` | torch at import time; a concrete family | infrastructure may import `nodes/` (the catalog) |
| `frontend/` | itself | anything on the Python side by name | talks to the server over `/api/v1` only |

**Pinned by (target):** one import-boundary check (a new `scripts/check_imports.py`, wired into
`scripts/full_gate.sh` beside the existing `check_quality.py` and `check_docs.py`) that parses imports with `ast` and fails on any edge
this table forbids. It must exist **before** the first family file is written;
the rule has no value if it is only a document.

### 2.2 "Where does this logic go?" decision list

Ask in order; the first yes wins.

1. Is it true only because the model is SDXL (or Anima)? -> `families/<name>/`.
2. Is it true of several models but needs torch? -> `families/general/`.
3. Does the server need it without torch (listing, detecting, validating)? ->
   `families/api` (types, protocols) or a `fingerprint.py` next to the family.
4. Is it about *running a graph* (ordering, events, completion)? -> `nodes/core.py`
   and the runtime, never a node.
5. Is it about *drawing or editing* a node? -> catalog metadata (`Group`,
   `Port.ui`, `MODULES`) plus the generic editor code, never a node-specific
   branch in the frontend.
6. Is it a step of the training process that a person may want to replace? -> a
   node (one clear job, ports in/out).
7. Is it a number that every training step needs fast? -> `engine/` (plan 70),
   not a node and not a handle method that allocates.
8. Anything else is probably a utility: put it where its **only caller** lives
   and promote it when a second caller appears. No `utils.py`, no `common/`.

### 2.3 One file, one job

A file that needs a section header comment to separate two topics is two files.
Soft limits that trigger a split discussion in review: 400 lines for a node file,
300 for a family area file, 500 for a frontend module. They are not tests; a
reviewer may accept an exception that is written down in the file's docstring.

---

## 3. Anatomy of a node (today + target)

### 3.1 The smallest correct node (today)

```python
class FloatConstantNode(Node):
    INPUTS  = {"value": Port(name="value", type=float, required=False, default=0.0)}
    OUTPUTS = {"value": Port(name="value", type=float, required=True)}

    def build(self, **inputs):
        self.validate_inputs(inputs)
        result = {"value": float(inputs.get("value", self.INPUTS["value"].default))}
        self.validate_outputs(result)
        return result
```

(`nodes/primitive/values.py`.) Port dictionaries are class variables, so the
catalog reads them without instantiating. `__init_subclass__` validates the
declarations at import time (`visible_when` typos, port-name mismatches): a
broken node fails at import, not at the person's first run.

### 3.2 A full node (target)

```python
class SwapTagsNode(Node):
    DISPLAY_NAME = "Swap tags"
    CATEGORY     = "Data / Captions"            # palette location
    GROUPS = (
        Group("main",  "Tags"),
        Group("seed",  "Randomness", collapsed=True, advanced=True),
    )
    INPUTS = {
        "prompts": Port(name="prompts", type=PromptTable, required=True),
        "per_image": Port(name="per_image", type=int, default=1, group="main",
                          doc="Tags kept for each image in one epoch."),
        "seed": Port(name="seed", type=int, default=None, group="seed",
                     ui={"placeholder": "derived from graph seed"}),
    }
    OUTPUTS = {"prompts": Port(name="prompts", type=PromptTable)}
    EVENTS_IN  = {"epoch_ending": EventIn(EpochEnding, handler="on_epoch_ending")}
    EVENTS_OUT = {"prompts_changed": EventOut(PromptsChanged)}
    READS_SETTINGS = ("seed.base",)
    MODULES = (Scalar("tags_per_image", "Tags per image", source="tags_per_image"),)

    def build(self, **inputs): ...
    def on_epoch_ending(self, ev, ctx): ...
```

What each declaration is for, and what **not** to do instead:

| Declaration | For | Do not |
|---|---|---|
| `INPUTS` / `OUTPUTS` | the wire contract; widget ports are fields | add a field by reading `params` dicts by string key in `build()` |
| `GROUPS` + `Port.group` | the editor layout (plan 30) | put 9+ ports in `main`; the lint test fails the class |
| `Port.ui` | widget hints only | encode behaviour in a hint; hints are advisory |
| `EVENTS_IN/OUT` | the backward plane | call another node, or reach the hub by topic string |
| `READS_SETTINGS` | the dialog's "used by" and validation | read a setting you did not declare |
| `MODULES` | what the node can **show** (chart, scalar, bars, text, progress) | import or call anything from the monitor stack |
| `resolve_inputs/outputs` | shapes that depend on params (editing-time) | touch the GPU, read tensors, or take more than ~100 ms |
| `has_diagnostics` | live advice about the current parameters | run the node's real work |

### 3.3 `build()` contract

1. **Declared in, declared out.** Everything `build()` returns is an `OUTPUTS`
   key; everything it reads is an `INPUTS` key or a setting. `validate_inputs`
   and `validate_outputs` are the first and last lines (today's pattern).
2. **A builder is not a runtime object.** `build()` constructs and returns; it
   does not train, loop for hours, or hold state the next node mutates. Anything
   with per-step state is a *runtime object* created by the builder and handed
   on as a handle. (This is the repo's first design principle; the managed
   trainer is the one allowed exception, and plan 70 shrinks it.)
3. **Instances are not retained** unless the node declares `EVENTS_IN`
   (plan 10); then the runtime keeps it until the run ends. So a reactive node
   must keep its state on `self`, a non-reactive node must keep it in the
   handle it returns.
4. **No module-level mutable state.** Not a cache, not a "current" object.
   (Found in the audit: the LoRA timestep gate is a module global,
   `_current_gate`; two graphs in one process would share it. Plan 43 moves it
   onto the adapter.) Caches live on an object whose lifetime is a run.
5. **Idempotent and cancel-aware.** `ctx.should_cancel()` between units of work
   larger than about a second. Raise on failure; never return a half-built
   output and a log line.
6. **Completion** (today, doc 16): a node that can stop early reports
   `Completion.unfinished_at(...)`; a node that must still run after Stop sets
   `ACCEPTS_UNFINISHED = True` (savers). Do not invent another stop mechanism.

### 3.4 Handles (what travels on wires)

* A handle is a class. The class **is** the wire type, so a new kind of value
  means a new class, not a string tag.
* Handles are **small and lazy**: they hold paths, specs and references, not
  tensors, until a consumer needs them (`ModelWeights` is the pattern; `ModelPart`
  generalises it, plan 40). The graph holds all handles for the whole run, so a
  handle that holds a tensor holds VRAM.
* Handles carry a **`FamilyRef`** where the family matters, so two incompatible
  handles are refused at validation time, not as a shape error at step 3000.
* Never pass a **dict of untyped things** across a wire. If there are three
  values, make a small frozen dataclass.
* Heavy data goes in shared stores addressed by handle (latent cache, prompt
  table). Events carry **indices and versions**, never the data (plan 10).

---

## 4. Families: how to use and how to extend (target, plans 40 to 43)

### 4.1 Using a family from a node

```python
fam = families.get(part.family)                 # registry, lazy import
cond = fam.conditioning.encode(parts, prompts)  # opaque Conditioning
x_t, target = fam.objective.make_pair(latent, t, noise)
```

* A node asks the family; it never `if family == "sdxl":`. That `if` is the
  smell that the family interface is missing a method. Add the method to
  **both** families (or give a documented default in the Protocol), do not
  branch in the node.
* A node takes `family` from the handle it receives; a node that needs to
  *choose* one has a `choice` port fed by the registry.
* **Conditioning is opaque.** Do not assume a pair of tensors, a dtype, or a
  sequence length; use `Conditioning.to / cat / index / blend / nbytes`.

### 4.2 Adding a family

A checklist, in order. The first four steps need no model weights.

1. Fingerprint (`fingerprint.py`): pure, header-only, with golden header
   fixtures in `tests/fixtures/headers/<family>/` (key names, shapes, dtypes
   only: a few KB, no weights).
2. Types: the `ModelPart` subclasses, `LatentSpec`, `AdapterLayout`
   (`families/api` additions are rare; review them as public API).
3. Kits: `conditioning`, `latents`, `objective`, `adapters`, `memory`, each
   with a CPU-runnable smoke test on tiny random weights.
4. Registry entry and a `Family` object. Nothing else in the repo changes. If
   something else must change, the interface was wrong: fix the interface.
5. Starter graph (plan 60) for the family; it is the integration test.
6. Hardware numbers (VRAM, steps per second) go to
   `docs/known-issues/` **only after they are measured**, never guessed.

### 4.3 Adding a model type to an existing family

A fingerprint plus a kit entry. Detection tests with a header fixture. No node
changes.

---

## 5. Ports, groups and the editor (target, plan 30)

* **Group** every node that has more than eight widget ports; mark the rarely
  touched ones `advanced`. The lint test fails otherwise.
* Port names are **API** once saved graphs and starters exist: they are stored in
  graph JSON. Before the swap (plan 90) names are free to fix; after it, a rename
  means updating `contracts/`, the starters and every user in one commit. Labels
  are always free to change.
* A **widget port** (a field) and a **wire port** (a connection) are the same
  `Port`; do not make two ports for one concept to get a different look.
* `visible_when` hides fields that do not apply; hidden fields keep their value
  and are ignored.
* **Dynamic ports** (`DYNAMIC`) are for *detected* shapes only (Load Model,
  adapters). Do not use them for user-defined shapes; use a `multi` port.
* **Multi ports** (several wires into one input) declare their element type and
  their ordering rule (wire order is the argument order). The node must not care
  how many arrive.
* Never write frontend code that mentions a node class by name. If the editor
  needs a new behaviour, it needs a new **catalog field**, usable by every node.
  Pinned by (target) a frontend test that greps the editor sources for known
  node class names.

## 6. Resolve and diagnostics (today + target, plans 30, 41)

* `resolve_inputs/outputs(params)` answer: *given these field values, what
  ports exist?* They run in the server process, on a debounce, many times while
  someone types. Rules: pure, fast, **no torch import**, no disk beyond one
  bounded header read through the shared cache, deterministic, never raise
  for bad input (return an `error`/`needs_choice` status with the field named).
* `diagnostics(inputs)` are advisory strings about the current parameters
  (VRAM estimates, incompatible choices). Same constraints; they never block a
  run. Blocking problems belong in validation, which has codes.
* Long computations (hashing a 6 GB checkpoint) are not resolve work. Detection
  reads headers only; anything heavier is a run-time step with progress.

## 7. Settings (target, plan 20)

* If two nodes would have to agree on a value, or a person would ask "why did I
  set this five times", it is a graph setting, not a port.
* Read through `self.context.settings.<group>.<key>`; declare it in
  `READS_SETTINGS`. Never read `os.environ`, `config.toml` or `torch.device`
  defaults directly in a node.
* A node may *override* a setting only if it declares a reason and shows an
  `Override` group that is empty by default.
* Adding a setting: a `SettingSpec` (key, label, doc, type, default, bounds,
  group, advanced) in the owning family or node class; one smoke test that the
  catalog serves it; its default is the old behaviour.

## 8. Events (target, plan 10)

Decide between a wire and an event with this table:

| The consumer needs... | Use |
|---|---|
| a value, once, before it starts | a data wire |
| to be told "something happened" while it is running | an event |
| to change something an earlier node produced | an event handled by the producer, or a new version of a shared store; **never** a backward data wire |

Rules:

1. Declare event ports; do not subscribe by topic string.
2. Handlers are short, deterministic, non-blocking; no sleeps, no network, no
   GPU sync.
3. Payloads are frozen dataclasses with a `summary()` (small, JSON-safe).
   Heavy objects go in shared handles; the payload carries an index or version.
4. Emit only from the run thread, at a safe point you define. Worker threads
   call `hub.post()`.
5. A handler assumes the emitter is paused, and must leave shared state
   consistent when it returns (that is the barrier).
6. Emit `EpochEnding` before the next pass starts; `StepFinished` after
   metrics are final. Do not invent a synonym of an existing event; extend the
   payload.
7. Never turn events into a hidden control flow of your own. If a node needs
   five events from the same source to work, the design wants a different
   contract.

## 9. Data, datasets and latents (target, plan 50)

* Latent spec (channels, factor, scale) comes from the family, never a
  literal; the audit found scale 0.13025, 4 channels and factor 8 spread over at least four files (`vae.py`, `vae_decode.py`, `manager/builder.py`, `unet_wrapper.py`).
* Caches are keyed by everything that changes their meaning (VAE identity,
  family, resolution policy, caption version) and carry that key in their
  header. A cache never silently serves the wrong key.
* A stage that rewrites prompts bumps `PromptTable.version` and emits
  `PromptsChanged`; downstream caches that depend on prompts invalidate by
  version.
* Randomness: derive from `seed.base` and the node's stable `uid`. No
  `random.random()` without a seed.

## 10. Concurrency, memory and the device

* The run is **single-threaded for graph logic**; threads are for I/O and
  prefetch only (`PrefetchingBatchSource`). Worker threads do not touch the
  hub, the monitor bus or any node state; they hand results over through a
  queue.
* Device and dtype come from settings and the part's own record, never from a
  literal `"xpu"` or `torch.bfloat16` in a node. (The repo's ADR 0004 hardcodes
  `xpu` as policy; the setting is where that single choice lives.)
* **Account for memory where you allocate:** anything that holds more than a
  few MB on the device reports through the memory interface (ADR 0005), so
  admission control sees it. A large allocation that bypasses it is a bug even
  if it "fits on my machine".
* Never call `empty_cache`/`synchronize` from a node; those belong to
  `engine/runtime`, which knows what the step is doing. (Found: `text_encoder.py`,
  `clip_encoder.py`, `vae_decode.py`, `supervised.py` and `managed.py` all call
  `empty_cache` themselves.)
* CPU first: every node must run on CPU with tiny fixtures so that CI and
  sandboxes can test it. Hardware-only behaviour goes in `gpu/` tests and is
  labelled as such.

## 11. Errors and logging

* Fail early and name the thing: node uid, port, file, the value. A message
  that cannot be acted on without a debugger is a bug.
* Do not catch `Exception` to continue. Catch the specific exception, add
  context, re-raise or convert to a validation code.
* Warnings to the person go through the node's `diagnostics` or an `info`
  display, not `print` and not the log alone.
* No silent fallbacks: if a requested option is unavailable, say so and stop,
  or select the fallback **visibly** (an info line on the node).

## 12. Comments, docstrings and docs (a real problem in this repo)

The audit counted about 41 percent of lines under `nodes/` as comments or
docstrings, many of them hardware measurements, history ("this used to...") and
incident narrative. They go stale and they bury the code. The rules:

| Put it in | What |
|---|---|
| the code's name or type | what a thing is |
| a short comment | **why** a non-obvious choice was made, one to three lines, true now |
| the docstring | the contract: what a caller must know, no history |
| `docs/decisions/NNNN-*.md` | a decision with a cost and a test that pins it |
| `docs/known-issues/*.md` | measured numbers, incidents, open or resolved findings; code **cites** the entry, it does not copy it |
| `CHANGELOG.md` | what changed for a user |
| git history | how it used to be |

Hard rules:

1. **No measured numbers in code comments** (VRAM, speed, thresholds). They go
   to `docs/known-issues/` with the date and hardware, and the comment points
   there. `scripts/check_docs.py` already checks numbers asserted in docs against
   the code; extend it, do not duplicate the habit.
2. **No history in comments** ("previously", "used to", "after the fix").
3. A comment that restates the next line is deleted.
4. If a long explanation is needed to make the code understandable, the design
   is the problem; say so in a decision record and simplify.
5. Do not delete or rename a `docs/known-issues` entry that a source comment
   cites (the known-issues README says this; keep it).
6. Docs are checked by `scripts/check_docs.py` in `full_gate.sh`: a new fact
   about the code that is a number or a name should be registered there.

## 13. Testing

* Style: standalone scripts with the repo's `check()` helper (no pytest), under
  `next/tests`, `backend/tests`, `frontend/tests` (`node --test`).
* **Policy for the rework (owner, 2026-10-13):** much is rewritten, so detailed
  per-feature suites are not written. What exists: (1) cheap **structural checks**
  over the whole tree (compile, import boundaries, a declaration lint that loads
  every node class and validates ports, groups, events, settings); (2) a few
  **pure self-checks** for the graph core, detection and events, because they run
  anywhere and catch the commonest AI rewrite errors; (3) **one acceptance test**
  that runs the starter graphs end to end on synthetic data (plan 90, section 6);
  (4) the owner's hardware run. GPU-only behaviour is labelled as such.
* **Optional math spot checks** against the old tree while it still exists: a few
  numbers (noise schedule, loss target, LoRA forward, VAE scaling). After the swap
  they are replaced by saved golden values.
* **A bug fix still gets the test that fails without it** once the acceptance test
  exists; before that, fix and note it in the ledger.
* Determinism: seeded everything; no wall-clock assertions; time injected.
* Fixtures never contain weights. Header fixtures are JSON; tiny random
  models are built in the test (a few thousand parameters).
* Do not mock the layer you are testing. Fakes are for ports owned by
  `application/`; nodes are tested with real handles.

## 14. Anti-patterns (each one found in this repository)

| Anti-pattern | Instead |
|---|---|
| `if family == "sdxl"` in a node | a family method (section 4.1) |
| The same environment value as a port on five nodes (`device` appears on five) | a graph setting |
| Module-level "current" object (`_current_gate`) | state on an object with run lifetime |
| A node that is also a controller (the VRAM budget node) | a setting; the runtime builds the controller |
| A handle with six consumers and **no producer node** (six such types exist) | build the node that produces it, or delete the handle |
| A 31-port node with no structure | groups now, then split the job (plan 70) |
| Hard-coded latent constants (`0.13025`, `8`) | `LatentSpec` from the family |
| Per-node cache flush calls | `engine/runtime` owns them |
| Two documents for one truth (a template holds a graph copy) | a view inside the graph (plan 60) |
| Node ids as stable addresses | `uid` |
| Comments that carry measurements and history | section 12 |
| A frontend branch per node class | a catalog field |
| A resolve hook that opens the model | header-only detection with a size cap |
| Silent fallbacks | visible choice or failure (section 11) |
| `utils.py` | the caller's package |

## 15. Changing a public contract

Public contracts: port names, handle classes, event types, graph JSON keys,
catalog format, setting keys, REST routes, DB schemas. They are written in
`contracts/` (plan 90, section 5). To change one:

1. Change the contract file first, with the reason in a line.
2. Search for every user (`grep`), update them all in the same commit, including
   the starters and the frontend.
3. Bump `catalog_format` or the graph `format` if the editor must tell.
4. No compatibility layer is kept (owner decision, plan 90). If that decision is
   ever reversed, migrations are a **new** plan, not an exception here.

## 16. Checklists

### 16.1 New node

- [ ] Right package (section 2); right category in the palette.
- [ ] Ports typed with real classes; names final (they are API).
- [ ] Groups if more than eight fields; advanced marked.
- [ ] Reads environment through settings and declares `READS_SETTINGS`.
- [ ] No family branches; no module state; cancel-aware.
- [ ] `MODULES` if it has anything worth watching.
- [ ] Events declared if it needs backward flow; payloads are frozen.
- [ ] Declaration test + logic test + graph test; CPU only.
- [ ] Docstring is a contract; comments are why-only; no measurements.
- [ ] CHANGELOG line if a user sees it; `docs` registered if it asserts a number.

### 16.2 Review (what a reviewer asks)

1. Which layer owns this, and did the import test pass?
2. Could it have been a setting, an event or a family method instead?
3. What breaks in the starters, the contracts or the frontend, and were they updated in the same change?
4. What happens on Stop, on failure, on a second run in the same process?
5. Where does its memory show up in admission control?
6. Would the next agent understand why, from the code and one decision record?

## 17. Process rules for agents working on this repo

* Read `rework/00-index.md` (during the rework) or `docs/design/README.md`
  (after) before editing.
* Facts about code go in `audit/` or the doc with file:line; do not re-derive
  them each session.
* A decision the owner made is recorded with a date in the plan or ADR it
  concerns; later agents do not reopen it without a new reason.
* A change that cannot be verified here (hardware, weights, network) is marked
  unverified in the commit message and in
  `docs/known-issues/pending-testing.md` with how to confirm it.
* Small commits; each runs the gate it touches. Do not reformat files you do
  not otherwise change.
* The sandbox-only workflow (patches via `git format-patch`) does not change
  any rule above.
* During the rework follow the method of `90-build-plan.md` section 3 (unit of work
  is one new file, triage and lessons first, contracts before bodies, ledger
  updated with every file). It is written for how an AI works best: read, then
  rewrite against the plan; do not port line by line.

## 18. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Where this guide lives | one file; a folder | `docs/developing-nodes.md` (short, linked) plus the plans kept as `docs/design/` records after the rework |
| D2 | Enforcement | review only; tests for the mechanical rules | Tests for imports, groups, no frontend node names, no module-level mutable state in `nodes/`; review for the rest |
| D3 | Comment policy | advisory; checked | Advisory plus one cheap checker for digits-with-units (`MB`, `GB`, `it/s`, `ms`) in comments, warning only |
| D4 | Plan documents after the rework | delete; keep | Move to `docs/design/` as numbered records, mark superseded where reality differed |

## 19. Open questions for the owner

1. Do you accept the comment policy of section 12 (measurements to
   `docs/known-issues`, no history in code)? It means a one-time cleanup of
   existing comments, which I would do in its own commit, not mixed with logic.
2. Is the 400/300/500 line guidance acceptable as soft limits, or do you want
   a hard check?
3. Do you want the import-boundary test (section 2.1) to be a required part of
   `full_gate.sh` from the first commit of the rework? Recommended.
