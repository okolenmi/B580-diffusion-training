# Cleanup plan (living document)

Started per the project owner's direction: "wrap `core/`, don't rewrite
it" is no longer the rule. Where a proven, independent (non-`core/`-
wrapping) alternative already exists, it's the canonical route and the
legacy wrapper should go. Where no alternative exists yet, that's future
work, not cleanup — don't build new things just to have something to
delete. The one exception, off-limits entirely: the Resources Controller
route (`resources_controller.py`, `lora_training_config.py`,
`lora_training_resources.py`, `sdxl_architecture.py`, `resource_inspection.py`)
— better design, not finished, may replace the current training flow
later. Leave it, including its own stale internal claims (e.g. its
docstring still says Phase 6 "not built yet").

Update this file as work happens. Each item: status, what it is, why.

## Done

- **`nodes/resource_policy.py` removed.** `ResourcePolicy`/`ManualResourcePolicy`
  were unreachable from the graph editor — no Node ever produced a
  `ResourcePolicy`, so `ComfyUNetLoRANode`'s `resource_policy` Port could
  only ever be populated by hand-written Python (only its own smoke
  test did). `ResourceBudget` (same file, unrelated class, actually used
  by `checkpoint_placement.py` and `memory/`) was split out into its own
  `nodes/resource_budget.py` rather than deleted with the rest.
- **Stale NF4 docstring fixed.** `nf4_weight_store.py` claimed NF4 was
  "not yet wired into a real forward pass" — no longer true since
  `nf4_lora_layer.py` + `adapter_strategy.py`'s `PlainLoRAAdapter` landed.
- **AdamW unified onto `ComposedAdamWOptimizerNode`.** Deleted
  `adamw.py` (`AdamWOptimizerNode`, `SimpleAdamWOptimizerNode`).
  `SimpleAdamWOptimizerNode` was a straight duplicate. `AdamWOptimizerNode`
  wrapped `CPUAdamW` for CPU-resident optimizer state, a real tradeoff
  in the abstract — but for a full-parameter fine-tune, which nothing in
  this codebase can produce (every `TrainableModel` is LoRA-injected;
  see `nodes/model/handle.py`). No equivalence gap either way: AdamW's
  math doesn't branch on parameter size or shape.
- **CAME unified onto `ComposedCAMEOptimizerNode`.** Deleted `came.py`
  (`CAMEOptimizerNode`) and `foreach_came.py` (`ForeachCAMEOptimizerNode`).
  Both fully proven equivalent (float32 ~4e-6 max abs diff, bf16 bounded
  growing divergence attributed to ordinary low-precision noise, not a
  missing code path — see `smoke_test_came_equivalence.py`). CAME has no
  tiny-parameter special case anywhere in `core/optimizers.py`, checked
  directly.
- Updated all cross-references for the above: `server/nodegraph_registry.py`,
  `server/nodegraph_introspect.py`, `server/routes_nodegraph.py`,
  `server/smoke_tests/smoke_test_graph_executor.py`,
  `server/smoke_tests/smoke_test_nodegraph_introspect.py`,
  `nodes/optimizer/handle.py`, `nodes/optimizer/composed_adamw.py`,
  `nodes/optimizer/composed_came.py`, `nodes/optimizer/composed_adafactor.py`,
  `nodes/smoke_tests/smoke_test_device_resident_retrofit.py`,
  `nodes/smoke_tests/smoke_test_lora_injector_extraction.py`. Deleted
  `smoke_test_simple_adamw.py`, `smoke_test_resource_policy.py`.
- **`introspect_optimizer_nodes()` + `/nodegraph/optimizers` route removed**
  (`server/nodegraph_introspect.py`, `server/routes_nodegraph.py`). A
  second, independent "bad competitor" found along the way: a hand-
  maintained, hardcoded duplicate of the generic `/nodegraph/registry`
  endpoint. Already stale before this cleanup even started — its
  hardcoded class list was missing `ForeachCAMEOptimizerNode` and every
  `Composed*` node. Nothing outside its own two files referenced it
  (checked).
- **Relocated, not resolved:** the `nodes/model/`/`nodes/dataset/managed.py`
  `core`/`manager` coupling was tracked here as a "still to check" item,
  but it isn't cleanup debt — no competing implementation exists to
  retire in favor of, just a domain nobody's rewritten yet, same as
  `optimizer/` before this pass started. Moved to
  `docs/design/09-prioritized-backlog.md` as a real, sized future item
  instead of carrying it here indefinitely.
- Fixed a trivial, unrelated stale reference found in passing:
  `core/optimizers.py`'s own runtime log message recommended switching
  to `ForeachCAMEOptimizerNode` by name; pointed at
  `ComposedCAMEOptimizerNode(strategy="foreach")` instead.

## Corrected mid-stream (leaving the record — this is exactly the kind
## of mistake worth catching, not hiding)

- Initially planned to also delete `adafactor.py`, `foreach_adafactor.py`,
  `fused_adafactor.py` as "proven redundant" the same way as CAME.
  **Wrong — caught before deleting**, though it took two more rounds of
  correction to get the actual picture right (recorded here in full
  since it's a good example of why "read the summary" isn't enough):
  first pass, wrongly assumed all three legacy classes share one
  tiny-parameter (< 10,000 element) mechanism, based on
  `smoke_test_adafactor_equivalence.py`'s scoping comment. Reading
  `core/optimizers.py` directly instead shows three *different* things:
  `ChunkedXPUAdafactor` ties every tiny parameter in the whole optimizer
  together into one shared clip/EMA state (cross-parameter batching, not
  a per-parameter algorithm concern); `FusedXPUAdafactor` has a real but
  *different*, genuinely per-parameter elementwise EMA; `ForeachXPUAdafactor`
  has no tiny-parameter special case at all — its factored/unfactored
  paths use the same math `AdafactorAlgorithm` already implements,
  for every parameter size. So `ForeachAdafactorOptimizerNode` may
  already be safe to delete with zero algorithm changes — a script now
  exists to confirm this (see below) rather than assuming it from
  reading the source alone, given the track record in this section.
  Separately, reading the exact lines around the tiny-parameter code
  also surfaced a real, unrelated momentum-corruption bug in both
  `ChunkedXPUAdafactor` and `FusedXPUAdafactor` (not `ForeachXPUAdafactor`)
  — see `docs/known-issues/open.md`.
- Also caught deleting `nodes/resource_policy.py` wholesale without
  first checking whether anything else in that file was still live —
  `ResourceBudget` was, and got recovered into its own file rather than
  silently broken.

## Not yet done

### Optimizer domain
- [x] **Ran `nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py`
      on real torch (2026-09-16, user-run, results reported back).**
      (A) confirmed: all four (dtype × momentum) configurations came
      back at floating-point-noise magnitude (float32 1.2e-07–4.8e-07;
      bf16 0–2.4e-04) — no systematic divergence.
      `foreach_adafactor.py`/`ForeachAdafactorOptimizerNode` deleted as
      a result, cross-references fixed (`server/nodegraph_registry.py`,
      `smoke_test_device_resident_retrofit.py`,
      `composed_adafactor.py`, `docs/design/10-node-surface-and-precision-control.md`).
      (B) confirmed a real gap, but specifically for *factored* (2D+)
      parameters (9.8e-04–7.8e-03, an order of magnitude+ above the (A)
      noise floor) — the *unfactored* (1D) case came back at noise
      level too, which makes structural sense on reflection:
      `AdafactorAlgorithm`'s regular path for a 1D parameter is already
      a plain elementwise EMA, same as the tiny path would be, since
      row/col factoring only exists for 2D+ tensors in the first place.
      `FusedAdafactorOptimizerNode` stays registered.
      (C) confirmed a real gap in both factored and unfactored cases
      (3.6e-03/1.8e-04 float32, 7.8e-03/2.0e-03 bf16) — the
      cross-parameter batching contaminates the 1D result too, unlike
      Fused's independent per-parameter hooks. `AdafactorOptimizerNode`
      stays registered; closing it needs new `ExecutionStrategy`-level
      machinery (something like `ShapeGroupedBatchStrategy`, but
      grouping "under a size threshold" instead of "same shape"), not
      an algorithm change — real, separate feature work, not attempted.
      Full numbers preserved permanently in the script's own docstring
      rather than just here, since it's the more durable home for them.
- [x] **Found and fixed a related but separate issue while investigating:**
      the float32+momentum corruption bug doesn't affect all three
      legacy classes the way it was first written up — only
      `ChunkedXPUAdafactor` (large-parameter path only) and
      `FusedXPUAdafactor` (all sizes); `ForeachXPUAdafactor` uses a
      non-in-place `.mul()` and is unaffected. `docs/known-issues/open.md`
      corrected.
- [x] **Found a pre-existing planning doc for this exact work while
      fixing cross-references:** `docs/design/10-node-surface-and-precision-control.md`
      section 11.1 ("Optimizer node consolidation") turned out to
      already sketch this whole consolidation, written earlier and
      never executed. Two things worth knowing about it: it also
      wrongly grouped `AdafactorOptimizerNode` with the "safe to
      retire" set (same over-generalization corrected above, now fixed
      in that doc too) — and it argued for *soft-deprecating* nodes
      (docstring warnings, keep the class) rather than deleting them
      outright, specifically because saved graphs might reference them
      by name. That's not the approach actually taken (outright
      deletion, per explicit direction, git history as the safety net)
      — recorded in that section now, not silently overridden.
      **Real open question surfaced by the same doc, not yet decided:**
      it separately argued `SimpleAdamWOptimizerNode` (bare
      `torch.optim.AdamW`, no custom code) vs.
      `ComposedAdamWOptimizerNode` (this project's own implementation,
      more features) was a genuine "trust the standard library" vs.
      "our own code" choice, not redundancy — a real, different kind of
      argument than `AdamWOptimizerNode`'s (which had no actual
      consumer). `SimpleAdamWOptimizerNode` was already deleted in the
      first commit of this pass, before this doc was found, without
      weighing that argument. Flagged for the project owner to confirm
      or overturn, not decided here either way.
- [x] **Surfaced the momentum-corruption bug precisely** — turned out to
      affect `ChunkedXPUAdafactor` (main/large-parameter path only) and
      `FusedXPUAdafactor` (all parameter sizes), not `ForeachXPUAdafactor`
      at all (confirmed by reading the exact in-place-vs-copy pattern in
      each, not assumed to be shared). Full writeup in
      `docs/known-issues/open.md`.
- [ ] **Implemented the `FusedXPUAdafactor` tiny-parameter fix (Part B's
      gap) — needs a real torch run to confirm before trusting it.**
      `AdafactorAlgorithm` grew an opt-in `tiny_parameter_threshold`
      (default `None`, unchanged behavior everywhere except below) —
      turned out to be a small change, not new algorithm work: the
      existing elementwise `"vs"` state/update path already handles any
      parameter shape correctly (it was only ever *selected* by
      dimensionality, `len(param_shape) >= 2`, not by a separate
      formula for 1D vs. tiny-2D), so this just changes *when* that
      already-correct path gets chosen. `ComposedFusedAdafactorOptimizerNode`
      now passes `10_000`; `ComposedAdafactorOptimizerNode` (chunked/
      foreach/simple/shape_grouped) deliberately doesn't, since Foreach's
      Part A match depends on *not* special-casing tiny parameters.
      One small, documented, deliberate simplification: doesn't
      replicate `FusedXPUAdafactor`'s lazy first-update initialization
      quirk exactly — estimated ~1e-4 relative on the first update only,
      versus the ~1e-3–1e-2 gap being closed; if Part D's real numbers
      don't bear that estimate out, that's a real problem to fix, not a
      tolerance to widen. `nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py`
      grew a Part D to check this — **needs to be run** before
      `fused_adafactor.py`/`FusedAdafactorOptimizerNode` can be deleted
      the way `foreach_adafactor.py` was.
- [ ] `AdafactorOptimizerNode`'s gap (Part C, cross-parameter batching)
      — real, separate `ExecutionStrategy`-level feature work, not an
      algorithm fix like the above. Relocated to
      `docs/design/09-prioritized-backlog.md` as a sized future item
      rather than tracked here — not cleanup debt in the same sense as
      the rest of this file, since it needs new capability to be built,
      not just a wrapper retired.

### Dataset domain — second "bad competitor" found, done
- [x] **`nodes/dataset/renoise.py` migrated off `core.noise_schedule`
      onto `components/diffusion.py`.** `_renoise()` now uses
      `DiscreteLinearNoiseSchedule`/`EpsParameterization`/
      `VPredParameterization` (constructed once in `__init__`, reused
      per batch) instead of importing `get_alpha_sigma`/`eps_to_x0`/
      `eps_to_vpred`/`vpred_to_x0` from `core.noise_schedule`. Matches
      what `train/step_pipeline.py`/`train/supervised.py`/`train/loss.py`
      already used. `sample_timestep` stays a deferred `core.noise_schedule`
      import -- random-draw strategy, not diffusion-process math, no
      `components/` equivalent exists, not a competing implementation.
      `smoke_test_renoise.py` needed no changes (black-box, builds its
      own independent reference via `core.noise_schedule` directly).

### Docs
- [x] `docs/status/progress.md`: fixed the "wrap-don't-copy... deliberately
      untouched" framing, and its now-inaccurate `ResourcePolicy` entry
      (rewrote to describe `ResourceBudget` alone, note the removal and
      why, and correct the pre-existing overstatement that `group_policy`
      routed through `ResourcePolicy` -- it never did, separate
      mechanism, `ParameterGroupPolicy`).
- [x] `docs/architecture.md`: fixed the same "wrap `core/`, don't
      rewrite it" framing (the pipelines table + the prose under it) to
      describe the actual current rule -- `core/`/`manager/` stay
      unmodified, but `nodes/` retires a wrapper once it has a proven
      independent replacement, and points at `docs/CLEANUP_TODO.md`.
- [x] **`ResourcePolicy`/`ManualResourcePolicy` design-doc rewrite done**
      across all six files: `03-training-step-orchestration.md` (the
      full 2.2 section -- kept all the real "why 3 of 7 methods, why
      each cut method still has its real home" reasoning, reframed to
      past tense, added the actual removal reasoning, fixed the stale
      `resource_policy` port mention at the old line ~194 too),
      `06-composition-walkthrough.md` (the code walkthrough no longer
      constructs a `ManualResourcePolicy` -- uses the three real,
      current direct Ports instead), `07-deferred-or-rejected.md`
      (`AutoResourcePolicy` entry no longer implies `ResourcePolicy`
      still exists), `08-validation-and-implementation-status.md` (the
      status table row rewritten -- also fixes the `group_policy`
      mischaracterization at its source), `09-prioritized-backlog.md`
      (one-line note added to the completed-items list),
      `10-node-surface-and-precision-control.md` (two precedent
      citations, tense-adjusted). Teaching content (the forward-
      reference-type-hint pattern, the "why each of the 7 sketched
      methods got cut" reasoning) kept, not deleted -- it's still
      accurate about *why* the current Ports are shaped the way they
      are, just no longer describes a `ResourcePolicy` that exists.
- [x] Trimmed `docs/review_notes.md` from 333 to ~80 lines per its own
      stated rule ("delete resolved items rather than leaving them as a
      changelog") -- most of it was resolved items marked "Fixed in
      this/a follow-up pass," directly contradicting that rule. Kept
      only the 6 genuinely still-open items, renumbered, each
      independently re-verified against the actual current repo state
      before being kept (not just carried forward from the old text --
      one claim in my own first draft turned out to be wrong on
      inspection, about `known-issues/pending-testing.md`/`resolved.md`
      contents, corrected before committing). Also fixed `README.md`'s
      "Current status" section, which claimed `docs/status/progress.md`
      "predates" the Resources Controller/VRAM-budget/state_precision
      work -- checked, and it doesn't; `progress.md` already covers all
      of that. That claim (and a matching "has drifted" claim further
      down) was itself stale, independent of anything in this cleanup.
      **Side effect, not fixed:** two files in the off-limits
      `docs/design/resources-controller/` route cite the old
      `review_notes.md` list by item number (`README.md` line 13 "item
      7", line 40 "item 1"; `08-consolidation.md` line 193 "item 8") --
      those numbers no longer point at the same content after the trim.
      Not fixed here since fixing them means editing files in the
      protected route. `README.md`'s line 40 citation ("progress.md has
      drifted -- see item 1") is now doubly wrong: the item it cites
      doesn't exist at the new numbering, *and* the claim itself
      (progress.md has drifted) is no longer true either, per the fix
      above. Whoever next touches that route should know both things.
- [x] **Residual docs/design/*.md check, done — found clean, no changes
      needed.** Read `01-design-goals-and-constraints.md`,
      `02-foundational-ontology.md`,
      `04-lora-adapter-mechanics-and-loss-weighting.md`,
      `05-coordination-registry-observability.md` (the four not yet
      touched by anything above). No references to any deleted class,
      no stale claims found. `docs/design/README.md`'s own stated
      policy ("sections describing something now implemented keep their
      rationale but no longer repeat the illustrative class code") is
      already being followed — these four files have almost no
      illustrative code left (1–2 code blocks each, `01` has none). One
      marginal, defensible exception left as-is:
      `02-foundational-ontology.md` still illustrates `Port`/`Builder`
      (the design-doc-era name for what's now `Node`/`Port` in
      `nodes/core.py`) in full — arguably should point at the real file
      instead per the stated policy, but it's the foundational
      vocabulary the *rest* of these documents' own illustrations are
      written in terms of, so keeping one example of it is a reasonable
      call, not an oversight. Not changed.

### Still to check (broader sweep, not yet done)
- [x] Swept `nodes/memory/`, `nodes/train/`, `nodes/primitive/`,
      `nodes/monitor/` for competing implementations of the same job
      (the pattern optimizer/ and dataset/ both had). Nothing found:
      `memory/`'s pieces (`MemoryManager`, `ResourceCoordinator`/
      `OffloadOrchestrator`, `ResourceControlHandle`, `ResourceProfile`,
      `VRAMBudgetControllerNode`) are each a distinct, complementary
      concern, not duplicate jobs. `train/`, `primitive/`, `monitor/`
      are all small and single-purpose by class-name inspection -- no
      wrap-vs-independent split like optimizer/ had anywhere in them.
