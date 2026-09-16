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
- [ ] **Run `nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py`
      and report the output back.** Written 2026-09-16, not yet run (no
      torch in this sandbox). Tests three things separately, since the
      three legacy classes don't share one mechanism (see above):
      (A) the hypothesis that `ForeachXPUAdafactor` has no gap at all —
      if confirmed, `foreach_adafactor.py`/`ForeachAdafactorOptimizerNode`
      can be deleted immediately, the same way `came.py` was, no
      algorithm work needed; (B) the actual size of `FusedXPUAdafactor`'s
      real per-parameter gap; (C) the actual size of `ChunkedXPUAdafactor`'s
      cross-parameter-batching gap, for reference. Next steps depend on
      what comes back:
      - If (A) confirms no gap: delete `foreach_adafactor.py` +
        `ForeachAdafactorOptimizerNode` right away.
      - (B) is self-contained (a per-parameter `AdafactorAlgorithm`
        branch) and could be implemented once the script confirms the
        exact numbers to match — but doing this *unconditionally* would
        make `ComposedAdafactorOptimizerNode(strategy="foreach")` start
        diverging from `ForeachXPUAdafactor` (which currently matches
        *because* neither side special-cases tiny parameters) — so this
        needs the Algorithm to know which family it's being used in, or
        a separate Fused-only override, not a blanket change. Needs a
        real design decision, not just an implementation.
      - (C) needs new `ExecutionStrategy`-level machinery (something
        like `ShapeGroupedBatchStrategy`, but grouping "under a size
        threshold" instead of "same shape") — real, separate feature
        work, bigger than a formula fix. Not attempted; `AdafactorOptimizerNode`
        stays registered regardless of what (A)/(B) show.
- [x] **Surfaced the momentum-corruption bug precisely** — turned out to
      affect `ChunkedXPUAdafactor` (main/large-parameter path only) and
      `FusedXPUAdafactor` (all parameter sizes), not `ForeachXPUAdafactor`
      at all (confirmed by reading the exact in-place-vs-copy pattern in
      each, not assumed to be shared). Full writeup in
      `docs/known-issues/open.md`.
- [ ] Minor: `core/optimizers.py:470` has a runtime message that still
      recommends switching to `ForeachCAMEOptimizerNode` by name — that
      class no longer exists. Not fixed here (`core/` untouched by this
      cleanup), flagging for whoever next touches that file.

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
- [ ] `docs/design/resources-controller/*` — left alone, same as the
      code (off-limits route).
- [ ] **Residual, smaller scope than originally listed:** the general
      docs/design/*.md trim (per the stated criteria -- keep only
      hard-to-derive-from-code info, specific design rationale,
      resources, actively-updated plan) hasn't had a full line-by-line
      pass beyond the specific inaccuracies fixed above. Files not yet
      read closely for this: `01-foundational-ontology.md` (name
      uncertain, check `docs/design/README.md`'s index), `02-...`,
      `04-...`, `05-...`. Lower priority than everything above -- these
      weren't flagged as *wrong*, just not yet checked for redundant-
      with-code narrative bulk.

### Still to check (broader sweep, not yet done)
- [ ] `nodes/model/` (LoRA/UNet injection), `nodes/model/text_encoder.py`,
      `nodes/dataset/managed.py` — all still import `core.lora`/
      `core.unet_wrapper`/`core.clip_encode`/`manager.loader` directly,
      with no independent alternative built yet anywhere (unlike
      optimizer/, where one already exists). This is *not* a "bad
      competitor" situation — there's only one implementation, just
      still `core`-coupled — so it's real future work, not cleanup, per
      "don't remove/replace not-yet-duplicated features." Flagged for
      awareness, not scheduled.
- [x] Swept `nodes/memory/`, `nodes/train/`, `nodes/primitive/`,
      `nodes/monitor/` for competing implementations of the same job
      (the pattern optimizer/ and dataset/ both had). Nothing found:
      `memory/`'s pieces (`MemoryManager`, `ResourceCoordinator`/
      `OffloadOrchestrator`, `ResourceControlHandle`, `ResourceProfile`,
      `VRAMBudgetControllerNode`) are each a distinct, complementary
      concern, not duplicate jobs. `train/`, `primitive/`, `monitor/`
      are all small and single-purpose by class-name inspection -- no
      wrap-vs-independent split like optimizer/ had anywhere in them.
