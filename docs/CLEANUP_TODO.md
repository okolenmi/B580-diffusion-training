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
  **Wrong — caught before deleting.** `ChunkedXPUAdafactor`/
  `ForeachXPUAdafactor`/`FusedXPUAdafactor` all route parameters under
  10,000 elements through a real, structurally different tiny-parameter
  fast path (plain elementwise second-moment EMA, not the row/col
  factored approximation) — confirmed directly in `core/optimizers.py`
  (`TINY_NUMEL`/`_tiny_vs`/`_tiny_vs_map`), and confirmed as a real,
  deliberate scope exclusion in the equivalence tests themselves
  (`smoke_test_adafactor_equivalence.py`, `smoke_test_fused_adafactor_equivalence.py`
  both deliberately test only parameters >= 10,000 elements, for this
  exact reason). `AdafactorAlgorithm` doesn't implement that branch at
  all. Many individual LoRA matrices are smaller than 10,000 elements,
  so this isn't an edge case — it's common-case behavior these three
  Nodes still uniquely provide. **All three legacy Adafactor nodes stay
  registered for now.** See "Not yet done" below for the real fix.
- Also caught deleting `nodes/resource_policy.py` wholesale without
  first checking whether anything else in that file was still live —
  `ResourceBudget` was, and got recovered into its own file rather than
  silently broken.

## Not yet done

### Optimizer domain
- [ ] **Implement the tiny-parameter (`< 10,000` element) branch in
      `AdafactorAlgorithm`** (`nodes/optimizer/algorithms/adafactor.py`):
      a plain elementwise second-moment EMA (see `core/optimizers.py`
      lines ~1239-1249 for `FusedXPUAdafactor`'s version, ~180-250 for
      `ChunkedXPUAdafactor`'s batched version) in place of the row/col
      factored approximation, gated on total element count rather than
      dimensionality (the legacy classes check `TINY_NUMEL` *before*
      checking factored-vs-not). Touches `init_state()` (different state
      shape below the threshold) and both `compute_update()` paths
      (safe + in-place). **Needs a real torch environment to verify** —
      not attempted blind in this sandbox (no torch available, and
      getting per-parameter numerical code wrong silently is a real
      training-correctness risk, not a style issue). Once done and
      verified equivalent (same rigor as the CAME check), `adafactor.py`,
      `foreach_adafactor.py`, and `fused_adafactor.py` become safe to
      delete the same way `came.py`/`foreach_came.py` already were.
- [ ] **Surface `FusedXPUAdafactor`'s float32+momentum bug.** Found
      while reading `smoke_test_fused_adafactor_equivalence.py`: for a
      float32 parameter with `beta1` (momentum) set, `core.optimizers.
      FusedXPUAdafactor`'s momentum buffer gets silently corrupted every
      step (`g = self.exp_avg[i]` aliases the buffer; the following
      `.to(dtype=p.dtype)` is a no-op for float32, so the buffer gets
      mutated in place by the next line instead of a copy). Confirmed
      directly, not theorized (`check_legacy_float32_momentum_bug()` in
      that smoke test). `AdafactorAlgorithm` does not have this bug.
      `core/` is out of scope for this cleanup (untouched legacy math,
      not part of the `nodes/` rewrite), so this doesn't get fixed here
      — but anyone choosing `AdafactorOptimizerNode`/
      `ForeachAdafactorOptimizerNode` with float32 + momentum should
      know. Add to `docs/known-issues/open.md` (real, live, currently
      unflagged there) and/or a warning on the relevant Port docs.
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
- [ ] General pass per the stated criteria (hard-to-derive-from-code
      info, specific design-decision rationale, resources, actively-
      updated issues/plan -- not narrative duplicating what code
      docstrings already say). Do this after the code changes above
      settle and the `ResourcePolicy` rewrite above is done, not before.
- [ ] `docs/design/resources-controller/*` — left alone, same as the
      code (off-limits route).

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
