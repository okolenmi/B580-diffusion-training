# Iteration 3: final revision -- fixes only

Per the definition given for this pass ("iteration 3 should be just
fixes"): no new abstractions, no new techniques. Everything below is
either (a) an item from `docs/theoretical_pipeline_design_iteration2.md`'s
own "Left for iteration 3" list, resolved and pointed at where it actually
landed, or (b) something a genuine re-read caught that wasn't on that
list. Both kept short -- this document is a changelog, not a design doc;
the actual content lives in the edits it describes, in
`docs/theoretical_pipeline_design.md` and
`docs/theoretical_pipeline_design_iteration2.md`.

## Resolving iteration 2's leftover list, item by item

1. **"Finish `MinSNRLossWeighting`'s v-prediction branch."** On closer
   look, this was mis-filed in iteration 2 -- it's not a fix to the
   *design*, it's a small, already-scoped completion of *real, existing*
   code (`nodes/train/loss.py`), which doesn't belong on a design-fixes
   list at all. Reclassified: it's now backlog item 6 in
   `docs/theoretical_pipeline_design.md`'s "Prioritized backlog," grouped
   with `P2LossWeighting` (same file, different kind of work, stated as
   such there) -- not resolved here, correctly filed instead.
2. **"Redo `docs/theoretical_pipeline_design.md`'s section 4 (gap
   analysis) properly."** Done. Section 4 (`## Gap analysis`) is rewritten
   in place -- new "already matches" rows (2), new subpackage gaps for
   every iteration-2 addition (`ParameterGroupPolicy`,
   `AdapterStrategy`/`LoRAScalingPolicy`/`FrozenWeightStore`,
   `CheckpointPlacementPolicy`, the loss-weighting completion), and its
   opening paragraph now states plainly that the earlier version was
   provisional and this supersedes it, rather than sitting next to it.
3. **"Reconcile the Prioritized backlog."** Done as part of the same
   edit -- one renumbered sequence (12 items, not two separate lists),
   plus a new "Further out" tier for the items that need something the
   numbered sequence doesn't provide by itself (real profiling data, a
   quality-comparison pass rather than an equivalence test, or an actual
   training-behavior change rather than a refactor): `DoRAAdapter`,
   `NF4WeightStore`, `CheckpointPlacementPolicy`/`GreedyRatioPlacement`,
   the zero-terminal-SNR-plus-v-prediction pair, and a one-line note on
   `LoRAPlusGroups` itself once its prerequisite lands.
4. **"Naming consistency pass."** Two forward-pointer notes added at the
   points iteration 2 revised, so a reader of `docs/theoretical_pipeline_design.md`
   alone doesn't get a stale impression from a section that was later
   superseded: `NoiseSchedule` (section 1.4) now notes it was narrowed to
   discrete-time explicitly in iteration 2's A.1; `ResourcePolicy`
   (section 2.2) now notes it grew from 3 methods to 7 in iteration 2's
   A.2, with a pointer to where. `DiffusionProcess.__post_init__`'s
   hardcoded `isinstance` check (iteration 2, B.1) is left as-is,
   deliberately -- there's still only one known-incompatible pair, and
   generalizing to a registry for a set of size one is exactly what "don't
   overcomplicate" rules out; noted there already, not changed now.
5. **"`ResourceBudget`'s `vram_budget_mb` unit ambiguity."** Fixed at the
   source: `ResourceBudget`'s own docstring (`docs/theoretical_pipeline_design.md`,
   section 2.2) now states explicitly that it measures against the
   allocator's *reserved* memory, not allocated -- and why (reserved is
   what an actual out-of-memory error is bounded by; allocated alone
   would understate the allocator's own held-but-unused pool). Every
   consumer, including iteration 2's `GreedyRatioPlacement`, now has one
   agreed-on meaning to compare against.

## Found during the fresh read-through, not on the original list

A "final revision" pass should catch real things a checklist written in
advance wouldn't anticipate -- two did:

6. **`RescaledZeroTerminalSNRSchedule`'s terminal `sigma` is exactly
   `inf`, not just "very large."** Checked by hand, not assumed: after
   the Lin et al. rescale, `alphas_cumprod[-1] == 0.0` exactly, so
   `sigma_t[-1]` is a literal IEEE-754 `inf` (division by an exact-zero
   tensor, not an exception). This is *correct* -- it's the whole point
   of "zero terminal SNR" -- and `VPredParameterization.to_x0()` was
   checked to stay well-defined in that limit (`x0 -> -raw` as
   `sigma -> inf`, a clean finite result), which is the actual substance
   behind B.1's stated v-prediction requirement, not just a restatement
   of the paper's claim. Documented directly in
   `docs/theoretical_pipeline_design_iteration2.md`'s B.1 section, right
   after the code that produces it, including the real footgun: anything
   touching raw `sigma_t` *outside* the `Parameterization` abstraction
   would hit that `inf` and needs to account for it.
7. **`Parameterization`'s `x_t` convention was left implicit.** The
   formulas in section 1.4 (`x0 = x_t - sigma*raw`, etc.) are only
   correct for the k-diffusion/`ModelSamplingDiscrete`-style `x_t`
   `NoiseSchedule` already implies (alpha-normalized, `sigma` as a
   noise-to-signal ratio) -- not the raw DDPM `x_t` a dataset loader
   actually produces. That conversion is `ModelInputTransform`'s job,
   defined two paragraphs later, but nothing said so explicitly before
   this pass. Now stated directly in `docs/theoretical_pipeline_design.md`,
   right after the `Parameterization` code block, instead of left for a
   reader to infer from the schedule's own formula.

## What wasn't touched, and why

No new class, no new interface, no new technique anywhere in this
document or its edits -- checked against the "just fixes" definition
given for this pass before each change above was made, not after. The
gap-analysis rewrite (item 2) reorganizes and extends what iteration 1
and 2 already designed; it doesn't design anything new itself. Items 6
and 7 are corrections to precision/completeness of existing claims, not
new design surface.

The comparison against `nodes/` (`docs/theoretical_pipeline_design.md`'s
`## Gap analysis` and `## Prioritized backlog` sections) is now the final
version, current as of this pass -- not provisional, and not superseded
by anything else in this document set.
