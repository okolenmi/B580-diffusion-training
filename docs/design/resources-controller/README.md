# Resources Controller & precision redesign -- step-by-step plan

*New here? Start at the root [`README.md`](../../../README.md). This is
the most recently active piece of work in the repo -- the status banner
right below is the most current single summary of it.*

This folder used to be one file,
`docs/resources_controller_redesign_plan.md` (~1,100 lines). It's been
split into one file per phase so each phase can be found, linked, and
updated on its own as work continues -- the content itself is
unchanged from the original, only reorganized (with one small in-place
correction, noted in
[`docs/review_notes.md`](../../review_notes.md) item 7: an "open
design question" section that had been left contradicting its own
later resolution).

**Status: Phase 1 + Phase 2 done, Phase 3's suggestion-menu question
resolved (node_kind/presets metadata), Phase 4's core object-
construction mechanics done, a consolidation pass done connecting this
to the main design docs' remaining open items --
including `Port.choices` (section 11.4, now in
[`../10-node-surface-and-precision-control.md`](../10-node-surface-and-precision-control.md)),
now built and landed as part of that consolidation, with real frontend
code (`server/static/nodegraph.js`) beyond Phases 1-2's backend-only
scope. Phase 5 done: the Resources Controller node produces a
verified, NOT-yet-LoRA-injected resource pack
(`LoRATrainingResources`) -- see that phase's own status for how its
scope got corrected from an earlier, wider version that did injection
itself, and for what's still browser-unverified. Phase 5 also landed
real, generic editor mechanics (`Port.visible_when`, `Port.widget_only`,
a live `Node.diagnostics()` endpoint) any future node can use, not
just this one. Phase 6's own `LoRATrainingConfigNode` done too -- takes
that resource pack and actually injects LoRA (rank/alpha/frozen-weight-
storage), including locking rank when continuing training from an
existing LoRA. Phase 9 now closes the loop with its own trainer
(`ManagedLoRATrainerNode`, taking `trainer` directly rather than
adapting into the main route's ports -- a first attempt at the latter
was tried and reverted, see that phase's own status), built around
deterministic per-phase residency instead of the main route's reactive-
only offloading, plus a `strict` flag on
the existing VRAM budget enforcer make the route's actual point --
staying clear of a VRAM ceiling, with a way to make that a hard
guarantee instead of best-effort -- a real, wired capability, not just
plumbing. Not run on real hardware yet.** This tracks a real,
multi-session redesign, not a single patch -- update it as phases land
or as open questions get resolved, the same way
[`docs/status/progress.md`](../../status/progress.md) tracks the rest
of this project (though that file has drifted -- see
`docs/review_notes.md` item 1 -- this one hasn't, so far).

**Priority rule, made explicit rather than left implicit:** where this
plan and the main design docs (`docs/design/`) conflict, this plan
wins -- it reflects the more recently decided direction. Section 11's
older items get updated or superseded as needed (see "Consolidation"
below for the concrete cases found so far), not treated as equally
authoritative history that both documents have to be reconciled around
forever. `docs/design/` stays the record of what actually shipped and
why for everything outside this redesign's scope; it isn't frozen,
just not the tie-breaker for anything this plan touches.

## Reading order

| File | What's in it |
|---|---|
| [`01-context-and-ground-truth.md`](01-context-and-ground-truth.md) | Why this redesign exists, the real ground-truth facts it's built against (not assumed), and the composition-mechanism question that blocked Phase 4 (now resolved). |
| [`02-phase-1-and-2.md`](02-phase-1-and-2.md) | Lazy resource references + header-only inspection; the server query endpoint. Both done. |
| [`03-phase-3-interactive-node-support.md`](03-phase-3-interactive-node-support.md) | Editor + `core.py` + introspection support for interactive nodes. |
| [`04-phase-4-resource-preset-abstraction.md`](04-phase-4-resource-preset-abstraction.md) | The `ResourcePreset` abstraction -- core construction mechanics done. |
| [`05-phase-5-resources-controller-node.md`](05-phase-5-resources-controller-node.md) | The Resources Controller node itself -- done, scope-corrected along the way. |
| [`06-phase-6-lora-training-config.md`](06-phase-6-lora-training-config.md) | `LoRATrainingConfigNode` -- done. Downstream `TrainerNode` integration itself moved to 09 (this file's original sketch for it turned out wrong). |
| [`07-post-phase-6-bugfixes.md`](07-post-phase-6-bugfixes.md) | Four real bugs found by actually using the editor after Phase 6 landed. |
| [`08-consolidation.md`](08-consolidation.md) | Resolving low-synergy items against the main design docs instead of leaving them to drift. |
| [`09-trainer-integration-and-vram-safety.md`](09-trainer-integration-and-vram-safety.md) | `ManagedLoRATrainerNode` -- the route's own independent step loop, deterministic residency instead of the main route's reactive-only offloading -- plus a `strict` VRAM-budget mode. Done, not run on real hardware. |
