# Review notes -- flagged during the docs restructuring pass

Written by a first-time reader of this repository while reorganizing
`docs/` from four large, mixed-topic files
(`PROGRESS.md`, `docs/training_pipeline_design.md`,
`docs/resources_controller_redesign_plan.md`,
`docs/suspicious_findings.md`) into the topic-separated folder
structure under `docs/` today, then extended by a follow-up
verification pass that checked specific claims in the restructured
docs against the actual codebase. Two kinds of entries below: things
that read as **confusing or internally inconsistent** (worth fixing
for clarity, not necessarily bugs), and things that look **potentially
outdated** (worth someone who knows the current real state confirming,
then updating or deleting). Nothing here has been fixed as part of
this pass unless explicitly marked "**Fixed in this pass**" -- this
file is a to-do list, not a changelog of everything that already
happened.

This file should shrink over time, not grow -- once an item below is
checked and resolved, delete it rather than marking it "done" and
leaving it here.

## What the verification pass actually checked, and found

A follow-up pass read every claim in `docs/status/progress.md`'s
"Implemented" section against the real code (file exists, class/method
named as claimed, behavior matches the description) rather than
trusting the prose, then did the same for every entry in
`docs/known-issues/`. Most claims checked out exactly as written --
listed here for completeness, not because the outcome was doubtful:

- Every class/file named in `docs/status/progress.md`'s "Implemented"
  section exists as described, with one exception (below).
- The newer Resources Controller / precision work
  `docs/status/progress.md` is missing (item 1 below) is confirmed real
  in code, not just claimed in commit messages: `ResourcesControllerNode`,
  `LoRATrainingConfigNode`, `ResourceControlHandle`/
  `BudgetedResourceControlHandle`, `Port.choices`/`visible_when`/
  `widget_only`, `Node.diagnostics()`, `Int8BlockStateStore` all exist
  and do what's claimed.
- `docs/known-issues/resolved.md`'s entries were spot-checked against
  the actual fix code (the `_STRATEGIES`-duplication consolidation into
  `nodes/optimizer/strategy_registry.py`, the `ZeroDivisionError`
  tensor-space-clamp fix in `AdafactorAlgorithm.compute_update()`, the
  dataset-loader `ValueError` fix, the missing-node-registration fix) --
  all confirmed present and matching their description.
- `docs/design/resources-controller/07-post-phase-6-bugfixes.md`'s four
  bugfix claims (the `/api` URL prefix, `remeasureAndRedraw()`, the
  `alpha=64.0` default, and the "no leaked `Port.doc` planning-note
  text" claim -- checked with an actual AST-style scan of every
  `Port(...)` call in the two affected files, not a guess) are all
  confirmed accurate in the real code.
- `docs/known-issues/open.md`'s `DeviceResident.footprint_bytes()`
  entry is **confirmed, not just suspected**: every concrete
  `footprint_bytes()` implementation in `nodes/` (14 checked) computes
  a size from tensor shape/dtype alone (`numel() * element_size()`);
  none check `.device` to confirm the tensors are actually where the
  object claims. Still genuinely open.

Three real problems were found and fixed:

1. **`docs/status/progress.md` said `Builder`/`Port` live in
   `nodes/core.py`.** There is no class named `Builder` anywhere in the
   codebase -- the real class is `Node`. The design doc's own
   implementation-status table (`docs/design/08-validation-and-implementation-status.md`)
   already gets this right ("`Builder`/`Port` (1.1) | `nodes/core.py`'s
   `Node`/`Port`"); `docs/status/progress.md` just didn't carry that
   distinction over. **Fixed in this pass**: corrected to `Node`/`Port`
   with a note explaining `Builder` was the design doc's illustrative
   name for the same concept.

2. **`docs/known-issues/deferred.md` claimed DoRA's `magnitude`
   parameter wasn't wired into checkpoint save/load -- confirmed false
   by reading the actual load and save paths.** Load:
   `nodes/model/lora_checkpoint_loader.py`'s `_load_dora_layers()` calls
   `layer.load_dora_weights(A, B, state_dict[magnitude_key])`. Save:
   `ComfyUNetTrainableModel.trained_state_dict()`
   (`nodes/model/lora_injector.py`) calls
   `nodes/model/lora_phases.py`'s `extract_combined_weights()`, which
   includes magnitude for an unsplit DoRA layer -- both wired, matching
   what `docs/status/progress.md`'s own "Model / LoRA" section already
   said. This was true when the entry was written and got resolved by
   later work without the entry being updated. **Fixed in this pass**:
   moved to `docs/known-issues/resolved.md` with an explanation, rather
   than silently deleted, so nobody re-diagnoses it from an old clone
   or an old `docs/status/progress.md` snapshot.

3. **A real editing defect in what's now `docs/known-issues/pending-testing.md`,
   found by git archaeology, predating this docs restructuring
   entirely.** Commit `2e0ca29` (2026-08-18, "adafactor:
   AdafactorAlgorithm.compute_update_batched()...") replaced an entry's
   header line ("- **[2026-07] Persistent ~500MB VRAM growth after
   preview generation.**") with a new, unrelated entry's header, but
   left the original entry's body paragraph in place underneath --
   orphaning it, silently, under the wrong heading. A reader hit a
   paragraph about VRAM growth after preview generation sitting inside
   what looked like the CAME/Adafactor optimizer-speed entry, with no
   way to tell it was actually a separate, once-independent finding.
   **Fixed in this pass**: restored the missing header line at the
   correct location. (Confirmed via `git log --all -S "..." --
   docs/suspicious_findings.md` that this predates the restructuring --
   not something introduced by the file split.)

## Source-code references to the old flat paths -- now fixed

**All 52 source-code comments/docstrings that referenced the old flat
file paths by exact string have been updated to point at the correct
split file.** This was deliberately deferred out of the original
restructuring pass (getting the docs into a good, logical structure
first, fixing references to them from source code afterward, rather
than letting reference-preservation constrain the structure) and
completed in a follow-up pass. What was fixed, so the work doesn't
need rediscovering if something was missed:

| Old path | Split into | Fixed in source (was: referenced by) |
|---|---|---|
| `docs/training_pipeline_design.md` | `docs/design/*.md` (see that folder's `README.md` for the section-number-to-file mapping) | 30 source files |
| `docs/resources_controller_redesign_plan.md` | `docs/design/resources-controller/*.md` | 17 source files |
| `docs/suspicious_findings.md` | `docs/known-issues/*.md` | 4 source files |
| `PROGRESS.md` | `docs/status/progress.md` | 1 source file |

Most of these references cited a specific section or phase number
(e.g. "design 3.1," "Phase 5," "section 11.3") rather than just the
bare filename, so each one was resolved individually against the
section-number-to-file mapping in `docs/design/README.md` and
`docs/design/resources-controller/README.md` rather than mechanically
renamed -- a handful of citations spanned two different target files
in one breath (e.g. "section 3.3/10", "Phase 2/4") and needed
splitting into two separate pointers rather than picking one. All
50 changed `.py` files were re-parsed with `ast.parse()` after editing
to confirm no syntax errors were introduced; all 81 doc-to-doc links
across `docs/` were re-verified to resolve.

**All doc-to-doc references (docs linking to other docs) were already
fixed in the verification pass** that preceded this one -- the initial
restructuring pass updated the top-level navigation docs, but missed
several references *within* the split design docs themselves; those
were found via a full `grep -rn` sweep and fixed at that time.

## Likely outdated -- worth verifying against real current state

1. **`docs/status/progress.md` is significantly behind real work and
   should be resynced, not lightly edited.** Its own trailer claims
   "Last synced against `docs/design/` (formerly
   `docs/training_pipeline_design.md`) at commit `2c1f0ff`
   (2026-08-25)." The repository has ~20 more commits after that
   (through `2991618`, 2026-09-10), including work
   `docs/status/progress.md` doesn't mention *at all* (confirmed real
   in code, see above):
   - The entire Resources Controller / precision redesign, Phases 3
     through 6 (`docs/design/resources-controller/`) --
     `ResourcesControllerNode`, `LoRATrainingConfigNode`,
     `LoRATrainingResources`/`LoRATrainingSkeleton`, frozen-LoRA
     merging, continue-training support, LoRA-file inspection.
   - New generic editor mechanics: `Port.choices`, `Port.visible_when`,
     `Port.widget_only`, `Node.diagnostics()`, `Node.NODE_KIND`/
     `NodePreset`/`Node.DISPLAY_NAME`.
   - `ResourceControlHandle`/`BudgetedResourceControlHandle` -- a live,
     per-step VRAM budget enforcer, genuinely different from (and newer
     than) the `OffloadOrchestrator` that `docs/status/progress.md`
     does mention.
   - `state_precision` -- block-wise 8-bit optimizer-state quantization
     (`OptimizerStateStore`/`Int8BlockStateStore`).

   A concrete, checkable symptom of the drift:
   `docs/status/progress.md` says "51 smoke tests under
   `nodes/smoke_tests/`... plus 5 more under `manager/`/`server/`."
   The real counts today are 56 (`nodes/`), 5 (`server/`), 1
   (`manager/`) -- 62 total, not 56.
   `docs/design/resources-controller/08-consolidation.md`'s own count
   ("the full existing `nodes/smoke_tests/` suite (56 files...)")
   matches reality exactly, which is itself evidence that doc has been
   kept current while `docs/status/progress.md` hasn't.

   **Recommendation:** don't patch `docs/status/progress.md`
   piecemeal -- do a full pass reading every commit since `2c1f0ff` (or
   since whatever commit a future resync starts from) the same way the
   original file was built, and update its own trailer to the new sync
   point. The verification pass corrected one specific inaccuracy found
   along the way (the `Builder`/`Node` naming, item 1 above) but did
   not attempt the full resync -- that's real, separate work.

2. **`docs/known-issues/`'s newest dated entry is 2026-08-21, before
   the Resources Controller Phases 4-6 landed (2026-08-28 through
   2026-09-05) and before the post-Phase-6 bug-fix session
   (2026-09-06).** `docs/design/resources-controller/07-post-phase-6-bugfixes.md`
   lists four real bugs found by using the editor (a missing `/api`
   URL prefix, wires not redrawing after a node resize, a too-weak
   default LoRA `alpha`, leaked planning-note text in tooltips) --
   structurally exactly the kind of thing `docs/known-issues/`
   otherwise tracks, but none of them appear there. Worth checking
   with whoever did that work whether this was a deliberate scoping
   choice (bugs fixed same-session, in the same doc that found them,
   don't also get a `docs/known-issues/` entry) or just something that
   fell through -- if the former, it'd be worth stating that scoping
   rule explicitly in `docs/known-issues/README.md`, since right now a
   reader has no way to tell the difference between "not logged
   because it didn't need to be" and "not logged because it was
   missed."

3. **Several entries in `docs/known-issues/pending-testing.md` and the
   "not yet confirmed on real hardware" items in `resolved.md` are
   still waiting on real-hardware confirmation that may have happened
   since without the doc being updated.** Not checkable from the repo
   alone -- these need whoever has the actual B580 hardware. Specific
   items: the LoRA timestep gate and VRAM-ratchet fix in
   `pending-testing.md`; the CAME/Adafactor `shape_grouped` speedup and
   non-square dataset re-ingestion cap in `resolved.md`. All dated
   2026-07/2026-08; it's been roughly a month of further work since.
   (The `footprint_bytes()` and VRAM-pressure-hang entries in `open.md`
   that used to be listed alongside these are now resolved as
   verification items -- see "What the verification pass actually
   checked" above: `footprint_bytes()` is confirmed still genuinely
   open, and the VRAM-pressure hang is confirmed still out of scope for
   `nodes/`, both by reading code rather than by assumption.)

4. **`convert-cfg.toml` (repo root) contains what looks like one
   specific person's real setup, not an anonymized template**: a
   literal filesystem path (`comfy_dir = "/home/okolenmi/comfy/ComfyUI/"`),
   a dataset named `"test"`, and what looks like a real trigger word
   (`"ttw001"`) in the preview prompts. If this is meant as a
   copy-and-edit example for new users, it'd be worth genericizing the
   path and calling that out in a comment; if it's someone's actual
   working config that ended up committed, it's worth confirming
   whether it should be gitignored instead. Not changed here since
   it's a judgment call for whoever owns that file, not a docs
   question.

5. **This project's Intel Arc B580 / XPU focus is stated in the docs
   (root `README.md`, `docs/setup.md`) as an inference, not a
   confirmed fact from an authoritative source.** It's a reasonable
   inference (the repo's own name, `device = "xpu"` throughout example
   configs, and a `docs/known-issues/` entry that references "the same
   B580 hardware" when comparing against another project's discussion)
   -- but nothing in the repo states this as a design constraint
   directly. Worth a maintainer confirming the wording in those two
   docs is accurate, and correcting it if there's more nuance (e.g.
   whether other hardware is also actively supported/tested).

## Confusing / internally inconsistent -- worth fixing for clarity

6. **Dangling references to two deleted design docs,
   `docs/nodes_package_design.md` and
   `docs/optimizer_execution_redesign_plan.md`.**
   `docs/known-issues/README.md` (formerly
   `docs/suspicious_findings.md`'s header note) says both files were
   deleted and that dangling pointers to them were "cleaned up" -- but
   that cleanup only touched that one doc. **Fixed in a follow-up
   pass**: the 9 remaining references across `server/routes_nodegraph.py`,
   `server/main.py`, `server/nodegraph_introspect.py`,
   `nodes/model/lora_phases.py`, `nodes/model/lora_saver.py`,
   `nodes/components/README.md`, and `manager/builder.py` were each
   checked individually. Where the cited content has a clear current
   home (e.g. the "fused optimizer family" citation ->
   `docs/design/10-node-surface-and-precision-control.md` section 11.2;
   the VRAM-findings citation in `manager/builder.py` -> confirmed by
   matching the exact "+560MB in one step" figure to
   `docs/known-issues/pending-testing.md`'s ratchet entry), the
   reference now points there. Where the exact original content
   doesn't survive anywhere findable (a few specific quoted phrases --
   "strictly better, since there's now an actual contract to read", "no
   longer needs torch importable at all", the "TrainerNode
   scope-reduction list"), the comment was reworded to state the
   underlying fact in its own words instead of citing a source that no
   longer exists -- and in each such case, the underlying factual claim
   was re-verified against current code first, not just carried over
   from the old (possibly also-stale) comment.

7. **`docs/design/resources-controller/01-context-and-ground-truth.md`'s
   "Open design question blocking Phase 4" section contradicted the
   rest of the same document** (back when it was still one file,
   `docs/resources_controller_redesign_plan.md`). Near the top, that
   section said the composition-vs-inheritance mechanism was "Not yet
   decided which one this project uses" -- but the Phase 4 section
   (now `docs/design/resources-controller/04-phase-4-resource-preset-abstraction.md`)
   and the file's own top-of-file status banner said this was already
   decided in favor of multiple inheritance, concrete-mixin-first. A
   reader going top-to-bottom hit the "still open" framing before ever
   reaching the resolution, with nothing in that section itself
   pointing forward to where it got resolved. **Fixed in this pass**:
   added a one-line forward-pointer at the top of that section noting
   it was resolved in Phase 4, without deleting the original reasoning
   (it's still useful context for *why* inheritance was chosen).

8. **The "Last synced against `docs/training_pipeline_design.md` at
   commit `2c1f0ff` (2026-08-25)" trailer used to be identical,
   word-for-word, at the bottom of both `PROGRESS.md` and
   `docs/resources_controller_redesign_plan.md`** (now
   `docs/status/progress.md` and
   `docs/design/resources-controller/08-consolidation.md`
   respectively; both trailers were updated across the two docs passes
   to point at `docs/design/` instead of the old single filename, but
   the underlying ambiguity below is still unresolved). This was
   confusing on its face even before the restructuring: the
   resources-controller file's own content (Phases 5, 6, and the
   post-Phase-6 bug fixes) was dated well after 2026-08-25, so either
   "last synced" means something narrower than "last edited" (e.g.
   "last time this file's claims were cross-checked against the main
   design docs specifically, independent of this file's own unrelated
   edits") or the trailer itself was stale. Worth a maintainer
   clarifying what this trailer is actually supposed to mean, now that
   it lives in two places, before it gets copied into a third.

9. **No single command runs this project's entire test suite.**
   `nodes/smoke_tests/run_all.py` is a convenience runner, but it only
   covers `nodes/smoke_tests/` -- `server/smoke_tests/` (5 files) and
   `manager/smoke_tests/` (1 file) each need to be run individually, as
   noted in `docs/setup.md`. Small enough that it's probably not worth
   much design thought, but worth a one-line top-level runner if this
   becomes annoying enough to notice again.

10. **Cross-document section-number references are structurally
    fragile -- a real, ongoing risk, not a one-time cleanup item.**
    Both docs and source-code comments reference `docs/design/`'s
    files by section number (e.g. "design 3.1," "section 11.3") --
    those numbers stayed meaningful through the split and every
    reference (doc-to-doc and source-to-doc) is now correct as of this
    pass, but nothing *automates* that correctness. If a section ever
    gets renumbered, inserted, or moved to a different file in the
    future, every citation of its old number -- across `docs/design/`
    itself, `docs/review_notes.md`'s own examples, and dozens of source
    files -- silently goes stale again with no mechanism to catch it.
    Not something this restructuring (or its follow-up) attempts to
    solve structurally (e.g. with stable per-section anchor IDs instead
    of numbers) -- flagged as a standing design trade-off, not a task
    with an end state.

11. **`docs/known-issues/` undersells its own reliability.** Its
    header describes it as "an informal, unaudited collection, not a
    spec," and `docs/status/progress.md` echoes that ("an informal
    collection, not authoritative"). In practice, most entries are
    dated, carefully traced to a real root cause (often "confirmed by
    reading X directly, not assumed"), and organized into
    `open.md`/`resolved.md`/`deferred.md`/`pending-testing.md` with
    real specificity -- closer to a lightweight issue tracker than an
    "informal list" in the way that phrase usually implies (hearsay,
    unconfirmed hunches). The verification pass's own experience
    supports this: every substantive, code-level claim checked (not
    just the ones already known to be current) turned out accurate
    except the one DoRA entry now fixed above. Not miscalibrated enough
    to be worth rewriting the framing, but worth knowing this
    collection is more trustworthy than its own disclaimer suggests.
