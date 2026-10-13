# 90. Build plan: a clean parallel tree instead of a migration

Status: **proposal, replaces the earlier `90-migration.md`.**

## 0. Owner decision (2026-10-13, 18:21 Kyiv) - DECIDED direction

* Migration is **not important**: the project is in early development and the owner
  is the only user. Losing datasets, saved graphs, templates or node classes is
  acceptable.
* Build a **parallel nodes folder** and write files one by one, using the knowledge
  of the existing implementations. This also removes the excessive comments.
  Outside `nodes/` the choice is mine.
* Testing: much will be rewritten, so detailed tests have little sense now; **one
  big test of everything** after the improved implementation exists.
* Method: for an AI, reading old files and rewriting them against the new plan is
  the shortest path. The owner suggested recording this in the rework rules (done,
  section 3 and `00-index.md`).

Consequences, applied to every plan file: **no shims, no aliases, no deprecation
windows, no old-port compatibility, no template or saved-graph conversion, no
parity gates between old and new.** Anything in plans 10 to 80 that says otherwise
is superseded by this file (each plan carries a note pointing here).

## 1. What disappears from the earlier plans

| Dropped | Was in |
|---|---|
| Class alias tables, port rename tables, graph format migration | old 90, plans 20, 41, 42, 60 |
| Re-export shims for moved files, deprecated node classes | old 90, 41, 42, 70 |
| `VRAMBudgetControllerNode` prompt-then-retire path | plan 20, section 8 |
| `[graph_defaults]` is kept, but as a plain feature, not a compatibility device | plan 20 |
| Template to graph+view conversion, `/templates` routes | plan 60, sections 10, 11 |
| "node field wins over setting for one release" | plan 20, section 6 |
| Bit-identical parity gates per phase | old 90, plans 40, 50, 70 |
| Dataset format v2 compatibility; stage A/B split of the latent cache | plan 50: build the target design directly |
| One-release windows, `deprecated_*` validator codes | everywhere |

Because nothing has to stay compatible, **port and node names can be chosen once,
cleanly**. They still become fixed when the swap (section 7) happens, so naming is
settled in `00-index.md` (glossary) before building.

## 2. The parallel tree

```
repo/
  nodes/ manager/ ...        old code: stays untouched and runnable until the swap;
                             read-only reference afterwards (moved to archive/)
  next/                      NEW tree; inside it only relative imports
    graph/                   node base, ports, groups, events, settings, validator,
                             runtime, graph JSON, catalog serializer, headless runner
    families/                api / general / sdxl / anima (plan 40)
    data/                    library, latent cache, prompt table, batching, ingest
    engine/                  step + runtime (plan 70)
    nodes/                   thin node classes by domain
    tests/                   the one big test + a few pure self-checks
```

* **Relative imports only** inside `next/`, so the swap is `git mv next/* .` with no
  rewriting. (`next` is a working name; the swap removes it.)
* The old app keeps working while this is built (decision B1, open question 1).
  `next/` is exercised by its own **headless runner** (`python -m next.graph.run
  graph.json`: validate, run, print events) and, from phase 6, through new routes.
* The import-boundary rules of plan 80, section 2.1, apply inside `next/` from the
  first file; the check script is the first thing written.

### Other regions (my choice, as the owner allowed)

| Region | Strategy | Why |
|---|---|---|
| `nodes/` | **parallel rewrite** in `next/` | owner's request; 41% comments; structure issues from the audits |
| `manager/` (ingest, loader, 2.9k lines) | rewritten into `next/data/` | the dataset design changes (per-VAE cache, prompt table, events) so it is not a move |
| `backend/` (26k lines, layered, tested) | **keep and extend**; add a thin v2 graph adapter over `next/graph` | its layering is the best-structured part of the repo; the graph validator and runtime logic move into `next/graph` because they are node-contract code |
| `frontend/js` (10k lines) | **editor modules rewritten in a parallel folder** (`frontend/js/graph2/`, page `graph2.html`); the rest untouched | plan 30 changes the render model (incremental), so it is not a patch; other pages are unaffected |
| `docs/` | rewritten at the swap from the plans; old design docs move to `archive/` | |

## 3. Working method (the "AI-native" rules)

The owner's observation is right: model habits come from human-written projects
(small diffs, careful migration, many unit tests). Those are costs for a human
team; for this job the efficient moves are different.

1. **The unit of work is one new file**, written after reading the old file(s) it
   replaces *and* the plan section that governs it. Never port line by line: read,
   understand what the code is *for*, write the new file against the plan.
2. **Triage before touching an area** (not all 120 files upfront, which would cost
   about 300k tokens of reading): for each old file in the area the ledger records
   `keep` (moved, comments cleaned), `rewrite`, or `drop`. Algorithms that are
   fine (optimizers, quantization, attention kernels) are `keep`.
3. **Harvest lessons first, per area.** Old comments contain measured hardware
   facts, OOM causes and incident history. A rewrite without comments would lose
   them. Before rewriting an area, write `lessons/<area>.md`: each fact with its
   source file, what it implies for the new design, and whether a test or a setting
   must carry it. These do not go into code comments (plan 80, section 12); the
   new code cites `docs/known-issues` or `lessons/` entries.
4. **Contracts before bodies.** `graph/` types and the catalog contract
   (`contracts/*.md`) are written and frozen before families, data and engine code,
   so every later file has something to be consistent with. A contract change goes
   to the contract file first.
5. **Big batches, cheap verification.** Write many files per pass; verify with the
   cheap whole-tree checks (`py_compile`, import check, a declaration lint that
   loads every node class and validates its ports, groups, events, settings, and
   the frontend's `node --test` for pure logic). These catch most of what an AI
   rewrite gets wrong (names, missing imports, signature drift) for almost no
   tokens.
6. **Renames are cheap now, expensive after the swap.** Use the freedom: fix names
   now, then freeze them.
7. **Keep a ledger** (`PORTING.md`): old file, planned destination, triage, state.
   It is the resume point after context compaction; update it with every file.
8. **One sub-agent at a time**, only for self-contained files with a precise brief
   (the plan section plus the old file). The owner's usage limit makes parallel
   agents fail all at once.
9. **Do not preserve old structure for diff readability.** Git history keeps the old
   code; the new code is read as new code.
10. **Be honest about what cannot be run here** (section 6) and mark such code
    `unverified` in the ledger until the owner runs it.

## 4. Phases

Each phase ends with something checkable. "Check" lists what runs **here**.

| # | Phase | Delivers | Check here |
|---|---|---|---|
| 0 | Contracts and rules | `contracts/` (catalog JSON, graph JSON, event wire format, settings, resolve API), glossary, import-check script, comment-lint, empty `next/` skeleton | scripts run on the skeleton |
| 1 | `next/graph` | node base (`Node`, `Port`, `Group`, ports, modules), settings registry, events (`EventType`, `EventHub`), validator (data + event edges, dynamic ports, dangling wires), runtime (topological, completion, cancel, reactive node retention), graph JSON with `uid`, catalog serializer, headless runner | self-check script with toy nodes: validation errors, event barrier, dangling wires, settings resolution (pure Python, runs here) |
| 2 | Families API and detection | `families/api`, `general/safetensors_io` (header reader with size cap), SDXL fingerprint, registry, `Load Model` node with resolve (dynamic outputs), scopes and `ProjectLayout` scopes | synthetic safetensors headers for each file kind, including ambiguous and unknown; no weights needed |
| 3 | General and SDXL (torch) | `general/{clip,vae,adapters,schedules}`, `sdxl/{denoiser,conditioning,latents,objective,adapters,memory}`, conditioning cache, `Conditioner`, adapter plan and injection | with torch on CPU: tiny random SDXL-shaped model, forward, LoRA inject/save/load round trip |
| 4 | Data | `next/data`: library, per-VAE latent cache, prompt table and stages, bucketer, batcher with epoch barrier and `PromptsChanged`, prefetch, `Noiser`, ingest; dataset nodes | synthetic images and a tiny random VAE: epochs, event order, tag swap |
| 5 | Engine and training nodes | `engine/step`, `engine/runtime` (residency, memory admission, block swap, capture, profiler), `ObjectiveNode`, optimizers, savers, probes, the single trainer | a 5-step run of a tiny model on CPU through the headless runner |
| 6 | Server and editor (parallel track from phase 1) | backend v2 graph adapter (catalog, validate, resolve, run, library), settings dialog, `graph2` editor: incremental render, groups, collapse with pins, node states, multi ports, dynamic ports, event wires | `node --test` for pure editor logic; backend tests for the adapter; visual smoke script |
| 7 | Views and starters | `views` in the graph, monitor/page boards, starters (SDXL LoRA, Anima LoRA), per-run metrics | starters validate and run headless |
| 8 | Anima family | kits, fingerprints, flow-matching objective, block swap, key scheme | tiny random Anima-shaped weights on CPU; real header and run are the owner's |
| 9 | Acceptance and swap | the big test (section 6); old trees to `archive/`; `next/*` promoted; docs rewritten | the big test; owner's B580 run |

Dependencies: 1 -> 2, 3, 4, 5, 6. 2 -> 3. 3 -> 4 (VAE) and 5. 4 -> 5. 6 starts after
the catalog contract of phase 0/1 is frozen and proceeds beside 2 to 5. 7 needs 5
and 6. 8 needs 5. 9 needs everything it tests.

## 5. Contracts that unlock parallel work

`contracts/` files, each a short normative document written in phase 0 and frozen
in phase 1; the plans give the substance, the contracts give the exact JSON:

* `catalog.md`: node info (ports, groups, `ui`, `multi`, `dynamic`, events, modules,
  settings read), settings specs, family list.
* `graph-json.md`: nodes with `uid`, display id, params, `layout`; data edges;
  `event_edges`; `settings`; `views`.
* `events.md`: event type table, hub rules, SSE mirroring.
* `resolve.md`: request, statuses (`ready`, `processing`, `needs_choice`, `error`),
  computed choices, info line.
* `runs.md`: run lifecycle, node states, progress frames.

## 6. Verification: the one big test, and what this sandbox can show

* **Big test** (phase 9, `next/tests/acceptance`): load each **starter** graph,
  validate, run end to end on the synthetic fixture (a few images, a tiny random
  model and VAE, 5 to 10 steps), assert it finishes, writes a loadable adapter, and
  that events fired in order (epoch barrier, saver on `StepFinished`). Then the same
  starters on the owner's B580 with a real model.
* Cheap structural checks run throughout (section 3, rule 5); they are not tests of
  behaviour, only of consistency.
* **Math spot checks (my recommendation, optional).** While the old tree still
  exists, a handful of numeric comparisons (noise schedule, loss target for eps and
  v-pred, LoRA forward, VAE scaling) between old and new cost very little and catch
  the quiet failure of a rewrite: the new code runs and trains wrongly. I would keep
  these few and drop everything else; the owner may say no.
* **The sandbox has no torch preinstalled** and Python 3.13 (the project asks for
  3.14). A CPU torch can be installed from PyPI here (an install is in progress at
  the time of writing). **Nothing here can run on an Intel XPU**, so speed, memory,
  residency and block swap are checked only by the owner. Everything that touches
  the device is marked `unverified` in the ledger.

## 7. Swap

1. Big test green here; the owner runs both starters on the B580.
2. `git mv nodes archive/nodes_v1`, same for `manager/` and the old graph routes;
   `next/*` to the root; `graph.html` replaced by `graph2.html`.
3. Backend v1 graph routes removed; v2 becomes `/api/v1`.
4. Docs regenerated from the plans; the plans move to `docs/design/`.

Rollback: git. (There are no users to protect, which is the point of the decision.)

## 8. Size and session budget (honest estimate)

* Old code: 27.6k lines, about 12.7k of code. New code is expected around 15 to 20k
  lines for the Python tree (Anima and the new features add; comments leave) plus a
  rewritten editor (about 6 to 8k lines of JS).
* At roughly 10 to 14 tokens per line, writing is on the order of **250 to 400k
  output tokens**, and reading what is ported another **200 to 300k**. The owner's
  stated working budget is near 100k tokens per 5 hours, so this is **several
  working windows (my estimate: 6 to 10)**. That is why every phase leaves the tree
  consistent, the ledger and contracts are the resume point, and big context-heavy
  reading (triage) happens per area.
* Order of risk: phases 0 to 2 are pure Python and fully checkable here and should
  go first; phase 3 and 5 are where hardware-only errors can hide.

## 9. Risks

| # | Risk | Mitigation |
|---|---|---|
| B1 | Rewriting loses hardware knowledge that lived in comments | lessons harvest per area before rewriting (method rule 3) |
| B2 | Quiet numerical regressions in a rewrite | the few math spot checks; the owner's B580 acceptance |
| B3 | Code that cannot run here is wrong | `unverified` marks, compile and import checks, tiny CPU models, owner's run |
| B4 | The parallel tree rots or diverges from plans | plans updated in the same pass as code; ledger |
| B5 | The old app changes while I rewrite (the owner's agent keeps fixing it) | the ledger records the old `main` hash a file was read at; before the swap, diff old files since that hash and re-apply real fixes |
| B6 | Scope: Anima plus everything else | Anima is phase 8, after SDXL works end to end |
| B7 | Editor rewrite stalls the rest | contracts decouple it; it can ship last |
| B8 | Token budget exhausted mid-phase | small phases, ledger, contracts |

## 10. Decisions

| # | Decision | Recommendation |
|---|---|---|
| B1 | Old app during the build | Keep it working until the swap (new tree has its own runner and v2 routes) |
| B2 | New tree location | `next/` with relative imports; promoted at the swap |
| B3 | Backend | Extend in place; thin v2 adapter; graph core in `next/graph` |
| B4 | Editor | Parallel `graph2` folder and page |
| B5 | Datasets | New format directly; old datasets are re-ingested or discarded |
| B6 | Testing | One acceptance test, cheap structural checks, optional math spot checks |
| B7 | Order | contracts, graph core, detection, SDXL, data, engine; editor beside them; Anima after SDXL works |
| B8 | Harvest | Lessons file per area before porting it |

## 11. Open questions for the owner

1. **Must the old app keep working while I build?** My default is yes (you can keep
   training with it). If not, the build is simpler: I would edit `backend/` in place
   and skip the v2 adapter.
2. Do you want the **math spot checks** (a few old-versus-new numeric comparisons) or
   strictly only the one big test?
3. May I keep the **optimizers and quantization code** as `keep` (moved with comments
   cleaned) unless triage finds structural problems? They were not flagged in the
   audits, and rewriting numerics costs the most for the least gain.
4. Is a **new dataset format** (images + captions, per-VAE latent caches; old
   datasets re-ingested) acceptable? It follows from plan 50 and from dropping
   compatibility.
5. Will you **supply a real Anima header** (and later a short real run) for phase 8?
6. The estimate in section 8 is long. Do you want phases 0 to 2 first (about one
   window, all checkable here) and a review before the heavy phases?
