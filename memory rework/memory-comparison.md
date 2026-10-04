# Memory rework: my plan versus what is pushed

My plan is `memory-plan-mine.md`, written against `f4e5a91` and sealed
(sha256 prefix `21ae00911bde46ab`, 2026-10-04 15:30 UTC) before I fetched.
The pushed work is three commits after it: `e11f459` (`sacrificable` state and
a negative measurement), `941d300` (measuring what a reservation would have
to sum), `5f767e5` (`DeviceReservations` + `PeakRecord`, "observed mode as a
persistent record"). Evidence tags: **[R]** I ran it, **[C]** I read it.

## 1. Where the two agree
* The graph is the configurable object; admission is at graph start by
  reservation; no preemption; a graph that does not fit does not start.
* An unknown demand must never be read as zero (their three-valued rule; my
  "unknown is not zero" in 2.1). Their `admit()` that *requires* a number
  after the `or 0.0` hole is the right fix.
* Offloading is demand-driven only (4.7x slowdown lesson).
* The record that makes step 0 safe must exist *before* the run starts.
* Nothing is wired into a trainer until measured.

## 2. What the pushed work does better than my plan
1. **Measurement.** Residents are exact (5,611 MB = model 4,897 + optimizer
   714, constant across batch) and workspace is the hard half (1,619 /
   2,055 / 3,343 MB at batch 1/2/4, not linear). My plan keyed learned peaks
   by "the same graph" and used 1.05 x last peak; that is wrong the moment
   batch or resolution changes. Their **fingerprint of peak-relevant
   configuration** (model, batch, latent h/w, rank, checkpointing,
   optimizer) is the right key. Take it.
2. **The negative result on `sacrificable`.** The training peak contains the
   model, so no budget is rescued by evicting it from inside the trainer. This
   saves me from building eviction inside the trainer; my Level 2 must target
   *other nodes' holdings*, not the trainer's own.
3. **Monotonic semantics.** A peak never ratchets down; a stated claim is
   never lowered by a measurement below it; a refused run holds nothing;
   re-claim replaces. All correct; all should survive the rewrite.
4. **Persistent observation read at admission** (the user's correction: the
   risk is mid-run, not step 0) is the right insight and removes my
   within-run learning altogether.
5. **The seam is named correctly:** nodes cannot state their demand today
   (`Port` has no cost field; `Node` has nothing pre-build).

## 3. What my plan covers that the pushed work does not
1. **The process boundary.** Each graph is a child process; the only party
   that sees all claims is the server. The pushed registry is an in-process,
   `threading.Lock`-protected dict in `nodes/memory/`, and the memory section
   of the docs never mentions processes. **[R]** Two child processes each
   claiming 7,000 MB on a 12,216 MB card: each sees 0 MB held by the other, so
   both are admitted (script `r22`).
2. **Every GPU user, not only trainers.** **[C]** `StartGraphExecution` checks
   graphs only; `StartDatasetTask` checks tasks **of the same dataset** only.
   A graph and a dataset task (or two datasets' tasks) can share the card
   today. A ledger must cover graphs, dataset tasks and the device probe.
3. **Device-visible accounting.** **[C]** `DeviceReservations.total_mb` is the
   whole card, so the pushed demo admits 7,816 + 4,400 = 12,216 MB, i.e. 100%
   of it. The project's own measurement says the device reading was 10,140 MB
   for 8,592 MB reserved: about 950 MB belongs to the desktop and about
   600 MB is overhead the allocator does not report. A ledger in allocator MB
   over-admits by roughly a gigabyte plus one overhead per process. Needs
   `foreign_reserve_mb` and `process_overhead_mb`.
4. **A physical check for foreign users** (ComfyUI, the desktop):
   `mem_get_info().free` in the child before it loads anything.
5. **A backstop:** `set_per_process_memory_fraction` so a node that exceeds
   its grant fails inside its own process, not on the card. (Exists in torch
   2.14; must be re-checked on the 2.12 xpu build.)
6. **Level 2, nodes asking the graph to release.** The user's sentence
   ("nodes may ask to release some for their needs") is an *intra-graph*
   arbitration; the pushed work only has the `register`/`ensure_loaded` seam
   inside one trainer. My lease API is a candidate for the case their
   measurement does not cover: a later node needing room while an earlier
   node's output (for example a trained model) is still held by the graph.
7. **A refusal that explains itself** (every holder, free, what would fit) and
   fault-injection tests for reserve -> spawn -> finish.

## 4. Defects in the pushed code (all small, all fixable now)
1. **[R] `DeviceReservations` does not coordinate processes.** See 3.1.
2. **[R] `PeakRecord` loses updates across processes and ends lower than a
   recorded peak.** Its docstring says "safe across processes by writing
   atomically ... never a wrong [admission]". Atomic *replace* prevents a torn
   file; it does not make read-modify-write atomic. Six processes recording
   distinct peaks at the same instant: **69 of 150 trials** ended with a stored
   peak below the highest one recorded. That breaks the property the commit
   message calls load-bearing ("a peak never ratchets down") and produces
   exactly the failure it exists to prevent: admit on a number a previous run
   already exceeded. The commit also says the peak is "written after each
   step", so concurrent runs write the *whole file* constantly, and a lost
   update also drops other fingerprints' entries. Fix: one writer (the
   server), `INSERT ... ON CONFLICT DO UPDATE SET peak = MAX(peak, excluded.peak)`
   in SQLite, or an `fcntl` lock around the read-modify-write.
3. **[C] `admit()` has a permissive branch that contradicts its own rule.**
   If `total_mb is None` the claim is recorded without any check. A caller
   that forgets `total_mb=` (the constructor default) gets "everything fits".
   This is the shape the author names in the same commit: an `Optional` on a
   safety check is a silent "no check performed". Make the total required, or
   make an unknown total a refusal.
4. **[C] "Refuse the first ever run" is a heavy price for safety.** One
   refusal per new configuration, always, unless the user types a number. A
   cheaper rule that keeps the invariant: an unmeasured run is admitted only
   if **nothing else holds the card**, and then it claims **all of the free
   capacity** (not zero), runs exclusively, and records its peak. Concurrent
   and unknown still refuses.
5. **[C] The reservation is only as good as the fingerprint.** The fields come
   from the trainer; the server needs them *before* spawning, from the graph
   definition and the dataset (latent h/w). Nothing yet derives them
   statically. That is the real first slice, ahead of any estimator.

## 5. Synthesis (what the directions file asks the AI to build)
One architecture, taking the best of each:

* **Server-side ledger** (mine) holding **device-visible** claims for
  graphs, dataset tasks and the probe; derived from DB rows; atomic.
* **Demand from three sources, one mechanism** (theirs): stated, observed
  (persistent record keyed by their fingerprint, written by the *server* from
  the child's reported peak with MAX semantics), or unknown (exclusive
  exploratory run, never a zero claim).
* **Graph settings object** in the graph format (mine) carrying min / max /
  auto and the policy, with the learned last-peak shown (theirs).
* **Child-side `GraphMemory`** (mine) enforcing the grant, with their
  three registration states kept and mapped (never / offloadable /
  sacrificable), `request/lease` for inter-node asks, built only when a
  concrete consumer exists, and measured before each node is migrated.
* Their rules kept verbatim: monotonic peaks, stated never lowered, refused
  run holds nothing, corrupt record reads as unknown, residents exact from
  shapes, workspace measured and fitted with a loud failure outside the fitted
  range.

## 6. Verdict
The pushed work is the better *empirical* foundation: it knows what the
numbers are and what cannot pay. My plan is the better *structural* answer to
where the claims live and who can see them. Neither alone is safe: the pushed
registry would admit two graphs onto one card, and my plan alone would have
keyed its learning wrongly. The directions file merges them.

## 7. Limits of this comparison
No XPU, no ComfyUI, one CPU core. `r22` demonstrates the process and
lost-update properties with real `multiprocessing`; it says nothing about
Level-2 behaviour, which neither side has built. I judged the pushed code by
reading and by those two runs; the 22 + 14 checks in its smoke tests I did not
re-run.
