*[← docs/design index](README.md)*

# 10. Prioritized backlog

**New: first-run installer and dependency gate** --
[`11-first-run-and-installer.md`](11-first-run-and-installer.md). A
separate item rather than one on this list because it has a different
shape: the work is a sequence that is only worth doing as a sequence
(check -> gate -> download -> first-run state -> wizard -> multi-root
model search), and the second half of it is a *design* item rather than a
build item. The part that is pure friction-removal and can be taken
independently is Phase B: make `run_server.sh` check its dependencies and
say what is missing, instead of starting and letting the failure surface
as a `ModuleNotFoundError` traceback. That is a small change with most of
the user-visible benefit.

Two things in that plan are worth noticing before anything is built. We
would be installing into a venv the user did not create -- made safe by
splitting the requirements into "everything" and "additions only" and
installing the latter under a constraints file of what that venv already
holds, so nothing already installed can move. And model lookup would
become main-then-reserve rather than the single-root-with-override it is
today, which changes what existing code has to assume in five places.

1. ~~**Verify `NF4WeightStore`'s quality against a real training run**~~
   (3.3) -- **the quality half is measured, 2026-10-04.** A matched pair
   on the B580 (managed, batch 2, `1024 aes`, 40 steps, seed 1234, only
   `--weight-store` differing) puts NF4's documented ~9% weight-space RMSE
   at **0.102% of loss at step 0** and **0.147% mean across 40 steps**,
   with both curves descending by the same 39.9% -- three orders of
   magnitude smaller in function space than in weight space, and no effect
   on training. What it *does* cost is **16% of throughput** (0.707 ->
   0.593 steps/sec) to save **180 MB** of peak, so at this operating point
   it is a bad trade. Numbers and the limit of the measurement (40 steps is
   ~0.4 epochs, so this is "does not break training", not "identical
   quality over a full fine-tune") are in
   [`../known-issues/resolved.md`](../known-issues/resolved.md).
   *Still open on this item:* the `MemoryManager`-backed scratch buffer for
   the dequantised tensor (real, separate optimization, not blocking
   correctness) -- see `nf4_lora_layer.py`'s own docstring -- and
   `DoRAAdapter` not honoring `NF4WeightStore` (`QDoRA`).
   A `MemoryManager`-backed scratch buffer for the dequantized tensor
   (real, separate optimization, not blocking correctness) is the other
   remaining piece -- see `nf4_lora_layer.py`'s own docstring.
   `DoRAAdapter` doesn't honor `NF4WeightStore` yet (`QDoRA`) -- real,
   separate follow-up, not done here.

**A real gap surfaced while landing `ResourceProfile`, not yet its own
backlog item because nothing above needs it yet:** there is no single
shared `MemoryManager` instance reachable from the trainer's `build()` --
each optimizer execution strategy
that uses one (`ChunkedScratchBufferStrategy`) constructs its own private
instance when none is injected, and no `Composed*OptimizerNode` exposes
a port to inject a shared one. Harmless today (each strategy's own
private manager is internally consistent), but it means
`ResourceProfile.memory_manager_stats` is `None` in every real run, and
it'll become a real problem the moment two things need to share one VRAM
budget on purpose (e.g. a future `AutoResourcePolicy`, or several
strategies deliberately pooling scratch space) -- worth threading a
shared `MemoryManager` through optimizer construction as part of
whichever future item first has a concrete reason to, rather than
speculatively now.

**Run the test suites in parallel. Done** (`277b16f`, and the `nodes/`
runner alongside it).

The backend suite was **21.5s serial** with the top five files 53% of it
and the CPU 58% idle; `backend/tests/run_all.py` now runs the files in a
`ProcessPoolExecutor`, **22.69s -> 6.32s on 6 jobs**. Each file keeps its
own `TMPDIR`, removed afterwards -- which both contains the litter every
test file leaves behind (see `docs/known-issues/open.md`) and is what makes
the files independent enough to run at once. Output order is preserved:
`pool.map` yields in submission order, because a log interleaving 26 files
makes a failure much harder to find.

`nodes/smoke_tests/run_all.py` was **119.9s serial** over 66 files, with
the cost spread evenly (ten slowest = 26%, longest single file 4.4s), so
it parallelises well -- but **not all of it**. An explicit serial list runs
first and alone for the tests that assert on *device state*
(`device_context_equivalence` asserts the `xpu_empty_cache` call order,
`memory_manager` asserts allocator and residency, `sdxl_text_encoder_offload`
asserts memory was actually released). These do not strain VRAM -- that was
measured, not assumed -- but two of them at once are measuring each other's
allocations, so a failure would mean nothing. **113.5s -> 46.5s**, and the
serial list is now 43% of what remains: the next lever is trimming that
list, which trades risk-reduction for speed and should be a deliberate
call rather than a default.

Multiprocessing there uses the **spawn** context, not the default `fork`:
a forked child inheriting a torch/XPU-initialised parent crashes the
pool, and it does so as `ConnectionResetError` from the forkserver rather
than anything legible.

**Validation work, not construction -- real code exists for both, what's
missing is a real run:**

- **`RescaledZeroTerminalSNRSchedule` + `VPredParameterization`, end to
  end** (1.4). What's missing is a real training run and qualitative
  image-quality evaluation -- does it actually fix medium-brightness
  clustering on this project's own data. Unlike everything numbered
  above, there's no old code path to equivalence-test against; this needs
  real training runs to trust, not a unit test.
- **`LoRAPlusGroups`, actually run** (3.4). What's missing is running it
  on a real LoRA training job and comparing against a `UniformGroups`
  baseline, at whatever `ratio` (the `16.0` default is an unverified
  starting point) turns out to matter for this project's own data.
- **`GreedyRatioPlacement`, wired and run** (2.3). Nothing has actually
  run the instrumentation against this project's own UNet yet, and
  `ComfyUNetLoRANode` has no port to select `ProfilingCheckpointing` or
  `GreedyRatioPlacement` in the first place -- both real, separate,
  smaller follow-ups once a first profiled run's numbers exist to wire
  a real placement decision against.
- **`DoRAAdapter`, a real training run** (3.1). Not yet run on this
  project's own data to confirm the quality improvement DoRA reports in
  its own published benchmarks (LLaMA/LLaVA/VL-BART, not diffusion
  UNets) actually shows up here too. Checkpoint save/load (direction +
  `.dora_scale` magnitude + alpha) is real now for the common, unsplit
  case -- see 9.1/9.2 -- so this item is validation-only, same as the
  others in this list.
- ~~**Why the monitor's VRAM graph lists CLIP while the text encoder is
  released**~~ -- **answered and partly fixed, 2026-10-04.** CLIP is
  legitimately resident when the residency controller decides to keep it,
  and *illegitimately reported as resident* when prewarm had already put
  it in host RAM. Two distinct things, and the second was a bug.

  **Resident, on purpose.** Peak 9,226 MB against a 9,889 MB usable budget,
  so the controller's decision set is empty, so
  `should_release("text_encoder")` is False, so
  `EncodeConditioningPhase` never offloads it:

      [residency] measured peak=9226MB over 3 calibration step(s),
                  usable budget=9889MB -- nothing -- staying fully resident
      [step 0] residents: model=4897MB optimizer=714MB text_encoder=1561MB

  **And not resident, while still being reported as 1,561 MB of it.** A/B on
  the B580, batch 2 / `1024 aes`, `--profile`:

  | | peak reserved | residents |
  |---|---|---|
  | `prewarm_text_encoder` off | 9,228 MB | model 4,897 / optimizer 714 / **text_encoder 1,561** |
  | `prewarm_text_encoder` on | **7,666 MB** | model 4,897 / optimizer 714 / **text_encoder 0** |

  `prewarm_text_encoder` is the port that does exactly what it should:
  warm the cache over every key training will ask for, then `unload()` the
  encoder for the rest of the run. Every step's encode is then a cache hit
  that never touches the model. **1,562 MB of peak for free** -- and
  throughput unchanged (0.703-0.760 steps/sec across runs, which is this
  card's own run-to-run noise, so this is a VRAM win and not a speed one).

  It reported 1,561 MB because `unload()` frees the device inside
  `SDXLClipEncoder.unload()` without setting the `_device_before_offload`
  flag that `footprint_bytes()` consults, so under exactly the setting
  that frees CLIP, every consumer of the footprint -- the monitor's VRAM
  graph first among them -- was told about memory that was not there.
  Fixed by having `unload()` record the device too, with the port's own
  contract quoted ("0 while offloaded, not the byte count of whatever's
  now sitting in host RAM instead") and a test for both routes. The
  `resolved.md` entry has it.

  **Resolved 2026-10-04: `prewarm_text_encoder` now defaults to on**, the
  project's call. The trade, all measured on the B580 at batch 2 / rank 64 /
  `1024 aes`:

  | | cost |
  |---|---|
  | warm pass | **3.5-3.8 s**, once (2 keys on `1024 aes`, 75 on `non-square`) |
  | host RAM for the cache | **+6.1 MB**, by RSS delta |
  | extra dataset iteration | one pass; batch materialisation is ~0.2 s |

  The cache is negligible next to what it replaces, which was the open
  question: **6.1 MB of host RAM against 1,561 MB of device memory.** It
  scales at ~0.6 MB per distinct `(prompt, batch_size)` -- 77x2048 fp32 --
  so a 512-prompt dataset would hold ~310 MB, still under a fifth of CLIP,
  and `max_entries` caps it regardless. `non-square` adds 0.0 MB on top
  because its 75 keys are 2 prompt keys and 73 resolution keys, and the
  resolution half is three orders of magnitude smaller.

  **Peak: 9,228 MB -> 7,666 MB, at rank 64 as well as the default rank.**
  Throughput unchanged across every combination run (0.671-0.760
  steps/sec), which is this card's own spread.

  **What it buys against real cards.** An 8 GB card is ~7,634 MB usable:

  | | peak | fits 8 GB? |
  |---|---|---|
  | batch 2, 1024, rank 64 | 7,666 MB | **no -- 32 MB over, 0.4%** |
  | batch 1, 1024, rank 64 | 7,230 MB | yes, ~400 MB spare |

  So 8 GB becomes *nearly* viable at batch 2 and comfortably viable at batch
  1, where before the flip neither was. Batch 2 costs only 436 MB more than
  batch 1, because the UNet (4,897 MB) and optimizer (714 MB) dominate and
  do not scale with batch -- with gradient checkpointing on, which is the
  default, activations are a small share of the total.

  **One assumption could not be left to trust, so it is bounded.** Prewarm
  needs `batches` finite per iteration, and a source that never ends would
  hang the trainer at startup with no output and no error -- the worst way
  for a default to fail. `discover_dataset_keys` takes `max_batches` now,
  called with `MAX_DISCOVERY_BATCHES` (100,000, far above any real dataset:
  the largest here is 152 batches). On reaching it, discovery stops and
  warns. Truncation costs time and not correctness, because the keys past
  the bound are cache misses, and a miss self-loads and returns the right
  answer.

  **A harness bug this exposed, worth keeping.** `hw_validate.py` passed
  `prewarm_text_encoder=args.prewarm_text_encoder` unconditionally, pinning
  the measurement harness to its own argparse default (`store_true`, so
  False) rather than the node's. The first run after the flip still measured
  9,228 MB and looked like the flip had not worked. The flag is now
  `BooleanOptionalAction` with `default=None`, and the kwarg is only passed
  when set -- so unset means "let the node decide" and the harness measures
  the shipped default instead of silently pinning its own. A harness that
  pins a default stops measuring it, and reports the old number with total
  confidence.

- **Prewarm does not scale to a million objects, and now says so with
  numbers** — raised by the user for 1M+ object datasets. The measurements
  are in; the *design* answer (a windowed warm) is not built, and this is
  where to take it from.

  **What already scales, which corrects half the premise.** The dataset
  side is fine: `ShardLoader.load()` memory-maps each shard and reads
  tensors on demand ("Tensors are NOT loaded into RAM"), so 1M images
  never sit in host RAM at once. Loading is ~800 samples/s measured, so
  ~21 min per epoch of pure I/O for 1M — bounded by the disk, as it should
  be. The metadata index is 1.6 KB/sample, so **1.6 GB of host RAM for
  1M** — worth knowing, but not the cliff.

  **The cliff is prewarm, and both of its costs are linear in *distinct
  prompts*** (measured, XPU, 1024, batch 2):

  | | per distinct prompt | 10,000 | 100,000 | 1,000,000 |
  |---|---|---|---|---|
  | warm time | **30.6 ms** (1,505 ms on CPU) | 5.1 min | 51 min | 8.5 h |
  | host RAM | **621 KB** | 6.1 GB | 60.6 GB | **606 GB** |

  Plus the discovery pass, **1.24 ms per dataset sample** (201 samples in
  0.25 s), because it reads every latent off disk to read `x_t.shape`. For
  1M that is 21 minutes before step 0 regardless of how few prompts there
  are. So a 1M-object dataset with 1M distinct captions wants ~9 hours and
  606 GB today: infeasible on both axes, and the RAM axis binds first, at
  roughly 50k distinct prompts on a 32 GB machine.

  **Done now: the limits are observable, so "too much" is a number.**
  - `CachingTextEncoder.cache_bytes()` sums the tensors actually stored.
    `footprint_bytes()` answers a *device* question and correctly reports
    0 for these caches, which is useless for the question that decides
    whether warming is affordable.
  - The warm pass prints what it cost — key count, distinct-prompt split,
    one-time setup separated from the marginal per-prompt rate, and the
    resulting host RAM. It prints rather than logs, because nothing here
    configures logging and a measurement nobody can see is not one.
  - `PREWARM_HOST_RAM_BUDGET_BYTES` (8 GiB ≈ 13,000 distinct prompts) emits
    a warning naming the marginal cost and the alternative. Deliberately
    **not** a Port: raising it does not make warming a million captions a
    good idea, it only moves the failure later.

  **Done 2026-10-04: capacity now comes from a RAM budget, not from the
  dataset.** `PREWARM_HOST_RAM_BUDGET_BYTES` (8 GiB) /
  `PREWARM_BYTES_PER_PROMPT_ENTRY` (621 KB, measured) =
  **13,508 prompt entries**, and `max_entries` takes that instead of
  `len(prewarm_keys)`. This was the actual coupling: sizing the cache to the
  dataset made host RAM a function of dataset size, which is what made
  prewarm look unscalable at all. It had a second, quieter failure too --
  with capacity below the distinct-prompt count, the warm pass's tail
  encodes, inserts, and is LRU-evicted on the very next insert, so it warms
  *nothing* for those prompts and they every one miss later. So capping the
  cache required capping what gets warmed, in the same change:
  `warm_and_unload` filters prompt keys to capacity and **reports how many
  it skipped and what a skip costs** (~790 ms, measured). Resolution keys are
  ~0.3 KB each so they are not filtered -- `non-square` warms all 44 of them
  in 0.07 s.

  Verified by cutting the budget ~4096x: capacity drops to 3, a 10-prompt
  fake warms 3, warns with the real numbers, and the cache holds exactly 3.

  **Correction to the framing below, from measurement after the user
  proposed an on-demand alternative.** The user proposed dropping the
  up-front warm for a per-step loop: check the cache, miss, load CLIP,
  encode, evict CLIP, check again -- with eviction of other residents to
  RAM if there is no room. Measured, that loop is the *most* expensive
  option, not the cheapest, because the churn dwarfs the work:

  | | cost |
  |---|---|
  | CLIP load RAM->XPU | **364 ms** |
  | CLIP evict XPU->RAM | **396 ms** |
  | encode one prompt (resident) | **30 ms** |
  | **an on-demand miss, room available** | **790 ms** -- 96% churn |
  | UNet evict XPU->RAM / reload | **1,449 / 1,145 ms** |
  | **an on-demand miss, no room** (evict UNet too) | **3,384 ms = 139% of a 2.43 s step** |

  Eviction *is* cheap next to a whole step, which is the intuition behind
  it -- but the comparison that decides anything is against the 30 ms
  encode, and load+evict is **25x** that. So on-demand is the right
  *fallback* and the wrong *primary*: prewarm pays 30 ms once per distinct
  prompt and never again, where on-demand pays 790 ms (or 3,384 ms on a
  tight card) per miss, and misses are per distinct prompt. It is also
  already what happens -- `CachingTextEncoder` self-loads on a miss, so the
  user's loop describes the existing degradation path accurately, and it is
  what makes a miss *correct* rather than wrong. It just must not be the
  plan.

  **So the windowed warm stands, and it is the answer for the regime the
  RAM limit creates (>50k distinct prompts on 32 GB).** What the exchange
  did settle is *why* it needs order-knowability rather than a bigger
  budget: because on-demand is not a cheaper alternative to fall back on,
  so past the RAM limit the choices are a window, a second GPU, or
  accepting a per-prompt 790 ms.

  **Plan of record: the user's per-step design, and what it costs.** Not
  built. Written here because it is a real change to step ordering, not a
  tuning knob, and the trade-offs belong next to the numbers.

  *The design.* Per step: (1) the trainer needs an encoded prompt, and this
  is checked **before** the major objects go into VRAM; (2) on a miss the
  prewarm starts; (3) if there is not enough memory for CLIP, evict other
  residents to RAM -- acceptable, because that is quick next to a whole
  step; (4) CLIP encodes and then self-evicts to RAM; (5) the cache is
  checked again and the prompt is there; (6) model/optimizer/etc. load from
  RAM; (7) the step trains; (8) repeat. Each part does its own work without
  depending on what the others are doing.

  **What exists today, stated as a diff.** The model is put on the device
  at build time (`managed.py` ~1302), so step (1) cannot precede it.
  Prewarm runs once at startup (~1356), not per step; the only per-step
  behaviour is `CachingTextEncoder`'s existing self-load on a miss, which
  is steps (2)-(4) *without* the ordering, and it reloads CLIP rather than
  treating it as a one-shot job. Step (6) does not happen: model and
  optimizer stay resident for the run, which is what
  `AdaptiveResidencyController` is for.

  **The trade that decides it: step (6).** Making conditioning precede the
  model's VRAM load means evicting and reloading the model *per step*, and
  that is measured at **1,449 ms down + 1,145 ms up = 2,594 ms**, against
  a 2,430 ms step. So the design as literally stated is a **~2.1x slowdown**
  -- evicting 4,897 MB of UNet to save a 30 ms encode, 86 times over.

  *Positive.* Conditioning genuinely should not need the card, and the
  per-step loop makes the cache self-healing with no planner and no
  prediction of the loader's shuffle. It is also the only variant that
  works unchanged at any prompt count -- no window, no second GPU, no
  discovery pass.

  *Negative.* The model round trip is not a detail, it is the whole cost,
  and it is paid every step rather than once. On this card the UNet round
  trip alone (2,594 ms) is more than the entire encode of every prompt in
  a 5,000-prompt warm (55 s / 5,000 steps' worth).

  **The resolution, which keeps the design and drops the expensive part.**
  Split the two reasons the model is resident. Residency exists so a step
  is not dominated by host-device traffic -- true, and worth 2,594 ms when
  a step is 2,430 ms. It does *not* have to be unconditional. So:

  1. Keep the model resident by default. Unchanged, and the common case
     stays 7,666 MB peak.
  2. On a conditioning **miss** with the model resident, do *not* evict it.
     Encode with CLIP brought alongside, at the measured cost of the
     transient peak (1,561 MB on top of 4,897 + 714 = 7,172 MB resident,
     which is what the 7,666 MB peak already proves fits). Self-evict CLIP
     immediately after. This is steps (2)-(5) with no round trip, because
     on this card there is no need for one.
  3. Evict the model to make room **only when there genuinely is not
     room** -- a smaller card, or a larger batch. That path costs 3,384 ms
     per miss (measured) and should be reached rarely, which the 5,000-entry
     cache guarantees in steady state.
  4. Reorder within the step so the cache check happens before the model's
     forward regardless, so the miss is discovered while there is still
     time to act on it rather than after the expensive part is done.

  That is the user's design with the per-step model round trip made
  conditional instead of mandatory, which is the only part of it that
  costs more than it saves.

  **Prompt encoding is not a constraint -- measured, both dtypes.** The
  model already batches: `SDClipModel.encode(rows)` returns `(N, 77, D)`
  with the batch axis intact. `encode_token_ids` above it deliberately
  *joins* rows along the sequence axis instead, because that is how CLIP
  prompt sections work, so a batched path needs the lower-level call.

  | | ms/prompt | 5,000 prompts | vs serial |
  |---|---|---|---|
  | serial fp16 (today) | 32.2 | 2.7 min | -- |
  | batched fp16, batch 64 | 2.53 | 12.7 s | 12.7x, **19% different output** |
  | batched fp32, batch 64 | 10.99 | 55 s | 2.9x, 0.17% different |

  So batching must be **fp32**: fp16 batched diverges 19% from fp16 serial
  (relative to activation magnitude), fp32 batched only 0.17%, so it is
  fp16 accumulation over 32 layers and not the kernel. Neither path is a
  bottleneck either way -- 5,000 prompts is ~2,500 training steps, about an
  hour, so serial encoding is already only ~4% of the time spent using the
  result. **The limit that matters is the 3.0 GiB resident, not the encode
  time**, which is why 5,000 is a memory number.

  **Sequencing.** (4) and the `encode(rows)` batched path are independent
  and small; the cache limit of 5,000 is done. The conditional-eviction
  change in (3) is the one that alters step behaviour and wants measuring on
  the B580 at both 12 GB (where it should never trigger) and a constrained
  budget (where it must).

  **The one thing that looks wasteful right now, and is.** For all three
  real datasets there is **1 distinct prompt**, so the warm pass costs
  30 ms while `discover_dataset_keys` costs **1.24 ms per sample** -- at
  1M samples that is 21 minutes of discovery to warm one prompt. The
  discovery pass is ~99.99% of prewarm's cost on these datasets and 100% of
  it is avoidable when the prompt count is tiny. A cheap bound exists that
  needs no order-knowability: stop discovering once N consecutive batches
  have introduced no new *prompt* (new *resolution* keys are nearly free --
  1.6 ms each, and `non-square` has 43 of them against 1 prompt). Heuristic,
  and the failure is a miss rather than a wrong answer, so it is safe in
  the same way `MAX_DISCOVERY_BATCHES` is.

  **The strongest of the user's three proposals is the second GPU.** A spare
  card preparing prompts ahead removes the limit outright, because encoding
  stops being on the critical path and the host-RAM ceiling stops being the
  binding constraint. Larger than a windowed warm and it is the only option
  here that does not need to predict the shuffle.

- **Memory admission belongs to the graph, not to a node and not to each
  trainer** — the user's architectural call, 2026-10-04, recorded not built.

  **How unloading works today, so the proposal has a baseline.** There *is*
  a shared mechanism — `ResourceControlHandle._make_room()`, reached through
  `before_step()` and `ensure_loaded(name)`. It offloads eligible,
  currently-loaded residents in **registration order**, skipping exclusions,
  when measured `reserved_mb` exceeds the budget, and `release()` does the
  same deterministically right after a phase ends. Two shapes exist:
  pressure-triggered (what `CachingTextEncoder` uses) and
  declared-upfront (a resident whose idle windows are known). Its own
  docstring says which residents get which treatment "is each trainer's own,
  disclosed choice; this module provides both mechanisms, not a
  one-size-fits-all policy".

  **So the gap is real and is precisely where the user says it is.** There is
  no way for an arbitrary node to say "I need N MB now" and have anything
  arbitrate. A node that needs a big transient either knows about
  `ResourceControlHandle` and the registration order, or it does not
  participate — and `VRAMBudgetControllerNode` has to be *wired into a
  particular trainer's* `resource_control` input, so it is scoped to one
  trainer rather than to the graph.

  **The proposal, in the user's terms.** One controller, callable by any node
  on the graph, taking a required VRAM number, deciding what to unload. Not a
  node — the existing budget node is the wrong altitude — but something one
  level above, carrying per-nodegraph settings so different graphs can hold
  different policies. `VRAMBudgetControllerNode`'s own docstring already
  anticipates part of this shape ("add the ABC split if and when a second one
  genuinely shows up"), which is the same instinct applied one layer down.

  **Why it is the right altitude, which is the argument worth keeping.**
  Memory admission is a *whole-graph* property because the scarce resource is
  shared by everything on it. Two trainers on one card is the failure the
  supervisor's own comments keep returning to, and today each trainer
  independently believes it is the only thing on the card — each has its own
  budget, its own coordinator, and its own registration order. A per-trainer
  controller cannot see the conflict; a per-graph one is the only thing that
  can.

  **Admission is at graph start, by reservation -- settled 2026-10-04 by
  the user, and it settles the question I had framed as open.** Each graph
  holds its own budget and allocates only from the pool nothing else has
  claimed. If the sum of demands exceeds the card, the run **cannot start** --
  there is no in-flight eviction to design, no preemption, and no controller
  that can fail halfway through and leave a graph half-admitted. My "refuse,
  or evict and retry" was the wrong shape: it assumed admission happens during
  a run, and the whole point is that it does not.

  That also disposes of the only question I thought was sharp. A controller
  that can only fail *is* useless -- unless failure is admission, decided
  before the first step, where refusing is the complete and correct answer
  rather than a corner case.

  **The object is the graph, not a global setting and not a node.** Global
  settings were considered first and rejected as wrong on both counts: they
  cannot differ between graphs on one card, and a node is below the altitude
  of a decision that is about everything on it. The **graph is the
  configurable object** -- it is what carries a budget, and it has "a lot of
  room for improvement" as a configurable thing.

  **Left to design when this is picked up**, and these are naming the work
  rather than answering it: what a graph's settings object contains beyond the
  budget (ordering policy between its own nodes is the obvious second thing,
  since two nodes in one graph can also conflict); whether a budget is a
  hard reservation or a ceiling that nodes *request* against; and what a
  graph looks like when two graphs are declared against one card and do not
  both fit -- which under reservation is a start-up error, so the only design
  question is what the message says and what it offers.

- **The CLIP vocabulary, actually vendored** — the one item design doc 12
  §7 left open, and *not* a validation task: the code is done and tested,
  what is missing is a *file*. `default_vocabulary_dir()` still falls back
  to ComfyUI's `sd1_tokenizer/`, so a checkout on a machine with no ComfyUI
  cannot tokenize.

  **The technical objection is gone; only the licence decision is left.**
  Checked 2026-10-04 against `openai/clip-vit-large-patch14`: `merges.txt` is
  byte-identical, and `vocab.json` is semantically identical — 49,408 entries
  both sides, identical key sets, same id for every entry — differing only in
  whether the writer used `indent=2`. There is **no ComfyUI-authored content
  in these files at all**, so this is regenerating a published artifact
  rather than vendoring a third party's, and the bytes can be re-derived and
  diffed instead of trusted. 1.49 MB compact, and the whole tokenizer test
  passes against the published copy. Full numbers, the derivation, and the
  `pip download clip` name collision that makes this awkward to repeat are in
  [`12-installer-and-comfy-decoupling.md`](12-installer-and-comfy-decoupling.md)
  §8. Deliberately last on this list: it is the project's call, not a
  finding, and nothing is blocked until it is taken.

- ~~**A tiny-parameter (`< 10,000` element) `ExecutionStrategy` for
  Adafactor's cross-parameter batching case** (11.1)~~ -- **closed
  2026-10-02 by retiring the wrapper instead of building it.**
  `nodes/optimizer/adafactor.py` is deleted; `ComposedAdafactorOptimizerNode`
  is the only Adafactor node. `ChunkedXPUAdafactor` tied every tiny
  parameter in the optimizer together into one shared clip/EMA state --
  a batching-strategy concern no `Algorithm` can see (by design, see
  `algorithms/base.py`) -- and the proposed fix was a
  `ShapeGroupedBatchStrategy` variant grouping "under a size threshold,
  any shape". Part C of `smoke_test_adafactor_tiny_parameter_gap.py` had
  already measured that the shared state *contaminates*: a parameter's
  update depends on unrelated parameters' gradients sharing its batch. So
  the strategy would have existed to reintroduce coupling the canonical
  math deliberately lacks. The formula gap that was real is closed
  (`AdafactorAlgorithm.tiny_parameter_threshold`, Fused's per-parameter
  mechanism). What remains is one unmeasured performance delta -- see
  `docs/known-issues/open.md`, which also carries the B580 measurement
  that should precede writing any such strategy.
- **`core`/`manager` coupling in the model and dataset domains** (new
  section, not yet numbered above -- see `docs/architecture.md`). The
  `optimizer/` domain's `Algorithm`/`ExecutionStrategy` split proved a
  domain can be fully separated from `core/`'s legacy implementation,
  verified equivalent, and the old wrapper retired; text encoding then
  followed on 2026-10-02 (`SDXLClipEncoder` -> `nodes/model/clip_encoder.py`,
  a relocation, `core/clip_encode.py` kept as a shim for `core/`'s and
  `manager/`'s own use). Three sub-pieces remain, and they are not
  equally hard:

  - ~~**`nodes/model/` LoRA/UNet injection** (`core.lora`,
    `core.unet_wrapper`)~~ -- **done 2026-10-02.** This was the one
    flagged as "not a thin import": `core.unet_wrapper` is the *model*
    every LoRA path is built on, and `core.lora._inject_lora` resolved
    `LoRALinear`/`LoRAConv2d` as module-level names at call time, which
    is what `adapter_injection.py` patched in place to substitute DoRA
    and NF4 layers. Owning the file made that patch unnecessary --
    `_inject_lora` now takes the classes as an argument -- which removed
    a documented concurrent-build race, deleted `lora_class_cache.py`
    entirely, and fixed four `isinstance` gates that had been silently
    skipping every DoRA and NF4 layer. `nodes/` now imports nothing from
    `core/`; `core/{lora,unet_wrapper,clip_encode,seed}.py` are re-export
    shims for `core/`'s own use.
  - **`nodes/dataset/managed.py`** (`manager.loader`, and through it
    `manager/t_sampling.py` -> `core.noise_schedule.sample_timestep` and
    `core.model_io.make_init_noise`, `core.seed.derive_seed`). Note the
    shape here is *inverted* relative to the other two: `nodes/` doesn't
    depend on `manager/`, `manager/` is the implementation and `nodes/`
    calls into it. Worth deciding deliberately which side should own it.
  - **`manager/builder.py`'s own use of `core/`** (9 modules), which is
    the backend's dataset ingestion path rather than anything `nodes/`
    reaches.

  So dataset ingestion is now the only domain left to unwire. It is
  worth doing deliberately rather than by following the pattern above:
  the other two were relocations of self-contained modules, whereas here
  the dependency runs the opposite direction -- `manager/` is the
  implementation and `nodes/dataset/managed.py` calls into it. Deciding
  which side owns `sample_timestep`, `make_init_noise` and the dataset
  loading is the actual content of that piece, and the move is only the
  easy half.

**Not recommended as near-term work, with reasoning kept where it's
argued in full:** `ComponentRegistry`/`TrainingRecipe`/`PipelineFactory`
(5.3, 5.4 -- nothing built through `nodes/components/` so far is
graph-editor-selectable, so the side-by-side-registration problem these
solve hasn't materialized). **Deliberately deferred or rejected, for the
reasons section 7 gives in full:** `AutoResourcePolicy`, automatic
eviction inside `MemoryManager`, layer-wise base offload, flow matching,
GaLore, 8-bit optimizer moments.
