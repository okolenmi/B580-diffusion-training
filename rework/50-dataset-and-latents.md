# 50. Dataset, latents and prompts: the Data part

Status: **proposal.** The owner accepted the earlier dataset plan (library,
caption stages, batcher) and asked for improvements where visible. Needs 10
(events), 40 (`LatentSpec`, `Conditioning`), 41 (`VAEPart`), 42 (prompt
channels). Inputs: component-map doc, `docs/design/backend/04-dataset-format.md`,
`manager/db.py`, `manager/builder.py`, `nodes/dataset/*`, `step_pipeline.py`.

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What the owner asked for

* VAE belongs to Data; CLIP to Conditioning; UNet to Model.
* A designed part for the **connection of images and prompts** (a "latent-prompt
  controller"): when the dataset prompts change a lot it announces it; at the end
  of an epoch (last element consumed or cached) it signals so the tag-swapping
  node runs again.
* Dataset plan: good enough; improve if there is room.

## 2. What exists

* **Dataset format v2** (SQLite per dataset): `shards` (latent tensors,
  layout `single_latent` or `compressed_traj`), `trajectories` (one row per
  sample: `shard_id`, `shard_index`, `prompt`, `neg_prompt`, `model_type`
  `eps|vpred`, `cfg`, `source_path`), `sources`, `training_sets` and
  `set_members`. The schema carries its distillation heritage (teacher sources,
  `cfg`, trajectories).
* **Latents are produced at ingest and are model-specific**: the backend's
  dataset task calls `manager/builder.py`, which loads the VAE from a
  checkpoint (`first_stage_model.`), encodes each image (scale 0.13025, factor 8,
  4 channels) and writes shards. Nothing in the dataset records *which* VAE or
  family made the latents. The trainer never touches a VAE.
* **Noising happens in the loader**: `manager/loader.py` draws `t`, noise, builds
  `x_t` and the target (`eps_to_vpred`), so a batch already contains `x_t`, `t`,
  `target` (the prewarm code reads `batch["x_t"].shape`). That keeps data prep in
  the prefetch worker, overlapping device compute.
* `ManagedDatasetSourceNode` has 16 ports mixing library choice, batching,
  shape bucketing, timestep sampling, caption dropout, caption pool.
* An epoch = one `iter()` pass; the trainer re-iters on `StopIteration`
  (`FetchBatchPhase`); the prefetcher starts a fresh worker thread per pass.

## 3. Problems this plan fixes

1. A dataset is **welded to one VAE and one family**: Anima (16 channels,
   Qwen-Image VAE) cannot use a dataset encoded for SDXL, and nothing stops the
   mix-up.
2. **Prompts live beside latents** in `trajectories.prompt`; changing prompts
   means mutating the dataset's own records, which is wrong for "swap one tag per
   epoch".
3. **Noising is SDXL-shaped** in the loader (eps/v-pred); flow matching needs a
   different formula.
4. **One node does five jobs** (16 ports).
5. There is **no owner of the epoch**: the boundary is implicit in a trainer
   phase, so no node can react to it.

## 4. The Data part as a chain

```
 Dataset Library ──► Latent Cache ──► Caption Stages ──► Bucketer ──► Batcher ──► Prefetch ──► Trainer
   (ImageSet)         (LatentSet)      (PromptTable)      (BatchPlan)  (BatchSource)
        ▲                  ▲                  ▲                              │
        │            vae: VAEPart        AutoCaption                  noiser: Noiser
     library id      (Load Model)       Caption Pool                  (Diffusion part, plan 70)
                                        Caption Dropout
                                        Tag Swap  ◄── epoch_ending (event)
```

### 4.1 Handles

| Handle | What | Mutable? |
|---|---|---|
| `ImageSet` | sample ids, image paths, sizes, **captions per channel**; metadata only | no |
| `LatentSet` | id -> latent location in a cache keyed by (image hash, VAE hash, size policy); knows its `LatentSpec` | no (append-only cache) |
| `PromptTable` | base captions per id and channel, plus an ordered list of stages; `prompt_for(id, epoch)`, `version`, `universe()` when enumerable | **yes**, versioned |
| `BatchPlan` | which ids go in which batch, with shape buckets | per epoch |
| `TrainingBatchSource` | exists; iterable of batches | stream |
| `Noiser` | turns `(latent, t-policy, noise)` into `(x_t, t, target)` using the family objective (plan 70) | no |

### 4.2 Nodes

| Node | Ports (grouped) | Replaces / new |
|---|---|---|
| `DatasetLibraryNode` | `dataset`, `set` (training set); group **Filter**: min size, caption required | part of `ManagedDatasetSourceNode` |
| `LatentCacheNode` | `images: ImageSet`, `vae: VAEPart`; group **Size policy**: base resolution, bucket multiple (from `LatentSpec.pixel_multiple`), crop mode; group **Cache**: reuse, rebuild | **new**; absorbs the VAE-encode step of `manager/builder.py` |
| `AutoCaptionNode` | adds `output_channel` | exists |
| `CaptionPoolNode`, `CaptionDropoutNode` (new node, was a port), `TagSwapNode` (new), `TagShuffleNode` (new) | each takes a `PromptTable` and returns one with a stage appended | `CaptionPool` exists; the rest new |
| `BucketerNode` | `latents`, group **Buckets**: `shape_bucket_multiple`, `max_shapes` (merges the cheapest shapes until at most this many remain; each new shape costs kernel compilation and driver memory), `batch_area` (pixel budget per batch), `pack` (fills short batches: left-over samples top up a larger existing shape's short batch, then combine in the smallest existing shape that holds them, never past the padding share given and never past the batch-area size; it never creates a shape; with packing every batch carries a valid mask), `fill` (`zero`, or `mirror`: the padding of the noisy network input is a reflection of the sample's own noisy input, so it adds no information), `group_by_size_only`, `keep_incomplete_batches`; input `bucket_balance` | part of the source node |
| `BatcherNode` | `latents`, `prompts`, `plan`, `noiser`; group **Batching**: `batch_size`, `shuffle`, `seed`; event outs `epoch_ending`, `epoch_started` | part of the source node + loader |
| `PrefetchNode` | exists | |

A single combined **Dataset** node can ship as a *template* (a pre-wired chain)
so a beginner still places one thing; the chain is the real structure.

## 5. The pairing: prompts and latents join late

**Latents are immutable and keyed by image; prompts are mutable and keyed by the
same id; they meet only in the `Batcher`, at batch time.** That is the "latent-prompt
controller" the owner described. It owns:

* the **epoch counter** and the end-of-epoch barrier (below);
* the **join** on sample id (a prompt for an id with no latent is an error at
  plan time, not at step 4000);
* `prompts_version`, copied into every batch so a consumer can tell which
  prompts a batch used.

### 5.1 The epoch barrier (verified seam, plan 10)

`BatcherNode.__iter__` is a generator that runs in the consumer's thread:

```
def __iter__():
    emit EpochStarted(epoch)                 # handlers may change prompts for this epoch
    for batch in prefetched(plan_for(epoch)):  # a fresh prefetch worker per pass
        yield batch
    emit EpochEnding(epoch)                  # last batch already handed out
    epoch += 1                               # next __iter__ starts the next pass
```

Because the prefetcher already creates a new worker per `__iter__`, **nothing of
epoch N+1 is read before `EpochEnding(N)` handlers return**. "Last element
consumed" is literally the point after the final `yield`. Cached-ahead data
(latents) is not affected; prompts are looked up per batch, so a changed table
takes effect on the first batch of the next epoch. (`ManagedDatasetLoader.__iter__`
today already re-draws caption-pool prompts per pass.)

*To verify before implementing:* that nothing outside the prefetcher reads ahead
across the boundary (the loader's own shuffle and pool draws happen inside
`__iter__`); that `FetchBatchPhase` keeps calling `iter()` exactly once per
`StopIteration`.

### 5.2 `PromptsChanged`

Any stage or node that mutates the table calls `table.bump(changed_ids)`, which
raises `version` and emits `PromptsChanged(version, changed, of)`. The conditioner
(plan 42) reacts by warming only the changed prompts and dropping stale cache
keys. This replaces the whole-dataset "universe" computation the earlier
pool/blend patch had to add to the prewarm.

### 5.3 The "one tag per image" experiment, concretely

`Tag Swap(mode=one_per_image, seed=...)`: handler on `EpochEnding` picks, for each
image, one tag from its base caption's tag list (deterministic from `seed` and
`epoch`), replaces that image's active prompt, bumps the version. State derives
from `epoch`, so a resume at epoch 7 reproduces the same choice.

## 6. Latents belong to the VAE, not to the dataset

* `LatentCacheNode` takes a `VAEPart` from Load Model and a size policy. It writes
  latents to a cache **keyed by the VAE's identity** (hash of its header: key
  names, shapes, size, and the family's `LatentSpec`). Switch from the SDXL VAE to
  Anima's VAE and the cache builds a second set; **nothing is mutated or
  re-ingested by hand**.
* The library stores images and captions, which are family-independent. The
  latent cache is a **new SQLite format** designed for this (plan 90: old datasets
  are re-ingested or discarded) and records `latent_spec` and `vae_hash` so a
  mismatch is refused:
  "this dataset was encoded for sdxl-vae-4ch, the graph's VAE is
  qwen-image-vae-16ch".
* `LatentSpec.pixel_multiple` (plan 40) lets the family say what image sizes are
  legal (SDXL: 8; Anima: latent factor 8 times the transformer's spatial patch
  of 2 is 16), and the bucketer uses it.
* **No staging** (decision D1): the target design is built directly: the image
  library (images and captions, family-independent) plus one latent cache per VAE
  hash, with the VAE encode inside `LatentCacheNode`.
* The Datasets page keeps managing images and captions and shows, per dataset,
  which VAEs it has caches for.

## 7. Noising moves behind a `Noiser`

The batcher gets a `noiser` input (plan 70 builds it from the family's
`objective` kit: process, parameterization, timestep policy). The worker thread
still produces `x_t`, `t`, `target`, so prefetch overlap is preserved, but the
formula is the family's: eps or v-prediction for SDXL, rectified-flow velocity
for Anima. Timestep sampling (`t_mode`, `t_low`, `t_high`, `t_values`) leaves the
dataset node for the Diffusion part. The batch dict keeps its keys, so
`EncodeConditioningPhase`, prewarm discovery and the trainer see no change.

## 8. Room for improvement the owner asked for

| Idea | Why |
|---|---|
| **Stable sample ids from image content hash** | prompts and caches survive moving or re-ingesting images |
| **Caption channels** (`text`, `tags`, `description`) | dual captions (plan 42); AutoCaption can write a channel without overwriting |
| **Dataset report display** (resolution histogram, duplicates, empty captions, tag frequency) as a `Node.MODULES` display | catches data problems before a 6-hour run; reuses the display system |
| **Tag vocabulary** (counts per tag) held by the `PromptTable` | feeds Tag Swap, Tag Shuffle, dropout by tag frequency |
| **Per-stage seed derived from `seed.base`** (plan 20) | one number reproduces data order and caption draws |
| **Bucket policy from the family** | no hard-coded 64 px; the right multiple per family |
| **Preview grid** display (`image-grid`) on the batcher node | see actual crops and prompts without running |
| **Cache progress events** (`cache_progress`) | the node shows "encoding 1,240 / 5,000" instead of a frozen graph |

## 9. Tests planned

* Epoch barrier test with a fake hub: events fire in order; no batch of epoch N+1
  is produced before `EpochEnding(N)` handlers return; works with the real
  prefetcher at depth 1, 2, 4.
* Tag Swap determinism: same `seed` and `epoch` give the same table; resume at
  epoch k equals a straight run.
* VAE-hash refusal and cache keying with two synthetic VAE headers.
* PromptTable stage algebra: pool, dropout, swap compose in order; `version` bumps.
* Starter graph: the SDXL starter runs its dataset chain end to end on synthetic
  images (part of the one acceptance test, plan 90 section 6).
* Optional math spot check while the old loader still exists: noising output
  (`x_t`, `target`) equals the old loader's for a fixed seed.

## 10. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Latent ownership | in the dataset; per-VAE cache | Per-VAE cache, built directly (no staging, section 6) |
| D2 | Prompt mutability | rewrite DB rows; versioned in-memory table with stages | Versioned table; base captions never rewritten by stages |
| D3 | Epoch signal source | trainer phase; batcher iterator | Batcher iterator (barrier proven by the per-`__iter__` worker) |
| D4 | Noising location | trainer; loader worker | Worker, through an injected `Noiser`, to keep prefetch overlap |
| D5 | One node vs chain | keep one node; chain | Chain, with a pre-wired template for beginners |
| D6 | Sample ids | row ids; content hash | Content hash |
| D7 | Schema change | keep v2 plus metadata; new format | New format; no compatibility with v2 (plan 90) |

## 11. Risks

* Moving noising to a `Noiser` touches the hottest path. Mitigation: the optional math
  spot check against the old loader, before enabling anything for Anima.
* Stage chains mean more nodes per graph. Mitigation: templates (plan 60) and
  frames (plan 30); the combined pre-wired chain.
* A shared mutable `PromptTable` can be mutated outside events. Mitigation: only
  stages and `bump()` mutate; every mutation raises `version`; handlers run at the
  barrier so no reader is active.
* Cache disk use grows per VAE. Mitigation: show per-cache size in the Datasets
  page; allow deleting a cache.

## 12. Open questions for the owner

1. Is it acceptable that datasets stop being "pre-encoded for one model" and
   become images + captions with per-VAE latent caches? (Old datasets are
   re-ingested or discarded; plan 90.)
2. Should Tag Swap pick randomly per epoch, or walk each image's tags in order?
   (Both can exist as modes; which is the experiment?)
3. Caption channel file naming: `image.txt` as the default channel plus
   `image.tags.txt` / `image.desc.txt` sidecars, or a different convention?
