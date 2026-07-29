# Dataset renoising, CLIP prewarm, LoRA checkpoint loading, 2026-07-28 (cont.)

Continuation of `docs/vram_and_lora_phase_split.md`, same day. Three
things, driven directly by user feedback on the first round: a real bug
in how the dataset was described (my mistake, corrected here), the CLIP
VRAM fix actually built (design question the user asked directly,
answered by building it), and the missing "continue training" node.

## The fixed-timestep-grid bug (user caught this, not me)

I initially told the user their dataset's target computation "matches
what kohya/diffusers/OneTrainer do: real image, add real noise at a
sampled timestep, predict that noise." The target formula part is true.
The *sampled* part is not, and the user caught it: `manager/builder.py`'s
`run_ingestion_task` builds a timestep grid **once**, before the
per-image loop, and reuses the identical grid for every image:

```python
if t_mode == "uniform":
    ...
    t_grid = [...]                       # built once, ~20 values
for i, img_path in enumerate(...):       # per image
    for t_val in t_grid:                 # same grid, every image
        ...
```

Verified directly (not re-asserted from memory) at
`manager/builder.py` around line 486. Every sample in a dataset built
this way sits at one of only ~20 exact timestep values -- 18 jittered by
a fixed draw, 2 endpoints (t_low/t_high) completely unjittered --
repeated identically across the entire dataset and every epoch. That's a
real, structural difference from mainstream tools (kohya/diffusers
resample a fresh continuous timestep per image every time it's used),
and a plausible, direct mechanism for the destructive-early-training
symptom the user described: a low-rank LoRA adapter has a real path to
learning a shortcut keyed to those exact ~20 timestep embeddings rather
than a smooth function of noise level, especially since inference
samplers visit a different, unrelated set of timestep values entirely.

**Fix, without re-ingesting.** Re-ingesting means a full VAE re-encode of
every image -- expensive, and unnecessary here, because x0 (the clean
latent) is exactly recoverable from what's already stored. Ingestion's
forward process is `x_t = x0 + sigma * eps`, `target = eps` (or
`eps_to_vpred(eps, ...)` for vpred models) -- confirmed against the real
formula in `manager/builder.py`, not assumed -- and `core/noise_schedule.py`
already has the exact inverses (`eps_to_x0`, `vpred_to_x0`), reused here
via composition rather than re-derived. `nodes/dataset/renoise.py`'s
`RenoiseBatchSource` wraps any `TrainingBatchSource`: recovers x0 from
each batch's stored `(x_t, target, t)`, draws a **fresh, independent**
timestep and noise per sample (not a shared draw for the whole batch),
and reapplies the exact same forward formula at the new timestep. Wire it
between the dataset source and the trainer; nothing else changes.

`model_type` (needed to pick the eps or vpred inverse) comes from the
batch's own `"metadata"` JSON string, same `json.loads(...)["model_type"]`
convention `manager/loader.py` already uses (default `"eps"` on the same
fallback). alpha/sigma for both the stored and the new timestep come from
`core.noise_schedule.get_alpha_sigma`, the exact function ingestion
itself calls -- not a re-derivation that could drift from it.

**Known limitation, stated plainly in the module docstring too:**
`target_p`/`target_n` (the dual-CFG-pass fields) only regenerate
correctly for real-image ingestion, where `manager/builder.py` sets
`target_p == target_n == target` by construction (verified directly).
There's no teacher model available here to regenerate genuinely distinct
positive/negative targets -- don't use this on an actual teacher-
trajectory distillation dataset (different code path, different
invariant).

**Verification:** real torch, exact numeric round-trip -- x0 recovered
from a batch built with the literal ingestion formula, then recovered
*again* from the renoised output, both compared against the real x0 with
`atol=1e-5`. Also checked: resampled timesteps actually span the range
(30 batches all "ingested" at the same fixed t=500 come out spread across
roughly [2, 928], not clustered), not just "different from before."

**What this doesn't fix by itself:** existing already-ingested datasets
still only ever had ~20 *distinct clean images worth of noise draws*
baked in per image at ingestion time (this doesn't add information that
was never captured -- it only stops re-serving noise/timestep pairs that
were baked at fixed points that repeat, letting the injected noise/
timestep diversity be much wider than 20 discrete values without needing
to re-run VAE encoding). Whether this is *the* fix for the described
structural-degradation symptom, versus a contributing factor, isn't
something to claim confidently without an actual before/after training
run -- flagging the honest uncertainty rather than overclaiming a fix
that hasn't been tested yet.

## CLIP prewarm-then-unload (the user asked for a specific design opinion)

Asked directly: "Should it be a new functional node that is alternative
to default CLIP input and have list of tensors as replacement? Or maybe
something else?" Answer: neither a replacement input nor a list of
tensors -- a decorator that preserves the existing `TextEncoder`
interface exactly (`encode(prompt, batch_size, height, width) -> (ctx,
y)`, `unload()`), so `SupervisedLoRATrainerNode` needs zero changes.
Reasons this beat the alternatives considered:

- **A "list of tensors" replacement input** would mean precomputing every
  batch's conditioning ahead of time and handing the trainer a static
  collection instead of a live encoder -- this only works if the exact
  batch order/composition is known and fixed ahead of the training run,
  which isn't true in general (shuffling, bucketing/clumping in
  `manager/loader.py`'s own `__iter__` reorders things every epoch). It
  would also mean `SupervisedLoRATrainerNode` needs a second, different
  code path for "conditioning already provided" vs. "call the encoder" --
  more surface area for what should be a pure optimization.
- **A genuinely different port/node type** (something other than
  `TextEncoder`) would leak this into every node that currently expects a
  `TextEncoder`, for no real benefit -- the whole point of the existing
  `TextEncoder` abstraction is that callers don't need to know how
  conditioning gets produced.

So: `nodes/model/text_encoder_prewarm.py`'s `PrewarmedTextEncoderNode`
takes the real encoder plus the dataset, does **one real pass over the
dataset** (cheap -- a `ManagedDatasetLoader`-backed source just reads
already-stored tensors, no VAE/CLIP involved) to discover every
`(prompt, batch_size, height, width)` key training will actually request
-- derived the same way `SupervisedLoRATrainerNode._run_step` derives it
(`batch["x_t"].shape[2] * 8`, the VAE downsample factor) so it isn't
guessing at the key format independently. Then it warms
`CachingTextEncoder` (from the first round's `nodes/model/text_encoder_cache.py`,
reused via composition, not re-implemented) with exactly those keys, and
calls `unload()`. Mirrors `core/trainer.py`'s own cache-then-unload
pattern, just keyed off the dataset directly instead of enumerating raw
prompt strings by hand.

Degrades rather than breaks if something outside the discovered set is
ever requested later: `unload()` only moves the underlying model to CPU,
it doesn't destroy it, so an unexpected cache miss still returns a
correct answer, just recomputed on CPU. Verified with a counting fake
encoder + a fake dataset with a deliberate repeat and a deliberate
distinct-resolution bucket: exactly 3 real encode calls for 4 batches (2
unique prompts x 1 resolution + 1 of them again at a second resolution),
`unload()` called exactly once, post-warmup calls served from cache with
zero further real calls, and an out-of-band key after warmup still
returns a correctly-shaped result instead of raising.

## LoRACheckpointLoaderNode

`nodes/model/lora_checkpoint_loader.py`. One node serves both requested
uses (resume training further, or provide a frozen base for
`LoRAPhaseSplitNode`) since they're the same underlying operation --
load weights into the currently-injected registry -- differing only in
what's wired downstream afterward. Reuses `core.lora.load_lora_into_model`
directly (already exercised for real in `smoke_test_lora_phase_split.py`'s
round-trip checks, not new/unverified code). That function's own coverage
check is permissive (silently skips anything not found in the file);
this node adds a loud check in front of it instead -- every currently-
injected layer's expected key must be present, and rank must match, or it
raises with the specific mismatched keys/ranks named, rather than a
partial silent load or an assertion from three calls deep. Verified:
real save-then-load round trip reproduces the original trained forward
output exactly; both the missing-keys and rank-mismatch paths raise with
the expected content in the message, not just "raises something."

Only meaningful *before* a phase-split (loads onto plain
`core.lora.LoRALinear`/`LoRAConv2d`, not `LoRAGeneration` layers --
`load_lora_into_model`'s isinstance gate would silently skip the latter).
Documented directly in the module docstring: load first, split after.

## Files touched

- `nodes/dataset/renoise.py`, `nodes/model/text_encoder_prewarm.py`,
  `nodes/model/lora_checkpoint_loader.py` (new)
- `nodes/model/lora_phases.py` (`_lora_key` -> `lora_key`, made public so
  the checkpoint loader can reuse the exact same key-naming convention
  instead of a third private copy of the same two lines)
- `server/nodegraph_registry.py` (registered all three new nodes)
- `nodes/smoke_tests/smoke_test_renoise.py`,
  `smoke_test_text_encoder_prewarm.py`,
  `smoke_test_lora_checkpoint_loader.py` (new)
