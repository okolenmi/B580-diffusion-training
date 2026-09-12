*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 4 -- `ResourcePreset` abstraction

**Status: the core object-construction mechanics are done.** Composition
vs. inheritance decided -- inheritance, concrete-mixin-first ordering.

`nodes/model/sdxl_architecture.py`'s `SDXLArchitecture`: pure SDXL
mechanics, no dtype awareness at all, exactly as specified.
`split_checkpoint()`/`build_text_encoder()`/`inject_lora()` all
delegate to already-real, already-tested code (`resource_inspection.classify_key()`,
`text_encoder.py`'s existing CLIP masking, `build_lora_injected_unet()`)
rather than reimplementing anything -- this class is assembly, not new
logic. `build_text_encoder()` masks SDXL's real two text encoders
(CLIP-L, OpenCLIP-G) behind one simple `TextEncoder` object, per the
explicit design requirement -- but honestly flags a real, current
limitation found while building it: `core.clip_encode.SDXLClipEncoder`
(frozen legacy code) hardcodes its own dtype, no parameter exists yet
to make it configurable.

`nodes/model/lora_training_resources.py`'s `LoRATrainingSkeleton`
(abstract, declares the three architecture-specific methods above) and
`SDXL_LoraTrainer(SDXLArchitecture, LoRATrainingSkeleton)` (the
concrete combination -- base order load-bearing, not stylistic, see
that class's own docstring for the exact MRO reasoning). Construction
*is* processing, per the original design conversation: `__init__` runs
the real pipeline (split → inject LoRA → mask CLIP) and the resulting
instance has real `.unet`/`.clip`/`.vae_sd`/`.lora` attributes with no
construction machinery riding along, matching "the beauty of this
method" from that conversation exactly.

**Real memory/objects management inside, per the explicit follow-up
ask -- and a real, reassuring answer to "not sure we have a good base
for it": there already was one, already real and already
production-used, just not yet connected here.**
`nodes/memory/coordinator.py`'s `ResourceCoordinator` is the same
thing `nodes/train/supervised.py`'s `SupervisedLoRATrainerNode` already
registers its own `model`/`optimizer`/`text_encoder` against in a real,
shipping node. `LoRATrainingSkeleton` is now a real `DeviceResident`
itself (`nodes/memory/handle.py`) -- registers its own `.unet`/`.clip`
against an internal coordinator at construction time, so
`footprint_bytes()`/`offload()`/`reload()`/`release()` delegate to
real, already-tested machinery (each already implements
`DeviceResident` itself) instead of a second, hand-rolled, parallel
implementation -- replaced the hand-summed `footprint_bytes()` from the
first pass of this phase. `vae_sd` (real tensor state, not yet wrapped
in any object -- see the deferred item below) is moved/dropped by hand
alongside the coordinator's own work in all three lifecycle methods,
not silently left out of them just because there's no resident object
to register it with yet.

**Two things deliberately, honestly deferred, not silently left
half-built:** "continue training" (loading an existing saved LoRA into
the freshly-injected model, per the original sketch's own checkbox) --
`.lora` stays `None` unconditionally; the real loading mechanics
already exist (`LoRACheckpointLoaderNode`, DoRA-aware) and this class's
`.unet` is exactly what that loader operates on, but wiring them
together is a real, separate next increment. No VAE object either --
`.vae_sd` stays the raw split-out state dict, since nothing in `nodes/`
builds a VAE wrapper anywhere yet (only legacy `core.vae_decode.VAEDecoder`,
unused by anything in `nodes/` today).

**Verified**, `nodes/smoke_tests/smoke_test_lora_training_resources.py`:
`split_checkpoint()` correctness (every key in exactly one bucket);
`build_text_encoder()`/`inject_lora()` each proven to genuinely
delegate (real call-arg recording, not assumed); a full end-to-end
`SDXL_LoraTrainer` construction from a synthetic checkpoint, checking
real `.unet`/`.clip`/`.vae_sd`/`.lora` and a correctly-summed
`footprint_bytes()`; the `DeviceResident` implementation genuinely
moves/drops all three of `.unet`/`.clip`/`.vae_sd` through
`offload()`/`reload()`/`release()` -- not just the two with an obvious
resident object to delegate to, `vae_sd`'s own raw tensors checked by
their real `.device` too, both the explicit-device and no-arg
`reload()` paths exercised, and a released trainer correctly reports 0
footprint; and, the one that actually matters most for this phase's
central decision -- **the negative case**: a version with the bases
listed in the wrong order genuinely fails to instantiate with
`TypeError`, proving the shipped order is load-bearing and not just
happening to work. Extended (not forked) the existing `_RecordingWrapper`/
`_FakeClipEncoder` test fixtures with `.to()`/`.unload()`/`.dtype` they
were missing once `offload()`/`reload()`/`release()` needed to
genuinely exercise them, rather than a second, subtly-different copy.
Adjacent tests (`smoke_test_resource_coordinator`,
`smoke_test_resource_policy`, `smoke_test_resource_inspection`,
`smoke_test_gradient_checkpointing`, `smoke_test_dataset_model_contracts`,
and the extraction test itself after being extended) still pass. No
non-CPU device is available in this sandbox, so the offload/reload
checks prove the real mechanics run correctly, not an actual
cross-device tensor move -- worth a real check on real hardware before
fully trusting the device-transition behavior specifically.

**Frozen LoRA -- done.** `nodes/model/lora_merge.py`'s
`merge_lora_into_state_dict(base_sd, lora_sd, strength)`: merges a
saved LoRA directly into a checkpoint's raw weights before injection --
`W_merged = W + strength * (alpha/rank) * (B @ A)`, matching
`core.lora.LoRALinear.merge()`/`LoRAConv2d.merge()`'s own formula
exactly (checked against them directly, not an independent
reimplementation trusted on its own). No frozen-LoRA object exists
after construction -- its effect is baked into the weight tensors,
nothing else about it survives. `strength` corresponds to
`core.lora.LoRALinear`'s own `weight` constructor argument (also a
scaling multiplier), applied here at merge time instead of injection
time. `LoRATrainingSkeleton.__init__` gained `frozen_lora_sd`/
`frozen_lora_strength` parameters -- the merge runs on the UNet
component before `inject_lora()`, so the trainable LoRA gets injected
on top of the already-merged weights. `frozen_lora_sd=None` (the
default) is a true no-op, checked directly: the wrapper receives the
checkpoint's own unmodified weights.

**Continue training -- done.** `nodes/model/lora_checkpoint_loader.py`'s
`LoRACheckpointLoaderNode.build()` was extracted the same way
`ComfyUNetLoRANode.build()` was in the earlier extraction patch --
`load_lora_into_registry(registry, state_dict, source_description)`
now holds the real validation (missing keys, rank mismatches, both
raising with specifics rather than silently loading a partial LoRA)
and loading (plain layers via `core.lora.load_lora_into_model`, DoRA
layers via the existing `_load_dora_layers()`), reused by both the
node and `LoRATrainingSkeleton.__init__`'s new `continue_lora_sd`
parameter. A different feature from frozen-LoRA merging -- this one
loads into `self.unet`'s own trainable adapter, after injection, so
training continues from these weights rather than starting fresh, and
stays trainable afterward (frozen-LoRA merging doesn't). `self.lora`
holds `continue_lora_sd` itself when given, `None` otherwise -- a
plain reference, matching `self.vae_sd`'s own raw-dict pattern, not
the weights themselves (those live inside `self.unet`'s registry once
loaded). Checked to coexist correctly with `frozen_lora_sd` given
together -- both apply, neither interferes with the other, since they
operate on genuinely different things (base weights vs. the trainable
adapter). A validation error from `load_lora_into_registry` (a real
rank mismatch, say) propagates straight out of construction rather
than being swallowed.

**LoRA-file inspection -- done, closing the gap flagged since Phase 2.**
`resource_inspection.py`'s new `inspect_lora(path)`: dtype and rank for
a saved LoRA, read from the header only (`get_shape()` alongside
`get_dtype()`, same header-only mechanics as checkpoint inspection).
`asset_paths.inspect()` now accepts `kind="lora"` too -- response shape
`{kind, path, dtype, rank, key_count}`, distinct from `kind="checkpoint"`'s
per-component shape. `kind="dataset"` is still the one real remaining
gap. Verified the same way as checkpoint inspection: real answer
against an explicit allowlist, response provably narrow, path
traversal rejected, plus the absent/mixed distinction for both dtype
and rank.

**Not yet built:** validators (per-input human-readable detection text
-- both the checkpoint and LoRA inspection functions this needs now
exist), the parameter-value dictionary (dtype choices), and
list-of-inputs -- the actual `ResourcePreset`/`NodePreset`-satisfying
interface pieces that make this usable *as a node*. Those, plus wiring
this whole thing into an actual `Node` subclass, are Phase 5.

**A working-approach note, not a technical one, worth recording
because it changes how the rest of this redesign should be built:**
earlier framing in this document leaned toward declaring an input only
once its full implementation was ready, to avoid a `Node` contract that
advertises something not yet functional. That instinct is wrong for
this project specifically -- the redesign's whole point is a complete,
correct foundation, even where some of it goes temporarily unused, and
retrofitting a contract after the fact costs more than building it
right the first time. Phase 5's `NodePreset`/`ResourcePreset`
interface should declare the full intended shape of the Resources
Controller now, not grow it incrementally as each piece happens to be
implemented -- and each independent unit (this merge function, the
inspection functions, the construction classes) should be designed
from its own inputs and outputs first, not from how it currently fits
into what already exists; existing code changes to match a better
design when the two conflict, not the other way around.

**Dependency:** none technically (pure abstraction design, could
proceed in parallel with Phases 1-3), but Phase 5's concrete preset
needs it finished first -- it now is, for the core construction path.
