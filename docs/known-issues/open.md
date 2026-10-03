*[← docs/known-issues index](README.md)*

# Open

**Nothing outstanding at the moment of writing.** The one entry below was a
measured question, asked and answered on hardware, including the part that
was still open when it was first written: whether checkpointing's recompute
penalty applies at this operating point. That is now answered too — the
answer is that there is no un-checkpointed configuration there to measure it
against. It is kept here rather than in [`resolved.md`](resolved.md)
because it is not a bug and there is nothing to fix. Check here first if
something odd has happened — and its being empty is itself the finding.

## Attention-checkpointing density and the memory-floor levers, at 1024 / batch 2

**The question.** Gradient checkpointing costs a ~25% recompute penalty,
which looked attackable two ways: checkpoint fewer blocks, or shrink the
resident floor so fewer blocks need checkpointing. Both were measured on
real hardware — Intel Arc B580, 12 GB, `scripts/hw_validate.py`, batch 2,
dataset 1024, 40 steps, using a new `--attn-ckpt-fraction` knob plus new
per-stage floor capture in `summary.json`'s `floor_stages` and the first
step's `component_footprints_mb`.

**The floor, before anything is checkpointed.** 7,529 MB allocated at loop
entry, 7,888 MB at step 0, split as UNet+LoRA 4,897 MB + text encoder
1,561 MB + optimizer states 714 MB (lazy; they appear on step 1) + ~716 MB
of gradients, conds and misc. Both routes (main and managed) have the
identical floor.

**Density is binary.** Full density (1.0) peaks at 9,268 MB reserved —
0.768 steps/sec main, 0.716 managed, against a bs1 reference of 8,592 MB
peak and ~0.946 steps/sec. **0.75 and 0.5 both OOM on step 0**, at ~10.7 GiB
allocated mid-forward with 14–22 MB free. 0.75 still OOMs on the managed
route even with the floor released to 6,327 MB, because
`AdaptiveResidencyController` calibrates on three fully-resident steps and
step 0 dies before any release can happen. Skipping even a quarter of the
blocks costs more than the ~2 GB of headroom that exists: there is no
useful middle ground at this operating point.

**The floor levers, each against the managed 0.716 steps/sec / 9,268 MB
baseline.**

| lever | floor given back | throughput | why |
|---|---|---|---|
| `nf4` weight store | 1,361 MB (566 MB of peak) | 0.574 steps/sec (−20%) | dequant on every forward |
| `int8_blockwise` optimizer states | 529 MB — and peak *rises* to 9,440 MB | 0.491 steps/sec (−31%) | per-step cast cost |
| `--budget 8000 --cache-text-encoder` | allocated down to 5,613–6,327 MB, reserved only to 8,960 MB | 0.331 steps/sec (−54%) | freed memory lingers in the allocator pool; dominated by re-uploading the 714 MB of optimizer states needed every step |

Releasing the **text encoder alone** was the one theoretically cheap lever
— 1,561 MB, and nearly free once `cache_text_encoder` holds the
conditioning — but it was unreachable as designed: the controller releases
candidates smallest-footprint-first, so the always-needed optimizer is
released first and drags its per-step transfer cost along with anything
else.

**What landed instead: `ManagedLoRATrainerNode`'s `prewarm_text_encoder`
Port.** Prewarming sidesteps the ordering problem entirely rather than
reordering the controller — the encoder is unloaded once at build, before
calibration, and never re-uploaded. Every later step's encode is a cache
hit; misses self-load through the cache's bound handle; and a 0-footprint
candidate is one `AdaptiveResidencyController` stops considering.

Measured after (managed, batch 2, dataset 1024, 40 steps): floor
7,888 → 6,327 MB at step 0, peak reserved 9,268 → 7,666 MB, throughput
0.716 → **0.789 steps/sec (+10%)** — the per-step CLIP forward is gone too.

That also confirms the causal story above, because the floor cut now
happens *before* calibration, which the budget path could not do (it only
releases after three fully-resident calibration steps). Density 0.75 now
completes, at a peak of 10,922 MB — ~300 MB under the wall, so feasible
rather than comfortable. **It is not a speed win**: 0.759 steps/sec against
0.789 at density 1.0, while adding 3.3 GB of activation residency.
Recomputing these blocks is cheaper than carrying their activations.

**Verdict.** Keep density 1.0 and prewarm. Density tuning is closed, and so
are the quantized and offload levers — nothing in these measurements
rescues them, because they were never blocked on floor.

**Was still open, and is now answered below:** whether the ~25% recompute
penalty figure measured elsewhere still applies at this operating point,
where checkpointing's activation residency costs less than the compute it
saves.

### Answered: the penalty cannot be measured here, because there is nothing to measure it against

Re-run after design doc 12 section 7.3 reimplemented the diffusion path, the
VAE and both CLIP towers — so these numbers are now measurements of *this*
project's code, with `comfy` provably absent from the process (see the note
on the harness below). Same operating point: `main`, batch 2, dataset
`1024 aes`, 40 steps, rank 64 / alpha 32, seed 1234, `attn_ckpt_fraction`
1.0, `div_4.safetensors`.

**The baseline reproduces.** Floor at loop entry **7,529 MB allocated**,
against the 7,529 MB recorded above — an exact match. Steady throughput
**0.813 steps/sec** against 0.768 (+6%), and peak reserved **9,228 MB**
against 9,268 (−40 MB, −0.4%). So the reimplementation did not move this
operating point; both differences are within what this harness's own
run-to-run spread covers.

One component does differ, and it is recorded rather than explained: the
`unet_lora_on_device` stage measures **5,254 MB allocated** against the
4,897 MB in the breakdown above, while the text encoder's contribution
matches closely (+1,571 MB against +1,561 MB) and the loop-entry *total*
matches exactly. So the components shifted and the total did not. This file
does not claim to know why; a breakdown that sums correctly is worth more
than one whose components are individually tidy.

**And the question closes negatively: `--no-checkpoint` OOMs on step 0.**

    torch.OutOfMemoryError: XPU out of memory. Tried to allocate 34.00 MiB.
    GPU 0 has a total capacity of 11.93 GiB of which 34.20 MiB is free. Of
    the allocated memory 10.80 GiB is allocated by PyTorch.

It dies in the UNet's first convolution, before step 1 completes. So at
1024 / batch 2 there is **no un-checkpointed configuration to compare
against**, and the ~25% recompute penalty is not a trade anyone can take at
this operating point — it is the price of running at all. That is a stronger
answer than a percentage would have been: the question asked whether
checkpointing's residency was worth its compute here, and the measurement
says the alternative does not exist.

Whether the ~25% figure applies at a *smaller* operating point, where the
alternative does exist, is a different question and is not answered here.
Nothing in the measurements above moves it.

### A note on the harness, because it is what makes the re-measurement mean anything

`scripts/hw_validate.py` used to put ComfyUI on `sys.path` on the strength
of a comment saying the node route's model construction imported `comfy.*`.
That stopped being true when section 7.3 landed, and the insertion is gone.
It was worth removing on its own terms rather than as dead code: with
ComfyUI on `sys.path`, a run that *looked* like it was measuring this
project's reimplementation could silently have imported ComfyUI's and
reported numbers for code nobody intended to measure. The checkpoint files
are still read from ComfyUI's `models/` directory — that is a path to a data
file, resolved by `paths`, and it stays. `scripts/probe_lora_shapes.py` had
the same insertion and the same removal.

Verified rather than assumed: a full harness run with `sys.modules` inspected
afterwards reports `comfy imported during a full run: False`.