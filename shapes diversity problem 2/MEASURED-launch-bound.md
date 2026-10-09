# The step is launch-bound: measurements for L4/L5

Numbers measured on the Intel Arc B580 (12,216 MB, torch 2.12.1+xpu), dataset
`non-square`, SDXL LoRA, managed route. Reusable mechanism for each item first,
then the measurement.

## The premise, re-measured

`scripts/count_launches.py` (copied in from the task file) counts the
kernel-launching aten ops in one production step on the `meta` device, through
the project's own UNet and LoRA layers. It reproduces the established number:

| configuration | adapted | forward | backward | total |
|---|---|---|---|---|
| base only (no LoRA), checkpointing off | 0 | 3,503 | 3,234 | 6,737 |
| LoRA, checkpointing off | 564 | 6,887 | 7,945 | 14,832 |
| **LoRA, checkpointing ON (production)** | 564 | 6,887 | 15,410 | **22,297** |

The task file says 22,286; this build gives 22,297, an 11-launch difference,
so the count is stable to about 0.05% and the discrepancy is code drift since
the file was written rather than a measurement error.

At the measured 1.06 s of CPU per fast step that is **48 us per launch**, which
is a normal eager PyTorch XPU launch. So the step is launch-bound, and step
time is (launches) x (48 us).

## L5.2 — cutting launches in the LoRA branch: measured, and **not**
recommended

The LoRA branch is 8,095 launches (36% of a step). One `LoRALinear.forward`
costs 7 counted launches:

```
mm(base)        F.linear(x, base_weight, base_bias)  -> bf16
_to_copy        x.to(fp32)                           activation cast UP
mm              (x32 @ lora_A.T)                     -> fp32
mul.Tensor      lora_B.T * scaling
mm              (@ that)                             -> fp32
_to_copy        lora_out.to(bf16)                    activation cast DOWN
add.Tensor      result + lora_out
```

Three variants were measured over the whole step
(`measure_lora_launches.py`, meta device, no GPU):

| variant | launches | vs current | what it costs |
|---|---|---|---|
| current | 25,149 | — | — |
| `fp32-fused` | 25,011 | **−0.5%** | none |
| `bf16-mm` | 23,327 | **−7.2%** | up to **2.27%** relative error on the adapter's own delta |
| `cached` (A/B casts hoisted) | 19,407 | **−22.8%** | needs an invalidation design that does not exist yet |

**There is no free win here, and the reason is arithmetic rather than taste.**
`addmm` is what folds `scaling` and the residual add into one launch, but it
requires all three operands to share a dtype:

- keeping the adapter in fp32 means casting the base result *up* to fp32, and
  that extra cast cancels the saving almost exactly (−0.5%);
- casting the small matrices *down* to bf16 instead avoids that cast, but makes
  the rank-64 matmul a bf16 one.

The bf16 numerics were measured, not assumed (`measure_lora_numerics.py`, CPU,
one real layer). Relative error on the adapter's own delta:

| rank | `\|B\|`=0.01 | `\|B\|`=0.10 |
|---|---|---|
| 16 | 1.29% | 0.51% |
| 64 | 1.90% | 0.63% |
| 128 | 2.27% | 0.73% |

It **grows with rank**, so it is not one rounding (bf16 is ~0.4%) — it is two
matmuls each rounding once. On the layer's full output the error is much
smaller (0.06–0.31%) because the frozen base weight dominates, but the quantity
that carries the adapter's contribution is the delta.

Note `lora.py`'s own docstring already names this arrangement as mainstream
("every mainstream LoRA implementation (HF PEFT, diffusers) keeps trainable
adapter weights in fp32 even when the frozen base model is fp16/bf16, casting
down for the forward matmul"), and the parameters here *do* stay fp32 — it is
only the compute that moves. But "mainstream" is not a measurement, and −7.2%
of launches (≈8% of step time) for up to 2.27% on the adapter's contribution is
a trade, not a free win.

**The −22.8% ceiling is not shippable as measured.** It needs a cache of bf16
copies of `lora_A`/`lora_B`. The obvious invalidation key is
`(param._version, param.data_ptr())`, and measured on this build:

| mutation | `_version` | `data_ptr` |
|---|---|---|
| `AdamW.step()` | **bumps** (0 → 2) | unchanged |
| `param.data = other` | unchanged | **changes** |
| `param.data.copy_(other)` | **unchanged** | **unchanged** |

and `LoRALinear.load_lora_weights` in this very file writes through
`param.data.copy_()`. So the obvious key misses the one writer this codebase
actually has, and a stale cache there would silently train on old weights — the
failure mode that is hardest to notice and worst to debug. It needs either a
generation counter bumped once per optimizer step (two call sites, both
existing) or every writer invalidating explicitly. Neither is built.

The measurement script is worth keeping for the reason its own docstring gives:
an earlier version of the `cached` variant *recomputed* the casts inside the
counter and so measured exactly the same as `bf16-mm` while claiming to model a
cache. A measurement that reports the number it was built to improve is worse
than no measurement.

## L5.3 — XPU graph capture: **2.05–2.08x, and it works**

The highest-ceiling item, because replay attacks all ~22,000 launches at once
instead of shaving a few hundred. `hasattr(torch.xpu, "XPUGraph")` is **True**
on this 2.12.1+xpu build (the task file said to check on 2.14; it is already
here).

`probe_xpu_graph.py` captures one real UNet forward+backward at one shape with
static input buffers and times replay against eager, same buffers, same
process.

| arm | eager | replay | speedup | vs production eager | pool cost |
|---|---|---|---|---|---|
| production (default SDPA, checkpointing ON) | 878.3 ms | — | — | — | — |
| **MATH SDPA + capture** | 1010.1 ms | **421.2 ms** | 2.40x | **2.08x** | 459 MB |
| **EFFICIENT SDPA + capture** | 993.4 ms | **430.5 ms** | 2.31x | **2.04x** | 517 MB |
| FLASH SDPA + capture | 828.6 ms | — | **cannot capture** | — | — |

**The blocker is the SDPA backend, and for FLASH it is not ours to fix.**
Capturing with the default backend fails inside
`F.scaled_dot_product_attention` with

```
RuntimeError: Graph nodes cannot depend on events from outside the graph.
```

Forcing FLASH_ATTENTION gives a different and much more specific failure:

```
RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not
yet available for use with the SYCL Graph extension.
```

That is a SYCL runtime limitation: the flash kernel wants scratch memory the
graph extension cannot yet express. No setting in this project changes it, and
FLASH is the *fastest* eager backend (828.6 ms), so the choice is MATH or
EFFICIENT — each ~10-15% slower eager than the default, and both more than
repay that under capture.

The activation-checkpointing `fork_rng` hazard did **not** bite: capture
succeeded with checkpointing ON, and the recompute inside backward was captured
with it.

The replay is verified, not assumed: the probe changes the input buffer and
replays, and the captured loss moves (MATH 0.120859 → 0.108437, EFFICIENT
0.110228 → 0.117422). A replay that ignored its inputs would look exactly like
a fast one.

**What is still open before this is a feature**, in rough order of risk:

1. **One graph per shape.** Bucketing leaves 3 static shapes; each pool is
   ~460-520 MB, so ~1.5 GB retained. Production peak is 7,220 MB of 12,216 MB,
   which leaves room, but this has not been measured.
2. **The optimizer step is not captured.** It is a separate set of kernels. It
   could be captured, but then the graph would own the parameter update.
3. **In-place updates are load-bearing.** A captured graph reads the adapter
   weights from fixed addresses, so it picks up new values only because the
   optimizer updates in place. An optimizer that replaced `param.data` would
   leave the graph reading freed memory — the same hazard as L5.2's cache, and
   for the same reason.
4. **Dropout RNG.** Dropout is 0 in this configuration. `XPUGraph` does have
   `register_generator_state`, so this is answerable but untested.
5. **Static conditioning buffers.** `ctx_emb`/`y` would have to be copied into
   static tensors each step, which is where L1's per-sample sizes land.

### Built 2026-10-09: integrated on the managed route (`use_xpu_graph`)

`nodes/train/xpu_graph_step.py` (`XPUGraphStepRunner`) + `GraphForwardLossBackwardPhase`
replacing Forward/Loss/Backward-backward when the Port is on (default off).
Per (batch, H, W): statics + hoisted (B,) weight vector, 3 genuine eager
warmups, capture, replay; optimizer step, clipping, encoding, monitoring
eager. Loss math shared with LossPhase (`masked_per_sample_mse`,
`apply_loss_weighting` in `loss.py`). Loud refusals: fused optimizer,
Dropout(p>0) (scan reaches through the TrainableModel façade),
grad_accum>1 (design handles it, parity run only covers 1 -- prove before
lifting), unscannable model. Model pinned resident (sacrificable=False);
grads zeroed in place (optimizer.zero_grad reassigns to None, which replay
would write past). Fallbacks loud, never fatal.

Acceptance (B4: loss parity over 200 steps, same seed; non-square, batch
2, x32, non-fused AdamW):

| | eager | graph | 
|---|---|---|
| steps/s steady | 0.991 | **1.353 (+37%)** |
| peak reserved | 7,220 MB | 8,730 MB (+1.5 GB pools) |
| max abs per-step loss diff | — | **2.96e-03** |
| mean abs per-step diff | — | 2.15e-04 |
| loss step 0 / step 199 | 0.007306 / 0.165788 | 0.007300 / 0.165932 |

Parity holds; the residual is MATH-vs-default SDPA bf16 noise (6.5e-06 at
step 0, slow drift). The acceptance caught one real bug first: reporting
handles aliased warmup-eager tensors instead of capture-pool outputs
froze the reported loss per shape (stale, correct-looking numbers --
grads were live throughout). Fixed by keeping the loss/per_sample objects
created inside the capture context.

Still open (unchanged): main-route integration, grad_accum>1 proof,
dropout>0 (needs register_generator_state), mid-run shape counts past
the 4-graph cap stay eager.

## L5.1 — activation checkpointing: the probe said 1.56x, the real run OOMs

**This corrects a number reported earlier in this file.** The capture probe
measured checkpointing OFF at 564.5 ms against 878.3 ms ON — 1.56x — and that
figure was written up as "1.56x faster for +3,460 MB, whether a full run fits
unmeasured". The full run does not fit:

| | steps/s | ms/step | steps completed | peak reserved |
|---|---|---|---|---|
| checkpointing **ON** (production) | 1.023 | 978 | **150 / 150** | 7,220 MB |
| checkpointing **OFF** | 1.227 | 815 | **4 / 150** | 10,282 MB |

```
torch.OutOfMemoryError: XPU out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 11.93 GiB of which 5.62 MiB is free.
Of the allocated memory 10.43 GiB is allocated by PyTorch,
and 94.57 MiB is reserved by PyTorch but unallocated.
```

**Checkpointing OFF is ~1.2x faster and dies at step 4.** The probe's 1.56x was
real for what it measured — a bare UNet with no optimizer, no text encoder and
no residency controller — and wrong as a statement about training. Two things
it left out, both of which cost memory rather than time: the optimizer's state
and the LoRA gradients, and the residency controller holding the text encoder's
budget. This is the clearest instance in this task of a correct isolated
measurement being the wrong answer, and the only reason it is not still
circulating is that the full run was actually executed.

`--attn-ckpt-fraction` (0.5, 0.25, …) already exists and is the real sweep: it
trades a fraction of the 1.2x for a fraction of the +3 GB. Not run — the
question L5.1 asks is answered, and the useful follow-up is "which fraction",
which needs its own measurement.

### Follow-up run 2026-10-09: the fraction sweep (batch 2, x32, seed 1234)

| fraction | steps/s | peak reserved | drift | loss path |
|---|---|---|---|---|
| 1.0 (control) | 0.996 | 7,220 MB | 2 MB | identical |
| **0.5** | **1.166 (150 steps) / 1.238 (300 steps)** | 9,354 MB | 972 MB once, then flat | identical |
| 0.25 | 0.582 | 10,590 MB | −2 MB | identical |
| 0.0 | OOM at step 4 (prior result, stands) | 10,282 MB | — | — |

Loss trajectories agree to the 4th decimal (0.00730 → ~0.1177 in all
arms): fraction changes numerics not at all, as exact recompute predicts.
The curve is **non-monotone**. 0.5 buys +17–24% for a one-time +2.1 GB
(the 300-step run holds peak 9,354 flat from the first 50 steps, so the
972 MB is a single allocator step, not a progressive ratchet). 0.25 is a
cliff: uniformly 1.8x slower steps (median 1.70 s vs 0.95 s, not stalls —
2 slow steps in every arm), the allocator-pressure regime at 87% full
where every allocation fragments. So 0.5 is the only useful setting, and
only with headroom; the 1.0 default stands for tight budgets.

## L4 — step time IS flat to batch 4, so per-sample captions are worth doing

The first measure, no code. `hw_validate.py --batch 1/2/4/8` on `non-square`,
150 steps, `shape_bucket_multiple=32` (so 3 shapes and the compile cost is out
of the way).

| batch | steps/s | ms/step | **images/s** | vs batch 1 | peak MB | reserved drift |
|---|---|---|---|---|---|---|
| 1 | 0.988 | 1,012 | 0.99 | 1.00x | 7,226 | 10 MB |
| 2 | 0.996 | 1,004 | 1.99 | 2.02x | 7,220 | 2 MB |
| 4 | 0.996 | 1,004 | 3.98 | **4.03x** | 7,376 | 2 MB |
| 8 | 0.645 | 1,550 | 5.16 | 5.22x | 8,336 | **640 MB** |

**Step time is flat within 5% from batch 1 to batch 4** — 0.988 → 0.996 steps/s,
which is noise — while images/s goes up **4.03x**. It stops being flat at batch
8: 1,550 ms/step, so batch 8 buys only 1.30x over batch 4 and costs +960 MB
more reserved plus 640 MB of allocator drift (MEM-09's ratchet, reappearing
across shapes once activations are big enough to matter).

This is the premise L4's structural change rests on, and it holds with room to
spare: because a step costs the same at batch 1 as at batch 4, a batch that may
contain four different captions instead of one costs *nothing extra*, and the
loader no longer has to put every image with a unique caption in a batch of its
own. **Batch 4 is the sweet spot on this card** — the last size that is still
free.

Two limits on what this establishes:

- **`non-square` has exactly one distinct caption** (every prewarm line in this
  task's runs: "1 distinct prompt(s)"). So this measures that a *bigger* batch
  is free. It does **not** measure that *heterogeneous captions* in one batch
  are cheap — that needs per-sample `ctx_emb`/`y`, which is the code L4 defers
  until this premise is checked, and which adds a text-encoder encode per
  distinct caption in the batch. On a one-caption dataset that is one encode
  either way, so this run cannot speak to it at all.
- **LoRA conditioning dominates the per-step CPU cost, not the text encoder.**
  At 22,297 launches per step there is no room in the budget for a per-sample
  conditioning path to be free by default; it has to earn its place.

### Follow-up run 2026-10-09: hetero captions measured on synthetic data

`datasets/multi-caption`: byte-identical copy of `non-square` latents with
8 synthetic distinct prompts round-robin (34–35 samples each; datasets/ is
gitignored — regenerate per `measure_l4_caption_cost.py`'s docstring).
Measurement only; quality is irrelevant. Batch 4, shuffle, seed 1234:

| dataset | bucket | keep_incomplete | trained/epoch | batches |
|---|---|---|---|---|
| non-square (1 caption) | off | False | 184/273 | 46 |
| non-square | off | True | 273/273 | 97 |
| non-square | x32 | False | 268/273 | 67 |
| **multi-caption (8)** | off | False | **28/273** | 7 |
| multi-caption | off | True | 273/273 | 187 |
| multi-caption | x32 | False | 232/273 | 58 |
| multi-caption | x32 | True | 273/273 | 79 |

**Caption fragmentation, not shapes, is what starves the run.** Eight
captions × 63 shapes → groups of ~4, and the default drops every group
under the batch: 28 of 273 train per epoch. Bucketing mitigates much of
it (8 × 3 buckets → groups of ~11: 232/273) because it collapses the
shape axis of the grouping key. `keep_incomplete_batches=True` recovers
everything but pays 187 batches for 273 samples unbucketed. Batches carry
exactly one prompt by construction (`_merge_samples` takes
`samples[0]`) — hetero batches cannot form under this grouping at all.

Text-encode cost (real towers, XPU, `encode_prompts`): **31.5 ms per cold
distinct prompt, linear** (1/4/8 prompts: 32/126/251 ms), **0.0 ms warm**
(cache-consistent, verified equal tensors). One-time device init ~800 ms
lives outside any prompt's number. Worst case per step — a full batch of
4 never-seen captions — is ~126 ms against a ~1,000 ms step, paid once
per caption and never again.

So the hetero-caption price is ~zero in steady state, and L4's remaining
cost is the loader regrouping plus `y` assembly, not encodes. What L4
buys on multi-caption data is not speed but **data**: 28 → 273 usable
samples per epoch at batch 4 unbucketed.

### Built 2026-10-09: size-only grouping + per-sample conditioning

`ManagedDatasetLoader(group_by_size_only=True)` groups by size bucket
alone (refuses without bucketing — unbucketed mixed shapes cannot
collate and there is no mask); batches carry a per-sample `prompts`
list (`prompt` kept as `prompts[0]` for old consumers).
`encode_with_true_sizes(..., prompts=)` routes uniform lists to the
unchanged single-prompt call (byte-identical) and mixed lists to the new
`TextEncoder.encode_per_sample_prompts` (one `encode_prompt_only(p, 1)`
per distinct prompt, cache-served; per-sample ctx/pooled/size rows).
Both `EncodeConditioningPhase`s pass `batch.get("prompts")`; default
grouping never touches new code. Node Port + `hw_validate.py`
`--group-by-size-only` wired.

GPU proof (`L4_sizeonly`: multi-caption, batch 4, x32, 150 steps, seed
1234): 150/150 ok, loss 0.157 → 0.091, **67/67 batches hetero**
(2–4 captions), 268/273 trained (5 incomplete-chunk drops, expected),
0.906 steps/s vs same-seed single-caption control 0.939 (−3.5%,
noise-level), peak identical 7,376 MB. Hetero costs nothing per step;
it buys the data (232 → 268 trained at x32; 28 → 273 unbucketed).

## What is not worth doing, and why

- **Shape pre-warm.** Measured and closed in
  `archive/shapes diversity problem/MEASURED-shape-stall.md`: activation-bound at
  ~12 GB for a single shape, so no room for a second thread. After bucketing,
  three serial compiles cost ~9 s per run, so the remaining prize is small.
- **Turning activation checkpointing off.** ~1.2x, and it OOMs at step 4 in a
  real run (L5.1 above). `--attn-ckpt-fraction` is the real lever, not the
  on/off switch.
- **A cross-process primitive cache** would fix the whole class, and oneDNN has
  no supported persistent form. Platform limit, not a missing setting.
- **LoRA launch micro-optimisation**, per L5.2 above: the levers that work all
  cost precision or need a correctness answer first, against a 2x that graph
  capture delivers with neither.

## Where the throughput went, and what is left

Starting point and the order things were measured in:

| | steps/s | vs start |
|---|---|---|
| 44 shapes, launch-bound (the original problem) | 0.690 | — |
| + primitive cache sized to the shape count | 0.757 | 1.10x |
| + shape bucketing x32 | 1.076 | **1.56x** |
| + XPU graph capture, MATH SDPA (one shape, not yet integrated) | 2.15 equiv. | **~2.0x more** |

Bucketing and graph capture are close to additive — one removes compile
stalls, the other removes launch cost — because they attack different parts of
the same 48 us-per-launch budget. The capture number is *not* a drop-in: it is
one shape, no optimizer step, no conditioning, and it needs MATH or EFFICIENT
SDPA in place of the faster default.

The thing none of this fixes: **at batch 8 the GPU becomes the limit**, and
nothing in the launch budget can help there. The remaining headroom on this
card is in getting to a useful batch size cheaply, which is L4's structural
lever and not a throughput one.

