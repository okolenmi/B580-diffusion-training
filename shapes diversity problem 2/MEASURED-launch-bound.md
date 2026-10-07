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

## L5.1 — activation checkpointing costs 1.56x of step time

Incidental to the capture probe, and larger than expected. Same model, same
buffers, one variable:

| | eager step | reserved |
|---|---|---|
| checkpointing **ON** (production) | 878.3 ms | 6,294 MB |
| checkpointing **OFF** | **564.5 ms** | **9,754 MB** |

**1.56x faster for +3,460 MB.** Whether a full training run fits without it is
unmeasured — the production run peaks at 7,220 MB with checkpointing on, so
the naive projection is ~10.7 GB against a 12,216 MB card, which is inside the
harness's 11,500 MB budget but not comfortably. `--attn-ckpt-fraction` already
exists for the intermediate points and is the next thing to sweep; this probe
used all-or-nothing only because it is not the question the probe was written
for.

## L4 — not measured yet

`hw_validate.py --batch 1/2/4/8` on a dataset whose images share captions
(`non-square` has exactly one distinct prompt, confirmed by every prewarm line
in this task's runs: "1 distinct prompt(s)"). Whether step time stays flat as
batch grows is what decides the whole per-sample-captioning change, and it is
the one L4 item that needs no code to answer.

## What is not worth doing, and why

- **Shape pre-warm.** Measured and closed in
  `shapes diversity problem/MEASURED-shape-stall.md`: activation-bound at
  ~12 GB for a single shape, so no room for a second thread. After bucketing,
  three serial compiles cost ~9 s per run, so the remaining prize is small.
- **A cross-process primitive cache** would fix the whole class, and oneDNN has
  no supported persistent form. Platform limit, not a missing setting.
- **LoRA launch micro-optimisation**, per L5.2 above: the levers that work all
  cost precision or need a correctness answer first, against a 2x that graph
  capture delivers with neither.
