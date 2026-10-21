# The SDXL step, measured: what the INT8 path costs and saves

*Measured 2026-10-21 on the B580 (i5-8600K, Arc B580 12.2 GB, torch 2.12.1+xpu,
oneAPI 2025.3).  The instrument is `xpu-kernels`' `debug/probe_sdxl_trainer_step.py`;
it borrows this project's own UNet from `next/` and changes nothing in this
tree.  The numbers below are for the *denoiser's* forward, loss and backward
-- the part of the step that touches the UNet.*

## The setup

| | |
|---|---|
| the model | this project's own SDXL UNet (`next/families/general/ldm_unet`), 2.57 B parameters, random weights (the timing does not depend on the values) |
| the step | the base frozen, the input requiring grad: the **LoRA-shaped** step, so the backward computes the activation gradient and no weight gradients |
| the memory | this project's own checkpointing (`ActivationCheckpointer`, every block), because without it the step's activations do not leave room for a graph capture on a 12 GB card |
| the capture | this project's own pattern (`next/engine/step/graphed.py`): the forward, the loss and the backward inside one `torch.xpu.graph`, with the mem-efficient attention (which on XPU falls back to MATH -- the same choice the trainer makes) |
| the INT8 path | the library's `Int8Linear` for every linear whose contraction is a multiple of 256 (the library's own rule: 662 of them), and for arm C the library's **fused** GEGLU FFN, `int8.ffn(..., act="geglu_erf")` |

Three arms: **A** the UNet as it is (bf16), **B** the eligible linears
replaced by the library's `Int8Linear`, **C** B with the GEGLU's FFN replaced
by the library's fused FFN.

## The numbers

768×768, batch 1 (the trainer's default resolution):

| arm | forward | backward | eager step | **captured step** | host time | aten calls | weights | peak |
|---|---|---|---|---|---|---|---|---|
| A: bf16 | 127.5 ms | 273.6 ms | 400.7 ms | **565.6 ms** | 1058 ms | 44 277 | 4.89 GB | 5.23 GB |
| B: INT8 linears | 288.2 ms | 705.4 ms | 986.0 ms | **628.1 ms** | 2735 ms | 85 822 | 2.84 GB | 3.21 GB |
| C: INT8 + the fused GEGLU | 290.0 ms | 740.6 ms | 1038.6 ms | **624.4 ms** | 2741 ms | 85 822 | 2.85 GB | 3.21 GB |

512×512, batch 1:

| arm | forward | backward | eager step | **captured step** | host time | aten calls | weights | peak |
|---|---|---|---|---|---|---|---|---|
| A: bf16 | 116.5 ms | 270.8 ms | 366.9 ms | **190.0 ms** | 970 ms | 44 277 | 4.89 GB | 5.04 GB |
| B: INT8 linears | 264.1 ms | 669.9 ms | 929.5 ms | **210.3 ms** | 2541 ms | 85 822 | 2.84 GB | 3.00 GB |
| C: INT8 + the fused GEGLU | 262.2 ms | 669.6 ms | 951.3 ms | **209.3 ms** | 2506 ms | 85 822 | 2.85 GB | 3.00 GB |

The captured column is the number that matters (below). The run-to-run spread
on it is 5-10 %: the 768 bf16 arm measured 503-566 ms over three runs, the
512 bf16 arm 190-219 ms.

## What the numbers say

**1. The step is host-bound, and the capture is the lever.** 44 277 aten calls
and ~1 s of host time against a ~370 ms eager wall: the host is the
bottleneck, exactly as `next/manual/speed.md` says. The captured step at 768
(503-566 ms) lands inside the trainer's own measured range for replayed steps
(510-550 ms), so the two measurements agree.

**2. The INT8 path doubles the host cost, so it is only viable under the
capture.** 85 822 aten calls against 44 277, and 2.6× the host time: the
library's path launches more kernels (the rotate, the quantize, the
epilogue, the dequantize -- four extra per linear). Eagerly, the step becomes
**2.5-2.7× slower**. Under the capture that cost is gone, and what remains is
the device time.

**3. Under the capture the INT8 path costs about 10-12 % of the step and saves
42 % of the weight memory and 39 % of the peak.** 628 ms against 566 ms at
768; 210 against 190 at 512. The weights go from 4.89 GB to 2.84 GB (the
eligible 84 % of the parameters at one byte) and the peak from 5.23 GB to
3.21 GB. On a 12 GB card that is the difference between fitting and not.

**4. The library's fused GEGLU does not move the whole step.** In isolation it
is 1.2-1.5× faster than the unfused INT8 chain (measured in the library,
`xpu-kernels/docs/performance.md`), but in the step the FFN is a small part
of the work and the difference is inside the noise (1.10× against 1.11× at
768). It is worth having -- it is free, it is bit-identical, and it removes
three launches per FFN -- but it is not a lever on the step.

## Where the device time goes

torch's profiler with the XPU activity gives the whole table for one step
(512×512, the bf16 arm): **0.263 s of device time over 12 171 kernel
calls**.

| kernel | calls | device time | share |
|---|---|---|---|
| `gemm_kernel` | 2 199 | 59.4 ms | 23 % |
| `aten::mm` | 1 579 | 32.4 ms | 12 % |
| `aten::addmm` | 620 | 27.0 ms | 10 % |
| the convolutions (`convolution_overrideable`, `gen_conv`, `conv_reorder`, the backward) | 555 | 57.2 ms | **22 %** |
| the attention (the flash backward) | 280 | 14.2 ms | 5 % |
| the elementwise, norm, cat and the rest | ~6 900 | 65 ms | 25 % |

So: **the GEMMs are 45 % of the step's device time, the convolutions 22 %,
the attention 5 %.** Two things follow:

* the INT8 path works on the 45 %, and it costs 10-12 % of the step to save
  42 % of the weight memory -- the quantize/epilogue traffic eats most of
  what the faster GEMM wins;
* the **convolutions are 22 %** and the library's INT8 path does not reach
  them at all (the ConvRot is a linear format: a conv's contraction is
  spatial, so the activation's scale would be per channel, and that cannot
  fold into one epilogue scale). For an SDXL UNet they are 334.8 M of the
  2.57 B parameters -- the largest remaining piece of the model that the
  library cannot quantize.

## Two things about profiling on this card

**The XPU profiler works.** It registers PTI callbacks with the Level Zero
loader, prints a registration warning (`XPTI Callback Registration returned:
-2147467261`), and still produces a complete device-time table -- the
warning is noise, not a failure. `COMFY_PROFILE_FWD` works the same way.

**What is not survivable is an out-of-memory during the profiled step.** The
device is lost (`UR_RESULT_ERROR_DEVICE_LOST`), the profiler's cleanup then
fails with `PTI_ERROR_NOT_IMPLEMENTED`, and that exception escapes a
destructor -- so the process *dies* instead of raising, and the profiler's
table is lost with it. It looks like the profiler crashing; it is the
device being gone. The protection is the memory budget this project already
has (the checkpointing, the planner, `memory.vram_max_mb`), not a profiler
setting.

## What it means for the INT8 base

The trade is *about 11 % of the step for 2 GB of the peak*, and it only holds
with the graph capture on. The trainer's own `Int8 frozen base weights
through xpu-kernels` (commit `cbb1112`) is the natural place for it: the base
becomes 2.84 GB instead of 4.89 GB, which is what lets a bigger batch or a
higher resolution fit.

Two things to keep in mind when reading the numbers:

* the INT8 base is a *different function* from the bf16 base (0.9-1.2 %
  against the unquantized weight), so the loss curve has to be re-taken with
  it;
* the backward through an INT8 base is the `mode="int8"` path (the weight
  stays int8, `dy` is quantized): no copy of the weight, and about 1.3× the
  bf16 backward's time. The exact backward (`mode="dense"`) needs a
  full-precision copy of the weight, 4.3 GB for this base, which does not fit
  beside the activations.
