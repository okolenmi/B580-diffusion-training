# 73. Making attention capturable (brief for a separate project)

Status: **open for a fused kernel; a pure-torch stopgap (route D) is built.** Written to be handed to the `xpu-kernels` project or an upstream report.

## Problem
- The trainer's replayed step (`next/engine/step/graphed.py`) captures forward, loss and backward in one `torch.xpu.XPUGraph`. The default SDPA kernel
  on XPU (oneDNN flash, reached through `aten/native/mkldnn/xpu/Attention.cpp`) cannot be captured: the capture raises
  `The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the SYCL Graph extension`. By the error text the kernel
  takes its work-group local memory size at launch time (dynamic scratch), and the SYCL Graph extension does not record that for a kernel node.
- `SDPBackend.EFFICIENT_ATTENTION` does not exist on XPU (it warns and runs the math backend). So a capture runs the math attention: on the SDXL UNet
  (512, batch 1) forward+backward is 403 ms with flash and 549-559 ms with math. Capture removes the launch cost (about 22 k launches) but pays
  about 36% of the device time, so bf16 replay measured 1324 ms against 1134 ms eager (batch 4, 512).
- Everything else in the step already captures (the library's rope/adaLN kernels and the int8 path replay bit-identically).
  A capturable attention that is close to flash speed would give the step the roughly 2x the launch-bound model predicts.

## What the workload needs
- bf16 in and out, fp32 accumulation; no mask, no dropout, not causal; no grouped keys.
- Head dim 64. Heads 10 (640 channels) and 20 (1280 channels). Self-attention length per image: (res/16)^2 at 640 channels and (res/32)^2 at 1280, i.e. 1024
  and 256 at 512 px, 4096 and 1024 at 1024 px. Cross-attention: the same query lengths, 77 keys (not a multiple of any block size: needs masking).
- Batch 1 to 8, shapes fixed per captured graph. About 140 attention calls per forward (70 transformer blocks, self and cross).
- Backward needs `dq`, `dk`, `dv` (the adapters sit upstream of q, k, v). No weight gradients in attention. Deterministic backward is preferred:
  separate `dq` and `dk/dv` kernels instead of atomics, so `runtime.deterministic` stays honest.

## Routes, best first
1. **A capturable attention kernel in `xpu_kernels.triton`** (forward and backward, autograd wrapper). The package already has Triton kernels that capture
   (rope, adaLN) and the pattern for it: reference chain, verify once per process before first use (outside a capture), one environment switch, `status()`.
   - Check first, in an hour: does a Triton kernel that uses on-chip shared memory (a `tl.dot` with a layout conversion, a row reduction across a tile)
     capture? Triton allocates static shared memory, not the dynamic scratch the error names, so it should. If it does not, route A is dead and route C remains.
   - Start from Intel's Triton flash-attention implementation (intel-xpu-backend-for-triton, its tutorial and benchmarks) and re-tune for the B580 (Xe2); verify
     the cross-attention shape (77 keys) and tiny lengths (256).
   - Acceptance: matches `F.scaled_dot_product_attention` in fp32 reference (forward relative error ~1e-3 in bf16, backward comparable to autograd of the math
     backend); replays bit-identically to eager inside a capture (`debug/verify_graph_capture.py` pattern, forward and backward); device time at the shapes
     above reported against flash and math. Target: well under math (at least 2x faster) and within about 1.3x of flash; any result between flash and math
     already turns the captured step from a loss into a win.
   - Capture rules: no host synchronisation, no allocation except through the torch allocator (it comes from the graph's pool), the first call (gate) outside a capture.
2. **Upstream fix.** Report with a minimal reproducer: `torch.xpu.XPUGraph()`, capture `F.scaled_dot_product_attention(q, k, v)` on bf16 `[1, 10, 1024, 64]` tensors
   with the default backend. Components that could change: the SYCL Graph extension (record the scratch size property on a kernel node) or oneDNN/torch-xpu-ops
   (use a statically sized local memory when the stream is being captured). The trainer can only wait for this; check with each new torch release, since capture
   of the default kernel is then picked up by trying it before the math attention (`attentions` list in the graph factory).
3. **Piecewise capture in the trainer.** Capture the segments between attention calls as separate graphs and run each attention eagerly (flash) between them. It
   needs per-segment autograd handling in forward, recompute and backward (about 280 to 420 graph launches per step instead of 22 k kernel launches) and is a
   substantial piece of work; only worth it if A and B both fail.

## Route D, built in the trainer meanwhile (untested on the card)
`next/families/general/ldm_unet/attention_modes.py`: `trainer.graph_attention=math_bf16` (math backend with the fp32 conversion switched off) and `chunked_bf16`
(pure torch: bf16 matmuls over blocks of query rows, fp32 softmax, probabilities recomputed in backward, fp32 accumulation of dk and dv; deterministic, no
atomics). Both round the scores to bf16, so quality needs the loss comparison in TESTING 28. They do not replace route A: a fused kernel keeps the scores
in registers and still launches far fewer kernels. The `bench` attention rows rank every backend and mode at the SDXL shapes, with a capture attempt each.

## What would change in the trainer when it exists
`nodes/train/trainer.py` builds the capture's `attentions` list (today only the math backend); a capturable kernel goes first, and the existing per-shape timing
(`GraphedCompute._judge`) keeps a graph only when it beats the eager step, so a slower kernel cannot make the run worse.
