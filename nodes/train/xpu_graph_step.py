"""Single-shape XPU graph capture for the managed training step (L5.3).

The step is launch-bound (~22,000 kernel launches at ~48 us each), and the
feasibility probe (`shapes diversity problem 2/probe_xpu_graph.py`)
measured 2.04-2.08x replay vs production eager for forward+backward under
MATH/EFFICIENT SDPA (FLASH cannot capture -- SYCL limit). This integrates
that probe into the managed trainer: forward + masked weighted loss +
backward replay per bucket shape, optimizer step and everything else eager.

What is captured per (batch, H, W) shape, after 3 genuine eager sightings
(no wasted work -- the first sightings are real training steps):
model.forward on static buffers -> masked_per_sample_mse -> weighting by
an eagerly-hoisted per-sample vector -> scaled backward. Capture itself
executes once (real side effects), so grads are zeroed before capture and
again before the replay that counts.

What stays eager, and why:
- text encoding (per-batch, cache-served), noise/t/sigma sampling, the
  loss-weight vector and bucket-balance vector (tiny CPU + one small copy
  each -- building them inside the graph would bake data-dependent
  construction and host syncs into the capture);
- ZeroGrad's update_lr; grad zeroing itself moves into the runner because
  the optimizer's zero_grad() reassigns p.grad = None, which would leave a
  replay writing gradients into the captured (freed) pool addresses while
  p.grad stays None -- in-place zero_ keeps the addresses the graph owns;
- the optimizer step (hundreds of launches vs 22,000) and grad clipping
  (reads live .grad, boundary only);
- ProbePhase's own forwards, monitoring reads, mid-run LoRA saves.

Refusals (build time, loud -- each would silently corrupt or no-op):
- fused optimizer: its update fires inside backward hooks, which a graph
  cannot capture (same reason clipping is refused for fused);
- any Dropout(p > 0) in the model: replay would freeze the RNG;
- XPUGraph absent: loud one-time warning, everything eager.

Invariants the caller (ManagedLoRATrainerNode) upholds: the model is
registered non-sacrificable for a graph run (a mid-run offload would leave
replays reading freed memory -- silent, so it is structural, not checked
per step); p.grad is never reassigned while captured (only the runner
zeroes, in place); extras["pred"] is NOT populated on replay steps (the
prediction lives in the pool -- only LossPhase ever read it, and the graph
path replaces that phase).

Fallbacks (loud, never fatal): capture exception -> that shape eager
forever; more than max_shapes distinct shapes -> eager; batch size != the
configured full size (incomplete tails) -> eager on live tensors.
"""

from __future__ import annotations


class XPUGraphStepRunner:
    """Owns static buffers, per-shape graphs, and the eager/graph decision.

    model must expose .forward(xc, t, ctx, y) and .trainable_parameters()
    (the ComfyUNetTrainableModel surface the phases already use).
    loss_weighting is a LossWeighting; backward_scale is 1.0/grad_accum.
    device is a torch device string. Shapes are keyed by (batch, H, W):
    incomplete tails (keep_incomplete_batches) get their own keys, bounded
    by max_shapes -- with the default loader they are dropped, so keys are
    exactly the bucket shapes.
    """

    def __init__(self, model, loss_weighting, device, backward_scale=1.0,
                 bucket_balance=None,
                 warmup_sightings=3, max_shapes=4):
        import torch
        self._torch = torch
        self._model = model
        self._loss_weighting = loss_weighting
        self._device = torch.device(device)
        self._scale = float(backward_scale)
        self._bucket_balance = bucket_balance
        self._warmup_sightings = max(int(warmup_sightings), 1)
        self._max_shapes = max(int(max_shapes), 1)
        # Materialized once: replay writes grads to these parameters'
        # captured addresses, so the objects must be stable (the optimizer
        # updates param.data in place -- same object, same address).
        self._params = [p for p in model.trainable_parameters()
                        if p.requires_grad]
        # Capture is XPU-only by definition: on any other device every step
        # is the eager-live fallback (same math, no statics), which is also
        # what makes this class unit-testable on CPU.
        self._can_capture = torch.device(device).type == "xpu"
        self._graphs: dict = {}
        self._eager_shapes: set = set()
        self._pool = None
        self._sdpa_ctx = None
        self._no_graph_warned = False
        self._params_logged = False

    # -- build-time guard --------------------------------------------------

    def refuse_if_unsupported(self) -> None:
        """ValueError for configurations capture would silently corrupt."""
        import torch
        mods = None
        if hasattr(self._model, "modules"):
            mods = self._model.modules()
        else:
            # ComfyUNetTrainableModel is a façade, not an nn.Module: reach
            # the wrapped UNet through .raw, the documented escape hatch
            # for callers that need the full model rather than the
            # TrainableModel contract.
            raw = getattr(self._model, "raw", None)
            inner = getattr(raw, "model", None) if raw is not None else None
            if inner is not None and hasattr(inner, "modules"):
                mods = inner.modules()
        if mods is None:
            raise ValueError(
                "use_xpu_graph cannot verify a deterministic forward: the "
                f"model ({type(self._model).__name__}) exposes neither "
                ".modules() nor .raw.model.modules(). Capture without a "
                "dropout scan would risk a frozen RNG -- refusing instead.")
        bad = [f"{type(m).__name__}(p={m.p})"
               for m in mods
               if isinstance(m, torch.nn.Dropout) and float(m.p) > 0.0]
        if bad:
            raise ValueError(
                "use_xpu_graph needs deterministic forward: model contains "
                f"active dropout ({', '.join(bad[:4])}) whose RNG would freeze "
                "at capture. Use dropout=0 with graph capture, or no capture "
                "with dropout.")

    # -- per-window grad zeroing (in place: never reassign .grad) ----------

    def zero_window(self) -> None:
        for p in self._params:
            g = p.grad
            if g is not None:
                g.zero_()

    # -- the micro-step ----------------------------------------------------

    def step(self, *, micro, xc, t, ctx_emb, y, target, sigma, mask):
        """Forward+loss+backward for one micro-step.

        Caller zeroes the window first (zero_window) at micro == 0.
        Returns (loss_tensor, per_sample_tensor, how) where how is "replay",
        "eager-static" (a warming sighting -- real training step), or
        "eager-live" (fallback, no statics). Grads are in .grad either way.
        """
        B = int(xc.shape[0])
        key = (B, int(xc.shape[2]), int(xc.shape[3]))
        entry = self._graphs.get(key)
        if entry is not None and entry["graph"] is not None:
            self._copy_in(entry, xc, t, ctx_emb, y, target, sigma, mask)
            entry["graph"].replay()
            return entry["loss"], entry["per_sample"], "replay"
        if micro == 0:
            # No-op unless this shape is warm (3 eager sightings): capture
            # zeroes grads again after its throwaway execution, so the
            # replay below counts as this step.
            self.maybe_capture(key)
            entry = self._graphs.get(key)
            if entry is not None and entry["graph"] is not None:
                self._copy_in(entry, xc, t, ctx_emb, y, target, sigma, mask)
                entry["graph"].replay()
                return entry["loss"], entry["per_sample"], "replay"
        return self._first_sightings(key, xc, t, ctx_emb, y, target,
                                     sigma, mask)

    # -- internals ---------------------------------------------------------

    def _sdpa(self):
        if self._sdpa_ctx is None:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            # MATH measured fastest under capture (2.08x); EFFICIENT also
            # captures (2.04x); FLASH cannot (SYCL scratch-memory limit).
            # Held for the run's lifetime: capture and every replay must
            # dispatch identical kernels, and eager warmups on the same
            # buffers should too.
            self._sdpa_ctx = sdpa_kernel(SDPBackend.MATH)
            self._sdpa_ctx.__enter__()
        return self._sdpa_ctx

    def _alloc_statics(self, key, like):
        torch = self._torch
        B, H, W = key
        xc0, t0, ctx0, y0, tgt0, mask0 = like
        dev = self._device
        return {
            "key": key,
            "xc": torch.empty((B, 4, H, W), dtype=xc0.dtype, device=dev),
            "t": torch.empty((B,), dtype=t0.dtype, device=dev),
            "ctx": torch.empty_like(ctx0),
            "y": torch.empty_like(y0),
            "target": torch.empty_like(tgt0),
            "mask": (torch.empty_like(mask0) if mask0 is not None else None),
            # (B,) fp32, like LossPhase's own weights vector.
            "weights": torch.empty((B,), dtype=torch.float32, device=dev),
            "w_bucket": None,
            "sightings": 0,
            "graph": None,
            "loss": None,
            "per_sample": None,
        }

    def _weight_vector(self, sigma):
        # Eager, per step, like LossPhase: one small host sync in a step
        # that already syncs for reporting. (B,) fp32 on device.
        torch = self._torch
        sigmas = sigma.float().reshape(-1)
        return torch.tensor(
            [self._loss_weighting.weight(float(s)) for s in sigmas.tolist()],
            dtype=torch.float32, device=self._device)

    def _copy_in(self, entry, xc, t, ctx_emb, y, target, sigma, mask):
        entry["xc"].copy_(xc)
        entry["t"].copy_(t)
        entry["ctx"].copy_(ctx_emb)
        entry["y"].copy_(y)
        entry["target"].copy_(target)
        if entry["mask"] is not None:
            entry["mask"].copy_(mask)
        entry["weights"].copy_(self._weight_vector(sigma))
        if self._bucket_balance is not None:
            wb = self._bucket_balance.weight_for_t(
                t, dtype=self._torch.float32, device=self._device)
            if wb is not None:
                if entry["w_bucket"] is None:
                    entry["w_bucket"] = self._torch.empty_like(wb)
                entry["w_bucket"].copy_(wb)
                return
        entry["w_bucket"] = None

    def _forward_loss_backward(self, entry):
        """The captured computation, also used for eager warmups and the
        throwaway capture execution: forward on statics -> masked weighted
        loss -> scaled backward. Returns (loss, per_sample) live tensors."""
        from .loss import masked_per_sample_mse
        pred = self._model.forward(entry["xc"], entry["t"], entry["ctx"],
                                   entry["y"])
        per_sample = masked_per_sample_mse(pred, entry["target"],
                                           entry["mask"])
        w = entry["weights"].to(per_sample.dtype)
        if entry["w_bucket"] is not None:
            w = w * entry["w_bucket"].to(per_sample.dtype)
        loss = (per_sample * w).mean()
        if self._scale != 1.0:
            loss = loss * self._scale
        loss.backward()
        return loss, per_sample.detach()

    def _first_sightings(self, key, xc, t, ctx_emb, y, target, sigma, mask):
        torch = self._torch
        if (not self._can_capture
                or not hasattr(torch.xpu, "XPUGraph")
                or len(self._graphs) >= self._max_shapes
                or key in self._eager_shapes):
            if not self._no_graph_warned:
                reason = ("CPU device -- capture is XPU-only"
                          if self._can_capture is False and
                          hasattr(torch.xpu, "XPUGraph")
                          else "torch.xpu.XPUGraph absent")
                print(f"[xpu-graph] {reason} -- graph capture off, every "
                      f"step eager (loud once, never fatal)")
                self._no_graph_warned = True
            if (len(self._graphs) >= self._max_shapes
                    and key not in self._graphs
                    and hasattr(torch.xpu, "XPUGraph")):
                print(f"[xpu-graph] shape {key} past the {self._max_shapes}-"
                      f"shape cap -- eager (correct, just not replayed)")
                self._eager_shapes.add(key)
            return self._eager_live(xc, t, ctx_emb, y, target, sigma, mask)
        entry = self._graphs.get(key)
        if entry is None:
            self._sdpa()
            if self._pool is None:
                self._pool = torch.xpu.graph_pool_handle()
            entry = self._alloc_statics(
                key, (xc, t, ctx_emb, y, target, mask))
            self._graphs[key] = entry
            if not self._params_logged:
                print(f"[xpu-graph] {len(self._params)} trainable params, "
                      f"pool shared across shapes, MATH SDPA held")
                self._params_logged = True
        self._copy_in(entry, xc, t, ctx_emb, y, target, sigma, mask)
        # Genuine eager training step on the statics (grads accumulate for
        # real); maybe_capture fires once warm.
        loss, per_sample = self._forward_loss_backward(entry)
        entry["sightings"] += 1
        entry["loss"], entry["per_sample"] = loss, per_sample
        return loss, per_sample, "eager-static"

    def maybe_capture(self, key) -> str:
        """Capture the graph for key. Call only at micro == 0 with grads
        already zeroed: the capture execution has real side effects (it
        accumulates once), so grads are zeroed again before the replay
        that counts as this step. Returns what happened."""
        torch = self._torch
        entry = self._graphs.get(key)
        if (entry is None or entry["graph"] is not None
                or entry["sightings"] < self._warmup_sightings):
            return "not-ready"
        try:
            torch.xpu.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, pool=self._pool):
                loss_c, per_c = self._forward_loss_backward(entry)
            torch.xpu.synchronize()
        except Exception as exc:  # noqa: BLE001 -- the failure IS handled
            print(f"[xpu-graph] capture failed for shape {key} "
                  f"({type(exc).__name__}: {exc}) -- that shape stays eager")
            self._eager_shapes.add(key)
            del self._graphs[key]
            return "failed"
        entry["graph"] = graph
        # Handles MUST come from inside the capture context: warmup loss
        # objects alias eager memory, which replay never touches -- only
        # pool addresses created during capture are overwritten by replay.
        # Reporting the warmup objects freezes the reported loss per shape
        # (caught by the 200-step loss-parity acceptance, not by timing).
        entry["loss"], entry["per_sample"] = loss_c, per_c
        self.zero_window()
        print(f"[xpu-graph] captured shape {key} "
              f"({len(self._graphs)} graph(s) alive)")
        return "captured"

    def _eager_live(self, xc, t, ctx_emb, y, target, sigma, mask):
        """Fallback with no statics: the same math on live tensors, so a
        shape that never captures still trains on the identical expression."""
        from .loss import apply_loss_weighting, masked_per_sample_mse
        pred = self._model.forward(xc, t, ctx_emb, y)
        per_sample = masked_per_sample_mse(pred, target, mask)
        sigmas = sigma.float().reshape(-1)
        w_bucket = None
        if self._bucket_balance is not None:
            w_bucket = self._bucket_balance.weight_for_t(
                t, dtype=per_sample.dtype, device=per_sample.device)
        loss = apply_loss_weighting(per_sample, sigmas,
                                    self._loss_weighting, w_bucket)
        if self._scale != 1.0:
            loss = loss * self._scale
        loss.backward()
        return loss, per_sample.detach(), "eager-live"
