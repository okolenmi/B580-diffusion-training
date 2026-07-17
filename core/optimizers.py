"""Optimizers: CPUAdamW, ChunkedXPUAdafactor, FusedXPUAdafactor."""

import gc
import math

import torch
from tqdm import tqdm


def _move_optional_list(lst, device):
    """In-place move of a list that may contain None entries."""
    for i in range(len(lst)):
        if lst[i] is not None:
            lst[i] = lst[i].to(device, non_blocking=False)


# ---------------------------------------------------------------------------
# CPUAdamW — FP32 states on CPU, saved to disc as BF16
# ---------------------------------------------------------------------------

class CPUAdamW:
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2):
        # Unroll param groups (produced by radial LR strategy)
        if (isinstance(params, (list, tuple)) and len(params) > 0
                and isinstance(params[0], dict)):
            self.params = []
            for group in params:
                p_list = (group['params'] if isinstance(group['params'], (list, tuple))
                          else [group['params']])
                for p in p_list:
                    self.params.append(p)
        else:
            self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        # FP32 on CPU — no per-step cast overhead
        self.m = [torch.zeros_like(p.data, device="cpu", dtype=torch.float32)
                  for p in self.params]
        self.v = [torch.zeros_like(p.data, device="cpu", dtype=torch.float32)
                  for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, n_steps: int = 1):
        self.t += n_steps
        lr_t = self.lr * math.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.detach().to("cpu", dtype=torch.float32)
            w = p.data.to("cpu", dtype=torch.float32)
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            del g
            upd = self.m[i] / self.v[i].sqrt().add_(self.eps)
            if self.wd != 0:
                # AdamW: weight decay uses the base LR, not the bias-corrected lr_t.
                # Using lr_t would make decay near-zero at the start and grow over time,
                # which is wrong. Decoupled decay should be constant per the AdamW paper.
                w.mul_(1 - self.lr * self.wd)
            w.add_(upd, alpha=-lr_t)
            del upd
            p.data.copy_(w.to(device=p.device, dtype=p.dtype))
            del w

    def load_archived_states(self, archive: dict, device: str):
        """Restore states from a RAM archive into live tensors on device."""
        self.t = archive.get("t", self.t)
        def _to_dev(t):
            if t is None: return None
            return t.to(device, non_blocking=False)
        
        if "m" in archive:
            self.m = [_to_dev(t) for t in archive.get("m", [])]
            self.v = [_to_dev(t) for t in archive.get("v", [])]

    def free_states(self):
        del self.m, self.v
        gc.collect()

    def offload_states_to_cpu(self):
        """No-op: CPUAdamW states already live on CPU."""
        pass

    def reload_states_to_device(self, device=None):
        """No-op: CPUAdamW states already live on CPU."""
        pass


# ---------------------------------------------------------------------------
# ChunkedXPUAdafactor — GPU Adafactor with memory-pool & scratch buffer
# ---------------------------------------------------------------------------

class ChunkedXPUAdafactor:
    def __init__(self, params, lr=1e-5, eps=(1e-08, 1e-3),
                 clip_threshold=1.0, beta1=None,
                 weight_decay=1.0, scale_parameter=True, device="xpu"):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            self.params = []
            self.param_lr = []
            for group in params:
                p_list = group['params'] if isinstance(group['params'], (list, tuple)) else [group['params']]
                group_lr = group.get('lr', lr)
                for p in p_list:
                    self.params.append(p)
                    self.param_lr.append(group_lr)
        else:
            self.params = list(params)
            self.param_lr = [lr] * len(self.params)

        self.lr = lr
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.beta1 = beta1
        self.wd = weight_decay
        self.scale_parameter = scale_parameter
        self.device = device
        self.t = 0

        # States live on XPU (tiny after flattening)
        self.vr = [None] * len(self.params)
        self.vc = [None] * len(self.params)
        self.vs = [None] * len(self.params)
        self.exp_avg = [None] * len(self.params)

        self._scratch = None
        self._use_pool = False
        self._pool = None
        self._initialized = False
        self._tiny_vs = None

    def _init_scratch(self):
        if self._initialized:
            return
        dev = self.device
        max_numel = max((p.numel() for p in self.params), default=0)
        if max_numel == 0:
            self._initialized = True
            return

        try:
            self._pool = torch.xpu.MemPool()
            self._use_pool = True
        except Exception:
            self._use_pool = False

        try:
            if self._use_pool:
                with torch.xpu.use_mem_pool(self._pool):
                    self._scratch = torch.empty(max_numel, dtype=torch.float32, device=dev)
            else:
                self._scratch = torch.empty(max_numel, dtype=torch.float32, device=dev)
        except Exception:
            self._scratch = None

        self._initialized = True

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, n_steps: int = 1):
        # n_steps: number of training steps this optimizer call represents.
        # With grad_accum=K, step() is called once per K training steps, so
        # pass n_steps=K to keep self.t (and therefore rho_t) consistent with
        # FusedXPUAdafactor, which increments t every training step via hooks.
        self.t += n_steps
        self._init_scratch()
        dev = self.device

        rho_t = max(1e-4, 1.0 - (self.t ** -0.8))

        ctx = torch.xpu.use_mem_pool(self._pool) if self._use_pool else None
        if ctx:
            ctx.__enter__()

        try:
            # Fast path: batch all tiny params into a single flat FP32 tensor
            tiny_idx = [i for i, p in enumerate(self.params)
                        if p.grad is not None and p.numel() < 10_000]

            if tiny_idx:
                grads = torch.cat([self.params[i].grad.detach().float().flatten()
                                   for i in tiny_idx])
                ws = torch.cat([self.params[i].data.float().flatten()
                                for i in tiny_idx])

                rms_g = grads.norm() / (grads.numel() ** 0.5 + 1e-8)
                grads.mul_(float(torch.clamp(self.clip_threshold / rms_g, max=1.0)))

                g2 = grads.pow(2)

                # Shape validation: if param ordering changed between sessions
                if not hasattr(self, '_tiny_vs') or self._tiny_vs is None:
                    self._tiny_vs = g2.add(self.eps1)
                elif self._tiny_vs.numel() != g2.numel():
                    tqdm.write(f"    [Adafactor] _tiny_vs shape mismatch "
                               f"({self._tiny_vs.numel()} vs {g2.numel()}) — resetting.")
                    self._tiny_vs = g2.add(self.eps1)
                else:
                    if self._tiny_vs.device != grads.device:
                        self._tiny_vs = self._tiny_vs.to(grads.device)
                    self._tiny_vs.mul_(rho_t).add_(g2.add_(self.eps1), alpha=1.0 - rho_t)

                grads.div_(self._tiny_vs.sqrt().add_(self.eps1))

                offsets = [0]
                for i in tiny_idx:
                    offsets.append(offsets[-1] + self.params[i].numel())

                for k, i in enumerate(tiny_idx):
                    s, e = offsets[k], offsets[k + 1]
                    g_chunk = grads[s:e]
                    p = self.params[i]
                    n = p.numel()
                    lr = self.param_lr[i]

                    if self.scale_parameter:
                        p_rms = ws[s:e].norm() / (n ** 0.5 + 1e-8)
                        alpha_t = torch.clamp(p_rms.pow(2),
                                              min=max(self.eps1, self.eps2**2)) * lr
                    else:
                        alpha_t = max(self.eps1, 1.0) * lr

                    if self.wd != 0:
                        ws[s:e].mul_(1.0 - alpha_t * self.wd)

                    ws[s:e].add_(g_chunk, alpha=-float(alpha_t)
                                 if isinstance(alpha_t, float)
                                 else -alpha_t.item())
                    p.grad = None

                offset = 0
                for i in tiny_idx:
                    p = self.params[i]
                    n = p.numel()
                    p.data.copy_(ws[offset:offset + n].reshape(p.shape).to(dtype=p.dtype))
                    offset += n

            # Main path: medium and large params
            # Use a set for O(1) skip checks instead of O(n) list membership.
            tiny_set = set(tiny_idx)
            for i, p in enumerate(self.params):
                if p.grad is None or i in tiny_set:
                    continue

                lr = self.param_lr[i]
                n = p.numel()
                orig_shape = p.shape

                if self._scratch is not None and n <= self._scratch.numel():
                    g = self._scratch[:n].reshape(orig_shape)
                    g.copy_(p.grad.detach())
                else:
                    g = p.grad.detach().float()
                p.grad = None

                # Gradient clipping — convert scalar tensor to Python float to
                # avoid 0-dim × large-tensor broadcast syncs on XPU.
                rms_g = g.norm() / (n ** 0.5 + 1e-8)
                g.mul_(float(torch.clamp(self.clip_threshold / rms_g, max=1.0)))

                if self.scale_parameter:
                    p_rms = p.data.norm(dtype=torch.float32) / (n ** 0.5 + 1e-8)
                    # float() here: alpha_t used as scalar multiplier below —
                    # keeping it as a 0-dim tensor would cause broadcast syncs.
                    alpha_t = float(torch.clamp(p_rms.pow(2), min=max(self.eps1, self.eps2**2)) * lr)
                else:
                    alpha_t = max(self.eps1, 1.0) * lr  # plain Python float

                factored = g.dim() >= 2
                if factored:
                    g_view = g.reshape(g.shape[0], -1)
                    g2r = g_view.pow(2).mean(dim=1)
                    g2c = g_view.pow(2).mean(dim=0)

                    if self.vr[i] is None:
                        self.vr[i] = g2r.add_(self.eps1)
                        self.vc[i] = g2c.add_(self.eps1)
                    else:
                        if self.vr[i].device != dev:
                            self.vr[i], self.vc[i] = self.vr[i].to(dev), self.vc[i].to(dev)
                        self.vr[i].mul_(rho_t).add_(g2r.add_(self.eps1), alpha=1.0 - rho_t)
                        self.vc[i].mul_(rho_t).add_(g2c.add_(self.eps1), alpha=1.0 - rho_t)

                    # float(): vr_mean_sqrt is used as a scalar broadcast multiplier;
                    # keeping it as a 0-dim tensor triggers an XPU command-stream sync.
                    vr_mean_sqrt = float(self.vr[i].mean().add_(self.eps1).sqrt_())
                    vr_sqrt = self.vr[i].sqrt().add_(self.eps1)
                    vc_sqrt = self.vc[i].sqrt().add_(self.eps1)
                    g_view.div_(vr_sqrt.unsqueeze(1))
                    g_view.div_(vc_sqrt.unsqueeze(0))
                    g_view.mul_(vr_mean_sqrt)
                else:
                    g2 = g.pow(2)
                    if self.vs[i] is None:
                        self.vs[i] = g2.add_(self.eps1)
                    else:
                        if self.vs[i].device != dev:
                            self.vs[i] = self.vs[i].to(dev)
                        self.vs[i].mul_(rho_t).add_(g2.add_(self.eps1), alpha=1.0 - rho_t)
                    g.div_(self.vs[i].sqrt().add_(self.eps1))

                if self.beta1 is not None:
                    if self.exp_avg[i] is None:
                        self.exp_avg[i] = torch.zeros_like(g)
                    self.exp_avg[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                    g = self.exp_avg[i]

                if self.wd != 0:
                    p.data.mul_(1.0 - self.wd * alpha_t)

                # alpha_t is now a plain Python float — mul_ and sub_ stay async.
                p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))

        finally:
            if ctx:
                ctx.__exit__(None, None, None)

    def reset_states(self):
        """Reset second-moment states but keep parameter references and time step."""
        for i in range(len(self.params)):
            self.vr[i] = None
            self.vc[i] = None
            self.vs[i] = None
            self.exp_avg[i] = None
        self._tiny_vs = None
        print("    [Adafactor] Optimizer states reset.")

    def decay_states(self, factor: float):
        """Scale second-moment states by factor (0.0 = reset, 1.0 = no-op)."""
        if factor <= 0:
            return self.reset_states()
        
        for i in range(len(self.params)):
            if self.vr[i] is not None: self.vr[i].mul_(factor)
            if self.vc[i] is not None: self.vc[i].mul_(factor)
            if self.vs[i] is not None: self.vs[i].mul_(factor)
            if self.exp_avg[i] is not None: self.exp_avg[i].mul_(factor)
        
        if self._tiny_vs is not None:
            self._tiny_vs.mul_(factor)
        print(f"    [Adafactor] Optimizer states decayed by factor {factor:.2f}.")

    def load_archived_states(self, archive: dict, device: str):
        """Restore states from a RAM archive into live tensors on device."""
        self.t = archive.get("t", self.t)
        def _to_dev(t):
            if t is None: return None
            return t.to(device, non_blocking=False)
        self.vr = [_to_dev(t) for t in archive.get("vr", [])]
        self.vc = [_to_dev(t) for t in archive.get("vc", [])]
        self.vs = [_to_dev(t) for t in archive.get("vs", [])]
        self.exp_avg = [_to_dev(t) for t in archive.get("ea", [])]
        self._tiny_vs = _to_dev(archive.get("tiny"))

    def free_states(self):
        del self.vr, self.vc, self.vs, self.exp_avg
        self._scratch = None
        self._tiny_vs = None
        self._pool = None
        gc.collect()

    def offload_states_to_cpu(self):
        """Move second-moment states off the XPU/GPU to free device memory
        (used between cyclic-training cache rebuilds)."""
        _move_optional_list(self.vr, "cpu")
        _move_optional_list(self.vc, "cpu")
        _move_optional_list(self.vs, "cpu")
        _move_optional_list(self.exp_avg, "cpu")
        if self._tiny_vs is not None:
            self._tiny_vs = self._tiny_vs.to("cpu")

    def reload_states_to_device(self, device=None):
        dev = device if device is not None else self.device
        _move_optional_list(self.vr, dev)
        _move_optional_list(self.vc, dev)
        _move_optional_list(self.vs, dev)
        _move_optional_list(self.exp_avg, dev)
        if self._tiny_vs is not None:
            self._tiny_vs = self._tiny_vs.to(dev)


# ---------------------------------------------------------------------------
# ChunkedXPUCAME — CAME optimizer (Luo et al., ACL 2023), ported to this
# codebase's XPU-resident, per-parameter-streaming style.
# ---------------------------------------------------------------------------

class ChunkedXPUCAME:
    """CAME (Confidence-guided Adaptive Memory Efficient Optimization).

    Algorithm faithfully follows the official reference implementation
    (github.com/yangluo7/CAME, came_pytorch/CAME.py) -- Luo et al., "CAME:
    Confidence-guided Adaptive Memory Efficient Optimization", ACL 2023.
    Verified against that source directly rather than reconstructed from the
    paper alone.

    Same interface as ChunkedXPUAdafactor (params/param_lr lists, step(n_steps=),
    zero_grad(), offload/reload/decay/reset hooks for cyclic training) so it
    drops into the same call sites. Unlike Adafactor, CAME keeps:
      - the same factorized row/col second-moment of grad^2 (vr/vc/vs -- same
        names as ChunkedXPUAdafactor so the existing checkpoint save/load code,
        which is generic on these attribute names, works unmodified)
      - a full-size momentum buffer per parameter (exp_avg) -- CAME's one
        genuinely bigger buffer vs Adafactor. For LoRA training this is
        proportional to adapter size only (not the base model), so the added
        VRAM is small in absolute terms.
      - a second factorized row/col pair (res_r/res_c) tracking how much
        each step's update disagrees with its own momentum -- the
        "confidence-guided" term that gives CAME faster, more stable
        convergence than plain Adafactor at nearly the same memory cost.

    beta1/beta2/beta3 are fixed EMA decay rates as published (0.9, 0.999,
    0.9999 by default) -- not reinterpreted to use Adafactor's time-varying
    rho_t schedule. No adaptive per-parameter LR scaling (Adafactor's
    scale_parameter) -- the reference algorithm doesn't have one; lr is used
    directly, same as AdamW.
    """

    def __init__(self, params, lr=1e-4, eps=(1e-30, 1e-16),
                 clip_threshold=1.0, betas=(0.9, 0.999, 0.9999),
                 weight_decay=0.0, device="xpu"):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            self.params = []
            self.param_lr = []
            for group in params:
                p_list = group['params'] if isinstance(group['params'], (list, tuple)) else [group['params']]
                group_lr = group.get('lr', lr)
                for p in p_list:
                    self.params.append(p)
                    self.param_lr.append(group_lr)
        else:
            self.params = list(params)
            self.param_lr = [lr] * len(self.params)

        self.lr = lr
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.beta1, self.beta2, self.beta3 = betas
        self.wd = weight_decay
        self.device = device
        self.t = 0

        n = len(self.params)
        self.vr = [None] * n
        self.vc = [None] * n
        self.vs = [None] * n
        self.exp_avg = [None] * n
        self.res_r = [None] * n
        self.res_c = [None] * n

        self._scratch = None
        self._use_pool = False
        self._pool = None
        self._initialized = False

    def _init_scratch(self):
        if self._initialized:
            return
        dev = self.device
        max_numel = max((p.numel() for p in self.params), default=0)
        if max_numel == 0:
            self._initialized = True
            return
        try:
            self._pool = torch.xpu.MemPool()
            self._use_pool = True
        except Exception:
            self._use_pool = False
        try:
            if self._use_pool:
                with torch.xpu.use_mem_pool(self._pool):
                    self._scratch = torch.empty(max_numel, dtype=torch.float32, device=dev)
            else:
                self._scratch = torch.empty(max_numel, dtype=torch.float32, device=dev)
        except Exception:
            self._scratch = None
        self._initialized = True

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def _factored_normalize(self, sq, row_buf, col_buf, beta, dev):
        """EMA-update row_buf/col_buf (in place) from sq=x^2+eps, return
        (row_buf, col_buf, row_sqrt, col_sqrt, row_mean_sqrt) such that
        dividing some tensor by row_sqrt[:,None] and col_sqrt[None,:], then
        multiplying by row_mean_sqrt, approximates dividing by sqrt of the
        factored EMA of sq. Same formula as ChunkedXPUAdafactor's vr/vc
        normalization -- reused here for both the grad^2 term and CAME's
        confidence term (only the beta and which buffers get passed differ).

        Zero-initializes on first call and always applies the beta blend
        (matching the reference implementation's torch.zeros_like init +
        unconditional mul_(beta).add_(new, alpha=1-beta) exactly) rather than
        initializing directly to the first observed value the way
        ChunkedXPUAdafactor's rho_t-scheduled vr/vc does. CAME uses fixed
        betas with no bias correction, so this zero-init is a deliberate,
        published "cold start" -- skipping it changes early-step behavior
        (verified this by numerically diffing against the reference: skipping
        it under-normalizes the first update by ~30x at beta2=0.999).
        """
        row_new = sq.mean(dim=1)
        col_new = sq.mean(dim=0)
        if row_buf is None:
            row_buf = torch.zeros_like(row_new)
            col_buf = torch.zeros_like(col_new)
        elif row_buf.device != dev:
            row_buf, col_buf = row_buf.to(dev), col_buf.to(dev)
        row_buf.mul_(beta).add_(row_new, alpha=1.0 - beta)
        col_buf.mul_(beta).add_(col_new, alpha=1.0 - beta)
        row_mean_sqrt = float(row_buf.mean().add(self.eps1).sqrt())
        row_sqrt = row_buf.sqrt().add(self.eps1)
        col_sqrt = col_buf.sqrt().add(self.eps1)
        return row_buf, col_buf, row_sqrt, col_sqrt, row_mean_sqrt

    def step(self, n_steps: int = 1):
        self.t += n_steps
        self._init_scratch()
        dev = self.device

        ctx = torch.xpu.use_mem_pool(self._pool) if self._use_pool else None
        if ctx:
            ctx.__enter__()

        try:
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                lr = self.param_lr[i]
                n = p.numel()
                orig_shape = p.shape

                if self._scratch is not None and n <= self._scratch.numel():
                    g = self._scratch[:n].reshape(orig_shape)
                    g.copy_(p.grad.detach())
                else:
                    g = p.grad.detach().float()
                p.grad = None

                factored = g.dim() >= 2
                if factored:
                    g_view = g.reshape(g.shape[0], -1)
                    g2 = g_view.pow(2).add(self.eps1)

                    self.vr[i], self.vc[i], vr_sqrt, vc_sqrt, vr_mean_sqrt = \
                        self._factored_normalize(g2, self.vr[i], self.vc[i], self.beta2, dev)

                    # Normalized gradient: grad * approx_sq_grad(row, col)
                    g_view.div_(vr_sqrt.unsqueeze(1))
                    g_view.div_(vc_sqrt.unsqueeze(0))
                    g_view.mul_(vr_mean_sqrt)
                else:
                    g2 = g.pow(2).add(self.eps1)
                    if self.vs[i] is None:
                        self.vs[i] = torch.zeros_like(g2)
                    elif self.vs[i].device != dev:
                        self.vs[i] = self.vs[i].to(dev)
                    self.vs[i].mul_(self.beta2).add_(g2, alpha=1.0 - self.beta2)
                    g.div_(self.vs[i].sqrt().add(self.eps1))

                # Clip the normalized update by its own RMS -- CAME's own
                # clipping (distinct from Adafactor's raw-gradient clipping),
                # applied to `g` after normalization, matching the reference.
                rms_g = g.norm() / (n ** 0.5 + 1e-8)
                clip_div = float(torch.clamp(rms_g / self.clip_threshold, min=1.0))
                if clip_div != 1.0:
                    g.div_(clip_div)

                # Momentum of the (clipped, normalized) update.
                ea = self.exp_avg[i]
                if ea is None:
                    ea = torch.zeros_like(g)
                elif ea.device != dev:
                    ea = ea.to(dev)
                ea.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                self.exp_avg[i] = ea

                if factored:
                    # Confidence-guided term: squared disagreement between
                    # this step's normalized update and its own momentum.
                    # High disagreement -> lower confidence -> smaller
                    # effective step. Reference implementation only computes
                    # this for factored (2D+) params -- skipped entirely for
                    # 1D params below, so no point computing it here either.
                    res = (g - ea).pow_(2).add_(self.eps2)
                    res_view = res.reshape(res.shape[0], -1)
                    self.res_r[i], self.res_c[i], rr_sqrt, rc_sqrt, rr_mean_sqrt = \
                        self._factored_normalize(res_view, self.res_r[i], self.res_c[i],
                                                 self.beta3, dev)
                    update = ea.reshape(res.shape[0], -1).clone()
                    update.div_(rr_sqrt.unsqueeze(1))
                    update.div_(rc_sqrt.unsqueeze(0))
                    update.mul_(rr_mean_sqrt)
                    update = update.reshape(orig_shape)
                else:
                    # Reference implementation does not apply the confidence
                    # term for non-factored (1D) params at all -- it uses the
                    # momentum directly. Not something LoRA hits (A/B are
                    # always 2D) but kept faithful for the dense-finetune case.
                    update = ea.clone()

                if self.wd != 0:
                    p.data.add_(p.data, alpha=-self.wd * lr)

                p.data.add_(update.to(dtype=p.dtype), alpha=-lr)

        finally:
            if ctx:
                ctx.__exit__(None, None, None)

    def reset_states(self):
        for i in range(len(self.params)):
            self.vr[i] = None; self.vc[i] = None; self.vs[i] = None
            self.exp_avg[i] = None
            self.res_r[i] = None; self.res_c[i] = None
        print("    [CAME] Optimizer states reset.")

    def decay_states(self, factor: float):
        if factor <= 0:
            return self.reset_states()
        for i in range(len(self.params)):
            for lst in (self.vr, self.vc, self.vs, self.exp_avg,
                        self.res_r, self.res_c):
                if lst[i] is not None:
                    lst[i].mul_(factor)
        print(f"    [CAME] Optimizer states decayed by factor {factor:.2f}.")

    def free_states(self):
        del self.vr, self.vc, self.vs, self.exp_avg, self.res_r, self.res_c
        self._scratch = None
        self._pool = None
        gc.collect()

    def offload_states_to_cpu(self):
        for lst in (self.vr, self.vc, self.vs, self.exp_avg,
                    self.res_r, self.res_c):
            _move_optional_list(lst, "cpu")

    def reload_states_to_device(self, device=None):
        dev = device if device is not None else self.device
        for lst in (self.vr, self.vc, self.vs, self.exp_avg,
                    self.res_r, self.res_c):
            _move_optional_list(lst, dev)


# ---------------------------------------------------------------------------
# ForeachXPUAdafactor — vectorized update using torch._foreach ops
# ---------------------------------------------------------------------------

class ForeachXPUAdafactor:
    """XPU Adafactor optimized for hundreds of small parameters (LoRA).
    
    Uses torch._foreach_* operations to batch multiple parameter updates
    into single kernels, drastically reducing Python overhead and CPU
    bottlenecking.
    """
    def __init__(self, params, lr=1e-5, eps=(1e-08, 1e-3),
                 clip_threshold=1.0, beta1=None,
                 weight_decay=1.0, scale_parameter=True, device="xpu"):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            self.params = []
            self.param_lr = []
            for group in params:
                p_list = group['params'] if isinstance(group['params'], (list, tuple)) else [group['params']]
                group_lr = group.get('lr', lr)
                for p in p_list:
                    self.params.append(p)
                    self.param_lr.append(group_lr)
        else:
            self.params = list(params)
            self.param_lr = [lr] * len(self.params)

        self.lr = lr
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.beta1 = beta1
        self.wd = weight_decay
        self.scale_parameter = scale_parameter
        self.device = device
        self.t = 0

        # State storage (tensors on device)
        self.vr = [None] * len(self.params)
        self.vc = [None] * len(self.params)
        self.vs = [None] * len(self.params)
        self.exp_avg = [None] * len(self.params)
        
        # Pre-calculated constants
        self._min_clamp = max(self.eps1, self.eps2**2)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, n_steps: int = 1):
        self.t += n_steps
        dev = self.device
        rho_t = max(1e-4, 1.0 - (self.t ** -0.8))
        
        # We group parameters by their update path to use foreach ops
        # Path 1: Factored (dim >= 2)
        # Path 2: Unfactored (dim < 2)
        factored_indices = []
        unfactored_indices = []
        
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            if p.dim() >= 2: factored_indices.append(i)
            else: unfactored_indices.append(i)

        # 1. Factored updates
        if factored_indices:
            self._step_factored(factored_indices, rho_t, dev)
            
        # 2. Unfactored updates
        if unfactored_indices:
            self._step_unfactored(unfactored_indices, rho_t, dev)

    def _step_factored(self, indices, rho_t, dev):
        # Implementation using foreach where possible, or optimized loops
        # for complex factoring logic.
        for i in indices:
            p = self.params[i]
            lr = self.param_lr[i]
            n = p.numel()
            g = p.grad.detach().float()
            p.grad = None

            # Gradient clipping — keep as on-device scalar tensor to avoid host-device sync.
            g_rms = g.norm().mul_(n ** -0.5)
            clip_t = (self.clip_threshold / (g_rms + 1e-8)).clamp_(max=1.0)
            g.mul_(clip_t)

            if self.scale_parameter:
                # keep alpha_t as on-device scalar tensor
                p_rms_sq = p.data.norm(dtype=torch.float32).pow_(2).div_(n)
                alpha_t = p_rms_sq.clamp_(min=self._min_clamp).mul_(lr)
            else:
                alpha_t = lr

            g_view = g.reshape(g.shape[0], -1)
            g2r = g_view.pow(2).mean(dim=1)
            g2c = g_view.pow(2).mean(dim=0)

            if self.vr[i] is None:
                self.vr[i] = g2r.add_(self.eps1)
                self.vc[i] = g2c.add_(self.eps1)
            else:
                self.vr[i].mul_(rho_t).add_(g2r.add_(self.eps1), alpha=1.0 - rho_t)
                self.vc[i].mul_(rho_t).add_(g2c.add_(self.eps1), alpha=1.0 - rho_t)

            # All these remain on-device scalar tensors — no host-device sync.
            vr_mean_sqrt = self.vr[i].mean().add_(self.eps1).sqrt_()
            vr_sqrt = self.vr[i].sqrt().add_(self.eps1)
            vc_sqrt = self.vc[i].sqrt().add_(self.eps1)
            
            g_view.div_(vr_sqrt.unsqueeze(1))
            g_view.div_(vc_sqrt.unsqueeze(0))
            g_view.mul_(vr_mean_sqrt)

            if self.beta1 is not None:
                if self.exp_avg[i] is None:
                    self.exp_avg[i] = torch.zeros_like(g)
                self.exp_avg[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                g = self.exp_avg[i]

            if self.wd != 0:
                p.data.mul_(1.0 - alpha_t * self.wd)
            
            # Optimization: avoid .item() sync by doing tensor multiplication
            p.data.add_(g.to(dtype=p.dtype).mul(alpha_t), alpha=-1)

    def _step_unfactored(self, indices, rho_t, dev):
        # Unfactored path is perfect for foreach
        ps = [self.params[i].data for i in indices]
        gs = [self.params[i].grad.detach().float() for i in indices]
        for i in indices: self.params[i].grad = None
        
        # 1. Clipping
        norms = torch._foreach_norm(gs)
        # sqrts are scalars (CPU)
        sqrts = [math.sqrt(p.numel()) + 1e-8 for p in ps]
        # rms_gs are tensors (XPU)
        rms_gs = [n / s for n, s in zip(norms, sqrts)]
        # clips are tensors (XPU)
        clips = [(self.clip_threshold / r).clamp_(max=1.0) for r in rms_gs]
        torch._foreach_mul_(gs, clips)
        
        # 2. Alpha calculation
        if self.scale_parameter:
            p_norms = torch._foreach_norm(ps)
            # p_rms_sq are tensors (XPU)
            alphas = [(n**2 / s**2).clamp_(min=self._min_clamp).mul_(self.param_lr[idx])
                      for n, s, idx in zip(p_norms, sqrts, indices)]
        else:
            alphas = [self.param_lr[idx] for idx in indices]
            
        # 3. State update
        g2s = torch._foreach_pow(gs, 2)
        torch._foreach_add_(g2s, self.eps1)
        
        for k, i in enumerate(indices):
            if self.vs[i] is None:
                self.vs[i] = g2s[k]
            else:
                self.vs[i].mul_(rho_t).add_(g2s[k], alpha=1.0 - rho_t)
        
        # 4. Gradient normalization
        vs_sqrts = torch._foreach_sqrt([self.vs[i] for i in indices])
        torch._foreach_add_(vs_sqrts, self.eps1)
        torch._foreach_div_(gs, vs_sqrts)
        
        # 5. Momentum
        if self.beta1 is not None:
            eas = []
            for i in indices:
                if self.exp_avg[i] is None: self.exp_avg[i] = torch.zeros_like(self.params[i], dtype=torch.float32)
                eas.append(self.exp_avg[i])
            torch._foreach_mul_(eas, self.beta1)
            torch._foreach_add_(eas, gs, alpha=1.0 - self.beta1)
            gs = eas
            
        # 6. Weight decay and Apply
        for k, i in enumerate(indices):
            p = self.params[i]
            a = alphas[k]
            if self.wd != 0:
                p.data.mul_(1.0 - a * self.wd)
            
            # Optimization: avoid .item() sync by doing tensor multiplication
            p.data.add_(gs[k].to(dtype=p.dtype).mul(a), alpha=-1)

    def decay_states(self, factor: float):
        if factor <= 0:
            for i in range(len(self.params)):
                self.vr[i] = self.vc[i] = self.vs[i] = self.exp_avg[i] = None
            return
        
        states = [s for s in (self.vr + self.vc + self.vs + self.exp_avg) if s is not None]
        if states:
            torch._foreach_mul_(states, factor)

    def load_archived_states(self, archive: dict, device: str):
        self.t = archive.get("t", self.t)
        def _to_dev(t): return t.to(device, non_blocking=False) if t is not None else None
        self.vr = [_to_dev(t) for t in archive.get("vr", [])]
        self.vc = [_to_dev(t) for t in archive.get("vc", [])]
        self.vs = [_to_dev(t) for t in archive.get("vs", [])]
        self.exp_avg = [_to_dev(t) for t in archive.get("ea", [])]

    def free_states(self):
        del self.vr, self.vc, self.vs, self.exp_avg
        gc.collect()

    def offload_states_to_cpu(self):
        """Move second-moment states off the XPU/GPU to free device memory
        (used between cyclic-training cache rebuilds)."""
        _move_optional_list(self.vr, "cpu")
        _move_optional_list(self.vc, "cpu")
        _move_optional_list(self.vs, "cpu")
        _move_optional_list(self.exp_avg, "cpu")

    def reload_states_to_device(self, device=None):
        dev = device if device is not None else self.device
        _move_optional_list(self.vr, dev)
        _move_optional_list(self.vc, dev)
        _move_optional_list(self.vs, dev)
        _move_optional_list(self.exp_avg, dev)


# ---------------------------------------------------------------------------
# FusedXPUAdafactor — fused into backward pass via grad hook
# ---------------------------------------------------------------------------

class FusedXPUAdafactor:
    TINY_NUMEL = 10_000

    def __init__(self, params, lr=1e-5, eps=(1e-08, 1e-3),
                 clip_threshold=1.0, beta1=None,
                 weight_decay=1.0, scale_parameter=True, device="xpu"):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            self.params = []
            self.param_lr = []
            for group in params:
                p_list = group["params"] if isinstance(group["params"], (list, tuple)) else [group["params"]]
                group_lr = group.get("lr", lr)
                for p in p_list:
                    self.params.append(p)
                    self.param_lr.append(group_lr)
        else:
            self.params = list(params)
            self.param_lr = [lr] * len(self.params)

        self.lr = lr
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.beta1 = beta1
        self.wd = weight_decay
        self.scale_parameter = scale_parameter
        self.device = device
        self.t = 0

        self._param_to_idx = {id(p): i for i, p in enumerate(self.params)}
        self.vr = [None] * len(self.params)
        self.vc = [None] * len(self.params)
        self.vs = [None] * len(self.params)
        self.exp_avg = [None] * len(self.params)

        self._hooks = []
        self._tiny_vs_map: dict = {}  # per-param second moments for tiny params
        self._in_backward = False
        self._rho_t = 1e-4

        # Internal accumulation support for multi-pass steps (distillation)
        self.sub_steps_required = 1
        self._current_sub_step = 0

        # Precompute constants
        self._eps2_sq = self.eps2 ** 2
        self._min_clamp = max(self.eps1, self._eps2_sq)

        # Cache per-param LR as plain Python floats for use inside hooks
        # (avoids any tensor allocation/indexing on the hot path).
        self._param_lr_f = list(self.param_lr)  # plain float list, updated by update_lr()

    def _update_param(self, p):
        """Per-parameter Adafactor update via backward hook."""
        i = self._param_to_idx.get(id(p))
        if i is None or p.grad is None:
            return

        # Handle logical step tracking across multiple backward() calls
        if not self._in_backward:
            self._in_backward = True
            self._current_sub_step += 1
            # Rho (time-decay) only updates when we actually commit an update
            if self._current_sub_step >= self.sub_steps_required:
                self.t += 1
                self._rho_t = max(1e-4, 1.0 - (self.t ** -0.8))

        # ACCUMULATION PASS: If we expect more passes, don't update weights yet.
        # Autograd automatically adds gradients to p.grad. We just exit early.
        if self._current_sub_step < self.sub_steps_required:
            return

        # UPDATE PASS: We have the combined gradient from all distillation passes.
        rho_t = self._rho_t
        lr = self._param_lr_f[i]
        n = p.numel()

        g = p.grad.detach().float()
        p.grad = None # Memory saving: clear gradient immediately after conversion to float

        # Gradient clipping
        g_rms = g.norm().mul_(n ** -0.5)
        clip_t = (self.clip_threshold / (g_rms + 1e-8)).clamp_(max=1.0)
        g.mul_(clip_t)

        if self.scale_parameter:
            p_rms_sq = p.data.norm(dtype=torch.float32).pow_(2).div_(n)
            alpha_t = p_rms_sq.clamp_(min=self._min_clamp).mul_(lr)
        else:
            alpha_t = lr

        if n < self.TINY_NUMEL:
            g2 = g.pow(2)
            if i not in self._tiny_vs_map:
                self._tiny_vs_map[i] = g2.add(self.eps1)
            else:
                tv = self._tiny_vs_map[i]
                if tv.device != g.device:
                    tv = tv.to(g.device)
                    self._tiny_vs_map[i] = tv
                tv.mul_(rho_t).add_(g2.add_(self.eps1), alpha=1.0 - rho_t)
            g.div_(self._tiny_vs_map[i].sqrt().add_(self.eps1))
        else:
            factored = g.dim() >= 2
            if factored:
                g_view = g.reshape(g.shape[0], -1)
                g2r = g_view.pow(2).mean(dim=1)
                g2c = g_view.pow(2).mean(dim=0)
                if self.vr[i] is None:
                    self.vr[i] = g2r.add_(self.eps1)
                    self.vc[i] = g2c.add_(self.eps1)
                else:
                    if self.vr[i].device != p.device:
                        self.vr[i] = self.vr[i].to(p.device)
                        self.vc[i] = self.vc[i].to(p.device)
                    self.vr[i].mul_(rho_t).add_(g2r.add_(self.eps1), alpha=1.0 - rho_t)
                    self.vc[i].mul_(rho_t).add_(g2c.add_(self.eps1), alpha=1.0 - rho_t)
                vr_mean_sqrt = self.vr[i].mean().add_(self.eps1).sqrt_()
                vr_sqrt = self.vr[i].sqrt().add_(self.eps1)
                vc_sqrt = self.vc[i].sqrt().add_(self.eps1)
                g_view.div_(vr_sqrt.unsqueeze(1))
                g_view.div_(vc_sqrt.unsqueeze(0))
                g_view.mul_(vr_mean_sqrt)
            else:
                g2 = g.pow(2)
                if self.vs[i] is None:
                    self.vs[i] = g2.add_(self.eps1)
                else:
                    if self.vs[i].device != p.device:
                        self.vs[i] = self.vs[i].to(p.device)
                    self.vs[i].mul_(rho_t).add_(g2.add_(self.eps1), alpha=1.0 - rho_t)
                g.div_(self.vs[i].sqrt().add_(self.eps1))

        if self.beta1 is not None:
            if self.exp_avg[i] is None:
                self.exp_avg[i] = torch.zeros_like(g)
            self.exp_avg[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            g = self.exp_avg[i]

        if self.wd != 0:
            p.data.mul_(1.0 - self.wd * alpha_t)

        p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))

    def register_hooks(self):
        self._in_backward = False
        for p in self.params:
            if p.requires_grad:
                h = p.register_post_accumulate_grad_hook(self._update_param)
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def begin_step(self, sub_steps: int = 1):
        """Reset flags for a new logical training step.
        sub_steps: how many physical backward() calls form one update (e.g. 2 for distillation).
        """
        self._in_backward = False
        self.sub_steps_required = sub_steps
        self._current_sub_step = 0

    def prepare_next_pass(self):
        """Reset per-backward flag so the next backward() increments current_sub_step."""
        self._in_backward = False

    def step(self):
        pass

    def zero_grad(self):
        pass

    def update_lr(self, new_lr):
        if hasattr(self, "_radial_mults"):
            self.param_lr    = [m * new_lr for m in self._radial_mults]
            self._param_lr_f = [m * new_lr for m in self._radial_mults]
        else:
            self.param_lr    = [new_lr] * len(self.params)
            self._param_lr_f = [new_lr] * len(self.params)
        self.lr = new_lr

    def reset_states(self):
        for i in range(len(self.params)):
            self.vr[i] = None
            self.vc[i] = None
            self.vs[i] = None
            self.exp_avg[i] = None
        self._tiny_vs_map = {}
        print("    [FusedAdafactor] Optimizer states reset.")

    def decay_states(self, factor: float):
        if factor <= 0:
            return self.reset_states()
        for i in range(len(self.params)):
            if self.vr[i] is not None: self.vr[i].mul_(factor)
            if self.vc[i] is not None: self.vc[i].mul_(factor)
            if self.vs[i] is not None: self.vs[i].mul_(factor)
            if self.exp_avg[i] is not None: self.exp_avg[i].mul_(factor)
        for v in getattr(self, "_tiny_vs_map", {}).values():
            v.mul_(factor)
        print(f"    [FusedAdafactor] Optimizer states decayed by factor {factor:.2f}.")

    def load_archived_states(self, archive: dict, device: str):
        self.t = archive.get("t", self.t)
        def _to_dev(t):
            if t is None: return None
            return t.to(device, non_blocking=False)
        self.vr = [_to_dev(t) for t in archive.get("vr", [])]
        self.vc = [_to_dev(t) for t in archive.get("vc", [])]
        self.vs = [_to_dev(t) for t in archive.get("vs", [])]
        self.exp_avg = [_to_dev(t) for t in archive.get("ea", [])]
        tiny_map = archive.get("tiny_map", {})
        self._tiny_vs_map = {int(k): _to_dev(v) for k, v in tiny_map.items()}
        self._param_lr_f = list(self.param_lr)

    def free_states(self):
        self.remove_hooks()
        del self.vr, self.vc, self.vs, self.exp_avg
        self._tiny_vs_map = {}
        gc.collect()

    def offload_states_to_cpu(self):
        """Move second-moment states off the XPU/GPU to free device memory
        (used between cyclic-training cache rebuilds). Backward hooks are
        left registered — they are harmless while no backward() is running,
        which is the only time this method is called."""
        _move_optional_list(self.vr, "cpu")
        _move_optional_list(self.vc, "cpu")
        _move_optional_list(self.vs, "cpu")
        _move_optional_list(self.exp_avg, "cpu")
        self._tiny_vs_map = {k: v.to("cpu") for k, v in self._tiny_vs_map.items()}

    def reload_states_to_device(self, device=None):
        dev = device if device is not None else self.device
        _move_optional_list(self.vr, dev)
        _move_optional_list(self.vc, dev)
        _move_optional_list(self.vs, dev)
        _move_optional_list(self.exp_avg, dev)
        self._tiny_vs_map = {k: v.to(dev) for k, v in self._tiny_vs_map.items()}
