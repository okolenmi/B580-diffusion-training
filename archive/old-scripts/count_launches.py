#!/usr/bin/env python3
"""Count the kernel-launching ops in one SDXL LoRA training step, with NO GPU and NO memory.

Why: scripts/ measurements show the training process uses exactly 1.00 CPU cores even in fast
steps (CPU time per step == wall time per step), i.e. the GPU is fed by one thread issuing
launches. If that is the bottleneck, step time ~ (launches per step) x (CPU cost per launch),
and the lever is the launch count, not the shape. This counts launches exactly, on the `meta`
device (tensors have shape and dtype but no storage), through the project's own UNet and LoRA
layers, using a torch dispatch hook.

Run from the repository root:   python3 scripts/count_launches.py [--batch 2] [--latent 64]
Counted as a launch: every dispatched aten op that is not a view/alias and not a pure allocation.
That is a slight over-count of device kernels (some ops are metadata-only) and a slight under-count
of CPU work (Python module overhead, autograd bookkeeping), which is why the implied cost per
launch is printed next to the measured CPU time per step, for a sanity check.
"""
from __future__ import annotations
import argparse, collections, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ALLOC_ONLY = {"aten::empty", "aten::empty_like", "aten::empty_strided", "aten::new_empty", "aten::new_empty_strided",
              "aten::_local_scalar_dense", "aten::detach", "aten::alias", "aten::lift_fresh", "aten::is_same_size",
              # metadata-only (reshape without a copy): no kernel is launched
              "aten::_unsafe_view", "aten::view", "aten::_reshape_alias", "aten::_unsafe_index_put_"}


def main(argv=None) -> int:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from nodes.model.unet import UNetModel
    from nodes.model.unet_wrapper import ComfyUNetWrapper as W
    from nodes.model.lora import LoRAConfig, inject_lora_into_unet

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--latent", type=int, default=64, help="latent side (64 = 512 px)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--measured-cpu-s", type=float, default=1.06, help="measured CPU seconds per fast step, for the implied cost per launch")
    args = ap.parse_args(argv)

    class Counter(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.n = 0; self.by_op = collections.Counter(); self.total = 0
        def __torch_dispatch__(self, func, types, a=(), kw=None):
            self.total += 1
            name = str(func)
            ret = func._schema.returns
            is_view = bool(ret) and ret[0].alias_info is not None and not ret[0].alias_info.is_write
            if not is_view and func.name() not in ALLOC_ONLY:
                self.n += 1; self.by_op[func.name()] += 1
            return func(*a, **(kw or {}))

    def run(label, lora, checkpoint):
        if checkpoint:   # exactly what lora_injector.build_lora_injected_unet does for use_checkpoint=True (a process-wide patch: keep this variant last)
            from nodes.model.gradient_checkpointing import FrozenParamSafeCheckpointing
            FrozenParamSafeCheckpointing().apply()
        cfg = dict(W.SDXL_CONFIG); cfg["use_checkpoint"] = checkpoint
        prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                unet = UNetModel(**cfg)
        finally:
            torch.set_default_dtype(prev)
        unet.requires_grad_(False)
        n_adapted = 0
        if lora:
            reg = inject_lora_into_unet(unet, LoRAConfig(rank=args.rank, alpha=1.0))
            n_adapted = len(reg) if hasattr(reg, "__len__") else -1
        kw = dict(device="meta", dtype=torch.bfloat16)
        x = torch.randn(args.batch, 4, args.latent, args.latent, **kw).requires_grad_(not lora)
        t = torch.randint(0, 1000, (args.batch,), device="meta")
        ctx = torch.randn(args.batch, 77, cfg["context_dim"], **kw); y = torch.randn(args.batch, cfg["adm_in_channels"], **kw)
        c = Counter()
        with c:
            out = unet(x, t, ctx, y)
            fwd = c.n
            out.float().pow(2).mean().backward()
        return label, n_adapted, fwd, c.n - fwd, c.n, c.by_op

    rows = [run("base only (no LoRA), checkpointing off", False, False),
            run("LoRA, checkpointing off", True, False),
            run("LoRA, checkpointing ON  (the production setting)", True, True)]
    print(f"batch {args.batch}, latent {args.latent}x{args.latent}, rank {args.rank}; counts are kernel-launching ops\n")
    print(f"{'configuration':<52}{'adapted':>8}{'forward':>9}{'backward':>10}{'total':>9}")
    for label, na, f, b, tot, _ in rows:
        print(f"{label:<52}{na:>8}{f:>9}{b:>10}{tot:>9}")
    base, nock, prod = rows[0][4], rows[1][4], rows[2][4]
    print(f"\nLoRA adds {nock - base} launches over the frozen base ({(nock - base) / base:.0%} more); "
          f"checkpointing adds {prod - nock} more for the recompute.")
    print(f"production step: {prod} launches; at the measured {args.measured_cpu_s:.2f} s of CPU per step that is "
          f"{args.measured_cpu_s / prod * 1e6:.0f} us of CPU per launch")
    print("  (a PyTorch XPU eager launch costs on the order of tens of microseconds, so an implied cost in that range")
    print("   means the step IS launch-bound and the launch count is the lever; far above it means something else is slow)")
    print("\nlargest contributors in the production step:")
    for name, n in rows[2][5].most_common(10):
        print(f"  {n:6d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
