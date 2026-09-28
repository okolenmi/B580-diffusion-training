"""Real-hardware validation harness for the entries in
docs/known-issues/pending-testing.md (and related VRAM questions).

Not a smoke test: this loads the real SDXL checkpoint, the real dataset,
and trains on the real XPU. It exists because every pending-testing entry
says the same thing -- "verified against fakes, not run on hardware" --
and there was no scripted way to actually run them. One process per
experiment (fresh allocator, fresh model load, no cross-contamination),
driven from the shell, e.g.:

    VENV_PYTHON=/path/to/venv/bin/python

    # main route, uniform dataset, attention checkpointing as shipped
    python scripts/hw_validate.py main --label A_after --dataset 1024 --steps 40

    # same, but with enable_attention_block_checkpointing() neutered --
    # reproduces the pre-fix "checkpointing only reaches ResBlock" behavior
    HW_DISABLE_ATTENTION_CKPT=1 python scripts/hw_validate.py main \
        --label A_before --dataset 1024 --steps 40

    # strict / under-pressure runs (pending-testing's control-handle entry)
    python scripts/hw_validate.py main --label C_strict --dataset 1image \
        --steps 10 --budget 2500 --strict

    # managed (Resources Controller) route
    python scripts/hw_validate.py managed --label D_managed --dataset 1024 \
        --steps 40 --budget 11500

Outputs land in runs/hw_validation/<label>/:
  - steps.jsonl   one row per step, flushed as it's written (so a run that
                  dies mid-way -- including a real OOM -- still leaves every
                  step it did complete to read afterwards)
  - summary.json  configuration + aggregate numbers + outcome
  - console.log   whatever the nodes themselves printed (profile output etc.)

Outcome classification: "ok", "strict_raise" (BudgetedResourceControlHandle
doing its job under --strict), or "oom" (a real device-side out-of-memory).
An OOM is a *result*, not a harness failure -- exit code 2 for it, so a
driver loop can keep going and still tell the cases apart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "runs" / "hw_validation"


# ------------------------------------------------------------------ memory
class MemProbe:
    """Per-step device-memory accounting straight from torch.xpu, independent
    of the project's own DeviceContext stats (so a bug in the latter can't
    silently make every experiment look healthy)."""

    def __init__(self):
        import torch
        self._torch = torch

    @staticmethod
    def _mb(n: int) -> float:
        return round(n / 2**20, 1)

    def snapshot(self) -> dict:
        t = self._torch
        out = {}
        for key, fn in (
            ("reserved_mb", "memory_reserved"),
            ("allocated_mb", "memory_allocated"),
            ("peak_reserved_mb", "max_memory_reserved"),
            ("peak_allocated_mb", "max_memory_allocated"),
        ):
            try:
                out[key] = self._mb(getattr(t.xpu, fn)())
            except Exception:
                out[key] = None
        return out

    def reset_peak(self) -> None:
        try:
            self._torch.xpu.reset_peak_memory_stats()
        except Exception:
            pass


def make_on_step(jsonl_path: Path, probe: MemProbe, state: dict):
    fh = jsonl_path.open("a")

    def on_step(step: int, loss) -> None:
        now = time.monotonic()
        row = {"step": int(step), "loss": float(loss), "wall": round(time.time(), 3)}
        prev = state.get("t_prev")
        if prev is not None:
            row["dt_sec"] = round(now - prev, 4)
        state["t_prev"] = now
        row.update(probe.snapshot())
        # First row covers checkpoint/text-encoder load + step 0 -- flagged,
        # and excluded from steady-state aggregates in summarize().
        row["covers_load"] = not state.get("reset_done")
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        probe.reset_peak()
        state["reset_done"] = True

    return on_step, fh


def summarize(jsonl_path: Path) -> dict:
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    steady = [r for r in rows if not r.get("covers_load") and "dt_sec" in r]
    dts = [r["dt_sec"] for r in steady if r["dt_sec"] > 0]
    peaks = [r["peak_reserved_mb"] for r in steady if r.get("peak_reserved_mb")]
    reserved = [r["reserved_mb"] for r in steady if r.get("reserved_mb")]
    out = {
        "steps_recorded": len(rows),
        "steady_steps": len(steady),
        "loss_first": rows[0]["loss"] if rows else None,
        "loss_last": rows[-1]["loss"] if rows else None,
        "steps_per_sec_steady": round(len(dts) / sum(dts), 3) if dts else None,
        "per_step_peak_reserved_mb": {"max": max(peaks) if peaks else None,
                                      "first": peaks[0] if peaks else None,
                                      "last": peaks[-1] if peaks else None},
        "reserved_mb_series": {"first": reserved[0] if reserved else None,
                               "last": reserved[-1] if reserved else None,
                               "max": max(reserved) if reserved else None,
                               "min": min(reserved) if reserved else None,
                               "drift": round(reserved[-1] - reserved[0], 1) if len(reserved) > 1 else None},
    }
    return out


# ------------------------------------------------------------------ graphs
def build_common(ctx, args, probe: MemProbe):
    """Pieces shared by both routes: weights, batches, LR schedule, VRAM budget."""
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.dataset.managed import ManagedDatasetSourceNode
    from nodes.train.schedule import CosineLRScheduleNode
    from nodes.memory.vram_budget_controller import VRAMBudgetControllerNode

    weights = SafetensorsCheckpointNode(ctx).build(path=args.checkpoint)["weights"]
    batches = ManagedDatasetSourceNode(ctx).build(
        dataset_root=args.dataset, batch_size=args.batch, shuffle=True)["batches"]
    schedule = CosineLRScheduleNode(ctx).build(
        lr=args.lr, total_steps=args.steps)["schedule"]
    control = VRAMBudgetControllerNode(ctx).build(
        vram_budget_mb=args.budget, strict=args.strict, vram_reserve_mb=args.reserve)["control"]
    return weights, batches, schedule, control


def run_main_route(args, ctx) -> str:
    from nodes.model.lora_injector import ComfyUNetLoRANode
    from nodes.model.text_encoder import SDXLTextEncoderNode
    from nodes.model.text_encoder_cache import CachingTextEncoderNode
    from nodes.model.parameters import ModelParametersNode
    from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
    from nodes.train.supervised import SupervisedLoRATrainerNode

    probe = MemProbe()
    weights, batches, schedule, control = build_common(ctx, args, probe)

    model = ComfyUNetLoRANode(ctx).build(
        weights=weights, rank=args.rank, alpha=args.alpha,
        use_checkpoint=not args.no_checkpoint)["model"]
    encoder = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    encoder = CachingTextEncoderNode(ctx).build(
        encoder=encoder, resource_control=control)["encoder"]
    params = ModelParametersNode(ctx).build(model=model)["params"]
    optimizer = ComposedAdamWOptimizerNode(ctx).build(
        params=params, lr=args.lr, state_precision=args.state_precision)["optimizer"]

    load_stats = probe.snapshot()
    probe.reset_peak()

    SupervisedLoRATrainerNode(ctx).build(
        model=model, batches=batches, optimizer=optimizer, text_encoder=encoder,
        lr_schedule=schedule, steps=args.steps, resource_control=control,
        on_step=args._on_step, profile=args.profile)
    return load_stats


def run_managed_route(args, ctx) -> str:
    from nodes.model.resources_controller import ResourcesControllerNode
    from nodes.model.lora_training_config import LoRATrainingConfigNode
    from nodes.model.trainer_parameters import TrainerParametersNode
    from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
    from nodes.train.managed import ManagedLoRATrainerNode

    probe = MemProbe()
    _, batches, schedule, control = build_common(ctx, args, probe)

    resources = ResourcesControllerNode(ctx).build(
        preset="lora_sdxl", checkpoint_path=args.checkpoint)["resources"]
    trainer = LoRATrainingConfigNode(ctx).build(
        resources=resources, rank=args.rank, alpha=args.alpha,
        unet_weight_store=args.weight_store,
        use_checkpoint=not args.no_checkpoint,
        cache_text_encoder=args.cache_text_encoder)["trainer"]
    params = TrainerParametersNode(ctx).build(trainer=trainer)["params"]
    optimizer = ComposedAdamWOptimizerNode(ctx).build(
        params=params, lr=args.lr, state_precision=args.state_precision)["optimizer"]

    load_stats = probe.snapshot()
    probe.reset_peak()

    ManagedLoRATrainerNode(ctx).build(
        trainer=trainer, batches=batches, optimizer=optimizer,
        lr_schedule=schedule, steps=args.steps, resource_control=control,
        on_step=args._on_step, profile=args.profile,
        calibration_steps=args.calibration_steps,
        residency_safety_margin=args.safety_margin,
        empty_cache_every_n_steps=args.empty_cache_every)
    return load_stats


# ------------------------------------------------------------------ driver
def maybe_disable_attention_checkpointing() -> bool:
    if os.environ.get("HW_DISABLE_ATTENTION_CKPT") != "1":
        return False
    import nodes.model.attention_checkpointing as ac
    ac.enable_attention_block_checkpointing = lambda *a, **k: None
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="route", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--label", required=True)
    common.add_argument("--dataset", default="1024")
    common.add_argument("--steps", type=int, default=40)
    common.add_argument("--batch", type=int, default=1)
    common.add_argument("--checkpoint", default="div_4.safetensors")
    common.add_argument("--rank", type=int, default=64)
    common.add_argument("--alpha", type=float, default=32.0)
    common.add_argument("--lr", type=float, default=1e-4)
    common.add_argument("--state-precision", default="float32",
                        choices=["float32", "int8"])
    common.add_argument("--budget", type=float, default=11500.0)
    common.add_argument("--reserve", type=float, default=512.0)
    common.add_argument("--strict", action="store_true")
    common.add_argument("--profile", action="store_true",
                        help="per-phase VRAM/timing prints (adds synchronize overhead)")
    common.add_argument("--no-checkpoint", action="store_true",
                        help="use_checkpoint=False (disables ALL activation checkpointing)")
    common.add_argument("--weight-store", default="bf16", choices=["bf16", "nf4"])
    common.add_argument("--cache-text-encoder", action="store_true")

    managed = sub.add_parser("managed", parents=[common])
    managed.add_argument("--calibration-steps", type=int, default=3)
    managed.add_argument("--safety-margin", type=float, default=0.1)
    managed.add_argument("--empty-cache-every", type=int, default=1)
    sub.add_parser("main", parents=[common])

    args = p.parse_args()

    import torch
    from core.comfy_setup import setup_comfy
    from nodes.core import ExecutionContext

    # The node route's model/text-encoder construction imports comfy.*
    # (ComfyUI's own packages) -- server/main.py ends up here via
    # core.comfy_setup too; a standalone script has to do it explicitly.
    setup_comfy()

    out_dir = OUT_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "steps.jsonl"
    jsonl_path.unlink(missing_ok=True)

    attention_disabled = maybe_disable_attention_checkpointing()

    probe = MemProbe()
    on_step, fh = make_on_step(jsonl_path, probe, {"t_prev": None})
    args._on_step = on_step

    config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    config.update({
        "route": args.route,
        "attention_checkpointing_disabled": attention_disabled,
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(0) if torch.xpu.is_available() else "cpu",
        "device_total_mb": (torch.xpu.get_device_properties(0).total_memory // 2**20
                            if torch.xpu.is_available() else None),
    })
    summary = {"config": config, "outcome": None}

    t0 = time.monotonic()
    try:
        ctx = ExecutionContext()
        if args.route == "main":
            load_stats = run_main_route(args, ctx)
        else:
            load_stats = run_managed_route(args, ctx)
        summary["load_stats"] = load_stats
        summary["outcome"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- classified below, full trace kept
        fh.flush()
        summary["traceback"] = traceback.format_exc()
        msg = str(exc)
        if "out of memory" in msg.lower() or type(exc).__name__ == "OutOfMemoryError":
            summary["outcome"] = "oom"
        elif isinstance(exc, RuntimeError) and ("budget" in msg.lower() or "strict" in msg.lower()):
            summary["outcome"] = "strict_raise"
        else:
            summary["outcome"] = f"error: {type(exc).__name__}: {msg[:300]}"
    finally:
        fh.close()
        summary["elapsed_sec"] = round(time.monotonic() - t0, 1)
        try:
            summary["run_aggregates"] = summarize(jsonl_path)
        except Exception as exc:  # noqa: BLE001
            summary["run_aggregates"] = {"error": str(exc)}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== hw_validate [{args.label}] outcome={summary['outcome']} "
          f"elapsed={summary['elapsed_sec']}s -> {out_dir}/summary.json")
    agg = summary.get("run_aggregates", {})
    if agg and agg.get("steps_per_sec_steady") is not None:
        print(f"    steady: {agg['steps_per_sec_steady']} steps/sec, "
              f"per-step peak reserved max={agg['per_step_peak_reserved_mb']['max']}MB, "
              f"reserved drift={agg['reserved_mb_series']['drift']}MB")

    if summary["outcome"] == "oom":
        sys.exit(2)
    if summary["outcome"] not in ("ok", "strict_raise"):
        sys.exit(1)


if __name__ == "__main__":
    main()
