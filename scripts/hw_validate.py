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
    python scripts/hw_validate.py main --label A_after --dataset "1024 aes" --steps 40

    # checkpointing sweep: how much of the shipped 100% attention
    # checkpointing is actually needed (floor/peak composition lands in
    # summary.json's floor_stages + steps.jsonl's component_footprints_mb)
    python scripts/hw_validate.py main --label F_frac50 --dataset "1024 aes" \
        --batch 2 --attn-ckpt-fraction 0.5

    # managed route: prewarm the prompt cache from the training batches
    # and unload the text encoder for the whole run (floor -1561MB,
    # peak -1602MB, +10% throughput vs the managed baseline)
    python scripts/hw_validate.py managed --label M_prewarm \
        --dataset "1024 aes" --batch 2 --steps 40 --prewarm-text-encoder

    # same, but with enable_attention_block_checkpointing() neutered --
    # reproduces the pre-fix "checkpointing only reaches ResBlock" behavior
    HW_DISABLE_ATTENTION_CKPT=1 python scripts/hw_validate.py main \
        --label A_before --dataset "1024 aes" --steps 40

    # strict / under-pressure runs (pending-testing's control-handle entry)
    python scripts/hw_validate.py main --label C_strict --dataset "non-square" \
        --steps 10 --budget 2500 --strict

    # managed (Resources Controller) route
    python scripts/hw_validate.py managed --label D_managed --dataset "1024 aes" \
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
import random
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
        # Per-component device footprints, captured once on the first step
        # (after optimizer states exist -- they're lazy). None-valued
        # entries (component exposes no footprint_bytes) are dropped.
        if not state.get("probes_done"):
            comps = {}
            for name, fn in (state.get("probes") or {}).items():
                if fn is None:
                    continue
                try:
                    comps[name] = round(fn() / 2**20, 1)
                except Exception as exc:  # noqa: BLE001 -- probe must never kill a run
                    comps[name] = f"error: {exc}"[:80]
            if comps:
                row["component_footprints_mb"] = comps
                state["probes_done"] = True
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
def _optimizer_node(args):
    """The optimizer class this run should build.

    Imported lazily: three modules, each pulling its own strategy set, and
    a run that only wants AdamW should not pay for the other two.
    """
    if args.optimizer == "adamw":
        from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
        return ComposedAdamWOptimizerNode
    if args.optimizer == "came":
        from nodes.optimizer.composed_came import ComposedCAMEOptimizerNode
        return ComposedCAMEOptimizerNode
    from nodes.optimizer.composed_adafactor import ComposedAdafactorOptimizerNode
    return ComposedAdafactorOptimizerNode


def _strategy(args) -> dict:
    """``{"strategy": ...}`` when asked for, empty otherwise.

    Empty rather than a default, because each node has its own default
    and passing one here would silently override it -- so "whatever this
    node does by default" could not otherwise be expressed.
    """
    return {"strategy": args.strategy} if args.strategy else {}


def build_common(ctx, args, probe: MemProbe):
    """Pieces shared by both routes: weights, batches, LR schedule, VRAM budget."""
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.dataset.managed import ManagedDatasetSourceNode
    from nodes.train.schedule import CosineLRScheduleNode
    from nodes.memory.vram_budget_controller import VRAMBudgetControllerNode

    weights = SafetensorsCheckpointNode(ctx).build(path=args.checkpoint)["weights"]
    args._floor_stages = {"weights_host": probe.snapshot()}
    batches = ManagedDatasetSourceNode(ctx).build(
        dataset_root=args.dataset, batch_size=args.batch, shuffle=True,
        keep_incomplete_batches=getattr(args, "keep_incomplete_batches", False),
    )["batches"]
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
    from nodes.train.supervised import SupervisedLoRATrainerNode

    probe = MemProbe()
    weights, batches, schedule, control = build_common(ctx, args, probe)

    model = ComfyUNetLoRANode(ctx).build(
        weights=weights, rank=args.rank, alpha=args.alpha,
        use_checkpoint=not args.no_checkpoint)["model"]
    args._floor_stages["unet_lora_on_device"] = probe.snapshot()
    encoder = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    encoder = CachingTextEncoderNode(ctx).build(
        encoder=encoder, resource_control=control)["encoder"]
    args._floor_stages["text_encoder_on_device"] = probe.snapshot()
    params = ModelParametersNode(ctx).build(model=model)["params"]
    optimizer = _optimizer_node(args)(ctx).build(
        params=params, lr=args.lr, state_precision=args.state_precision,
        **_strategy(args))["optimizer"]

    load_stats = probe.snapshot()
    probe.reset_peak()
    # Optimizer states are lazy (allocated on the first step) -- register
    # post-step footprint probes so the first steady-state steps.jsonl row
    # records where the floor actually sits once training is underway.
    args._probes.update({
        "unet_lora": getattr(model, "footprint_bytes", None),
        "text_encoder": getattr(encoder, "footprint_bytes", None),
        "optimizer": getattr(optimizer, "footprint_bytes", None),
    })

    SupervisedLoRATrainerNode(ctx).build(
        model=model, batches=batches, optimizer=optimizer, text_encoder=encoder,
        lr_schedule=schedule, steps=args.steps, resource_control=control,
        on_step=args._on_step, profile=args.profile)
    return load_stats


def run_managed_route(args, ctx) -> str:
    from nodes.model.resources_controller import ResourcesControllerNode
    from nodes.model.lora_training_config import LoRATrainingConfigNode
    from nodes.model.trainer_parameters import TrainerParametersNode
    from nodes.train.managed import ManagedLoRATrainerNode

    probe = MemProbe()
    _, batches, schedule, control = build_common(ctx, args, probe)

    resources = ResourcesControllerNode(ctx).build(
        preset="lora_sdxl", checkpoint_path=args.checkpoint)["resources"]
    args._floor_stages.update({"resources_controller": probe.snapshot()})
    trainer = LoRATrainingConfigNode(ctx).build(
        resources=resources, rank=args.rank, alpha=args.alpha,
        unet_weight_store=args.weight_store,
        use_checkpoint=not args.no_checkpoint,
        cache_text_encoder=args.cache_text_encoder)["trainer"]
    args._floor_stages["trainer_model"] = probe.snapshot()
    params = TrainerParametersNode(ctx).build(trainer=trainer)["params"]
    optimizer = _optimizer_node(args)(ctx).build(
        params=params, lr=args.lr, state_precision=args.state_precision,
        **_strategy(args))["optimizer"]

    load_stats = probe.snapshot()
    probe.reset_peak()
    args._probes.update({
        "resources": getattr(resources, "footprint_bytes", None),
        "trainer": getattr(trainer, "footprint_bytes", None),
        "optimizer": getattr(optimizer, "footprint_bytes", None),
    })

    ManagedLoRATrainerNode(ctx).build(
        trainer=trainer, batches=batches, optimizer=optimizer,
        lr_schedule=schedule, steps=args.steps, resource_control=control,
        on_step=args._on_step, profile=args.profile,
        calibration_steps=args.calibration_steps,
        residency_safety_margin=args.safety_margin,
        empty_cache_every_n_steps=args.empty_cache_every,
        prewarm_text_encoder=args.prewarm_text_encoder)
    return load_stats


# ------------------------------------------------------------------ driver
def configure_attention_checkpointing(args) -> str:
    """Applies --attn-ckpt-fraction (and the HW_DISABLE_ATTENTION_CKPT=1
    escape hatch) by pre-patching the module attribute the strategy's
    lazy import reads at call time. Returns the mode string recorded in
    summary.json's config."""
    if os.environ.get("HW_DISABLE_ATTENTION_CKPT") == "1":
        import nodes.model.attention_checkpointing as ac
        ac.enable_attention_block_checkpointing = lambda *a, **k: None
        return "disabled"
    frac = float(getattr(args, "attn_ckpt_fraction", 1.0))
    if frac <= 0.0:
        import nodes.model.attention_checkpointing as ac
        ac.enable_attention_block_checkpointing = lambda *a, **k: None
        return "disabled(fraction=0)"
    if frac < 1.0:
        import nodes.model.attention_checkpointing as ac
        real = ac.enable_attention_block_checkpointing
        ac.enable_attention_block_checkpointing = lambda: real(fraction=frac)
        return f"fraction:{frac}"
    return "all"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="route", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--label", required=True)
    common.add_argument(
        "--dataset", default="1024 aes",
        help="dataset name *inside the datasets directory*, not a path to "
             "it. It is sandboxed by paths.resolve_safe_dataset_path(), so "
             "'datasets/1024 aes' resolves to datasets/datasets/1024 aes "
             "and finds nothing. Run `ls datasets/` for the names that exist.",
    )
    common.add_argument("--steps", type=int, default=40)
    common.add_argument("--batch", type=int, default=1)
    common.add_argument("--checkpoint", default="div_4.safetensors")
    common.add_argument("--rank", type=int, default=64)
    common.add_argument("--alpha", type=float, default=32.0)
    common.add_argument("--lr", type=float, default=1e-4)
    common.add_argument("--optimizer", default="adamw",
                        choices=["adamw", "adafactor", "came"],
                        help="which Composed optimizer node to build. Affects "
                             "step time and optimizer footprint only -- the "
                             "equivalence results in nodes/smoke_tests/ are "
                             "about the maths, which is not what this measures")
    common.add_argument("--strategy", default=None,
                        help="optimizer batching strategy (names in "
                             "nodes/optimizer/strategy_registry.py). Unset "
                             "means each node's own default, which is "
                             "'simple' -- so naming it explicitly is what "
                             "makes a comparison honest")
    common.add_argument("--seed", type=int, default=1234,
                        help="seeds the dataset order and the model init, so "
                             "two runs that differ only in what is being "
                             "measured do identical work. Without this a "
                             "comparison mixes the thing under test with "
                             "whichever images each run happened to draw")
    common.add_argument("--state-precision", default="float32",
                        choices=["float32", "int8_blockwise"])
    common.add_argument("--budget", type=float, default=11500.0)
    common.add_argument("--reserve", type=float, default=512.0)
    common.add_argument("--strict", action="store_true")
    common.add_argument("--profile", action="store_true",
                        help="per-phase VRAM/timing prints (adds synchronize overhead)")
    common.add_argument("--no-checkpoint", action="store_true",
                        help="use_checkpoint=False (disables ALL activation checkpointing)")
    common.add_argument("--attn-ckpt-fraction", type=float, default=1.0,
                        help="fraction of attention blocks to checkpoint: 1.0=all (shipped "
                             "behavior), 0.5=every 2nd, 0.0=none (ResBlock checkpointing "
                             "unaffected either way)")
    common.add_argument("--weight-store", default="bf16", choices=["bf16", "nf4"])
    common.add_argument("--keep-incomplete-batches", action="store_true",
                        help="keep samples in (prompt, size) groups smaller than a "
                             "batch, as smaller batches, instead of dropping them. "
                             "Costs extra distinct batch shapes, so this is the "
                             "measurement of whether that stalls the device: on the "
                             "non-square dataset at batch 4 it takes the shapes from "
                             "22 to 73 and recovers 89 of 273 samples that are "
                             "otherwise never trained on")
    common.add_argument("--cache-text-encoder", action="store_true")
    common.add_argument("--prewarm-text-encoder", action="store_true",
                        help="managed route only: warm every (prompt, bs, h, w) key "
                             "from the training batches into a cache around trainer.clip, "
                             "then unload the encoder for the whole run")

    managed = sub.add_parser("managed", parents=[common])
    managed.add_argument("--calibration-steps", type=int, default=3)
    managed.add_argument("--safety-margin", type=float, default=0.1)
    managed.add_argument("--empty-cache-every", type=int, default=1)
    sub.add_parser("main", parents=[common])

    args = p.parse_args()

    import torch
    from nodes.core import ExecutionContext

    # ComfyUI used to be put on sys.path here, with the comment "the node
    # route's model/text-encoder construction imports comfy.*". That stopped
    # being true when design doc 12 section 7.3 reimplemented the diffusion
    # path, the VAE and both CLIP towers, and the insertion is gone.
    #
    # It is worth more than the dead code it was, though: with ComfyUI on
    # sys.path, a run that *looked* like it was measuring this project's
    # reimplementation could silently have imported ComfyUI's, and reported
    # numbers for code nobody intended to measure. This is the project's
    # real-hardware measurement harness, so its import graph being exactly
    # what it appears to be is the whole point. The checkpoint files are
    # read from ComfyUI's models directory -- that is data, resolved by
    # `paths`, and stays.

    # Seeded before anything is constructed, because the dataset source
    # builds a shuffling sampler at construction time: seeding afterwards
    # would fix the dropout and the init but not the batch order, which is
    # the thing that actually differed between runs when this was added.
    # The first comparison run (foreach vs simple) showed different
    # loss_first values for exactly this reason.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(args.seed)

    out_dir = OUT_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "steps.jsonl"
    jsonl_path.unlink(missing_ok=True)

    attn_ckpt_mode = configure_attention_checkpointing(args)

    args._probes = {}   # name -> callable -> bytes, filled by the route builders
    args._floor_stages = {}
    probe = MemProbe()
    on_step, fh = make_on_step(jsonl_path, probe, {"t_prev": None, "probes": args._probes})
    args._on_step = on_step

    config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    config.update({
        "route": args.route,
        "attention_checkpointing": attn_ckpt_mode,
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
        summary["floor_stages"] = args._floor_stages
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
