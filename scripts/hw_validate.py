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

    def on_step(step: int, loss, shape=None) -> None:
        now = time.monotonic()
        row = {"step": int(step), "loss": float(loss), "wall": round(time.time(), 3)}
        # The latent shape this step ran on. Both trainer routes pass it (see
        # MonitoringPhase in nodes/train/step_pipeline.py for why it belongs in
        # the record). Key omitted when None rather than written as null: the
        # monitor's rule is that a series with no source draws a gap, and this
        # file is read by the same tooling.
        if shape is not None:
            row["latent_shape"] = shape
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


class _Tee:
    """Writes to the real stdout and to a file, and is still a stdout.

    Minimal on purpose: it has to behave like the stream it replaces for the
    `print(..., flush=True)` calls the nodes make and for anything that reads
    `sys.stdout.fileno()`, or the tee itself becomes the bug. `isatty` and
    `fileno` delegate to the real stream rather than guessing.
    """

    def __init__(self, real, handle):
        self._real = real
        self._handle = handle

    def write(self, data):
        self._handle.write(data)
        return self._real.write(data)

    def flush(self):
        self._handle.flush()
        return self._real.flush()

    def isatty(self):
        return self._real.isatty()

    def fileno(self):
        return self._real.fileno()

    def __getattr__(self, name):
        return getattr(self._real, name)


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


# --------------------------------------------------------- fixed holdout eval
#
# Answers the one question a step time cannot: does a setting change *what the
# model learned*? A run's own loss is measured on whatever it happened to
# train on, so comparing two arms' training losses compares two different sets
# of samples at two different shapes -- it cannot distinguish "the setting
# damaged the model" from "the setting changed the data".
#
# So the evaluation set is built FIRST, from a loader with bucketing off, and
# is byte-identical between runs: the same samples, the same noise, the same t,
# the same conditioning. The RNG is reseeded immediately before building it,
# so the identity does not depend on how much randomness the training run
# happened to consume first (which differs between arms by construction).
# `holdout.digest` in summary.json is the check: if two arms disagree on it,
# the comparison's premise is broken and the numbers mean nothing.
#
# **It is NOT a held-out set, and the flag name overstates what it measures.**
# The samples come from the same dataset the model trains on, so this scores
# *fit*, not generalization. Measured on `non-square` at 16 batches: the set is
# 31 images, and after 300 steps the bucketed arms had trained on all 31 while
# an unbucketed arm had trained on 29 -- 6% of the scored set was unseen, and
# that arm was being *penalised* on it. A name like `--eval-batches` would be
# honest; the flag keeps its name because it is already in run configs and
# renaming it would change recorded configuration for no gain. What the number
# supports is "did training make this set worse", and a genuinely held-out
# split is still unbuilt.
#
# What it DOES support, because the arms are paired: same image, same noise,
# same t, per batch, so per-batch difficulty cancels in the difference. The
# raw spread across batches is ~47x on this dataset and never enters a paired
# difference at all.



def build_fixed_holdout(args, ctx, batches: int, seed: int) -> list[dict]:
    """`batches` unpadded batches, fixed, with no gradient anywhere.

    Unpadded and it stays that way: the loader is built with
    shape_bucket_multiple=0 whatever this run trains with, and shuffle=False so
    the sample order cannot depend on the run. Each batch's tensors are copied
    out of the loader's own objects, because the loader redraws noise on every
    iteration and these have to survive more than one pass.
    """
    import torch
    from nodes.dataset.managed import ManagedDatasetSourceNode

    # Reseed immediately before, so the holdout depends only on `seed` and not
    # on anything drawn before this point.
    random.seed(seed)
    torch.manual_seed(seed)
    source = ManagedDatasetSourceNode(ctx).build(
        dataset_root=args.dataset, batch_size=args.batch, shuffle=False,
        keep_incomplete_batches=True, shape_bucket_multiple=0)["batches"]
    holdout = []
    for batch in source:
        if batch.get("valid_mask") is not None:
            raise AssertionError(
                "the holdout must be unpadded, but this batch carries a "
                "valid_mask -- the loader was not built with "
                "shape_bucket_multiple=0")
        holdout.append({
            "x_t": batch["x_t"].clone(),
            "target": batch["target"].clone(),
            "t": batch["t"].clone(),
            "prompt": batch["prompt"],
        })
        if len(holdout) >= batches:
            break
    return holdout


def holdout_digest(holdout: list[dict]) -> str:
    """A short digest of the holdout's contents, so two runs can prove they
    evaluated the same thing instead of assuming it. Hashes the bytes of every
    tensor plus the prompt, so any difference in sample, noise, t or caption
    shows up as a different digest."""
    import hashlib
    h = hashlib.sha256()
    for item in holdout:
        h.update(item["prompt"].encode("utf-8"))
        for key in ("x_t", "target", "t"):
            tensor = item[key].detach().to("cpu").contiguous()
            h.update(key.encode("ascii"))
            h.update(str(tuple(tensor.shape)).encode("ascii"))
            h.update(tensor.numpy().tobytes())
    return h.hexdigest()[:16]


def evaluate_holdout(holdout: list[dict], model, text_encoder, process,
                     device: str) -> dict:
    """Unweighted MSE per holdout batch, and the mean.

    Deliberately the *plain* masked-mean expression with no loss weighting and
    no LoRA gate: this is a fixed yardstick for comparing two runs, so it must
    not itself depend on the run's weighting, and the gate is a training-time
    device that has no meaning at inference. Both are stated in the report so
    the number is not mistaken for "the run's loss".

    Forward only, under no_grad, and the optimizer is never touched: this
    measures the model as training left it.
    """
    import torch
    from nodes.model.lora import lora_gate_override

    model.eval()
    per_batch = []
    try:
        with torch.no_grad(), lora_gate_override(None):
            for item in holdout:
                x_t = item["x_t"].to(device=device, dtype=torch.bfloat16)
                target = item["target"].to(device=device, dtype=torch.bfloat16)
                t = item["t"].to(device=device, dtype=torch.long).view(-1)
                _, sigma = process.schedule.alpha_sigma(t)
                xc = process.input_transform.scale_input(x_t, sigma)
                # Unpadded by construction, so the size is x_t's own shape --
                # the same call the trainer makes when no mask is present.
                ctx_emb, y = text_encoder.encode(
                    item["prompt"], batch_size=x_t.shape[0],
                    height=x_t.shape[2] * 8, width=x_t.shape[3] * 8)
                pred = model.forward(
                    xc, t, ctx_emb.to(device=device, dtype=torch.bfloat16),
                    y.to(device=device, dtype=torch.bfloat16))
                mse = float((pred.float() - target.float()).pow(2).mean())
                per_batch.append({"shape": [int(x_t.shape[-2]), int(x_t.shape[-1])],
                                  "mse": mse})
    finally:
        model.train()
    values = [b["mse"] for b in per_batch]
    return {
        "batches": len(values),
        "mse_mean": sum(values) / len(values) if values else None,
        "mse_min": min(values) if values else None,
        "mse_max": max(values) if values else None,
        "per_batch": per_batch,
    }


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
        shape_bucket_multiple=getattr(args, "shape_bucket_multiple", 0),
        group_by_size_only=getattr(args, "group_by_size_only", False),
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
    _score_holdout_if_asked(args, model, encoder, ctx)
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

    # Only passed when the flag was actually given. Passing it
    # unconditionally would pin the harness to whatever this script's
    # argparse default is and silently stop exercising the node's own --
    # which is how `prewarm_text_encoder` measured as off-by-default in
    # this harness after the node had been flipped, and how a default
    # change goes unnoticed by the thing that exists to measure it.
    prewarm_kwarg = (
        {} if args.prewarm_text_encoder is None
        else {"prewarm_text_encoder": args.prewarm_text_encoder})
    ManagedLoRATrainerNode(ctx).build(
        trainer=trainer, batches=batches, optimizer=optimizer,
        lr_schedule=schedule, steps=args.steps, resource_control=control,
        on_step=args._on_step, profile=args.profile,
        calibration_steps=args.calibration_steps,
        residency_safety_margin=args.safety_margin,
        empty_cache_every_n_steps=args.empty_cache_every,
        **prewarm_kwarg,
        probe_every_n_steps=getattr(args, "probe_every_n_steps", 0),
        probe_items=getattr(args, "probe_items", 2),
        probe_points_per_bucket=getattr(args, "probe_points_per_bucket", 2),
        probe_grad_alignment=getattr(args, "probe_grad_alignment", False),
        use_xpu_graph=getattr(args, "use_xpu_graph", False))
    # After training, so the model is scored as training left it. The encoder
    # may have been offloaded by the residency controller, so it is brought
    # back before use rather than assumed resident.
    if getattr(args, "_holdout", None):
        from nodes.model.resources_controller import ResourcesControllerNode
        control.ensure_loaded("text_encoder")
        _score_holdout_if_asked(args, trainer.unet, trainer.clip, ctx)
    return load_stats


def _score_holdout_if_asked(args, model, text_encoder, ctx) -> None:
    """Evaluate the fixed holdout after training, into summary.json.

    A no-op unless --holdout-batches was given, so every existing invocation
    of this harness behaves exactly as before.

    The diffusion process is rebuilt rather than taken from the trainer: it is
    a frozen configuration dataclass with no state the run mutated, and the
    trainer node does not expose it. It is built with **the trainer node's own
    defaults, copied** (ManagedLoRATrainerNode.build's
    `inputs.get("diffusion_process") or DiffusionProcess(...)` line) -- using
    anything else here would quietly score on a different input transform than
    the run trained with, and the number would mean nothing.
    """
    if not getattr(args, "_holdout", None):
        return
    from nodes.components.diffusion import (DiffusionProcess,
                                            DiscreteLinearNoiseSchedule,
                                            EpsParameterization,
                                            KarrasInputScaler)
    process = DiffusionProcess(DiscreteLinearNoiseSchedule(), EpsParameterization(),
                               KarrasInputScaler())
    report = evaluate_holdout(args._holdout, model, text_encoder, process,
                              device="xpu")
    args._summary["holdout"] = {
        "digest": args._holdout_digest,
        "seed": args.holdout_seed,
        "unweighted_mse_note": (
            "plain mean squared error over a fixed unpadded holdout, no loss "
            "weighting and no LoRA gate, forward-only -- a yardstick that does "
            "not depend on the run's configuration, so it is comparable "
            "between runs"),
        **report,
    }
    print(f"  [holdout] mse mean {report['mse_mean']:.6f} over "
          f"{report['batches']} batch(es) "
          f"(min {report['mse_min']:.6f}, max {report['mse_max']:.6f})")


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
    # L4: size-only batch grouping (off by default). Lets one batch carry
    # several captions; needs --shape-bucket-multiple > 1 (the loader
    # refuses otherwise). Conditioning is assembled per sample, one
    # text-encode per distinct caption, cache-served.
    common.add_argument("--group-by-size-only", action="store_true",
                        help="group batches by size bucket alone instead of "
                             "(caption, size): multi-caption datasets train "
                             "every sample instead of only full "
                             "single-caption groups")
    common.add_argument("--use-xpu-graph", action="store_true",
                        help="L5.3: capture forward+loss+backward into an "
                             "XPUGraph replay per shape (managed route only; "
                             "refused with fused optimizers or dropout)")
    # Optional shape bucketing (off by default). Pads latents up to a
    # multiple of N so a multi-resolution dataset trains on few shapes, at a
    # permanent +compute cost -- see the node port's own docstring.
    common.add_argument("--shape-bucket-multiple", type=int, default=0,
                        help="0 = off; N > 1 pads each latent up to the next "
                             "multiple of N (32 collapses non-square's 44 "
                             "shapes to 3, measured 1.83x over 150 steps, at "
                             "+1%% per step and 13.3%% mean pad -- run with "
                             "--holdout-batches before believing the speed "
                             "number is free)")
    common.add_argument(
        "--holdout-batches", type=int, default=0,
        help="after training, evaluate the model on this many FIXED unpadded "
             "batches and record the unweighted MSE in summary.json (0 = off). "
             "Built before training, with bucketing off and shuffle off, so two "
             "runs with the same --holdout-seed score byte-identical inputs; "
             "summary.json's holdout.digest is how they prove it. NOT a "
             "held-out split: the samples come from the training dataset, so "
             "this measures fit, not generalization. What it does buy is a "
             "*paired* comparison -- identical image, noise and t per batch, so "
             "batch difficulty cancels -- which a run's own loss cannot give "
             "because it is measured on whatever that run happened to train on")
    common.add_argument(
        "--holdout-seed", type=int, default=99991,
        help="seed for the fixed holdout, deliberately different from --seed "
             "so the holdout's samples and noise are not correlated with the "
             "training draw; the holdout is reseeded with this immediately "
             "before it is built, so its contents depend on it alone")
    common.add_argument("--keep-incomplete-batches", action="store_true",
                        help="keep samples in (prompt, size) groups smaller than a "
                             "batch, as smaller batches, instead of dropping them. "
                             "Costs extra distinct batch shapes, so this is the "
                             "measurement of whether that stalls the device: on the "
                             "non-square dataset at batch 4 it takes the shapes from "
                             "22 to 73 and recovers 89 of 273 samples that are "
                             "otherwise never trained on")
    common.add_argument("--cache-text-encoder", action="store_true")
    # default=None rather than store_true's False: unset means "say
    # nothing and let the node's own default decide", so this harness
    # measures the shipped default instead of pinning its own.
    common.add_argument("--prewarm-text-encoder", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="managed route only: warm every (prompt, bs, h, w) key "
                             "from the training batches into a cache around trainer.clip, "
                             "then unload the encoder for the whole run. On by default "
                             "on the node since 2026-10-04; pass "
                             "--no-prewarm-text-encoder to measure the old behaviour "
                             "(CLIP resident for the run, 1,562 MB more peak)")

    managed = sub.add_parser("managed", parents=[common])
    managed.add_argument("--probe-every-n-steps", type=int, default=0,
                        help="run the training-diagnostics probe every N "
                             "optimizer steps, and once as soon as the probe "
                             "images are captured. 0 = off. The probe's step-1 "
                             "record is what says rel ~ 1.000 and drift ~ 0 on "
                             "a real LoRA, because that is the only point at "
                             "which B is still exactly zero")
    managed.add_argument("--probe-items", type=int, default=2)
    managed.add_argument("--probe-points-per-bucket", type=int, default=2)
    managed.add_argument("--probe-grad-alignment", action="store_true",
                        help="also measure per-bucket gradient norms and "
                             "cosines; needs a backward per probe point, so "
                             "this is the flag whose VRAM peak has to be "
                             "compared against the training step's")
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

    # stdout is teed into console.log for the whole run. The file is listed in
    # this script's own docstring as one of the three outputs and was never
    # written -- so the node-side reports (the dataset node's bucketing pad
    # fractions, the residency controller's calibration line, the prewarm
    # summary) existed only in the terminal scrollback, which a later analysis
    # run cannot read. Found by an analysis script reading `pad fraction` out
    # of it and getting nothing.
    #
    # Tee, not redirect: these prints are operator-facing and this script's
    # output is how a driver loop sees progress. Replacing sys.stdout would
    # keep the file and lose the terminal.
    log_fh = (out_dir / "console.log").open("w", errors="replace")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, log_fh)

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
    args._summary = summary

    t0 = time.monotonic()
    try:
        ctx = ExecutionContext()
        # The holdout is built BEFORE training, and its digest recorded, so two
        # runs can prove they scored the same thing rather than assume it. It
        # is built here rather than inside a route builder because it is a
        # property of the run, not of a route.
        args._holdout = None
        args._holdout_digest = None
        if args.holdout_batches > 0:
            args._holdout = build_fixed_holdout(
                args, ctx, args.holdout_batches, args.holdout_seed)
            args._holdout_digest = holdout_digest(args._holdout)
            print(f"  [holdout] {len(args._holdout)} fixed unpadded batch(es), "
                  f"digest {args._holdout_digest}")
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
        # Restore stdout before the final report, so the summary line lands in
        # the terminal even if the log file has become unwritable.
        sys.stdout = real_stdout
        log_fh.close()
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
