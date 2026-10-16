"""One-shot hardware check of the new pipeline on real SDXL weights (run on the Intel Arc machine).

    python scripts/xpu_smoke.py --models D:/models --file sdxl_base.safetensors
    python scripts/xpu_smoke.py ... --device cpu          # slow, to separate "wrong" from "XPU-specific"
    python scripts/xpu_smoke.py ... --images D:/some/images --resolution 1024 --report smoke.json

``--models`` is a folder that holds a ``checkpoints`` subfolder (or the checkpoints themselves when there is none);
``--file`` is relative to that checkpoints folder, so ``sdxl_base.safetensors``, not ``checkpoints/sdxl_base.safetensors``. Each step prints PASS/FAIL/INFO with numbers; send the whole output back. A failed
step does not stop the others, except that later steps need the earlier loads. Peak memory and timings are
reported because they are the point of the exercise. Nothing is written except the optional report and a
temporary latent cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from next.families.api.adapters import TrainableSelection  # noqa: E402
from next.families.api.library import ModelLibrary  # noqa: E402
from next.families.api.parts import make_part  # noqa: E402
from next.families.registry import detect_file, get_family  # noqa: E402

RESULTS: list[dict] = []


def record(status: str, name: str, detail: str = "", **numbers) -> None:
    RESULTS.append({"status": status, "name": name, "detail": detail, **numbers})
    extra = "  " + json.dumps(numbers, default=float) if numbers else ""
    print(f"[{status}] {name}{(': ' + detail) if detail else ''}{extra}", flush=True)


def step(name):
    """Run a function as one reported step; an exception is a FAIL with a traceback, not a crash."""
    def wrap(fn):
        def run(*args, **kwargs):
            started = time.time()
            try:
                result = fn(*args, **kwargs)
                record("INFO", f"{name} finished", seconds=round(time.time() - started, 1))
                return result
            except Exception:  # noqa: BLE001
                record("FAIL", name, traceback.format_exc(limit=6))
                return None
        return run
    return wrap


class Device:
    def __init__(self, name: str):
        from next.engine.devices import Accelerator

        self.name = name
        self.module = getattr(torch, name, None) if name != "cpu" else None
        self.accel = Accelerator(name)

    def sync(self) -> None:
        self.accel.synchronize()

    def reset_peak(self) -> None:
        self.accel.reset_peak()

    def memory(self) -> dict:
        """GB: peak live tensors, peak held by the allocator, and the driver's view of the card."""
        return {f"{key}_gb": round(value / 1024, 2) for key, value in self.accel.memory().items()}

    def free(self) -> None:
        gc.collect()
        if self.module is not None and hasattr(self.module, "empty_cache"):
            self.module.empty_cache()


def check(condition: bool, name: str, detail: str = "", **numbers) -> None:
    record("PASS" if condition else "FAIL", name, detail, **numbers)


def test_images(folder: str | None, count: int, size: int):
    from PIL import Image

    paths = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}) if folder else []
    images = []
    for path in paths[:count]:
        with Image.open(path) as image:
            images.append(image.convert("RGB").resize((size, size)))
    if not images:  # smooth gradients plus texture: not a photo, but exercises the same code paths
        gen = torch.Generator().manual_seed(0)
        axis = torch.linspace(0, 1, size)
        for i in range(count):
            base = torch.stack([axis[None, :].expand(size, size), axis[:, None].expand(size, size),
                                torch.full((size, size), 0.2 + 0.2 * i)])
            data = (base * 0.8 + 0.2 * torch.rand(3, size, size, generator=gen)).clamp(0, 1)
            images.append(Image.fromarray((data.permute(1, 2, 0) * 255).byte().numpy()))
        record("INFO", "no --images given", "using synthetic images; latent statistics are only indicative")
    return images


@step("environment")
def environment(dev: Device) -> None:
    record("INFO", "torch", torch.__version__)
    if dev.name == "xpu":
        ok = hasattr(torch, "xpu") and torch.xpu.is_available()
        check(ok, "torch.xpu available")
        if ok:
            props = torch.xpu.get_device_properties(0)
            record("INFO", "device", props.name, total_gb=round(props.total_memory / 2**30, 1))


@step("detect checkpoint")
def detect(args) -> dict | None:
    library = ModelLibrary({args.scope: [Path(args.models) / args.scope]} if (Path(args.models) / args.scope).is_dir()
                           else {args.scope: [args.models]})
    found = detect_file(library, args.scope, args.file)
    check(found.ok, "detection ok", found.message or "")
    if not found.ok:
        return None
    check(found.family_id == "sdxl", "family is sdxl", found.family_id or "")
    path = str(library.resolve(args.scope, args.file))
    parts = {p.role: make_part(found.family_id, path, p) for p in found.parts}
    check({"denoiser", "clip_l", "clip_g", "vae"} <= set(parts), "checkpoint holds denoiser, both CLIPs and VAE",
          roles=sorted(parts))
    return parts


@step("tokenizer")
def tokenizer() -> None:
    from next.families.general.clip.tokenizer import CLIPTokenizer, default_vocabulary_dir

    record("INFO", "vocabulary folder", str(default_vocabulary_dir()))
    tok = CLIPTokenizer()
    out = tok.tokenize("a photo of a cat, red hair")
    ids = out["l"][0]
    check(ids[0] == 49406 and 49407 in ids and len(ids) == 77, "BOS ... EOS padded to 77", length=len(ids))
    long = tok.tokenize(", ".join(["word"] * 120))
    check(len(long["l"]) >= 2, "a long prompt becomes several sections", sections=len(long["l"]))


@step("text encoder")
def text_encoder(parts, dev: Device):
    family = get_family("sdxl")
    prompts = ["a photo of a cat", "1girl, red hair, smiling, outdoors", "", "a very detailed landscape painting"]
    dev.reset_peak()
    half = family.load_text_encoder(parts, device=dev.name, dtype=torch.float16)
    one_by_one = [half.encode_text([p])[0] for p in prompts]
    check(all(torch.isfinite(e.ctx.float()).all() and torch.isfinite(e.pooled.float()).all() for e in one_by_one),
          "fp16 encodings are finite")
    check(tuple(one_by_one[0].ctx.shape) == (1, 77, 2048) and tuple(one_by_one[0].pooled.shape) == (1, 1280),
          "encoding shapes (1,77,2048) and (1,1280)", shape=list(one_by_one[0].ctx.shape))
    record("INFO", "fp16 towers", **dev.memory())
    ref = family.load_text_encoder(parts, device="cpu", dtype=torch.float32)
    full = ref.encode_text(prompts)
    worst = 0.0
    for a, b in zip(one_by_one, full):
        a_, b_ = a.ctx.float().cpu(), b.ctx.float()
        worst = max(worst, ((a_ - b_).norm() / b_.norm()).item())
    check(worst < 0.05, "fp16 on device agrees with fp32 on CPU", worst_relative_error=round(worst, 5))
    batched = ref.batch_size
    ref.batch_size = len(prompts)
    together = ref.encode_text(prompts)
    gap = max(((x.ctx - y.ctx).norm() / y.ctx.norm()).item() for x, y in zip(together, full))
    check(gap < 1e-4, "fp32 batched equals one at a time", relative_error=gap, batch=batched)
    del ref
    return half, one_by_one


@step("vae")
def vae(parts, images, dev: Device, resolution: int):
    import numpy as np

    family = get_family("sdxl")
    codec = family.load_vae(parts["vae"], device=dev.name, dtype=torch.float32)
    batch = torch.stack([torch.from_numpy(np.asarray(i)).permute(2, 0, 1) for i in images])
    dev.reset_peak()
    started = time.time()
    latents = codec.encode(batch[:1])
    dev.sync()
    record("INFO", "vae encode", seconds=round(time.time() - started, 2), **dev.memory())
    check(tuple(latents.shape) == (1, 4, resolution // 8, resolution // 8), "latent shape", shape=list(latents.shape))
    check(bool(torch.isfinite(latents).all()), "latents finite")
    record("INFO", "latent statistics (scaled; real photos give std near 1)", mean=latents.mean().item(),
           std=latents.std().item())
    decoded = codec.decode(latents).float() / 255.0
    original = batch[:1].float() / 255.0
    psnr = (10 * torch.log10(1.0 / ((decoded - original) ** 2).mean())).item()
    check(psnr > 22, "encode then decode reproduces the image", psnr_db=round(psnr, 1))
    return codec


@step("unet forward and backward with LoRA")
def unet(parts, cond, dev: Device, resolution: int, rank: int, checkpoint: bool, batch: int):
    from next.families.sdxl.objective import SDXLObjective

    family = get_family("sdxl")
    dev.free()
    record("INFO", "before loading the UNet (everything else released)", **dev.memory())
    dev.reset_peak()
    denoiser = family.load_denoiser(parts["denoiser"], device=dev.name, dtype=torch.bfloat16)
    record("INFO", "unet loaded (bf16)", **dev.memory())
    latent = resolution // 8
    x0 = torch.randn(batch, 4, latent, latent)
    objective = SDXLObjective()
    t = torch.full((batch,), 500)
    x_t, target, _sigma = objective.noisy(x0, torch.randn_like(x0), t)
    ctx = cond.ctx.to(dev.name, torch.bfloat16).expand(batch, -1, -1).contiguous()
    y = cond.y.to(dev.name, torch.bfloat16).expand(batch, -1).contiguous()
    x_t, tt = x_t.to(dev.name, torch.bfloat16), t.to(dev.name)
    with torch.no_grad():
        out = denoiser(x_t, tt, ctx, y)
    check(out.shape == x_t.shape and bool(torch.isfinite(out.float()).all()), "forward output finite, input-shaped")
    adapted = family.inject_denoiser(denoiser, TrainableSelection(), rank=rank, alpha=float(rank))
    if checkpoint:
        from next.engine.runtime.checkpointing import ActivationCheckpointer

        denoiser.set_checkpointer(ActivationCheckpointer(), 1.0)
    params = adapted.trainable_parameters()
    record("INFO", "adapter", modules=len(adapted.layers), trainable_params=sum(p.numel() for p in params))
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    times = []
    record("INFO", "adapter injected", **dev.memory())
    for i in range(4):
        last = i == 3
        if last:
            dev.reset_peak()
        dev.sync()
        started = time.time()
        pred = denoiser(x_t, tt, ctx, y)
        loss = torch.nn.functional.mse_loss(pred.float(), target.to(dev.name).float())
        if last:
            dev.sync()
            record("INFO", "after forward", **dev.memory())
        loss.backward()
        if last:
            dev.sync()
            record("INFO", "after backward", **dev.memory())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dev.sync()
        times.append(time.time() - started)
        if last:
            record("INFO", "after optimizer step", **dev.memory())
            dev.free()
            record("INFO", "after emptying the cache", **dev.memory())
    check(bool(torch.isfinite(loss)), "loss finite after steps", loss=loss.item())
    record("INFO", "training step" + (" (checkpointed)" if checkpoint else ""), seconds_first=round(times[0], 2), seconds_steady=round(times[-1], 2),
           batch=batch)
    del denoiser, adapted, optimizer
    dev.free()


@step("dataset chain on real VAE")
def dataset_chain(codec, images, dev: Device, resolution: int) -> None:
    from next.data.batcher import Batcher
    from next.data.images import SizePolicy
    from next.data.latent_cache import LatentCache
    from next.data.library import ImageLibrary
    from next.data.noise import Noiser
    from next.data.prompts import PromptTable, TagSwap
    from next.families.sdxl.latents import LATENTS
    from next.families.sdxl.objective import SDXLObjective

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        folder = tmp / "images"
        folder.mkdir()
        for i, image in enumerate(images):
            image.save(folder / f"{i}.png")
            (folder / f"{i}.txt").write_text(f"tag a, tag b{i}, tag c")
        library = ImageLibrary(tmp / "lib.sqlite")
        library.ingest(folder)
        snapshot = library.snapshot()
        policy = SizePolicy(base=resolution, mode="fit", multiple=8)
        cache = LatentCache(tmp / "caches", "smoke", LATENTS, label="smoke")
        stats = cache.build(snapshot, policy, codec, batch_size=2)
        check(stats.encoded_images == len(snapshot) and not stats.skipped, "latent cache encodes every image",
              encoded=stats.encoded_images)
        latents = cache.view(snapshot, policy)
        prompts = PromptTable.from_images(snapshot, seed=1).with_stage(TagSwap("random"))
        batcher = Batcher(latents, prompts, Noiser(SDXLObjective()), batch_size=2, keep_incomplete=True, pin=dev.name == "xpu")
        batches = list(batcher)
        check(len(batches) >= 1 and all(torch.isfinite(b["x_t"]).all() for b in batches), "batcher serves finite batches",
              batches=len(batches))
        cache.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True, help="folder with the model files")
    parser.add_argument("--file", required=True, help="all-in-one SDXL checkpoint, relative to <models>/<scope>")
    parser.add_argument("--scope", default="checkpoints")
    parser.add_argument("--device", default="xpu", choices=("xpu", "cpu", "cuda"))
    parser.add_argument("--images", help="folder of real images (a few are used)")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1, help="batch size of the UNet step")
    parser.add_argument("--checkpoint", action="store_true", help="recompute activations in the UNet step (what the trainer does)")
    parser.add_argument("--report", help="write the results as JSON here")
    args = parser.parse_args()

    dev = Device(args.device)
    environment(dev)
    parts = detect(args)
    tokenizer()
    if parts:
        encoded = text_encoder(parts, dev)
        images = test_images(args.images, 2, args.resolution)
        codec = vae(parts, images, dev, args.resolution)
        if encoded:
            half, per_prompt = encoded
            cond = half.assemble(per_prompt[0], (args.resolution, args.resolution))
            del half
            dev.free()
            unet(parts, cond, dev, args.resolution, args.rank, args.checkpoint, args.batch)
        if codec is not None:
            dataset_chain(codec, images, dev, args.resolution)
    failed = [r["name"] for r in RESULTS if r["status"] == "FAIL"]
    print(f"\n{sum(r['status'] == 'PASS' for r in RESULTS)} passed, {len(failed)} failed" + (f": {failed}" if failed else ""))
    if args.report:
        Path(args.report).write_text(json.dumps(RESULTS, indent=1, default=float))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
