"""Training engine contract: a few real steps on a tiny SDXL-shaped UNet with adapters, through the real batcher.

    python -m next.tests.check_engine

Covers what must keep holding: only adapter weights train, one optimizer step per accumulation window, events
and stopping at step boundaries, fused and plain optimizers agreeing, loss weighting bounds, and the optimizer's
state offload and quantized-state round trips.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ..data.batcher import Batcher
from ..data.conditioning_cache import ConditioningCache
from ..data.images import SizePolicy
from ..data.latent_cache import LatentCache
from ..data.library import ImageLibrary
from ..data.noise import Noiser
from ..data.prompts import PromptTable
from ..engine.conditioner import Conditioner
from ..engine.optim.algorithms import AdamW
from ..engine.optim.handle import FusedOptimizer, Optimizer, ParamGroup
from ..engine.optim.state import Int8BlockStore
from ..engine.step.loss import MinSNR, Uniform, per_sample_mse
from ..engine.runtime.checkpointing import ActivationCheckpointer
from ..engine.step.schedule import Constant, Cosine, Warmup
from ..engine.step.state import StepState
from ..engine.step.trainer import Trainer
from ..families.api.adapters import TrainableSelection
from ..families.general.ldm_unet.unet import UNetModel
from ..families.sdxl.conditioning import SDXLConditioning
from ..families.sdxl.denoiser import SDXLDenoiser
from ..families.sdxl.family import FAMILY
from ..families.sdxl.latents import LATENTS
from ..families.sdxl.objective import SDXLObjective
from ..graph.events import StepFinished
from .check_data import FakeEncoder, make_images
from .check_sdxl import TINY_UNET
from .support import check, run


class FakeKit:
    """Deterministic embeddings per prompt in the tiny UNet's widths."""

    def cache_key(self, prompt, size=None):
        return ("fake", prompt, tuple(size) if size else None)

    def encode(self, prompts, sizes=None):
        out = []
        for p in prompts:
            g = torch.Generator().manual_seed(abs(hash(p)) % 2**31)
            out.append(SDXLConditioning(torch.randn(1, 5, 48, generator=g), torch.randn(1, 40, generator=g)))
        return out

    def stack(self, items):
        return SDXLConditioning.stack(items)

    def bytes_per_prompt(self):
        return 5 * 48 * 4 + 40 * 4


def rig(tmp: Path, *, fused: bool = False, accumulation: int = 1, steps: int = 3, **extra):
    folder = tmp / "images"
    if not folder.exists():
        folder.mkdir(parents=True)
        make_images(folder)
        for p in folder.glob("*.png"):
            from PIL import Image
            Image.open(p).resize((128, 128)).save(p)
    library = ImageLibrary(tmp / "lib.sqlite")
    library.ingest(folder)
    images = library.snapshot()
    cache = LatentCache(tmp / "caches", "v", LATENTS)
    policy = SizePolicy(base=128, mode="center_crop", multiple=8)
    cache.build(images, policy, FakeEncoder(), batch_size=2)
    batcher = Batcher(cache.view(images, policy), PromptTable.from_images(images), Noiser(SDXLObjective()),
                      batch_size=2, keep_incomplete=True, seed=7)
    torch.manual_seed(0)
    denoiser = SDXLDenoiser(UNetModel(**TINY_UNET), dtype=torch.float32)
    base = {k: v.clone() for k, v in denoiser.model.state_dict().items()}
    adapted = FAMILY.inject_denoiser(denoiser, TrainableSelection(), rank=4, alpha=4.0)
    with torch.no_grad():
        for p in adapted.trainable_parameters():
            if p.shape[0] and p.ndim == 2 and p.abs().sum() == 0:
                p.normal_(std=0.02)  # lora_B starts at zero; a nonzero start makes every factor receive gradient
    groups = [ParamGroup(g["params"], g["lr_mult"]) for g in adapted.param_groups()]
    optimizer = (FusedOptimizer if fused else Optimizer)(AdamW(weight_decay=0.0), groups, 1e-3)
    events: list = []
    trainer = Trainer(batches=batcher, denoiser=denoiser, conditioner=Conditioner(ConditioningCache(FakeKit()), dtype=torch.float32),
                      optimizer=optimizer, schedule=Constant(1e-3), weighting=Uniform(), steps=steps, accumulation=accumulation,
                      gate=adapted.gate, emit=events.append, **extra)
    return trainer, adapted, denoiser, base, events, optimizer


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        print("training run")
        trainer, adapted, denoiser, base, events, optimizer = rig(tmp / "a")
        before = {id(p): p.detach().clone() for p in adapted.trainable_parameters()}
        result = trainer.run()
        check(result.steps_done == 3 and not result.stopped and result.last_loss is not None and result.last_loss == result.last_loss,
              "three steps run and the loss is finite")
        moved = [not torch.equal(before[id(p)], p) for p in adapted.trainable_parameters()]
        check(all(moved), f"every adapter parameter received an update ({len(moved)} parameters, {moved.count(False)} unchanged)")
        frozen = [k for k, v in denoiser.model.state_dict().items() if k in base and not torch.equal(v, base[k])]
        check(not frozen, f"no base weight changed {frozen[:2]}")
        steps = [e for e in events if isinstance(e, StepFinished)]
        check([e.step for e in steps] == [1, 2, 3] and all({"loss", "lr", "step_ms"} <= set(e.metrics) for e in steps),
              "one StepFinished per optimizer step with loss, lr and timing")

        print("accumulation and stopping")
        trainer, *_ , optimizer2 = rig(tmp / "b", accumulation=2, steps=2)
        trainer.run()
        check(optimizer2.algorithm.t == 2, "two optimizer steps for two accumulation windows of two micro-steps")
        flag = {"stop": False}
        trainer, _, _, _, ev, _ = rig(tmp / "c", steps=5, should_stop=lambda: flag["stop"])
        trainer.emit = lambda e: (ev.append(e), flag.__setitem__("stop", True))
        stopped = trainer.run()
        check(stopped.stopped and stopped.steps_done == 1, "stopping takes effect at the next step boundary")

        print("fused optimizer")
        plain, a1, *_ = rig(tmp / "d", steps=2)
        fused, a2, *_ = rig(tmp / "e", steps=2, fused=True)
        plain.run(), fused.run()
        gap = max((p - q).abs().max().item() for p, q in zip(a1.trainable_parameters(), a2.trainable_parameters()))
        check(gap < 1e-6, f"fused and plain optimizers give the same adapters (max gap {gap:.1e})")

    print("activation checkpointing")
    grads = {}
    for label, fraction in (("off", None), ("all", 1.0), ("half", 0.5)):
        tmp2 = Path(tempfile.mkdtemp())
        trainer, adapted, denoiser, *_ = rig(tmp2, steps=1)
        if fraction is not None:
            denoiser.set_checkpointer(ActivationCheckpointer(), fraction)
        state = StepState(0, 0, "cpu")
        for phase in trainer.pipeline.phases[:5]:
            phase.run(state)
        state.loss.backward()
        grads[label] = torch.cat([p.grad.flatten() for p in adapted.trainable_parameters()])
    check(torch.allclose(grads["off"], grads["all"], atol=1e-6) and torch.allclose(grads["off"], grads["half"], atol=1e-6),
          "checkpointed gradients equal plain ones (all blocks and half of the attention blocks)")

    print("loss, schedules, optimizer state")
    sigma = torch.tensor([0.1, 1.0, 10.0])
    check(bool((MinSNR(5.0).weight(sigma) <= 1.0).all()) and bool((MinSNR(5.0, "v").weight(sigma) <= 1.0).all()), "Min-SNR weights stay at or below one")
    pred, target = torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2)
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]], [[[1.0, 1.0], [1.0, 1.0]]]])
    check(torch.allclose(per_sample_mse(pred, target, mask), torch.ones(2)), "masked MSE averages over valid positions only")
    warm = Warmup(Cosine(1.0, 100), 10)
    check(warm.value(0) < warm.value(5) < warm.value(10) and abs(Cosine(1.0, 100).value(99) - 0.05) < 1e-9, "warmup ramps into the cosine schedule")
    ps = [torch.nn.Parameter(torch.randn(8, 8)) for _ in range(3)]
    opt = Optimizer(AdamW(), [ParamGroup(ps)], 1e-2, store=Int8BlockStore())
    for p in ps:
        p.grad = torch.randn_like(p)
    opt.step()
    held = opt.footprint_bytes()
    opt.offload()
    check(held > 0 and opt.footprint_bytes() == 0, "offloading state frees its footprint")
    opt.reload("cpu")
    for p in ps:
        p.grad = torch.randn_like(p)
    opt.step()
    check(opt.footprint_bytes() == held, "reloaded state steps again with the same footprint")


if __name__ == "__main__":
    run(main)
