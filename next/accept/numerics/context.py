"""What the checks run on: the device, and a subject (a real checkpoint with an adapter, or a tiny stand-in) built once on demand."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ...engine.runtime.checkpointing import ActivationCheckpointer
from ...families.api.adapters import TrainableSelection
from ...families.general.adapters.layers import AdaptedConv2d, AdaptedLinear
from ...families.general.ldm_unet.unet import UNetModel
from ...families.registry import get_family
from ...families.sdxl.adapters import inject_denoiser
from ...families.sdxl.denoiser import SDXLDenoiser
from ..fixtures import DenoiserInputs
from ..loading import load_denoiser

TINY_UNET = {
    "in_channels": 4, "out_channels": 4, "model_channels": 32, "num_res_blocks": [1, 1, 1], "channel_mult": [1, 2, 4],
    "num_head_channels": 8, "transformer_depth": [0, 0, 1, 1, 2, 2], "transformer_depth_middle": 2,
    "transformer_depth_output": [0, 0, 0, 1, 1, 1, 2, 2, 2], "context_dim": 48, "use_linear_in_transformer": True,
    "num_classes": "sequential", "adm_in_channels": 40,
}


@dataclass
class Snapshot:
    """One forward and backward: the output, the gradient of the input, and the gradient of every adapter tensor (host, float32)."""

    out: Tensor
    dx: Tensor
    grads: dict[str, Tensor]


@dataclass
class Context:
    """``tiny`` builds a small random UNet instead of loading ``file`` (for a machine without the checkpoint, and for testing the checks)."""

    device: torch.device
    models: str | None = None
    file: str | None = None
    lora: str | None = None
    resolution: int = 256
    tiny: bool = False
    seed: int = 0
    memory_mb: int = 1024  # the size of the device memory pattern test
    _subject: Subject | None = field(default=None, repr=False)

    @property
    def subject(self) -> Subject:
        if self._subject is None:
            self._subject = Subject(self)
        return self._subject

    def release(self) -> None:
        gc.collect()
        backend = getattr(torch, self.device.type, None)
        if hasattr(backend, "empty_cache"):
            backend.empty_cache()


class Subject:
    """A denoiser with a LoRA on every eligible layer. The factors are fixed pseudo-random values of a size a trained adapter has (or the
    values of ``--lora``), so every check sees the same adapter and the layers do not start from a zero branch."""

    RANK = 16

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        if ctx.tiny:
            torch.manual_seed(ctx.seed)
            self.denoiser = SDXLDenoiser(UNetModel(**TINY_UNET), device=ctx.device, dtype=torch.bfloat16)
            self.dims = {"context_dim": 48, "adm_dim": 40, "tokens": 5}
            self.resolution, rank = 128, 4
        else:
            if not ctx.models or not ctx.file:
                raise SystemExit("the model checks need --models and --file (or --tiny)")
            self.denoiser = load_denoiser(ctx.models, ctx.file, str(ctx.device))
            self.dims = {}
            self.resolution, rank = ctx.resolution, self.RANK
        self.adapted = inject_denoiser(self.denoiser, TrainableSelection(), rank=rank, alpha=float(rank))
        self.family = None if ctx.tiny else get_family("sdxl")
        self._factors()
        self.checkpointer = ActivationCheckpointer(preserve_rng=False)
        self.loaded_lora = False
        if ctx.lora and not ctx.tiny:
            self._load_lora(ctx.lora)

    def _factors(self) -> None:
        gen = torch.Generator().manual_seed(self.ctx.seed + 1)
        for layer in self.adapted.layers.values():
            with torch.no_grad():
                layer.lora_B.copy_(torch.randn(layer.lora_B.shape, generator=gen) * 0.02)

    def _load_lora(self, path: str) -> None:
        state, meta = self.family.read_adapter(path)
        scheme = self.family.schemes.get(meta.get("scheme") or "kohya")
        report = self.family.load_checked(scheme, self.adapted, state)
        self.loaded_lora = True
        print(f"  adapter {path}: {report.matched} layers matched, {len(report.missing)} missing, {len(report.unexpected)} unexpected keys", flush=True)

    @property
    def layers(self) -> list:
        return [layer for layer in self.adapted.layers.values() if isinstance(layer, (AdaptedLinear, AdaptedConv2d))]

    def set_compute(self, name: str) -> None:
        """``float32`` or ``activation``: the dtype the low-rank path runs in."""
        for layer in self.layers:
            layer.compute_in_activation(name == "activation")

    def set_checkpointing(self, on: bool) -> None:
        self.denoiser.set_checkpointer(self.checkpointer if on else None, 1.0)

    def inputs(self, batch: int = 1, seed: int = 0) -> DenoiserInputs:
        steps = [250, 500, 750, 900, 100, 400, 600, 800][:batch]
        return DenoiserInputs(resolution=self.resolution, timesteps=steps, device=self.ctx.device, dtype=torch.bfloat16, seed=seed, **self.dims)

    def evaluate(self, inputs: DenoiserInputs | None = None) -> Snapshot:
        """Forward and backward against the inputs' fixed output gradient, leaving no gradient behind."""
        inputs = inputs or self.inputs()
        self.zero()
        out, dx = inputs.run(self.denoiser)
        grads = {}
        for path, layer in self.adapted.layers.items():
            for name, tensor in layer.adapter_state().items():
                if tensor.grad is not None:
                    grads[f"{path}.{name}"] = tensor.grad.detach().float().cpu().clone()
        self.zero()
        return Snapshot(out.detach().float().cpu(), dx.detach().float().cpu(), grads)

    def zero(self) -> None:
        for p in self.adapted.trainable_parameters():
            p.grad = None
