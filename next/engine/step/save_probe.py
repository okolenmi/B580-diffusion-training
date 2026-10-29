"""One forward pass with the adapter exactly as training left it, taken right before a file is written.

A saved adapter must make the model compute what the training process was computing. The probe records that computation
(inputs, conditioning, the prediction with the adapter live and with it switched off) in a file next to the adapter, so
the saved file can be loaded again later, here or in another program, and made to answer the same questions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import torch
from safetensors.torch import save_file

from ..handles import TrainableDenoiser

PROMPTS = (
    "a red fox standing in a snowy forest at dawn, soft light",
    "portrait of an old fisherman with a weathered face, 35mm photograph",
    "a small wooden cabin by a mountain lake, reflections on the water",
    "a futuristic city street at night, neon signs and rain",
    "a bowl of ripe oranges on a rustic kitchen table",
    "an astronaut riding a horse across a desert, wide shot",
    "a watercolor painting of a quiet harbor with sailing boats",
    "a close-up of a green parrot sitting on a branch",
)
TIMESTEPS = (200.0, 600.0, 900.0)


class SaveProbe:
    """``seed`` picks the prompt and the noise; the empty prompt is always probed as well (it is what captionless training sees)."""

    def __init__(self, model: TrainableDenoiser, conditioner, seed: int = 0, size: tuple[int, int] = (1024, 1024)):
        self.model, self.conditioner, self.seed, self.size = model, conditioner, seed, size
        self.prompts = (PROMPTS[seed % len(PROMPTS)], "")

    def facts(self) -> dict[str, str]:
        """What the live computation depended on besides the weights."""
        layers = list(self.model.adapted.layers.values())
        flags = {bool(getattr(layer, "in_activation", False)) for layer in layers}
        denoiser_facts = getattr(self.model.denoiser, "runtime_facts", dict)()
        return {"compute": "mixed" if len(flags) > 1 else ("activation" if flags == {True} else "float32"),
                "base_precision": str((self.model.info or {}).get("base_precision", "native")), "kind": self.model.kind,
                "rank": str(self.model.rank), "layers": str(len(layers)), **{k: str(v) for k, v in denoiser_facts.items()}}

    def _noise(self) -> torch.Tensor:
        height, width = self.size[0] // 8, self.size[1] // 8
        return torch.randn((1, 4, height, width), generator=torch.Generator().manual_seed(self.seed))

    def record(self, path: str, step: int, adapter_file: str, extra: Mapping[str, str] | None = None) -> None:
        model, denoiser = self.model, self.model.denoiser
        x = self._noise()
        tensors: dict[str, torch.Tensor] = {"x": x}
        cases: list[dict] = []
        was_training = denoiser.module().training
        denoiser.eval()
        try:
            with torch.no_grad():
                for prompt in self.prompts:
                    conditioning = self.conditioner.encode([prompt], [self.size])
                    for t in TIMESTEPS:
                        n = len(cases)
                        timestep = torch.tensor([t])
                        args = (x.to(model.device), timestep.to(model.device), conditioning)
                        with model.gate.override(None):
                            live = denoiser.predict(*args)
                        with model.gate.override(torch.zeros(1, device=model.device)):
                            base = denoiser.predict(*args)
                        tensors.update({f"t{n}": timestep, f"ctx{n}": conditioning.ctx.float().cpu().clone(), f"y{n}": conditioning.y.float().cpu().clone(),
                                        f"live{n}": live.float().cpu(), f"base{n}": base.float().cpu()})
                        cases.append({"prompt": prompt, "t": t})
        finally:
            denoiser.train(was_training)
        metadata = {"step": step, "adapter": adapter_file, "seed": self.seed, "size": list(self.size), "cases": cases,
                    "state": "adapter as trained, taken before the file was written", **self.facts(), **dict(extra or {})}
        save_file({k: v.contiguous() for k, v in tensors.items()}, path, metadata={k: json.dumps(v) if not isinstance(v, str) else v for k, v in metadata.items()})
