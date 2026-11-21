"""Encode Prompt: a positive and a negative prompt through the text encoders, which leave the device when it is done."""

from __future__ import annotations

from typing import Any

import torch

from engine.devices import Accelerator, part_dtype, resolve_device
from engine.handles import PromptEncoding
from families.api.parts import ModelPart
from families.registry import get_family
from graph import Node, Port


class EncodePromptNode(Node):
    DISPLAY_NAME = "Encode Prompt"
    CATEGORY = "Generation"
    READS_SETTINGS = ("device.default", "precision.encoders")
    INPUTS = {
        "encoders": Port("encoders", ModelPart, multi=True, doc="The text-encoder parts of the model (wire each)."),
        "prompt": Port("prompt", str, required=False, default="", ui={"widget": "textarea", "rows": 3}),
        "negative": Port("negative", str, required=False, default="", ui={"widget": "textarea", "rows": 2},
                         doc="What the image should move away from. Empty is the model's empty prompt."),
    }
    OUTPUTS = {"encoding": Port("encoding", PromptEncoding)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        parts = {p.role: p for p in inputs["encoders"]}
        families = {p.family_id for p in parts.values()}
        if len(families) != 1:
            raise ValueError(f"Encode Prompt: the encoders come from different families ({sorted(families)})")
        family = get_family(families.pop())
        device = resolve_device(self.context.settings.get("device.default"))
        setting = self.context.settings.get("precision.encoders")
        dtype = {role: part_dtype(p.dtype, setting, torch.float16) for role, p in parts.items()}
        prompt, negative = inputs.get("prompt") or "", inputs.get("negative") or ""
        with self.context.task("Encoding the prompt"):
            encoder = family.load_text_encoder(parts, device="cpu", dtype=dtype)
            kit = family.conditioning_kit(encoder)
            encoder.to(device)
            try:
                with torch.no_grad():
                    positive, opposite = (text.to("cpu") for text in kit.encode_text([prompt, negative]))
            finally:
                encoder.to("cpu")  # the kit keeps the towers for assembling sizes; they wait on the host
                Accelerator(device).empty_cache()
        return {"encoding": PromptEncoding(family, kit, positive, opposite, prompt, negative)}
