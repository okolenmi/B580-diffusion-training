"""Load Model: one node for every supported model file.

The scope only picks a folder. What the file holds is read from its header, and the node's output
ports are the parts found: an SDXL checkpoint gives four, a VAE file gives one. Nothing is loaded
into memory here; the outputs are handles that consumers load when they need them.
"""

from __future__ import annotations

from typing import Any

from ...families.api.detection import Detection
from ...families.api.library import SCOPES
from ...families.api.parts import ROLE_CLASSES, make_part
from ...families.registry import FAMILY_IDS, detect_file
from ...graph import DynamicShape, Environment, Node, Port, Resolution

_SERVICE = "models"
_DTYPES = ("auto", "bf16", "fp16", "fp32")

_OUTPUT_DOC = {
    "denoiser": "The denoising network (UNet or DiT, by family).",
    "clip_l": "CLIP-L text encoder.",
    "clip_g": "CLIP-G text encoder.",
    "qwen3": "Qwen3 text encoder.",
    "vae": "Latent autoencoder.",
    "adapter": "Sub-network stored inside the denoiser file.",
}


def _output_ports(detection: Detection) -> dict[str, Port]:
    return {
        part.role: Port(part.role, ROLE_CLASSES[part.role], label=detection.label_of(part.role),
                        doc=_OUTPUT_DOC.get(part.role, ""))
        for part in detection.parts
    }


def _detect(params: dict, env: Environment | None) -> tuple[Detection | None, str | None]:
    """(detection, problem). A missing file is not a problem: the node simply has no outputs yet."""
    scope = params.get("scope") or "checkpoints"
    relative = params.get("file") or ""
    if scope not in SCOPES:
        return None, f"unknown scope {scope!r}"
    if not relative:
        return None, None
    if env is None or not env.has(_SERVICE):
        return None, "no model library is configured on this machine"
    family = params.get("family") or "auto"
    return detect_file(env.service(_SERVICE), scope, relative, family), None


class LoadModelNode(Node):
    DISPLAY_NAME = "Load Model"
    CATEGORY = "Model"
    DYNAMIC = DynamicShape(depends_on=("scope", "file", "family"))

    INPUTS = {
        "scope": Port("scope", str, required=False, default="checkpoints", choices=SCOPES,
                      doc="Which folder list to browse. It does not decide what the file contains."),
        "file": Port("file", str, required=False, default="", picker="model_file",
                     doc="A .safetensors file under the scope's folders (subfolders allowed)."),
        "family": Port("family", str, required=False, default="auto", choices=("auto", *FAMILY_IDS),
                       doc="Leave on auto unless the file matches several families."),
        "dtype": Port("dtype", str, required=False, default="auto", choices=_DTYPES, advanced=True,
                      doc="Requested compute dtype. auto follows the graph setting, then the file."),
    }
    OUTPUTS: dict[str, Port] = {}

    @classmethod
    def resolve(cls, params: dict, env: Environment | None = None) -> Resolution:
        inputs = dict(cls.INPUTS)
        detection, problem = _detect(params, env)
        if problem:
            return Resolution("error", inputs, {}, message=problem,
                              focus="scope" if "scope" in problem else None)
        if detection is None:
            return Resolution("ready", inputs, {}, info="choose a file")
        if detection.status == "error":
            focus = "file" if detection.error_kind in {"no_file", "outside_scope", "missing", "no_folder"} else None
            return Resolution("error", inputs, {}, message=detection.message, focus=focus,
                              detail={"kind": detection.error_kind, "report": list(detection.report)})
        if detection.status == "needs_choice":
            tied = tuple(m.family_id for m in detection.matches
                         if abs(m.score - detection.matches[0].score) < 1e-9)
            return Resolution("needs_choice", inputs, {}, choices={"family": tied}, focus="family",
                              message=detection.message)
        return Resolution("ready", inputs, _output_ports(detection), info=detection.summary(),
                          detail={"family": detection.family_id, "warnings": list(detection.warnings),
                                  "parts": [{"role": p.role, "dtype": p.dtype, "bytes": p.nbytes,
                                             "keys": p.key_count} for p in detection.parts]})

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        env = self.context.env
        detection, problem = _detect(inputs, env)
        if problem or detection is None:
            raise ValueError(problem or "Load Model: no file chosen")
        if not detection.ok:
            raise ValueError(f"Load Model: {detection.message}")
        library = env.service(_SERVICE)
        path = str(library.resolve(inputs.get("scope") or "checkpoints", inputs["file"]))
        dtype = inputs.get("dtype") or "auto"
        return {p.role: make_part(detection.family_id, path, p, dtype) for p in detection.parts}
