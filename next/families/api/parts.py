"""What Load Model hands out: lazy references to one part of a model file.

A part is a *description* (which file, which key prefix, what was found there); no tensor is read
until a consumer loads it. Each role has its own class so the graph's wire check refuses, say, a
Qwen3 encoder wired into a CLIP-G input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class PartKind(str, Enum):
    DENOISER = "denoiser"
    TEXT_ENCODER = "text_encoder"
    VAE = "vae"
    ADAPTER = "adapter"
    OTHER = "other"


@dataclass(frozen=True)
class PartInfo:
    """What was found in a header for one part."""

    kind: PartKind
    role: str
    prefix: str
    key_count: int
    dtype: str | None
    nbytes: int
    config: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class PartSource:
    path: str
    prefix: str


@dataclass(frozen=True)
class AdapterMerge:
    """A saved adapter to bake into the weights when the part is loaded."""

    file: str
    strength: float = 1.0
    format: str = "kohya"


@dataclass(frozen=True)
class ModelPart:
    family_id: str
    role: str
    source: PartSource
    info: PartInfo
    dtype: str = "auto"  # the dtype the person asked for; 'auto' leaves the choice to the loader
    merges: tuple[AdapterMerge, ...] = ()  # applied in order by the loader of a denoiser

    ROLE: ClassVar[str] = ""
    KIND: ClassVar[PartKind] = PartKind.OTHER

    def __post_init__(self) -> None:
        if self.ROLE and self.role != self.ROLE:
            raise ValueError(f"{type(self).__name__} holds role {self.ROLE!r}, got {self.role!r}")


class DenoiserPart(ModelPart):
    ROLE = "denoiser"
    KIND = PartKind.DENOISER


class ClipLPart(ModelPart):
    ROLE = "clip_l"
    KIND = PartKind.TEXT_ENCODER


class ClipGPart(ModelPart):
    ROLE = "clip_g"
    KIND = PartKind.TEXT_ENCODER


class Qwen3Part(ModelPart):
    ROLE = "qwen3"
    KIND = PartKind.TEXT_ENCODER


class VAEPart(ModelPart):
    ROLE = "vae"
    KIND = PartKind.VAE


class AdapterPart(ModelPart):
    """A sub-network that lives inside another file (the LLM adapter inside an Anima DiT)."""

    ROLE = "adapter"
    KIND = PartKind.ADAPTER


ROLE_CLASSES: dict[str, type[ModelPart]] = {
    cls.ROLE: cls for cls in (DenoiserPart, ClipLPart, ClipGPart, Qwen3Part, VAEPart, AdapterPart)
}

ROLE_LABELS = {"denoiser": "denoiser", "clip_l": "CLIP_L", "clip_g": "CLIP_G", "qwen3": "Qwen3",
               "vae": "VAE", "adapter": "LLM adapter"}


def make_part(family_id: str, path: str, info: PartInfo, dtype: str = "auto") -> ModelPart:
    cls = ROLE_CLASSES.get(info.role)
    if cls is None:
        raise KeyError(f"no part class for role {info.role!r}")
    return cls(family_id, info.role, PartSource(path, info.prefix), info, dtype)
