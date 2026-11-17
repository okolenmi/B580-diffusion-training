"""What this project needs installed, and why, as data the setup page and the installer share.

Three tiers: ``required`` (the server does not start without it), ``training`` (the page works but a run does not) and
``optional`` (a feature degrades quietly). A row's ``approx_mb`` is its size on disk once installed (the XPU torch brings its own runtime libraries), None where no honest size is known.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

REQUIRED = "required"
TRAINING = "training"
OPTIONAL = "optional"

TIERS: tuple[str, ...] = (REQUIRED, TRAINING, OPTIONAL)

#: Installed from this index instead of PyPI: PyPI's torch is a CUDA or CPU build, and the trainer runs on the Intel XPU build.
TORCH_INDEX_URL = "https://download.pytorch.org/whl/xpu"
ACCELERATOR_PACKAGES: tuple[str, ...] = ("torch",)
#: The torch release the trainer is tested with on the card; the setup page installs it unless told otherwise.
TORCH_TESTED_VERSION = "2.12.1"


def torch_requirement(choice: str) -> str:
    """The pip entry for a torch choice: ``"tested"``, ``"latest"`` (whatever the XPU index offers) or an exact version like ``2.11.0``."""
    import re

    if choice == "tested":
        return f"torch=={TORCH_TESTED_VERSION}"
    if choice == "latest":
        return "torch"
    if re.fullmatch(r"\d+(\.\d+){1,3}", choice):
        return f"torch=={choice}"
    raise ValueError(f"'{choice}' is not a torch choice; use 'tested', 'latest' or a version such as {TORCH_TESTED_VERSION}")


@dataclass(frozen=True, slots=True)
class Requirement:
    """``distribution`` is the pip name, which is also what ``importlib.metadata`` reports."""

    distribution: str
    tier: str
    why: str
    approx_mb: int | None = None

    @property
    def import_name(self) -> str:
        from .environment import IMPORT_NAMES

        return IMPORT_NAMES.get(self.distribution, self.distribution)


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("fastapi", REQUIRED, "the API framework the server is built on"),
    Requirement("uvicorn", REQUIRED, "the ASGI server that serves it"),
    Requirement("python-multipart", REQUIRED, "multipart body parsing, for the upload endpoints"),
    Requirement("tomli_w", REQUIRED, "writing training config TOML back out"),
    Requirement("packaging", REQUIRED, "PEP 440 version comparison and requirement parsing in the installer"),
    Requirement("torch", TRAINING, "every training step runs through it; without it no run starts (the Intel XPU build)", approx_mb=5400),
    Requirement("numpy", TRAINING, "array maths throughout the trainer and nodes", approx_mb=20),
    Requirement("safetensors", TRAINING, "the only checkpoint format read or written", approx_mb=3),
    Requirement("pillow", TRAINING, "image loading for dataset items", approx_mb=4),
    Requirement("tqdm", TRAINING, "progress bars while datasets are built"),
    Requirement("regex", TRAINING, "CLIP's word splitter needs \\p{L} and \\p{N}, which the standard library's re does not have"),
    Requirement("tokenizers", TRAINING, "Anima's Qwen and T5 tokenizers; without it only Anima text encoding is unavailable"),
    Requirement("transformers", OPTIONAL, "the auto-caption node's local image-to-text backend; imported only when that backend is used"),
)
if sys.platform.startswith("linux"):
    # Its wheel is built for Linux only; elsewhere pip would try to compile the sdist and fail the whole install.
    REQUIREMENTS += (Requirement("xpu-kernels", TRAINING, "fused INT8 and NF4 base-weight kernels, fused GEGLU and the capturable attention for graph replay; "
                                                       "without it those stay off (slower steps, more memory)"),)

#: Every package a fresh environment gets.
FULL_INSTALL: tuple[str, ...] = tuple(r.distribution for r in REQUIREMENTS)


def by_tier(tier: str) -> tuple[Requirement, ...]:
    return tuple(r for r in REQUIREMENTS if r.tier == tier)


def total_approx_mb() -> int:
    """Rough size on disk of a full install, for the disk decision."""
    return sum(r.approx_mb or 0 for r in REQUIREMENTS)
