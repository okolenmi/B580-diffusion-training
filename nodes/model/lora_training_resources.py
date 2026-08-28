"""LoRATrainingSkeleton + SDXL_LoraTrainer: the task-specific half of
Phase 4's resources-controller design
(docs/resources_controller_redesign_plan.md), combined with
nodes/model/sdxl_architecture.py's SDXLArchitecture via inheritance --
this project's own choice after working through the real MRO
mechanics for combining two independent axes (task x architecture),
not composition (which was seriously considered -- see this module's
own note on SDXL_LoraTrainer's base ordering for why the alternative
matters here).

LoRATrainingSkeleton owns everything about "what LoRA training needs"
that has nothing to do with which architecture it's running against:
given a base checkpoint's raw state dict (and, later, an existing LoRA
to continue from), pack it into a real unet + clip + vae + lora
resources object -- exactly the "processor" step from the original
design conversation, and exactly why __init__ does real, non-trivial
work here rather than the usual thin-constructor convention: this
class's whole point is that construction *is* processing, so the
resulting object needs no separate build step and carries no
construction machinery with it afterward.

Three methods are declared abstract rather than implemented here
because they're architecture-specific, not task-specific -- an
architecture mixin (SDXLArchitecture today) has to provide them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .sdxl_architecture import SDXLArchitecture


class LoRATrainingSkeleton(ABC):

    @abstractmethod
    def split_checkpoint(self, state_dict: dict) -> dict[str, dict]:
        """{"unet": ..., "clip": ..., "vae": ...} -- see
        SDXLArchitecture.split_checkpoint for the real implementation
        this skeleton depends on but doesn't provide itself."""

    @abstractmethod
    def build_text_encoder(self, clip_sd: dict, device: str):
        """A single object masking however many real text encoders the
        architecture actually has -- see
        SDXLArchitecture.build_text_encoder."""

    @abstractmethod
    def inject_lora(self, unet_sd: dict, **kwargs):
        """The real adapter-injection entry point -- see
        SDXLArchitecture.inject_lora."""

    def __init__(self, base_model_sd: dict, *, device: str = "xpu", dtype=None,
                 rank: int = 64, alpha: float = 1.0, **inject_kwargs):
        """The processor step. dtype/rank/alpha/**inject_kwargs are
        resolved DECISIONS this is handed, not detected here -- see
        docs/resources_controller_redesign_plan.md's Phase 4 design: a
        Resources Controller node's own dtype-choice UI (or, for now, a
        direct caller) is what decides these; this class only acts on
        the decision, exactly like SDXLArchitecture never touches dtype
        either.

        Real, deliberately deferred rather than half-built: "continue
        training" (loading an existing saved LoRA's weights into the
        freshly-injected model, per the original design sketch's own
        "Continue training" checkbox) isn't implemented in this pass --
        self.lora stays None regardless of any future lora_sd-shaped
        argument, since there isn't one yet. The real loading mechanics
        already exist (nodes/model/lora_checkpoint_loader.py's
        LoRACheckpointLoaderNode, DoRA-aware since two patches ago) and
        this class's own inject_lora()-produced self.unet is exactly
        the kind of freshly-injected model that loader operates on --
        wiring them together is a real, separate next increment, not
        attempted here to keep this one honestly scoped.

        No VAE object either -- self.vae_sd stays the raw split-out
        state dict. Nothing in nodes/ builds a VAE wrapper today (only
        core.vae_decode.VAEDecoder, legacy, unused by anything in
        nodes/) -- a real, current gap, not something invented here
        just to fill out this class's shape.
        """
        components = self.split_checkpoint(base_model_sd)
        self.unet = self.inject_lora(components["unet"], device=device, dtype=dtype,
                                      rank=rank, alpha=alpha, **inject_kwargs)
        self.clip = self.build_text_encoder(components["clip"], device=device)
        self.vae_sd = components["vae"]
        self.lora = None

    def footprint_bytes(self) -> int:
        """Matches the pattern already established by
        ComfyUNetTrainableModel.footprint_bytes()/FrozenWeightStore.footprint_bytes()
        -- summed from self.unet's and self.clip's own real accessors
        (both already implement this), plus vae_sd's raw tensors
        directly, since there's no object wrapping them yet to ask."""
        total = self.unet.footprint_bytes() + self.clip.footprint_bytes()
        total += sum(t.numel() * t.element_size() for t in self.vae_sd.values())
        return total


class SDXL_LoraTrainer(SDXLArchitecture, LoRATrainingSkeleton):
    """The concrete "LoRA training on SDXL" resources object -- an
    instance has real .unet/.clip/.vae_sd/.lora attributes the moment
    it's constructed, nothing about how it was built riding along, per
    the original design conversation's own "beauty of this method"
    framing.

    Base ordering (SDXLArchitecture before LoRATrainingSkeleton) is
    load-bearing, not stylistic. Python resolves a method by the first
    match walking the MRO left to right -- listing LoRATrainingSkeleton
    first would mean split_checkpoint/build_text_encoder/inject_lora
    all resolve to its own @abstractmethod stubs (found first in the
    MRO), which never call super(), so SDXLArchitecture's real
    implementations -- sitting right there as the second base -- would
    never actually be reached, and this class would fail to instantiate
    at all ("Can't instantiate abstract class"). This exact ordering
    was worked out and confirmed correct before landing, not assumed --
    see check_wrong_base_order_fails_to_instantiate in this module's
    smoke test for the negative-case proof, not just the positive one.
    """
    pass
