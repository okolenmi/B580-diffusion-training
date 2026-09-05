"""LoRATrainingSkeleton + SDXL_LoraTrainer: builds a real, LoRA-injected
unet + clip + vae + lora resources object from a base checkpoint's
state dict, for the not-yet-built training node
(docs/resources_controller_redesign_plan.md's Phase 6) that actually
decides rank/alpha and creates the adapter.

LoRATrainingResources + SDXL_LoRATrainingResources: the earlier stage --
loaded, verified, NOT yet injected -- ResourcesControllerNode
(nodes/model/resources_controller.py, Phase 5) actually produces today.
Two separate classes rather than one doing both stages, direct
correction on an earlier version of this codebase's Resources
Controller node, which had blurred that line by calling inject_lora()
itself: see LoRATrainingResources's own docstring below for the full
reasoning. Named to standardize the two stages' output types under one
family ("...Resources" for the verified-but-uninjected pack Phase 5
produces, "...Skeleton" for the injected, trainable object Phase 6 will
produce from it) rather than leaving Phase 5's own output type
generically named ("VerifiedResourcePack", what it was called before
this naming pass) -- direct feedback that a consistent, LoRA-training-
specific type name here sets up how the next node's own input should
be typed, not just how this one's output happens to be named.

LoRATrainingSkeleton owns everything about LoRA training that doesn't
depend on architecture; SDXLArchitecture (sdxl_architecture.py) owns
the SDXL-specific mechanics. The two combine via inheritance --
SDXL_LoraTrainer(SDXLArchitecture, LoRATrainingSkeleton), base order
required, see that class's own docstring. LoRATrainingResources/
SDXL_LoRATrainingResources below mirror this exact same pattern for the
earlier stage, reusing SDXLArchitecture's own split_checkpoint()/
build_text_encoder() rather than a second implementation of either.

Construction does the real work: __init__ runs the actual pipeline
(merge an optional frozen LoRA into the base weights, split checkpoint,
inject LoRA, build the text encoder, load an optional continue-from
LoRA into the freshly-injected adapter) and the resulting object
already has real .unet/.clip/.vae_sd/.lora attributes -- no separate
build step.

Three methods below are abstract because they're architecture-specific:
an architecture class (SDXLArchitecture) has to provide them.

LoRATrainingSkeleton is also a DeviceResident (nodes/memory/handle.py),
built on nodes/memory/coordinator.py's ResourceCoordinator -- the same
mechanism nodes/train/supervised.py's SupervisedLoRATrainerNode already
uses for its own model/optimizer/text_encoder. .unet and .clip register
with an internal coordinator; footprint_bytes()/offload()/reload()/
release() delegate to it and to .unet/.clip's own DeviceResident
implementations. .vae_sd isn't a DeviceResident (see __init__), so it's
moved/dropped directly alongside the coordinator's work rather than
through it. describe() (below) gives a read-only summary of the same
information -- the "universal interface other nodes may use later"
docs/resources_controller_redesign_plan.md's Phase 5 asks for -- built
entirely out of those same existing methods, not a second parallel
introspection path. LoRATrainingResources implements the same
DeviceResident contract + describe() shape for its own, earlier-stage
fields.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..memory.coordinator import ResourceCoordinator
from ..memory.handle import DeviceResident, sum_tensor_bytes
from .resource_inspection import dtype_to_str
from .sdxl_architecture import SDXLArchitecture


def _lora_rank(lora_sd: dict) -> int | None:
    """The rank a saved LoRA's own lora_down.weight tensors actually
    have -- shared by LoRATrainingResources.describe() below (detected
    information) and nodes/model/lora_training_config.py's own rank-
    locking (Phase 6: a continuing LoRA's shape isn't a free choice,
    nothing here resizes it). None when the modules don't agree on one
    single rank -- a genuinely malformed/mixed file, not a value to
    silently pick one of."""
    ranks = {int(t.shape[0]) for k, t in lora_sd.items() if k.endswith("lora_down.weight")}
    return ranks.pop() if len(ranks) == 1 else None


class LoRATrainingSkeleton(DeviceResident, ABC):

    @abstractmethod
    def split_checkpoint(self, state_dict: dict) -> dict[str, dict]:
        """{"unet": ..., "clip": ..., "vae": ...} -- see
        SDXLArchitecture.split_checkpoint for the implementation."""

    @abstractmethod
    def build_text_encoder(self, clip_sd: dict, device: str):
        """A single object masking however many real text encoders the
        architecture has -- see SDXLArchitecture.build_text_encoder."""

    @abstractmethod
    def inject_lora(self, unet_sd: dict, **kwargs):
        """The adapter-injection entry point -- see
        SDXLArchitecture.inject_lora."""

    def __init__(self, base_model_sd: dict, *, device: str = "xpu", dtype=None,
                 rank: int = 64, alpha: float = 1.0,
                 frozen_lora_sd: dict | None = None, frozen_lora_strength: float = 1.0,
                 continue_lora_sd: dict | None = None,
                 **inject_kwargs):
        """dtype/rank/alpha/**inject_kwargs are resolved values this is
        handed, not detected here -- whatever calls this decides them.

        frozen_lora_sd: a saved LoRA (lora_unet_*.lora_down.weight/
        lora_up.weight/alpha format -- lora_merge.py) merged directly
        into the UNet's base weights before injection. Permanent,
        untrainable: the merged LoRA has no separate identity
        afterward, no object holds it, only its effect on the weights
        remains. frozen_lora_strength scales the merge -- see
        lora_merge.merge_lora_into_state_dict() for exactly what it
        does to the math.

        continue_lora_sd: an existing saved LoRA (same file format as
        frozen_lora_sd) loaded into self.unet's own trainable adapter
        instead of merged into the base -- training continues from
        these weights rather than starting fresh. A different feature
        from frozen_lora_sd: this one stays trainable afterward,
        frozen_lora_sd doesn't and has no separate identity at all.
        Uses load_lora_into_registry() (lora_checkpoint_loader.py) --
        the same validation and loading LoRACheckpointLoaderNode uses
        for the node-graph case, so a missing key or a rank mismatch
        between continue_lora_sd and rank/target_modules above raises
        rather than silently loading a partial LoRA. self.lora is
        continue_lora_sd itself when given, None otherwise -- a plain
        reference (matching self.vae_sd's own raw-dict pattern), not
        the weights themselves, which live inside self.unet's own
        registry once loaded.

        self.vae_sd stays the raw split-out state dict, not a VAE
        object -- nothing in nodes/ builds one yet (only legacy
        core.vae_decode.VAEDecoder, unused elsewhere in nodes/).

        Starts from a raw checkpoint and does the whole pipeline itself
        -- split, merge, inject, build the text encoder, all in one
        call. from_resources() below is the other entry point, for
        nodes/model/resources_controller.py's own
        LoRATrainingResources, which already did everything except
        injection -- the two share _inject() (below) rather than this
        method and that one each having their own copy of "inject, load
        continue_lora, set up the coordinator."
        """
        components = self.split_checkpoint(base_model_sd)
        frozen_lora_merged_count = 0
        if frozen_lora_sd is not None:
            from .lora_merge import merge_lora_into_state_dict
            components["unet"], frozen_lora_merged_count = merge_lora_into_state_dict(
                components["unet"], frozen_lora_sd, strength=frozen_lora_strength)
        clip = self.build_text_encoder(components["clip"], device=device)
        self._inject(components["unet"], clip, components["vae"],
                     device=device, dtype=dtype, rank=rank, alpha=alpha,
                     continue_lora_sd=continue_lora_sd, **inject_kwargs)
        self.frozen_lora_merged_count = frozen_lora_merged_count

    @classmethod
    def from_resources(cls, resources: "LoRATrainingResources", *,
                        rank: int = 64, alpha: float = 1.0, **inject_kwargs):
        """The other entry point -- skips split/frozen-merge/dtype-
        convert/build_text_encoder entirely, because `resources`
        (nodes/model/resources_controller.py's own output, Phase 5)
        already did every one of those. Re-deriving any of them here a
        second time from a raw checkpoint resources itself no longer
        even exposes would be exactly the duplication this whole
        redesign's Consolidation section exists to flag -- this is the
        real reason ResourcesControllerNode and this class stayed two
        separate things rather than one node doing both stages.
        frozen_lora_merged_count is 0 here always -- whatever frozen
        LoRA was merged already happened inside `resources`, this
        object has no visibility into that count, only into its
        already-applied effect on unet_sd."""
        self = cls.__new__(cls)
        self._inject(resources.unet_sd, resources.clip, resources.vae_sd,
                     device=resources._device, dtype=None, rank=rank, alpha=alpha,
                     continue_lora_sd=resources.continue_lora_sd, **inject_kwargs)
        self.frozen_lora_merged_count = 0
        return self

    def _inject(self, unet_sd: dict, clip, vae_sd: dict, *, device: str, dtype,
                continue_lora_sd: dict | None, rank: int, alpha: float, **inject_kwargs):
        """The one real implementation of "inject LoRA into this UNet,
        attach the already-built clip/vae, load an optional continue-
        from LoRA into the freshly-injected adapter, set up device-
        residency" -- __init__ and from_resources() above both end
        here rather than each having their own copy. dtype is None from
        from_resources() (resources.unet_sd is already whatever dtype
        ResourcesControllerNode resolved it to -- converting it again
        here would be a second, redundant conversion, not a correction
        of anything)."""
        self._rank = rank
        self._unet_dtype = dtype
        self.unet = self.inject_lora(unet_sd, device=device, dtype=dtype,
                                      rank=rank, alpha=alpha, **inject_kwargs)
        self.clip = clip
        self.vae_sd = vae_sd

        self.lora = continue_lora_sd
        if continue_lora_sd is not None:
            from .lora_checkpoint_loader import load_lora_into_registry
            load_lora_into_registry(self.unet.raw.lora_registry, continue_lora_sd,
                                     source_description="continue_lora_sd")

        self._device = device

        self._coordinator = ResourceCoordinator()
        self._coordinator.register("unet", self.unet)
        self._coordinator.register("clip", self.clip)


    def footprint_bytes(self) -> int:
        """Coordinator's total (unet + clip) plus vae_sd's raw tensors,
        via the same sum_tensor_bytes() helper other DeviceResident
        implementations in this project use for a list of tensors."""
        return self._coordinator.total_footprint_bytes() + sum_tensor_bytes(self.vae_sd.values())

    def offload(self) -> None:
        """unet/clip via the coordinator's bulk offload; vae_sd's raw
        tensors moved to CPU directly since they aren't registered."""
        self._coordinator.offload_all_except(set())
        self.vae_sd = {k: v.cpu() for k, v in self.vae_sd.items()}

    def reload(self, device: str | None = None) -> None:
        """None reloads to the device this was constructed for."""
        target = device or self._device
        self._coordinator.reload("unet", target)
        self._coordinator.reload("clip", target)
        self.vae_sd = {k: v.to(target) for k, v in self.vae_sd.items()}

    def release(self) -> None:
        """Not reversible. unet/clip via their own release() (each
        moves to CPU first, then drops); vae_sd cleared directly."""
        self.unet.release()
        self.clip.release()
        self.vae_sd = {}

    def describe(self) -> dict[str, dict]:
        """Universal, read-only summary of what this container
        currently holds -- for anything that wants to know what's
        inside without knowing this is specifically a LoRA/SDXL
        trainer: a graph-editor diagnostics call
        (nodes/model/resources_controller.py's ResourcesControllerNode),
        a future preview/monitoring node, a log line. One dict entry
        per resource, {"dtype": str | None, "footprint_bytes": int,
        ...a couple of resource-specific extras}. Built entirely from
        this class's own existing methods (per_resident_footprint_bytes(),
        trainable_parameters(), footprint_bytes()) plus the dtype/rank
        __init__ already resolved and stored -- no new introspection
        logic of its own, so it can't drift from what those already
        report.

        vae's dtype is read directly off one of its own tensors, not
        stored at __init__ -- unlike unet_dtype/rank above, vae_sd's
        dtype is whatever split_checkpoint() found in the source
        checkpoint (this class never converts it, see __init__'s own
        docstring), so there's no "resolved value this was handed" to
        cache; reading it back from the tensors themselves is the only
        source of truth and it's already fully materialized by the time
        describe() could ever run.
        """
        footprints = self._coordinator.per_resident_footprint_bytes()
        lora_params = self.unet.trainable_parameters()
        vae_tensor = next(iter(self.vae_sd.values()), None)
        return {
            "unet": {
                "dtype": dtype_to_str(self._unet_dtype),
                "footprint_bytes": footprints["unet"],
            },
            "clip": {
                "footprint_bytes": footprints["clip"],
            },
            "vae": {
                "dtype": dtype_to_str(vae_tensor.dtype) if vae_tensor is not None else None,
                "footprint_bytes": sum_tensor_bytes(self.vae_sd.values()),
            },
            "lora_adapter": {
                "dtype": dtype_to_str(lora_params[0].dtype) if lora_params else None,
                "rank": self._rank,
                "param_count": sum(p.numel() for p in lora_params),
            },
        }


class SDXL_LoraTrainer(SDXLArchitecture, LoRATrainingSkeleton):
    """LoRA training resources for SDXL. An instance has real
    .unet/.clip/.vae_sd/.lora attributes the moment it's constructed.

    Base order (SDXLArchitecture before LoRATrainingSkeleton) is
    required, not stylistic. Python resolves a method by the first
    match walking the MRO left to right. Listing LoRATrainingSkeleton
    first would mean split_checkpoint/build_text_encoder/inject_lora
    all resolve to its @abstractmethod stubs (found first), which never
    call super() -- so SDXLArchitecture's real implementations would
    never be reached and the class would fail to instantiate at all.
    See this module's smoke test for a negative-case check that the
    wrong order genuinely does fail this way."""
    pass


class LoRATrainingResources(DeviceResident, ABC):
    """Ready-to-use, verified base resources for LoRA training -- NOT
    yet LoRA-injected. That's deliberately a separate, later node's job
    (docs/resources_controller_redesign_plan.md's own Phase 6, not
    built yet) -- direct correction on an earlier version of this
    codebase's Resources Controller node, which called inject_lora()
    itself: rank/alpha/frozen-weight-storage are LoRA-injection
    specifics, not properties of a verified *resource*, and belong on
    whatever node actually creates the adapter (which can then do its
    own real job, like sizing a continued LoRA's adapter to that LoRA's
    own actual rank -- this class has no opinion on rank at all, on
    purpose).

    Exactly four things -- what any LoRA training run over any
    architecture needs to start from, independent of rank/alpha/how the
    frozen base ends up stored once something is actually injected:
    unet_sd (raw state dict, ready to hand to an injector), clip (a
    real, already-loaded, ready-to-use text encoder object --
    build_text_encoder() does real work, loading real weights, and
    isn't LoRA-specific at all, so there's no reason to leave it raw),
    vae_sd (raw state dict -- nothing in nodes/ builds a VAE object
    yet), continue_lora_sd (raw state dict, optional -- there is no
    adapter yet at this stage for it to load into).

    Same split_checkpoint()/build_text_encoder() architecture-specific
    methods LoRATrainingSkeleton above needs, reused (not
    reimplemented) by whichever concrete class mixes in an Architecture
    class -- see SDXL_LoRATrainingResources below for how, the same
    SDXLArchitecture(...) combines with either ABC via multiple
    inheritance in the same shape. Deliberately does NOT need
    inject_lora() at all -- the type system itself reflects the scope
    boundary this class exists to enforce.
    """

    @abstractmethod
    def split_checkpoint(self, state_dict: dict) -> dict[str, dict]:
        """{"unet": ..., "clip": ..., "vae": ...} -- see
        SDXLArchitecture.split_checkpoint for the implementation."""

    @abstractmethod
    def build_text_encoder(self, clip_sd: dict, device: str):
        """A single object masking however many real text encoders the
        architecture has -- see SDXLArchitecture.build_text_encoder."""

    def __init__(self, base_model_sd: dict, *, device: str = "xpu", dtype=None,
                 frozen_lora_sd: dict | None = None, frozen_lora_strength: float = 1.0,
                 continue_lora_sd: dict | None = None):
        """dtype is a resolved value this is handed, not detected here
        -- whatever calls this decides it (e.g. "inherited" resolving
        against the checkpoint's own detected dtype is
        ResourcesControllerNode's own job, not this class's).

        frozen_lora_sd/frozen_lora_strength: merged directly into the
        UNet's base weights right here, before anything else -- see
        LoRATrainingSkeleton's own __init__ docstring for the full
        merge-vs-continue distinction, identical here. Permanent: the
        merged LoRA has no separate identity afterward, which is
        exactly why it isn't one of this class's own four fields.

        continue_lora_sd is stored as given, not validated/loaded here
        -- there's no adapter yet at this stage to load it into or
        validate its rank/shape against; that happens wherever this
        gets injected later.
        """
        components = self.split_checkpoint(base_model_sd)
        self.frozen_lora_merged_count = 0
        if frozen_lora_sd is not None:
            from .lora_merge import merge_lora_into_state_dict
            components["unet"], self.frozen_lora_merged_count = merge_lora_into_state_dict(
                components["unet"], frozen_lora_sd, strength=frozen_lora_strength)
        if dtype is not None:
            components["unet"] = {k: v.to(dtype) for k, v in components["unet"].items()}

        self.unet_sd = components["unet"]
        self.vae_sd = components["vae"]
        self.continue_lora_sd = continue_lora_sd
        self.clip = self.build_text_encoder(components["clip"], device=device)
        self._device = device  # LoRATrainingSkeleton.from_resources()'s own default
        # injection device (nodes/model/lora_training_config.py, Phase 6) -- also
        # reload()'s own fallback below, same pattern LoRATrainingSkeleton uses.
        self._coordinator = ResourceCoordinator()
        self._coordinator.register("clip", self.clip)

    def footprint_bytes(self) -> int:
        return (self.clip.footprint_bytes()
                + sum_tensor_bytes(self.unet_sd.values())
                + sum_tensor_bytes(self.vae_sd.values())
                + (sum_tensor_bytes(self.continue_lora_sd.values())
                   if self.continue_lora_sd else 0))

    def offload(self) -> None:
        self.clip.offload()
        self.unet_sd = {k: v.cpu() for k, v in self.unet_sd.items()}
        self.vae_sd = {k: v.cpu() for k, v in self.vae_sd.items()}
        if self.continue_lora_sd is not None:
            self.continue_lora_sd = {k: v.cpu() for k, v in self.continue_lora_sd.items()}

    def reload(self, device: str | None = None) -> None:
        """None reloads to the device this was constructed for -- was
        hardcoded "xpu" regardless of that, a real bug fixed here rather
        than shipped: reload(None) after offload() on anything built for
        "cpu" (a sandbox with no XPU, say) would have silently moved
        everything to a device that was never actually asked for."""
        target = device or self._device
        self.clip.reload(target)
        self.unet_sd = {k: v.to(target) for k, v in self.unet_sd.items()}
        self.vae_sd = {k: v.to(target) for k, v in self.vae_sd.items()}
        if self.continue_lora_sd is not None:
            self.continue_lora_sd = {k: v.to(target) for k, v in self.continue_lora_sd.items()}

    def release(self) -> None:
        """Not reversible -- same posture as LoRATrainingSkeleton's own."""
        self.clip.release()
        self.unet_sd = {}
        self.vae_sd = {}
        self.continue_lora_sd = None

    def describe(self) -> dict[str, dict]:
        """Same shape/reasoning as LoRATrainingSkeleton.describe()
        above -- built entirely out of this class's own fields/methods,
        no lora_adapter entry (there's no adapter yet), a continue_lora
        entry instead when one was given (dtype/rank detected directly
        off its own tensors -- the same "on the fly verification" the
        Resources Controller's own diagnostics() already does against
        the file before this object even existed, reported here too
        since it's cheap and this is the resulting object's own honest
        self-description, not a claim about the file)."""
        def _component(sd: dict) -> dict:
            tensor = next(iter(sd.values()), None)
            return {
                "dtype": dtype_to_str(tensor.dtype) if tensor is not None else None,
                "footprint_bytes": sum_tensor_bytes(sd.values()),
            }
        result = {
            "unet": _component(self.unet_sd),
            "clip": {"footprint_bytes": self.clip.footprint_bytes()},
            "vae": _component(self.vae_sd),
        }
        if self.continue_lora_sd is not None:
            result["continue_lora"] = {
                **_component(self.continue_lora_sd),
                "rank": _lora_rank(self.continue_lora_sd),
            }
        return result


class SDXL_LoRATrainingResources(SDXLArchitecture, LoRATrainingResources):
    """LoRATrainingResources for SDXL -- same base-order requirement and
    reasoning as SDXL_LoraTrainer above (SDXLArchitecture first, or
    split_checkpoint/build_text_encoder resolve to LoRATrainingResources's
    own @abstractmethod stubs instead of SDXLArchitecture's real ones)."""
    pass
