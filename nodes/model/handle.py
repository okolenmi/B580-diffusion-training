"""Runtime contracts for loaded checkpoints and trainable models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..memory.handle import DeviceResident


class ModelWeights:
    """A checkpoint's UNet state dict, separated from everything else
    (CLIP/VAE) in the same file.

    Lazy by default (Phase 1 of the resources-controller redesign,
    docs/resources_controller_redesign_plan.md): constructed from a
    resolved path, not materialized tensors. unet_sd/non_unet_sd stay
    plain @property attribute access -- every existing consumer
    (ComfyUNetLoRANode, the SDXL text encoder node) reads them exactly
    as it always has; the laziness is invisible to them by design, not
    something they need to opt into. The actual safetensors load only
    happens the first time either property is touched, and is cached
    after that -- so a checkpoint attached to a graph but never built
    from, or only ever asked about via inspect_dtypes() below, never
    pays for a full multi-GB load at all.

    See from_state_dicts() for the eager alternative -- already-
    materialized dicts, no file involved. That's what this class
    unconditionally was before Phase 1; still real and still needed,
    e.g. for a test fixture built directly from synthetic tensors with
    no safetensors file backing it at all.
    """

    def __init__(self, path):
        self._path = path
        self._unet_sd: dict | None = None
        self._non_unet_sd: dict | None = None
        self._loaded = False

    @classmethod
    def from_state_dicts(cls, unet_sd: dict, non_unet_sd: dict) -> "ModelWeights":
        """Already-materialized data, no lazy loading and no path --
        what this class's only constructor did before Phase 1. Real
        use: a test building a fixture directly, not a safetensors
        file being loaded."""
        weights = cls.__new__(cls)
        weights._path = None
        weights._unet_sd = unet_sd
        weights._non_unet_sd = non_unet_sd
        weights._loaded = True
        return weights

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._path is None:
            # Unreachable through either public constructor as written
            # today -- from_state_dicts() always sets _loaded=True, and
            # __init__ always sets a real path. Guarded explicitly
            # anyway: a silent "loads an empty dict" here would be a
            # far worse failure mode than a clear error the moment it
            # actually happens, e.g. if a future refactor adds a third
            # construction path that forgets to set one of the two.
            raise RuntimeError(
                "ModelWeights: no path to load from and not already loaded -- "
                "this should be unreachable through this class's own public "
                "constructors; something constructed an instance in an "
                "inconsistent state."
            )
        from safetensors.torch import load_file
        from .resource_inspection import _is_unet_key

        sd = load_file(str(self._path))
        self._unet_sd = {k: v for k, v in sd.items() if _is_unet_key(k)}
        self._non_unet_sd = {k: v for k, v in sd.items() if not _is_unet_key(k)}
        self._loaded = True

    @property
    def unet_sd(self) -> dict:
        self._ensure_loaded()
        return self._unet_sd

    @property
    def non_unet_sd(self) -> dict:
        self._ensure_loaded()
        return self._non_unet_sd

    def inspect_dtypes(self):
        """Per-component (unet/clip/vae) detected dtype -- see
        nodes/model/resource_inspection.py's inspect_checkpoint_dtypes()
        for the real mechanics (safetensors header only, no tensor data
        touched, and independent of/doesn't affect the lazy full-load
        cache above -- calling this never triggers unet_sd/non_unet_sd
        to materialize, and materializing them never invalidates an
        already-computed inspection).

        Only meaningful for a path-backed instance. A from_state_dicts()
        instance already has real, materialized tensors -- their dtype
        is directly on each tensor (t.dtype), there's nothing left to
        cheaply inspect instead of just reading."""
        if self._path is None:
            raise RuntimeError(
                "ModelWeights.inspect_dtypes(): this instance was built via "
                "from_state_dicts(), not a real file -- there's no header to "
                "peek. Read dtype directly off the tensors you already have "
                "(e.g. next(iter(weights.unet_sd.values())).dtype) instead."
            )
        from .resource_inspection import inspect_checkpoint_dtypes
        return inspect_checkpoint_dtypes(self._path)


class ParameterList(list):
    """A list of trainable tensors, typed distinctly from a plain `Any` so
    an OptimizerNode's `params` input can reject an accidental TrainableModel
    connection outright instead of accepting it (Any accepts everything)
    and crashing later inside the real optimizer constructor. Genuinely
    just a list otherwise -- torch optimizers accept any iterable, so this
    is a free drop-in, not a wrapper callers need to unwrap."""


class TrainedWeightsExportable(ABC):
    """Anything that can hand back a plain, save-ready adapter state dict
    (CPU tensors, keyed for safetensors). What LoRACheckpointSaverNode
    actually depends on. TrainableModel extends this below rather than
    the saver node depending on TrainableModel directly, specifically so
    a second, non-trainable implementer -- a frozen, already-extracted
    snapshot with no forward/train/eval/to (see FrozenLoRASnapshot in
    nodes/model/lora_phases.py) -- is a normal, separately-typed ABC
    implementer the graph's own issubclass() port check accepts at the
    same saver input, not a special case the saver has to know about."""

    @abstractmethod
    def trained_state_dict(self) -> dict:
        ...


class TrainableModel(TrainedWeightsExportable, DeviceResident, ABC):
    """Runtime contract for a model ready to be trained. Extending
    TrainedWeightsExportable means every TrainableModel is required to
    say how to export what it learned -- a reasonable universal
    requirement (not a LoRA-specific one; a hypothetical future
    full-finetune TrainableModel would just export its own full weight
    diff through the same method), and the concrete reason
    LoRACheckpointSaverNode's model input can stay typed at this ABC
    without narrowing to a concrete class.

    Extends DeviceResident (nodes/memory/handle.py) -- footprint_bytes()
    needs a FrozenWeightStore to answer for the frozen base's
    contribution (see nodes/model/frozen_weight_store.py). to()/train()/
    eval() are a general device/dtype move and mode switch;
    offload()/reload()/release() are narrower-purpose siblings: offload/
    reload are specifically the cheap, reversible, stay-resident-on-host
    operation, release is the non-reversible drop."""

    @abstractmethod
    def forward(self, x_t, timestep, context, y):
        ...

    @abstractmethod
    def trainable_parameters(self) -> list:
        ...

    @abstractmethod
    def train(self) -> "TrainableModel":
        ...

    @abstractmethod
    def eval(self) -> "TrainableModel":
        ...

    @abstractmethod
    def to(self, device=None, **kwargs) -> "TrainableModel":
        ...
