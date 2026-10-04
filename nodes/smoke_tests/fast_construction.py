"""Constructing a model whose initial values are about to be thrown away.

`nodes/model/unet.py`'s `UNetModel` at SDXL's configuration holds 2.57
billion parameters. Building it runs PyTorch's default initialiser over
every one of them, and that is the single most expensive thing several
smoke tests do:

    UNetModel(**SDXL_CONFIG)            10.38 s
    the same with init skipped            0.05 s

`load_state_dict` then overwrites all of it in 1.22 s, and reading the
6.6 GB checkpoint takes 0.11 s. So in a test that builds the real UNet in
order to load the real weights into it, roughly ten seconds is spent
filling memory that is discarded microseconds later -- and four of the five
builds in `smoke_test_unet.py` never even load weights, comparing shapes or
counting injection targets instead.

**This is only safe when every parameter is about to be overwritten**, and
that is the whole hazard. An uninitialised tensor is whatever was in the
allocator's memory: it can be zeros, it can be a previous test's weights,
and it can be NaN. A model built this way and then used without loading
produces plausible-looking numbers derived from nothing, or NaN, and a
test that asserts "finite" passes or fails for a reason that has nothing to
do with the code under test.

So `assert_fully_covered()` is here to be called after the load, and it
exists because the shortcut is only defensible with the check attached.
Measured on this machine's SDXL checkpoint: 1680 parameters, **1680 of them
present in the file, 0 not**, `load_state_dict` reporting 0 missing and 0
unexpected, every sampled weight equal to the file's, and a forward pass
finite.

Which is why this is a helper and not a change to `UNetModel`: the model
has no idea whether its caller is about to overwrite it, and a constructor
that silently skips initialisation would be a trap in production rather
than a shortcut in a test.

Not used where the initial values *are* the subject -- `smoke_test_clip.py`
and `smoke_test_vae.py` compare forwards against ComfyUI's with weights
drawn from one seeded generator, and there the init is the point.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from torch import nn

__all__ = [
    "assert_fully_covered",
    "find_a_checkpoint",
    "read_state_dicts",
    "skipped_parameter_init",
]

#: Every layer whose `__init__` calls `reset_parameters`. An exhaustive list
#: is not possible to guarantee against future torch versions, so this covers
#: what this project builds and `assert_fully_covered` is the backstop.
_RESETTABLE = tuple(
    cls for cls in (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose2d, nn.Embedding, nn.LayerNorm, nn.GroupNorm,
        nn.BatchNorm1d, nn.BatchNorm2d,
    )
    if hasattr(cls, "reset_parameters")
)


def _do_nothing(self) -> None:  # noqa: ANN001 -- a torch method signature
    """Replacement for `reset_parameters`: leave the memory alone."""


@contextlib.contextmanager
def skipped_parameter_init():
    """Build models under `__init__` without initialising their weights.

    Allocation still happens -- the tensors exist, they are simply not
    filled -- so memory use is unchanged and a subsequent `load_state_dict`
    behaves exactly as before.

    Restores every patched method on the way out, including on an exception:
    a leaked patch would silently change the initialisation of every model
    built for the rest of the process, which is the kind of failure that
    looks like a numerical mystery.
    """
    saved: dict[type, object] = {}
    try:
        for cls in _RESETTABLE:
            saved[cls] = cls.reset_parameters
            cls.reset_parameters = _do_nothing
        yield
    finally:
        for cls, original in saved.items():
            cls.reset_parameters = original


def assert_fully_covered(model: nn.Module, state: dict,
                         name: str = "") -> list[str]:
    """Every tensor is either in `state` or initialised by an adapter.

    The backstop for `skipped_parameter_init`. Anything not covered and not
    an adapter's own would still be holding uninitialised memory, so this
    raises rather than warns: a warning is one scroll from being missed and
    the consequence shows up as a NaN three layers later.

    :param state: the state dict about to be loaded.
    :returns: the tensors accepted as an adapter's own, so a caller can check
        *their* initialisation rather than take it on trust.
    """
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))

    #: Paths whose module carries `lora_A`, i.e. an injected target.
    adapter_roots = {path for path, module in model.named_modules()
                     if hasattr(module, "lora_A")}

    def checkpoint_key(key: str) -> str | None:
        """The state-dict key this tensor came from, or None if it is new.

        LoRA injection *replaces* the target `Linear` with a `LoRALinear`
        that keeps the frozen weight as a **buffer** called `base_weight`
        (and the bias as `base_bias`). So 972 of the checkpoint's keys stop
        matching by name the moment adapters are injected -- 972 measured on
        SDXL at rank 8 -- and a naive coverage check reports them all as
        uninitialised. They are the checkpoint's weights under another name.

        `lora_A` and `lora_B` return None: they are genuinely new, and they
        are initialised by the adapter with explicit `kaiming_uniform_` and
        `zeros_` rather than through the `reset_parameters` this patch
        removes. That is a claim, so the caller is handed the list and
        expected to check it.
        """
        for suffix in (".base_weight", ".base_bias"):
            if key.endswith(suffix):
                return key[:-len(suffix)] + suffix.replace("base_", "")
        return None

    def is_adapter_tensor(key: str) -> bool:
        if key.endswith(("lora_A", "lora_B")):
            return True
        parent = key.rsplit(".", 1)[0]
        return parent in adapter_roots

    def covered(key: str) -> bool:
        equivalent = checkpoint_key(key)
        if equivalent is not None and equivalent in state:
            return True
        return key in state or is_adapter_tensor(key)

    missing = sorted(key for key in tensors if not covered(key))
    if missing:
        prefix = f"{name}: " if name else ""
        raise AssertionError(
            f"{prefix}{len(missing)} of {len(tensors)} tensors are neither in "
            f"the state dict (directly or under an adapter's base_weight) nor "
            f"an adapter's own, so skipping initialisation would leave them "
            f"uninitialised: {missing[:5]}"
        )
    return sorted(key for key in tensors if key not in state)


def find_a_checkpoint(minimum_bytes: int = 1_000_000_000) -> Path | None:
    """The largest SDXL-looking safetensors under ComfyUI's checkpoints.

    `None` rather than raising when there is none, because "this machine has
    no checkpoint" is a legitimate state a test should report as a skip and
    not as a failure.
    """
    try:
        import paths as project_paths
        directory = Path(project_paths.get_comfy_dir()) / "models" / "checkpoints"
    except Exception:  # noqa: BLE001 -- no ComfyUI is a valid state
        return None
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.safetensors"),
                        key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        return None
    return next((c for c in candidates if c.stat().st_size > minimum_bytes),
                candidates[0])


def read_state_dicts(path: Path) -> tuple[dict, dict, dict]:
    """``(unet, conditioner, vae)`` state dicts, keys as the reimplementations
    expect them.

    The three prefixes are exactly what ComfyUI writes into a single file,
    so one read serves all three. 0.11 s for a 6.6 GB file -- safetensors
    memory-maps it, which is why this is never the expensive part.
    """
    from safetensors import safe_open

    unet: dict = {}
    clip: dict = {}
    vae: dict = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith("model.diffusion_model."):
                unet[key[len("model.diffusion_model."):]] = handle.get_tensor(key)
            elif key.startswith("conditioner."):
                clip[key] = handle.get_tensor(key)
            elif key.startswith("first_stage_model."):
                vae[key[len("first_stage_model."):]] = handle.get_tensor(key)
    return unet, clip, vae
