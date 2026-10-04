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


#: How far above its measured peak footprint a GPU smoke test's own
#: allocation may climb before an out-of-memory stops being read as
#: contention and is reported as growth instead. 300 MB, not 0: the
#: failing allocation reads under the last successful one (the card
#: refused 16-20 MB against a 10,669 MB peak), and device-free dropped
#: 173 MB across one test's conditioning phase even after its cache was
#: emptied -- driver-level or desktop usage that max_memory_allocated
#: never sees.
OOM_FOOTPRINT_TOLERANCE_MB = 300.0


def _device_memory_mb() -> tuple[float, float, float]:
    """`(peak allocated, free, total)` in MB for the device in use.

    Zeros when there is no device to ask: this runs inside an
    out-of-memory handler, where replacing the failure with a second
    exception would lose the classification the caller asked for.
    """
    import torch

    try:
        if torch.xpu.is_available():
            free_b, total_b = torch.xpu.mem_get_info()
            peak_b = torch.xpu.max_memory_allocated()
        elif torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            peak_b = torch.cuda.max_memory_allocated()
        else:
            return 0.0, 0.0, 0.0
    except Exception as exc:  # noqa: BLE001 -- a broken reading must not
        # replace the out-of-memory being classified. Zeros still
        # classify (peak 0 is inside any real footprint) and are printed
        # as zeros, so they cannot be mistaken for a measurement.
        print(f"  (device memory readings unavailable: {exc})")
        return 0.0, 0.0, 0.0
    return peak_b / 1_048_576, free_b / 1_048_576, total_b / 1_048_576


def oom_outcome(exc: BaseException, *, footprint_mb: float,
                failures: list[str], name: str,
                peak_mb: float | None = None,
                free_mb: float | None = None,
                total_mb: float | None = None) -> int | None:
    """Classify a device out-of-memory: contention gets an exit code,
    growth gets `None` so the caller re-raises.

    The two tests that load the real SDXL UNet in float32 -- float32
    because their claims are float32 tolerances (a 1e-5 merge identity,
    1e-8 gradient floors) that bfloat16 rounding would drown -- measure a
    peak of 10,669 MB allocated on this 12,216 MB card (9,804 MB of
    weights plus a latent-32 backward). Foreign usage was measured at
    1,100-1,500 MB, and a further ~173 MB of non-PyTorch device usage
    appeared across one test's conditioning phase, so how they fit
    depends on a margin they do not own: the same code passed and failed
    on the same day, once with 10 MB of slack.

    An out-of-memory on its own therefore says nothing about whether the
    test changed. What it was holding says more:

    * `peak_mb` at or below `footprint_mb + OOM_FOOTPRINT_TOLERANCE_MB`
      -- the test held what it has always held and foreign usage held
      the rest. Printed as a SKIP carrying every number (peak vs
      footprint, device free, and the out-of-memory's own free/allocated
      line), exit code 0: the posture a run with no accelerator already
      has, and no check is claimed to have passed.
    * Above that -- the test's own allocation grew past its measurement:
      a regression. Printed as such and `None` returned, so the caller
      re-raises and the gate fails.
    * `failures` already recorded are never downgraded by the card
      running out: they print and 1 returns. What failed is still the
      reason the run fails.

    Pass `peak_mb`/`free_mb`/`total_mb` together or leave all three as
    None: the former for a test of this classification that has no card
    to exhaust, the latter in a real handler, which measures.
    """
    if peak_mb is None or free_mb is None or total_mb is None:
        peak_mb, free_mb, total_mb = _device_memory_mb()

    headline = str(exc).strip().splitlines()[0]
    ceiling = footprint_mb + OOM_FOOTPRINT_TOLERANCE_MB
    print("\n" + "=" * 60)

    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S) -- the device OOM "
              f"below is context, not a replacement")
        for failure in failures:
            print(f"  - {failure}")
        print(f"  peak {peak_mb:,.0f} MB against a {footprint_mb:,.0f} MB "
              f"measured footprint; device free {free_mb:,.0f} MB of "
              f"{total_mb:,.0f} MB")
        print(f"  {headline}")
        return 1

    if peak_mb <= ceiling:
        print(f"  SKIP: {name}: the card ran out inside this test's own "
              f"measured footprint -- VRAM contention from foreign "
              f"usage, not a change in the test")
        print(f"    peak allocated {peak_mb:,.0f} MB against the measured "
              f"footprint of {footprint_mb:,.0f} MB "
              f"(+{OOM_FOOTPRINT_TOLERANCE_MB:,.0f} MB tolerance)")
        print(f"    device free {free_mb:,.0f} MB of {total_mb:,.0f} MB")
        print(f"    {headline}")
        print("SMOKE TEST: SKIPPED (VRAM contention; nothing verified)")
        return 0

    print(f"SMOKE TEST: device OOM at {peak_mb:,.0f} MB allocated -- above "
          f"this test's measured footprint of {footprint_mb:,.0f} MB "
          f"(+{OOM_FOOTPRINT_TOLERANCE_MB:,.0f} MB tolerance)")
    print(f"  {name} grew past its own measurement: the test changing, "
          f"not contention, so the exception is re-raised")
    print(f"    device free {free_mb:,.0f} MB of {total_mb:,.0f} MB")
    print(f"    {headline}")
    return None
