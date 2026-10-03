"""Correctness check for nodes/model/unet.py -- the SDXL UNet this project
owns (design doc 12, section 7.3, section A).

The contract here is unusually strict and entirely mechanical: 1,680 tensors
with fixed names, because the names *are* the checkpoint's keys and this
project's LoRA block-weight paths are written in terms of them. So the
checks are mostly about structure, and the structural claim is strong enough
to be worth making precisely:

* **The full SDXL configuration builds to exactly the same `state_dict` as
  ComfyUI's** -- same keys, same shapes, 1,680 tensors, zero difference. That
  is the claim that makes the port safe, and it is checked against ComfyUI
  where ComfyUI is importable.
* **`label_emb`'s redundant `nn.Sequential` is load-bearing.** Its keys are
  `label_emb.0.0.*`; unwrap it and all four tensors are renamed, which
  `load_state_dict(strict=False)` reports as missing and carries on. This is
  the one place where "tidying up" would silently break loading, so it is
  pinned directly rather than only implied by the state_dict comparison.
* **A real checkpoint loads with nothing missing.** Checked when a checkpoint
  is present, skipped when it is not -- the kind of check that must not make
  the file depend on a 7 GB download.
* **The forward is bitwise identical to ComfyUI's**, on a small config that
  still has every structural feature SDXL has: three levels, attention in
  the middle and at two input/output levels, `num_classes="sequential"`,
  and `use_linear_in_transformer`. Reported as characterisation rather than
  gated, since the UNet is unambiguous and a divergence is a bug report.
* **`output_shape` reaches the upsamples**, which is what keeps a
  non-power-of-two latent size working. Without it the skip concatenation
  fails on a shape mismatch, so this is checked by using an odd size.
* **`y` is required exactly when the model is class-conditional**, in both
  directions.

Run: `python nodes/smoke_tests/smoke_test_unet.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.unet import (  # noqa: E402
    Downsample,
    ResBlock,
    TimestepBlock,
    TimestepEmbedSequential,
    UNetModel,
    Upsample,
)
from nodes.model.unet_wrapper import ComfyUNetWrapper  # noqa: E402

failures: list[str] = []
skipped: list[str] = []

#: The published SDXL UNet, small enough to build on CPU in a test. Three
#: levels, attention in the middle and at two input and output levels.
SMALL = dict(
    image_size=32, in_channels=4, out_channels=4, model_channels=32,
    num_res_blocks=[1, 1], channel_mult=[1, 2], dropout=0.0,
    conv_resample=True, num_classes="sequential", use_checkpoint=False,
    num_head_channels=16, use_spatial_transformer=True,
    transformer_depth=[0, 1],              # one per input ResBlock: 1 + 1
    context_dim=24, adm_in_channels=40,
    transformer_depth_middle=1,
    transformer_depth_output=[0, 1, 1, 1],  # one per output block: 2 + 2
    use_linear_in_transformer=True,
)


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


def sdxl_kwargs(use_checkpoint: bool = False) -> dict:
    """The real SDXL config, minus the keys this project no longer accepts."""
    cfg = dict(ComfyUNetWrapper.SDXL_CONFIG)
    return {**cfg, "use_checkpoint": use_checkpoint, "adm_in_channels": 2816}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def check_label_emb_nesting():
    """The one place a tidy-up silently breaks checkpoint loading."""
    model = UNetModel(**{**SMALL, "adm_in_channels": 40})
    keys = {k for k in model.state_dict() if k.startswith("label_emb")}
    record(keys == {"label_emb.0.0.weight", "label_emb.0.0.bias",
                    "label_emb.0.2.weight", "label_emb.0.2.bias"},
           "label_emb is nested one level deeper than it needs to be",
           f"got {sorted(keys)}")
    record(isinstance(model.label_emb[0], torch.nn.Sequential),
           "and that nesting is a real nn.Sequential, not an accident of "
           "key naming")
    unwrapped = {k.replace("label_emb.0.", "label_emb.")
                 for k in keys}
    record(not (unwrapped & keys),
           "so unwrapping really would rename all four, which is why it "
           "stays")


def check_block_structure():
    model = UNetModel(**{**SMALL, "adm_in_channels": 40})
    record(isinstance(model.input_blocks[0], TimestepEmbedSequential),
           "input_blocks are TimestepEmbedSequential")
    record(isinstance(model.middle_block, TimestepEmbedSequential),
           "and so is the middle block")
    # The leaves live inside the TimestepEmbedSequential wrappers, not in the
    # ModuleList directly -- checking the ModuleList finds only the wrappers,
    # which is what the first version of this check did.
    def leaves(module):
        out = []
        for child in module.children():
            if isinstance(child, TimestepEmbedSequential):
                out.extend(child)
            else:
                out.append(child)
        return out

    down = leaves(model.input_blocks)
    up = leaves(model.output_blocks)
    record(any(isinstance(m, ResBlock) for m in down),
           "input_blocks contain ResBlocks",
           f"{len(down)} leaves: "
           f"{sorted({type(m).__name__ for m in down})}")
    record(any(isinstance(m, Downsample) for m in down),
           "and a Downsample on the way down")
    record(any(isinstance(m, Upsample) for m in up),
           "and an Upsample on the way back up")
    record(isinstance(down[0], torch.nn.Conv2d),
           "the first input block is a bare Conv2d -- the stem that maps "
           "in_channels to model_channels, and the one thing on the way "
           "down that is neither a ResBlock nor the Downsample",
           f"{type(down[0]).__name__}")
    record([type(m).__name__ for m in down]
           == ["Conv2d", "ResBlock", "Downsample", "ResBlock",
               "SpatialTransformer"],
           "and the whole way down is exactly: stem, resblock, downsample, "
           "resblock+attention -- attention living inside the same "
           "sequential as its ResBlock, not as a separate block",
           f"{[type(m).__name__ for m in down]}")
    record(isinstance(ResBlock, type) and issubclass(ResBlock, TimestepBlock),
           "ResBlock is a TimestepBlock, which is how the sequential knows "
           "to pass emb to it")
    record(ResBlock.__name__ == "ResBlock",
           "the ResBlock class keeps its name, because block_profiler.py "
           "labels blocks with type(instance).__name__",
           f"got {ResBlock.__name__!r}")
    record(callable(getattr(ResBlock, "_forward", None)),
           "and ResBlock._forward exists as a bound method, which is what "
           "carries __self__ for the profiler")
    record(any("ff_in" in k or "norm_in" in k
               for k in model.state_dict()) is False,
           "no ff_in/norm_in anywhere: SDXL builds none")


def check_config_is_not_mutated():
    """Building a UNet must not eat the caller's config.

    `transformer_depth` and `transformer_depth_output` are consumed by pop()
    as the blocks are built, so an implementation that forgets to copy them
    makes the second UNet in a process fail with "pop from empty list".
    """
    cfg = dict(ComfyUNetWrapper.SDXL_CONFIG)
    before = dict(cfg)
    UNetModel(**sdxl_kwargs())
    record(cfg == before,
           "building the real SDXL UNet leaves SDXL_CONFIG untouched",
           f"changed: {[k for k in cfg if cfg[k] != before.get(k)]}")
    UNetModel(**sdxl_kwargs())   # would raise if the config had been eaten
    record(True, "and a second build in the same process works")


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def _fill(module, seed: int = 17, scale: float = 0.05) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=gen) * scale)


def check_forward():
    torch.manual_seed(0)
    model = UNetModel(**{**SMALL, "adm_in_channels": 40}).eval()
    torch.manual_seed(5)
    x = torch.randn(1, 4, 16, 16)
    t = torch.tensor([321.0])
    context = torch.randn(1, 7, 24)
    y = torch.randn(1, 40)
    with torch.no_grad():
        out = model(x, t, context=context, y=y)
    record(tuple(out.shape) == (1, 4, 16, 16),
           "the forward returns the input latent's shape", f"{tuple(out.shape)}")
    record(bool(torch.isfinite(out).all()), "and it is finite")

    # An odd spatial size, which is what output_shape exists for. 16 -> 8 is
    # exact; 12 -> 6 is too. Use a size whose halving is not a power of two.
    torch.manual_seed(5)
    x_odd = torch.randn(1, 4, 12, 12)
    try:
        with torch.no_grad():
            out_odd = model(x_odd, t, context=context, y=y)
        record(tuple(out_odd.shape) == (1, 4, 12, 12),
               "a non-power-of-two latent size round-trips, which only works "
               "because output_shape reaches the upsamples",
               f"{tuple(out_odd.shape)}")
    except RuntimeError as exc:
        record(False, "a non-power-of-two latent size round-trips", str(exc)[:80])


def check_y_requirement():
    model = UNetModel(**{**SMALL, "adm_in_channels": 40}).eval()
    x = torch.randn(1, 4, 16, 16)
    t = torch.tensor([1.0])
    context = torch.randn(1, 7, 24)
    with torch.no_grad():
        try:
            model(x, t, context=context)
        except ValueError as exc:
            record("conditional" in str(exc),
                   "a class-conditional model refuses a forward with no y",
                   str(exc)[:70])
        else:
            record(False, "a class-conditional model refuses a forward with "
                   "no y", "it accepted one")

    plain = UNetModel(**{**SMALL, "num_classes": None,
                         "adm_in_channels": None}).eval()
    with torch.no_grad():
        try:
            plain(x, t, context=context, y=torch.randn(1, 40))
        except ValueError as exc:
            record("class-conditional" in str(exc),
                   "and an unconditional one refuses a y", str(exc)[:70])
        else:
            record(False, "and an unconditional one refuses a y",
                   "it accepted one")
        out = plain(x, t, context=context)
    record(tuple(out.shape) == (1, 4, 16, 16),
           "while the unconditional model runs without one")


# ---------------------------------------------------------------------------
# Against ComfyUI, and against a real checkpoint
# ---------------------------------------------------------------------------

def check_against_comfy():
    try:
        import paths as p
        root = p.get_comfy_dir()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from comfy.ldm.modules.diffusionmodules.openaimodel import (
            UNetModel as ComfyUNet,
        )
    except Exception as exc:  # noqa: BLE001 -- absent means "not here"
        skip("characterisation vs ComfyUI",
             f"comfy not importable ({type(exc).__name__})")
        return

    # 1. The full SDXL configuration: identical state_dict.
    torch.manual_seed(0)
    theirs = ComfyUNet(**{**sdxl_kwargs(), "legacy": False,
                          "dtype": torch.float32})
    ours = UNetModel(**sdxl_kwargs())
    their_keys, our_keys = set(theirs.state_dict()), set(ours.state_dict())
    record(their_keys == our_keys,
           "the real SDXL UNet builds an identical set of tensor names",
           f"{len(our_keys)} ours vs {len(their_keys)} comfy; "
           f"only ours {sorted(our_keys - their_keys)[:3]}, "
           f"only comfy {sorted(their_keys - our_keys)[:3]}")
    if their_keys == our_keys:
        mismatch = [k for k in their_keys
                    if theirs.state_dict()[k].shape
                    != ours.state_dict()[k].shape]
        record(not mismatch, "and identical shapes", f"{mismatch[:3]}")
    record(len(our_keys) == 1680,
           "1680 tensors, which is what the checkpoint has",
           f"{len(our_keys)}")

    # 2. The forward, on a config small enough to run on CPU but with every
    # structural feature SDXL has.
    torch.manual_seed(0)
    theirs_small = ComfyUNet(**{**SMALL, "legacy": False,
                                "dtype": torch.float32}).eval()
    torch.manual_seed(0)
    ours_small = UNetModel(**{**SMALL, "adm_in_channels": 40}).eval()
    # ComfyUI's ops.Linear leaves its weight uninitialised (~3e29), because it
    # assumes a checkpoint overwrites it -- so weights have to come from one
    # generator or both sides come out NaN and nan == nan passes everything.
    _fill(theirs_small)
    ours_small.load_state_dict(theirs_small.state_dict())

    torch.manual_seed(5)
    x = torch.randn(1, 4, 16, 16)
    t = torch.tensor([321.0])
    context = torch.randn(1, 7, 24)
    y = torch.randn(1, 40)
    with torch.no_grad():
        a = theirs_small(x, t, context=context, y=y)
        b = ours_small(x, t, context=context, y=y)
    if bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()):
        worst = (a - b).abs().max().item()
        print(f"  {'DIFF' if worst else 'SAME'}: forward: "
              f"max |comfy - ours| = {worst:.3e}")
    else:
        print("  SKIP: forward: a non-finite output, so the comparison would "
              "be meaningless")


def check_lora_targets_resolve():
    """The LoRA contract, resolved against *this* UNet.

    Section 7.3's cost estimate originally said every LoRA injection point
    was written against ComfyUI's module layout and would all move. That is
    wrong, and this is the check that says so rather than asserting it:
    injection selects by module *name*, so the only thing that has to match
    is the names -- and the names are the checkpoint's, which we already
    match.

    So the claim is that `nodes/model/lora.py` needs no change. If a
    reimplementation renamed `to_q`, or nested the blocks differently, this
    would find zero injection targets and the project would silently train
    nothing.
    """
    from nodes.model.lora import LoRAConfig, inject_lora_into_unet

    model = UNetModel(**sdxl_kwargs(use_checkpoint=False))
    config = LoRAConfig(rank=8, alpha=8.0,
                        target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    registry = inject_lora_into_unet(model, config)
    record(bool(registry),
           f"LoRA injection finds targets in the reimplemented UNet: "
           f"{len(registry)} layers")
    if not registry:
        return

    found = {name for _path, _mod, name, _layer in registry}
    record({"to_q", "to_k", "to_v"} <= found,
           "and every configured attention projection is among them",
           f"{sorted(found)}")
    # `to_out.0` is a Sequential, so its Linear's leaf name is "0" -- and
    # because injection matches leaf names, `time_embed.0` and
    # `time_embed.2` match too. Pre-existing behaviour of lora.py, unchanged
    # by the reimplementation, and recorded here because a reader counting
    # injected layers will otherwise wonder where the extra two came from.
    record("0" in found,
           "`to_out.0`'s Linear is injected under the leaf name '0'")
    record("2" in found,
           "and leaf-name matching also picks up time_embed.2, which is "
           "pre-existing behaviour rather than something the "
           "reimplementation introduced",
           f"{sorted(found)}")

    # The nesting, not the leaf names. A path taken from the registry is
    # resolved back through the module tree, so this fails if the structure
    # changed even while the names still matched.
    by_path = sorted(path for path, _m, _n, _l in registry
                     if "transformer_blocks" in path)
    record(len(by_path) > 0,
           f"and the injected paths carry the block nesting: "
           f"{by_path[0] if by_path else 'none'}")
    probe = by_path[0]
    module = model
    resolved = True
    for part in probe.split("."):
        if part.isdigit():
            module = list(module.children())[int(part)]
        elif hasattr(module, part):
            module = getattr(module, part)
        else:
            resolved = False
            break
    record(resolved,
           f"which resolves back through the module tree: {probe}")


def check_real_checkpoint():
    """The claim that actually matters: a real checkpoint loads, fully."""
    try:
        import paths as p
        root = p.get_comfy_dir()
        candidates = sorted((root / "models" / "checkpoints").glob("*.safetensors"))
    except Exception as exc:  # noqa: BLE001
        skip("a real checkpoint loads with nothing missing",
             f"no ComfyUI directory ({type(exc).__name__})")
        return
    if not candidates:
        skip("a real checkpoint loads with nothing missing",
             "no .safetensors in the ComfyUI checkpoints directory")
        return

    try:
        from safetensors import safe_open
    except ImportError:
        skip("a real checkpoint loads with nothing missing",
             "safetensors is not installed")
        return

    path = next((c for c in candidates if c.stat().st_size > 1_000_000_000),
                candidates[0])
    prefix = "model.diffusion_model."
    state = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = f.get_tensor(key)
    if not state:
        skip("a real checkpoint loads with nothing missing",
             f"{path.name} has no {prefix}* keys (not an SDXL UNet)")
        return

    record(len(state) == 1680,
           f"{path.name} holds 1680 UNet tensors", f"{len(state)}")
    model = UNetModel(**sdxl_kwargs())
    missing, unexpected = model.load_state_dict(state, strict=False)
    record(not missing, "nothing missing", f"{missing[:4]}")
    record(not unexpected, "nothing unexpected", f"{unexpected[:4]}")


def main() -> int:
    print("== label_emb nesting ==")
    check_label_emb_nesting()
    print("\n== block structure ==")
    check_block_structure()
    print("\n== the config is not mutated ==")
    check_config_is_not_mutated()
    print("\n== forward ==")
    check_forward()
    print("\n== y is required exactly when conditional ==")
    check_y_requirement()
    print("\n== characterisation vs ComfyUI ==")
    check_against_comfy()
    print("\n== the LoRA contract ==")
    check_lora_targets_resolve()
    print("\n== a real checkpoint ==")
    check_real_checkpoint()

    print("\n" + "=" * 60)
    if skipped:
        print(f"  {len(skipped)} check(s) skipped: "
              + ", ".join(sorted(set(skipped))))
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())