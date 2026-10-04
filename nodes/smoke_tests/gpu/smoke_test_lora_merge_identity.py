"""Merging a LoRA must not change what the model computes.

Design doc 12, section 7.3. Merging folds `B @ A * scale` into the base
weight and discards the adapter, which is how a trained LoRA gets exported
as a plain checkpoint. It is a mathematical rewrite of the forward pass, and
nothing else in the suite checks it:

* `smoke_test_adapter_injection.py` covers targeting -- which modules get
  wrapped.
* `smoke_test_lora_injector_extraction.py` covers getting the adapters back
  out of the registry.
* The three LoRA *training* tests install a stub `comfy` module so the
  checkpointing patches can import, and none of them runs a forward.

So a merge that is subtly wrong -- a transposed factor, a missing `alpha`
scale, the wrong multiplier -- would produce a valid-looking checkpoint that
conditions every downstream image slightly wrongly, and nothing would fail.
That is the same class of defect as the five ComfyUI bugs section 7.3 found
and did not copy, and it is the reason this file exists.

**The identity under test is exact and hardware-independent:** merging is
supposed to leave the function unchanged, so the forward output before and
after must be the same *to within float32 rounding*, on any device, with no
reference implementation involved. Where that does not hold, the failure is
in the merge.

**Three things are checked against each other**, all on the real SDXL UNet
with real LoRA adapters at rank 8:

1. unmerged forward vs merged forward
2. unmerged forward vs the forward after a save/load round trip through the
   on-disk weight format
3. the on-disk key format itself against ComfyUI's, skipped when ComfyUI is
   absent -- because a format this project's own loader accepts but ComfyUI's
   rejects still means every exported LoRA is unusable outside this project,
   and that is exactly the interop question.

Non-zero adapter weights are the point of any of it: a fresh LoRA has `B` at
zero, so the delta is zero and *every* comparison here would pass trivially.
The test trains the adapters for a few steps against a real loss first, and
asserts the delta is non-trivial before relying on any of it.

A device out-of-memory *inside* this test's measured 10,669 MB footprint is
foreign usage, not this test: `fast_construction.oom_outcome` prints peak,
footprint and device-free, then skips. One above that footprint is the test
having grown, and is re-raised as the regression it is.

Run: `python nodes/smoke_tests/gpu/smoke_test_lora_merge_identity.py`
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from nodes.smoke_tests.fast_construction import (  # noqa: E402
    assert_fully_covered,
    find_a_checkpoint,
    oom_outcome,
    read_state_dicts,
    skipped_parameter_init,
)

failures: list[str] = []
skipped: list[str] = []

PROMPT = "a photograph of an astronaut riding a horse on mars"
LATENT = 32
TI = 500

#: float32 has ~1.2e-7 relative resolution. The merge is a rewrite, so the
#: two forwards accumulate their differences differently and this is a
#: tolerance, not an equality -- but it is a *tight* one: the two ways of
#: computing the same function should agree to rounding, and a transposed
#: factor or a missed `alpha` would miss this by orders of magnitude.
TOLERANCE = 1e-5

#: The measured footprint of this test: the same fp32 UNet and the same
#: latent-32 backward as `smoke_test_real_training_step`, whose own peak
#: print is 10,669 MB (9,804 MB of weights). Its comparison forwards run
#: under `no_grad`, which stores no graph, and `merge_lora` folds in
#: place, so nothing here adds a second copy of the model. An
#: out-of-memory at or below this (plus `OOM_FOOTPRINT_TOLERANCE_MB`) is
#: foreign usage holding the rest of the card -- skipped with the
#: numbers printed; see `oom_outcome`.
FOOTPRINT_MB = 10_669.0

#: ComfyUI's LoRA key prefix, from `comfy/lora.py:model_lora_keys_unet`:
#: `"lora_unet_{}".format(key_lora)` where `key_lora` is the module path with
#: `.` replaced by `_`.
PREFIX = "lora_unet_"


def record(ok: bool, name: str, detail: str | None = "") -> None:
    shown = detail if (detail is not None and ok) else None
    suffix = f": {shown}" if shown else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


def free(device: str) -> None:
    gc.collect()
    if device == "xpu" and torch.xpu.is_available():
        torch.xpu.empty_cache()


def load_unet_sd():
    """The real checkpoint's UNet tensors, or None.

    The shared reader rather than a third copy of this: it strips
    `model.diffusion_model.` from the UNet keys, which is what the model's own
    names are, so the coverage check below compares like with like.
    """
    path = find_a_checkpoint()
    if path is None:
        skip("a real checkpoint", "none found under ComfyUI's checkpoints")
        return None
    unet, _clip, _vae = read_state_dicts(path)
    print(f"  using {path.name}: {len(unet)} unet tensors")
    return unet


def relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """`max|a - b|` against the scale of `a`, which is the honest yardstick.

    An absolute tolerance would be meaningless here: the output's scale
    depends on the input draw, and a large output deserves a larger absolute
    error.
    """
    return float((a - b).abs().max()) / max(float(a.abs().max()), 1e-12)


def main() -> int:
    if not (torch.xpu.is_available() or torch.cuda.is_available()):
        skip("the LoRA merge identity", "neither XPU nor CUDA is available")
        print("\nSMOKE TEST: ALL CHECKS PASSED (nothing to run)")
        return 0
    device = "xpu" if torch.xpu.is_available() else "cuda"
    print(f"== merging a LoRA does not change what the model computes "
          f"({device}) ==")

    unet_sd = load_unet_sd()
    if unet_sd is None:
        print("\nSMOKE TEST: ALL CHECKS PASSED (nothing to run)")
        return 0

    from nodes.components.diffusion import (DiffusionProcess,
                                           DiscreteLinearNoiseSchedule,
                                           EpsParameterization,
                                           KarrasInputScaler)
    from nodes.model.clip_encoder import SDXLClipEncoder
    from nodes.model.lora import LoRAConfig
    from nodes.model.unet_wrapper import ComfyUNetWrapper

    # Conditioning first, off the card, so the UNet has room for its 10.3 GB.
    # `unet_sd` was already read from this file above; only the conditioner
    # half is still needed, and reading the whole thing again to get it is
    # what this used to do.
    _u, clip_sd, _v = read_state_dicts(find_a_checkpoint())
    encoder = SDXLClipEncoder(clip_sd, device=device)
    context, _pooled = encoder.encode_prompt(PROMPT)
    context = context.to(torch.float32).to(device)
    del encoder, clip_sd
    free(device)

    # Base weights are replaced by the checkpoint's; skip initialising them
    # first. `lora_B` still starts at exactly zero -- `LoRALinear` uses
    # explicit `zeros_`, not `reset_parameters` -- and this file's whole
    # premise is that a *trained* adapter differs from that starting point,
    # so it asserts the adapters moved before relying on any comparison.
    with skipped_parameter_init():
        wrapper = ComfyUNetWrapper(
            unet_sd, device=device, dtype=torch.float32, use_checkpoint=False,
            adm_in_channels=2816,
            lora_config=LoRAConfig(rank=8, alpha=8.0,
                                   target_modules=["to_q", "to_k", "to_v",
                                                   "to_out.0"]))
    assert_fully_covered(wrapper.model, unet_sd, name="UNet")
    model = wrapper.model

    process = DiffusionProcess(DiscreteLinearNoiseSchedule(),
                               EpsParameterization(), KarrasInputScaler())
    generator = torch.Generator(device="cpu").manual_seed(0)
    x0 = torch.randn(1, 4, LATENT, LATENT, generator=generator)
    eps = torch.randn(1, 4, LATENT, LATENT, generator=generator)
    t = torch.tensor([TI], dtype=torch.long, device=device)
    _, sigma = process.schedule.alpha_sigma(t)
    x_t = x0.to(device) + sigma * eps.to(device)
    xc = process.input_transform.scale_input(x_t, sigma)
    adm = torch.zeros(1, 2816, device=device, dtype=torch.float32)

    def forward() -> torch.Tensor:
        with torch.no_grad():
            return wrapper.forward(xc, t, context, adm).detach().float().cpu()

    print("\n== train the adapters, so the delta is not trivially zero ==")
    for step in range(3):
        model.zero_grad(set_to_none=True)
        pred = wrapper.forward(xc, t, context, adm)
        loss = (pred.float() - eps.to(device).float()).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    p.add_(p.grad, alpha=-0.05)

    b_weights = {n: p for n, p in model.named_parameters()
                 if p.requires_grad and "lora_B" in n}
    moved = [n for n, p in b_weights.items() if p.abs().max() > 0]
    record(bool(moved),
           f"the adapters are no longer at their zero initialisation "
           f"({len(moved)}/{len(b_weights)} lora_B tensors are nonzero) "
           f"after 3 steps -- without this every comparison below would "
           f"pass trivially",
           None)

    before = forward()

    print("\n== 1. save and reload through the on-disk weight format ==")
    exported = wrapper.get_lora_weights()
    record(bool(exported),
           f"the wrapper exports {len(exported)} tensors", None)
    down = [k for k in exported if k.endswith("lora_down.weight")]
    up = [k for k in exported if k.endswith("lora_up.weight")]
    alpha = [k for k in exported if k.endswith(".alpha")]
    record(len(down) == len(up) == len(alpha),
           f"they come in matched sets -- one down, one up and one alpha "
           f"per injected layer: {len(down)}/{len(up)}/{len(alpha)}",
           None)

    # Round-trip through safetensors, so this is the real on-disk path and
    # not just a dict that never left memory.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "round_trip.safetensors"
        from safetensors import safe_open
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in exported.items()}, str(path))
        with safe_open(str(path), framework="pt") as f:
            reloaded = {k: f.get_tensor(k) for k in f.keys()}
    record(set(reloaded) == set(exported),
           f"the file round-trips all {len(reloaded)} tensors with the same "
           f"keys")
    record(all(torch.equal(reloaded[k], exported[k]) for k in reloaded),
           "and byte-identical values")

    wrapper.load_lora_weights(reloaded)
    after_reload = forward()
    reload_diff = relative_difference(before, after_reload)
    record(reload_diff <= TOLERANCE,
           f"loading them back changes nothing: relative difference "
           f"{reload_diff:.3e} against a tolerance of {TOLERANCE:.0e}",
           f"{reload_diff:.3e}")

    print("\n== 2. merging is a rewrite, so the output must not move ==")
    wrapper.merge_lora()
    merged = forward()
    merge_diff = relative_difference(before, merged)
    record(merge_diff <= TOLERANCE,
           f"merged forward matches unmerged: relative difference "
           f"{merge_diff:.3e} against a tolerance of {TOLERANCE:.0e}. A "
           f"larger difference would mean the merge is not a faithful "
           f"rewrite -- check the factor order and that the alpha scaling "
           f"is applied",
           None)

    print("\n== and the adapters really are gone, not just bypassed ==")
    from nodes.model.lora import lora_param_count
    record(lora_param_count(wrapper.lora_registry) > 0,
           "the registry still lists the adapters")
    merged_export = wrapper.get_lora_weights()
    record(all(not torch.equal(merged_export[k], exported[k])
               for k in down if k in merged_export),
           "but their weights are gone -- merging consumed them, so a second "
           "merge would be a no-op rather than a double-apply",
           "the exported weights are unchanged after merge_lora()")
    free(device)

    print("\n== 3. the on-disk keys are ones ComfyUI can read ==")
    # ComfyUI's contract is `comfy/lora.py:load_lora(lora, to_load)`, which
    # looks for `{base}.alpha`, then hands `{base}` to the adapters in
    # `comfy/weight_adapter/`, which read `{base}.lora_down.weight` and
    # `{base}.lora_up.weight`. Those three suffixes are read out of comfyi's
    # *source* here rather than hardcoded, so this tracks comfyi if the names
    # ever move -- a test that hardcoded them would keep passing after the
    # thing it exists to check had changed.
    their_suffixes: set[str] = set()
    their_key_shape = None
    try:
        import re as _re
        import paths as project_paths
        comfy_root = project_paths.get_comfy_dir()
        if str(comfy_root) not in sys.path:
            sys.path.insert(0, str(comfy_root))
        for relative in ("comfy/lora.py", "comfy/weight_adapter/lora.py"):
            source = (Path(comfy_root) / relative).read_text(encoding="utf-8")
            their_suffixes |= set(
                _re.findall(r'\{\}\.([a-z0-9_.]+)', source))
        # comfyi/lora.py:model_lora_keys_unet builds
        #   "lora_unet_{}".format(k[:-len(".weight")].replace(".", "_"))
        # so both the prefix and the `.` -> `_` mangling are the contract.
        sd_source = (Path(comfy_root) / "comfy/lora.py").read_text(
            encoding="utf-8")
        their_key_shape = _re.search(
            r'key_lora = k\[[^\]]+\]\.replace\("([^"]+)", "([^"]+)"\)',
            sd_source)
        their_key_shape = their_key_shape.groups() if their_key_shape else None
        prefix = _re.search(r'key_map\["([a-z_]+)_\{\}"\]', sd_source)
        global PREFIX
        if prefix:
            PREFIX = prefix.group(1) + "_"
    except Exception as exc:  # noqa: BLE001
        skip("the on-disk key format",
             f"could not read comfyi's own key format ({type(exc).__name__}: "
             f"{exc})")
        print("     (a format this project's loader accepts but comfyi's "
              "rejects still means every exported LoRA is unusable outside "
              "this project)")
    else:
        wanted = {"lora_down.weight", "lora_up.weight", "alpha"}
        record(wanted <= their_suffixes,
               f"the suffixes we export are among the ones comfyi's own "
               f"source builds: {sorted(wanted & their_suffixes)}",
               None)

        ours = {k.rsplit(".", 1)[-1] for k in exported}
        record(ours <= their_suffixes,
               f"and every exported key ends in one of them "
               f"({len(ours)} distinct suffixes: {sorted(ours)}); anything "
               f"else comfyi's loader would silently not read",
               None)

        record(their_key_shape is not None
               and their_key_shape == (".", "_"),
               f"comfyi builds its key shape by replacing "
               f"{their_key_shape} in the module path; a different shape "
               f"would mean the mangling assumption in the rest of this "
               f"section does not apply",
               None)

        # The half that matters: our keys must be comfyi's `lora_unet_` form.
        # That is the checkpoint contract end to end -- it would fail if a
        # reimplementation renamed or renested anything, which is exactly
        # what would make every exported LoRA silently unloadable.
        if their_key_shape == (".", "_"):
            prefixed = [k for k in down if k.startswith(PREFIX)]
            record(len(prefixed) == len(down),
                   f"all {len(down)} use comfyi's `{PREFIX}` prefix and its "
                   f"dot-to-underscore mangling, {len(prefixed)} of "
                   f"{len(down)}",
                   None)

            # ComfyUI's rule, read out of its source above, applied to the
            # paths this project actually injected. Comparing the two
            # directions is not possible -- `_` -> `.` is ambiguous, because
            # `time_embed.0` mangles to `time_embed_0` and unmangles to
            # `time.embed.0` -- so the comparison runs forwards, where there
            # is no ambiguity.
            injected = [full for full, _p, _a, _l in wrapper.lora_registry]
            expected = {f"{PREFIX}{p.replace('.', '_')}" for p in injected}
            got = {k[: -len(".lora_down.weight")] for k in down}
            record(expected == got,
                   f"and applying comfyi's own rule to the {len(injected)} "
                   f"injected paths reproduces our {len(got)} exported keys "
                   f"exactly, {len(expected ^ got)} differ",
                   None)

            # And the injected paths hold a correctly-shaped adapter. The
            # original Linear is *replaced* rather than nested --
            # `LoRALinear` keeps the frozen weight as a `base_weight` buffer
            # -- so "is it still an nn.Linear" is the wrong question, and
            # would fail on a perfectly good injection. The right one is
            # whether the factor geometry matches the base weight it was
            # built against, because that is what silently produces a broken
            # export if the target set ever drifts onto a module of the wrong
            # shape.
            wrong_type, wrong_shape = [], []
            for path in injected:
                try:
                    target = model.get_submodule(path)
                except AttributeError:
                    wrong_type.append(f"{path} (does not resolve)")
                    continue
                if not (hasattr(target, "lora_A")
                        and hasattr(target, "lora_B")
                        and hasattr(target, "base_weight")):
                    wrong_type.append(f"{path} ({type(target).__name__})")
                    continue
                down_w, up_w = target.lora_A, target.lora_B
                base = target.base_weight
                if not (down_w.shape[1] == base.shape[1]
                        and up_w.shape[0] == base.shape[0]
                        and down_w.shape[0] == up_w.shape[1]):
                    wrong_shape.append(
                        f"{path}: A{tuple(down_w.shape)} B{tuple(up_w.shape)} "
                        f"against base{tuple(base.shape)}")
            record(not wrong_type,
                   f"and every injected path resolves and holds an adapter "
                   f"with the three expected pieces -- lora_A, lora_B and the "
                   f"frozen base_weight ({len(injected)} checked), "
                   f"{len(injected) - len(wrong_type)} do",
                   None)
            record(not wrong_shape,
                   "whose factor geometry matches the base weight it was "
                   "built against, so the merge is a valid multiplication",
                   None)

            roots = sorted({p.split(".")[0] for p in injected})
            record(set(roots) <= {"input_blocks", "middle_block",
                                  "output_blocks", "time_embed", "label_emb"},
                   f"and all of them sit inside the UNet's own top-level "
                   f"blocks: {roots}",
                   None)

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
    try:
        raise SystemExit(main())
    except torch.OutOfMemoryError as exc:
        outcome = oom_outcome(
            exc, footprint_mb=FOOTPRINT_MB, failures=failures,
            name="the LoRA merge identity")
        if outcome is None:
            raise
        raise SystemExit(outcome)
