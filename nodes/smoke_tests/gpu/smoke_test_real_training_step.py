"""A training step through the reimplemented SDXL stack, on real weights.

Design doc 12, section 7.3. Everything the reimplementation covers is
verified somewhere, but never *together*: the CLIP test uses random weights,
the UNet test checks a forward with no backward, the LoRA tests use tiny
synthetic models, and `smoke_test_step_pipeline_correctness.py` proves the
pipeline's arithmetic against a `pred = p` stand-in. So nothing ran a
gradient through the real 2.57 B UNet with real LoRA adapters.

That is the gap, and it is where the reimplementation's risk sits.
`nodes/model/checkpoint.py` had two bugs -- a crash on frozen parameters and
a CUDA-only autocast re-entry -- and neither is visible in a forward pass.

**The central claim is that activation checkpointing is numerically
transparent, and on this card that cannot be tested by comparing for
bitwise equality.** The B580's kernels are not bit-reproducible: two
identical forwards differ by up to 9.5e-07 against an output scale of
4.85, and two identical backwards differ by up to 5.1e-09 in their
gradients. So "the checkpointed and un-checkpointed gradients are equal"
is not a statement this hardware can make, and a test asserting it would be
asserting something false and passing or failing for the wrong reason.

What can be said is the stronger, more useful thing: **checkpointing's
contribution is inside the run-to-run variation the card already has**. So
the test measures that floor first, from two un-checkpointed backwards, and
then requires the checkpointed-against-uncheckpointed difference to sit
inside it. A checkpointing bug that changed the math would show up as a
difference orders of magnitude larger than the floor; one that merely
reordered an accumulation would not, and should not fail.

This is also why the design doc's "bitwise identical" claims are all CPU
measurements. They are true of CPU. They would not be true here.

**Half the gradients are exactly zero, and that is not a defect.** LoRA
initialises `B` to zero, so the delta is `B(A(x)) == 0` and
`dL/dA = B^T . dL/dy = 0` exactly. At rank 8 with 564 injected layers that
is 564 of the 1128 trainable tensors with a gradient of precisely zero, and
zero compares bitwise equal to zero -- which is what makes a naive
"how many gradients match" count read as exactly half no matter what else
is true. The tests below exclude them and say so.

**One model on the card at a time.** The UNet is 10.3 GB in float32 against
an 11.93 GB card. CLIP runs first, its conditioning moves to the host, and
only then does the UNet load. The latent is 32 (a 256x256 image) because
the *uncheckpointed* backward at 64x64 OOMs trying to allocate 58 MB with
6 MB free -- which is itself the argument for `SDXL_CONFIG` defaulting
`use_checkpoint` to True.

Skipped rather than failed with no accelerator or no checkpoint: both are
legitimate states, and neither is a reason to report a green run that never
ran.

Run: `python nodes/smoke_tests/gpu/smoke_test_real_training_step.py`
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from nodes.smoke_tests.fast_construction import (  # noqa: E402
    assert_fully_covered,
    find_a_checkpoint,
    read_state_dicts,
    skipped_parameter_init,
)

failures: list[str] = []
skipped: list[str] = []

PROMPT = "a photograph of an astronaut riding a horse on mars"
LATENT = 32
TI = 500
LR = 1e-4

#: How much larger than the card's own run-to-run variation a real
#: checkpointing difference would have to be before this test fails. 4x, not
#: 1x: the floor is a single sample of the variation, so a checkpointed run
#: can legitimately land slightly outside one measurement of it. 4x still
#: leaves orders of magnitude between "noise" and "wrong arithmetic" -- a
#: dropped gradient term shows up at 1e+00 here, not at 1e-08.
NOISE_HEADROOM = 4.0


def record(ok: bool, name: str, detail: str | None = "") -> None:
    """One check.

    `detail` is the *measurement*, and is printed whether the check passed
    or failed -- a passing line that does not say what it measured is a
    claim, not a check. A failed check adds the detail to the failure list
    too, so the summary line is self-contained.

    `detail=None` says there is nothing to add beyond the message, which is
    right when the message already carries the numbers.
    """
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


def peak_mb(device: str) -> float:
    if device == "xpu" and torch.xpu.is_available():
        return torch.xpu.max_memory_allocated() / 1024 / 1024
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_peak(device: str) -> None:
    if device == "xpu" and torch.xpu.is_available():
        torch.xpu.reset_peak_memory_stats()
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def load_state_dicts():
    """``(unet, clip)`` from a real checkpoint, or None.

    The shared reader, because two copies of this function had already
    drifted: one stripped `model.diffusion_model.` from the UNet keys and one
    did not, and the difference only showed up as a coverage assertion
    reporting 972 tensors missing that were present under the other
    convention.
    """
    path = find_a_checkpoint()
    if path is None:
        skip("a real checkpoint", "none found under ComfyUI's checkpoints")
        return None
    unet, clip, _vae = read_state_dicts(path)
    print(f"  using {path.name}: {len(unet)} unet, {len(clip)} clip tensors")
    return unet, clip


def build_conditioning(clip_sd, device):
    """CLIP alone, then off the card, so the UNet has room."""
    from nodes.model.clip_encoder import SDXLClipEncoder

    encoder = SDXLClipEncoder(clip_sd, device=device)
    context, pooled = encoder.encode_prompt(PROMPT)
    y_time = encoder.resolution_embedding(LATENT * 8, LATENT * 8)
    # The pipeline's EncodeConditioningPhase hands the UNet bfloat16
    # conditioning (nodes/train/step_pipeline.py), so build it that way
    # rather than in float32 and casting inside the test.
    context = context.to(torch.bfloat16).cpu()
    adm = torch.cat([pooled.to(torch.float32), y_time.to(torch.float32)],
                    dim=-1).to(torch.bfloat16).cpu()
    record(tuple(context.shape) == (1, 77, 2048),
           "CLIP gives a (1, 77, 2048) context on real weights",
           f"{tuple(context.shape)}")
    record(tuple(adm.shape) == (1, 2816),
           "and SDXL's (1, 2816) y", f"{tuple(adm.shape)}")
    del encoder
    free(device)
    return context, adm


def build_wrapper(unet_sd, device, use_checkpoint):
    """The production entry point, which is what freezes and injects."""
    from nodes.model.lora import LoRAConfig
    from nodes.model.unet_wrapper import ComfyUNetWrapper

    config = LoRAConfig(rank=8, alpha=8.0,
                        target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    # The base weights are all replaced by the checkpoint's below, so skip
    # initialising 2.57 B of them first (10.3 s). The LoRA adapters are not
    # affected: `LoRALinear` initialises `lora_A` and `lora_B` with explicit
    # `kaiming_uniform_`/`zeros_` calls rather than through `reset_parameters`,
    # so `lora_B` still starts at exactly zero -- which is what the gradient
    # checks below depend on.
    with skipped_parameter_init():
        wrapper = ComfyUNetWrapper(unet_sd, device=device, dtype=torch.float32,
                                   use_checkpoint=use_checkpoint,
                                   adm_in_channels=2816, lora_config=config)
    adapters = assert_fully_covered(wrapper.model, unet_sd, name="UNet")
    # `assert_fully_covered` accepts the adapters' own tensors as covered,
    # because LoRALinear initialises them itself rather than through
    # `reset_parameters`. That is a claim, so it is checked here rather than
    # assumed: lora_B must start at exactly zero, which is what makes this
    # test's "exactly half the gradients are zero" observation true.
    parameters = dict(wrapper.model.named_parameters())
    lora_b = [parameters[a] for a in adapters if a.endswith("lora_B")]
    if not all(bool((p == 0).all()) for p in lora_b):
        raise AssertionError(
            f"skipped_parameter_init disturbed the adapters: "
            f"{sum(0 if bool((p == 0).all()) else 1 for p in lora_b)} of "
            f"{len(lora_b)} lora_B tensors are not exactly zero")
    return wrapper


def set_checkpointing(model, flag: bool) -> int:
    """Toggle the flag on every block that reads it, in place.

    `ResBlock.forward` passes `self.use_checkpoint` to `checkpoint()` on
    every call, so this needs no rebuild and no second copy of 2.57 B
    parameters.

    It reaches 18 modules and only those: the spatial transformers are
    checkpointed unconditionally by `attention_checkpointing.py`, which
    found that ComfyUI's `BasicTransformerBlock` accepts a `checkpoint`
    argument and silently discards it, so there is no flag on those blocks
    to toggle and none is invented here. Both runs therefore have the
    transformers checkpointed, and what this toggles is the ResBlocks.
    """
    touched = 0
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = flag
            touched += 1
    return touched


class Step:
    """One training step's worth of fixed inputs, built once.

    Both backwards must see identical inputs or the comparison is measuring
    the inputs, so the noise is drawn once from one seeded generator here
    rather than per call.
    """

    def __init__(self, context, adm, device):
        from nodes.components.diffusion import (DiffusionProcess,
                                               DiscreteLinearNoiseSchedule,
                                               EpsParameterization,
                                               KarrasInputScaler)
        from nodes.model.lora import set_lora_gate
        from nodes.train.loss import UniformLossWeighting

        self.process = DiffusionProcess(DiscreteLinearNoiseSchedule(),
                                         EpsParameterization(),
                                         KarrasInputScaler())
        self.weighting = UniformLossWeighting()
        self.device = device
        self.context = context.to(device)
        self.adm = adm.to(device)

        generator = torch.Generator(device="cpu").manual_seed(0)
        x0 = torch.randn(1, 4, LATENT, LATENT, generator=generator)
        eps = torch.randn(1, 4, LATENT, LATENT, generator=generator)
        self.eps = eps.to(device)
        self.t = torch.tensor([TI], dtype=torch.long, device=device)

        _, sigma = self.process.schedule.alpha_sigma(self.t)
        self.sigma = sigma
        # x_t = x0 + sigma*eps, the convention NoiseSchedule documents.
        x_t = x0.to(device) + sigma * self.eps
        set_lora_gate(None)            # gate disabled: cleared every step
        self.xc = self.process.input_transform.scale_input(x_t, sigma)

    def loss_of(self, pred):
        per_sample = (pred.float() - self.eps.float()).pow(2)
        per_sample = per_sample.view(1, -1).mean(dim=1)
        return per_sample.mean() * self.weighting.weight(
            float(self.sigma.float().mean()))

    def forward(self, wrapper):
        return wrapper.forward(self.xc, self.t, self.context, self.adm)


def lora_grads(model) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().float().cpu().clone()
            for name, p in model.named_parameters() if p.grad is not None}


def backward(wrapper, model, step) -> tuple[float, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = step.loss_of(step.forward(wrapper))
    loss.backward()
    return float(loss.detach()), lora_grads(model)


def worst_difference(a: dict, b: dict) -> tuple[float, str, int]:
    """Largest absolute gradient difference, over the *nonzero* gradients.

    LoRA's `A` gradients are exactly zero -- `B` starts at zero, so the delta
    is zero and `dL/dA` is zero -- and zero is bitwise equal to zero whatever
    else happened. Including them makes every comparison look half-identical
    no matter what, which is exactly the false signal this file had before.
    """
    nonzero = [k for k in a if a[k].abs().max() > 0 and k in b]
    if not nonzero:
        return 0.0, "none", 0
    worst_name, worst = max(
        ((k, float((a[k] - b[k]).abs().max())) for k in nonzero),
        key=lambda kv: kv[1])
    return worst, worst_name, len(nonzero)


def main() -> int:
    if not (torch.xpu.is_available() or torch.cuda.is_available()):
        skip("a training step on real weights",
             "neither XPU nor CUDA is available")
        print("\nSMOKE TEST: ALL CHECKS PASSED (nothing to run)")
        return 0
    device = "xpu" if torch.xpu.is_available() else "cuda"
    name = (torch.cuda.get_device_name(0) if device == "cuda"
            else torch.xpu.get_device_name(0))
    print(f"== a training step through the reimplemented SDXL stack on "
          f"{device} ==")
    print(f"  {name}")

    state = load_state_dicts()
    if state is None:
        print("\nSMOKE TEST: ALL CHECKS PASSED (nothing to run)")
        return 0
    unet_sd, clip_sd = state

    print("\n== conditioning, measured alone on the card ==")
    context, adm = build_conditioning(clip_sd, device)

    print("\n== the model, with LoRA injected by the production path ==")
    wrapper = build_wrapper(unet_sd, device, use_checkpoint=False)
    model = wrapper.model
    step = Step(context, adm, device)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record(bool(trainable),
           f"LoRA injected and trainable: {len(trainable)} tensors, "
           f"{total/1e6:.2f}M parameters")
    record(all("lora_" in n for n in trainable),
           "and every trainable tensor is a LoRA one, so the 2.57 B base "
           "parameters are frozen",
           f"not LoRA: {[n for n in trainable if 'lora_' not in n][:3]}")
    record(len(trainable) < 2000,
           "a small fraction of the model -- which is why a LoRA step needs "
           "no 10.3 GB of gradient buffers",
           f"{len(trainable)} trainable tensors")
    free(device)

    print("\n== the card's own reproducibility floor ==")
    with torch.no_grad():
        first = step.forward(wrapper).detach().float().cpu().clone()
    with torch.no_grad():
        second = step.forward(wrapper).detach().float().cpu().clone()
    forward_noise = float((first - second).abs().max())
    forward_scale = float(first.abs().max())
    record(forward_noise > 0,
           f"two identical forwards are not bitwise identical on {device}, "
           f"which is why the gradient comparison below measures a floor "
           f"first instead of asserting equality: "
           f"{forward_noise:.3e} against a scale of {forward_scale:.3e}",
           None)
    print(f"     (relative: {forward_noise / max(forward_scale, 1e-12):.1e}) "
          "-- so the design doc's bitwise claims are CPU measurements, "
          "and are labelled as such")

    print("\n== two UNCHECKPOINTED backwards: the floor for the comparison ==")
    set_checkpointing(model, False)
    reset_peak(device)
    loss_a, grads_a = backward(wrapper, model, step)
    free(device)
    reset_peak(device)
    loss_b, grads_b = backward(wrapper, model, step)
    peak_off = peak_mb(device)
    floor_off, floor_name, n_nonzero = worst_difference(grads_a, grads_b)
    record(all(torch.isfinite(g).all() for g in grads_a.values()),
           f"every gradient is finite (loss {loss_a:.9f}, {len(grads_a)} "
           f"tensors)")
    record(n_nonzero > 0,
           f"and {n_nonzero} of them are nonzero, so this is not a silent "
           f"no-op",
           f"max |grad| "
           f"{max(g.abs().max() for g in grads_a.values()):.3e}")
    print(f"     loss run-to-run: {abs(loss_a - loss_b):.3e}")
    print(f"     gradient floor (uncheckpointed, same inputs): "
          f"{floor_off:.3e} on {floor_name}")
    record(floor_off > 0,
           "the floor is non-zero, which is what makes it a floor rather "
           "than an assertion that happens to hold",
           f"{floor_off:.3e}")

    print("\n== two CHECKPOINTED backwards, and the difference between them ==")
    toggled = set_checkpointing(model, True)
    record(toggled > 0,
           f"the flag reaches the ResBlocks that read it ({toggled} modules; "
           f"the spatial transformers are checkpointed unconditionally by "
           f"attention_checkpointing.py and have no flag)")
    reset_peak(device)
    loss_c, grads_c = backward(wrapper, model, step)
    free(device)
    reset_peak(device)
    loss_d, grads_d = backward(wrapper, model, step)
    peak_on = peak_mb(device)
    floor_on, _, _ = worst_difference(grads_c, grads_d)
    print(f"     gradient floor (checkpointed, same inputs):  {floor_on:.3e}")
    print(f"     peak allocated: off {peak_off:.0f} MB, on {peak_on:.0f} MB")

    print("\n== checkpointing's contribution, against that floor ==")
    effect, effect_name, _ = worst_difference(grads_a, grads_c)
    ceiling = max(floor_off, floor_on) * NOISE_HEADROOM
    record(effect <= ceiling,
           f"checkpointing's gradient difference ({effect:.3e}) is inside "
           f"the card's own run-to-run variation "
           f"(floor {floor_off:.3e} uncheckpointed, {floor_on:.3e} "
           f"checkpointed, headroom {NOISE_HEADROOM}x), "
           f"largest on {effect_name}, ceiling {ceiling:.3e}",
           None)

    loss_effect = abs(loss_a - loss_c)
    # The gradient ceiling above is `max(observed floors) * headroom`, and the
    # gradient floors are reliably non-zero -- measured between 5.7e-09 and
    # 2.0e-08 across many runs. The loss's are not: both pairs of runs often
    # come back *bitwise* identical, so both floors are 0.0, the ceiling is
    # 0.0, and any difference at all fails. That is what the gate caught on
    # 2026-10-04, with the loss differing by 2.98e-08 -- about 0.6 ulp of a
    # float32 near 0.44.
    #
    # So the loss gets a floor of its own, from the representation rather than
    # from the hardware: a float32 scalar of this magnitude cannot agree with
    # itself to better than its own ulp, so demanding that is asking for
    # something the format does not offer. Four ulps is generous for a
    # reduction over 128k elements and still orders of magnitude below the
    # 1e+00 a genuinely dropped gradient term would produce.
    loss_floor = max(abs(loss_a - loss_b), abs(loss_c - loss_d))
    loss_ceiling = max(loss_floor * NOISE_HEADROOM,
                       4.0 * torch.finfo(torch.float32).eps * abs(loss_a))
    record(loss_effect <= loss_ceiling,
           f"and the loss likewise, against a ceiling of {loss_ceiling:.3e} "
           f"built from its run-to-run variation ({loss_floor:.3e}) and from "
           f"float32's own resolution at this magnitude: {loss_effect:.3e}",
           None)

    record(peak_on <= peak_off,
           f"and checkpointing does not cost memory: {peak_on:.0f} MB against "
           f"{peak_off:.0f} MB -- at a {LATENT}x{LATENT} latent the 10.3 GB of "
           f"frozen weights dominate, so there is almost nothing for it to "
           f"save here and this is not evidence about a real training "
           f"resolution",
           None)

    print("\n== an optimizer step moves the adapters, and only those ==")
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}
    base_before = {n: p.detach().float().cpu().clone()
                   for n, p in model.named_parameters() if not p.requires_grad}
    set_checkpointing(model, False)
    backward(wrapper, model, step)
    with torch.no_grad():
        for _n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad, alpha=-LR)

    current = dict(model.named_parameters())
    moved = {n for n, b in before.items() if not torch.equal(b, current[n])}
    unmoved = set(before) - moved
    expected_unmoved = {n for n in before if n in grads_a
                        and grads_a[n].abs().max() == 0}
    record(moved == set(before) - expected_unmoved,
           f"exactly the {len(moved)} adapters with a nonzero gradient moved, "
           f"and the {len(unmoved)} with an exactly-zero gradient did not",
           f"moved when they should not: "
           f"{sorted(moved - (set(before) - expected_unmoved))[:3]}; "
           f"did not move when they should have: "
           f"{sorted(unmoved - expected_unmoved)[:3]}")
    record(all("lora_A" in n for n in expected_unmoved),
           "and the zero-gradient ones are all `lora_A`, which is correct: "
           "LoRA starts B at zero, so the delta is zero and dL/dA is zero",
           f"zero-gradient tensors that are not lora_A: "
           f"{sorted(n for n in expected_unmoved if 'lora_A' not in n)[:3]}")

    after_base = {n: p.detach().float().cpu()
                  for n, p in model.named_parameters() if not p.requires_grad}
    changed_base = [n for n, b in base_before.items()
                    if not torch.equal(b, after_base[n])]
    record(not changed_base,
           f"and none of the {len(base_before)} frozen base tensors did",
           f"{len(changed_base)} moved: {changed_base[:3]}")

    free(device)

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
