"""Conditioning blend: ctx = p*ctx(caption) + (1-p)*ctx("").

CPU, fake encoder with readable values: the caption's context is all 1.0 and
the blank's all 0.0 (so a blended row's value *is* p), the resolution columns of
y carry (h, w) so a disturbed resolution is visible. Pins: the formula; per-
sample p in range and seeded; p=1 bit-exact and p=0 exactly blank; the
resolution part untouched; the inactive blend (default) leaves encode_with_true_
sizes byte-identical and never asks for the blank prompt; the blank is one
("",1) key whatever the batch size; both trainers take the port; prewarm sets
gain the blank key. Whether it helps a real LoRA is not tested here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.model.text_encoder import TextEncoder
from nodes.train.bucket_sizes import encode_with_true_sizes
from nodes.train.conditioning_blend import (BLANK_PROMPT, ConditioningBlend,
                                            ConditioningBlendNode)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class _Enc(TextEncoder):
    """Caption -> ones, blank -> zeros; y = [pooled(2) | h*10000+w]."""

    def __init__(self):
        self.prompt_calls = []

    def encode_prompt_only(self, prompt, batch_size):
        self.prompt_calls.append((prompt, batch_size))
        v = 0.0 if prompt == BLANK_PROMPT else 1.0
        return (torch.full((batch_size, 3, 4), v), torch.full((batch_size, 2), v))

    def resolution_embedding(self, height, width, batch_size):
        return torch.full((batch_size, 1), float(height) * 10000 + width)

    def unload(self): ...
    def offload(self): ...
    def reload(self): ...
    def release(self): ...
    def footprint_bytes(self): return 0


def _batch(b=4, mask=True):
    x_t = torch.randn(b, 4, 8, 8)
    return x_t, (torch.ones(b, 4, 8, 8) if mask else None)


def check_the_formula_and_ranges():
    print("[p drawn per sample in range; row value == p; resolution untouched]")
    enc = _Enc()
    x_t, m = _batch(6)
    blend = ConditioningBlend(0.25, 0.75, seed=3)
    ctx, y = encode_with_true_sizes(enc, "cap", 6, x_t, m, blend=blend)
    ps = ctx[:, 0, 0]
    check(bool(((ps >= 0.25) & (ps <= 0.75)).all()), ps)
    check(len(set(ps.tolist())) > 1, "p did not vary per sample")
    check(bool(torch.allclose(ctx, ps.view(-1, 1, 1).expand_as(ctx))), "ctx not uniform per row")
    check(bool(torch.allclose(y[:, :2], ps.view(-1, 1).expand(-1, 2))), "pooled part of y")
    check(bool((y[:, 2] == 64 * 10000 + 64).all()), "resolution column disturbed")
    print("    PASS")


def check_endpoints_are_exact():
    print("[p=1 bit-exact caption; p=0 exactly the blank]")
    x_t, m = _batch(3)
    plain_ctx, plain_y = encode_with_true_sizes(_Enc(), "cap", 3, x_t, m)
    c1, y1 = encode_with_true_sizes(_Enc(), "cap", 3, x_t, m, blend=ConditioningBlend(1.0, 1.0))
    check(torch.equal(c1, plain_ctx) and torch.equal(y1, plain_y), "inactive changed values")
    # active blend whose draws are all 1.0 through where(): use range pinned at 1 via draw()
    blend = ConditioningBlend(0.0, 1.0, seed=1)
    ctx = torch.randn(3, 3, 4); y = torch.randn(3, 3)
    c0 = torch.randn(3, 3, 4); y0 = torch.randn(3, 3)
    out_c, out_y = blend.mix(ctx, y, c0, y0, [1.0, 0.0, 1.0])
    check(torch.equal(out_c[0], ctx[0]) and torch.equal(out_c[2], ctx[2]), "p=1 not exact")
    check(torch.equal(out_c[1], c0[1]) and torch.equal(out_y[1], y0[1]), "p=0 not blank")
    check(torch.equal(out_y[0], y[0]), "p=1 y not exact")
    print("    PASS")


def check_inactive_never_touches_the_blank():
    print("[default blend / none: identical values, the blank prompt is never requested]")
    for blend in (None, ConditioningBlend()):
        enc = _Enc()
        x_t, m = _batch(2)
        encode_with_true_sizes(enc, "cap", 2, x_t, m, blend=blend)
        check(all(p != BLANK_PROMPT for p, _ in enc.prompt_calls), enc.prompt_calls)
        check(blend is None or not blend.active, "default blend reports active")
    print("    PASS")


def check_the_blank_is_one_key_at_batch_size_one():
    print("[blank requested as (\"\", 1) for any batch size; unmasked batch works too]")
    for b, mask in ((1, True), (5, True), (3, False)):
        enc = _Enc()
        x_t, m = _batch(b, mask)
        encode_with_true_sizes(enc, "cap", b, x_t, m, blend=ConditioningBlend(0.5, 0.5))
        blanks = {c for c in enc.prompt_calls if c[0] == BLANK_PROMPT}
        check(blanks == {(BLANK_PROMPT, 1)}, blanks)
    print("    PASS")


def check_mixed_caption_batches_blend_per_sample():
    print("[mixed captions + blend: each row blended from its own caption]")
    class Enc2(_Enc):
        def encode_prompt_only(self, prompt, batch_size):
            v = {"": 0.0, "a": 2.0, "b": 4.0}[prompt]
            return (torch.full((batch_size, 3, 4), v), torch.full((batch_size, 2), v))
    x_t, m = _batch(2)
    ctx, _ = encode_with_true_sizes(Enc2(), "a", 2, x_t, m, prompts=["a", "b"],
                                    blend=ConditioningBlend(0.5, 0.5))
    check(torch.allclose(ctx[0], torch.full((3, 4), 1.0)), ctx[0])
    check(torch.allclose(ctx[1], torch.full((3, 4), 2.0)), ctx[1])
    print("    PASS")


def check_seed_replays_and_validation():
    print("[seeded draws replay; bad ranges refused]")
    a, b = ConditioningBlend(0.1, 0.9, seed=5), ConditioningBlend(0.1, 0.9, seed=5)
    check(a.draw(8) == b.draw(8), "seed did not replay")
    for lo, hi in ((0.8, 0.2), (-0.1, 0.5), (0.5, 1.5)):
        try:
            ConditioningBlend(lo, hi)
        except ValueError:
            continue
        check(False, (lo, hi))
    print("    PASS")


def check_node_and_trainer_ports_and_prewarm_key():
    print("[node defaults inactive; both trainers carry the port; prewarm gains the blank key]")
    out = ConditioningBlendNode().build()["blend"]
    check(not out.active, "node default is active")
    check(ConditioningBlendNode().build(seed=-1)["blend"].seed is None, "seed -1")
    from nodes.train.managed import ManagedLoRATrainerNode
    from nodes.train.supervised import SupervisedLoRATrainerNode
    for cls in (ManagedLoRATrainerNode, SupervisedLoRATrainerNode):
        port = cls.INPUTS["conditioning_blend"]
        check(port.required is False and port.default is None, cls)
    from nodes.model.text_encoder_prewarm import Discovery, StopReason
    from nodes.train.conditioning_blend import with_blank_prompt
    d = Discovery(keys=frozenset({("x", 2, 64, 64, 2)}), batches_seen=1,
                  prompts_found=1, reason=StopReason.COMPLETED)
    d2 = with_blank_prompt(d)
    check(("", 1, 64, 64, 2) in d2.keys and ("x", 2, 64, 64, 2) in d2.keys, d2.keys)
    print("    PASS")


def check_phases_pass_the_blend_through():
    print("[EncodeConditioningPhase (both routes) applies the blend to ctx_emb / y]")
    from nodes.train.managed import EncodeConditioningPhase as ManagedPhase
    from nodes.train.managed import ManagedStepState
    from nodes.train.step_pipeline import EncodeConditioningPhase, StepState

    class _RC:
        def ensure_loaded(self, name): ...
        def release(self, name): ...

    class _Ctl:
        def should_release(self, name): return False

    cpu = torch.device("cpu")
    for route in ("supervised", "managed"):
        for blend, want in ((ConditioningBlend(0.5, 0.5), 0.5), (None, 1.0)):
            x_t, m = _batch(2)
            batch = {"prompt": "cap", "prompts": ["cap", "cap"]}
            if route == "supervised":
                state = StepState(step=0, batch=batch, model=None, device=cpu)
                phase = EncodeConditioningPhase(_Enc(), blend=blend)
            else:
                state = ManagedStepState(step=0, batch=batch, model=None, device=cpu)
                phase = ManagedPhase(_Enc(), _RC(), _Ctl(), blend=blend)
            state.extras["x_t"], state.extras["valid_mask"] = x_t, m
            phase.run(state)
            got = state.extras["ctx_emb"].float()
            check(bool(torch.allclose(got, torch.full((2, 3, 4), want))), (route, blend, got))
    print("    PASS")


def main():
    check_the_formula_and_ranges()
    check_endpoints_are_exact()
    check_inactive_never_touches_the_blank()
    check_the_blank_is_one_key_at_batch_size_one()
    check_mixed_caption_batches_blend_per_sample()
    check_seed_replays_and_validation()
    check_node_and_trainer_ports_and_prewarm_key()
    check_phases_pass_the_blend_through()
    print("\nSMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
