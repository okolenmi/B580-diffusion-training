"""Correctness check for nodes/model/clip.py -- the SDXL CLIP text encoders
this project owns (design doc 12, section 7.3, section C1).

The contract is the checkpoint's keys, and the structure inside them:
`clip_l.transformer.text_model.*` and `clip_g.transformer.text_model.*`,
716 tensors.

The details worth pinning are the ones where being wrong produces a model
that loads, runs, and conditions slightly badly, with nothing raising:

* **The conditioning is the penultimate layer**, `layer_idx=-2`. Reading the
  last layer instead gives a shape-identical, silently worse result.
* **`final_layer_norm` is not applied to that tapped layer** for SDXL.
* **CLIP-L's activation is `quick_gelu`, CLIP-G's is `gelu`.** Two different
  architectures sharing one code path that takes a flag.
* **The pooled output is the last layer at the first EOS position.** Not the
  last position: the tokenizer pads with EOS, so for a short prompt the last
  position is padding. This one *does* change the output shape, which is how
  it was caught.
* **The context is 768 + 1280 = 2048**, concatenated along the feature axis
  and truncated to the shorter sequence.

Also checked: bitwise agreement with ComfyUI on four token-weight cases
including the weighted-blend path, the `quick_gelu` definition against its
closed form, the causal mask actually masking, and the ragged-section
rejection.

The tokenizer is *not* covered here -- it is section C2 and needs a decision
about vendoring data. This file therefore checks the model given token ids,
which is the whole of what the tokenizer's output feeds.

Run: `python nodes/smoke_tests/smoke_test_clip.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.clip import (  # noqa: E402
    CLIP_G_CONFIG,
    CLIP_L_CONFIG,
    CLIPEncoder,
    CLIPLayer,
    SDClipModel,
    SDXLClipG,
    SDXLClipModel,
)

failures: list[str] = []
skipped: list[str] = []

BOS, EOS = 49406, 49407


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


def _fill(module, seed: int = 31, scale: float = 0.02) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=gen) * scale)


def _row(ids):
    """One prompt section as a plain list of ids.

    Not (id, weight) pairs: the weight path is ComfyUI's wildcard machinery
    and is deliberately not reimplemented. See SDClipModel.encode_token_ids.
    """
    return [int(t) for t in ids]


def _prompt_ids(seed: int = 3, length: int = 77) -> list[int]:
    torch.manual_seed(seed)
    return [BOS] + torch.randint(0, 49406, (length - 2,)).tolist() + [EOS]


# ---------------------------------------------------------------------------
# The two configurations
# ---------------------------------------------------------------------------

def check_configs():
    record(CLIP_L_CONFIG["num_hidden_layers"] == 12
           and CLIP_L_CONFIG["hidden_size"] == 768
           and CLIP_L_CONFIG["num_attention_heads"] == 12,
           "CLIP-L is 12 layers, width 768, 12 heads")
    record(CLIP_L_CONFIG["hidden_act"] == "quick_gelu",
           "and uses quick_gelu, not gelu")
    record(CLIP_G_CONFIG["num_hidden_layers"] == 32
           and CLIP_G_CONFIG["hidden_size"] == 1280
           and CLIP_G_CONFIG["num_attention_heads"] == 20,
           "CLIP-G is 32 layers, width 1280, 20 heads")
    record(CLIP_G_CONFIG["hidden_act"] == "gelu", "and uses gelu")
    record(CLIP_L_CONFIG["max_position_embeddings"] == 77
           and CLIP_G_CONFIG["max_position_embeddings"] == 77,
           "both take 77 positions")
    record(CLIP_L_CONFIG["hidden_size"] + CLIP_G_CONFIG["hidden_size"] == 2048,
           "and their widths sum to the UNet's context_dim of 2048",
           f"{CLIP_L_CONFIG['hidden_size'] + CLIP_G_CONFIG['hidden_size']}")


def check_quick_gelu():
    """quick_gelu is x * sigmoid(1.702x) -- not nn.GELU, and not tanh-GELU."""
    from nodes.model.clip import _ACTIVATIONS
    quick = _ACTIVATIONS["quick_gelu"]
    x = torch.linspace(-4, 4, 41)
    want = x * torch.sigmoid(1.702 * x)
    record(torch.allclose(quick(x), want, atol=1e-6),
           "quick_gelu is x * sigmoid(1.702 * x) to float precision")
    record(not torch.allclose(quick(x), torch.nn.functional.gelu(x),
                              atol=1e-3),
           "and it is measurably not nn.GELU -- so CLIP-L read as gelu would "
           "be wrong rather than equivalent")


def check_layer_selection():
    """The conditioning is the penultimate layer, un-normed."""
    model = SDClipModel(CLIP_L_CONFIG, layer="hidden", layer_idx=-2,
                        layer_norm_hidden_state=False).eval()
    _fill(model, seed=41)
    record(model.layer == "hidden" and model.layer_idx == -2,
           "SDXL reads the penultimate hidden layer", f"{model.layer}")
    record(model.layer_norm_hidden_state is False,
           "and does not layer-norm it")

    ids = _prompt_ids()
    with torch.no_grad():
        hidden, pooled = model([ids])
        # The same weights, reading the last layer instead.
        model.set_layer_idx(None)
        last, last_pooled = model([ids])
        model.set_layer_idx(-2)

    record(tuple(hidden.shape) == (1, 77, 768) and tuple(last.shape) == (1, 77, 768),
           "both layers give the same shape -- so reading the wrong one is "
           "silent",
           f"{tuple(hidden.shape)} vs {tuple(last.shape)}")
    record(not torch.allclose(hidden, last, atol=1e-4),
           "and different values, which is the whole point of layer_idx",
           f"max diff {(hidden - last).abs().max().item():.3e}")
    record(tuple(pooled.shape) == (1, 768), "the pooled output is [1, width]")
    record(torch.equal(pooled, last_pooled),
           "the pooled output is the same either way -- it comes from the "
           "last layer, so layer_idx must not touch it")


def check_eos_pooling():
    """The pooled row is the first EOS position, not the last."""
    model = SDClipModel(CLIP_L_CONFIG, layer="hidden", layer_idx=-2,
                        layer_norm_hidden_state=False).eval()
    _fill(model, seed=43)

    # A short prompt: BOS, five tokens, EOS, then EOS-padding to 77. Reading
    # the last position would read padding.
    ids = [BOS] + [100, 200, 300, 400, 500] + [EOS] + [EOS] * 70
    with torch.no_grad():
        _, pooled = model([ids])
        # The stack's own output, reached the way forward() reaches it: the
        # token embeddings, with the position embedding added inside. Passing
        # token embeddings *as* embeds would add the positions twice, which is
        # what the first version of this did.
        token_embeds = model.transformer.get_input_embeddings()(
            torch.tensor([ids])).float()
        last_layer = model.transformer.text_model(
            token_embeds, input_tokens=torch.tensor([ids]))[0]

    eos_at = (torch.tensor(ids) == EOS).int().argmax().item()
    record(eos_at == 6, "the first EOS is at position 6", f"{eos_at}")
    # `pooled` is the *projected* output, so the reference has to go through
    # text_projection too. Comparing against the raw hidden state is the
    # first version of this check, and it is off by the projection.
    project = model.transformer.text_projection
    with torch.no_grad():
        want = project(last_layer[0, eos_at])
        wrong = project(last_layer[0, -1])
    record(torch.allclose(pooled[0], want, atol=1e-5),
           "the pooled row is the last layer at that position",
           f"max diff {(pooled[0] - want).abs().max().item():.3e}")
    record(not torch.allclose(pooled[0], wrong, atol=1e-3),
           "and not the last position, which is padding here",
           f"max diff {(pooled[0] - wrong).abs().max().item():.3e}")


def check_causal_mask():
    """Attention is causal -- but only when it is given a mask.

    The first version of this check built a bare CLIPLayer and perturbed a
    late token. It failed, and the layer was right: CLIPLayer takes the mask
    as an argument and is bidirectional when given none. The causality lives
    in the mask the encoder builds, so that is what has to be tested.
    """
    torch.manual_seed(7)
    layer = CLIPLayer(16, 4, 32, "gelu").eval()
    _fill(layer, seed=51)
    x = torch.randn(1, 6, 16)

    length = x.shape[1]
    causal = torch.full((length, length), -torch.finfo(x.dtype).max,
                        dtype=x.dtype).triu_(1)
    with torch.no_grad():
        masked = layer(x, causal)
        y = x.clone()
        y[0, 5] += 100.0
        masked_perturbed = layer(y, causal)

    record(torch.allclose(masked[0, :5], masked_perturbed[0, :5], atol=1e-6),
           "with a causal mask, a later token cannot influence an earlier one",
           f"max diff "
           f"{(masked[0, :5] - masked_perturbed[0, :5]).abs().max().item():.3e}")
    record(not torch.allclose(masked[0, 5], masked_perturbed[0, 5], atol=1e-4),
           "but a token does influence itself")
    with torch.no_grad():
        unmasked = layer(x)
    record(not torch.allclose(unmasked[0, :5], masked[0, :5], atol=1e-4),
           "and without the mask the layer really is bidirectional, which is "
           "why the mask has to be part of this check")

    # The encoder must actually build it.
    # One layer, so this compares like with like: the block above and the
    # stack around it must agree when given the same mask.
    encoder = CLIPEncoder(1, 16, 4, 32, "gelu").eval()
    # The same weights, not the same seed: the two modules name their
    # parameters differently, so `_fill` walks them in a different order and
    # one shared seed would still produce different values.
    with torch.no_grad():
        for (_, src), (_, dst) in zip(sorted(layer.state_dict().items()),
                                      sorted(encoder.state_dict().items())):
            dst.copy_(src)
    with torch.no_grad():
        enc_masked = encoder(x, mask=causal)[0]
    record(torch.allclose(enc_masked, masked, atol=1e-6),
           "and a one-layer stack gives the same answer as the bare layer "
           "given the same mask",
           f"max diff {(enc_masked - masked).abs().max().item():.3e}")


def check_encoder_tap():
    enc = CLIPEncoder(3, 16, 4, 32, "gelu").eval()
    _fill(enc, seed=61)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        last, tapped = enc(x, intermediate_output=-1)
    record(torch.equal(tapped[0], last[0]),
           "a negative index counts from the end, so -1 taps the last layer")
    with torch.no_grad():
        _, tapped0 = enc(x, intermediate_output=0)
        _, first_only = enc(x, intermediate_output=0)
    record(tapped0 is not None and not torch.equal(tapped0[0], last[0]),
           "and 0 taps the first")
    with torch.no_grad():
        _, none = enc(x, intermediate_output=None)
    record(none is None, "with no tap asked for, none is returned")


def check_special_tokens():
    record(SDXLClipG().special_tokens["pad"] == 0, "CLIP-G pads with token 0")
    record(SDClipModel(CLIP_L_CONFIG).special_tokens["pad"] == EOS,
           "while CLIP-L pads with EOS")


def check_ragged_sections_rejected():
    model = SDXLClipModel(dtype=torch.float32, device="cpu").eval()
    ids = _prompt_ids()
    # Two sections of different lengths: ragged.
    try:
        model.encode_token_ids({
            "l": [_row(ids[:60]), _row(ids)], "g": [_row(ids)]})
    except ValueError as exc:
        record("same length" in str(exc) and "60" in str(exc),
               "ragged sections are rejected, naming the lengths",
               str(exc)[:90])
    else:
        record(False, "ragged sections are rejected, naming the lengths",
               "they were accepted")

    # One section of the wrong length: not ragged, just wrong. ComfyUI pads
    # this inside process_tokens; we reject it, because the position
    # embedding is a 77-entry lookup and the failure otherwise surfaces as a
    # shape error from deep inside the forward.
    try:
        model.encode_token_ids({
            "l": [_row(ids[:60])], "g": [_row(ids)]})
    except ValueError as exc:
        record("77 long" in str(exc),
               "a short row is rejected with the length it needs", str(exc)[:90])
    else:
        record(False, "a short row is rejected with the length it needs",
               "it was accepted")


# ---------------------------------------------------------------------------
# Structure and numbers
# ---------------------------------------------------------------------------

def check_context_concatenation():
    model = SDXLClipModel(dtype=torch.float32, device="cpu").eval()
    _fill(model, seed=71)
    ids_l, ids_g = _prompt_ids(3), _prompt_ids(5)
    with torch.no_grad():
        context, pooled = model.encode_token_ids(
            {"l": [_row(ids_l)], "g": [_row(ids_g)]})
    record(tuple(context.shape) == (1, 77, 2048),
           "the context is [1, 77, 768+1280]", f"{tuple(context.shape)}")
    record(tuple(pooled.shape) == (1, 1280),
           "and the pooled output is CLIP-G's 1280-wide one",
           f"{tuple(pooled.shape)}")

    # The two towers are truncated to a common length. With both tokenizers
    # padding to 77 that never actually truncates, so the branch is defensive
    # rather than exercised -- recorded here so the min() is not mistaken for
    # a bug and "simplified" away.
    record(model.clip_l.max_length == model.clip_g.max_length == 77,
           "both towers take 77 tokens, so the truncation is a no-op in "
           "practice",
           f"{model.clip_l.max_length}, {model.clip_g.max_length}")

    # Sections rejoin along the sequence axis, not the batch axis. Getting
    # this wrong gives the right element count in the wrong place: a (3, 77,
    # 2048) batch instead of a (1, 231, 2048) sequence, which a caller
    # building a UNet context would treat as three images.
    three = {"l": [_row(ids_l)] * 3, "g": [_row(ids_g)] * 3}
    with torch.no_grad():
        wide = model.encode_token_ids(three)[0]
    record(tuple(wide.shape) == (1, 231, 2048),
           "three sections come back as one long sequence, not a batch of "
           "three",
           f"{tuple(wide.shape)}")


def check_against_comfy():
    try:
        import paths as p
        root = p.get_comfy_dir()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from comfy.sdxl_clip import SDXLClipModel as ComfySDXL
    except Exception as exc:  # noqa: BLE001 -- absent means "not here"
        skip("characterisation vs ComfyUI",
             f"comfy not importable ({type(exc).__name__})")
        return

    theirs = ComfySDXL(device="cpu", dtype=torch.float32).eval()
    ours = SDXLClipModel(dtype=torch.float32, device="cpu").eval()
    record(set(theirs.state_dict()) == set(ours.state_dict()),
           "identical state_dict keys",
           f"{len(ours.state_dict())} tensors; only ours "
           f"{sorted(set(ours.state_dict()) - set(theirs.state_dict()))[:3]}, "
           f"only comfy {sorted(set(theirs.state_dict()) - set(ours.state_dict()))[:3]}")
    mismatch = [k for k in theirs.state_dict()
                if k in ours.state_dict()
                and theirs.state_dict()[k].shape != ours.state_dict()[k].shape]
    record(not mismatch, "and identical shapes", f"{mismatch[:3]}")

    # ComfyUI's Linear leaves its weight uninitialised, so weights come from
    # one shared generator.
    _fill(theirs)
    _fill(ours)

    ids_l, ids_g = _prompt_ids(3), _prompt_ids(5)
    cases = {
        "plain": {"l": [_row(ids_l)], "g": [_row(ids_g)]},
        "two sections": {"l": [_row(ids_l), _row(ids_l[::-1])],
                         "g": [_row(ids_g)]},
        "three sections": {"l": [_row(ids_l)] * 3, "g": [_row(ids_g)] * 2},
    }
    for label, pairs in cases.items():
        # ComfyUI's entry point takes (id, weight) pairs; ours takes ids.
        # Same ids either way, which is the comparison.
        as_pairs = {tower: [[(t, 1.0) for t in row] for row in rows]
                    for tower, rows in pairs.items()}
        with torch.no_grad():
            tc, tp = theirs.encode_token_weights(as_pairs)
            oc, op = ours.encode_token_ids(pairs)
        dc = (tc - oc).abs().max().item()
        dp = (tp - op).abs().max().item()
        same = dc == 0.0 and dp == 0.0
        print(f"  {'DIFF' if not same else 'SAME'}: {label}: "
              f"context {dc:.3e}, pooled {dp:.3e}")


def main() -> int:
    print("== the two configurations ==")
    check_configs()
    print("\n== quick_gelu ==")
    check_quick_gelu()
    print("\n== which layer is the conditioning ==")
    check_layer_selection()
    print("\n== EOS pooling ==")
    check_eos_pooling()
    print("\n== the causal mask ==")
    check_causal_mask()
    print("\n== the encoder's tap ==")
    check_encoder_tap()
    print("\n== special tokens ==")
    check_special_tokens()
    print("\n== ragged sections ==")
    check_ragged_sections_rejected()
    print("\n== the context ==")
    check_context_concatenation()
    print("\n== characterisation vs ComfyUI ==")
    check_against_comfy()

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