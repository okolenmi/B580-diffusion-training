"""Correctness check for nodes/model/clip_state_dict.py -- the checkpoint
key translation this project now owns instead of importing from ComfyUI's
`utils` (docs/design/12-installer-and-comfy-decoupling.md section 7.2).

The tables are the easy part; what is worth a test is everything about them
that is surprising, because all of it is behaviour a tidy-up would break:

- **Both functions mutate the dict they are given.** They pop from it and
  return it. `state_dict_prefix_replace(filter_keys=True)` returns a
  *different* dict, so each key ends up in exactly one of the two -- a
  matched key in the return value and gone from the caller's, an unmatched
  key the other way round.
- **`transformers_convert` stores views, not copies.** Splitting
  `attn.in_proj_{weight,bias}` slices the popped tensor, so q/k/v alias one
  buffer. Adding a `.contiguous()` would be a behaviour change.
- **The iteration must not mutate the dict it walks.** The matched keys are
  collected before any pop; iterating the live dict raises.
- **`text_projection` has two spellings** and the transposed one is
  transposed on the way in.

Compares against ComfyUI as characterisation, not a gate, and skips rather
than fails when ComfyUI is not importable -- the same reasoning as
smoke_test_timestep_embedding.py: this project is being decoupled from those
files, and a test that needed them present would put the coupling back.

Run: `python nodes/smoke_tests/smoke_test_clip_state_dict.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nodes.model.clip_state_dict import (  # noqa: E402
    clip_text_transformers_convert,
    state_dict_prefix_replace,
    transformers_convert,
)

failures: list[str] = []
skipped: list[str] = []

PREFIXES = {
    "conditioner.embedders.0.transformer.text_model":
        "clip_l.transformer.text_model",
    "conditioner.embedders.1.model.": "clip_g.",
}


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


# ---------------------------------------------------------------------------
# state_dict_prefix_replace
# ---------------------------------------------------------------------------

def check_prefix_replace():
    sd = {
        "a.one": 1, "a.two": 2,
        "b.one": 3,
        "c.unmatched": 4,
    }

    out = state_dict_prefix_replace(sd, {"a.": "A.", "b.": "B."}, filter_keys=True)
    record(sorted(out) == ["A.one", "A.two", "B.one"],
           "filter_keys=True returns only the renamed keys",
           f"got {sorted(out)}")
    record(sorted(sd) == ["c.unmatched"],
           "and leaves the unmatched key in the caller's dict",
           f"got {sorted(sd)}")
    record(out is not sd, "the returned dict is not the one passed in")
    record(out["A.one"] == 1 and out["B.one"] == 3, "values came across")

    # filter_keys=False: same object, old and new spellings both present.
    sd2 = {"a.one": 1, "b.one": 3}
    out2 = state_dict_prefix_replace(sd2, {"a.": "A."}, filter_keys=False)
    record(out2 is sd2, "filter_keys=False returns the dict it was given")
    record(sorted(out2) == ["A.one", "b.one"],
           "and keeps the unmatched key alongside the renamed one",
           f"got {sorted(out2)}")

    # No prefix matches: nothing moves, nothing is lost.
    sd3 = {"x.y": 1}
    out3 = state_dict_prefix_replace(sd3, {"q.": "Q."}, filter_keys=True)
    record(out3 == {} and sd3 == {"x.y": 1},
           "no matching prefix leaves both dicts intact",
           f"returned {out3}, caller holds {sd3}")

    # Many keys under one prefix -- this is where mutating the dict being
    # iterated would raise RuntimeError.
    sd4 = {f"a.k{i}": i for i in range(50)}
    out4 = state_dict_prefix_replace(sd4, {"a.": "A."}, filter_keys=True)
    record(len(out4) == 50 and sd4 == {},
           "50 keys under one prefix convert without a dict-size error",
           f"returned {len(out4)}, caller holds {len(sd4)}")


# ---------------------------------------------------------------------------
# transformers_convert
# ---------------------------------------------------------------------------

def _one_clip(prefix: str = "p.", blocks: int = 1) -> dict:
    """A minimal but complete input in the checkpoint's spelling."""
    sd = {
        f"{prefix}positional_embedding": torch.randn(77, 32),
        f"{prefix}token_embedding.weight": torch.randn(77, 32),
        f"{prefix}ln_final.weight": torch.randn(32),
        f"{prefix}ln_final.bias": torch.randn(32),
    }
    for i in range(blocks):
        for name, shape in (
            ("ln_1.weight", (32, 32)), ("ln_1.bias", (32,)),
            ("ln_2.weight", (32, 32)), ("ln_2.bias", (32,)),
            ("mlp.c_fc.weight", (64, 32)), ("mlp.c_fc.bias", (64,)),
            ("mlp.c_proj.weight", (32, 64)), ("mlp.c_proj.bias", (32,)),
            ("attn.out_proj.weight", (32, 32)), ("attn.out_proj.bias", (32,)),
            ("attn.in_proj_weight", (96, 32)), ("attn.in_proj_bias", (96,)),
        ):
            sd[f"{prefix}transformer.resblocks.{i}.{name}"] = torch.randn(*shape)
    return sd


def check_transformers_convert():
    sd = _one_clip()
    out = transformers_convert(sd, "p.", "q.", 1)

    # The four top-level renames, named exactly.
    record("q.embeddings.position_embedding.weight" in out,
           "positional_embedding -> embeddings.position_embedding.weight")
    record("q.embeddings.token_embedding.weight" in out,
           "token_embedding.weight -> embeddings.token_embedding.weight")
    record("q.final_layer_norm.weight" in out and "q.final_layer_norm.bias" in out,
           "ln_final -> final_layer_norm (weight and bias)")

    # The five per-block renames, at block 0.
    layer = "q.encoder.layers.0"
    for expected in ("layer_norm1.weight", "layer_norm1.bias",
                     "layer_norm2.weight", "layer_norm2.bias",
                     "mlp.fc1.weight", "mlp.fc1.bias",
                     "mlp.fc2.weight", "mlp.fc2.bias",
                     "self_attn.out_proj.weight", "self_attn.out_proj.bias"):
        if f"{layer}.{expected}" not in out:
            record(False, f"{expected} is renamed into the layer")
            return
    record(True, "all five per-block renames land on encoder.layers.0")

    # Nothing left behind under the old spellings.
    stale = [k for k in out if "transformer.resblocks" in k or "ln_final" in k
             or k.startswith("p.")]
    record(not stale, "no source key survives the conversion", f"{stale[:4]}")

    # The block count is honoured exactly: asking for one block converts
    # block 0 and leaves block 1 entirely untouched, rather than converting
    # everything or half-converting the tail.
    out1 = transformers_convert(_one_clip(blocks=2), "p.", "q.", 1)
    record("q.encoder.layers.0.layer_norm1.weight" in out1,
           "asking for one block converts block 0")
    record("p.transformer.resblocks.1.ln_1.weight" in out1,
           "and leaves block 1 under its source name",
           f"block 1 keys: "
           f"{[k for k in out1 if 'resblocks.1' in k][:2]}")
    record("q.encoder.layers.1.layer_norm1.weight" not in out1,
           "block 1 is not converted under a new name either")
    out2 = transformers_convert(_one_clip(blocks=2), "p.", "q.", 2)
    record("q.encoder.layers.1.layer_norm1.weight" in out2,
           "asking for two blocks converts block 1 as well")


def check_in_proj_is_split_into_views():
    """The fused QKV split, and the part that is easy to change by accident."""
    packed = torch.arange(96 * 32, dtype=torch.float32).reshape(96, 32)
    sd = {"p.transformer.resblocks.0.attn.in_proj_weight": packed}
    out = transformers_convert(sd, "p.", "q.", 1)

    pre = "q.encoder.layers.0.self_attn"
    record(all(f"{pre}.{n}.weight" in out for n in ("q_proj", "k_proj", "v_proj")),
           "in_proj_weight becomes q_proj, k_proj and v_proj")

    q = out[f"{pre}.q_proj.weight"]
    k = out[f"{pre}.k_proj.weight"]
    v = out[f"{pre}.v_proj.weight"]
    record(q.shape == k.shape == v.shape == (32, 32),
           "each projection is a third of the packed tensor",
           f"{tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    record(torch.equal(q, packed[:32]) and torch.equal(k, packed[32:64])
           and torch.equal(v, packed[64:]),
           "and each holds the right third, in q/k/v order")

    # The aliasing. This is the claim that makes a `.contiguous()` here a
    # behaviour change rather than a tidy-up, so it is checked by storage
    # identity -- writing q[0,0] would not touch v[0,0], because that is a
    # different row of the same buffer.
    record(q.untyped_storage().data_ptr() == v.untyped_storage().data_ptr(),
           "the three projections share one buffer, i.e. are views")
    record((q.storage_offset(), k.storage_offset(), v.storage_offset())
           == (0, 32 * 32, 2 * 32 * 32),
           "at offsets 0, third and two-thirds of it",
           f"{(q.storage_offset(), k.storage_offset(), v.storage_offset())}")

    # The bias splits the same way.
    packed_b = torch.arange(96, dtype=torch.float32)
    out2 = transformers_convert(
        {"p.transformer.resblocks.0.attn.in_proj_bias": packed_b}, "p.", "q.", 1)
    record(torch.equal(out2[f"{pre}.q_proj.bias"], packed_b[:32])
           and torch.equal(out2[f"{pre}.v_proj.bias"], packed_b[64:]),
           "in_proj_bias splits the same way")


def check_text_projection_forms():
    src = torch.randn(32, 8)

    sd = {"clip_g.text_projection": src.clone()}
    out = clip_text_transformers_convert(sd, "clip_g.", "clip_g.transformer.")
    target = "clip_g.transformer.text_projection.weight"
    record(target in out, "the bare 'text_projection' key is renamed")
    record(torch.equal(out[target], src.transpose(0, 1).contiguous()),
           "and transposed, because the checkpoint stores it that way")

    sd2 = {"clip_g.text_projection.weight": src.clone()}
    out2 = clip_text_transformers_convert(sd2, "clip_g.", "clip_g.transformer.")
    record(torch.equal(out2[target], src),
           "the already-suffixed form is renamed without a transpose")

    # Both present: the second overwrites the first. ComfyUI's order, kept
    # deliberately; pinning it so a "tidier" either/or is noticed.
    other = torch.randn(32, 8)
    sd3 = {"clip_g.text_projection.weight": src.clone(),
           "clip_g.text_projection": other.clone()}
    out3 = clip_text_transformers_convert(sd3, "clip_g.", "clip_g.transformer.")
    record(torch.equal(out3[target], other.transpose(0, 1).contiguous()),
           "with both spellings present the transposed one wins")
    record(len(out3) == 1, "and only one key results, not two",
           f"{sorted(out3)}")


# ---------------------------------------------------------------------------
# Characterisation against ComfyUI
# ---------------------------------------------------------------------------

def _comfy_module():
    import paths as _paths
    root = _paths.get_comfy_dir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import comfy.utils
    return comfy.utils


def check_against_comfy():
    try:
        cu = _comfy_module()
    except Exception as exc:  # noqa: BLE001 -- absent means "not here"
        skip("characterisation vs ComfyUI",
             f"comfy not importable ({type(exc).__name__})")
        return

    # A whole CLIP-G, same seed on both sides, compared key by key.
    def clip_g(text_projection_key: str):
        sd = _one_clip("clip_g.", blocks=2)
        sd[text_projection_key] = torch.randn(32, 8)
        return sd

    for label, key in (("with text_projection.weight",
                        "clip_g.text_projection.weight"),
                       ("with text_projection", "clip_g.text_projection")):
        torch.manual_seed(3)
        mine = clip_g(key)
        torch.manual_seed(3)
        theirs = clip_g(key)
        clip_text_transformers_convert(mine, "clip_g.", "clip_g.transformer.")
        cu.clip_text_transformers_convert(theirs, "clip_g.", "clip_g.transformer.")
        same_keys = set(mine) == set(theirs)
        same_values = same_keys and all(
            torch.equal(mine[k], theirs[k]) for k in theirs)
        record(same_keys, f"same key set as ComfyUI, {label}",
               f"{len(mine)} keys vs {len(theirs)}; "
               f"only ours {sorted(set(mine) - set(theirs))[:3]}, "
               f"only comfy {sorted(set(theirs) - set(mine))[:3]}")
        record(same_values, f"every value equal to ComfyUI's, {label}")

    # The aliasing claim, checked against ComfyUI's rather than assumed.
    torch.manual_seed(5)
    packed = torch.arange(96 * 32, dtype=torch.float32).reshape(96, 32)
    mine = {"clip_g.transformer.resblocks.0.attn.in_proj_weight": packed.clone()}
    theirs = {"clip_g.transformer.resblocks.0.attn.in_proj_weight": packed.clone()}
    clip_text_transformers_convert(mine, "clip_g.", "clip_g.transformer.")
    cu.clip_text_transformers_convert(theirs, "clip_g.", "clip_g.transformer.")
    pre = "clip_g.transformer.text_model.encoder.layers.0.self_attn"

    def aliases(sd):
        # Each implementation must alias *its own* q/k/v. Comparing one
        # implementation's storage pointer to another's would be comparing
        # two separate allocations, which are never equal.
        return (sd[f"{pre}.q_proj.weight"].untyped_storage().data_ptr()
                == sd[f"{pre}.k_proj.weight"].untyped_storage().data_ptr()
                == sd[f"{pre}.v_proj.weight"].untyped_storage().data_ptr())

    record(aliases(mine) and aliases(theirs),
           "and the QKV aliasing matches ComfyUI's, not just our own",
           f"ours={aliases(mine)} comfy={aliases(theirs)}")
    record(
        mine[f"{pre}.q_proj.weight"].storage_offset()
        == theirs[f"{pre}.q_proj.weight"].storage_offset()
        == 0
        and mine[f"{pre}.v_proj.weight"].storage_offset()
        == theirs[f"{pre}.v_proj.weight"].storage_offset(),
        "with the same offsets into it",
    )

    # transformers_convert on its own.
    torch.manual_seed(11)
    a = _one_clip("p.", blocks=1)
    torch.manual_seed(11)
    b = _one_clip("p.", blocks=1)
    transformers_convert(a, "p.", "q.", 1)
    cu.transformers_convert(b, "p.", "q.", 1)
    record(set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in b),
           "transformers_convert matches ComfyUI on its own")


def main() -> int:
    print("== state_dict_prefix_replace ==")
    check_prefix_replace()
    print("\n== transformers_convert ==")
    check_transformers_convert()
    print("\n== the fused QKV split ==")
    check_in_proj_is_split_into_views()
    print("\n== text_projection ==")
    check_text_projection_forms()
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