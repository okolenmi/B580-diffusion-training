"""Checkpoint key translation for the SDXL text encoders -- owned here.

Design doc 12, section 7.2. `clip_encoder.py` needs two helpers from
ComfyUI's `utils` to rename the conditioner keys in a `.safetensors` state
dict into the `clip_l.` / `clip_g.` names its modules expect. Those helpers
call a third one, so the real dependency is about sixty lines, not the nine
the design doc originally claimed.

Why own it. `import comfy.utils` costs **2.71 s and 2136 modules** on this
machine (measured, against 0.01 s and nothing for an empty interpreter),
and `_extract_and_convert_clip_state_dict` runs it lazily at model-load
time -- so the first SDXL text-encoder load pays it.

What this is: a table of old checkpoint key spellings and new module
attribute names. The tables are a reading of the checkpoint format, not an
architectural decision, and the split of one fused `in_proj` tensor into
three projections is arithmetic on a shape. Nothing here expresses an
opinion about how a transformer should be built, which is why owning it is
safe in a way that owning `UNetModel` is not.

Provenance: `state_dict_prefix_replace` and `clip_text_transformers_convert`
are from ComfyUI's `comfy/utils.py` (Apache-2.0), and `transformers_convert`
from the same file. The behaviour below is reproduced deliberately,
including the two things that surprise people:

* **Both functions mutate their input.** They `pop` from the dict they are
  given and return it. `state_dict_prefix_replace` with `filter_keys=True`
  is the strange one: it returns a *different* dict containing only the
  renamed keys, while the caller's dict keeps the untouched keys and loses
  the renamed ones. A key is therefore in exactly one of the two dicts, and
  for a key that matches no prefix it is only in the caller's.
* **`transformers_convert` stores views, not copies.** Splitting
  `attn.in_proj_{weight,bias}` slices the popped tensor rather than cloning
  it, so the resulting entries alias one 3x-wide buffer. `clip_l`'s Q/K/V
  weights are three windows onto the same tensor. Preserved on purpose --
  copying would change what a later in-place edit to one projection does to
  the others -- but it means a `.contiguous()` added here is a behaviour
  change and not a tidy-up.
"""

from __future__ import annotations

__all__ = [
    "clip_text_transformers_convert",
    "state_dict_prefix_replace",
    "transformers_convert",
]


def state_dict_prefix_replace(
    state_dict: dict,
    replace_prefix: dict[str, str],
    filter_keys: bool = False,
) -> dict:
    """Rename keys by prefix, in place, optionally dropping what did not match.

    :param state_dict: the dict to rewrite. **Mutated** -- matched keys are
        popped from it.
    :param replace_prefix: old prefix -> new prefix. A prefix that is
        itself a prefix of another entry is not safe here; the first match
        in iteration order wins and the key is gone for the second.
    :param filter_keys: when False, returns `state_dict` itself with the
        renamed keys added, so old and new spellings both survive. When
        True, returns a *new* dict holding only the renamed keys.
    :return: see `filter_keys`.
    """
    out: dict = {} if filter_keys else state_dict
    for old_prefix, new_prefix in replace_prefix.items():
        for key in [k for k in state_dict if k.startswith(old_prefix)]:
            # `[k for k in ...]` rather than iterating the dict directly: the
            # loop body pops from `state_dict`, and mutating a dict during
            # iteration raises.
            out[new_prefix + key[len(old_prefix):]] = state_dict.pop(key)
    return out


#: HuggingFace transformer spellings -> the names `CLIPTextModel` uses.
#: ComfyUI's names for the same thing; they are the checkpoint's, and the
#: checkpoints are what fixes them.
_SD_KEYS_TO_REPLACE = {
    "positional_embedding": "embeddings.position_embedding.weight",
    "token_embedding.weight": "embeddings.token_embedding.weight",
    "ln_final.weight": "final_layer_norm.weight",
    "ln_final.bias": "final_layer_norm.bias",
}

#: Per-resblock submodule renames. Applied to 32 blocks, which is why this
#: cannot be a straight rename: the block index has to be threaded through.
_RESBLOCK_KEYS_TO_REPLACE = {
    "ln_1": "layer_norm1",
    "ln_2": "layer_norm2",
    "mlp.c_fc": "mlp.fc1",
    "mlp.c_proj": "mlp.fc2",
    "attn.out_proj": "self_attn.out_proj",
}

#: The three projections packed into one `in_proj`. Order matters and is the
#: checkpoint's: q, then k, then v.
_FUSED_QKV = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")


def transformers_convert(
    sd: dict,
    prefix_from: str,
    prefix_to: str,
    number: int,
) -> dict:
    """Convert one CLIP's keys to `CLIPTextModel` spellings, in place.

    :param sd: the dict to rewrite. **Mutated**.
    :param prefix_from: the checkpoint's prefix for this CLIP.
    :param prefix_to: the prefix for the destination module.
    :param number: how many transformer resblocks to walk. SDXL's CLIP-L is
        32 and CLIP-G is 32.
    :return: `sd`.
    """
    for old, new in _SD_KEYS_TO_REPLACE.items():
        key = f"{prefix_from}{old}"
        if key in sd:
            sd[f"{prefix_to}{new}"] = sd.pop(key)

    for resblock in range(number):
        for old, new in _RESBLOCK_KEYS_TO_REPLACE.items():
            for suffix in ("weight", "bias"):
                key = f"{prefix_from}transformer.resblocks.{resblock}.{old}.{suffix}"
                moved = f"{prefix_to}encoder.layers.{resblock}.{new}.{suffix}"
                if key in sd:
                    sd[moved] = sd.pop(key)

        for suffix in ("weight", "bias"):
            key = f"{prefix_from}transformer.resblocks.{resblock}.attn.in_proj_{suffix}"
            if key not in sd:
                continue
            packed = sd.pop(key)
            third = packed.shape[0] // 3
            for i, projection in enumerate(_FUSED_QKV):
                moved = f"{prefix_to}encoder.layers.{resblock}.{projection}.{suffix}"
                # A slice, not a clone -- see the module docstring.
                sd[moved] = packed[third * i:third * (i + 1)]

    return sd


def clip_text_transformers_convert(
    sd: dict,
    prefix_from: str,
    prefix_to: str,
) -> dict:
    """Convert a CLIP-G key set, in place.

    CLIP-L needs nothing here: its keys are already under the prefix the
    caller renamed them to, so only the CLIP-G packer needs unpacking.
    `prefix_from` and `prefix_to` both name the same encoder, differing only
    by where the `text_model.` part sits.

    The two `text_projection` forms are the OpenAI and HuggingFace
    spellings of the same weight -- the second is stored transposed, which
    is why it gets a transpose here. They are checked independently rather
    than as an either/or, so a checkpoint carrying both ends up with the
    second one's value. That is ComfyUI's behaviour and it is kept: it is
    not worth guessing which spelling is authoritative, and no checkpoint
    known to carry both would load either way.
    """
    sd = transformers_convert(sd, prefix_from, f"{prefix_to}text_model.", 32)

    target = f"{prefix_to}text_projection.weight"

    key = f"{prefix_from}text_projection.weight"
    if key in sd:
        sd[target] = sd.pop(key)

    key = f"{prefix_from}text_projection"
    if key in sd:
        sd[target] = sd.pop(key).transpose(0, 1).contiguous()

    return sd