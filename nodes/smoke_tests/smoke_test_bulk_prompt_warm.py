"""Bulk prompt warming -- `CachingTextEncoder.warm_prompts` and the batched
path behind it.

The batched path exists because one prompt per forward costs 32.2 ms on the
B580 where 64 at a time cost 2.53 ms. It cannot be used in float16: batched
float16 diverges from serial by 19% of mean activation magnitude, which
would mean a warmed prompt and a cache-missed prompt get different
conditioning in the same run. Float32 agrees to 1.5e-6 relative to the
largest activation, which is rounding.

What these checks hold onto is the property that makes the whole thing safe:
**a warmed prompt and a missed one must be the same value.** Everything else
here is speed, and speed is not worth a semantic difference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.model.clip_encoder import SDXLClipEncoder
from nodes.model.text_encoder import SDXLTextEncoder, TextEncoder
from nodes.model.text_encoder_cache import CachingTextEncoder

CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"    PASS: {message}")


class LoopOnlyEncoder(TextEncoder):
    """A real `TextEncoder` that never overrides `encode_prompts`.

    Subclasses `TextEncoder` **directly**, not `SDXLTextEncoder` --
    subclassing the latter would inherit its batched implementation and this
    would silently stop testing the loop it claims to test, which is what an
    earlier version of this file did.

    Implements only the abstracts, so `encode_prompts` is the ABC's loop:
    correct for any encoder, just slower. Without this double the loop would
    be untested code that every real encoder bypasses.
    """

    def __init__(self, inner):
        self._inner = inner

    def encode_prompt_only(self, prompt, batch_size):
        return self._inner.encode_prompt_and_pool(prompt, batch_size)

    def resolution_embedding(self, height, width, batch_size):
        return self._inner.resolution_embedding(height, width, batch_size)

    def unload(self):
        pass

    def footprint_bytes(self):
        return 0

    def offload(self):
        pass

    def reload(self, device=None):
        pass

    def release(self):
        pass


class ShortResultEncoder(SDXLTextEncoder):
    """Returns fewer results than prompts -- a subclass bug."""

    def encode_prompts(self, prompts, batch_size=1):
        return []


def _relative(a, b) -> float:
    """Max absolute difference over the largest magnitude in `a`.

    Normalised by the max, not the mean: activations here run to ~132 while
    averaging ~0.2, so a mean-normalised number reads 700x worse than it is
    and turns a rounding difference into an apparent failure.

    Both sides moved to CPU first: the cache stores CPU tensors by contract,
    so one side of any cache-vs-fresh comparison is always on CPU and the
    other may be on the device.
    """
    a = a.float().cpu()
    b = b.float().cpu()
    scale = a.abs().max().item()
    return (a - b).abs().max().item() / max(scale, 1e-12)


def build_inner() -> tuple[SDXLClipEncoder, dict]:
    """A real CLIP in float32, plus the state dict it came from.

    Float32 because that is the only precision in which batching agrees with
    the serial path; a float16 inner here would make the equivalence check
    fail for the documented reason rather than a real one.

    The raw dict comes back too, because the fallback checks below need a
    float16 encoder and re-reading the checkpoint to make one would print a
    714-key warning that looks like a real failure and is an artefact of how
    the second encoder was built.
    """
    from nodes.smoke_tests.fast_construction import find_a_checkpoint, read_state_dicts

    _, clip_sd, _ = read_state_dicts(find_a_checkpoint())
    device = "xpu" if torch.xpu.is_available() else "cpu"
    inner = SDXLClipEncoder(clip_sd, device=device)
    inner.clip_model.float()
    return inner, clip_sd


PROMPTS = [
    "short",
    "a much longer caption with a fair number of words in it indeed, "
    "with commas and a trailing clause",
    "с кириллицей в подписи",
    "你好世界 mixed with latin",
    "punctuation! (parens) & symbols #1",
]


def main() -> None:
    inner, clip_sd = build_inner()
    sync = (lambda: torch.xpu.synchronize()) if torch.xpu.is_available() else (lambda: None)

    print("[warm_prompts: stores a batch, and a batch of one is the same "
          "computation as a miss]")
    enc = SDXLTextEncoder(inner)
    cached = CachingTextEncoder(enc, max_entries=64)

    enc.encode_prompts(PROMPTS[:1])          # one-time device setup, not timed
    sync()
    stored = cached.warm_prompts(PROMPTS)
    sync()
    check(stored == len(PROMPTS),
          f"every fresh prompt stored (got {stored} of {len(PROMPTS)})")
    check(len(cached._prompt_cache) == len(PROMPTS),
          f"cache holds them all (got {len(cached._prompt_cache)})")

    # The equivalence that justifies batching at all. Same prompt, warmed in
    # a batch of 5, versus encoded alone through the miss path.
    probe = PROMPTS[2]
    hit = cached.encode_prompt_only(probe, 1)
    alone = enc.encode_prompts([probe])[0]
    worst = max(_relative(h, a) for h, a in zip(hit, alone))
    # 1e-3 rather than something tight around what we happen to observe. The
    # measured figure is ~6e-5 -- bfloat16 rounding, since the encoder casts
    # its output to bfloat16 and one bf16 ulp is 3.9e-3 relative. The failure
    # this has to catch is the float16 regime at 1.9e-1, which is 190x above
    # the threshold; a threshold pinned near 6e-5 would have under 2x of
    # headroom over ordinary rounding and would flake on a platform whose
    # reduction order differs. A check that cries wolf gets ignored, and then
    # it is not checking.
    check(worst < 1e-3,
          f"a warmed prompt and a freshly encoded one agree (worst relative "
          f"{worst:.2e}, threshold 1e-3, float16 failure mode is 1.9e-1); a "
          f"difference here would mean conditioning depended on cache state")

    print("\n[warm_prompts: already-cached prompts are not re-encoded]")
    again = cached.warm_prompts(PROMPTS)
    check(again == 0, f"nothing re-encoded on a repeat warm (got {again})")
    check(len(cached._prompt_cache) == len(PROMPTS),
          "and the cache did not grow")

    print("\n[warm_prompts: edges]")
    check(cached.warm_prompts([]) == 0,
          "empty input stores nothing and asks the encoder for nothing")

    mixed = CachingTextEncoder(enc, max_entries=64)
    mixed.warm_prompts(PROMPTS[:2])
    check(mixed.warm_prompts(PROMPTS) == len(PROMPTS) - 2,
          "a partly-warm set stores only the missing half -- an encoder call "
          "whose result is discarded is the waste a warm pass exists to avoid")

    print("\n[the ABC's loop default still works, for an encoder with no "
          "batched path]")
    loop_enc = CachingTextEncoder(LoopOnlyEncoder(inner), max_entries=8)
    check(loop_enc.warm_prompts(["only one prompt"]) == 1,
          "an encoder that never overrides encode_prompts warms correctly")
    loop_hit = loop_enc.encode_prompt_only("only one prompt", 1)
    check(tuple(loop_hit[0].shape) == tuple(hit[0].shape),
          "and through the loop it stores the same shape a batched warm does")

    print("\n[warm_prompts stores under the batch_size it was asked for]")
    # A real bug that shipped in the first version of this path: it keyed at
    # batch_size 1 regardless, so warming at 2 while training asks at 2
    # stored every entry under a key nothing reads, and every step missed.
    # Invisible on this project's own datasets, which have exactly one
    # distinct prompt each -- the first prompt goes through `encode`, which is
    # always right. So it is pinned here rather than left to a run.
    bs2 = CachingTextEncoder(enc, max_entries=8)
    bs2.warm_prompts(["at two"], batch_size=2)
    stored_key = next(iter(bs2._prompt_cache))
    check(stored_key == ("at two", 2),
          f"a warm at batch_size 2 is keyed (prompt, 2) (got {stored_key})")
    served = bs2.encode_prompt_only("at two", 2)
    check(tuple(served[0].shape)[0] == 2,
          f"and a batch_size-2 request gets a batch of 2 back, not 1 "
          f"(got {tuple(served[0].shape)})")
    solo = bs2._inner.encode_prompt_only("at two", 2)
    check(_relative(served[0], solo[0]) < 1e-3,
          "and it matches what the miss path would have produced")

    print("\n[a subclass returning the wrong count is caught, not truncated]")
    short = CachingTextEncoder(ShortResultEncoder(inner), max_entries=8)
    try:
        short.warm_prompts(["a", "b"])
        check(False, "a short encode_prompts result must raise")
    except ValueError as exc:
        check("one (ctx, pooled) pair per prompt" in str(exc),
              f"a short encode_prompts result raises rather than silently "
              f"leaving those prompts uncached (got {str(exc)[:60]!r})")

    print("\n[production is float16, so the fallback is what actually runs]")
    # The thing that was silently wrong: `SDXLClipEncoder` loads float16
    # ("CLIP runs in fp16"), and batching in float16 diverges from the
    # serial path by 19%. So the batched path must refuse, not quietly run
    # and disagree with the cache-miss path.
    fp16_inner = SDXLClipEncoder(clip_sd, device="cpu")
    check(not fp16_inner.batching_available(),
          "a float16 encoder reports batching unavailable rather than "
          "batching anyway")
    fp16_cached = CachingTextEncoder(SDXLTextEncoder(fp16_inner), max_entries=8)
    check(not fp16_cached.batching_available(),
          "and the cache reports it from the encoder rather than guessing")
    check(fp16_cached.warm_prompts(["fallback prompt"]) == 1,
          "the float16 fallback still warms correctly")
    served16 = fp16_cached.encode_prompt_only("fallback prompt", 1)
    solo16 = fp16_inner.encode_prompt_and_pool("fallback prompt", 1)
    check(_relative(served16[0], solo16[0]) < 1e-3,
          "and a fallback-warmed prompt matches the miss path exactly, which "
          "is the property the refusal exists to preserve")
    check(tuple(fp16_inner.encode_prompts(["x"])[0][0].shape)
          == tuple(fp16_inner.encode_prompt_and_pool("x", 1)[0].shape),
          "encode_prompts and encode_prompt_and_pool agree on shape whether "
          "batched or not")

    print("\n[both cache entry points agree on what an insert does]")
    check(hasattr(CachingTextEncoder, "_insert_prompt"),
          "the insert is one shared method, so a warm pass cannot produce a "
          "cache state the miss path could not")

    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {CHECKS} CHECKS PASSED")


if __name__ == "__main__":
    main()
