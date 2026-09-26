"""Pure-Python-plus-torch-tensors check of nodes/model/text_encoder_cache.py.
No mocking framework needed -- a tiny FakeEncoder that counts real calls
is enough to prove caching actually skips them, and real (small) torch
tensors make the "hands back a usable (ctx, y) pair" check meaningful
rather than trivially true. A tiny _RecordingResourceControl fake, same
reasoning, covers the optional resource_control wiring: ensure_loaded()
called on a miss (and only a miss), with the right resource_name, both
directly on CachingTextEncoder and threaded through
CachingTextEncoderNode.build().

_CountingEncoder counts encode_prompt_only()/resolution_embedding() calls
separately (not a single combined counter the way this file's previous
version did, back when CachingTextEncoder cached on one combined key) --
that split, and being able to tell the two apart in a test, is the whole
point of this session's change to text_encoder_cache.py: see that
module's own docstring for why a single (prompt, batch_size, height,
width) key defeated caching almost entirely for exactly the dataset
shape (many resolutions, one repeated or empty prompt) this project's
own real use has hit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.memory.control_handle import ResourceControlHandle
from nodes.model.text_encoder import TextEncoder, TextEncoderNode
from nodes.model.text_encoder_cache import CachingTextEncoder, CachingTextEncoderNode


class _CountingEncoder(TextEncoder):
    def __init__(self):
        self.prompt_calls = 0
        self.resolution_calls = 0
        self.unloaded = False

    def encode_prompt_only(self, prompt: str, batch_size: int):
        self.prompt_calls += 1
        # Distinct per-call values so a cache bug (returning the wrong
        # cached entry) would actually be visible, not accidentally masked
        # by every call returning the same constant.
        ctx = torch.full((batch_size, 2), float(self.prompt_calls))
        pooled = torch.full((batch_size, 1), float(self.prompt_calls) * 10)
        return ctx, pooled

    def resolution_embedding(self, height: int, width: int, batch_size: int):
        self.resolution_calls += 1
        return torch.full((batch_size, 1), float(self.resolution_calls) * 100)

    def unload(self) -> None:
        self.unloaded = True

    def footprint_bytes(self) -> int:
        return 0

    def offload(self) -> None:
        pass

    def reload(self, device=None) -> None:
        pass

    def release(self) -> None:
        pass


class _RecordingResourceControl(ResourceControlHandle):
    """Records every ensure_loaded() call's name, in order -- enough to
    prove CachingTextEncoder calls it on a miss and not on a hit, without
    needing a mocking framework, same reasoning as _CountingEncoder above.
    register()/before_step()/release() are real no-ops here: CachingTextEncoder
    never calls any of them itself (see text_encoder_cache.py's own
    docstring -- it only ever calls ensure_loaded()), so nothing exercises
    them via this class, but ResourceControlHandle is an ABC and all are
    abstract.
    """

    def __init__(self):
        self.ensure_loaded_calls: list[str] = []

    def register(self, name: str, resident, offloadable: bool = False) -> None:
        pass

    def before_step(self, step: int) -> None:
        pass

    def ensure_loaded(self, name: str) -> None:
        self.ensure_loaded_calls.append(name)

    def release(self, name: str) -> None:
        pass

    def usable_budget_mb(self):
        return None


def check_contracts():
    print("[contracts]")
    assert not getattr(CachingTextEncoderNode, "__abstractmethods__", None)
    assert set(CachingTextEncoderNode.INPUTS) == {"encoder", "max_entries", "resource_control"}
    assert CachingTextEncoderNode.OUTPUTS == TextEncoderNode.OUTPUTS
    print("    PASS")


def check_hit_skips_the_real_call():
    print("[cache hit skips the wrapped encoder entirely]")
    inner = _CountingEncoder()
    cache = CachingTextEncoder(inner, max_entries=8)

    ctx1, y1 = cache.encode("a cat", batch_size=2, height=512, width=512)
    assert (inner.prompt_calls, inner.resolution_calls) == (1, 1)
    ctx2, y2 = cache.encode("a cat", batch_size=2, height=512, width=512)
    assert (inner.prompt_calls, inner.resolution_calls) == (1, 1), \
        "same key must not re-call the inner encoder, on either half"
    torch.testing.assert_close(ctx1, ctx2)
    torch.testing.assert_close(y1, y2)
    print("    PASS: identical (prompt, batch_size, height, width) reuses both cached halves")

    cache.encode("a cat", batch_size=4, height=512, width=512)
    assert (inner.prompt_calls, inner.resolution_calls) == (2, 2), \
        "a different batch_size changes both keys -- both halves must re-call"
    cache.encode("a dog", batch_size=2, height=512, width=512)
    assert (inner.prompt_calls, inner.resolution_calls) == (3, 3)
    print("    PASS: any differing key element forces a real call")


def check_prompt_and_resolution_caches_are_independent():
    print("[the real point of this session's change: same prompt at a new "
          "resolution skips CLIP entirely; same resolution with a new "
          "prompt skips the resolution embedding entirely]")
    inner = _CountingEncoder()
    cache = CachingTextEncoder(inner, max_entries=8)

    cache.encode("a cat", batch_size=1, height=512, width=512)
    assert (inner.prompt_calls, inner.resolution_calls) == (1, 1)

    # Same prompt, brand new resolution -- exactly this project's own real
    # "100 images without captions, every image a different resolution"
    # case: the (only) prompt should never be re-encoded.
    cache.encode("a cat", batch_size=1, height=768, width=1024)
    assert inner.prompt_calls == 1, \
        "same prompt at a new resolution must NOT re-run the CLIP forward pass"
    assert inner.resolution_calls == 2, \
        "a genuinely new (height, width) must still compute its own embedding"
    print("    PASS: repeated prompt, new resolution -- CLIP not re-run")

    # Same resolution as the very first call, brand new prompt -- the
    # symmetric case: the resolution embedding should never be recomputed.
    cache.encode("a dog", batch_size=1, height=512, width=512)
    assert inner.prompt_calls == 2, "a genuinely new prompt must still run CLIP"
    assert inner.resolution_calls == 2, \
        "a resolution already seen must NOT recompute its embedding"
    print("    PASS: repeated resolution, new prompt -- embedding not recomputed")


def check_resource_control_called_only_on_miss():
    print("[resource_control.ensure_loaded() called on a miss of either "
          "cache, not on a hit of both]")
    inner = _CountingEncoder()
    control = _RecordingResourceControl()
    cache = CachingTextEncoder(inner, max_entries=8, resource_control=control)

    cache.encode("a cat", batch_size=2, height=512, width=512)
    assert control.ensure_loaded_calls == ["text_encoder"], (
        "a miss must call ensure_loaded() exactly once, with the default "
        "resource_name, before the real encode")

    cache.encode("a cat", batch_size=2, height=512, width=512)
    assert control.ensure_loaded_calls == ["text_encoder"], (
        "a hit of both caches must not call ensure_loaded() again -- the "
        "whole point is that the inner encoder isn't touched")

    cache.encode("a cat", batch_size=2, height=999, width=999)
    assert control.ensure_loaded_calls == ["text_encoder", "text_encoder"], (
        "a prompt-cache hit alongside a resolution-cache miss is still a "
        "miss overall -- ensure_loaded() must fire")
    print("    PASS")

    print("[resource_name override is threaded through to ensure_loaded()]")
    inner2 = _CountingEncoder()
    control2 = _RecordingResourceControl()
    cache2 = CachingTextEncoder(inner2, resource_control=control2,
                                 resource_name="text_encoder_2")
    cache2.encode("a cat", batch_size=2, height=512, width=512)
    assert control2.ensure_loaded_calls == ["text_encoder_2"]
    print("    PASS")

    print("[no resource_control -> encode() still works, nothing to call]")
    inner3 = _CountingEncoder()
    cache3 = CachingTextEncoder(inner3)  # resource_control defaults to None
    cache3.encode("a cat", batch_size=2, height=512, width=512)
    assert inner3.prompt_calls == 1
    print("    PASS")


def check_eviction():
    print("[LRU eviction at max_entries, independently per cache]")
    inner = _CountingEncoder()
    cache = CachingTextEncoder(inner, max_entries=2)
    cache.encode("p1", 1, 64, 64)
    cache.encode("p2", 1, 64, 64)
    cache.encode("p3", 1, 64, 64)  # evicts p1's prompt entry (oldest, never re-touched)
    assert len(cache._prompt_cache) == 2
    assert ("p1", 1) not in cache._prompt_cache
    assert ("p3", 1) in cache._prompt_cache
    calls_before = inner.prompt_calls
    cache.encode("p1", 1, 64, 64)
    assert inner.prompt_calls == calls_before + 1, "p1 was evicted, must re-call"
    print("    PASS: oldest entry evicted once max_entries is exceeded")


def check_move_to_end_on_hit():
    print("[a cache hit refreshes recency, protecting it from eviction]")
    inner = _CountingEncoder()
    cache = CachingTextEncoder(inner, max_entries=2)
    cache.encode("p1", 1, 64, 64)
    cache.encode("p2", 1, 64, 64)
    cache.encode("p1", 1, 64, 64)  # hit -- should move p1 to most-recently-used
    cache.encode("p3", 1, 64, 64)  # should evict p2, not p1
    assert ("p1", 1) in cache._prompt_cache
    assert ("p2", 1) not in cache._prompt_cache
    print("    PASS: recently-hit entries survive eviction ahead of untouched ones")


def check_unload_and_clear_delegate():
    print("[unload()/clear_cache() delegate correctly, across both caches]")
    inner = _CountingEncoder()
    cache = CachingTextEncoder(inner, max_entries=8)
    cache.encode("p1", 1, 64, 64)
    cache.unload()
    assert inner.unloaded, "unload() must reach the wrapped encoder"
    cache.clear_cache()
    assert len(cache._prompt_cache) == 0
    assert len(cache._resolution_cache) == 0
    print("    PASS")


def check_node_build():
    print("[CachingTextEncoderNode.build()]")
    inner = _CountingEncoder()
    node = CachingTextEncoderNode()
    result = node.build(encoder=inner, max_entries=4)
    wrapped = result["encoder"]
    assert isinstance(wrapped, CachingTextEncoder)
    wrapped.encode("p", 1, 64, 64)
    wrapped.encode("p", 1, 64, 64)
    assert inner.prompt_calls == 1
    print("    PASS")

    print("[CachingTextEncoderNode.build() threads resource_control through]")
    inner2 = _CountingEncoder()
    control = _RecordingResourceControl()
    result2 = node.build(encoder=inner2, max_entries=4, resource_control=control)
    wrapped2 = result2["encoder"]
    wrapped2.encode("p", 1, 64, 64)
    assert control.ensure_loaded_calls == ["text_encoder"], (
        "build()'s resource_control must reach the constructed "
        "CachingTextEncoder, not be silently dropped")
    print("    PASS")


def main():
    check_contracts()
    check_hit_skips_the_real_call()
    check_prompt_and_resolution_caches_are_independent()
    check_resource_control_called_only_on_miss()
    check_eviction()
    check_move_to_end_on_hit()
    check_unload_and_clear_delegate()
    check_node_build()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
