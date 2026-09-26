"""Real torch tensors for batch shapes (needed since the node derives
height/width/batch_size from batch["x_t"].shape, exactly like
SupervisedLoRATrainerNode does) plus a counting fake encoder/dataset --
no need for real CLIP to verify the warm-then-unload logic itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.dataset.handle import TrainingBatchSource
from nodes.model.text_encoder import TextEncoder
from nodes.model.text_encoder_cache import CachingTextEncoder
from nodes.model.text_encoder_prewarm import PrewarmedTextEncoderNode


class _CountingEncoder(TextEncoder):
    def __init__(self):
        self.prompt_calls = []
        self.resolution_calls = []
        self.unloaded = False

    def encode_prompt_only(self, prompt: str, batch_size: int):
        self.prompt_calls.append((prompt, batch_size))
        return torch.zeros(batch_size, 2), torch.zeros(batch_size, 1)

    def resolution_embedding(self, height: int, width: int, batch_size: int):
        self.resolution_calls.append((height, width, batch_size))
        return torch.zeros(batch_size, 1)

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


class _FakeDataset(TrainingBatchSource):
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)

    def invalidate(self):
        pass


def _batch(prompt, batch_size, latent_h, latent_w):
    return {"x_t": torch.zeros(batch_size, 4, latent_h, latent_w), "prompt": prompt}


def check_contracts():
    print("[contracts]")
    assert not getattr(PrewarmedTextEncoderNode, "__abstractmethods__", None)
    assert set(PrewarmedTextEncoderNode.INPUTS) == {"encoder", "dataset"}
    print("    PASS")


def check_warms_exactly_the_keys_training_will_request():
    print("[warms exactly the (prompt, batch_size, height, width) keys the dataset implies]")
    inner = _CountingEncoder()
    # latent 64x64 -> pixel 512x512 (the *8 VAE factor); two distinct buckets.
    dataset = _FakeDataset([
        _batch("a cat", 2, 64, 64),
        _batch("a dog", 2, 64, 64),
        _batch("a cat", 2, 64, 64),  # repeat -- must not cause a second real call
        _batch("a cat", 2, 96, 96),  # different resolution -- must be its own key
    ])
    node = PrewarmedTextEncoderNode()
    result = node.build(encoder=inner, dataset=dataset)
    encoder = result["encoder"]
    assert isinstance(encoder, CachingTextEncoder)

    # NOT 3 (one per unique combined key, the pre-split-cache count) -- the
    # whole point of this session's text_encoder_cache.py change: "a cat" at
    # 512x512 and "a cat" at 768x768 share one prompt-encode; 512x512 for
    # "a cat" and 512x512 for "a dog" share one resolution-embed. See
    # text_encoder_cache.py's own module docstring.
    assert sorted(inner.prompt_calls) == sorted([("a cat", 2), ("a dog", 2)]), \
        f"expected exactly 2 distinct prompts encoded, got {inner.prompt_calls}"
    assert sorted(inner.resolution_calls) == sorted([(512, 512, 2), (768, 768, 2)]), \
        f"expected exactly 2 distinct resolutions embedded, got {inner.resolution_calls}"
    print(f"    PASS: {len(inner.prompt_calls)} real CLIP calls (2 unique prompts) + "
          f"{len(inner.resolution_calls)} real resolution-embed calls (2 unique resolutions) "
          f"across 4 batches / 3 unique combined keys -- fewer real calls than unique combined "
          f"keys, not just fewer than 4")


def check_unloads_after_warming():
    print("[underlying encoder is unloaded once warming finishes]")
    inner = _CountingEncoder()
    dataset = _FakeDataset([_batch("x", 1, 64, 64)])
    result = PrewarmedTextEncoderNode().build(encoder=inner, dataset=dataset)
    assert inner.unloaded
    print("    PASS")


def check_post_warmup_calls_are_free():
    print("[a request matching the warmed set makes zero further real calls]")
    inner = _CountingEncoder()
    dataset = _FakeDataset([_batch("x", 1, 64, 64)])
    result = PrewarmedTextEncoderNode().build(encoder=inner, dataset=dataset)
    encoder = result["encoder"]
    prompt_calls_after_warmup = len(inner.prompt_calls)
    resolution_calls_after_warmup = len(inner.resolution_calls)
    encoder.encode("x", 1, 512, 512)
    encoder.encode("x", 1, 512, 512)
    assert len(inner.prompt_calls) == prompt_calls_after_warmup, "should be served entirely from cache"
    assert len(inner.resolution_calls) == resolution_calls_after_warmup, "should be served entirely from cache"
    print("    PASS")


def check_unknown_key_degrades_not_breaks():
    print("[a genuinely new key after warmup still works -- degrades, doesn't crash]")
    inner = _CountingEncoder()
    dataset = _FakeDataset([_batch("x", 1, 64, 64)])
    result = PrewarmedTextEncoderNode().build(encoder=inner, dataset=dataset)
    encoder = result["encoder"]
    prompt_calls_before = len(inner.prompt_calls)
    resolution_calls_before = len(inner.resolution_calls)
    ctx, y = encoder.encode("a completely different prompt", 3, 512, 512)
    assert len(inner.prompt_calls) == prompt_calls_before + 1, \
        "an uncached prompt must still be served, via the unloaded encoder"
    assert len(inner.resolution_calls) == resolution_calls_before, \
        "512x512 was already warmed -- this new prompt shouldn't need a new resolution embed"
    assert ctx.shape == (3, 2)
    print("    PASS: falls back to a real (if now CPU-side) call rather than failing")


def main():
    check_contracts()
    check_warms_exactly_the_keys_training_will_request()
    check_unloads_after_warming()
    check_post_warmup_calls_are_free()
    check_unknown_key_degrades_not_breaks()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
