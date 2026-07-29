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
        self.calls = []
        self.unloaded = False

    def encode(self, prompt: str, batch_size: int, height: int, width: int):
        self.calls.append((prompt, batch_size, height, width))
        return torch.zeros(batch_size, 2), torch.zeros(batch_size, 1)

    def unload(self) -> None:
        self.unloaded = True


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

    assert len(inner.calls) == 3, f"expected exactly 3 real calls, got {inner.calls}"
    assert ("a cat", 2, 512, 512) in inner.calls
    assert ("a dog", 2, 512, 512) in inner.calls
    assert ("a cat", 2, 768, 768) in inner.calls
    print(f"    PASS: {len(inner.calls)} real encode calls for 3 unique keys "
          f"across 4 batches (one repeat correctly deduplicated)")


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
    calls_after_warmup = len(inner.calls)
    encoder.encode("x", 1, 512, 512)
    encoder.encode("x", 1, 512, 512)
    assert len(inner.calls) == calls_after_warmup, "should be served entirely from cache"
    print("    PASS")


def check_unknown_key_degrades_not_breaks():
    print("[a genuinely new key after warmup still works -- degrades, doesn't crash]")
    inner = _CountingEncoder()
    dataset = _FakeDataset([_batch("x", 1, 64, 64)])
    result = PrewarmedTextEncoderNode().build(encoder=inner, dataset=dataset)
    encoder = result["encoder"]
    calls_before = len(inner.calls)
    ctx, y = encoder.encode("a completely different prompt", 3, 512, 512)
    assert len(inner.calls) == calls_before + 1, "an uncached key must still be served, via the unloaded encoder"
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
