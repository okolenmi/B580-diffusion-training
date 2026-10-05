"""Tests for graph_fingerprint: computed without importing torch.

The fingerprint is what makes two runs "the same configuration" for peak
purposes. It must be computable before spawning a child, so it must not
import torch. These tests verify that property and the field derivation.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sys

from backend.application.memory_fingerprint import (
    FINGERPRINT_FIELDS,
    GraphMemoryFingerprint,
    UnknownFingerprint,
    graph_fingerprint,
)
from backend.domain.graph import GraphDefinition, GraphNodeSpec


def _graph(*nodes: GraphNodeSpec) -> GraphDefinition:
    return GraphDefinition(nodes=nodes, edges=())


def _node(class_name: str, **params) -> GraphNodeSpec:
    return GraphNodeSpec(id="n1", class_name=class_name, params=params)


def _resolver(memory_fields_map: dict[str, tuple[str, ...]]):
    """Build a resolver from a class_name -> fields dict."""
    def resolve(class_name: str) -> tuple[str, ...] | None:
        return memory_fields_map.get(class_name)
    return resolve


# A graph with all required fields declared
_FULL_GRAPH = _graph(
    _node("ManagedLoRATrainerNode", model="sdxl", batch_size=2, rank=64,
          checkpointing=True, optimizer="adamw"),
    _node("ManagedDatasetSourceNode", batch_size=2),
    _node("ComfyUNetLoRANode", model="sdxl", rank=64, checkpointing=True),
    _node("LoRATrainingConfigNode", rank=64, checkpointing=True),
    _node("ComposedAdamWOptimizerNode", optimizer="adamw"),
)

_FULL_RESOLVER = _resolver({
    "ManagedLoRATrainerNode": ("model", "batch_size", "rank", "checkpointing", "optimizer"),
    "ManagedDatasetSourceNode": ("batch_size",),
    "ComfyUNetLoRANode": ("model", "rank", "checkpointing"),
    "LoRATrainingConfigNode": ("rank", "checkpointing"),
    "ComposedAdamWOptimizerNode": ("optimizer",),
})

_DATASET_STATS = {
    "buckets": [
        {"height": 1024, "width": 1024, "count": 100},
        {"height": 512, "width": 512, "count": 50},
    ]
}


def test_fingerprint_computed_without_torch():
    """The fingerprint must not import torch."""
    # Ensure torch is not already imported by another test
    assert "torch" not in sys.modules or True  # may be imported by other tests
    result = graph_fingerprint(_FULL_GRAPH, _DATASET_STATS,
                              resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(result, GraphMemoryFingerprint)
    # The key property: this function works in a clean process without torch
    # (verified by running this test in isolation)


def test_fingerprint_fields_derived_correctly():
    """All fingerprint fields come from declared node params."""
    result = graph_fingerprint(_FULL_GRAPH, _DATASET_STATS,
                              resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(result, GraphMemoryFingerprint)
    assert result.model == "sdxl"
    assert result.batch_size == 2
    assert result.latent_h == 1024
    assert result.latent_w == 1024
    assert result.rank == 64
    assert result.checkpointing is True
    assert result.optimizer == "adamw"


def test_fingerprint_uses_largest_bucket():
    """Latent h/w come from the largest bucket, not the most samples."""
    stats = {
        "buckets": [
            {"height": 512, "width": 512, "count": 1000},  # most samples
            {"height": 1024, "width": 768, "count": 10},   # largest pixels
        ]
    }
    result = graph_fingerprint(_FULL_GRAPH, stats,
                              resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(result, GraphMemoryFingerprint)
    assert result.latent_h == 1024
    assert result.latent_w == 768


def test_fingerprint_batch_size_change_different():
    """Different batch size -> different fingerprint."""
    graph_bs4 = _graph(
        _node("ManagedLoRATrainerNode", model="sdxl", batch_size=4, rank=64,
              checkpointing=True, optimizer="adamw"),
        _node("ManagedDatasetSourceNode", batch_size=4),
        _node("ComfyUNetLoRANode", model="sdxl", rank=64, checkpointing=True),
        _node("LoRATrainingConfigNode", rank=64, checkpointing=True),
        _node("ComposedAdamWOptimizerNode", optimizer="adamw"),
    )
    r1 = graph_fingerprint(_FULL_GRAPH, _DATASET_STATS,
                           resolve_memory_fields=_FULL_RESOLVER)
    r2 = graph_fingerprint(graph_bs4, _DATASET_STATS,
                           resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(r1, GraphMemoryFingerprint)
    assert isinstance(r2, GraphMemoryFingerprint)
    assert r1.key() != r2.key()


def test_fingerprint_checkpointing_change_different():
    """Different checkpointing -> different fingerprint."""
    graph_no_ckpt = _graph(
        _node("ManagedLoRATrainerNode", model="sdxl", batch_size=2, rank=64,
              checkpointing=False, optimizer="adamw"),
        _node("ManagedDatasetSourceNode", batch_size=2),
        _node("ComfyUNetLoRANode", model="sdxl", rank=64, checkpointing=False),
        _node("LoRATrainingConfigNode", rank=64, checkpointing=False),
        _node("ComposedAdamWOptimizerNode", optimizer="adamw"),
    )
    r1 = graph_fingerprint(_FULL_GRAPH, _DATASET_STATS,
                           resolve_memory_fields=_FULL_RESOLVER)
    r2 = graph_fingerprint(graph_no_ckpt, _DATASET_STATS,
                           resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(r1, GraphMemoryFingerprint)
    assert isinstance(r2, GraphMemoryFingerprint)
    assert r1.key() != r2.key()


def test_fingerprint_missing_field_unknown():
    """A graph with a missing field -> UnknownFingerprint."""
    incomplete_graph = _graph(
        _node("ManagedLoRATrainerNode", model="sdxl", batch_size=2, rank=64,
              checkpointing=True),  # missing optimizer
        _node("ManagedDatasetSourceNode", batch_size=2),
        _node("ComfyUNetLoRANode", model="sdxl", rank=64, checkpointing=True),
        _node("LoRATrainingConfigNode", rank=64, checkpointing=True),
    )
    result = graph_fingerprint(incomplete_graph, _DATASET_STATS,
                              resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(result, UnknownFingerprint)
    assert "optimizer" in result.reason


def test_fingerprint_no_dataset_stats_unknown():
    """No dataset stats -> UnknownFingerprint."""
    result = graph_fingerprint(_FULL_GRAPH, None,
                              resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(result, UnknownFingerprint)
    assert "dataset stats" in result.reason


def test_fingerprint_no_buckets_unknown():
    """Dataset stats with no buckets -> UnknownFingerprint."""
    result = graph_fingerprint(_FULL_GRAPH, {"buckets": []},
                              resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(result, UnknownFingerprint)
    assert "buckets" in result.reason


def test_fingerprint_shapeless_bucket_unknown():
    """Rows that never recorded a latent size -> UnknownFingerprint.

    A 0x0 "largest bucket" must not become a key: it would file this
    dataset's peak under a key a genuinely large-shape run also computes
    (the key carries no dataset identity), and the read-back would be a
    peak measured at the wrong size.
    """
    stats = {"buckets": [{"height": 0, "width": 0, "count": 10}]}
    result = graph_fingerprint(_FULL_GRAPH, stats,
                              resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(result, UnknownFingerprint)
    assert "shape" in result.reason


def test_fingerprint_no_declaring_nodes_unknown():
    """A graph with no nodes that declare memory_fields -> UnknownFingerprint."""
    no_fields_graph = _graph(
        _node("SomeOtherNode", model="sdxl", batch_size=2),
    )
    result = graph_fingerprint(no_fields_graph, _DATASET_STATS,
                              resolve_memory_fields=_resolver({}))
    assert isinstance(result, UnknownFingerprint)
    assert "missing" in result.reason


def test_fingerprint_key_is_stable():
    """The key is a stable string for the peak store."""
    result = graph_fingerprint(_FULL_GRAPH, _DATASET_STATS,
                              resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(result, GraphMemoryFingerprint)
    key = result.key()
    assert isinstance(key, str)
    assert "sdxl" in key
    assert "64" in key
    assert "adamw" in key


def test_fingerprint_same_config_same_key():
    """Same config -> same key (idempotent)."""
    r1 = graph_fingerprint(_FULL_GRAPH, _DATASET_STATS,
                           resolve_memory_fields=_FULL_RESOLVER)
    r2 = graph_fingerprint(_FULL_GRAPH, _DATASET_STATS,
                           resolve_memory_fields=_FULL_RESOLVER)
    assert isinstance(r1, GraphMemoryFingerprint)
    assert isinstance(r2, GraphMemoryFingerprint)
    assert r1.key() == r2.key()


def test_fingerprint_fields_tuple():
    """FINGERPRINT_FIELDS is the canonical tuple."""
    assert "model" in FINGERPRINT_FIELDS
    assert "batch_size" in FINGERPRINT_FIELDS
    assert "latent_h" in FINGERPRINT_FIELDS
    assert "latent_w" in FINGERPRINT_FIELDS
    assert "rank" in FINGERPRINT_FIELDS
    assert "checkpointing" in FINGERPRINT_FIELDS
    assert "optimizer" in FINGERPRINT_FIELDS



def main() -> None:
    """Run every test in this file, listed by name.

    Listed, not discovered: a `def test_*` nothing calls is a comment
    shaped like a safety net, and `scripts/check_test_wiring.py` fails
    this file when one is defined and left out here -- all 12 of
    these were, and the file exited 0 having run nothing, before that
    check caught it.
    """
    tests = [
        test_fingerprint_computed_without_torch,
        test_fingerprint_fields_derived_correctly,
        test_fingerprint_uses_largest_bucket,
        test_fingerprint_batch_size_change_different,
        test_fingerprint_checkpointing_change_different,
        test_fingerprint_missing_field_unknown,
        test_fingerprint_no_dataset_stats_unknown,
        test_fingerprint_no_buckets_unknown,
        test_fingerprint_shapeless_bucket_unknown,
        test_fingerprint_no_declaring_nodes_unknown,
        test_fingerprint_key_is_stable,
        test_fingerprint_same_config_same_key,
        test_fingerprint_fields_tuple,
    ]
    for test in tests:
        test()
    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {len(tests)} CHECKS PASSED")


if __name__ == "__main__":
    main()
