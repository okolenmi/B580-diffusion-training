"""Real torch. Verifies nodes/dataset/renoise.py's core claim: x0 is
recovered *exactly* from a batch built the same way manager/builder.py's
real ingestion path builds one, and after renoising, x0 is still
recoverable exactly from the new (x_t, target, t) -- i.e. renoising
changes the noise level, not the underlying image.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from core.noise_schedule import eps_to_vpred, eps_to_x0, get_alpha_sigma, vpred_to_x0
from nodes.dataset.handle import TrainingBatchSource
from nodes.dataset.renoise import RenoiseBatchSource, RenoiseBatchSourceNode


class _FakeSource(TrainingBatchSource):
    """Yields exactly the batches given to it, once. Enough surface for
    RenoiseBatchSource, which only ever iterates, calls len(), and
    forwards invalidate()."""

    def __init__(self, batches):
        self._batches = batches
        self.invalidated = False

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)

    def invalidate(self):
        self.invalidated = True


def _build_ingestion_style_batch(x0: torch.Tensor, t_val: int, model_type: str, seed: int):
    """Exactly manager/builder.py's real formula (x_t = x0 + sigma*eps,
    target = eps or eps_to_vpred(eps, ...)), not a paraphrase of it."""
    at, st = get_alpha_sigma(t_val)
    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn(x0.shape, generator=gen, dtype=torch.float32)
    x_t = x0 + st.item() * eps
    if model_type == "vpred":
        target = eps_to_vpred(eps, x_t, at, st)
    else:
        target = eps
    import json
    metadata = json.dumps({"model_type": model_type})
    return {
        "x_t": x_t, "target": target, "t": torch.tensor([t_val] * x0.shape[0]),
        "target_p": target.clone(), "target_n": target.clone(),
        "prompt": "a cat", "neg_prompt": "", "metadata": metadata,
    }


def check_contracts():
    print("[contracts]")
    assert not getattr(RenoiseBatchSourceNode, "__abstractmethods__", None)
    assert set(RenoiseBatchSourceNode.INPUTS) == {"batches", "t_low", "t_high", "t_mode", "seed"}
    assert set(RenoiseBatchSourceNode.OUTPUTS) == {"batches"}
    print("    PASS")


def check_exact_round_trip_eps():
    print("[eps model: x0 recovered exactly, before and after renoising]")
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, 8, 8)
    batch = _build_ingestion_style_batch(x0, t_val=500, model_type="eps", seed=42)

    source = RenoiseBatchSource(_FakeSource([batch]), t_low=1, t_high=999, seed=7)
    out = next(iter(source))

    # Recover x0 from the *original* batch, exactly as the decorator does internally.
    at0, st0 = get_alpha_sigma(batch["t"])
    x0_recovered_before = eps_to_x0(batch["target"], batch["x_t"],
                                     at0.view(-1, 1, 1, 1), st0.view(-1, 1, 1, 1))
    torch.testing.assert_close(x0_recovered_before, x0, atol=1e-5, rtol=1e-4)

    # Recover x0 from the *renoised* output -- must be the same image.
    at1, st1 = get_alpha_sigma(out["t"])
    x0_recovered_after = eps_to_x0(out["target"], out["x_t"],
                                    at1.view(-1, 1, 1, 1), st1.view(-1, 1, 1, 1))
    torch.testing.assert_close(x0_recovered_after, x0, atol=1e-5, rtol=1e-4)
    print("    PASS: x0 recovered exactly from the original batch")
    print("    PASS: x0 still recovers exactly after renoising -- same image, new noise level")

    assert out["t"].tolist() != batch["t"].tolist(), "renoising should not reuse the original timestep"
    torch.testing.assert_close(out["target_p"], out["target"])
    torch.testing.assert_close(out["target_n"], out["target"])
    assert out["prompt"] == "a cat" and out["neg_prompt"] == ""
    print("    PASS: timestep actually changed; target_p/target_n collapsed to match target; "
          "unrelated fields passed through untouched")


def check_exact_round_trip_vpred():
    print("[vpred model: same round-trip, through the vpred inversion instead]")
    torch.manual_seed(1)
    x0 = torch.randn(3, 4, 8, 8)
    batch = _build_ingestion_style_batch(x0, t_val=250, model_type="vpred", seed=99)

    source = RenoiseBatchSource(_FakeSource([batch]), seed=3)
    out = next(iter(source))

    at1, st1 = get_alpha_sigma(out["t"])
    x0_recovered = vpred_to_x0(out["target"], out["x_t"], at1.view(-1, 1, 1, 1), st1.view(-1, 1, 1, 1))
    torch.testing.assert_close(x0_recovered, x0, atol=1e-4, rtol=1e-4)
    print("    PASS: x0 recovered exactly through the vpred path")


def check_timestep_diversity_breaks_the_fixed_grid():
    print("[resampled timesteps actually span the range, not stuck on ingestion's fixed grid]")
    torch.manual_seed(2)
    x0 = torch.randn(1, 4, 4, 4)
    fixed_grid_t = 500  # every batch "ingested" at exactly this one value
    batches = [_build_ingestion_style_batch(x0, t_val=fixed_grid_t, model_type="eps", seed=i)
               for i in range(30)]
    source = RenoiseBatchSource(_FakeSource(batches), t_low=1, t_high=999, seed=123)
    seen_t = [int(out["t"][0]) for out in source]
    assert len(set(seen_t)) > 15, f"expected wide spread, got {sorted(set(seen_t))}"
    assert min(seen_t) < 300 and max(seen_t) > 700, "expected real coverage of the range, not a narrow cluster"
    print(f"    PASS: {len(set(seen_t))}/30 distinct timesteps sampled, "
          f"range [{min(seen_t)}, {max(seen_t)}] -- the fixed-grid regularity is gone")


def check_delegation():
    print("[__len__/invalidate() delegate to the wrapped source]")
    inner = _FakeSource([_build_ingestion_style_batch(torch.randn(1, 4, 4, 4), 500, "eps", 1)])
    source = RenoiseBatchSource(inner)
    assert len(source) == len(inner)
    source.invalidate()
    assert inner.invalidated
    print("    PASS")


def check_node_build():
    print("[RenoiseBatchSourceNode.build()]")
    inner = _FakeSource([_build_ingestion_style_batch(torch.randn(1, 4, 4, 4), 500, "eps", 1)])
    node = RenoiseBatchSourceNode()
    result = node.build(batches=inner, t_low=1, t_high=999, t_mode="uniform", seed=5)
    assert isinstance(result["batches"], RenoiseBatchSource)
    print("    PASS")


def main():
    check_contracts()
    check_exact_round_trip_eps()
    check_exact_round_trip_vpred()
    check_timestep_diversity_breaks_the_fixed_grid()
    check_delegation()
    check_node_build()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
