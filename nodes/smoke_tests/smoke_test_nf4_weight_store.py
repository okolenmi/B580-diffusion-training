"""Correctness check for nodes/model/nf4_weight_store.py's NF4WeightStore.

Cross-checks the codebook against bitsandbytes' own real, published
constants (hardcoded below, taken directly from a real bitsandbytes
build -- not recomputed from this module's own code, which would prove
nothing). Reconstruction-error and compression-ratio checks use
realistic weight sizes (matching an actual SDXL cross-attention
projection's dimensions) since this class's real, block-padding-based
double quantization only pays off at that kind of scale -- see the
module's own docstring.

Run this directly: `python nodes/smoke_tests/smoke_test_nf4_weight_store.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.model.frozen_weight_store import FrozenWeightStore
from nodes.model.nf4_weight_store import (
    NF4WeightStore, _linear_dequantize_u8, _linear_quantize_u8, _nf4_codebook,
    _pack_nibbles, _quantize_blockwise_nearest, _unpack_nibbles,
)

failures = []


def record(ok: bool, name: str, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


# Taken directly from a real bitsandbytes build's NF4 codebook (16 sorted
# values) -- not derived from this project's own reproduction, so
# comparing against it is a real cross-check, not circular.
_BNB_REFERENCE_CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


def check_codebook_matches_bitsandbytes_real_constants():
    print("\n=== NF4 codebook matches bitsandbytes' own real, published values ===")
    codebook = _nf4_codebook(torch.device("cpu"), torch.float32)
    record(codebook.numel() == 16, "16 values", detail=str(codebook.numel()))
    ref = torch.tensor(_BNB_REFERENCE_CODEBOOK)
    max_diff = (codebook - ref).abs().max().item()
    record(max_diff < 1e-5,
           "matches bitsandbytes' real codebook to float32-rounding precision "
           "(~1e-7 expected, not a real discrepancy)", detail=f"max diff={max_diff}")
    record(codebook[7].item() == 0.0, "the exact-zero level is really exactly zero",
           detail=str(codebook[7].item()))
    record(torch.equal(codebook, codebook.sort().values), "codebook is sorted ascending")


def check_pack_unpack_nibbles_round_trip():
    print("\n=== nibble packing round-trips exactly ===")
    torch.manual_seed(0)
    indices = torch.randint(0, 16, (256,), dtype=torch.uint8)
    packed = _pack_nibbles(indices)
    record(packed.numel() == 128, "128 bytes for 256 4-bit indices (exactly 2x compression)",
           detail=str(packed.numel()))
    unpacked = _unpack_nibbles(packed, 256)
    record(torch.equal(unpacked, indices), "unpack(pack(x)) == x exactly")


def check_linear_u8_round_trip_error_is_small():
    print("\n=== linear 8-bit (de)quantization -- double-quant's own scheme -- "
          "has small, bounded error ===")
    torch.manual_seed(1)
    x = torch.randn(300) * 5
    q, lo, scale = _linear_quantize_u8(x)
    recon = _linear_dequantize_u8(q, lo, scale)
    max_err = (recon - x).abs().max().item()
    # 256 levels over the real [min,max] range -- worst case is half a
    # quantization step, i.e. (max-min)/255/2.
    expected_bound = float((x.max() - x.min()) / 255 / 2) + 1e-4
    record(max_err <= expected_bound,
           "every reconstructed value is within half a quantization step",
           detail=f"max_err={max_err} bound={expected_bound}")


def check_identity_shape_and_dtype():
    print("\n=== materialize() returns the right shape/dtype ===")
    torch.manual_seed(2)
    w = torch.randn(128, 96) * 0.02
    store = NF4WeightStore(w)
    recon = store.materialize()
    record(recon.shape == w.shape, "shape matches", detail=f"{recon.shape} vs {w.shape}")
    record(recon.dtype == torch.bfloat16, "defaults to bf16 for an fp32 source",
           detail=str(recon.dtype))

    w16 = w.to(torch.float16)
    store16 = NF4WeightStore(w16)
    record(store16.materialize().dtype == torch.float16,
           "keeps float16 if the source was float16 (doesn't force bf16 always)")


def check_reconstruction_error_beats_naive_uniform_4bit():
    print("\n=== NF4's Gaussian-quantile codebook reconstructs real (near-Gaussian) "
          "weight-like data more accurately than naive uniform 4-bit ===")
    torch.manual_seed(3)
    w = torch.randn(1280, 1280) * 0.02  # realistic SDXL cross-attn projection size
    store = NF4WeightStore(w)
    recon_nf4 = store.materialize().float()
    rmse_nf4 = (recon_nf4 - w).pow(2).mean().sqrt().item()

    flat = w.reshape(-1)
    blocks = flat.view(-1, 64)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    levels = torch.linspace(-1, 1, 16)
    idx = (blocks.unsqueeze(-1) / absmax.unsqueeze(-1) - levels.view(1, 1, -1)).abs().argmin(dim=-1)
    recon_naive = (levels[idx] * absmax).view(-1)
    rmse_naive = (recon_naive - flat).pow(2).mean().sqrt().item()

    record(rmse_nf4 < rmse_naive,
           "NF4's codebook gives lower RMSE than a naive uniform 4-bit codebook on "
           "Gaussian-distributed weight-like data", detail=f"nf4={rmse_nf4} naive={rmse_naive}")
    relative_rmse = rmse_nf4 / w.std().item()
    record(relative_rmse < 0.15,
           "relative RMSE (vs. the weight's own std) is in the expected range for "
           "4-bit quantization of near-Gaussian data", detail=f"relative_rmse={relative_rmse}")


def check_compression_ratio_and_double_quant_savings_at_realistic_scale():
    print("\n=== compression ratio and double-quantization's real savings, at a "
          "realistic weight size (padding overhead dominates for tiny tensors -- "
          "see this module's own docstring) ===")
    torch.manual_seed(4)
    w = torch.randn(1280, 1280) * 0.02
    store = NF4WeightStore(w)
    bf16_bytes = w.numel() * 2
    nf4_bytes = store.footprint_bytes()
    ratio = bf16_bytes / nf4_bytes
    record(3.5 < ratio < 4.0,
           "real compression ratio vs bf16 is close to the theoretical 4x "
           "(absmax overhead keeps it just under)", detail=f"ratio={ratio:.3f}x")

    num_first_level_blocks = w.numel() // 64
    codes_bytes = w.numel() // 2
    undoubled_absmax_bytes = num_first_level_blocks * 4  # one fp32 absmax per block
    undoubled_total = codes_bytes + undoubled_absmax_bytes
    record(nf4_bytes < undoubled_total,
           "double quantization is a real, measurable saving over storing absmax "
           "as plain fp32 at this scale", detail=f"doubled={nf4_bytes} undoubled={undoubled_total}")
    saved_bits_per_param = (undoubled_total - nf4_bytes) * 8 / w.numel()
    record(0.2 < saved_bits_per_param < 0.5,
           "savings are in the right ballpark vs. the QLoRA paper's own reported "
           "~0.37 bits/parameter for double quantization",
           detail=f"{saved_bits_per_param:.3f} bits/param")


def check_non_block_aligned_sizes_work():
    print("\n=== weight sizes that don't divide evenly into blocksize/double_quant_blocksize "
          "still round-trip correctly (padding/trimming bookkeeping) ===")
    torch.manual_seed(5)
    for numel in (1, 5, 63, 65, 127, 4097):
        w = torch.randn(numel) * 0.02
        store = NF4WeightStore(w, blocksize=64, double_quant_blocksize=256)
        recon = store.materialize()
        record(recon.shape == w.shape,
               f"numel={numel}: shape preserved through padding/trimming",
               detail=f"{recon.shape}")
        record(torch.isfinite(recon).all(),
               f"numel={numel}: no NaN/Inf introduced by padding blocks")


def check_footprint_bytes_matches_hand_computed_formula():
    print("\n=== footprint_bytes() matches a hand-computed formula exactly ===")
    torch.manual_seed(6)
    w = torch.randn(512, 512) * 0.02  # numel=262144, evenly divisible throughout
    store = NF4WeightStore(w, blocksize=64, double_quant_blocksize=256)
    numel = w.numel()
    num_blocks = numel // 64
    dq_num_blocks = -(-num_blocks // 256)
    expected = (
        numel // 2                    # packed 4-bit codes, 2 per byte
        + dq_num_blocks * 256          # absmax_q: padded to full dq blocks
        + dq_num_blocks * 4            # absmax_lo (fp32 per dq block)
        + dq_num_blocks * 4            # absmax_scale (fp32 per dq block)
        + 4                            # offset (one fp32 scalar)
    )
    record(store.footprint_bytes() == expected,
           "footprint_bytes() matches the documented formula exactly",
           detail=f"got={store.footprint_bytes()} expected={expected}")


def check_materialize_is_deterministic_across_calls():
    print("\n=== materialize() called twice gives exactly the same result "
          "(re-dequantizes correctly, doesn't drift) ===")
    torch.manual_seed(7)
    w = torch.randn(256, 128) * 0.02
    store = NF4WeightStore(w)
    a = store.materialize()
    b = store.materialize()
    record(torch.equal(a, b), "two materialize() calls return bit-identical tensors")


def check_conformance():
    print("\n=== NF4WeightStore conforms to FrozenWeightStore ===")
    store = NF4WeightStore(torch.randn(64, 64) * 0.02)
    record(isinstance(store, FrozenWeightStore), "isinstance FrozenWeightStore")


def check_rejects_odd_blocksize():
    print("\n=== rejects an odd blocksize (nibble-packing needs an even one) ===")
    try:
        NF4WeightStore(torch.randn(128) * 0.02, blocksize=63)
        record(False, "odd blocksize should raise ValueError")
    except ValueError:
        record(True, "odd blocksize raises ValueError")


def main():
    check_codebook_matches_bitsandbytes_real_constants()
    check_pack_unpack_nibbles_round_trip()
    check_linear_u8_round_trip_error_is_small()
    check_identity_shape_and_dtype()
    check_reconstruction_error_beats_naive_uniform_4bit()
    check_compression_ratio_and_double_quant_savings_at_realistic_scale()
    check_non_block_aligned_sizes_work()
    check_footprint_bytes_matches_hand_computed_formula()
    check_materialize_is_deterministic_across_calls()
    check_conformance()
    check_rejects_odd_blocksize()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
