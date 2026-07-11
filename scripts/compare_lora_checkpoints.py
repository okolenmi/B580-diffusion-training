"""
Directly compares two LoRA checkpoint .safetensors files and reports how much
weights actually moved. This bypasses the training loop's dW metric entirely,
so it's a good independent sanity check.

Usage:
    python compare_lora_checkpoints.py OLDER.safetensors NEWER.safetensors
    python compare_lora_checkpoints.py OLDER.safetensors NEWER.safetensors --verbose

Default output is a compact report: one line per architectural block (in true
block order, e.g. input_blocks_0, input_blocks_1, ..., middle_block,
output_blocks_0, ...) with aggregate movement stats, plus an overall summary
and a short list of any fully-frozen tensors. Pass --verbose for the full
one-line-per-tensor dump (1000+ lines on a typical SDXL LoRA run) instead.

How to get two checkpoints to compare:
  - Your training run periodically overwrites a single "resume" checkpoint
    (save_every, default 100 steps) at <lora_output>.resume.safetensors.
    Copy it aside now (e.g. `cp foo.resume.safetensors snapshot_A.safetensors`),
    let training run for another few hundred/thousand steps, copy it aside
    again (`snapshot_B.safetensors`), then run this script on the two copies.
  - Or just compare your current resume checkpoint against the final
    lora_output from a previous run, if you have one.

What "real movement" looks like: lora_up (the up-projection) starts at
exactly zero, so it should show unambiguous nonzero movement almost
immediately. lora_down only starts moving once its paired lora_up has moved
away from zero (expected LoRA behavior, not a bug), so it can lag behind
early in training but should catch up.
"""
import re
import sys
import statistics
from safetensors.torch import load_file


def natural_sort_key(s: str):
    """Split into text/number chunks so 'blocks_2' sorts before 'blocks_10'
    (plain alphabetical sort does not: '10' < '2' as strings)."""
    return [int(chunk) if chunk.isdigit() else chunk
            for chunk in re.split(r"(\d+)", s)]


def block_group(key: str) -> str:
    """Collapse a full tensor key down to its architectural block, e.g.
    'lora_unet_output_blocks_5_1_transformer_blocks_9_attn2_to_q.lora_down.weight'
    -> 'output_blocks_5'. Falls back to the first 2 underscore-segments for
    anything that doesn't match the usual input/middle/output block pattern
    (e.g. time_embed, label_emb)."""
    m = re.search(r"(input_blocks_\d+|middle_block|output_blocks_\d+)", key)
    if m:
        return m.group(1)
    parts = key.replace("lora_unet_", "").split("_")
    return "_".join(parts[:2])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    verbose = "--verbose" in sys.argv

    if len(args) != 2:
        print(__doc__)
        sys.exit(1)

    path_a, path_b = args
    a = load_file(path_a)
    b = load_file(path_b)

    common_keys = set(a.keys()) & set(b.keys())
    lora_keys = sorted(
        (k for k in common_keys if "lora_down" in k or "lora_up" in k),
        key=natural_sort_key,
    )

    missing_from_b = sorted(k for k in a.keys() if ("lora_down" in k or "lora_up" in k) and k not in b)
    missing_from_a = sorted(k for k in b.keys() if ("lora_down" in k or "lora_up" in k) and k not in a)

    if not lora_keys:
        print("No lora_down/lora_up keys found in common between the two files.")
        print(f"Keys in A ({path_a}): {list(a.keys())[:10]}")
        print(f"Keys in B ({path_b}): {list(b.keys())[:10]}")
        sys.exit(1)

    if missing_from_a or missing_from_b:
        print("WARNING: the two files don't have identical key sets -- this on its own")
        print("would explain 'some blocks look missing' if you were diffing raw output.")
        if missing_from_b:
            print(f"  {len(missing_from_b)} keys in A but not B, e.g.: {missing_from_b[:3]}")
        if missing_from_a:
            print(f"  {len(missing_from_a)} keys in B but not A, e.g.: {missing_from_a[:3]}")
        print()

    per_tensor = []  # (key, mean_abs_a, mean_abs_b, mean_abs_diff, max_abs_diff, moved)
    for k in lora_keys:
        ta = a[k].float()
        tb = b[k].float()
        if ta.shape != tb.shape:
            print(f"SHAPE MISMATCH on {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        diff = (tb - ta).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        per_tensor.append((k, ta.abs().mean().item(), tb.abs().mean().item(),
                            mean_diff, max_diff, mean_diff > 1e-8))

    if verbose:
        print(f"{'tensor':<70} {'mean|A|':>10} {'mean|B|':>10} {'mean|B-A|':>12} {'max|B-A|':>12} {'moved?':>7}")
        print("-" * 125)
        for k, ma, mb, md, xd, moved in per_tensor:
            print(f"{k:<70} {ma:10.6f} {mb:10.6f} {md:12.8f} {xd:12.8f} {'YES' if moved else 'NO':>7}")
        print()

    # --- Aggregate by architectural block, in true numeric block order ---
    groups: dict[str, list] = {}
    for row in per_tensor:
        groups.setdefault(block_group(row[0]), []).append(row)

    print(f"Comparing {len(per_tensor)} LoRA tensors between:")
    print(f"  A (older): {path_a}")
    print(f"  B (newer): {path_b}")
    print()
    print(f"{'block':<20} {'tensors':>8} {'moved':>6} {'frozen':>7} {'mean|B-A|':>12} {'max|B-A|':>12}")
    print("-" * 70)
    for group_name in sorted(groups.keys(), key=natural_sort_key):
        rows = groups[group_name]
        deltas = [r[3] for r in rows]
        maxes = [r[4] for r in rows]
        n_moved = sum(r[5] for r in rows)
        print(f"{group_name:<20} {len(rows):>8} {n_moved:>6} {len(rows) - n_moved:>7} "
              f"{statistics.mean(deltas):>12.8f} {max(maxes):>12.8f}")

    # --- Overall summary ---
    all_deltas = [r[3] for r in per_tensor]
    n_moved_total = sum(r[5] for r in per_tensor)
    n_frozen_total = len(per_tensor) - n_moved_total
    print()
    print(f"Overall: {len(per_tensor)} tensors | {n_moved_total} moved | {n_frozen_total} frozen")
    print(f"  mean|B-A| across all tensors   : {statistics.mean(all_deltas):.8f}")
    print(f"  median|B-A| across all tensors : {statistics.median(all_deltas):.8f}")
    print(f"  max|B-A| across all tensors    : {max(r[4] for r in per_tensor):.8f}")

    if n_frozen_total:
        print()
        print(f"Frozen tensors ({n_frozen_total}, bit-identical between the two checkpoints):")
        for k, *_rest, moved in per_tensor:
            if not moved:
                print(f"  {k}")

    print()
    if n_frozen_total == len(per_tensor):
        print("=> Every tracked tensor is frozen. Something is still wrong upstream")
        print("   of the checkpoints themselves (gradients, optimizer wiring, or the")
        print("   two files you gave this script are actually identical/mixed up).")
    elif n_frozen_total > 0:
        print("=> Most tensors moved. Frozen ones are very likely lora_down matrices")
        print("   whose paired lora_up hasn't moved away from zero yet (expected early")
        print("   -training behavior) -- worth re-checking only if the SAME tensors")
        print("   stay frozen across multiple comparisons over time.")
    else:
        print("=> Every tracked tensor moved. Training is genuinely updating weights.")


if __name__ == "__main__":
    main()
