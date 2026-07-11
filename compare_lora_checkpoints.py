"""
Directly compares two LoRA checkpoint .safetensors files and reports how much
each tensor actually moved. This bypasses the training loop's dW metric
entirely, so it's a good independent sanity check.

Usage:
    python compare_lora_checkpoints.py OLDER.safetensors NEWER.safetensors

How to get two checkpoints to compare:
  - Your training run periodically overwrites a single "resume" checkpoint
    (save_every, default 100 steps) at <lora_output>.resume.safetensors.
    Copy it aside now (e.g. `cp foo.resume.safetensors snapshot_A.safetensors`),
    let training run for another few hundred/thousand steps, copy it aside
    again (`snapshot_B.safetensors`), then run this script on the two copies.
  - Or just compare your current resume checkpoint against the final
    lora_output from a previous run, if you have one.

What "real movement" looks like: lora_B (the up-projection) starts at exactly
zero, so it should show unambiguous nonzero movement almost immediately.
lora_A only starts moving once lora_B has moved away from zero (this is
expected LoRA behavior, not a bug), so it may lag behind lora_B early in
training but should catch up.
"""
import sys
from safetensors.torch import load_file
import torch


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    a = load_file(path_a)
    b = load_file(path_b)

    common_keys = sorted(set(a.keys()) & set(b.keys()))
    lora_keys = [k for k in common_keys if "lora_down" in k or "lora_up" in k]

    if not lora_keys:
        print(f"No lora_down/lora_up keys found in common between the two files.")
        print(f"Keys in A ({path_a}): {list(a.keys())[:10]}")
        print(f"Keys in B ({path_b}): {list(b.keys())[:10]}")
        sys.exit(1)

    print(f"Comparing {len(lora_keys)} LoRA tensors between:")
    print(f"  A (older): {path_a}")
    print(f"  B (newer): {path_b}")
    print()
    print(f"{'tensor':<55} {'mean|A|':>12} {'mean|B|':>12} {'mean|B-A|':>12} {'max|B-A|':>12} {'moved?':>8}")
    print("-" * 115)

    n_moved = 0
    n_frozen = 0
    for k in lora_keys:
        ta = a[k].float()
        tb = b[k].float()
        if ta.shape != tb.shape:
            print(f"{k:<55} SHAPE MISMATCH: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        diff = (tb - ta).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        moved = mean_diff > 1e-8
        n_moved += moved
        n_frozen += not moved
        print(f"{k:<55} {ta.abs().mean().item():12.6f} {tb.abs().mean().item():12.6f} "
              f"{mean_diff:12.8f} {max_diff:12.8f} {'YES' if moved else 'NO':>8}")

    print()
    print(f"Summary: {n_moved} tensors moved, {n_frozen} tensors completely frozen "
          f"(bit-identical between the two checkpoints).")
    if n_frozen == len(lora_keys):
        print("=> Every tracked tensor is frozen. Something is still wrong upstream")
        print("   of the checkpoints themselves (gradients, optimizer wiring, or the")
        print("   two files you gave this script are actually identical/mixed up).")
    elif n_frozen > 0:
        print("=> Some tensors moved and some didn't. Frozen ones are very likely")
        print("   lora_A matrices whose paired lora_B hasn't moved away from zero yet")
        print("   (expected early-training behavior, not necessarily a bug) -- unless")
        print("   the same tensors stay frozen across multiple comparisons over time.")
    else:
        print("=> Every tracked tensor moved. Training is genuinely updating weights.")


if __name__ == "__main__":
    main()
