#!/usr/bin/env python3
"""lora_forensics.py -- offline inspection and ablation of a kohya-style SDXL LoRA.

No GPU, no training, no ComfyUI needed. Reads the LoRA (and optionally the base
checkpoint, lazily) and answers three questions:

  1. WHICH modules does this LoRA touch?  (attention vs. time_embed / label_emb / other)
  2. HOW STRONG is each module's change relative to the weight it modifies?
        ratio = ||scale * up @ down||_F / ||W_base||_F      scale = alpha / rank
     A module whose ratio is far above the attention modules' median is where the
     LoRA is "shouting".
  3. WHAT happens if a group is removed?  --write-without writes a copy of the LoRA
     with those groups deleted, so you can test it in ComfyUI immediately.

Usage
-----
  # what is in it, and how strong (add --base for the ratios):
  python lora_forensics.py bad.safetensors --base sdxl_checkpoint.safetensors

  # same, side by side with a LoRA you know is good (from another trainer):
  python lora_forensics.py bad.safetensors --base ckpt.safetensors --compare good.safetensors

  # ABLATION: delete the conditioning-path modules, write a new file, load it in ComfyUI
  python lora_forensics.py bad.safetensors --write-without time_embed,label_emb bad_no_cond.safetensors

  # the opposite ablation: keep ONLY those modules
  python lora_forensics.py bad.safetensors --write-only time_embed,label_emb bad_only_cond.safetensors

Groups: time_embed, label_emb, attn_input, attn_middle, attn_output, other_unet, text_encoder, unknown
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys

import numpy as np

try:
    import torch  # noqa: F401
    FRAMEWORK = "pt"
except ImportError:  # numpy fallback: fine for fp32/fp16 files, not for bf16
    FRAMEWORK = "numpy"

from safetensors import safe_open

DOWN, UP, ALPHA = ".lora_down.weight", ".lora_up.weight", ".alpha"
GROUPS = ["time_embed", "label_emb", "attn_input", "attn_middle", "attn_output",
          "other_unet", "text_encoder", "unknown"]


def to_np(t) -> np.ndarray:
    if FRAMEWORK == "pt":
        return t.detach().float().cpu().numpy().astype(np.float64)
    return np.asarray(t, dtype=np.float64)


def group_of(module_key: str) -> str:
    """module_key is the key without the .lora_down.weight / .lora_up.weight / .alpha suffix."""
    if module_key.startswith(("lora_te", "lora_te1", "lora_te2")):
        return "text_encoder"
    if not module_key.startswith("lora_unet_"):
        return "unknown"
    p = module_key[len("lora_unet_"):]
    if p.startswith("time_embed"):
        return "time_embed"
    if p.startswith("label_emb"):
        return "label_emb"
    if "attn" in p or "transformer_blocks" in p:
        if p.startswith("input_blocks"):
            return "attn_input"
        if p.startswith("middle_block"):
            return "attn_middle"
        if p.startswith("output_blocks"):
            return "attn_output"
    return "other_unet"


def read_lora(path: str) -> dict[str, dict]:
    """module_key -> {down, up, alpha}"""
    mods: dict[str, dict] = {}
    with safe_open(path, framework=FRAMEWORK) as f:
        for key in f.keys():
            for suffix, slot in ((DOWN, "down"), (UP, "up"), (ALPHA, "alpha")):
                if key.endswith(suffix):
                    mods.setdefault(key[: -len(suffix)], {})[slot] = f.get_tensor(key)
    return mods


def delta_fro(mod: dict) -> tuple[float, int, float]:
    """(||scale * up@down||_F, rank, scale) without materialising the full matrix."""
    down, up = to_np(mod["down"]), to_np(mod["up"])
    rank = down.shape[0]
    alpha = float(to_np(mod["alpha"]).reshape(-1)[0]) if "alpha" in mod else float(rank)
    scale = alpha / rank
    # ||U D||_F^2 = trace((U^T U)(D D^T))
    fro2 = float(np.sum((up.T @ up) * (down @ down.T)))
    return scale * np.sqrt(max(fro2, 0.0)), rank, scale


def base_weight_norms(ckpt: str, module_keys: list[str]) -> dict[str, float]:
    """Frobenius norm of the base weight each LoRA module modifies (lazy reads)."""
    out: dict[str, float] = {}
    with safe_open(ckpt, framework=FRAMEWORK) as f:
        lookup = {}
        for k in f.keys():
            if k.startswith("model.diffusion_model.") and k.endswith(".weight"):
                path = k[len("model.diffusion_model."):-len(".weight")]
                lookup["lora_unet_" + path.replace(".", "_")] = k
        for mk in module_keys:
            ck = lookup.get(mk)
            if ck is not None:
                w = to_np(f.get_tensor(ck))
                out[mk] = float(np.sqrt(np.sum(w * w)))
    return out


def analyse(path: str, base: str | None) -> dict[str, list[dict]]:
    mods = read_lora(path)
    norms = base_weight_norms(base, list(mods)) if base else {}
    by_group: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for mk, m in mods.items():
        if "down" not in m or "up" not in m:
            continue
        dfro, rank, scale = delta_fro(m)
        wn = norms.get(mk)
        by_group[group_of(mk)].append({
            "key": mk, "rank": rank, "scale": scale, "delta": dfro,
            "ratio": (dfro / wn) if wn else None,
        })
    return by_group


def fmt(x, nd=4):
    return "   n/a " if x is None else f"{x:.{nd}f}"


def report(title: str, by_group: dict[str, list[dict]], has_base: bool) -> None:
    print(f"\n=== {title} ===")
    print(f"{'group':<14}{'modules':>8}{'rank':>6}{'scale':>8}{'median |dW|':>13}{'max |dW|':>11}"
          + (f"{'median ratio':>14}{'max ratio':>11}" if has_base else ""))
    for g in GROUPS:
        rows = by_group[g]
        if not rows:
            continue
        d = [r["delta"] for r in rows]
        line = (f"{g:<14}{len(rows):>8}{rows[0]['rank']:>6}{rows[0]['scale']:>8.3f}"
                f"{statistics.median(d):>13.4f}{max(d):>11.4f}")
        if has_base:
            rr = [r["ratio"] for r in rows if r["ratio"] is not None]
            line += (f"{fmt(statistics.median(rr)):>14}{fmt(max(rr)):>11}" if rr else f"{'n/a':>14}{'n/a':>11}")
        print(line)

    if has_base:
        att = [r["ratio"] for g in ("attn_input", "attn_middle", "attn_output")
               for r in by_group[g] if r["ratio"] is not None]
        if att:
            med = statistics.median(att)
            print(f"\nattention median ratio = {med:.4f}.  Modules > 3x that:")
            flagged = sorted((r for g in GROUPS for r in by_group[g]
                              if r["ratio"] is not None and r["ratio"] > 3 * med),
                             key=lambda r: -r["ratio"])
            for r in flagged[:15]:
                print(f"   {r['ratio']:.4f}  ({r['ratio']/med:5.1f}x)  {r['key']}")
            if not flagged:
                print("   (none)")


def compare(a: dict, b: dict, name_a: str, name_b: str) -> None:
    print(f"\n=== key coverage: {name_a} vs {name_b} ===")
    for g in GROUPS:
        na, nb = len(a[g]), len(b[g])
        if na or nb:
            note = "  <-- only in " + (name_a if nb == 0 else name_b) if (na == 0) != (nb == 0) else ""
            print(f"{g:<14}{name_a}: {na:>5}   {name_b}: {nb:>5}{note}")


def write_filtered(src: str, dst: str, groups: set[str], keep: bool) -> None:
    if FRAMEWORK == "pt":
        from safetensors.torch import load_file, save_file
    else:
        from safetensors.numpy import load_file, save_file
    tensors = load_file(src)
    out, dropped = {}, 0
    for k, v in tensors.items():
        mk = re.sub(r"(\.lora_down\.weight|\.lora_up\.weight|\.alpha)$", "", k)
        in_set = group_of(mk) in groups
        if in_set == keep:
            out[k] = v
        else:
            dropped += 1
    save_file(out, dst)
    print(f"\nwrote {dst}: kept {len(out)} tensors, dropped {dropped}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lora")
    ap.add_argument("--base", help="SDXL checkpoint .safetensors (enables strength ratios)")
    ap.add_argument("--compare", help="a known-good LoRA to compare against")
    ap.add_argument("--write-without", nargs=2, metavar=("GROUPS", "OUT"),
                    help="comma-separated groups to DELETE, output path")
    ap.add_argument("--write-only", nargs=2, metavar=("GROUPS", "OUT"),
                    help="comma-separated groups to KEEP (everything else deleted), output path")
    args = ap.parse_args()

    mine = analyse(args.lora, args.base)
    report(args.lora, mine, bool(args.base))
    if args.compare:
        other = analyse(args.compare, args.base)
        report(args.compare, other, bool(args.base))
        compare(mine, other, "lora", "compare")

    for spec, keep in ((args.write_without, False), (args.write_only, True)):
        if spec:
            groups = {g.strip() for g in spec[0].split(",")}
            bad = groups - set(GROUPS)
            if bad:
                print(f"unknown group(s): {sorted(bad)}; valid: {GROUPS}", file=sys.stderr)
                return 2
            write_filtered(args.lora, spec[1], groups, keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
