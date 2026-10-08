#!/usr/bin/env python3
"""How different is an empty caption from a real one, on the real towers?

Motivation, measured not assumed: every caption in every dataset in this
repo is the empty string (273/273 non-square, 204/204 test2, 201/201
"1024 aes" -- confirmed in the sqlite `trajectories` table, so it is what
was ingested, not a loader artefact). So the LoRA was trained with a ctx
that comes from tokenising "" and nothing else.

This measures the gap that creates. If the empty-caption ctx is
*identical* to some other caption's, training was fine and the empty
string is only a convention. If it is a distinct point far from every
real prompt, then the LoRA learned its behaviour in a conditioning
regime that inference never reproduces.

Cosine similarity on the ctx rows (mean over token positions, which is
what cross-attention sees) and on the pooled vector (which feeds `y` and
therefore label_emb, one of the two groups the forensics flagged).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def cos(a, b):
    import torch
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> int:
    import torch
    from nodes.core import ExecutionContext
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.text_encoder import SDXLTextEncoderNode

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "div_4.safetensors"
    ctx = ExecutionContext()
    weights = SafetensorsCheckpointNode(ctx).build(path=checkpoint)["weights"]
    enc = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]

    empty = ""
    real = [
        "a photo of a cat",
        "portrait of a woman, detailed, sharp focus",
        "a landscape painting with mountains and a lake",
        "1girl, solo, masterpiece, best quality",
        "an intricate mechanical dragon, highly detailed",
    ]

    with torch.no_grad():
        e_ctx, e_pooled = enc.encode_prompt_only(empty, batch_size=1)
        rows = []
        for p in real:
            r_ctx, r_pooled = enc.encode_prompt_only(p, batch_size=1)
            # ctx: per-token-position cosine, then the mean over positions.
            # Zero-padding rows are included on purpose: they are what
            # cross-attention attends over, so they are part of the input.
            a = e_ctx[0].float()
            b = r_ctx[0].float()
            per_tok = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            rows.append({
                "prompt": p,
                "ctx_cos_mean_over_tokens": round(float(per_tok.mean()), 4),
                "ctx_cos_min_token": round(float(per_tok.min()), 4),
                "pooled_cos": round(cos(e_pooled, r_pooled), 4),
                "ctx_l2": round(float((a - b).norm()), 2),
                "n_tokens_real": int(r_ctx.shape[1]),
            })
            print(f"{p[:44]:<46} ctx_cos={rows[-1]['ctx_cos_mean_over_tokens']:.4f} "
                  f"(min tok {rows[-1]['ctx_cos_min_token']:.4f})  "
                  f"pooled_cos={rows[-1]['pooled_cos']:.4f}  "
                  f"|dctx|={rows[-1]['ctx_l2']}")

        # Self-comparison baseline: how similar are two REAL prompts to each
        # other? Without this, "empty vs real = 0.7" means nothing --
        # real prompts may simply all be similar to one another.
        print("\nreal-vs-real baseline (same metric):")
        base = enc.encode_prompt_only(real[0], batch_size=1)[0][0].float()
        for p in real[1:]:
            c = enc.encode_prompt_only(p, batch_size=1)[0][0].float()
            pt = torch.nn.functional.cosine_similarity(base, c, dim=-1)
            print(f"{real[0][:20]:<22} vs {p[:30]:<32} ctx_cos={float(pt.mean()):.4f}")

        # And the pool vector, which is the part `y` carries into label_emb.
        print("\npooled cosine, empty vs real:",
              [r["pooled_cos"] for r in rows])

    out = REPO / "runs" / "a5" / "caption_gap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())