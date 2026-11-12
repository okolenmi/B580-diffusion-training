"""Write what a safetensors file's header says, without reading any weights: key names, dtypes, shapes, metadata, the
top-level groups, and what each family's detector makes of it. Used to check detection against real files (Anima, Qwen3,
VAE) without sending gigabytes.

    python scripts/dump_header.py FILE.safetensors [FILE2 ...] --out headers

Writes headers/<file stem>.json (a few hundred KB) and prints one summary line per file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from next.families.api.detection import Match  # noqa: E402
from next.families.api.header import read_header  # noqa: E402
from next.families.registry import matchers  # noqa: E402


def groups(index, depth: int) -> dict:
    """Tensor count, bytes and dtypes per key prefix of `depth` dotted parts."""
    out: dict = defaultdict(lambda: {"tensors": 0, "bytes": 0, "dtypes": set()})
    for key, info in index.entries.items():
        g = out[".".join(key.split(".")[:depth])]
        g["tensors"] += 1
        g["bytes"] += info.nbytes
        g["dtypes"].add(info.dtype)
    return {k: {**v, "dtypes": sorted(v["dtypes"])} for k, v in sorted(out.items())}


def verdicts(index) -> dict:
    out = {}
    for family, match in matchers().items():
        try:
            result = match(index)
        except Exception as exc:  # a detector refusing is an answer too
            out[family] = f"rejected: {exc}"
            continue
        out[family] = "no match" if result is None else (
            {"score": result.score, "warnings": list(result.warnings), "parts": [str(p) for p in result.parts]}
            if isinstance(result, Match) else str(result)
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="headers")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        index = read_header(name)
        doc = {
            "file": Path(name).name, "size": index.file_size, "tensors_total": len(index), "metadata": index.metadata,
            "groups_1": groups(index, 1), "groups_2": groups(index, 2), "detection": verdicts(index),
            "tensors": [[k, v.dtype, list(v.shape)] for k, v in sorted(index.entries.items())],
        }
        (out / f"{Path(name).stem}.json").write_text(json.dumps(doc, indent=0))
        best = max((v["score"] for v in doc["detection"].values() if isinstance(v, dict)), default=None)
        print(f"{Path(name).name}: {len(index)} tensors, {index.file_size / 2**30:.1f} GiB, best match score {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
