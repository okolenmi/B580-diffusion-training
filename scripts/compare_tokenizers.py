"""Compare the old tokenizer with next/ on a prompt corpus, using the real CLIP vocabulary.

    CLIP_TOKENIZER_DIR=/path/to/sd1_tokenizer python scripts/compare_tokenizers.py [prompts.txt]

Prints every prompt where the two disagree and exits non-zero if there is any. Delete once the swap is done.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from next.families.general.clip.tokenizer import CLIPTokenizer  # noqa: E402
from nodes.model import tokenizer as old  # noqa: E402

CORPUS = ["a photo of a cat", "", "1girl, solo, masterpiece, best quality", "(word) <|startoftext|> héllo 日本語 😀 it's 123",
          "&amp;lt; double escaped", "x" * 300, "cat " * 120, "photophotophotophotophotophoto " * 4,
          "a very detailed illustration of a castle on a hill at sunset, volumetric light, 8k"]


def main() -> int:
    prompts = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if len(sys.argv) > 1 else CORPUS
    new, ref = CLIPTokenizer(), old.SDXLTokenizer()
    bad = [p for p in prompts if new.tokenize(p) != ref.tokenize(p)]
    for p in bad:
        print("DIFFERS:", repr(p[:80]))
    print(f"{len(prompts) - len(bad)}/{len(prompts)} prompts identical")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
