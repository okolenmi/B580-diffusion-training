"""D1: captions longer than 77 tokens -- whose chunk rule does what.

Evidence classes: project behavior measured (own tokenizer + tiny CLIP
forward, CPU); kohya's rule read in its source
(sd-scripts/library/strategy_base.py: hard cut every 75 content tokens,
BOS prepended + trailing id appended per chunk, no word logic); ComfyUI's
rule read in its source (comfy/sd1_clip.py: 77-sections with BOS/EOS and
the same max_word_length word-break as this project).

Result this pins: the project packs like ComfyUI (whole words, BOS/EOS
per 77-section, sections rejoined along the sequence axis, pooled from
the first section) and like kohya on caps (<=75 content + BOS/EOS) and
first-section pooling -- differing from kohya exactly on word-boundary
keeping. A 150-content-token caption is 2 sections everywhere.
"""

import sys
import os
sys.path.insert(0, os.environ.get("REPO", "."))

import torch


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _tokenizer():
    from nodes.model.tokenizer import SDTokenizer
    return SDTokenizer(pad_with_end=True)  # CLIP-L layout; -G differs only in pad id


def _caption_149(tok):
    """149 content tokens: whole-word packing wastes the 5 slots a hard cut
    would use, so a 6-token word straddling 75 *cannot* fit exactly-150
    content into 2 sections (70 kept + 80 remaining > 75 capacity) -- 149
    with the straddler at 74 packs [74][75] exactly. That impossibility is
    itself the difference under test: kohya's hard cut fits 150 into 2
    sections by splitting the word, whole-word keeping does not."""
    ones = [w for w in (
        "a", "the", "red", "blue", "cat", "dog", "sun", "oak", "iron",
        "lake", "hill", "fox", "owl", "ash", "elm", "rye", "cod", "eel")
        if len(tok.encode_text(w)) == 1]
    check(len(ones) >= 8, "need one-token words from the vocab")
    six = None
    for w in ("supercalifragilistic", "antidisestablishmentarian",
              "pneumonoultramicroscopicsilicovolcanoconiosis",
              "hippopotomonstrosesquippedaliophobia"):
        if len(tok.encode_text(w)) == 6:
            six = w
            break
    check(six is not None, "need a 6-token word from the vocab")
    words = (ones * 10)[:74] + [six]
    n = sum(len(tok.encode_text(w)) for w in words)
    check(n == 80, f"straddler must start at content 74, starts at {n - 6}")
    i = 0
    while n < 149:
        w = ones[i % len(ones)]
        words.append(w)
        n += len(tok.encode_text(w))
        i += 1
    check(n == 149, f"caption must carry exactly 149 content tokens, got {n}")
    return " ".join(words), six


def kohya_chunks(content_ids, bos=49406, last=49407):
    """sd-scripts strategy_base.py, v2/SDXL branch: hard cut every 75
    content ids, BOS prepended + trailing id appended per chunk."""
    return [[bos] + content_ids[i:i + 75] + [last]
            for i in range(0, len(content_ids), 75)]


def check_project_sections_are_lossless():
    print("[149 content tokens -> project sections: 2x77, BOS/EOS, lossless]")
    tok = _tokenizer()
    text, six = _caption_149(tok)
    sections = tok.tokenize(text)
    check(len(sections) == 2, f"want 2 sections, got {len(sections)}")
    for s in sections:
        check(len(s) == 77, f"section length {len(s)}, want 77")
        check(s[0] == tok.start_token, "section must start with BOS")
    full = tok.encode_text(text)
    check(len(full) == 149, f"content must be 149 ids, got {len(full)}")
    # Strip BOS/padding/EOS-tail per section, rejoin: every content id
    # exactly once, in order -- chunking must never lose or duplicate.
    rejoined = []
    for s in sections:
        body = [t for t in s[1:] if t != tok.end_token and t != tok.pad_token]
        # A section's own EOS sits right after its content (padding after).
        rejoined.extend(body)
    check(rejoined == full,
          "sections do not rejoin to the full content id list")
    print("    PASS")
    return text, six, sections


def check_word_boundary_vs_kohya(text, six, sections):
    print("[project keeps the straddling word whole; kohya's hard cut splits it]")
    tok = _tokenizer()
    full = tok.encode_text(text)
    six_ids = tok.encode_text(six)
    pos = None
    for i in range(len(full) - 6 + 1):
        if full[i:i + 6] == six_ids:
            pos = i
            break
    check(pos is not None, "straddler not found in content")
    check(pos == 74, f"straddler must start at 74, starts at {pos}")
    sec1 = [t for t in sections[0][1:]
            if t != tok.end_token and t != tok.pad_token]
    check(six_ids not in [sec1[i:i + 6] for i in range(len(sec1))],
          "project split a 6-token word across sections (it keeps words "
          "under max_word_length whole)")
    sec2 = [t for t in sections[1][1:]
            if t != tok.end_token and t != tok.pad_token]
    check(sec2[:6] == six_ids,
          "the kept-whole word must open the second section")
    chunks = kohya_chunks(full)
    check(len(chunks) == 2 and len(chunks[0]) == 77 and len(chunks[1]) == 76,
          "kohya rule must give 77 + 76 (75- and 74-content + BOS/EOS)")
    check(chunks[0][1:76] == full[:75] and chunks[1][1:-1] == full[75:149],
          "kohya chunks must be hard cuts at 75")
    # The hard cut lands inside the word (starts at 74, 6 long).
    check(full[74] == six_ids[0] and full[75] == six_ids[1],
          "cut position must be inside the straddling word")
    print(f"    straddler at content 74: project keeps whole, "
          f"kohya cuts after 1 of 6")
    print("    PASS")


def check_pooled_comes_from_the_first_section():
    print("[pooled ignores later sections (measured on a tiny tower)]")
    from nodes.model.clip import SDClipModel
    cfg = {"hidden_size": 32, "intermediate_size": 64,
           "num_attention_heads": 4, "num_hidden_layers": 4,
           "max_position_embeddings": 77, "hidden_act": "quick_gelu",
           "eos_token_id": 2, "vocab_size": 512}
    torch.manual_seed(41)
    tower = SDClipModel(cfg, layer="hidden", layer_idx=-2,
                        layer_norm_hidden_state=False,
                        special_tokens={"start": 1, "end": 2, "pad": 2},
                        return_projected_pooled=True).eval()
    torch.manual_seed(43)
    sec1 = [1] + torch.randint(3, cfg["vocab_size"], (75,)).tolist() + [2]
    sec2a = [1] + torch.randint(3, cfg["vocab_size"], (75,)).tolist() + [2]
    sec2b = [1] + torch.randint(3, cfg["vocab_size"], (75,)).tolist() + [2]
    check(sec2a != sec2b, "second sections must differ for the test to mean")
    with torch.no_grad():
        _, pooled_a = tower.encode_token_ids([sec1, sec2a])
        _, pooled_b = tower.encode_token_ids([sec1, sec2b])
    check(torch.equal(pooled_a, pooled_b),
          "changing only the second section moved pooled -- it must come "
          "from the first alone")
    print("    PASS")


def main():
    text, six, sections = check_project_sections_are_lossless()
    check_word_boundary_vs_kohya(text, six, sections)
    check_pooled_comes_from_the_first_section()
    print()
    print("conventions: project == ComfyUI (77-sections, BOS/EOS, whole-word "
          "keep; read in comfy/sd1_clip.py) ~= kohya (75-cap + BOS/EOS, "
          "pooled-first; read in sd-scripts strategy_base.py) except "
          "word-boundary keeping, where kohya hard-cuts.")
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
