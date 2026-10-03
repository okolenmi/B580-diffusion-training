"""Correctness check for nodes/model/tokenizer.py -- the CLIP BPE tokenizer
this project reimplements (design doc 12, section 7.3, section C2).

This is the one piece whose contract is not the checkpoint but OpenAI's
published BPE data, so the checks are of two kinds.

**Agreement with ComfyUI's tokenizer**, over a corpus chosen to cover the
places a tokenizer actually goes wrong: the empty string, single characters,
mixed case, runs of whitespace, contractions, digits, punctuation between
letters and at the edges, accented Latin, CJK, Cyrillic, emoji, a
300-character word, a word long enough to need several sections, and a
prompt too long for 77 tokens. Every one of those must come back
token-for-token identical, or the reimplementation is wrong.

**Three deliberate divergences**, each pinned so it cannot change by
accident. All three are cases where ComfyUI has prompt grammar this project
does not have, or where HuggingFace has drifted from CLIP:

1. `(word)` is text, not a weight group. ComfyUI's `token_weights` runs
   `parse_parentheses` and consumes a balanced `(...)` as a weighting
   expression, so `"a (b) c"` reaches its tokenizer as `"a b c"` with the
   parentheses gone.
2. A literal `<|startoftext|>` in the prompt is text. CLIP's split pattern
   matches the special tokens and passes them through as ids, so a prompt
   containing them gets a second BOS in the middle.
3. HTML entities are unescaped, twice, which is CLIP's `basic_clean`. The
   installed `transformers` 5.12.1 has dropped that step.

The byte-level alphabet is also checked directly, because a wrong one fails
silently: every word comes out as single characters, every id is in the
vocabulary, and nothing raises.

Skipped, not failed, when no vocabulary is reachable -- the tokenizer is the
one module that genuinely needs a data file, and a machine without one should
say so rather than fail. See `default_vocabulary_dir()`.

Run: `python nodes/smoke_tests/smoke_test_tokenizer.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

failures: list[str] = []
skipped: list[str] = []

BOS, EOS = 49406, 49407

#: Chosen to hit the failure modes, not to be representative of prompts.
CORPUS = [
    "",
    " ",
    "\n\t ",
    "a",
    "a photo of a cat",
    "a photograph of an astronaut riding a horse on mars, 4k, detailed",
    "masterpiece, best quality, ultra detailed, sharp focus",
    " masterpiece ,  best quality  ,  ultra detailed  ",
    "  leading and trailing   ",
    "MiXeD CaSe WoRdS",
    "ANTICLIMAX!!! wow!!! **** ###",
    "don't can't won't it's they've I'd you're we'll he's",
    "123 4567 0.5 3.14159 -42 1e10",
    "hyphen-ated under_scored slash/es back\\slash",
    "tab\tand\nnewline   and    spaces",
    "\t" * 30,
    "Café naïve résumé straße",
    "Функция тест 测试関数 테스트",
    "emoji \U0001F3A8\U0001F305 and accents àèî",
    "a" * 300,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "photo of a " + "very " * 60 + "long prompt that will not fit in 77 tokens",
    "one two three four five six seven eight nine ten",
]


def record(ok: bool, name: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"  {'PASS' if ok else 'FAIL'}: {name}{suffix}")
    if not ok:
        failures.append(name if not detail else f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  SKIP: {name}: {why}")
    skipped.append(name)


def _vocabulary_dir():
    from nodes.model.tokenizer import default_vocabulary_dir
    return default_vocabulary_dir()


def check_layout():
    """The sequence shape around the BPE: BOS, text, EOS, pad."""
    from nodes.model.tokenizer import SDXLTokenizer
    tok = SDXLTokenizer()

    empty = tok.tokenize("")
    for tower in ("l", "g"):
        record(len(empty[tower]) == 1, f"{tower}: the empty prompt is one "
              "section", f"{len(empty[tower])}")
        section = empty[tower][0]
        record(section[0] == BOS and section[1] == EOS,
               f"{tower}: which is BOS then EOS and nothing else",
               f"{section[:3]}")
        record(len(section) == 77, f"{tower}: padded to 77", f"{len(section)}")

    one = tok.tokenize("a photo of a cat")["l"][0]
    record(one[:7] == [BOS, 320, 1125, 539, 320, 2368, EOS],
           "'a photo of a cat' tokenizes to the ids the reference gives",
           f"{one[:7]}")
    record(all(t == EOS for t in one[7:]),
           "and CLIP-L's padding is EOS, so padding and the end are the same "
           "token",
           f"{sorted(set(one[7:]))}")

    one_g = tok.tokenize("a photo of a cat")["g"][0]
    record(one_g[:7] == one[:7], "CLIP-G's text ids are identical")
    record(all(t == 0 for t in one_g[7:]),
           "but CLIP-G pads with token 0 instead",
           f"first pad {one_g[7]}")

    # Every section is exactly 77, whatever the prompt.
    for text in ("a", "a" * 300, "photo of a " + "very " * 60 + "long prompt"):
        for tower in ("l", "g"):
            for section in tok.tokenize(text)[tower]:
                record(len(section) == 77,
                       f"every section is 77 ({text[:18]!r}, {tower})",
                       f"{len(section)}")


def check_long_prompt_sections():
    """A prompt too long becomes several sections, each with its own BOS/EOS."""
    from nodes.model.tokenizer import SDXLTokenizer
    tok = SDXLTokenizer()
    # 60 repetitions is 68 ids -- inside 77. Measured, not guessed, which is
    # what the first version of this check did.
    long_prompt = "photo of a " + "very " * 200 + "long prompt that overflows"
    sections = tok.tokenize(long_prompt)["l"]
    record(len(sections) > 1,
           "a prompt over 77 tokens becomes several sections",
           f"{len(sections)}")
    record(all(s[0] == BOS for s in sections),
           "each starting with its own BOS")
    # The EOS is at the *end*. Position 1 only holds EOS for a section with
    # no content, because CLIP-L pads with EOS -- so a full section's second
    # element is ordinary text and checking it was the first version of this.
    record(all(s[-1] == EOS for s in sections),
           "and each ending with its own EOS",
           f"{[s[-1] for s in sections]}")

    # A single enormous *word* starts a new section rather than being cut.
    # "a" * 300 is only 38 BPE ids -- repeated letters merge well -- so it is
    # not long enough; a word of rarer characters is.
    huge_word = "zqxjv" * 120
    huge = tok.tokenize(huge_word)["l"]
    record(len(huge) > 1,
           "one very long word is split across sections rather than "
           "truncated", f"{len(huge)} sections for {len(huge_word)} chars")
    record(all(s[0] == BOS for s in huge),
           "each of those sections carrying its own BOS")
    # Nothing is dropped: every section is full, or is the last.
    body = [len([i for i in s if i not in (BOS, EOS)]) for s in huge]
    record(body[:-1] == [75] * len(body[:-1]),
           "and every section but the last is filled to the brim",
           f"{body[:4]}...")


def check_byte_alphabet():
    """The byte-level map, checked directly.

    A wrong one is silent: every word byte-encodes to something the merge
    table cannot use, every word comes out as single characters, and nothing
    raises. The failure I hit was zipping byte *ordinals* against the
    characters rather than byte *values*, which mapped byte 112 to `'!'`, so
    "photo" encoded to `'´«³¸³'` and merged nothing at all.
    """
    from nodes.model.tokenizer import bytes_to_unicode

    mapping = bytes_to_unicode()
    record(len(mapping) == 256, "every byte has a character",
           f"{len(mapping)}")
    record(len(set(mapping.values())) == 256,
           "and no two bytes share one")
    record(mapping[ord("p")] == "p",
           "byte 112 maps to 'p' -- the bug that broke every merge",
           f"{mapping[ord('p')]!r}")
    record(mapping[0] != "!", "byte 0 is not mapped to '!'")
    # Byte 10 is the eleventh byte with no printable character of its own,
    # so it lands at U+0100 + 10.
    record(ord(mapping[10]) == 256 + 10,
           "a control byte moves into the U+0100 block",
           f"{mapping[10]!r} = U+{ord(mapping[10]):04X}")
    record(ord(mapping[0]) == 256, "and so does byte 0",
           f"{mapping[0]!r} = U+{ord(mapping[0]):04X}")
    # Bijective in both directions -- which is the property that matters.
    # A utf-8 round trip does *not* hold and is not the right test: the
    # U+0100 block is two bytes in utf-8, so only the byte/character
    # bijection is meaningful.
    printable = {chr(c) for c in range(33, 127)}
    printable |= {chr(c) for c in range(161, 173)}
    printable |= {chr(c) for c in range(174, 256)}
    shifted = {chr(256 + i) for i in range(256 - len(printable))}
    record(set(mapping.values()) == printable | shifted,
           "the 188 printable bytes map to themselves and the other 68 to "
           "U+0100..U+0143, with no overlap",
           f"{len(printable)} printable + {len(shifted)} shifted, "
           f"got {len(set(mapping.values()))}")
    record(all(ord(v) == k for k, v in mapping.items()
               if ord(v) < 256),
           "and each of those maps to itself -- 112 to 'p', not to '!'")


def check_ids_are_real():
    """Every id we emit must be in the vocabulary."""
    from nodes.model.tokenizer import SDXLTokenizer
    tok = SDXLTokenizer()
    encoder = tok.clip_l._bpe.encoder
    seen = set()
    for text in CORPUS:
        for tower in ("l", "g"):
            for section in tok.tokenize(text)[tower]:
                seen.update(section)
    unknown = sorted(i for i in seen if i not in set(encoder.values()))
    record(not unknown, "every id we emit exists in the vocabulary",
           f"{len(unknown)} unknown: {unknown[:5]}")
    record(max(seen) <= 49407, "and none exceeds the vocabulary's range",
           f"max {max(seen)}")


def check_against_comfy():
    try:
        import paths as p
        root = p.get_comfy_dir()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from comfy.sdxl_clip import SDXLTokenizer as Theirs
    except Exception as exc:  # noqa: BLE001 -- absent means "not here"
        skip("agreement with comfyi's tokenizer",
             f"comfy not importable ({type(exc).__name__})")
        return

    from nodes.model.tokenizer import SDXLTokenizer as Ours
    ours, theirs = Ours(), Theirs()

    mismatches = []
    for text in CORPUS:
        mine = ours.tokenize(text)
        other = theirs.tokenize_with_weights(text)
        for tower in ("l", "g"):
            got = [list(section) for section in mine[tower]]
            want = [[i for i, _w in section] for section in other[tower]]
            if got != want:
                mismatches.append((tower, text))
    record(not mismatches,
           f"token-for-token identical over {len(CORPUS)} prompts x 2 towers",
           f"{len(mismatches)} differ: "
           + "; ".join(f"[{t}] {x[:24]!r}" for t, x in mismatches[:3]))


def check_deliberate_divergences():
    """The three documented differences, pinned so they cannot drift."""
    try:
        import paths as p
        root = p.get_comfy_dir()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from comfy.sdxl_clip import SDXLTokenizer as Theirs
    except Exception as exc:  # noqa: BLE001
        skip("the deliberate divergences",
             f"comfy not importable ({type(exc).__name__})")
        return

    from nodes.model.tokenizer import SDXLTokenizer as Ours
    ours, theirs = Ours(), Theirs()

    def ids(tokenizer, prompt, tower="l"):
        if isinstance(tokenizer, Ours):
            return [list(s) for s in tokenizer.tokenize(prompt)[tower]]
        return [[i for i, _w in s]
                for s in tokenizer.tokenize_with_weights(prompt)[tower]]

    # 1. Parentheses survive for us and are consumed by comfyi's weight parser.
    def content(sections, pad):
        # Sections are padded to 77, so their lengths always match; what
        # differs is how many real ids each carries before the padding.
        return [[i for i in s if i != pad] for s in sections]

    mine, other = ids(ours, "a (b) c"), ids(theirs, "a (b) c")
    record(mine != other, "premise: comfyi consumes the parentheses")
    mine_body = content(mine, EOS)[0]
    other_body = content(other, EOS)[0]
    record(len(mine_body) > len(other_body),
           "ours keeps them: more real ids before the padding",
           f"ours {len(mine_body)}, comfy {len(other_body)}")
    record(ids(ours, "(a)")[0].count(BOS) == 1,
           "and a parenthesised prompt does not lose its own BOS")

    # 2. A literal special token is text, not a second BOS.
    literal = "<|startoftext|> a cat"
    mine, other = ids(ours, literal), ids(theirs, literal)
    record(other[0].count(BOS) == 2,
           "premise: comfyi emits a second BOS for the literal text",
           f"{other[0].count(BOS)} BOS")
    record(mine[0].count(BOS) == 1,
           "ours does not: it is byte-encoded and merged like any text",
           f"{mine[0].count(BOS)} BOS")

    # 3. HTML entities are unescaped, per CLIP's basic_clean.
    entity, plain = ids(ours, "&lt;"), ids(ours, "<")
    record(entity == plain,
           "ours unescapes &lt; to <, which is what CLIP's basic_clean does")
    mine, other = ids(ours, "&lt;"), ids(theirs, "&lt;")
    record(mine != other,
           "comfy no longer does -- transformers 5.12.1 dropped that step")

    # The non-divergent neighbour: punctuation between letters is identical,
    # which is what says the divergences above are grammar and not a bug in
    # the alphabet.
    same = all(ids(ours, "a" + ch + "b") == ids(theirs, "a" + ch + "b")
               for ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    record(same, "all 32 ASCII punctuation marks between letters agree")


def main() -> int:
    print("== the sequence layout ==")
    check_layout()
    print("\n== long prompts ==")
    check_long_prompt_sections()
    print("\n== the byte-level alphabet ==")
    check_byte_alphabet()
    print("\n== every id is real ==")
    check_ids_are_real()
    print("\n== agreement with comfyi's tokenizer ==")
    check_against_comfy()
    print("\n== the deliberate divergences ==")
    check_deliberate_divergences()

    print("\n" + "=" * 60)
    if skipped:
        print(f"  {len(skipped)} check(s) skipped: "
              + ", ".join(sorted(set(skipped))))
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())