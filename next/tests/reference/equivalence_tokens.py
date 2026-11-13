"""Our Qwen and T5 tokenizers against Hugging Face's own classes on the same vocabulary folders (needs ``transformers``).

    QWEN_TOKENIZER_DIR=<ComfyUI>/comfy/text_encoders/qwen25_tokenizer T5_TOKENIZER_DIR=<ComfyUI>/comfy/text_encoders/t5_tokenizer \\
        python -m next.tests.reference.equivalence_tokens
"""
import random

from transformers import Qwen2Tokenizer, T5TokenizerFast

from ...families.anima.tokens import QwenTokenizer, T5Tokenizer, vocabulary_dir

qdir, tdir = vocabulary_dir("qwen"), vocabulary_dir("t5")
mine_q, mine_t = QwenTokenizer(qdir), T5Tokenizer(tdir)
ref_q, ref_t = Qwen2Tokenizer.from_pretrained(str(qdir)), T5TokenizerFast.from_pretrained(str(tdir))
samples = ["1girl, solo, long hair, hatsune miku (vocaloid), looking at viewer", "A photo of a cat.\nSecond line\n\n  indented",
           "Ünïcödé: 日本語のテキスト, café, 🙂 emoji", "it's they're I'll 12345 3.14 $5 100%", "<|im_end|> literal special", "", " ",
           "tabs\tand   spaces   ", "é composed vs decomposed", "{braces} [brackets] <angle> \\backslash"]
alphabet = "abcdefgh ABC 0123 ,.!?()\n\t'\"éü日本🙂-_"
rng = random.Random(0)
samples += ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60))) for _ in range(300)]
bad = 0
for s in samples:
    q_ref = ref_q(s)["input_ids"] or [151643]
    t_ref = ref_t(s)["input_ids"]
    if mine_q.encode(s) != q_ref or mine_t.encode(s) != t_ref:
        bad += 1
        if bad <= 5:
            print("DIFF", repr(s), mine_q.encode(s)[:12], q_ref[:12], mine_t.encode(s)[:12], t_ref[:12])
print(f"{len(samples)} strings, {bad} differ")
