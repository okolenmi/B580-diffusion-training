"""Kernel names to categories. One ordered table: the first rule that matches wins, so a new kernel family is one line."""

from __future__ import annotations

import re

RULES: tuple[tuple[str, str], ...] = (
    (r"xpuint8|convrot|quantize|dequant|rotate|int_mm", "int8"),
    (r"flash|sdpa|scaled_dot|attention|softmax|rms_rope|rope", "attention"),
    (r"conv", "convolution"),
    (r"gemm|\bmm\b|addmm|bmm|matmul|linear|dpas|baddbmm", "gemm"),
    (r"layer_norm|group_norm|batch_norm|rms_norm|adaln|norm", "norm"),
    (r"_to_copy|copy_|copy|cast|contiguous|clone|\.to\b", "copy"),
    (r"sum|mean|amax|amin|max|min|reduce|var|std|argmax|cumsum", "reduce"),
    (r"fill|zero|memset|empty|memcpy|cat|index|gather|scatter|embedding|transpose|permute|reorder|stack|slice|select|view|reshape|as_strided|expand|chunk|split", "memory"),
    (r"add|mul|sub|div|silu|gelu|relu|sigmoid|tanh|exp|where|neg|pow|sqrt|rsqrt|clamp|elementwise|vectorized|unrolled|abs|lerp|addc|foreach|sign|round", "elementwise"),
)
CATEGORIES = ("gemm", "convolution", "attention", "norm", "elementwise", "copy", "reduce", "int8", "memory", "other")


class Categorizer:
    def __init__(self, rules: tuple[tuple[str, str], ...] = RULES):
        self._rules = [(re.compile(pattern, re.IGNORECASE), category) for pattern, category in rules]
        self.unknown: dict[str, int] = {}

    def category(self, name: str) -> str:
        for pattern, category in self._rules:
            if pattern.search(name):
                return category
        self.unknown[name] = self.unknown.get(name, 0) + 1
        return "other"
