"""Anima text conditioning: Qwen3 hidden states plus T5 token ids, which the adapter inside the DiT file turns into context.

What is cached per prompt is the adapter's *input* (``TextEncoding``): small, and independent of the DiT, so one cache serves
every DiT file. A batch (``AnimaConditioning``) is always ``tokens`` positions wide, so every step has the same shapes (captured
step graphs are reused) and the DiT sees the 512 context positions it was trained with; masks mark the real ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..api.parts import ModelPart
from ..general.tensors import read_prefixed
from .text import QwenConfig, QwenEncoder
from .tokens import AnimaTokenizer, T5_END

TOKENS = 512
WIDTH = QwenConfig.width
SEGMENT_NAMES = ("qwen", "t5")


def _pad(rows: Sequence[Tensor], length: int, fill=0) -> tuple[Tensor, Tensor]:
    """Rows (1, n, ...) padded with ``fill`` to ``length`` and stacked, with a (B, length) mask of the real positions."""
    out = rows[0].new_full((len(rows), length, *rows[0].shape[2:]), fill)
    mask = torch.zeros(len(rows), length, dtype=torch.bool, device=rows[0].device)
    for i, row in enumerate(rows):
        out[i, : row.shape[1]] = row[0]
        mask[i, : row.shape[1]] = True
    return out, mask


@dataclass(frozen=True)
class TextEncoding:
    """One prompt: ``hidden`` (1, n, 1024) from Qwen3, ``ids`` (1, m) from T5's tokenizer."""

    hidden: Tensor
    ids: Tensor

    def to(self, device=None, dtype=None) -> TextEncoding:
        return TextEncoding(self.hidden.to(device=device, dtype=dtype), self.ids.to(device=device))

    def nbytes(self) -> int:
        return self.hidden.numel() * self.hidden.element_size() + self.ids.numel() * self.ids.element_size()


@dataclass(frozen=True)
class AnimaConditioning:
    hidden: Tensor        # (B, tokens, 1024), zero where not real
    ids: Tensor           # (B, tokens), 0 where not real
    hidden_mask: Tensor   # (B, tokens) bool
    ids_mask: Tensor      # (B, tokens) bool

    @staticmethod
    def of(text: TextEncoding, tokens: int = TOKENS) -> AnimaConditioning:
        hidden, hidden_mask = _pad([text.hidden], tokens)
        ids, ids_mask = _pad([text.ids], tokens)
        return AnimaConditioning(hidden, ids, hidden_mask, ids_mask)

    @staticmethod
    def stack(items: Sequence[AnimaConditioning]) -> AnimaConditioning:
        return AnimaConditioning(*(torch.cat(parts) for parts in zip(*(i.tensors() for i in items))))

    def to(self, device=None, dtype=None) -> AnimaConditioning:
        return AnimaConditioning(self.hidden.to(device=device, dtype=dtype), *(t.to(device=device) for t in self.tensors()[1:]))

    def tensors(self) -> list[Tensor]:
        return [self.hidden, self.ids, self.hidden_mask, self.ids_mask]

    def map(self, fn) -> AnimaConditioning:
        return AnimaConditioning(*(fn(t) for t in self.tensors()))

    def index(self, i) -> AnimaConditioning:
        if isinstance(i, int):
            i = slice(i, i + 1)
        return AnimaConditioning(*(t[i] for t in self.tensors()))

    def segments(self) -> dict[str, Tensor]:
        return {"qwen": self.hidden, "t5": self.ids}

    def blend(self, other: AnimaConditioning, p, segments: Sequence[str] | None = None) -> AnimaConditioning:
        """``self * (1 - p) + other * p`` on Qwen's hidden states. T5 ids cannot be mixed: a sample takes ``other``'s ids
        once ``p`` reaches 0.5. ``p`` is a float or a per-sample tensor (B,)."""
        names = set(segments) if segments is not None else set(SEGMENT_NAMES)
        unknown = names - set(SEGMENT_NAMES)
        if unknown:
            raise ValueError(f"unknown segment(s) {sorted(unknown)}; choose from {SEGMENT_NAMES}")
        weight = torch.as_tensor(p, dtype=torch.float32, device=self.hidden.device).reshape(-1)
        if weight.numel() == 1:
            weight = weight.expand(self.hidden.shape[0])
        hidden, hidden_mask = self.hidden, self.hidden_mask
        if "qwen" in names:
            w = weight[:, None, None]
            hidden = (self.hidden.float() * (1 - w) + other.hidden.float() * w).to(self.hidden.dtype)
            # a position stays while either side still has weight there
            hidden_mask = ((self.hidden_mask & (weight < 1)[:, None]) | (other.hidden_mask & (weight > 0)[:, None]))
        ids, ids_mask = self.ids, self.ids_mask
        if "t5" in names:
            take = (weight >= 0.5)[:, None]
            ids, ids_mask = torch.where(take, other.ids, self.ids), torch.where(take, other.ids_mask, self.ids_mask)
        return AnimaConditioning(hidden, ids, hidden_mask, ids_mask)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


class AnimaTextEncoder:
    """The Qwen3 encoder plus both tokenizers. Prompts longer than ``tokens`` are cut (the DiT's context has that many
    positions); T5's ids keep their closing ``</s>``."""

    def __init__(self, qwen: QwenEncoder, tokenizer: AnimaTokenizer, device="cpu", out_dtype: torch.dtype = torch.bfloat16,
                 tokens: int = TOKENS, batch_size: int = 16):
        self.qwen, self.tokenizer, self.tokens = qwen, tokenizer, tokens
        self.device, self.out_dtype, self.batch_size = torch.device(device), out_dtype, batch_size
        qwen.eval().requires_grad_(False)
        qwen.to(self.device)

    def to(self, device) -> None:
        self.device = torch.device(device)
        self.qwen.to(self.device)

    def modules(self) -> tuple[torch.nn.Module]:
        return (self.qwen,)

    @property
    def encoder_dtype(self) -> torch.dtype:
        return self.qwen.model.norm.weight.dtype

    def batching_available(self) -> bool:
        """Batched and one-at-a-time encodings agree to the last bit only in float32; a cached prompt must not depend on
        which prompts were encoded beside it."""
        return self.encoder_dtype == torch.float32

    def _ids(self, text: str) -> tuple[list[int], list[int]]:
        qwen_ids, t5_ids = self.tokenizer.tokenize(text)
        if len(t5_ids) > self.tokens:
            t5_ids = t5_ids[: self.tokens - 1] + [T5_END]
        return qwen_ids[: self.tokens], t5_ids

    def encode_text(self, prompts: Sequence[str]) -> list[TextEncoding]:
        if not prompts:
            return []
        rows = [self._ids(p) for p in prompts]
        step = self.batch_size if self.batching_available() else 1
        out: list[TextEncoding] = []
        with torch.no_grad():
            for start in range(0, len(rows), step):
                chunk = rows[start:start + step]
                longest = max(len(q) for q, _ in chunk)
                ids = torch.zeros(len(chunk), longest, dtype=torch.long)
                mask = torch.zeros(len(chunk), longest, dtype=torch.long)
                for i, (q, _) in enumerate(chunk):
                    ids[i, : len(q)] = torch.tensor(q)
                    mask[i, : len(q)] = 1
                hidden = self.qwen(ids.to(self.device), mask.to(self.device) if len(chunk) > 1 else None)
                for i, (q, t) in enumerate(chunk):
                    out.append(TextEncoding(hidden[i:i + 1, : len(q)].to(self.out_dtype), torch.tensor([t], dtype=torch.long, device=self.device)))
        return out


class AnimaConditioningKit:
    """The family-facing surface of the encoder (see api.conditioning). Anima has no size conditioning."""

    def __init__(self, encoder: AnimaTextEncoder, typical_tokens: int = 128):
        self.encoder, self.typical_tokens = encoder, typical_tokens

    def cache_key(self, prompt: str) -> tuple:
        return ("anima", prompt)

    def encode_text(self, prompts: Sequence[str]) -> list[TextEncoding]:
        return self.encoder.encode_text(prompts)

    def assemble(self, text: TextEncoding, size=None) -> AnimaConditioning:
        return AnimaConditioning.of(text, self.encoder.tokens)

    def encode(self, prompts: Sequence[str], sizes=None) -> list[AnimaConditioning]:
        return [self.assemble(t) for t in self.encode_text(prompts)]

    @staticmethod
    def stack(items: Sequence[AnimaConditioning]) -> AnimaConditioning:
        return AnimaConditioning.stack(items)

    def blank(self, size=None) -> AnimaConditioning:
        return self.encode([""])[0]

    def dump_text(self, text: TextEncoding) -> dict[str, Tensor]:
        return {"hidden": text.hidden, "ids": text.ids}

    def load_text(self, tensors: dict[str, Tensor]) -> TextEncoding:
        return TextEncoding(tensors["hidden"], tensors["ids"])

    def bytes_per_prompt(self) -> int:
        itemsize = torch.empty((), dtype=self.encoder.out_dtype).element_size()
        return self.typical_tokens * (WIDTH * itemsize + 8)


def load_text_encoder(parts: Mapping[str, ModelPart], tokenizer: AnimaTokenizer | None = None, *, device="cpu",
                      dtype: torch.dtype = torch.bfloat16, out_dtype: torch.dtype = torch.bfloat16) -> AnimaTextEncoder:
    """Build the encoder from the ``qwen3`` part a Load Model node produced."""
    if "qwen3" not in parts:
        raise KeyError("Anima text encoding needs a 'qwen3' part")
    part = parts["qwen3"]
    layers = part.info.config.get("layers", QwenConfig.layers)
    if layers != QwenConfig.layers or part.info.config.get("hidden", WIDTH) != WIDTH:
        raise ValueError(f"the qwen3 file has {layers} layers of width {part.info.config.get('hidden')}; Anima uses Qwen3-0.6B")
    # Qwen3 is a bfloat16 model; float16 would overflow in its residual stream.
    state = read_prefixed(part.source.path, part.source.prefix, dtype=torch.bfloat16 if dtype == torch.float16 else dtype)
    with torch.device("meta"):
        qwen = QwenEncoder()
    qwen.model.load_state_dict(state, strict=True, assign=True)
    return AnimaTextEncoder(qwen, tokenizer or AnimaTokenizer(), device, out_dtype)
