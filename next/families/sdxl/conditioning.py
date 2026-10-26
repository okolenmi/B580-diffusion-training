"""SDXL text conditioning: the two CLIP towers, size embeddings, and the object that carries the result.

A conditioning is ``ctx`` (B, 77 * sections, 2048: CLIP-L's 768 then CLIP-G's 1280 features) and ``y``
(B, 2816: CLIP-G's pooled 1280 then six 256-wide size embeddings). ``y`` stays float32 and ``ctx`` is stored in
the encoder's output dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from ..api.parts import ModelPart
from ..general.clip.checkpoint import to_tower_state
from ..general.clip.model import CLIP_G, CLIP_L, CLIPConfig, CLIPTextTower
from ..general.clip.tokenizer import CLIPTokenizer
from ..general.embeddings import timestep_embedding
from ..general.tensors import read_prefixed

L_WIDTH, G_WIDTH, SIZE_WIDTH = 768, 1280, 256
SEGMENT_NAMES = ("clip_l", "clip_g", "pooled", "size")


@dataclass(frozen=True)
class SDXLConditioning:
    ctx: Tensor
    y: Tensor

    @staticmethod
    def stack(items: Sequence[SDXLConditioning]) -> SDXLConditioning:
        return SDXLConditioning(torch.cat([c.ctx for c in items]), torch.cat([c.y for c in items]))

    def to(self, device=None, dtype=None) -> SDXLConditioning:
        return SDXLConditioning(self.ctx.to(device=device, dtype=dtype), self.y.to(device=device))

    def tensors(self) -> list[Tensor]:
        return [self.ctx, self.y]

    def map(self, fn) -> SDXLConditioning:
        return SDXLConditioning(fn(self.ctx), fn(self.y))

    def index(self, i) -> SDXLConditioning:
        if isinstance(i, int):
            i = slice(i, i + 1)
        return SDXLConditioning(self.ctx[i], self.y[i])

    def segments(self) -> dict[str, Tensor]:
        return {"clip_l": self.ctx[..., :L_WIDTH], "clip_g": self.ctx[..., L_WIDTH:],
                "pooled": self.y[:, :G_WIDTH], "size": self.y[:, G_WIDTH:]}

    def blend(self, other: SDXLConditioning, p, segments: Sequence[str] | None = None) -> SDXLConditioning:
        """``self * (1 - p) + other * p`` on the chosen segments (all by default); the rest stay as in ``self``.

        ``p`` is a float or a per-sample tensor of shape (B,).
        """
        names = set(segments) if segments is not None else set(SEGMENT_NAMES)
        unknown = names - set(SEGMENT_NAMES)
        if unknown:
            raise ValueError(f"unknown segment(s) {sorted(unknown)}; choose from {SEGMENT_NAMES}")
        if self.ctx.shape[1:] != other.ctx.shape[1:]:
            raise ValueError(f"cannot blend contexts of shape {tuple(self.ctx.shape)} and {tuple(other.ctx.shape)}")
        weight = torch.as_tensor(p, dtype=torch.float32, device=self.ctx.device)
        weight = weight.view(-1, 1) if weight.ndim else weight.view(1, 1)
        ctx_w = torch.zeros(self.ctx.shape[-1], device=self.ctx.device)
        ctx_w[:L_WIDTH] = "clip_l" in names
        ctx_w[L_WIDTH:] = "clip_g" in names
        y_w = torch.zeros(self.y.shape[-1], device=self.y.device)
        y_w[:G_WIDTH] = "pooled" in names
        y_w[G_WIDTH:] = "size" in names
        ctx_mix = (self.ctx.float() * (1 - weight[:, :, None] * ctx_w) + other.ctx.float() * (weight[:, :, None] * ctx_w))
        y_mix = self.y * (1 - weight * y_w) + other.y.to(self.y.dtype) * (weight * y_w)
        return SDXLConditioning(ctx_mix.to(self.ctx.dtype), y_mix)

    def nbytes(self) -> int:
        return self.ctx.numel() * self.ctx.element_size() + self.y.numel() * self.y.element_size()


@dataclass(frozen=True)
class TextEncoding:
    """The size-independent half: what the towers produce from a prompt."""

    ctx: Tensor
    pooled: Tensor

    def to(self, device=None, dtype=None) -> TextEncoding:
        return TextEncoding(self.ctx.to(device=device, dtype=dtype), self.pooled.to(device=device, dtype=dtype))

    def nbytes(self) -> int:
        return self.ctx.numel() * self.ctx.element_size() + self.pooled.numel() * self.pooled.element_size()


def size_embedding(height: int, width: int, crop: tuple[int, int] = (0, 0),
                   target: tuple[int, int] | None = None, device="cpu") -> Tensor:
    """(1, 1536) embedding of height, width, crop top, crop left, target height, target width."""
    target_h, target_w = target if target is not None else (height, width)
    values = torch.tensor([height, width, crop[0], crop[1], target_h, target_w], dtype=torch.float32, device=device)
    return timestep_embedding(values, SIZE_WIDTH).reshape(1, -1)


class SDXLTextEncoder:
    """Both towers plus the tokenizer. Prompts longer than one 77-token section become several sections whose
    hidden states are joined along the sequence axis; the pooled vector comes from the first section."""

    def __init__(self, clip_l: CLIPTextTower, clip_g: CLIPTextTower, tokenizer, device="cpu",
                 out_dtype: torch.dtype = torch.bfloat16, batch_size: int = 64):
        self.clip_l, self.clip_g, self.tokenizer = clip_l, clip_g, tokenizer
        self.device, self.out_dtype, self.batch_size = torch.device(device), out_dtype, batch_size
        for tower in (clip_l, clip_g):
            tower.eval().requires_grad_(False)
            tower.to(self.device)

    def to(self, device) -> None:
        """Move both towers (to ``"cpu"`` to free the accelerator once everything is cached)."""
        self.device = torch.device(device)
        for tower in (self.clip_l, self.clip_g):
            tower.to(self.device)

    def modules(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        return self.clip_l, self.clip_g

    @property
    def tower_dtype(self) -> torch.dtype:
        return self.clip_g.text_model.final_layer_norm.weight.dtype

    def batching_available(self) -> bool:
        """Batched towers agree with one-at-a-time encoding only in float32 (see rework/lessons)."""
        return self.tower_dtype == torch.float32

    def encode_text(self, prompts: Sequence[str]) -> list[TextEncoding]:
        if not prompts:
            return []
        per_prompt = [self.tokenizer.tokenize(p) for p in prompts]
        rows_l, rows_g, owner = [], [], []
        for index, sections in enumerate(per_prompt):
            if len(sections["l"]) != len(sections["g"]):
                raise ValueError("the two towers tokenized a prompt into a different number of sections")
            for sec_l, sec_g in zip(sections["l"], sections["g"]):
                rows_l.append(sec_l)
                rows_g.append(sec_g)
                owner.append(index)
        step = self.batch_size if self.batching_available() else 1
        hidden, pooled_all = [], []
        with torch.no_grad():
            for start in range(0, len(rows_l), step):
                tok_l = torch.tensor(rows_l[start:start + step], dtype=torch.long, device=self.device)
                tok_g = torch.tensor(rows_g[start:start + step], dtype=torch.long, device=self.device)
                h_l, _ = self.clip_l(tok_l)
                h_g, pooled = self.clip_g(tok_g)
                hidden.append(torch.cat([h_l, h_g], dim=-1))
                pooled_all.append(pooled)
        hidden, pooled_all = torch.cat(hidden), torch.cat(pooled_all)
        out: list[TextEncoding] = []
        for index in range(len(prompts)):
            rows = [i for i, o in enumerate(owner) if o == index]
            ctx = torch.cat([hidden[i] for i in rows], dim=0).unsqueeze(0).to(self.out_dtype)
            out.append(TextEncoding(ctx, pooled_all[rows[0]].unsqueeze(0).to(self.out_dtype)))
        return out

    def assemble(self, text: TextEncoding, size: tuple[int, int], crop: tuple[int, int] = (0, 0),
                 target: tuple[int, int] | None = None) -> SDXLConditioning:
        """Join cached text with the size embeddings of one image (``size`` is height, width)."""
        emb = size_embedding(size[0], size[1], crop, target, device=text.pooled.device)
        return SDXLConditioning(text.ctx, torch.cat([text.pooled, emb], dim=-1))


class SDXLConditioningKit:
    """The family-facing surface of the encoder (see api.conditioning)."""

    def __init__(self, encoder: SDXLTextEncoder, default_size: tuple[int, int] = (1024, 1024)):
        self.encoder, self.default_size = encoder, default_size

    def cache_key(self, prompt: str) -> tuple:
        return ("sdxl", prompt)

    def encode_text(self, prompts: Sequence[str]) -> list[TextEncoding]:
        """The size-independent part, the thing worth caching."""
        return self.encoder.encode_text(prompts)

    def assemble(self, text: TextEncoding, size: tuple[int, int] | None = None) -> SDXLConditioning:
        """Join cached text with the size embedding of one image (``size`` is height, width)."""
        return self.encoder.assemble(text, tuple(size) if size is not None else self.default_size)

    def encode(self, prompts: Sequence[str], sizes: Sequence[tuple[int, int]] | None = None) -> list[SDXLConditioning]:
        texts = self.encode_text(prompts)
        sizes = sizes or [None] * len(prompts)
        return [self.assemble(t, s) for t, s in zip(texts, sizes)]

    @staticmethod
    def stack(items: Sequence[SDXLConditioning]) -> SDXLConditioning:
        return SDXLConditioning.stack(items)

    def blank(self, size=None) -> SDXLConditioning:
        return self.encode([""], [size])[0]

    def dump_text(self, text: TextEncoding) -> dict[str, Tensor]:
        return {"ctx": text.ctx, "pooled": text.pooled}

    def load_text(self, tensors: dict[str, Tensor]) -> TextEncoding:
        return TextEncoding(tensors["ctx"], tensors["pooled"])

    def bytes_per_prompt(self, sections: int = 1) -> int:
        itemsize = torch.empty((), dtype=self.encoder.out_dtype).element_size()
        return 77 * sections * (L_WIDTH + G_WIDTH) * itemsize + (G_WIDTH + 6 * SIZE_WIDTH) * 4


def _tower_from_part(part: ModelPart, config: CLIPConfig, dtype: torch.dtype) -> CLIPTextTower:
    layers = part.info.config.get("layers", config.layers)
    if layers != config.layers:
        raise ValueError(f"{part.role} file has {layers} layers, expected {config.layers}")
    state = read_prefixed(part.source.path, part.source.prefix, dtype=dtype)
    state = to_tower_state(state, part.info.config.get("layout", "hf"), config.layers)
    with torch.device("meta"):
        tower = CLIPTextTower(config)
    tower.load_state_dict(state, strict=True, assign=True)
    return tower


def load_text_encoder(parts: Mapping[str, ModelPart], tokenizer: CLIPTokenizer | None = None, *, device="cpu",
                      dtype: torch.dtype = torch.float16, out_dtype: torch.dtype = torch.bfloat16,
                      configs: tuple[CLIPConfig, CLIPConfig] = (CLIP_L, CLIP_G)) -> SDXLTextEncoder:
    """Build the encoder from the ``clip_l`` and ``clip_g`` parts a Load Model node produced."""
    for role in ("clip_l", "clip_g"):
        if role not in parts:
            raise KeyError(f"SDXL text encoding needs a {role!r} part")
    return SDXLTextEncoder(_tower_from_part(parts["clip_l"], configs[0], dtype),
                           _tower_from_part(parts["clip_g"], configs[1], dtype),
                           tokenizer or CLIPTokenizer(), device, out_dtype)
