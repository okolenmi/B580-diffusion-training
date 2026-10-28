"""What text each image trains with: the captions as stored, and the prompts after the stages of the graph, epoch by epoch."""

from __future__ import annotations

import statistics

from ..data.library import CHANNELS, ImageSet, split_tags
from ..data.prompts import PromptTable, TagSwap


def _median(values: list[int]) -> float:
    return statistics.median(values) if values else 0


class PromptAudit:
    """Reads an image set the way the trainer's prompt nodes do. ``tag_swap`` adds the Tag Swap stage the ``--tag-swap`` option adds."""

    def __init__(self, images: ImageSet, channel: str = "text", *, tag_swap: bool = False, epochs: int = 3, seed: int = 0, examples: int = 10):
        self.images, self.channel, self.tag_swap, self.epochs, self.seed, self.examples = images, channel, tag_swap, epochs, seed, examples
        self._tokens = self._tokenizer()

    @staticmethod
    def _tokenizer():
        try:
            from ..families.general.clip.tokenizer import CLIPTokenizer
            tokenizer = CLIPTokenizer()
            tokenizer.tokenize("a")
            return tokenizer
        except Exception:  # noqa: BLE001 - the vocabulary files may not be here; the token counts are then left out
            return None

    def token_count(self, text: str) -> int | None:
        return len(self._tokens.l.encode_text(text)) if self._tokens is not None and text.strip() else (0 if self._tokens is not None else None)

    def tables(self) -> list[tuple[int, dict[str, str]]]:
        table = PromptTable.from_images(self.images, self.channel, seed=self.seed)
        if self.tag_swap:
            table = table.with_stage(TagSwap("random", 0))
        out = []
        for epoch in range(self.epochs):
            table.set_epoch(epoch)
            out.append((epoch, table.snapshot()[1]))
        return out

    def _stats(self, texts: list[str]) -> str:
        empty = sum(1 for t in texts if not t.strip())
        tags = [len(split_tags(t)) for t in texts]
        counts = [c for c in (self.token_count(t) for t in texts) if c is not None]
        tokens = f", CLIP tokens median {_median(counts):.0f} and most {max(counts)}, {sum(c > 75 for c in counts)} past 75 (a second 77-token section)" if counts else ""
        return (f"{empty} empty ({empty / max(len(texts), 1):.0%}), {len({t for t in texts if t.strip()})} distinct, "
                f"comma-separated parts per text: median {_median(tags):.0f}, most {max(tags, default=0)}{tokens}")

    def markdown(self) -> str:
        ids = list(self.images.ids)
        lines = ["# Prompt audit", "", f"{len(ids)} images; the trainer reads channel `{self.channel}`; Tag Swap {'ON' if self.tag_swap else 'off'}.", "",
                 "## Captions as stored, by channel", ""]
        for channel in CHANNELS:
            lines.append(f"- `{channel}`{' (read by training)' if channel == self.channel else ''}: {self._stats([self.images.caption(i, channel) for i in ids])}")
        empty_read = [i for i in ids if not self.images.caption(i, self.channel).strip()]
        if empty_read:
            others = [c for c in CHANNELS if c != self.channel and any(self.images.caption(i, c).strip() for i in empty_read)]
            lines += ["", f"**{len(empty_read)} of {len(ids)} images have an empty `{self.channel}` caption**: they train with the empty prompt."
                      + (f" Those images do have text in channel(s) {', '.join(others)}." if others else "")]
        lines += ["", "## Prompts the trainer feeds, by epoch", ""]
        tables = self.tables()
        for epoch, active in tables:
            lines.append(f"- epoch {epoch}: {self._stats([active[i] for i in ids])}")
        lines += ["", "## Examples (image id, stored caption, prompt per epoch)", ""]
        step = max(len(ids) // self.examples, 1)
        for i in ids[::step][: self.examples]:
            lines.append(f"- `{i}`: stored: {self.images.caption(i, self.channel)[:110]!r}")
            lines += [f"    - epoch {epoch}: {active[i][:110]!r}" for epoch, active in tables]
        return "\n".join(lines) + "\n"
