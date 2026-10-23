"""Which part of a network a module belongs to, so statistics and errors can be reported by region instead of by thousands of layer names."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..perf.modules import region_of, role_of

_BLOCK = re.compile(r"^(input_blocks|output_blocks)\.(\d+)")


@dataclass(frozen=True)
class Region:
    group: str          # down, mid, up, embed (time and label embeddings), out, other
    block: int | None   # index of the input or output block, None for mid, embed and out

    @property
    def name(self) -> str:
        return self.group if self.block is None else f"{self.group}.{self.block}"


class Regions:
    """UNet paths (``input_blocks.4.1...``) to regions. ``coarse`` collapses blocks into down, mid and up."""

    def __init__(self, coarse: bool = False):
        self.coarse = coarse

    def of(self, path: str) -> Region:
        group = region_of(path)
        if group == "other":
            head = path.split(".")[0]
            group = "embed" if head in ("time_embed", "label_emb") else ("out" if head == "out" else "other")
        match = _BLOCK.match(path)
        return Region(group, None if self.coarse or match is None else int(match.group(2)))

    @staticmethod
    def role(path: str) -> str:
        return role_of(path)

    def order(self, regions) -> list[Region]:
        rank = {"embed": 0, "down": 1, "mid": 2, "up": 3, "out": 4, "other": 5}
        return sorted(regions, key=lambda r: (rank.get(r.group, 9), -1 if r.block is None else r.block))
