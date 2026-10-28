"""Where a trained adapter changed the network: the size of every layer's update by region and kind of layer, and how two adapters agree."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import torch
from safetensors import safe_open
from torch import Tensor

from ..families.general.adapters.schemes import read_adapter

_REGION = re.compile(r"(input_blocks|middle_block|output_blocks)(?:[._](\d+))?")
_KINDS = (("attn1", "self attention"), ("attn2", "cross attention"), ("ff", "feed-forward"), ("proj_in", "projection in"), ("proj_out", "projection out"),
          ("time_embed", "timestep embedding"), ("label_emb", "size embedding"))


def _kind(name: str) -> str:
    for token, label in _KINDS:
        if re.search(rf"(^|[._]){token}([._]|$)", name):
            return label
    return "other (convolutions and the rest)"


def _region(name: str) -> str:
    found = _REGION.search(name)
    if not found:
        return "elsewhere"
    return found.group(1) if found.group(2) is None else f"{found.group(1)}.{found.group(2)}"


def _coarse(region: str) -> str:
    return {"input_blocks": "down", "middle_block": "mid", "output_blocks": "up"}.get(region.split(".")[0], "elsewhere")


@dataclass
class Update:
    """One layer's low-rank update ``scale * up @ down`` kept as its two small factors (the full matrix is never built)."""

    name: str
    up: Tensor
    down: Tensor
    scale: float

    @property
    def rank(self) -> int:
        return self.down.shape[0]

    def energy(self) -> float:
        """``|update|_F ** 2`` from the two rank-by-rank Gram matrices."""
        return float(self.scale ** 2 * ((self.up.T @ self.up) * (self.down @ self.down.T)).sum())

    def dot(self, other: Update) -> float:
        return float(self.scale * other.scale * ((self.up.T @ other.up) * (self.down @ other.down.T)).sum())


def read_updates(path: str) -> dict[str, Update]:
    state, _ = read_adapter(path)
    out: dict[str, Update] = {}
    for key, tensor in state.items():
        match = re.fullmatch(r"(.+)\.(lora_down\.weight|lora_A(?:\.weight)?)", key)
        if not match:
            continue
        base = match.group(1)
        up = next((state[k] for k in (f"{base}.lora_up.weight", f"{base}.lora_B", f"{base}.lora_B.weight") if k in state), None)
        if up is None:
            continue
        down = tensor.double().reshape(tensor.shape[0], -1)
        up = up.double().reshape(up.shape[0], -1)
        alpha = state.get(f"{base}.alpha")
        scale = float(alpha.reshape(-1)[0]) / down.shape[0] if alpha is not None else 1.0
        out[base] = Update(base, up, down, scale)
    if not out:
        raise ValueError(f"{path}: no LoRA factors found (expected kohya or native keys)")
    return out


def weight_norms(updates: dict[str, Update], checkpoint: str) -> dict[str, float]:
    """``|W|_F`` of the base weight each update belongs to, read from the checkpoint tensor by tensor."""
    wanted = {re.sub(r"[.]", "_", u.name.split("lora_unet_", 1)[-1]): u.name for u in updates.values()}
    norms: dict[str, float] = {}
    with safe_open(checkpoint, framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118 - safetensors handles are not mappings
            if not key.endswith(".weight"):
                continue
            tail = key[:-len(".weight")]
            for marker in ("input_blocks", "middle_block", "output_blocks", "time_embed", "label_emb"):
                if marker in tail:
                    tail = tail[tail.index(marker):]
                    break
            name = wanted.get(re.sub(r"[.]", "_", tail))
            if name:
                norms[name] = float(handle.get_tensor(key).double().norm())
    return norms


class LoraMap:
    def __init__(self, updates: dict[str, Update], norms: dict[str, float] | None = None):
        self.updates, self.norms = updates, norms or {}

    def groups(self, key) -> dict[str, list[Update]]:
        out: dict[str, list[Update]] = {}
        for update in self.updates.values():
            out.setdefault(key(update.name), []).append(update)
        return dict(sorted(out.items()))

    def _row(self, label: str, members: list[Update], total: float) -> str:
        energy = [u.energy() for u in members]
        size = [e ** 0.5 for e in energy]
        median = statistics.median(size)
        dead = sum(1 for s in size if s <= 1e-6 * max(max(size), 1e-30))
        relative = [s / self.norms[u.name] for s, u in zip(size, members, strict=True) if u.name in self.norms]
        rel = f" | {statistics.median(relative):.2e} | {max(relative):.2e}" if relative else " | - | -"
        return f"| {label} | {len(members)} | {sum(energy) / total:.1%} | {median:.3e} | {dead}{rel} |"

    def markdown(self, title: str) -> str:
        total = sum(u.energy() for u in self.updates.values()) or 1.0
        header = "| group | layers | share of total update energy | median update size | layers with no update | median update / weight | largest update / weight |\n|---|---|---|---|---|---|---|"
        ranks = sorted({u.rank for u in self.updates.values()})
        lines = [f"# LoRA map: {title}", "", f"{len(self.updates)} layers, rank {', '.join(map(str, ranks))}. Update size is the Frobenius norm of `scale * up @ down`; energy is its square."
                 + ("" if self.norms else " (No checkpoint given, so no sizes relative to the weights.)"), ""]
        for name, key in (("by part of the network", lambda n: _coarse(_region(n))), ("by kind of layer", _kind), ("by block", lambda n: _region(n))):
            lines += [f"## {name}", "", header]
            lines += [self._row(label, members, total) for label, members in self.groups(key).items()]
            lines.append("")
        ranked = sorted(self.updates.values(), key=lambda u: -u.energy())
        lines += ["## The ten layers that changed most", "", "| layer | share of energy | size |", "|---|---|---|"]
        lines += [f"| `{u.name}` | {u.energy() / total:.1%} | {u.energy() ** 0.5:.3e} |" for u in ranked[:10]]
        top = sum(u.energy() for u in ranked[: max(len(ranked) // 10, 1)]) / total
        lines += ["", f"The most-changed tenth of the layers holds {top:.0%} of the energy."]
        return "\n".join(lines) + "\n"


def agreement(a: dict[str, Update], b: dict[str, Update], names: tuple[str, str]) -> str:
    """Cosine between two adapters' updates, layer by layer. Near 1: they learned the same change; near 0: different directions."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return f"No layer is in both {names[0]} and {names[1]}.\n"
    cos = {}
    for key in shared:
        denom = (a[key].energy() * b[key].energy()) ** 0.5
        cos[key] = a[key].dot(b[key]) / denom if denom else 0.0
    lines = [f"# Agreement: {names[0]} against {names[1]}", "", f"{len(shared)} shared layers ({len(a) - len(shared)} only in the first, {len(b) - len(shared)} only in the second).", "",
             "| group | layers | median cosine | lowest | highest |", "|---|---|---|---|---|"]
    for key in (lambda _: "all", lambda n: _coarse(_region(n)), _kind):
        groups: dict[str, list[float]] = {}
        for name, value in cos.items():
            groups.setdefault(key(name), []).append(value)
        lines += [f"| {g} | {len(v)} | {statistics.median(v):.3f} | {min(v):.3f} | {max(v):.3f} |" for g, v in sorted(groups.items())]
    ea, eb = sum(a[k].energy() for k in shared), sum(b[k].energy() for k in shared)
    dot = sum(a[k].dot(b[k]) for k in shared)
    lines += ["", f"Over all shared layers together the cosine is {dot / (ea * eb) ** 0.5:.3f}; the two adapters' total update sizes are {ea ** 0.5:.3e} and {eb ** 0.5:.3e}.",
              "", "Two adapters trained on the same data from different seeds share the part of the update the data demands and differ in the rest; "
              "a cosine near 0 for a layer means that layer's update is mostly not what the data asked for."]
    return "\n".join(lines) + "\n"


def lora_map_report(paths: list[str], checkpoint: str | None = None) -> str:
    loaded = [read_updates(p) for p in paths]
    parts = []
    for path, updates in zip(paths, loaded, strict=True):
        parts.append(LoraMap(updates, weight_norms(updates, checkpoint) if checkpoint else None).markdown(path.rsplit("/", 1)[-1]))
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            parts.append(agreement(loaded[i], loaded[j], (paths[i].rsplit("/", 1)[-1], paths[j].rsplit("/", 1)[-1])))
    torch.set_grad_enabled(True)
    return "\n".join(parts)
