"""Deciding what a model file contains, from its header alone."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Iterable

from .header import HeaderIndex
from .parts import ROLE_LABELS, PartInfo

Matcher = Callable[[HeaderIndex], "Match | None"]

_ADAPTER_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight", ".alpha",
                     ".lora_A.default.weight", ".lora_B.default.weight")


class Rejected(Exception):
    """A matcher's way to say 'this is not mine, and here is why', for the error message."""


@dataclass(frozen=True)
class Match:
    """One family's reading of a file: how sure, and which parts it found."""

    family_id: str
    score: float
    parts: tuple[PartInfo, ...]
    warnings: tuple[str, ...] = ()
    labels: dict = field(default_factory=dict, hash=False, compare=False)
    family_label: str = ""


@dataclass(frozen=True)
class Detection:
    """status: ready | needs_choice | error. ``family_id`` is set when ready."""

    status: str
    family_id: str | None
    parts: tuple[PartInfo, ...] = ()
    matches: tuple[Match, ...] = ()
    message: str | None = None
    warnings: tuple[str, ...] = ()
    labels: dict = field(default_factory=dict, hash=False, compare=False)
    report: tuple[tuple[str, int], ...] = ()
    file_size: int = 0
    family_label: str = ""
    error_kind: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ready"

    def label_of(self, role: str) -> str:
        return self.labels.get(role) or ROLE_LABELS.get(role, role)

    def summary(self) -> str:
        if not self.ok:
            return self.message or ""
        names = " + ".join(self.label_of(p.role) for p in self.parts) or "nothing recognised"
        dtypes = {p.dtype for p in self.parts if p.dtype}
        dtype = next(iter(dtypes)) if len(dtypes) == 1 else ("mixed dtypes" if dtypes else "")
        size = f"{self.file_size / 1e9:.1f} GB" if self.file_size >= 1e8 else f"{self.file_size / 1e6:.1f} MB"
        bits = [self.family_label or self.family_id or "", names, dtype.lower() if dtype else "", size]
        return ", ".join(b for b in bits if b)


def looks_like_adapter(index: HeaderIndex) -> bool:
    """A LoRA-style file: most keys are low-rank factors or alphas."""
    if not len(index):
        return False
    hits = sum(1 for k in index.entries if k.endswith(_ADAPTER_SUFFIXES))
    return hits / len(index) > 0.5


def _dedupe(parts: Iterable[PartInfo]) -> tuple[tuple[PartInfo, ...], tuple[str, ...]]:
    best: dict[str, PartInfo] = {}
    warnings: list[str] = []
    order: list[str] = []
    for part in parts:
        known = best.get(part.role)
        if known is None:
            best[part.role] = part
            order.append(part.role)
        else:
            warnings.append(f"more than one {part.role} found; using the one under {known.prefix!r}"
                            if known.key_count >= part.key_count else
                            f"more than one {part.role} found; using the one under {part.prefix!r}")
            if part.key_count > known.key_count:
                best[part.role] = part
    return tuple(best[r] for r in order), tuple(warnings)


def detect(index: HeaderIndex, matchers: dict[str, Matcher], family: str = "auto") -> Detection:
    """Run every family's matcher (or one, when ``family`` names it) and resolve the result."""
    report = tuple(index.top_prefixes(2))
    if family == "auto" and looks_like_adapter(index):
        return Detection("error", None, message="this file is an adapter (LoRA); use Load Adapter",
                         report=report, file_size=index.file_size, error_kind="adapter")
    if family != "auto" and family not in matchers:
        return Detection("error", None, message=f"unknown family {family!r}", report=report,
                         file_size=index.file_size, error_kind="unknown_family")
    chosen = matchers if family == "auto" else {family: matchers[family]}
    matches: list[Match] = []
    reasons: list[str] = []
    for fid, matcher in chosen.items():
        try:
            found = matcher(index)
        except Rejected as why:
            reasons.append(f"{fid}: {why}")
            continue
        if found is not None and found.parts:
            matches.append(found)
    matches.sort(key=lambda m: (-m.score, m.family_id))
    if not matches:
        what = f"does not look like a {family} model file" if family != "auto" else "is not a model file any family recognises"
        if reasons:
            what += " (" + "; ".join(reasons) + ")"
        return Detection("error", None, message=f"this file {what}", report=report,
                         file_size=index.file_size, error_kind="no_match")
    top = [m for m in matches if abs(m.score - matches[0].score) < 1e-9]
    if len(top) > 1:
        names = ", ".join(m.family_id for m in top)
        return Detection("needs_choice", None, matches=tuple(matches), report=report, file_size=index.file_size,
                         message=f"this file matches several families equally ({names}); choose one")
    best = matches[0]
    parts, dup = _dedupe(best.parts)
    return Detection("ready", best.family_id, parts=parts, matches=tuple(matches),
                     warnings=best.warnings + dup, labels=dict(best.labels), report=report,
                     file_size=index.file_size, family_label=best.family_label or best.family_id)
