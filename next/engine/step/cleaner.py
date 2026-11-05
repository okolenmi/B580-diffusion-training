"""LoRA cleaner: find which parts of a trained adapter carry its useful change, keep those, and let the rest start over.

Every weight of the adapter (``element``) or every rank direction of its change (``component``, after an exact rewrite of each layer into
its singular directions) gets a gate between 0 and 1 that multiplies it. The adapter itself is frozen; only the gates train, with the
training step's own loss (so the anchor term, when it is on, counts too): a gate is pushed up where removing its part would cost, and down
where it would help. After ``steps`` such steps the ``keep`` fraction with the highest gates stays and the rest is removed.

What was kept can then be frozen (the others train from zero, or from a fresh start for a removed direction) so that training goes on in the
room that was cleared. The gates are applied by replacing the layer's factor parameters with tensors computed from them for the length of
the cleaning, so no layer's forward knows about them."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

MODES = ("element", "component", "magnitude")  # magnitude: no gates, the largest weights stay (the control the learned gates must beat)
START_LOGIT = 3.0  # a gate of 0.95: the cleaning starts almost where the adapter is
TIE_BREAK = 1e-3  # gates that never moved are ranked by the size of their weight


class Gates:
    """The gated view of a set of LoRA layers: ``engage`` swaps the factors for gated tensors, ``refresh`` recomputes them from the logits, ``release`` puts the parameters back."""

    def __init__(self, layers: Mapping[str, Any], mode: str):
        self.layers, self.mode = dict(layers), mode
        self.logits: dict[str, dict[str, Tensor]] = {}
        self.params: dict[str, dict[str, nn.Parameter]] = {}
        self.effective: dict[str, dict[str, Tensor]] = {}

    def engage(self) -> None:
        for path, layer in self.layers.items():
            device = layer.lora_A.device
            a, b = layer._parameters.pop("lora_A"), layer._parameters.pop("lora_B")
            a.requires_grad_(False)
            b.requires_grad_(False)
            self.params[path] = {"lora_A": a, "lora_B": b}
            shapes = {"a": (a.shape[0],), "b": (b.shape[1],)} if self.mode == "component" else {"a": tuple(a.shape), "b": tuple(b.shape)}
            keys = ("a",) if self.mode == "component" else ("a", "b")
            self.logits[path] = {k: torch.full(shapes[k], START_LOGIT, device=device, requires_grad=True) for k in keys}
        self.refresh()

    @torch.no_grad()
    def refresh(self) -> None:
        """New gated tensors from the logits, as leaves: a step may run several backward passes (the loss and the anchor), so the chain back to the
        logits is applied once by hand in ``collect`` instead of through a graph the first backward would free."""
        self.effective = {}
        for path, layer in self.layers.items():
            a, b = self.params[path]["lora_A"], self.params[path]["lora_B"]
            g = {k: torch.sigmoid(v) for k, v in self.logits[path].items()}
            if self.mode == "component":
                layer.lora_A = (a * g["a"].view(-1, *([1] * (a.ndim - 1)))).requires_grad_()
                layer.lora_B = b
                self.effective[path] = {"a": layer.lora_A}
            else:
                layer.lora_A, layer.lora_B = (a * g["a"]).requires_grad_(), (b * g["b"]).requires_grad_()
                self.effective[path] = {"a": layer.lora_A, "b": layer.lora_B}

    @torch.no_grad()
    def collect(self) -> None:
        """The gradient of the loss with respect to each logit, from the gradient of the gated tensors."""
        for path, per in self.logits.items():
            for key, logit in per.items():
                grad = self.effective[path][key].grad
                if grad is None:
                    logit.grad = torch.zeros_like(logit)
                    continue
                weight = self.params[path]["lora_A" if key == "a" else "lora_B"]
                slope = torch.sigmoid(logit) * (1.0 - torch.sigmoid(logit))
                if self.mode == "component":
                    logit.grad = (grad * weight).reshape(weight.shape[0], -1).sum(dim=1) * slope
                else:
                    logit.grad = grad * weight * slope

    def release(self) -> None:
        for path, layer in self.layers.items():
            for name in ("lora_A", "lora_B"):
                layer.__dict__.pop(name, None)
                param = self.params[path][name]
                param.requires_grad_(True)
                layer.register_parameter(name, param)
        self.params = {}

    def leaves(self) -> list[Tensor]:
        return [t for per in self.logits.values() for t in per.values()]


def to_singular_directions(layers: Mapping[str, Any]) -> None:
    """Rewrites each layer's factors so that its rows of A are orthogonal directions and B's columns their partners, scaled by the square root
    of the singular value: the same change (B A is unchanged), but now each rank direction means something on its own. On the host."""
    with torch.no_grad():
        for layer in layers.values():
            a, b = layer.lora_A, layer.lora_B
            rank = a.shape[0]
            am, bm = a.detach().reshape(rank, -1).float().cpu(), b.detach().reshape(-1, rank).float().cpu()
            qb, rb = torch.linalg.qr(bm)
            qa, ra = torch.linalg.qr(am.T)
            u, s, vh = torch.linalg.svd(rb @ ra.T)
            root = s.sqrt()
            new_b, new_a = (qb @ u) * root, (vh * root[:, None]) @ qa.T
            a.copy_(new_a.reshape(a.shape).to(a.device, a.dtype))
            b.copy_(new_b.reshape(b.shape).to(b.device, b.dtype))


def fresh_direction(row: Tensor) -> None:
    """A random starting row for a removed direction, at the usual size of a down-projection row."""
    bound = 1.0 / math.sqrt(row.numel())
    row.uniform_(-bound, bound)


class LoraCleaner:
    """Cleans the adapter of ``layers`` at step ``at`` and, with ``freeze``, keeps what it kept fixed from then on."""

    def __init__(self, layers: Mapping[str, Any], *, keep: float, at: int = 0, steps: int = 200, mode: str = "element", lr: float = 0.1,
                 sparsity: float = 0.0, freeze: bool = True, directory: str = "", save: Callable[[str], None] | None = None):
        if not 0.0 < keep < 1.0 or steps < 1 or lr <= 0 or at < 0 or sparsity < 0:
            raise ValueError("cleaning needs a keep fraction between 0 and 1, at least one step, a positive learning rate and a start step of 0 or more")
        if mode not in MODES:
            raise ValueError(f"unknown cleaning mode {mode!r}; choose from {MODES}")
        for path, layer in layers.items():
            if layer.kind != "lora":
                raise ValueError(f"the cleaner is defined for LoRA, not {layer.kind} ({path})")
        self.layers, self.keep, self.at, self.steps, self.mode = dict(layers), keep, at, steps, mode
        self.lr, self.sparsity, self.freeze, self.directory, self._save = lr, sparsity, freeze, directory, save
        self._frozen: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}  # layer -> factor -> (kept mask, kept values)
        self.summary: dict[str, Any] = {}

    def run(self, fetch: Callable[[], dict], backward: Callable[[dict], float], zero: Callable[[], None]) -> dict[str, Any]:
        """``fetch`` gives a batch, ``backward`` adds its gradients (and returns its loss), ``zero`` clears the adapter's gradients."""
        if not any(bool(layer.lora_B.detach().ne(0).any()) for layer in self.layers.values()):
            raise ValueError("the adapter has nothing to clean (every B is zero): give --init-adapter, or clean at a later step")
        if self.mode == "magnitude":
            scores = {path: {"a": layer.lora_A.detach().abs().cpu(), "b": layer.lora_B.detach().abs().cpu()} for path, layer in self.layers.items()}
            self._apply(scores)
            self.summary = self._summarise(scores, [])
            self._write()
            return self.summary
        if self.mode == "component":
            to_singular_directions(self.layers)
        gates = Gates(self.layers, self.mode)
        gates.engage()
        optimizer = torch.optim.Adam(gates.leaves(), lr=self.lr)
        losses: list[float] = []
        try:
            for _ in range(self.steps):
                gates.refresh()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(backward(fetch())))
                gates.collect()
                self._pull_closed(gates)
                optimizer.step()
            scores = self._scores(gates)
        finally:
            gates.release()
            zero()
        self._apply(scores)
        self.summary = self._summarise(scores, losses)
        self._write()
        return self.summary

    def _pull_closed(self, gates: Gates) -> None:
        """Optionally adds a pull toward closed, in units of the typical gradient of that tensor so that the weight means the same everywhere."""
        if not self.sparsity:
            return
        for tensor in gates.leaves():
            if tensor.grad is not None:
                tensor.grad += self.sparsity * tensor.grad.abs().mean() * torch.sigmoid(tensor.detach())

    def _scores(self, gates: Gates) -> dict[str, dict[str, Tensor]]:
        """The gate logits of every part, with a tie-break by the part's size, on the host."""
        out: dict[str, dict[str, Tensor]] = {}
        for path, per in gates.logits.items():
            params = gates.params[path]
            out[path] = {}
            for key, logit in per.items():
                weight = params["lora_A" if key == "a" else "lora_B"].detach()
                if self.mode == "component":
                    size = weight.reshape(weight.shape[0], -1).abs().amax(dim=1) if key == "a" else weight.reshape(-1, weight.shape[1]).abs().amax(dim=0)
                else:
                    size = weight.abs()
                out[path][key] = (logit.detach() + TIE_BREAK * size / size.max().clamp(min=1e-12)).cpu()
        return out

    def _masks(self, scores: dict[str, dict[str, Tensor]]) -> dict[str, dict[str, Tensor]]:
        """Keep the top ``keep`` share of each kind of part (A weights, B weights, or directions) over the whole adapter."""
        masks: dict[str, dict[str, Tensor]] = {path: {} for path in scores}
        for key in {k for per in scores.values() for k in per}:
            flat = torch.cat([per[key].flatten() for per in scores.values() if key in per])
            count = max(1, int(round(self.keep * flat.numel())))
            threshold = flat.kthvalue(flat.numel() - count + 1).values if count < flat.numel() else flat.min()
            for path, per in scores.items():
                if key in per:
                    masks[path][key] = per[key] >= threshold
        return masks

    @torch.no_grad()
    def _apply(self, scores: dict[str, dict[str, Tensor]]) -> None:
        self._kept = self._masks(scores)
        for path, layer in self.layers.items():
            a, b = layer.lora_A, layer.lora_B
            masks = self._factor_masks(path, a, b)
            if self.mode == "component":
                for index in (~self._kept[path]["a"].to(a.device)).nonzero().flatten().tolist():  # a removed direction starts over: a new row, a zero column
                    fresh_direction(a[index])
                    b[:, index].zero_()
            else:
                a.mul_(masks["lora_A"])
                b.mul_(masks["lora_B"])
            self._freeze_layer(path, {"lora_A": (masks["lora_A"], a), "lora_B": (masks["lora_B"], b)})

    def _factor_masks(self, path: str, a: Tensor, b: Tensor) -> dict[str, Tensor]:
        """The kept masks at the shape of the factors (a direction's mask spans its whole row of A and column of B), on their device."""
        kept = self._kept[path]
        if self.mode == "component":
            keep = kept["a"].to(a.device)
            return {"lora_A": keep.view(-1, *([1] * (a.ndim - 1))).expand(a.shape), "lora_B": keep.view(1, -1, *([1] * (b.ndim - 2))).expand(b.shape)}
        return {"lora_A": kept["a"].to(a.device), "lora_B": kept["b"].to(b.device)}

    def _freeze_layer(self, path: str, parts: dict[str, tuple[Tensor, Tensor]]) -> None:
        if self.freeze:
            self._frozen[path] = {name: (mask.clone(), param.detach().clone()) for name, (mask, param) in parts.items()}

    @torch.no_grad()
    def after_update(self) -> None:
        """Puts the frozen part back after the optimizer's step (its momentum and weight decay would otherwise move it)."""
        for path, parts in self._frozen.items():
            layer = self.layers[path]
            for name, (mask, saved) in parts.items():
                param = getattr(layer, name)
                param.copy_(torch.where(mask, saved, param))

    def restore(self) -> None:
        """After a resume: the mask of an earlier cleaning is read back and the weights the resumed adapter holds are the ones to keep."""
        if not self.freeze:
            return
        path = os.path.join(self.directory, "mask.safetensors")
        if not os.path.exists(path):
            raise ValueError(f"the cleaning ran in an earlier session but {path} is missing, so the kept part cannot be frozen again")
        stored = load_file(path)
        for layer_path, layer in self.layers.items():
            self._frozen[layer_path] = {name: (stored[f"{layer_path}.{name}"].bool().to(getattr(layer, name).device), getattr(layer, name).detach().clone())
                                        for name in ("lora_A", "lora_B")}

    def _summarise(self, scores: dict[str, dict[str, Tensor]], losses: list[float]) -> dict[str, Any]:
        def share(names: list[str], key: str) -> float:
            kept = sum(int(self._kept[p][key].sum()) for p in names if key in self._kept[p])
            total = sum(self._kept[p][key].numel() for p in names if key in self._kept[p])
            return kept / total if total else float("nan")

        def by(label: Callable[[str], str]) -> dict[str, float]:
            groups: dict[str, list[str]] = {}
            for path in self.layers:
                groups.setdefault(label(path), []).append(path)
            return {name: round(share(paths, "a"), 3) for name, paths in sorted(groups.items())}

        window = max(1, len(losses) // 5)
        return {"mode": self.mode, "keep": self.keep, "steps": len(losses), "loss_first": sum(losses[:window]) / window if losses else float("nan"),
                "loss_last": sum(losses[-window:]) / window if losses else float("nan"),
                "kept_by_region": by(lambda p: p.split(".")[0]), "kept_by_part": by(_part)}

    def describe(self) -> str:
        """The cleaning in a few lines for the console."""
        s = self.summary
        trained = f", gate loss {s['loss_first']:.4f} -> {s['loss_last']:.4f}" if s["steps"] else ""
        frozen = "; the kept part is frozen" if self.freeze else ""
        return f"cleaning ({s['mode']}): kept {s['keep']:.0%} after {s['steps']} gate steps{trained}{frozen}\n  kept by region {s['kept_by_region']}\n  kept by part {s['kept_by_part']}"

    def _write(self) -> None:
        """The cleaned adapter (before any further training), the mask and a summary, for looking at what was kept."""
        if not self.directory:
            return
        os.makedirs(self.directory, exist_ok=True)
        masks = {f"{path}.{name}": mask.to(torch.uint8).cpu().contiguous()
                 for path, layer in self.layers.items() for name, mask in self._factor_masks(path, layer.lora_A, layer.lora_B).items()}
        save_file(masks, os.path.join(self.directory, "mask.safetensors"))
        with open(os.path.join(self.directory, "summary.json"), "w", encoding="utf8") as handle:
            json.dump(self.summary, handle, indent=1)
        if self._save is not None:
            self._save(os.path.join(self.directory, "lora_cleaned.safetensors"))


def _part(path: str) -> str:
    for name in ("attn1", "attn2", "ff", "proj_in", "proj_out", "emb_layers", "in_layers", "out_layers", "skip", "label_emb", "time_embed"):
        if name in path:
            return name
    return "other"
