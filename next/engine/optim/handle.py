"""The optimizer a trainer talks to: an update rule, parameter groups, and a place for the state.

Parameters of the same shape, dtype, device and learning-rate multiplier form a *stack*; the rule updates a
whole stack at once and its state lives in stacked tensors (persistent, so nothing is stacked or copied back
per step). ``group_size`` caps how many parameters share a stack: 0 means all of them (fastest), 1 means one
at a time (lowest temporary memory), anything between trades the two.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor, nn

from ..runtime.residents import DeviceResident
from .algorithms import Algorithm
from .state import Float32Store, StateStore


@dataclass
class ParamGroup:
    params: Sequence[nn.Parameter]
    lr_mult: float = 1.0
    name: str = ""


@dataclass
class _Stack:
    params: list[nn.Parameter]
    lr_mult: float
    state: object = field(default=None)


def _apply(params: list[Tensor], deltas: list[Tensor], decays: list, foreach: bool) -> None:
    """``param = param * decay - delta`` for every parameter; ``decays`` holds None, a float or a 0-dim tensor each."""
    if foreach:
        by_kind: dict[tuple, list[int]] = {}
        for i, p in enumerate(params):
            by_kind.setdefault((p.device, p.dtype), []).append(i)
        for (_, dtype), idx in by_kind.items():
            targets = [params[i].data for i in idx]
            scaled = [i for i in idx if decays[i] is not None]
            if scaled:
                torch._foreach_mul_([params[i].data for i in scaled],
                                    [torch.as_tensor(decays[i], dtype=torch.float32) for i in scaled])
            torch._foreach_sub_(targets, [deltas[i].to(dtype) for i in idx])
        return
    for p, delta, d in zip(params, deltas, decays):
        if d is not None:
            p.data.mul_(d)
        p.data.sub_(delta.to(p.dtype))


class Optimizer(DeviceResident):
    def __init__(self, algorithm: Algorithm, groups: Sequence[ParamGroup], lr: float, *, group_size: int = 0,
                 store: StateStore | None = None, foreach: bool = True, grad_clip: float | None = None):
        self.algorithm, self.foreach, self.grad_clip = algorithm, foreach, grad_clip
        self.store = store or Float32Store()
        self.stacks: list[_Stack] = []
        self._lr = lr
        self._offloaded = False
        for group in groups:
            self._add(group, group_size)
        if not self.stacks:
            raise ValueError("the optimizer has no parameters to train")

    def _add(self, group: ParamGroup, group_size: int) -> None:
        by_kind: dict[tuple, list[nn.Parameter]] = {}
        for p in group.params:
            by_kind.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)
        for (shape, _dtype, device), params in by_kind.items():
            step = group_size or len(params)
            for start in range(0, len(params), step):
                chunk = params[start:start + step]
                shapes = self.algorithm.state_shapes(shape)
                tensors = {name: torch.zeros((len(chunk), *s), dtype=torch.float32, device=device) for name, s in shapes.items()}
                self.stacks.append(_Stack(chunk, group.lr_mult, self.store.wrap(tensors)))

    @property
    def params(self) -> list[nn.Parameter]:
        return [p for stack in self.stacks for p in stack.params]

    @property
    def lr(self) -> float:
        return self._lr

    def set_lr(self, lr: float) -> None:
        self._lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def _clip(self) -> None:
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return
        total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]))
        scale = torch.clamp(self.grad_clip / (total + 1e-6), max=1.0)
        torch._foreach_mul_(grads, scale.to(grads[0].device))

    def step(self, n: int = 1) -> None:
        if self._offloaded:
            raise RuntimeError("optimizer state is offloaded; reload it before stepping")
        if self.grad_clip:
            self._clip()
        self.algorithm.begin_step(n)
        targets: list[Tensor] = []
        deltas: list[Tensor] = []
        decays: list = []
        for stack in self.stacks:
            live = [i for i, p in enumerate(stack.params) if p.grad is not None]
            if not live:
                continue
            state = self.store.checkout(stack.state)
            whole = len(live) == len(stack.params)
            part = state if whole else {k: v[live] for k, v in state.items()}
            grad = torch.stack([stack.params[i].grad.detach().float() for i in live])
            members = [stack.params[i] for i in live]
            delta, decay = self.algorithm.update(grad, part, self._lr * stack.lr_mult, members)
            if not whole:
                for k, v in part.items():
                    state[k][live] = v
            self.store.commit(stack.state, state)
            targets += members
            deltas += list(delta.unbind(0))
            decays += list(decay.unbind(0)) if isinstance(decay, Tensor) else [decay] * len(members)
        _apply(targets, deltas, decays, self.foreach)

    def decay_states(self, factor: float) -> None:
        for stack in self.stacks:
            state = self.store.checkout(stack.state)
            self.algorithm.decay_state(state, factor)
            self.store.commit(stack.state, state)

    def reset_states(self) -> None:
        for stack in self.stacks:
            state = self.store.checkout(stack.state)
            self.algorithm.reset_state(state)
            self.store.commit(stack.state, state)

    def footprint_bytes(self) -> int:
        return 0 if self._offloaded else sum(self.store.footprint_bytes(s.state) for s in self.stacks)

    def offload(self) -> None:
        for stack in self.stacks:
            stack.state = self.store.to(stack.state, "cpu")
        self._offloaded = True

    def reload(self, device: str | None = None) -> None:
        for stack in self.stacks:
            stack.state = self.store.to(stack.state, device or str(stack.params[0].device))
        self._offloaded = False

    def release(self) -> None:
        self.stacks = []
        gc.collect()

    def describe(self) -> str:
        return f"{type(self.algorithm).__name__}({len(self.params)} params in {len(self.stacks)} stacks, " \
               f"{type(self.store).__name__})"


class FusedOptimizer(Optimizer):
    """Applies each parameter's update the moment its gradient is ready, so gradients never pile up.

    ``begin_step(sub_steps)`` marks the start of an optimizer step that spans ``sub_steps`` backward passes
    (gradient accumulation): the update happens during the last one only. ``step`` and ``zero_grad`` do nothing.
    The state must be float32, because each update touches one member of a stack.
    """

    def __init__(self, algorithm: Algorithm, groups: Sequence[ParamGroup], lr: float, *, group_size: int = 0,
                 store: StateStore | None = None, foreach: bool = False, grad_clip: float | None = None):
        if grad_clip:
            raise ValueError("gradient clipping needs every gradient at once, which a fused optimizer never has")
        super().__init__(algorithm, groups, lr, group_size=group_size, store=store, foreach=foreach)
        if not self.store.in_place:
            raise ValueError("a fused optimizer needs float32 state")
        self._hooks = []
        self._sub_steps = 1
        self._seen = 0
        self._in_backward = False
        for stack in self.stacks:
            for member, p in enumerate(stack.params):
                if p.requires_grad:
                    self._hooks.append(p.register_post_accumulate_grad_hook(self._on_ready(stack, member)))

    def _on_ready(self, stack: _Stack, member: int):
        def hook(p: Tensor) -> None:
            if p.grad is None:
                return
            if not self._in_backward:
                self._in_backward, self._seen = True, self._seen + 1
            if self._seen < self._sub_steps:
                return
            grad = p.grad.detach().float().unsqueeze(0)
            p.grad = None
            view = {k: v[member:member + 1] for k, v in stack.state.items()}
            delta, decay = self.algorithm.update(grad, view, self._lr * stack.lr_mult, [p])
            _apply([p], [delta[0]], [decay[0] if isinstance(decay, Tensor) else decay], False)
        return hook

    def begin_step(self, sub_steps: int = 1) -> None:
        self.algorithm.begin_step(1)
        self._in_backward, self._sub_steps, self._seen = False, sub_steps, 0

    def next_pass(self) -> None:
        """Call between the backward passes of one accumulation window."""
        self._in_backward = False

    def step(self, n: int = 1) -> None:
        pass

    def zero_grad(self) -> None:
        pass

    def release(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        super().release()
