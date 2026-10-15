"""Activation checkpointing for blocks whose weights are partly frozen.

The block's forward runs without storing activations; in backward it runs again with the same random state
and autocast settings, and gradients flow to the block's inputs and to those of its parameters that require
grad (an adapter's factors, not the frozen base weights). Parameters are passed through the autograd function
as arguments so the graph stays connected even when no input requires grad.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn


def _autocast_state(inputs: list[Tensor]) -> tuple[str | None, dict]:
    device_type = inputs[0].device.type if inputs else "cpu"
    try:
        if not torch.amp.autocast_mode.is_autocast_available(device_type):
            return None, {}
    except Exception:  # noqa: BLE001  (an unknown device type: assume it is available)
        pass
    return device_type, {"enabled": torch.is_autocast_enabled(device_type),
                         "dtype": torch.get_autocast_dtype(device_type),
                         "cache_enabled": torch.is_autocast_cache_enabled()}


def _device_rng(tensors: list[Tensor]) -> tuple[str | None, list[int], list[Tensor]]:
    """Random state of the accelerator the tensors live on (None, [], [] on the CPU)."""
    for tensor in tensors:
        kind = tensor.device.type
        module = getattr(torch, kind, None)
        if kind == "cpu" or module is None or not (hasattr(module, "get_rng_state") and hasattr(module, "set_rng_state")):
            continue
        index = tensor.device.index if tensor.device.index is not None else getattr(module, "current_device", lambda: 0)()
        return kind, [index], [module.get_rng_state(index)]
    return None, [], []


class _Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run, count, *args):
        ctx.run, ctx.inputs, ctx.params = run, list(args[:count]), list(args[count:])
        ctx.autocast_device, ctx.autocast_kwargs = _autocast_state(ctx.inputs)
        ctx.cpu_rng = torch.get_rng_state()
        ctx.rng_kind, ctx.rng_devices, ctx.rng_states = _device_rng(ctx.inputs)
        with torch.no_grad():
            return run(*ctx.inputs)

    @staticmethod
    def backward(ctx, *grads):
        inputs = [x.detach().requires_grad_(x.is_floating_point()) for x in ctx.inputs]
        autocast = (torch.autocast(device_type=ctx.autocast_device, **ctx.autocast_kwargs)
                    if ctx.autocast_device else contextlib.nullcontext())
        with torch.random.fork_rng(devices=ctx.rng_devices, device_type=ctx.rng_kind or "cpu"):
            torch.set_rng_state(ctx.cpu_rng)
            if ctx.rng_devices:
                module = getattr(torch, ctx.rng_kind)
                for index, state in zip(ctx.rng_devices, ctx.rng_states):
                    module.set_rng_state(state, index)
            with torch.enable_grad(), autocast:
                outputs = ctx.run(*[x.view_as(x) for x in inputs])
        wanted = [x for x in inputs if x.requires_grad] + [p for p in ctx.params if p.requires_grad]
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        pairs = [(o, g) for o, g in zip(outputs, grads) if o.requires_grad and g is not None]
        found = iter(torch.autograd.grad([o for o, _ in pairs], wanted, [g for _, g in pairs], allow_unused=True)) if pairs else iter(())
        input_grads = tuple(next(found, None) if x.requires_grad else None for x in inputs)
        param_grads = tuple(next(found, None) if p.requires_grad else None for p in ctx.params)
        del ctx.inputs, ctx.params
        return (None, None) + input_grads + param_grads


class ActivationCheckpointer:
    """Callable with the ``(fn, inputs, params)`` shape the UNet's ``set_checkpointer`` expects."""

    def __call__(self, fn: Callable[..., Any], inputs: tuple, params: Iterable[nn.Parameter]) -> Any:
        return _Recompute.apply(fn, len(inputs), *inputs, *tuple(params))
