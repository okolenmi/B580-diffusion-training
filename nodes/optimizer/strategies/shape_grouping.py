"""build_shape_groups()/compute_grouped_updates(): the shared "batch the
core algorithm math across same-shape parameters" logic behind both
ShapeGroupedBatchStrategy (shape_grouped.py, per-member apply) and
ShapeGroupedForeachStrategy (shape_grouped_foreach.py, batched
torch._foreach_* apply) -- the two strategies differ only in how they
apply each group's already-computed (delta, decay) results, not in how
those results get computed. Extracted here once both needed the
identical grouping/computation logic, rather than the second copy of it
that used to be the plan -- see
docs/training_pipeline_design.md section 11.2 for the axis this
decomposition is built around, and docs/suspicious_findings.md's
`strategy_registry.py` entry for the real duplication bug (three copies
of a dict, one of which went stale) this project already paid for once
by not extracting shared structure proactively.
"""

from __future__ import annotations


def build_shape_groups(params, param_lr) -> list[list[int]]:
    """Groups parameter indices by exact (shape, dtype, device, lr).
    `lr` is part of the key, not just shape/dtype/device: two
    same-shaped parameters with genuinely different effective lr (via
    ParameterGroupPolicy, e.g. LoRAPlusGroups) must never be batched
    under one shared scalar lr, or one of them would silently get the
    wrong step size.

    Exact-shape grouping only, no padding of near-matching shapes --
    padding-based batching is a real extra correctness surface (masking,
    gradient leakage through pad regions) for a win exact-shape grouping
    already gets most of, since real UNet target modules repeat exact
    in/out-feature widths constantly (every q/k/v/out projection at a
    given attention width, across many blocks).

    Callers compute this once, lazily, on the first step() call --
    parameter shapes are stable for a LoRA's whole training run, no
    reason to recompute every step."""
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(params):
        key = (tuple(p.shape), p.dtype, p.device, param_lr[i])
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def compute_grouped_updates(algorithm, params, states, param_lr,
                             groups: list[list[int]]) -> list[tuple]:
    """A flat list of (param, delta, decay) triples, one per parameter
    that had a gradient this step -- computed via
    Algorithm.compute_update_batched() for any group with 2+ live
    members this step, or the plain per-member Algorithm.compute_update()
    for a singleton group (batching a group of one buys nothing, and
    compute_update_batched() isn't guaranteed to handle k=1 specially).

    Caller decides how to apply the results -- a per-member loop
    (strategies.base.apply_update(), what ShapeGroupedBatchStrategy
    does) or a batched torch._foreach_* pass
    (strategies.base.apply_updates_batched(), what
    ShapeGroupedForeachStrategy does). Deliberately returns data, not
    side effects, so it stays agnostic to that choice."""
    import torch

    entries = []
    for idx_group in groups:
        # Gradient accumulation / partial-graph edge case: a group's
        # membership (shape/dtype/device/lr) is fixed at construction,
        # but which members actually have a gradient *this specific
        # step* can vary -- filtered fresh every call.
        live = [i for i in idx_group if params[i].grad is not None]
        if not live:
            continue

        if len(live) == 1:
            i = live[0]
            grad = params[i].grad.detach().float()
            delta, decay = algorithm.compute_update(grad, params[i], states[i], param_lr[i])
            entries.append((params[i], delta, decay))
            continue

        grads = [params[i].grad.detach().float() for i in live]
        grad_stack = torch.stack(grads, dim=0)
        group_params = [params[i] for i in live]
        group_states = [states[i] for i in live]
        lr = param_lr[live[0]]  # identical within a group by construction (grouping key includes lr)
        delta_stack, decay = algorithm.compute_update_batched(
            grad_stack, group_params, group_states, lr)
        for j, i in enumerate(live):
            entries.append((params[i], delta_stack[j], decay))

    return entries
