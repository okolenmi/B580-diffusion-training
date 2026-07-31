"""Real torch. Targeted at SimpleAdamWOptimizerHandle -- new logic with
real risk (the state-tensor iteration used by offload/reload/decay/reset),
not covered by anything existing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.model.handle import ParameterList
from nodes.optimizer.adamw import SimpleAdamWOptimizerHandle, SimpleAdamWOptimizerNode


def check_contracts():
    print("[contracts]")
    assert not getattr(SimpleAdamWOptimizerNode, "__abstractmethods__", None)
    print("    PASS")


def _make_handle():
    p = torch.nn.Parameter(torch.randn(4, 4))
    node = SimpleAdamWOptimizerNode()
    result = node.build(params=ParameterList([p]), lr=0.1)
    return result["optimizer"], p


def check_step_actually_updates_and_matches_plain_torch_adamw():
    print("[step() matches an independent plain torch.optim.AdamW on the same data]")
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(4, 4))
    p2 = torch.nn.Parameter(p1.detach().clone())

    handle = SimpleAdamWOptimizerHandle(
        torch.optim.AdamW([p1], lr=0.1, foreach=True), device=p1.device)
    reference = torch.optim.AdamW([p2], lr=0.1)

    for _ in range(3):
        loss1 = (p1 * p1).sum()
        loss2 = (p2 * p2).sum()
        handle.zero_grad()
        reference.zero_grad()
        loss1.backward()
        loss2.backward()
        handle.step()
        reference.step()

    torch.testing.assert_close(p1, p2)
    print("    PASS")


def check_offload_and_reload_round_trip():
    print("[offload_states_to_cpu / reload_states_to_device round trip]")
    handle, p = _make_handle()
    x = torch.randn(4, 4)
    (p * x).sum().backward()
    handle.step()
    before = {k: v.clone() for st in handle._legacy.state.values()
              for k, v in st.items() if torch.is_tensor(v)}

    handle.offload_states_to_cpu()
    for state in handle._legacy.state.values():
        for key, val in state.items():
            if key != "step" and torch.is_tensor(val):
                assert val.device.type == "cpu"

    handle.reload_states_to_device()
    for state in handle._legacy.state.values():
        for key, val in state.items():
            if key != "step" and torch.is_tensor(val):
                assert val.device == p.device

    after = {k: v for st in handle._legacy.state.values()
             for k, v in st.items() if torch.is_tensor(v)}
    for k in before:
        torch.testing.assert_close(before[k], after[k])
    print("    PASS: values survive the round trip unchanged")


def check_decay_and_reset():
    print("[decay_states / reset_states]")
    handle, p = _make_handle()
    (p * torch.randn(4, 4)).sum().backward()
    handle.step()

    handle.decay_states(0.5)
    for state in handle._legacy.state.values():
        for key, val in state.items():
            if key != "step" and torch.is_tensor(val):
                assert not torch.all(val == 0), "decay(0.5) should not zero everything"

    handle.reset_states()
    for state in handle._legacy.state.values():
        for key, val in state.items():
            if key != "step" and torch.is_tensor(val):
                assert torch.all(val == 0)
    print("    PASS")


def check_free_states():
    print("[free_states clears optimizer state entirely]")
    handle, p = _make_handle()
    (p * torch.randn(4, 4)).sum().backward()
    handle.step()
    assert len(handle._legacy.state) > 0
    handle.free_states()
    assert len(handle._legacy.state) == 0
    print("    PASS")


def main():
    check_contracts()
    check_step_actually_updates_and_matches_plain_torch_adamw()
    check_offload_and_reload_round_trip()
    check_decay_and_reset()
    check_free_states()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
