"""Checks nodes.core.Port's new `choices` field and Node.validate_inputs()'s
enforcement of it -- docs/resources_controller_redesign_plan.md's
Consolidation section (11.4 / the ResourcePreset "parameter-value
dictionary", built as one generic Port mechanism rather than twice).

Four kinds of check: (1) Port.__post_init__ rejects every malformed
`choices` at construction time, not first use; (2) a well-formed choices
Port constructs and is immutable metadata, nothing more; (3)
Node.validate_inputs() accepts a valid explicit value, rejects an
invalid one, and never touches choices for a value that's simply absent
(missing-optional is a different, pre-existing check); (4) the two real
call sites this landed on -- ComposedAdamWOptimizerNode's `strategy` and
RenoiseBatchSourceNode/ManagedDatasetSourceNode's `t_mode` -- genuinely
read their choices from the same shared registries their doc strings
already cited (STRATEGIES, T_MODES), not a hand-copied second list, and
`device` (deliberately NOT given choices -- torch.device-parseable,
including indexed variants like "xpu:0", genuinely open-ended) stays
None.
"""

import sys
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.core import Node, Port


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_malformed_choices_rejected_at_construction():
    print("[Port(choices=...) rejects every malformed case at construction, not first use]")
    for kwargs, label in [
        (dict(type=int, choices=("a", "b")), "non-str type"),
        (dict(type=str, choices=()), "empty tuple"),
        (dict(type=str, choices=["a", "b"]), "list instead of tuple"),
        (dict(type=str, choices=(1, 2)), "non-str entries"),
        (dict(type=str, default="x", choices=("a", "b")), "default not in choices"),
    ]:
        try:
            Port(name="p", required=False, **kwargs)
            raise AssertionError(f"expected Port construction to reject: {label}")
        except (TypeError, ValueError):
            pass
    print("    PASS (5 malformed cases all rejected)")


def check_well_formed_choices_port_constructs():
    print("[A well-formed choices Port constructs and reports back its own data]")
    p = Port(name="mode", type=str, required=False, default="a", choices=("a", "b", "c"))
    check(p.choices == ("a", "b", "c"), f"choices not preserved: {p.choices!r}")
    # None (the default) stays the default -- a Port with no choices= at
    # all is untouched by any of the new validation.
    p2 = Port(name="free", type=str, required=False, default="anything")
    check(p2.choices is None, "a Port with no choices= should default to None")
    print("    PASS")


class _ChoicesTestNode(Node):
    """Synthetic, smoke-test-only: one required, one optional choices Port."""
    INPUTS: ClassVar[dict] = {
        "mode": Port(name="mode", type=str, required=True, choices=("uniform", "low")),
        "extra": Port(name="extra", type=str, required=False, default="x", choices=("x", "y")),
    }
    OUTPUTS: ClassVar[dict] = {"out": Port(name="out", type=str)}

    def build(self, **inputs):
        self.validate_inputs(inputs)
        return {"out": inputs["mode"]}


def check_validate_inputs_enforces_choices():
    print("[Node.validate_inputs() enforces choices: valid passes, invalid raises, "
          "missing-optional is untouched]")
    node = _ChoicesTestNode()
    node.validate_inputs({"mode": "uniform"})  # valid, required, no error
    node.validate_inputs({"mode": "uniform", "extra": "y"})  # valid optional
    node.validate_inputs({"mode": "uniform"})  # extra absent entirely -- fine, not "invalid"
    try:
        node.validate_inputs({"mode": "not_a_real_mode"})
        raise AssertionError("expected an out-of-choices 'mode' to raise")
    except ValueError as e:
        check("not_a_real_mode" in str(e), f"error message doesn't name the bad value: {e}")
    try:
        node.validate_inputs({"mode": "uniform", "extra": "not_x_or_y"})
        raise AssertionError("expected an out-of-choices 'extra' to raise")
    except ValueError:
        pass
    try:
        node.validate_inputs({})
        raise AssertionError("expected the missing required 'mode' to still raise, unrelated to choices")
    except ValueError:
        pass
    print("    PASS")


def check_real_strategy_ports_share_strategy_registry():
    print("[Real 'strategy' Ports read choices from the same STRATEGIES registry "
          "their doc string already cited, not a second hand-copied list]")
    from nodes.optimizer.strategy_registry import STRATEGIES
    from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
    from nodes.optimizer.composed_came import ComposedCAMEOptimizerNode
    from nodes.optimizer.composed_adafactor import ComposedAdafactorOptimizerNode
    expected = tuple(STRATEGIES)
    for cls in (ComposedAdamWOptimizerNode, ComposedCAMEOptimizerNode, ComposedAdafactorOptimizerNode):
        got = cls.INPUTS["strategy"].choices
        check(got == expected, f"{cls.__name__}.INPUTS['strategy'].choices = {got!r}, expected {expected!r}")
        got_device = cls.INPUTS["device"].choices
        check(got_device is None, f"{cls.__name__}.INPUTS['device'].choices should stay None (open-ended), got {got_device!r}")
    print(f"    PASS ({len(expected)} strategies: {expected})")


def check_real_t_mode_ports_share_timestep_modes():
    print("[Real 't_mode' Ports read choices from the same T_MODES constant "
          "their doc string already described]")
    from nodes.dataset.timestep_modes import T_MODES
    from nodes.dataset.renoise import RenoiseBatchSourceNode
    from nodes.dataset.managed import ManagedDatasetSourceNode
    for cls in (RenoiseBatchSourceNode, ManagedDatasetSourceNode):
        got = cls.INPUTS["t_mode"].choices
        check(got == T_MODES, f"{cls.__name__}.INPUTS['t_mode'].choices = {got!r}, expected {T_MODES!r}")
    print(f"    PASS ({T_MODES})")


def main():
    check_malformed_choices_rejected_at_construction()
    check_well_formed_choices_port_constructs()
    check_validate_inputs_enforces_choices()
    check_real_strategy_ports_share_strategy_registry()
    check_real_t_mode_ports_share_timestep_modes()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
