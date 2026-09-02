"""Checks server/nodegraph_introspect.py's display_name (design doc
section 11.5: class_name/__name__ stays the stable registry key a saved
graph resolves against; display_name is a second, independent string for
the palette label, auto-derived or overridden via Node.DISPLAY_NAME).

Three kinds of check, deliberately kept separate: (1) _auto_display_name()
pinned against real class names actually in nodes/ today -- this is the
part with real logic (a curated domain-token list, see that function's
own docstring for why a plain capital-boundary split isn't enough) and
the part most likely to silently regress; (2) the DISPLAY_NAME override
path, via synthetic classes so it doesn't depend on any real class
happening to set one; (3) every real node in the registry, to catch a
crash or an empty label on the actual, full set -- not just the
hand-picked examples in (1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.core import Node, Port
from server import nodegraph_registry
from server.nodegraph_introspect import (
    _auto_display_name, introspect_legacy_class, introspect_node_class,
    node_info_to_dict,
)


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


# Real class names from server/nodegraph_registry.py's _load(), paired with
# the label a person reading the palette should actually see. Chosen to
# cover every domain token _KNOWN_DISPLAY_TOKENS knows about (UNet, LoRA,
# DoRA, CAME, AdamW, SDXL, SNR, LR, P2), not just the design doc's one
# worked example.
_EXPECTED = {
    "ComfyUNetLoRANode": "Comfy UNet LoRA",
    "ComposedCAMEOptimizerNode": "Composed CAME Optimizer",
    "SimpleAdamWOptimizerNode": "Simple AdamW Optimizer",
    "SDXLTextEncoderNode": "SDXL Text Encoder",
    "LoRACheckpointLoaderNode": "LoRA Checkpoint Loader",
    "MinSNRLossWeightingNode": "Min SNR Loss Weighting",
    "CosineLRScheduleNode": "Cosine LR Schedule",
    "P2LossWeightingNode": "P2 Loss Weighting",
    "ManagedDatasetSourceNode": "Managed Dataset Source",
    # No trailing "Node" to strip -- shouldn't crash or eat a real letter.
    "Node": "Node",
    # Not ending in "Node" at all -- passed through word-by-word untouched.
    "AdamW": "AdamW",
}


def check_auto_display_name_matches_expected_examples():
    print("[_auto_display_name() matches expected labels for real class names]")
    for class_name, expected in _EXPECTED.items():
        got = _auto_display_name(class_name)
        check(got == expected, f"{class_name}: got {got!r}, expected {expected!r}")
    print("    PASS")


def check_display_name_override_wins_over_auto_derivation():
    print("[Node.DISPLAY_NAME, when set, is used verbatim instead of auto-derivation]")

    class WeirdlyNamedButOverriddenNode(Node):
        DISPLAY_NAME = "My Friendly Label"
        INPUTS: dict = {}
        OUTPUTS = {"value": Port(name="value", type=int)}

        def build(self, **inputs):
            return {"value": 1}

    info = introspect_node_class(WeirdlyNamedButOverriddenNode)
    check(info.display_name == "My Friendly Label", info.display_name)
    check(info.class_name == "WeirdlyNamedButOverriddenNode",
          "class_name must stay the real __name__ regardless of DISPLAY_NAME")
    print("    PASS")


def check_display_name_defaults_to_none_and_auto_derives():
    print("[a Node subclass that doesn't set DISPLAY_NAME still gets an auto-derived one]")

    class SomeGenericThingNode(Node):
        INPUTS: dict = {}
        OUTPUTS = {"value": Port(name="value", type=int)}

        def build(self, **inputs):
            return {"value": 1}

    check(SomeGenericThingNode.DISPLAY_NAME is None,
          "the base class default should still be None until overridden")
    info = introspect_node_class(SomeGenericThingNode)
    check(info.display_name == "Some Generic Thing", info.display_name)
    print("    PASS")


def check_legacy_class_without_display_name_attr_does_not_crash():
    print("[introspect_legacy_class() on a plain class with no DISPLAY_NAME at all]")

    class PlainOldClass:
        """A class that was never designed with a node interface in mind."""
        def __init__(self, x: int = 1):
            self.x = x

    info = introspect_legacy_class(PlainOldClass)
    check(info.display_name == "Plain Old Class", info.display_name)
    print("    PASS")


def check_node_info_to_dict_includes_display_name():
    print("[node_info_to_dict() carries display_name through to the JSON-facing dict]")
    from nodes.train.schedule import CosineLRScheduleNode
    d = node_info_to_dict(introspect_node_class(CosineLRScheduleNode))
    check(d["display_name"] == "Cosine LR Schedule", d["display_name"])
    check(d["class_name"] == "CosineLRScheduleNode", d["class_name"])
    print("    PASS")


def check_every_real_registered_node_gets_a_sane_display_name():
    print("[every node in the real registry gets a non-empty, whitespace-clean display_name]")
    registry = nodegraph_registry.get_registry()
    check(len(registry) > 0, "registry unexpectedly empty -- nothing to check")
    for class_name, cls in registry.items():
        info = introspect_node_class(cls)
        check(info.display_name, f"{class_name}: empty display_name")
        check(info.display_name == info.display_name.strip(),
              f"{class_name}: display_name has leading/trailing whitespace: {info.display_name!r}")
        check("  " not in info.display_name,
              f"{class_name}: display_name has a doubled space: {info.display_name!r}")
        if class_name.endswith("Node") and len(class_name) > 4 and cls.DISPLAY_NAME is None:
            check(not info.display_name.endswith("Node"),
                  f"{class_name}: auto-derived display_name still carries the "
                  f"meaningless trailing 'Node' suffix it's supposed to strip: "
                  f"{info.display_name!r}")
    print(f"    PASS ({len(registry)} registered nodes checked)")


def check_port_choices_propagate_and_stay_none_elsewhere():
    print("[introspect_node_class()/node_info_to_dict() carry Port.choices through "
          "as a JSON-friendly list, and stay None on a Port that never declared one]")
    from nodes.optimizer.composed_adamw import ComposedAdamWOptimizerNode
    d = node_info_to_dict(introspect_node_class(ComposedAdamWOptimizerNode))
    by_name = {p["name"]: p for p in d["inputs"]}
    check(isinstance(by_name["strategy"]["choices"], list),
          f"choices should serialize as a list, got {type(by_name['strategy']['choices'])}")
    check(set(by_name["strategy"]["choices"]) ==
          {"simple", "chunked", "foreach", "shape_grouped", "shape_grouped_foreach"},
          f"unexpected strategy choices: {by_name['strategy']['choices']!r}")
    check(by_name["device"]["choices"] is None,
          "device is deliberately open-ended (torch.device-parseable, e.g. 'xpu:0') "
          "-- should introspect with choices=None, not an invented closed list")
    check(by_name["betas"]["choices"] is None,
          "a Port that never set choices= should introspect with choices=None")
    print("    PASS")


def main():
    check_auto_display_name_matches_expected_examples()
    check_display_name_override_wins_over_auto_derivation()
    check_display_name_defaults_to_none_and_auto_derives()
    check_legacy_class_without_display_name_attr_does_not_crash()
    check_node_info_to_dict_includes_display_name()
    check_every_real_registered_node_gets_a_sane_display_name()
    check_port_choices_propagate_and_stay_none_elsewhere()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
