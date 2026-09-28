"""Checks nodes.core.Node.NODE_KIND/NodePreset/list_presets() and their
introspection into server.nodegraph_introspect.NodeInfo.node_kind/presets
-- Phase 3 of docs/design/resources-controller/03-phase-3-interactive-node-support.md: the
suggestion-menu search resolution ("each preset counts as its own
searchable entry, matched on required inputs/outputs only").

Four kinds of check, deliberately kept separate: (1) the enforcement in
Node.__init_subclass__ -- a dynamic-kind class that doesn't override
list_presets(), or an invalid NODE_KIND, both fail loudly at class
*definition* time, not the first time something calls list_presets();
(2) a synthetic dynamic node's presets introspect correctly, required-
only, optional ports excluded; (3) a real multi-level-inheritance edge
case (an abstract intermediate class provides list_presets(), a
concrete subclass doesn't re-override it) -- the override check has to
resolve through the real MRO, not just check "did *this* class's own
__dict__ have it"; (4) every real node in the registry is internally
consistent -- static means presets is None, dynamic (only
ResourcesControllerNode today) means real non-empty presets. Check (4)
was originally written as "every registered node is still static"
before Phase 4/5 shipped the first dynamic registered node, which is
what made that older wording stale, not anything about the registry.
"""

import sys
from abc import abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.core import Node, NodePreset, Port
from server import nodegraph_registry
from server.nodegraph_introspect import introspect_node_class, node_info_to_dict


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_dynamic_without_override_raises_at_definition_time():
    print("[NODE_KIND='dynamic' without overriding list_presets() fails at class "
          "definition, not the first call to list_presets()]")
    try:
        class _BadNode(Node):
            NODE_KIND = "dynamic"
            OUTPUTS = {"x": Port(name="x", type=int)}

            def build(self, **inputs):
                return {"x": 1}
        raise AssertionError("expected TypeError at class definition")
    except TypeError as e:
        check("list_presets" in str(e), str(e))
        print(f"    PASS: {e}")


def check_invalid_node_kind_raises():
    print("[an invalid NODE_KIND value fails loudly at class definition]")
    try:
        class _BadNode2(Node):
            NODE_KIND = "sort-of-dynamic"
            OUTPUTS = {"x": Port(name="x", type=int)}

            def build(self, **inputs):
                return {"x": 1}
        raise AssertionError("expected TypeError at class definition")
    except TypeError as e:
        check("static' or 'dynamic'" in str(e), str(e))
        print(f"    PASS: {e}")


class _TwoPresetDynamicNode(Node):
    """Synthetic dynamic node: two presets, each with its own required
    input, and a shared common INPUTS/OUTPUTS."""
    NODE_KIND = "dynamic"
    INPUTS = {"common": Port(name="common", type=str, required=False, default="shared")}
    OUTPUTS = {"result": Port(name="result", type=int)}

    @classmethod
    def list_presets(cls):
        return [
            NodePreset(
                name="preset_a",
                required_inputs={"a_only": Port(name="a_only", type=str)},
                required_outputs={"result": Port(name="result", type=int)},
            ),
            NodePreset(
                name="preset_b",
                required_inputs={"b_only": Port(name="b_only", type=float)},
                required_outputs={"result": Port(name="result", type=int)},
            ),
        ]

    def build(self, **inputs):
        return {"result": 1}


def check_required_inputs_rejects_a_self_contradictory_optional_port():
    print("[NodePreset rejects a required=False Port inside required_inputs/"
          "required_outputs at construction time -- self-contradictory, since "
          "these dicts are specifically the required-only subset]")
    try:
        NodePreset(
            name="bad",
            required_inputs={"oops": Port(name="oops", type=str, required=False)},
            required_outputs={"result": Port(name="result", type=int)},
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        check("required=False" in str(e) and "required_inputs" in str(e), str(e))
        print(f"    PASS: {e}")


def check_dynamic_node_introspection():
    print("[a synthetic dynamic node's presets introspect correctly -- "
          "names, required inputs, outputs, per preset]")
    info = introspect_node_class(_TwoPresetDynamicNode)
    check(info.node_kind == "dynamic", info.node_kind)
    check(info.presets is not None and len(info.presets) == 2, info.presets)

    preset_a = next(p for p in info.presets if p.name == "preset_a")
    check([p.name for p in preset_a.required_inputs] == ["a_only"],
          [p.name for p in preset_a.required_inputs])
    check([p.name for p in preset_a.required_outputs] == ["result"],
          [p.name for p in preset_a.required_outputs])

    preset_b = next(p for p in info.presets if p.name == "preset_b")
    check([p.name for p in preset_b.required_inputs] == ["b_only"],
          [p.name for p in preset_b.required_inputs])
    check(preset_b.required_inputs[0].type_str == "float", preset_b.required_inputs[0].type_str)
    print("    PASS")


def check_node_info_to_dict_serializes_presets():
    print("[node_info_to_dict() carries node_kind/presets through correctly]")
    d = node_info_to_dict(introspect_node_class(_TwoPresetDynamicNode))
    check(d["node_kind"] == "dynamic", d["node_kind"])
    check(len(d["presets"]) == 2, d["presets"])
    names = {p["name"] for p in d["presets"]}
    check(names == {"preset_a", "preset_b"}, names)
    preset_a = next(p for p in d["presets"] if p["name"] == "preset_a")
    check([i["name"] for i in preset_a["required_inputs"]] == ["a_only"],
          preset_a["required_inputs"])
    check("required_outputs" in preset_a and len(preset_a["required_outputs"]) == 1,
          preset_a)

    # And a static node's dict has presets: null, not an empty list -- a real,
    # meaningful distinction (see NodeInfo.presets' own docstring: None means
    # "not applicable", [] would wrongly suggest "dynamic with zero presets").
    from nodes.train.schedule import CosineLRScheduleNode
    static_d = node_info_to_dict(introspect_node_class(CosineLRScheduleNode))
    check(static_d["node_kind"] == "static", static_d["node_kind"])
    check(static_d["presets"] is None, static_d["presets"])
    print("    PASS")


def check_multi_level_inheritance_resolves_through_the_real_mro():
    print("[an abstract intermediate class provides list_presets(); a concrete "
          "subclass that doesn't re-override it is still correctly satisfied -- "
          "the override check walks the real MRO, not just cls.__dict__]")

    class _AbstractDynamicBase(Node):
        NODE_KIND = "dynamic"
        OUTPUTS = {"x": Port(name="x", type=int)}

        @classmethod
        def list_presets(cls):
            return [NodePreset(name="only_preset", required_inputs={},
                                required_outputs={"x": Port(name="x", type=int)})]

        @abstractmethod
        def build(self, **inputs):
            ...

    # _AbstractDynamicBase itself is still abstract (unimplemented build()) --
    # exempt from __init_subclass__'s checks entirely, same as any other
    # abstract intermediate class in this project (e.g. OptimizerNode).
    class _ConcreteFromBase(_AbstractDynamicBase):
        def build(self, **inputs):
            return {"x": 1}

    info = introspect_node_class(_ConcreteFromBase)
    check(info.node_kind == "dynamic", info.node_kind)
    check(info.presets is not None and info.presets[0].name == "only_preset", info.presets)
    print("    PASS")


def check_every_real_registered_node_is_statically_consistent():
    print("[every node in the real registry is internally consistent: "
          "static nodes have presets=None, dynamic nodes introspect as "
          "dynamic with real (non-empty) presets]")
    # Written when NODE_KIND was purely additive and every registered node
    # was static; Phase 4/5 then shipped the first genuinely dynamic
    # registered node (ResourcesControllerNode, preset "lora_sdxl" --
    # docs/design/resources-controller/05-phase-5-resources-controller-node.md),
    # which made the old "everything must be static" assertion stale, not
    # the registry. The invariant worth checking is the pair: static means
    # presets is None, dynamic means presets actually exist (a dynamic node
    # with zero presets would be useless and is already rejected at class
    # definition time by Node.__init_subclass__, but re-checking here
    # proves it holds through introspection/serialization too).
    registry = nodegraph_registry.get_registry()
    check(len(registry) > 0, "registry unexpectedly empty")
    dynamic_seen = []
    for class_name, cls in registry.items():
        info = introspect_node_class(cls)
        if cls.NODE_KIND == "static":
            check(info.node_kind == "static", f"{class_name}: node_kind is {info.node_kind!r}")
            check(info.presets is None, f"{class_name}: presets is {info.presets!r}, expected None")
        else:
            check(cls.NODE_KIND == "dynamic", f"{class_name}: NODE_KIND is {cls.NODE_KIND!r}")
            check(info.node_kind == "dynamic", f"{class_name}: node_kind is {info.node_kind!r}")
            check(info.presets is not None and len(info.presets) > 0,
                  f"{class_name}: dynamic but presets is {info.presets!r}")
            dynamic_seen.append(class_name)
    print(f"    PASS ({len(registry)} registered nodes checked, "
          f"{len(dynamic_seen)} dynamic: {dynamic_seen or 'none'})")


def main():
    check_dynamic_without_override_raises_at_definition_time()
    check_invalid_node_kind_raises()
    check_required_inputs_rejects_a_self_contradictory_optional_port()
    check_dynamic_node_introspection()
    check_node_info_to_dict_serializes_presets()
    check_multi_level_inheritance_resolves_through_the_real_mro()
    check_every_real_registered_node_is_statically_consistent()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
