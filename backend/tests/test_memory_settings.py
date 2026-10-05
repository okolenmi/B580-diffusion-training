"""Tests for MemorySettings and effective_memory.

The graph is the configurable object: each graph holds its own budget,
allocates only from the unallocated pool, and if it doesn't fit the run
cannot start. Format 1 graphs load with defaults; format 2 carries
settings.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.domain.graph import GraphDefinition, GraphNodeSpec
from backend.domain.memory_settings import (
    AUTO,
    EffectiveMemory,
    MemorySettings,
    effective_memory,
)


def _graph(*nodes: GraphNodeSpec) -> GraphDefinition:
    return GraphDefinition(nodes=nodes, edges=())


def _node(class_name: str, **params) -> GraphNodeSpec:
    return GraphNodeSpec(id="n1", class_name=class_name, params=params)


# -- MemorySettings ---------------------------------------------------------


def test_memory_settings_defaults():
    """Default settings are sensible."""
    s = MemorySettings()
    assert s.vram_min_mb == 0.0
    assert s.vram_max_mb == AUTO
    assert s.strict is True
    assert s.policy == "demand_driven"
    assert s.ram_max_mb == AUTO


def test_memory_settings_as_dict():
    """Round-trips through as_dict."""
    s = MemorySettings(vram_min_mb=1000.0, vram_max_mb=8000.0, strict=False)
    d = s.as_dict()
    assert d["vram_min_mb"] == 1000.0
    assert d["vram_max_mb"] == 8000.0
    assert d["strict"] is False


def test_memory_settings_from_dict():
    """Parses from a dict."""
    raw = {"vram_min_mb": 2000.0, "vram_max_mb": 9000.0, "strict": False}
    s = MemorySettings.from_dict(raw)
    assert s.vram_min_mb == 2000.0
    assert s.vram_max_mb == 9000.0
    assert s.strict is False


def test_memory_settings_from_none():
    """None -> defaults."""
    s = MemorySettings.from_dict(None)
    assert s.vram_min_mb == 0.0
    assert s.vram_max_mb == AUTO


def test_device_is_carried_validated_and_not_overridable():
    """MEM-05 #2: the graph file names the device the child checks."""
    from backend.application.errors import InvalidQueryError
    from backend.application.use_cases.save_graph import SaveGraph
    from backend.domain.memory_settings import _with_overrides
    from backend.presentation.schemas import GraphRunIn

    # Default, tolerant decode, round trip: graphs saved before MEM-05
    # have no device key and get the default like every other missing
    # field.
    assert MemorySettings().device == "xpu"
    assert MemorySettings().as_dict()["device"] == "xpu"
    assert MemorySettings.from_dict({}).device == "xpu"
    assert MemorySettings.from_dict({"device": "cuda"}).device == "cuda"
    again = MemorySettings.from_dict(MemorySettings(device="cuda:1").as_dict())
    assert again.device == "cuda:1"

    # The domain refuses what a device name cannot be, by name.
    for bad in ("", "   ", 123, True, None):
        try:
            MemorySettings(device=bad)
        except ValueError as exc:
            assert "device" in str(exc), (bad, exc)
        else:
            raise AssertionError(f"device={bad!r} was accepted")

    # The run edge: carried into the definition the endpoint builds,
    # and an empty name is a refusal here so it cannot escape the
    # schema as a domain ValueError (a 500 instead of a 422).
    body = GraphRunIn(nodes=[], memory={"device": "cuda"})
    assert body.to_definition().memory.device == "cuda"
    try:
        GraphRunIn(nodes=[], memory={"device": " "})
    except Exception as exc:
        assert "device" in str(exc), str(exc)
    else:
        raise AssertionError("an empty device name was accepted at the edge")

    # Overrides are the five known keys; device rides the graph, not
    # the request: the child runs the device the graph *file* names,
    # and an override would live only on the row it never sees.
    try:
        GraphRunIn(nodes=[], memory_overrides={"device": "cpu"})
    except Exception as exc:
        assert "device" in str(exc), str(exc)
    else:
        raise AssertionError("device was overridable per-request")
    merged = _with_overrides(
        MemorySettings(device="cuda"), {"device": "cpu", "strict": False}
    )
    assert merged.device == "cuda", "the domain's merge drops device too"
    assert merged.strict is False, "the five known keys still override"

    # The save edge: device is a known setting, and its values are
    # checked like every budget before the block is stored verbatim.
    kept = SaveGraph._payload(
        {"nodes": [], "edges": [], "memory": {"device": "cuda"}}
    )
    assert kept["memory"] == {"device": "cuda"}
    try:
        SaveGraph._payload(
            {"nodes": [], "edges": [], "memory": {"device": ""}}
        )
    except InvalidQueryError as exc:
        assert "device" in str(exc), str(exc)
    else:
        raise AssertionError("an empty device name was saved")


# -- Graph format round-trip -------------------------------------------------


def test_graph_format_is_2():
    """Graph format is now 2."""
    assert GraphDefinition().as_dict()["format"] == 2


def test_graph_as_dict_has_memory():
    """as_dict includes the memory key."""
    g = _graph(_node("ManagedLoRATrainerNode", model="sdxl", batch_size=2))
    d = g.as_dict()
    assert "memory" in d
    assert d["memory"]["vram_max_mb"] == AUTO


def test_graph_from_dict_round_trip():
    """from_dict -> as_dict preserves nodes and edges."""
    g = _graph(_node("ManagedLoRATrainerNode", model="sdxl", batch_size=2))
    d = g.as_dict()
    g2 = GraphDefinition.from_dict(d)
    assert len(g2.nodes) == 1
    assert g2.nodes[0].class_name == "ManagedLoRATrainerNode"
    assert g2.nodes[0].params["model"] == "sdxl"


def test_graph_from_dict_format_1_loads():
    """A format 1 graph (no memory key) loads with defaults, and
    re-saves unchanged except for the format number."""
    payload = {
        "format": 1,
        "nodes": [{"id": "n1", "class_name": "SomeNode", "params": {}}],
        "edges": [],
    }
    g = GraphDefinition.from_dict(payload)
    assert len(g.nodes) == 1
    assert g.nodes[0].class_name == "SomeNode"
    assert g.memory == MemorySettings()
    again = g.as_dict()
    assert again["format"] == 2
    assert again["nodes"] == payload["nodes"]
    assert again["edges"] == payload["edges"]
    assert again["memory"] == MemorySettings().as_dict()


def test_graph_from_dict_ignores_unknown_keys():
    """Unknown keys are ignored (tolerant by design)."""
    payload = {
        "format": 2,
        "nodes": [],
        "edges": [],
        "unknown_key": "ignored",
    }
    g = GraphDefinition.from_dict(payload)
    assert len(g.nodes) == 0


def test_graph_non_default_memory_round_trips():
    """Settings the graph was saved with survive storage, not just
    defaults -- the case every other round-trip test here missed."""
    g = GraphDefinition(
        memory=MemorySettings(
            vram_min_mb=1024.0,
            vram_max_mb=6144.0,
            strict=False,
            ram_max_mb=4096.0,
        ),
    )
    g2 = GraphDefinition.from_dict(g.as_dict())
    assert g2.memory == g.memory


# -- API submission ---------------------------------------------------------


def test_run_in_memory_reaches_definition():
    """A submitted `memory` block becomes the definition's settings."""
    from backend.presentation.schemas import GraphRunIn

    body = GraphRunIn(
        nodes=[],
        memory={"vram_min_mb": 512.0, "vram_max_mb": 4096.0, "strict": False},
    )
    definition = body.to_definition()
    assert definition.memory.vram_min_mb == 512.0
    assert definition.memory.vram_max_mb == 4096.0
    assert definition.memory.strict is False
    absent = GraphRunIn(nodes=[]).to_definition()
    assert absent.memory == MemorySettings()


def test_run_in_rejects_unknown_memory_setting():
    """An unknown setting is a 422 naming it, not silently defaulted."""
    from backend.presentation.schemas import GraphRunIn

    try:
        GraphRunIn(nodes=[], memory={"banana": 1})
    except Exception as exc:  # the rejection message must carry the key
        assert "banana" in str(exc), str(exc)
    else:
        raise AssertionError("an unknown memory setting was accepted")


def test_run_in_rejects_prose_for_auto():
    """`vram_max_mb` is a number or "auto" -- nothing else."""
    from backend.presentation.schemas import GraphRunIn

    try:
        GraphRunIn(nodes=[], memory={"vram_max_mb": "everything"})
    except Exception as exc:  # the rejection message must carry the value
        assert "everything" in str(exc), str(exc)
    else:
        raise AssertionError("vram_max_mb='everything' was accepted")


def test_save_graph_rejects_unknown_memory_setting():
    """The library save path rejects an unknown setting by name too.

    Saving stores the payload verbatim (forward compatibility), so this
    is the only moment a bad memory block can be refused with a clear
    message instead of being silently defaulted at run time.
    """
    from backend.application.errors import InvalidQueryError
    from backend.application.use_cases.save_graph import SaveGraph

    kept = SaveGraph._payload({
        "nodes": [],
        "edges": [],
        "memory": {"vram_max_mb": 4096.0},
    })
    assert kept["memory"] == {"vram_max_mb": 4096.0}
    try:
        SaveGraph._payload({
            "nodes": [],
            "edges": [],
            "memory": {"vram_max_mb": 4096.0, "banana": 1},
        })
    except InvalidQueryError as exc:
        assert "banana" in str(exc), str(exc)
    else:
        raise AssertionError("an unknown memory setting was saved")


# -- effective_memory ---------------------------------------------------------


def test_effective_memory_stated_beats_observed():
    """Stated beats observed (held = max of the two)."""
    settings = MemorySettings(vram_max_mb=8000.0)
    peak_record = {"fp_key": 7000.0}
    result = effective_memory(
        settings, None, peak_record, "fp_key", 12000.0
    )
    assert isinstance(result, EffectiveMemory)
    assert result.demand_mb == 8000.0  # stated wins
    assert result.demand_source == "stated"


def test_effective_memory_observed_when_higher():
    """Observed wins when it's higher than stated (with pillow)."""
    settings = MemorySettings(vram_max_mb=7000.0)
    peak_record = {"fp_key": 8000.0}
    result = effective_memory(
        settings, None, peak_record, "fp_key", 12000.0
    )
    assert result.demand_mb == 8150.0  # 8000 + 150 pillow
    assert result.demand_source == "observed"


def test_effective_memory_unknown_uses_capacity():
    """Unknown demand uses capacity as exploratory exclusive claim."""
    settings = MemorySettings(vram_max_mb=AUTO)
    result = effective_memory(
        settings, None, None, None, 12000.0
    )
    assert result.demand_mb == 12000.0
    assert result.demand_source == "unknown"


def test_effective_memory_overrides_win():
    """Request overrides win over graph settings."""
    settings = MemorySettings(vram_max_mb=8000.0, strict=True)
    overrides = {"vram_max_mb": 9000.0, "strict": False}
    result = effective_memory(
        settings, overrides, None, None, 12000.0
    )
    assert result.vram_max_mb == 9000.0
    assert result.strict is False


def test_effective_memory_no_overrides():
    """No overrides -> graph settings pass through."""
    settings = MemorySettings(vram_min_mb=1000.0, vram_max_mb=8000.0)
    result = effective_memory(
        settings, None, None, None, 12000.0
    )
    assert result.vram_min_mb == 1000.0
    assert result.vram_max_mb == 8000.0


def test_effective_memory_pillow_added_to_observed():
    """Observed peak gets the pillow added."""
    settings = MemorySettings(vram_max_mb=AUTO)
    peak_record = {"fp_key": 7000.0}
    result = effective_memory(
        settings, None, peak_record, "fp_key", 12000.0,
        pillow_mb=200.0,
    )
    assert result.demand_mb == 7200.0  # 7000 + 200
    assert result.demand_source == "observed"


def test_effective_memory_fingerprint_not_in_record():
    """Fingerprint not in record -> unknown."""
    settings = MemorySettings(vram_max_mb=AUTO)
    peak_record = {"other_key": 5000.0}
    result = effective_memory(
        settings, None, peak_record, "fp_key", 12000.0
    )
    assert result.demand_mb == 12000.0
    assert result.demand_source == "unknown"


def test_effective_memory_unknown_without_capacity_is_none():
    """No probe reading, unknown demand -> demand stays None, not zero.

    Unknown is never a zero claim: nothing was measured, so nothing is
    claimed. Admission (MEM-03) needs a real number, and a missing
    capacity reading is not one.
    """
    result = effective_memory(MemorySettings(), None, None, None, None)
    assert result.demand_mb is None
    assert result.demand_source == "unknown"


def test_effective_memory_round_trips_storage():
    """as_dict -> from_dict reproduces the admitted values exactly.

    What a restart reads back from ``memory_json`` must equal what was
    stored, including a null demand.
    """
    admitted = effective_memory(
        MemorySettings(vram_max_mb=6000.0), {"strict": False}, None, None, 12216.0
    )
    assert EffectiveMemory.from_dict(admitted.as_dict()) == admitted
    unknown = effective_memory(MemorySettings(), None, None, None, None)
    raw = unknown.as_dict()
    assert raw["demand_mb"] is None
    assert EffectiveMemory.from_dict(raw) == unknown


def test_run_in_overrides_keep_only_the_named_keys():
    """Partial overrides: absent keys mean 'keep the graph's value'."""
    from backend.presentation.schemas import GraphRunIn

    body = GraphRunIn(
        nodes=[], memory_overrides={"vram_max_mb": 4096.0, "strict": False}
    )
    assert body.memory_overrides is not None
    assert body.memory_overrides.as_overrides() == {
        "vram_max_mb": 4096.0,
        "strict": False,
    }
    auto = GraphRunIn(nodes=[], memory_overrides={"ram_max_mb": AUTO})
    assert auto.memory_overrides is not None
    assert auto.memory_overrides.as_overrides() == {"ram_max_mb": AUTO}
    # No memory_overrides at all -> the request overrides nothing.
    assert GraphRunIn(nodes=[]).memory_overrides is None


def test_run_in_rejects_unknown_override_key():
    """A typo in an override is refused, not silently 'keep the default'."""
    from backend.presentation.schemas import GraphRunIn

    try:
        GraphRunIn(nodes=[], memory_overrides={"vram_max": 4096.0})
    except Exception as exc:  # the rejection must name the offending key
        assert "vram_max" in str(exc), str(exc)
    else:
        raise AssertionError("an unknown memory override key was accepted")


# -- MEM-03H-01: unusable numbers -------------------------------------------
#
# The seeds came from the reproduction: through the real endpoint,
# memory={"vram_max_mb": -9000} was a 201 and the ledger then reported
# 19,592 MB free on an 11,192 MB card. Three independent layers each
# hold on their own -- API 422 here, domain ValueError below, and the
# ledger's own refusal in test_memory_ledger.py.

#: Every value that must never become a budget, on a ceiling field.
#: The last one is an int past the float range: it used to escape as an
#: OverflowError -- a 500 -- from every layer but the ledger.
_BUDGET_SEEDS = (
    -1, 0, float("nan"), float("inf"), float("-inf"), True, "lots", 10**400,
)
#: On the *floor*: zero means "no floor" and is the field's own default,
#: so it stays valid everywhere it means that.
_MIN_SEEDS = (
    -1, float("nan"), float("inf"), float("-inf"), True, "lots", 10**400,
)


def test_run_in_rejects_unusable_budgets():
    """API layer: each unusable value is a 422 naming the field."""
    from backend.presentation.schemas import GraphRunIn, MemoryOverridesIn

    for field, seeds in (
        ("vram_min_mb", _MIN_SEEDS),
        ("vram_max_mb", _BUDGET_SEEDS),
        ("ram_max_mb", _BUDGET_SEEDS),
    ):
        for value in seeds:
            try:
                GraphRunIn(nodes=[], memory={field: value})
            except ValueError as exc:  # pydantic's ValidationError is one
                assert field in str(exc), (field, value, exc)
            else:
                raise AssertionError(f"memory.{field}={value!r} was accepted")
    # The override path is the same endpoint: the softer way in is not
    # a way in.
    for value in _BUDGET_SEEDS:
        try:
            GraphRunIn(nodes=[], memory_overrides={"vram_max_mb": value})
        except ValueError as exc:
            assert "vram_max_mb" in str(exc), (value, exc)
        else:
            raise AssertionError(f"overrides vram_max_mb={value!r} was accepted")
    try:
        MemoryOverridesIn(vram_min_mb=-1)
    except ValueError as exc:
        assert "vram_min_mb" in str(exc), exc
    else:
        raise AssertionError("overrides vram_min_mb=-1 was accepted")
    # A zero floor is the default and stays valid.
    body = GraphRunIn(nodes=[], memory={"vram_min_mb": 0})
    assert body.memory is not None and body.memory.vram_min_mb == 0


def test_run_in_rejects_inverted_ranges():
    """min > max is refused wherever the two halves meet."""
    from backend.presentation.schemas import GraphRunIn, MemoryOverridesIn

    try:
        GraphRunIn(nodes=[], memory={"vram_min_mb": 5000, "vram_max_mb": 4096})
    except ValueError as exc:
        assert "5000" in str(exc) and "4096" in str(exc), str(exc)
    else:
        raise AssertionError("an inverted memory block was accepted")
    try:
        MemoryOverridesIn(vram_min_mb=5000, vram_max_mb=4096)
    except ValueError as exc:
        assert "5000" in str(exc) and "4096" in str(exc), str(exc)
    else:
        raise AssertionError("an inverted override pair was accepted")
    # The merged view: an override that lifts the floor above the
    # graph's ceiling would reach the domain and die there as a 500.
    try:
        GraphRunIn(
            nodes=[],
            memory={"vram_max_mb": 4096},
            memory_overrides={"vram_min_mb": 5000},
        )
    except ValueError as exc:
        assert "overrides applied" in str(exc), str(exc)
    else:
        raise AssertionError("an override above the graph's ceiling was accepted")
    # Ordered combinations still pass, including equal ends and an
    # "auto" ceiling (nothing numeric to order against).
    GraphRunIn(nodes=[], memory={"vram_min_mb": 4096, "vram_max_mb": 5000})
    GraphRunIn(nodes=[], memory={"vram_min_mb": 5000, "vram_max_mb": 5000})
    GraphRunIn(
        nodes=[],
        memory={"vram_max_mb": "auto"},
        memory_overrides={"vram_min_mb": 9000},
    )


def test_domain_rejects_unusable_budgets():
    """Domain layer: the same conditions raise ValueError -- straight
    through the constructor and out of ``from_dict``, so a stored or
    hand-edited graph file cannot smuggle one past the API."""
    from backend.domain.memory_settings import MemorySettings

    for field, seeds in (
        ("vram_min_mb", _MIN_SEEDS),
        ("vram_max_mb", _BUDGET_SEEDS),
        ("ram_max_mb", _BUDGET_SEEDS),
    ):
        for value in seeds:
            for build in (
                lambda f=field, v=value: MemorySettings(**{f: v}),
                lambda f=field, v=value: MemorySettings.from_dict({f: v}),
            ):
                try:
                    build()
                except ValueError as exc:
                    assert field in str(exc), (field, value, exc)
                else:
                    raise AssertionError(
                        f"{field}={value!r} was accepted (direct or from_dict)"
                    )
    assert MemorySettings(vram_min_mb=0).vram_min_mb == 0


def test_domain_rejects_inverted_range():
    """The ordering rule holds for constructed and stored settings."""
    from backend.domain.memory_settings import MemorySettings

    for build in (
        lambda: MemorySettings(vram_min_mb=5000, vram_max_mb=4096),
        lambda: MemorySettings.from_dict(
            {"vram_min_mb": 5000, "vram_max_mb": 4096}
        ),
    ):
        try:
            build()
        except ValueError as exc:
            assert "exceeds" in str(exc), str(exc)
        else:
            raise AssertionError("an inverted range was accepted by the domain")
    # Equal ends are ordered; a minimum above an "auto" ceiling has no
    # ceiling to exceed (the ordering rule is about numeric maxima).
    MemorySettings(vram_min_mb=5000, vram_max_mb=5000)
    MemorySettings(vram_min_mb=9000)


def test_save_graph_rejects_unusable_budget_values():
    """The save edge refuses bad values by name, verbatim storage intact."""
    from backend.application.errors import InvalidQueryError
    from backend.application.use_cases.save_graph import SaveGraph

    kept = SaveGraph._payload({
        "nodes": [], "edges": [], "memory": {"vram_max_mb": 4096.0},
    })
    assert kept["memory"] == {"vram_max_mb": 4096.0}  # still verbatim
    for value in (-9000, 0, float("nan"), True, "lots", 10**400):
        try:
            SaveGraph._payload({
                "nodes": [], "edges": [], "memory": {"vram_max_mb": value},
            })
        except InvalidQueryError as exc:
            assert "vram_max_mb" in str(exc), (value, exc)
        else:
            raise AssertionError(f"vram_max_mb={value!r} was saved")
    try:
        SaveGraph._payload({
            "nodes": [], "edges": [],
            "memory": {"vram_min_mb": 5000, "vram_max_mb": 4096},
        })
    except InvalidQueryError as exc:
        assert "exceeds" in str(exc), str(exc)
    else:
        raise AssertionError("an inverted memory range was saved")



def main() -> None:
    """Run every test in this file, listed by name.

    Listed, not discovered: a `def test_*` nothing calls is a comment
    shaped like a safety net, and `scripts/check_test_wiring.py` fails
    this file when one is defined and left out here -- all 16 of
    these were, and the file exited 0 having run nothing, before that
    check caught it.
    """
    tests = [
        test_memory_settings_defaults,
        test_memory_settings_as_dict,
        test_memory_settings_from_dict,
        test_memory_settings_from_none,
        test_device_is_carried_validated_and_not_overridable,
        test_graph_format_is_2,
        test_graph_as_dict_has_memory,
        test_graph_from_dict_round_trip,
        test_graph_from_dict_format_1_loads,
        test_graph_from_dict_ignores_unknown_keys,
        test_graph_non_default_memory_round_trips,
        test_run_in_memory_reaches_definition,
        test_run_in_rejects_unknown_memory_setting,
        test_run_in_rejects_prose_for_auto,
        test_save_graph_rejects_unknown_memory_setting,
        test_effective_memory_stated_beats_observed,
        test_effective_memory_observed_when_higher,
        test_effective_memory_unknown_uses_capacity,
        test_effective_memory_overrides_win,
        test_effective_memory_no_overrides,
        test_effective_memory_pillow_added_to_observed,
        test_effective_memory_fingerprint_not_in_record,
        test_effective_memory_unknown_without_capacity_is_none,
        test_effective_memory_round_trips_storage,
        test_run_in_overrides_keep_only_the_named_keys,
        test_run_in_rejects_unknown_override_key,
        test_run_in_rejects_unusable_budgets,
        test_run_in_rejects_inverted_ranges,
        test_domain_rejects_unusable_budgets,
        test_domain_rejects_inverted_range,
        test_save_graph_rejects_unusable_budget_values,
    ]
    for test in tests:
        test()
    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {len(tests)} CHECKS PASSED")


if __name__ == "__main__":
    main()
