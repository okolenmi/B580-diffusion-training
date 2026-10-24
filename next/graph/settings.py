"""Graph settings: the environment of a run, kept out of node ports.

See rework/contracts/settings.md and plan 20. A setting is data (a ``SettingSpec``);
nodes and families contribute specs; the registry validates a graph's values; a run
gets an immutable ``Settings`` snapshot and each node sees only the keys it declared in
``READS_SETTINGS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

TYPES = ("bool", "int", "float", "str", "choice")
AUTO = "auto"


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    type: str
    default: Any
    doc: str = ""
    choices: tuple[str, ...] | None = None
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    group: str = ""
    advanced: bool = False
    allow_auto: bool = False
    contributed_by: str = "core"

    def __post_init__(self) -> None:
        if "." not in self.key or self.key != self.key.lower():
            raise ValueError(f"setting key {self.key!r} must be lowercase 'group.name'")
        if self.type not in TYPES:
            raise ValueError(f"setting {self.key}: type must be one of {TYPES}")
        if self.type == "choice" and not self.choices:
            raise ValueError(f"setting {self.key}: a choice needs choices")
        if self.type != "choice" and self.choices is not None:
            raise ValueError(f"setting {self.key}: choices only on a choice setting")
        if not self.group:
            object.__setattr__(self, "group", self.key.split(".")[0].capitalize())
        problem = self.problem(self.default)
        if problem:
            raise ValueError(f"setting {self.key}: default is invalid: {problem}")

    def problem(self, value: Any) -> str | None:
        """Why ``value`` is not acceptable, or None."""
        if isinstance(value, str) and value == AUTO and (
                self.allow_auto or (self.type == "choice" and AUTO in (self.choices or ()))):
            return None
        if self.type == "bool":
            return None if isinstance(value, bool) else "expected true or false"
        if self.type in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "expected a number"
            if self.type == "int" and int(value) != value:
                return "expected a whole number"
            if self.minimum is not None and value < self.minimum:
                return f"below the minimum {self.minimum}"
            if self.maximum is not None and value > self.maximum:
                return f"above the maximum {self.maximum}"
            return None
        if not isinstance(value, str):
            return "expected text"
        if self.type == "choice" and value not in (self.choices or ()):
            return f"not one of {list(self.choices or ())}"
        return None

    def as_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "doc": self.doc, "type": self.type,
            "default": self.default, "choices": list(self.choices) if self.choices else None,
            "unit": self.unit, "minimum": self.minimum, "maximum": self.maximum, "step": self.step,
            "group": self.group, "advanced": self.advanced, "allow_auto": self.allow_auto,
            "contributed_by": self.contributed_by,
        }


_PRECISIONS = ("auto", "bf16", "fp16", "fp32")

CORE_SETTINGS: tuple[SettingSpec, ...] = (
    SettingSpec("memory.vram_max_mb", "VRAM budget", "float", AUTO, allow_auto=True, unit="MB",
                minimum=256, step=256, group="Memory",
                doc="Most device memory in use on the card, as the driver reports it (other programs included). "
                    "'auto' is the whole card, with head-room kept by the safety margin; a number above the card is refused."),
    SettingSpec("memory.margin_mb", "Memory margin", "float", 512, unit="MB", minimum=0, step=64, group="Memory", advanced=True,
                doc="The one safety margin: memory kept free below the budget for what cannot be predicted (kernels of a new shape, "
                    "driver growth). The allocator cap, the checkpoint tuner, block placement and the planner all use "
                    "budget - outside memory - this, with nothing added on top. Raise it if steps run out of memory when a new "
                    "shape first appears; lower it to use more of the card."),
    SettingSpec("memory.vram_min_mb", "VRAM floor", "float", 0, unit="MB", minimum=0, step=256,
                group="Memory", advanced=True, doc="Refuse to start when less than this is free. 0 is no floor."),
    SettingSpec("memory.strict", "Strict budget", "bool", True, group="Memory", advanced=True,
                doc="Going over the budget is a bug in a node, not something to train through."),
    SettingSpec("memory.ram_max_mb", "RAM budget", "float", AUTO, allow_auto=True, unit="MB",
                minimum=256, group="Memory", advanced=True, doc="Host-memory ceiling; recorded with the run."),
    SettingSpec("device.default", "Device", "str", "xpu", group="Device",
                doc="Where nodes compute unless one declares an override. torch device strings, e.g. xpu, cpu."),
    SettingSpec("precision.denoiser", "Denoiser precision", "choice", "auto", choices=_PRECISIONS, group="Device",
                doc="'auto' keeps the file's own dtype."),
    SettingSpec("precision.encoders", "Text encoder precision", "choice", "auto", choices=_PRECISIONS,
                group="Device", doc="'auto' keeps the file's own dtype."),
    SettingSpec("precision.vae", "VAE precision", "choice", "auto", choices=_PRECISIONS, group="Device",
                doc="'auto' keeps the file's own dtype."),
    SettingSpec("precision.autocast", "Compute precision", "choice", "bf16", choices=("bf16", "fp16", "off"),
                group="Device", advanced=True, doc="Dtype used for the arithmetic of a training step."),
    SettingSpec("runtime.graph_capture", "Graph capture", "bool", False, group="Runtime", advanced=True,
                doc="Capture the step as a device graph (static shapes only)."),
    SettingSpec("runtime.deterministic", "Deterministic run", "bool", False, group="Reproducibility", advanced=True,
                doc="Same graph, data and settings give the same numbers: deterministic kernels (slower where they differ), "
                    "the global generator seeded from the base seed (0 when unset), and batches never split by predicted "
                    "memory (a batch that does not fit then runs out of memory instead; lower the batch size)."),
    SettingSpec("runtime.residency.calibration_steps", "Residency calibration steps", "int", 3, minimum=0,
                group="Runtime", advanced=True, doc="Steps spent measuring memory before residency is settled."),
    SettingSpec("runtime.empty_cache_every_n_steps", "Empty cache every N steps", "int", 1, minimum=0,
                group="Runtime", advanced=True, doc="0 never empties the allocator cache."),
    SettingSpec("runtime.block_swap", "Blocks held on the host", "int", 0, minimum=0, group="Runtime",
                advanced=True, doc="Denoiser blocks kept in host memory and brought in per block."),
    SettingSpec("runtime.profile", "Profile steps", "bool", False, group="Runtime", advanced=True,
                doc="Time each phase of the step (the trainer's timing port starts from it)."),
    SettingSpec("perf.dir", "Performance report folder", "str", "", group="Performance", advanced=True,
                doc="Profile a few steps of the trainer and write report.md and report.json here (empty = off). The report rates every "
                    "kernel category and module against what the card measures it can do; see the performance manual page."),
    SettingSpec("perf.ceilings", "Ceilings cache file", "str", "", group="Performance", advanced=True,
                doc="Where the measured ceilings are cached (empty = ceilings.json in the report folder). Runs that share a file compare against the same numbers."),
    SettingSpec("perf.warmup_steps", "Performance warm-up steps", "int", 8, minimum=0, group="Performance", advanced=True,
                doc="Steps run before the profiled window: kernels get built, shapes seen and graphs captured. Replays start from the "
                    "second sighting of a shape, so keep this above the number of distinct shapes."),
    SettingSpec("perf.steps", "Performance window", "int", 3, minimum=1, group="Performance", advanced=True,
                doc="Steps profiled, each with the device idle at both ends. Profiling slows these steps."),
    SettingSpec("perf.modules", "Time every module", "bool", True, group="Performance", advanced=True,
                doc="Per-module forward, recompute and backward device time. Needs eager steps (a replayed graph runs no Python)."),
    SettingSpec("perf.provenance", "Tie kernels to operators and modules", "bool", False, group="Performance", advanced=True,
                doc="Records operator shapes and opens a profiler range per module call, so every kernel is reported with the operator, dtypes, shapes and module "
                    "that launched it and module times are device times. Slower to process."),
    SettingSpec("perf.trace", "Write the kernel trace", "bool", False, group="Performance", advanced=True,
                doc="Also write trace.json (open in a Chrome-trace viewer); large."),
    SettingSpec("conditioning.cache_mb", "Conditioning cache", "float", AUTO, allow_auto=True, unit="MB",
                minimum=0, group="Conditioning", advanced=True,
                doc="Size of the encoded-prompt cache; 'auto' derives it from the budget."),
    SettingSpec("conditioning.prewarm", "Prewarm encoder cache", "bool", True, group="Conditioning",
                doc="Encode the prompts before training starts."),
    SettingSpec("seed.base", "Base seed", "int", -1, minimum=-1, group="Reproducibility",
                doc="Nodes with an empty seed derive theirs from this and their uid. -1 leaves them at 0, so every run sees the same order and noise."),
)


class SettingsError(ValueError):
    pass


class SettingsRegistry:
    """The merged set of specs for one process."""

    def __init__(self, *contributions: Iterable[SettingSpec]):
        self._specs: dict[str, SettingSpec] = {}
        self.errors: list[str] = []
        for group in (CORE_SETTINGS, *contributions):
            for spec in group:
                if spec.key in self._specs:
                    self.errors.append(
                        f"duplicate setting {spec.key!r} from {spec.contributed_by!r} "
                        f"(already from {self._specs[spec.key].contributed_by!r})")
                    continue
                self._specs[spec.key] = spec

    def specs(self) -> tuple[SettingSpec, ...]:
        return tuple(self._specs.values())

    def spec(self, key: str) -> SettingSpec | None:
        return self._specs.get(key)

    def validate(self, values: Mapping[str, Any]) -> list[tuple[str, str, str]]:
        """(severity, key, message) for every problem in a graph's settings map."""
        problems: list[tuple[str, str, str]] = []
        for key, value in values.items():
            spec = self._specs.get(key)
            if spec is None:
                problems.append(("warning", key, "unknown setting (kept, not used)"))
                continue
            message = spec.problem(value)
            if message:
                problems.append(("error", key, message))
        low = values.get("memory.vram_min_mb", self._specs["memory.vram_min_mb"].default)
        high = values.get("memory.vram_max_mb", AUTO)
        if isinstance(low, (int, float)) and isinstance(high, (int, float)) \
                and not isinstance(low, bool) and not isinstance(high, bool) and low > high:
            problems.append(("error", "memory.vram_min_mb", "floor is above the budget"))
        return problems

    def snapshot(self, graph_values: Mapping[str, Any] | None = None,
                 app_defaults: Mapping[str, Any] | None = None) -> "Settings":
        """graph value > app default > spec default, for every known key."""
        resolved: dict[str, Any] = {}
        origin: dict[str, str] = {}
        for key, spec in self._specs.items():
            if graph_values and key in graph_values:
                resolved[key], origin[key] = graph_values[key], "graph"
            elif app_defaults and key in app_defaults:
                resolved[key], origin[key] = app_defaults[key], "app"
            else:
                resolved[key], origin[key] = spec.default, "default"
        return Settings(resolved, origin)


class Settings:
    """Immutable resolved values for one run."""

    def __init__(self, values: Mapping[str, Any], origin: Mapping[str, str] | None = None):
        self._values = dict(values)
        self._origin = dict(origin or {})

    def get(self, key: str) -> Any:
        try:
            return self._values[key]
        except KeyError:
            raise SettingsError(f"unknown setting {key!r}") from None

    def origin(self, key: str) -> str:
        return self._origin.get(key, "default")

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def view(self, allowed: Iterable[str], owner: str) -> "SettingsView":
        return SettingsView(self, frozenset(allowed), owner)


class SettingsView:
    """What one node sees: only the keys it declared in READS_SETTINGS."""

    def __init__(self, settings: Settings, allowed: frozenset[str], owner: str):
        self._settings = settings
        self._allowed = allowed
        self._owner = owner

    def get(self, key: str) -> Any:
        if key not in self._allowed:
            raise SettingsError(f"{self._owner} reads setting {key!r} without declaring it in READS_SETTINGS")
        return self._settings.get(key)
