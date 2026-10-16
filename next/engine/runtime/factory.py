"""The run-wide runtime a composition root hands to ``GraphRunner(runtime_factory=...)``."""

from __future__ import annotations

from dataclasses import dataclass

from ..devices import Accelerator, resolve_device
from .memory import MemoryPolicy, ResidencyManager


@dataclass
class Runtime:
    """What nodes reach as ``context.runtime``. Nodes must work when it is ``None`` (no residency management)."""

    residency: ResidencyManager


def make_runtime(context) -> Runtime:
    settings = context.settings
    manager = ResidencyManager(Accelerator(resolve_device(settings.get("device.default"))), MemoryPolicy.from_settings(settings),
                               report=lambda frame: context.report("memory", frame))
    return Runtime(manager)
