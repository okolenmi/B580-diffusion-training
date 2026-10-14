"""Graph core: node contract, document, validation, runtime, events and settings.

Pure Python; no torch. The server imports this package, so it must stay light.
"""

from .catalog import CATALOG_FORMAT, build_catalog, node_info
from .document import FORMAT, Edge, GraphDoc, GraphFormatError, NodeSpec, new_uid, parse
from .events import (CheckpointDue, EpochEnding, EpochStarted, EventContext, EventHub, EventType, MemoryPressure,
                     PromptsChanged, RunStarted, RunStopped, StepFinished, StopRequested)
from .modules import DisplayModule, Series
from .node import COMPLETE, Completion, ExecutionContext, Node, Resolution, is_field
from .ports import DynamicShape, EventIn, EventOut, Group, Port
from .registry import NodeRegistry
from .resolve import resolve_node
from .runtime import GraphRunner, NodeResult, RunOutcome
from .settings import CORE_SETTINGS, SettingSpec, Settings, SettingsRegistry
from .validate import Issue, Validation, validate

__all__ = [name for name in dir() if not name.startswith("_")]
