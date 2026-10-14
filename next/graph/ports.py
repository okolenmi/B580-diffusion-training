"""Port, Group and the other declarations a node class carries.

Everything here is pure data: nodes declare, the catalog serialises, the editor draws,
the validator checks. Nothing in this module runs a node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NUMBER_TYPES = (int, float)


@dataclass(frozen=True)
class Port:
    """One named input or output slot.

    A port with a primitive type and no wire is a *field*: its value lives in the
    node's params. Presentation hints (``label`` .. ``ui``) never affect execution.
    """

    name: str
    type: Any
    required: bool = True
    default: Any = None
    doc: str = ""
    choices: tuple[str, ...] | None = None
    visible_when: tuple[str, Any] | None = None
    widget_only: bool = False
    multi: bool = False
    picker: str | None = None
    label: str = ""
    group: str = ""
    advanced: bool = False
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    ui: dict = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        for bound in ("minimum", "maximum", "step"):
            value = getattr(self, bound)
            if value is not None and (isinstance(value, bool) or not isinstance(value, NUMBER_TYPES)):
                raise TypeError(f"Port {self.name!r}: {bound} must be a number, got {value!r}")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"Port {self.name!r}: minimum {self.minimum} exceeds maximum {self.maximum}")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Port {self.name!r}: step must be positive")
        if self.multi and self.widget_only:
            raise ValueError(f"Port {self.name!r}: a multi port needs a wire socket")
        if self.choices is not None:
            if self.type is not str:
                raise TypeError(f"Port {self.name!r}: choices need a str port")
            if not isinstance(self.choices, tuple) or not self.choices or not all(
                isinstance(c, str) for c in self.choices
            ):
                raise TypeError(f"Port {self.name!r}: choices must be a non-empty tuple of str")
            if self.default is not None and self.default not in self.choices:
                raise ValueError(f"Port {self.name!r}: default {self.default!r} is not in choices")
        if self.visible_when is not None:
            if not (isinstance(self.visible_when, tuple) and len(self.visible_when) == 2
                    and isinstance(self.visible_when[0], str)):
                raise TypeError(f"Port {self.name!r}: visible_when must be (port_name, value)")


@dataclass(frozen=True)
class Group:
    """A named section of fields on a node. Ports opt in with ``Port.group``."""

    id: str
    label: str
    collapsed: bool = False
    advanced: bool = False
    visible_when: tuple[str, Any] | None = None
    doc: str = ""


@dataclass(frozen=True)
class DynamicShape:
    """Marks a node whose ports depend on its params; ``resolve`` computes them."""

    depends_on: tuple[str, ...]
    debounce_ms: int = 300

    def __post_init__(self) -> None:
        if not self.depends_on or not all(isinstance(n, str) for n in self.depends_on):
            raise TypeError("DynamicShape.depends_on must be a non-empty tuple of port names")
        if self.debounce_ms < 0:
            raise ValueError("DynamicShape.debounce_ms must not be negative")


@dataclass(frozen=True)
class EventIn:
    payload: type
    handler: str
    doc: str = ""
    label: str = ""


@dataclass(frozen=True)
class EventOut:
    payload: type
    doc: str = ""
    label: str = ""


def type_name(t: Any) -> str:
    return getattr(t, "__name__", None) or str(t)


def type_mro_names(t: Any) -> list[str]:
    mro = getattr(t, "__mro__", None)
    return [c.__name__ for c in mro] if mro else [type_name(t)]


def types_compatible(out_type: Any, in_type: Any) -> bool:
    """True when a value of ``out_type`` may feed a port of ``in_type``."""
    if in_type is Any or out_type is Any:
        return True
    if isinstance(out_type, type) and isinstance(in_type, type):
        return issubclass(out_type, in_type)
    return True
