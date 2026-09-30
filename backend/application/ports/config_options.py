"""ConfigOptions port -- the editable schema of a training config.

A flat list of option descriptors (one per configurable field) that a
form renderer can build the config editor from: type, default,
bounds, choices, group, visibility conditions.

``schema()`` is a pure function of the config *model* plus the
hand-authored UI metadata -- it reads no config file and no
filesystem. Field *values* come from ``ConfigFiles.read``; keeping
the two apart means the schema is fetchable once and reused across
every config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConfigOptions(ABC):
    @abstractmethod
    def schema(self) -> list[dict[str, Any]]:
        raise NotImplementedError
