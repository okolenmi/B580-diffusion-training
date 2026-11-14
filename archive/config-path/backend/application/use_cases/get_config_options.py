"""GetConfigOptions -- the config editor's field schema.

Pure: depends on the config model + UI metadata only, so the result
can be fetched once and reused for every config file (values come
from ``GetConfig``).
"""

from __future__ import annotations

from typing import Any

from ..ports.config_options import ConfigOptions


class GetConfigOptions:
    def __init__(self, *, options: ConfigOptions) -> None:
        self._options = options

    def execute(self) -> list[dict[str, Any]]:
        return self._options.schema()
