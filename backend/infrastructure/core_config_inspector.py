"""CoreConfigInspector -- ConfigInspector over core.config_io/core.config_model.

The adapter owns all knowledge of the config format; the application
only ever sees ``ConfigSummary`` or the two application errors.
"""

from __future__ import annotations

from pathlib import Path

from ..application.errors import ConfigInvalidError, ConfigNotFoundError
from ..application.ports.config_inspector import ConfigInspector, ConfigSummary


class CoreConfigInspector(ConfigInspector):
    def summarize(self, config_path: Path) -> ConfigSummary:
        try:
            from core.config_io import read_config  # repo bridge
        except ImportError as exc:  # pragma: no cover - env breakage
            raise ConfigInvalidError(
                f"training config reader unavailable: {exc}"
            ) from exc
        try:
            config = read_config(config_path)
        except FileNotFoundError as exc:
            raise ConfigNotFoundError(
                f"config file not found: {config_path}"
            ) from exc
        except Exception as exc:
            raise ConfigInvalidError(
                f"config {config_path} is not a valid training config: {exc}"
            ) from exc
        return ConfigSummary(
            mode=str(config.tuning.method),
            total_steps=int(config.common.steps),
        )
