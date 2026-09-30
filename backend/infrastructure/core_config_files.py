"""CoreConfigFiles -- ConfigFiles over core.config_io / core.config_model.

The adapter owns all knowledge of the config format; the application
only ever sees plain dicts and the two config errors. Updates are
validate-then-write, so a rejected merge never touches the file.

Bridging note: ``core`` is imported lazily inside methods (same
deliberate-bridge posture as ``CoreConfigInspector``) so importing
the backend never drags the training stack into a process that only
serves HTTP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..application.errors import ConfigInvalidError, ConfigNotFoundError
from ..application.ports.config_files import ConfigFiles


def _core_io():
    try:
        from core import config_io, config_model  # repo bridge
    except ImportError as exc:  # pragma: no cover - env breakage
        raise ConfigInvalidError(f"training config reader unavailable: {exc}") from exc
    return config_io, config_model


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class CoreConfigFiles(ConfigFiles):
    def read(self, config_path: Path) -> dict[str, Any]:
        return self._load(config_path).model_dump(mode="json")

    def read_raw(self, config_path: Path) -> str:
        try:
            return config_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigNotFoundError(f"config file not found: {config_path}") from exc
        except OSError as exc:
            raise ConfigInvalidError(f"cannot read config {config_path}: {exc}") from exc

    def update(
        self, config_path: Path, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        if not overrides:
            return self.read(config_path)  # still validates; no write needed
        current = self._load(config_path)  # raises not-found/invalid first
        config_io, config_model = _core_io()
        merged = _deep_merge(current.model_dump(mode="json"), overrides)
        try:
            config = config_model.TrainingConfig.model_validate(merged)
        except Exception as exc:
            raise ConfigInvalidError(
                f"config update rejected, file unchanged: {exc}"
            ) from exc
        config_io.write_config(config_path, config)
        return config.model_dump(mode="json")

    def replace(self, config_path: Path, content: str) -> None:
        config_io, _ = _core_io()
        try:
            config = config_io.config_from_toml_string(content)
        except Exception as exc:
            raise ConfigInvalidError(
                f"invalid config document, file unchanged: {exc}"
            ) from exc
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_io.write_config(config_path, config)

    # -- shared --------------------------------------------------------

    def _load(self, config_path: Path):
        config_io, _ = _core_io()
        try:
            return config_io.read_config(config_path)
        except FileNotFoundError as exc:
            raise ConfigNotFoundError(f"config file not found: {config_path}") from exc
        except Exception as exc:
            raise ConfigInvalidError(
                f"config {config_path} is not a valid training config: {exc}"
            ) from exc
