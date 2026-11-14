"""CoreConfigFiles -- ConfigFiles over nodes.config_io / nodes.config_model.

The adapter owns all knowledge of the config format; the application
only ever sees plain dicts and the two config errors. Updates are
validate-then-write, so a rejected merge never touches the file, and the
raw editor stores the user's own text (comments and unknown keys
intact, docs 07 F-08) through a temp sibling + rename.

Bridging note: the config modules are imported lazily inside methods
(same deliberate-bridge posture as ``CoreConfigInspector``) so importing
the backend never drags the training stack into a process that only
serves HTTP.

The class keeps its ``Core``-prefixed name. It is now a *naming* remnant
rather than a description of anything -- the format is ``nodes``' own --
but renaming an adapter class is churn with no behaviour behind it, and
the ports it implements are what callers actually depend on.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..application.errors import ConfigInvalidError, ConfigNotFoundError
from ..application.ports.config_files import ConfigFiles


def _config_io():
    try:
        from nodes import config_io, config_model
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
        config_io, config_model = _config_io()
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
        """Validate the document, then store the user's own text.

        Serializing the parsed model back out would drop comments, key
        order and any key the model does not declare -- silent data loss
        in a file people keep notes in (docs 07 F-08). So the text is
        validated by parsing it and then written verbatim through a temp
        sibling + rename: a failed write never truncates either.
        """
        config_io, _ = _config_io()
        try:
            config_io.config_from_toml_string(content)
        except Exception as exc:
            raise ConfigInvalidError(
                f"invalid config document, file unchanged: {exc}"
            ) from exc
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_atomic(config_path, content)

    @staticmethod
    def _write_atomic(config_path: Path, content: str) -> None:
        temp = config_path.with_name(f".{config_path.name}.partial")
        try:
            with open(temp, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            temp.replace(config_path)
        except OSError as exc:
            try:
                temp.unlink()
            except OSError:
                pass  # nothing better to do about the temp file either
            raise ConfigInvalidError(f"cannot write config {config_path}: {exc}") from exc

    # -- shared --------------------------------------------------------

    def _load(self, config_path: Path):
        config_io, _ = _config_io()
        try:
            return config_io.read_config(config_path)
        except FileNotFoundError as exc:
            raise ConfigNotFoundError(f"config file not found: {config_path}") from exc
        except Exception as exc:
            raise ConfigInvalidError(
                f"config {config_path} is not a valid training config: {exc}"
            ) from exc
