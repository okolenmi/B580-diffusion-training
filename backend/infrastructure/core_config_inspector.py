"""CoreConfigInspector -- ConfigInspector over core.config_io/core.config_model.

The adapter owns all knowledge of the config format; the application
only ever sees the summary/description dataclasses and the two
application errors.

``describe`` resolves every configured path against the *settings-
aware* directories (via ``WorkspaceLayout``): absolute paths as-is,
relative names under the kind's base dir -- so availability reflects
what this server would actually hand to the trainer, not what a
CWD-relative guess would find.
"""

from __future__ import annotations

from pathlib import Path

from ..application.errors import ConfigInvalidError, ConfigNotFoundError
from ..application.ports.config_inspector import (
    ConfigDescription,
    ConfigInspector,
    ConfigSummary,
    StartOption,
)
from .workspace import WorkspaceLayout


class CoreConfigInspector(ConfigInspector):
    def __init__(self, layout: WorkspaceLayout) -> None:
        self._layout = layout

    def summarize(self, config_path: Path) -> ConfigSummary:
        config = self._load(config_path)
        return ConfigSummary(
            mode=str(config.tuning.method),
            total_steps=int(config.common.steps),
        )

    def describe(self, config_path: Path) -> ConfigDescription:
        config = self._load(config_path)
        method = str(config.tuning.method)

        options = {
            "teacher": self._option(
                str(config.paths.base_model or ""),
                kind="checkpoint",
                label="Base Model",
            ),
            "student": self._option(
                str(config.paths.student or ""),
                kind="checkpoint",
                label="Student",
            ),
            "resume": self._option(
                str(config.paths.resume_checkpoint or ""),
                kind="lora" if method == "lora" else "checkpoint",
                label="Resume",
            ),
        }
        if method == "lora":
            # Absent, not "unavailable": lora_checkpoint is not a
            # launch choice for non-LoRA configs at all.
            options["lora_checkpoint"] = self._option(
                str(config.tuning.lora_output or ""),
                kind="lora",
                label="LoRA Checkpoint",
            )

        return ConfigDescription(
            mode=method,
            total_steps=int(config.common.steps),
            start_from=options,
        )

    # -- helpers -------------------------------------------------------

    def _option(self, raw_path: str, *, kind: str, label: str) -> StartOption:
        if not raw_path:
            return StartOption(path="", available=False, label=label)
        base = (
            self._layout.loras_dir if kind == "lora" else self._layout.checkpoints_dir
        )
        candidate = Path(raw_path)
        resolved = candidate if candidate.is_absolute() else (base / candidate)
        return StartOption(path=raw_path, available=resolved.is_file(), label=label)

    def _load(self, config_path: Path):
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
        return config
