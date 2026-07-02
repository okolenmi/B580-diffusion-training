"""Option tree builder — generates frontend schema from UI metadata + config values."""

from typing import Any

from .config_ui import OPTION_TREE, SYNTHETIC_OPTIONS
from core.config_io import read_config
from core.config_model import TrainingConfig


def get_config_defaults() -> dict:
    """Return default values from TrainingConfig as a flat dotted-key dict."""
    return _flatten_config(TrainingConfig())


def _flatten_config(config: TrainingConfig, prefix: str = "") -> dict[str, Any]:
    """Flatten a TrainingConfig to dotted-key dict (e.g. 'common.steps')."""
    result: dict[str, Any] = {}
    for section_name in ("paths", "common", "tuning", "cache"):
        section = getattr(config, section_name)
        if isinstance(section, dict):
            values = section
        else:
            values = section.model_dump()
        for key, value in values.items():
            dotted = f"{section_name}.{key}" if prefix else f"{section_name}.{key}"
            result[dotted] = value
    result["start_from"] = config.start_from
    result["reset_optimizer"] = config.reset_optimizer
    return result


def _read_config_values(config_path: str | None) -> dict[str, Any]:
    """Read actual config values from file, returning flat dotted-key dict."""
    if not config_path:
        return {}
    from pathlib import Path
    from .config import settings
    p = Path(config_path)
    if not p.is_absolute():
        p = settings.project_root / config_path
    if not p.exists():
        return {}
    try:
        config = read_config(p)
        return _flatten_config(config)
    except Exception:
        return {}


def build_option_tree(config_path: str | None = None) -> list[dict]:
    """Generate the frontend option tree from UI metadata + config values.

    Returns a flat list of option dicts. The frontend groups them by
    the 'group' field.
    """
    config_values = _read_config_values(config_path)

    # Start with synthetic options (start_from, reset_optimizer)
    options: list[dict] = []
    for opt in SYNTHETIC_OPTIONS:
        opt = dict(opt)
        saved = config_values.get(opt["id"])
        if saved is not None:
            opt["default"] = saved
        options.append(opt)

    # Build options from UI metadata
    for dotted_key, meta in OPTION_TREE.items():
        default_from_config = config_values.get(dotted_key)

        opt = {
            "id": dotted_key,
            "label": meta.get("label", dotted_key),
            "type": meta.get("type", "text"),
            "default": default_from_config if default_from_config is not None else meta.get("default"),
            "group": meta.get("group", "General"),
        }

        for extra_key in ("choices", "min", "max", "step", "placeholder", "help"):
            if extra_key in meta:
                opt[extra_key] = meta[extra_key]

        if "visible_when" in meta:
            opt["visible_when"] = meta["visible_when"]

        # Clean None values (except default which can be 0 or False)
        opt = {k: v for k, v in opt.items()
               if v is not None or k == "default"}

        options.append(opt)

    return options
