"""Option tree builder — merges auto-derived schema (config_schema.py) with
hand-authored UI extras (config_ui.py) + actual config values into the flat
option list the frontend expects."""

from typing import Any

from . import config_schema
from .config_ui import EXTRAS, SYNTHETIC_OPTIONS
from core.config_io import read_config
from core.config_model import TrainingConfig

# start_from / reset_optimizer are real TrainingConfig fields (so
# config_schema.py picks them up automatically), but the UI should only
# ever show the per-launch SYNTHETIC_OPTIONS version of them -- see
# config_ui.py's docstring. Excluded here so they don't show up twice.
_EXCLUDE_FROM_SCHEMA = {"start_from", "reset_optimizer"}


def _humanize(dotted_key: str) -> str:
    name = dotted_key.rsplit(".", 1)[-1]
    return name.replace("_", " ").title()


def _naive_choice_label(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


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
    """Generate the frontend option tree from the auto-derived schema +
    hand-authored extras + actual config values.

    Returns a flat list of option dicts (same shape as before this was
    split into config_schema.py/config_ui.py -- the frontend doesn't need
    to change).
    """
    config_values = _read_config_values(config_path)
    schema_opts = config_schema.build_schema_options()

    options: list[dict] = []

    # Synthetic (per-launch) options first, same as before.
    for opt in SYNTHETIC_OPTIONS:
        opt = dict(opt)
        saved = config_values.get(opt["id"])
        if saved is not None:
            opt["default"] = saved
        options.append(opt)

    for dotted_key, base in schema_opts.items():
        if dotted_key in _EXCLUDE_FROM_SCHEMA:
            continue

        extra = EXTRAS.get(dotted_key, {})
        default_from_config = config_values.get(dotted_key)

        opt = {
            "id": dotted_key,
            "label": extra.get("label", _humanize(dotted_key)),
            "type": base["type"],
            "default": default_from_config if default_from_config is not None else base.get("default"),
            "group": extra.get("group", "General"),
        }

        if "choices" in base:
            choice_labels = extra.get("choice_labels", {})
            choice_order = extra.get("choice_order")
            raw_choices = base["choices"]
            if choice_order:
                # Preserve a curated display order; anything in raw_choices
                # not explicitly ordered is appended at the end (so a new
                # Union variant added later still shows up automatically
                # instead of silently disappearing).
                ordered = [v for v in choice_order if v in raw_choices]
                ordered += [v for v in raw_choices if v not in choice_order]
                raw_choices = ordered
            opt["choices"] = [
                {"value": v, "label": choice_labels.get(v, _naive_choice_label(v))}
                for v in raw_choices
            ]

        for key in ("min", "max"):
            if key in base:
                opt[key] = base[key]
        for key in ("step", "placeholder", "help"):
            if key in extra:
                opt[key] = extra[key]

        # Optional: for a path field that refers to a checkpoint or LoRA file,
        # the frontend adds a dropdown (populated from GET /api/files/{kind})
        # alongside the normal text input -- pick from what's actually on
        # disk, or type a path manually, your choice. See
        # option-tree.js's attachFilePicker().
        if "file_kind" in extra:
            opt["file_kind"] = extra["file_kind"]

        # Optional: explicit within-group render order (lower = earlier).
        # Fields without this stay in their natural pydantic-declaration
        # order relative to each other -- only the handful that actually
        # need to come first/follow something specific (e.g. Data Source
        # before Dataset Name, since the latter only makes sense once you
        # know the former) need to set this.
        if "order" in extra:
            opt["order"] = extra["order"]

        visible_when = dict(base.get("visible_when") or {})
        visible_when.update(extra.get("extra_visible_when") or {})
        if visible_when:
            opt["visible_when"] = visible_when

        opt = {k: v for k, v in opt.items()
               if v is not None or k == "default"}

        options.append(opt)

    return options
