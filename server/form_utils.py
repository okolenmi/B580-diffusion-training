"""Shared form-data parsing/coercion helpers.

Extracted from routes_config.py and routes_training.py, which each had
their own near-identical copy of deep_merge / value coercion / dotted-key
nesting / flat-key migration. One implementation now, used by both.
"""

from core.config_io import FLAT_TO_SECTION_MAP


def coerce_value(value: str):
    """Best-effort type coercion for a single form string value."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def coerce_dict(obj):
    """Recursively coerce string values in a nested dict."""
    if isinstance(obj, dict):
        return {k: coerce_dict(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return coerce_value(obj)
    return obj


def dotted_to_nested(dotted: dict) -> dict:
    """Convert dotted-key form data (e.g. {"common.steps": "10"}) to a
    nested dict (e.g. {"common": {"steps": "10"}})."""
    result: dict = {}
    for key, value in dotted.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def migrate_flat_keys(data: dict) -> dict:
    """Rename old flat keys (e.g. "steps") to section-based dotted keys
    (e.g. "common.steps") using the same map config_io.py uses for TOML
    migration."""
    result = {}
    for key, value in data.items():
        if key in FLAT_TO_SECTION_MAP:
            result[FLAT_TO_SECTION_MAP[key]] = value
        else:
            result[key] = value
    return result


def form_to_nested_overrides(form_data, ignore_keys: set = frozenset()) -> dict:
    """Full pipeline: raw form/query data -> migrated, coerced, nested dict
    of config overrides ready for deep_merge() against an existing config.

    ignore_keys: synthetic form fields that aren't config overrides at all
    (e.g. "config", "start_from", "reset_optimizer").
    """
    raw: dict[str, str] = {}
    for key, value in form_data.items():
        if key in ignore_keys:
            continue
        raw[key] = str(value)

    raw = migrate_flat_keys(raw)
    nested = dotted_to_nested(raw)
    return coerce_dict(nested)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides dict into base dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
